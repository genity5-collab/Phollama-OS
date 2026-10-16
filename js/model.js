'use strict';
/**
 * Nova 1.0 — a small neural language model, implemented from scratch in
 * JavaScript. No pretrained weights, no external APIs, no libraries.
 *
 * Architecture (a Bengio-style neural probabilistic language model):
 *   input   : `context` previous character tokens
 *   embed   : each token -> `embedDim` learned vector (concatenated)
 *   hidden  : tanh layer with `hiddenDim` units
 *   output  : softmax over the whole vocabulary (next-token prediction)
 *
 * Everything (forward pass, backpropagation, SGD training, sampling) is real
 * and runs in the browser. The honest consequence of a model this small: it
 * learns character-level statistics and short-range patterns from the corpus
 * you give it. It is NOT a ChatGPT-class assistant and does not understand
 * meaning. See README.md for the full capability statement.
 */
class Nova1Model {
  constructor(cfg) {
    cfg = cfg || {};
    this.config = {
      context: cfg.context || 3,
      embedDim: cfg.embedDim || 16,
      hiddenDim: cfg.hiddenDim || 96,
      lr: cfg.lr || 0.1,
      lrDecay: cfg.lrDecay || 0.7,
    };
    this.tokenizer = new NovaTokenizer();
    this.trained = false;
    this.trainSteps = 0;
    this.lastLoss = null;
    this.corpusChars = 0;
    this._initWeights(this.vocabSize);
  }

  get vocabSize() { return this.tokenizer.vocabSize(); }

  _initWeights(vocab) {
    const v = Math.max(vocab || this.vocabSize, 8);
    const { context, embedDim, hiddenDim } = this.config;
    const fanIn1 = context * embedDim;
    this.emb = Nova1Model._rand(v * embedDim, 0.08);
    this.W1 = Nova1Model._rand(fanIn1 * hiddenDim, Math.sqrt(2 / fanIn1));
    this.b1 = new Float32Array(hiddenDim);
    this.W2 = Nova1Model._rand(hiddenDim * v, Math.sqrt(2 / hiddenDim));
    this.b2 = new Float32Array(v);
  }

  static _rand(n, scale) {
    const a = new Float32Array(n);
    for (let i = 0; i < n; i++) a[i] = (Math.random() * 2 - 1) * scale;
    return a;
  }

  /** Grows embedding/output layers when the vocabulary expands (new chars). */
  _resizeForVocab(newVocab) {
    const { embedDim, hiddenDim } = this.config;
    // embeddings
    if (this.emb.length !== newVocab * embedDim) {
      const oldV = Math.floor(this.emb.length / embedDim);
      const newEmb = Nova1Model._rand(newVocab * embedDim, 0.08);
      for (let i = 0; i < oldV; i++) {
        newEmb.set(this.emb.subarray(i * embedDim, (i + 1) * embedDim), i * embedDim);
      }
      this.emb = newEmb;
    }
    // output layer
    if (this.W2.length !== hiddenDim * newVocab) {
      const oldV = this.b2.length;
      const newW2 = Nova1Model._rand(hiddenDim * newVocab, Math.sqrt(2 / hiddenDim));
      const newB2 = new Float32Array(newVocab);
      for (let j = 0; j < hiddenDim; j++) {
        for (let t = 0; t < oldV; t++) newW2[j * newVocab + t] = this.W2[j * oldV + t];
      }
      newB2.set(this.b2);
      this.W2 = newW2;
      this.b2 = newB2;
    }
  }

  /** Forward pass. ctxIds must have exactly `config.context` entries. */
  forward(ctxIds) {
    const { context, embedDim, hiddenDim } = this.config;
    const v = this.vocabSize;
    const x = new Float32Array(context * embedDim);
    for (let i = 0; i < context; i++) {
      const off = ctxIds[i] * embedDim;
      for (let j = 0; j < embedDim; j++) x[i * embedDim + j] = this.emb[off + j];
    }
    const h = new Float32Array(hiddenDim);
    const fanIn1 = x.length;
    for (let j = 0; j < hiddenDim; j++) {
      let s = this.b1[j];
      for (let k = 0; k < fanIn1; k++) s += x[k] * this.W1[k * hiddenDim + j];
      h[j] = Math.tanh(s);
    }
    const logits = new Float32Array(v);
    for (let t = 0; t < v; t++) {
      let s = this.b2[t];
      for (let j = 0; j < hiddenDim; j++) s += h[j] * this.W2[j * v + t];
      logits[t] = s;
    }
    return { x, h, logits };
  }

  softmax(logits, temp) {
    const n = logits.length;
    const out = new Float32Array(n);
    let mx = -Infinity;
    for (let i = 0; i < n; i++) if (logits[i] > mx) mx = logits[i];
    let sum = 0;
    for (let i = 0; i < n; i++) { const e = Math.exp((logits[i] - mx) / temp); out[i] = e; sum += e; }
    for (let i = 0; i < n; i++) out[i] /= sum;
    return out;
  }

  sample(logits, temp) {
    const p = this.softmax(logits, temp);
    let r = Math.random();
    for (let i = 0; i < p.length; i++) { r -= p[i]; if (r <= 0) return i; }
    return p.length - 1;
  }

  /**
   * One SGD step on a single (context -> next token) example.
   * Full backprop: output softmax -> hidden -> embedding. Returns cross-entropy loss.
   */
  _stepLoss(ctx, target, lr) {
    const { context, embedDim, hiddenDim } = this.config;
    const v = this.vocabSize;
    const { x, h, logits } = this.forward(ctx);
    const p = this.softmax(logits, 1);
    const loss = -Math.log(Math.max(p[target], 1e-9));

    // output grads
    const dlog = new Float32Array(v);
    for (let t = 0; t < v; t++) dlog[t] = p[t];
    dlog[target] -= 1;

    // hidden grads (uses pre-update W2)
    const dh = new Float32Array(hiddenDim);
    for (let j = 0; j < hiddenDim; j++) {
      let acc = 0;
      for (let t = 0; t < v; t++) acc += dlog[t] * this.W2[j * v + t];
      dh[j] = acc;
    }
    // apply W2, b2 updates
    for (let j = 0; j < hiddenDim; j++) {
      const hj = h[j];
      const row = j * v;
      for (let t = 0; t < v; t++) this.W2[row + t] -= lr * hj * dlog[t];
    }
    for (let t = 0; t < v; t++) this.b2[t] -= lr * dlog[t];

    // pre-activation grads through tanh
    const dhpre = new Float32Array(hiddenDim);
    for (let j = 0; j < hiddenDim; j++) dhpre[j] = dh[j] * (1 - h[j] * h[j]);

    // embedding grads (uses pre-update W1)
    const demb = new Float32Array(this.emb.length);
    for (let i = 0; i < context; i++) {
      const idOff = ctx[i] * embedDim;
      for (let j = 0; j < embedDim; j++) {
        const k = i * embedDim + j;
        const kH = k * hiddenDim;
        let acc = 0;
        for (let hh = 0; hh < hiddenDim; hh++) acc += this.W1[kH + hh] * dhpre[hh];
        demb[idOff + j] += acc;
      }
    }
    // apply W1, b1, emb updates
    const fanIn1 = context * embedDim;
    for (let k = 0; k < fanIn1; k++) {
      const kH = k * hiddenDim;
      const xk = x[k];
      for (let hh = 0; hh < hiddenDim; hh++) this.W1[kH + hh] -= lr * xk * dhpre[hh];
    }
    for (let hh = 0; hh < hiddenDim; hh++) this.b1[hh] -= lr * dhpre[hh];
    for (let i = 0; i < demb.length; i++) if (demb[i] !== 0) this.emb[i] -= lr * demb[i];

    return loss;
  }

  /**
   * Trains the model on `text`. Real SGD over every context window, chunked
   * with await so the UI keeps breathing. onProgress({epoch, epochs, step,
   * loss, sample}) fires periodically.
   */
  async train(text, opts) {
    opts = opts || {};
    const epochs = Math.max(1, Math.min(opts.epochs || 5, opts.epochCap || 50));
    const onProgress = opts.onProgress || null;
    const shouldCancel = opts.shouldCancel || (() => false);

    if (!text || text.trim().length < 50) {
      throw new Error('Nova needs at least ~50 characters of training text to learn anything.');
    }

    this.tokenizer.buildFromText(text);
    this._resizeForVocab(this.vocabSize);

    const ids = this.tokenizer.encode(text);
    const { context } = this.config;
    if (ids.length <= context) throw new Error('Training text is too short for the context window.');

    const starts = [];
    for (let i = 0; i + context < ids.length; i++) starts.push(i);
    // For big corpora we take a random subsample of windows per epoch
    // (real SGD sampling) so training stays responsive in the browser.
    const MAX_WINDOWS = opts.maxWindows || 12000;
    const usedPerEpoch = Math.min(starts.length, MAX_WINDOWS);

    let lr = this.config.lr;
    let step = 0;
    const reportEvery = Math.max(64, Math.floor(usedPerEpoch / 20));

    for (let e = 0; e < epochs; e++) {
      // Fisher-Yates shuffle
      for (let i = starts.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        const tmp = starts[i]; starts[i] = starts[j]; starts[j] = tmp;
      }
      let total = 0, count = 0;
      const ctx = new Array(context);
      for (let n = 0; n < usedPerEpoch; n++) {
        const w = starts[n];
        if (shouldCancel()) return { cancelled: true, loss: this.lastLoss, steps: step };
        for (let i = 0; i < context; i++) ctx[i] = ids[w + i];
        total += this._stepLoss(ctx, ids[w + context], lr);
        count++; step++;
        if (step % reportEvery === 0) {
          await new Promise((r) => setTimeout(r, 0));
          if (onProgress) onProgress({ epoch: e + 1, epochs, step, loss: total / count, sample: null });
        }
      }
      this.lastLoss = total / count;
      if (!Number.isFinite(this.lastLoss)) {
        throw new Error('Training diverged (numerical overflow). Try fewer epochs or shorter text.');
      }
      lr = Math.max(0.015, lr * this.config.lrDecay);
      if (onProgress) {
        onProgress({ epoch: e + 1, epochs, step, loss: this.lastLoss, sample: this.generate(120, 0.8) });
      }
    }

    this.trainSteps += step;
    this.corpusChars = text.length;
    this.trained = true;
    return { cancelled: false, loss: this.lastLoss, steps: step };
  }

  /**
   * Samples a completion from the model's ACTUAL next-token predictions.
   * Untrained weights still produce real (if meaningless) predictions —
   * nothing here is faked or rule-based.
   */
  generate(maxTokens, temperature, seedIds) {
    maxTokens = maxTokens || 200;
    temperature = temperature || 0.9;
    const { context } = this.config;
    const ctx = new Array(context).fill(NovaTokenizer.PAD);
    if (seedIds && seedIds.length) {
      // continue from the last `context` tokens of the seed (left-padded with <pad>)
      const tail = seedIds.slice(-context);
      for (let i = 0; i < tail.length; i++) ctx[context - tail.length + i] = tail[i];
    } else {
      ctx[context - 1] = NovaTokenizer.BOS;
    }
    const out = [];
    for (let t = 0; t < maxTokens; t++) {
      const { logits } = this.forward(ctx);
      const id = this.sample(logits, temperature);
      if (id === NovaTokenizer.EOS) break;
      if (id !== NovaTokenizer.PAD && id !== NovaTokenizer.BOS) out.push(id);
      ctx.shift(); ctx.push(id);
    }
    return this.tokenizer.decode(out);
  }

  reset() {
    this.tokenizer = new NovaTokenizer();
    this.trained = false;
    this.trainSteps = 0;
    this.lastLoss = null;
    this.corpusChars = 0;
    this._initWeights(this.vocabSize);
  }

  // ---------- persistence ----------

  toJSON() {
    return {
      version: 1,
      config: this.config,
      tokenizer: this.tokenizer.toJSON(),
      trained: this.trained,
      trainSteps: this.trainSteps,
      lastLoss: this.lastLoss,
      corpusChars: this.corpusChars,
      emb: Nova1Model._f32ToB64(this.emb),
      W1: Nova1Model._f32ToB64(this.W1),
      b1: Nova1Model._f32ToB64(this.b1),
      W2: Nova1Model._f32ToB64(this.W2),
      b2: Nova1Model._f32ToB64(this.b2),
    };
  }

  static fromJSON(o) {
    if (!o || o.version !== 1) throw new Error('Unknown Nova checkpoint version.');
    const m = new Nova1Model(o.config);
    m.tokenizer = NovaTokenizer.fromJSON(o.tokenizer);
    const v = m.vocabSize;
    const { context, embedDim, hiddenDim } = m.config;
    m.emb = Nova1Model._b64ToF32(o.emb);
    m.W1 = Nova1Model._b64ToF32(o.W1);
    m.b1 = Nova1Model._b64ToF32(o.b1);
    m.W2 = Nova1Model._b64ToF32(o.W2);
    m.b2 = Nova1Model._b64ToF32(o.b2);
    // validate shapes
    if (m.emb.length !== v * embedDim ||
        m.W1.length !== context * embedDim * hiddenDim ||
        m.b1.length !== hiddenDim ||
        m.W2.length !== hiddenDim * v ||
        m.b2.length !== v) {
      throw new Error('Saved checkpoint does not match the tokenizer vocabulary.');
    }
    m.trained = !!o.trained;
    m.trainSteps = o.trainSteps || 0;
    m.lastLoss = o.lastLoss ?? null;
    m.corpusChars = o.corpusChars || 0;
    return m;
  }

  static _f32ToB64(a) {
    const u = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
    let s = '';
    for (let i = 0; i < u.length; i += 8192) {
      s += String.fromCharCode.apply(null, u.subarray(i, i + 8192));
    }
    return btoa(s);
  }

  static _b64ToF64Safe(b64) { return Nova1Model._b64ToF32(b64); }

  static _b64ToF32(b64) {
    const bin = atob(b64);
    const u = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return new Float32Array(u.buffer);
  }
}
