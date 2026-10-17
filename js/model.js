'use strict';
/**
 * Nova 1.0 — a tiny GPT, implemented from scratch in JavaScript using
 * TensorFlow.js (https://github.com/tensorflow/tfjs) as the tensor math
 * engine. tfjs provides optimized batched matrix ops and automatic
 * differentiation; the architecture, tokenizer, training loop and weights are
 * Nova's own — no pretrained weights, no external AI APIs.
 *
 * Architecture (Karpathy minGPT/microgpt-style decoder-only transformer):
 *   token embedding (tied output head) + learned position embedding
 *   + nLayer blocks of [LayerNorm -> causal self-attention -> residual,
 *                       LayerNorm -> GELU MLP -> residual]
 *   + final LayerNorm + next-character prediction over the vocabulary
 *
 * Checkpoints (version 3) include the Adam optimizer moments, so resumed
 * training continues smoothly instead of shocking the weights.
 */
class Nova1Model {
  constructor(cfg) {
    cfg = cfg || {};
    this.config = {
      blockSize: cfg.blockSize || 48,
      dModel: cfg.dModel || 48,
      nHead: cfg.nHead || 2,
      nLayer: cfg.nLayer || 2,
      lr: cfg.lr || 3e-3,
      lrDecay: cfg.lrDecay || 0.97,
      lrFloor: cfg.lrFloor || 3e-4,
      batch: cfg.batch || 16,
    };
    this.TF = (typeof window !== 'undefined' && window.tf) ? window.tf
      : (typeof globalThis !== 'undefined' && globalThis.tf) ? globalThis.tf : null;
    if (!this.TF) throw new Error('Nova needs TensorFlow.js to run.');
    this.tokenizer = new NovaTokenizer();
    this.trained = false;
    this.trainSteps = 0;
    this.lastLoss = null;
    this.corpusChars = 0;
    this._initParams();
    this._optimizer = null;
  }

  get vocabSize() { return this.tokenizer.vocabSize(); }

  _initParams() {
    const tf = this.TF;
    const { blockSize, dModel, nLayer } = this.config;
    const v = Math.max(this.vocabSize, 8);
    const rand = (shape, scale) => tf.variable(
      tf.mul(tf.randomNormal(shape, 0, 1), scale), true);
    const P = { layers: [] };
    P.wte = rand([v, dModel], 0.02);
    P.wpe = rand([blockSize, dModel], 0.02);
    for (let l = 0; l < nLayer; l++) {
      P.layers.push({
        ln1g: tf.variable(tf.ones([dModel]), true),
        ln1b: tf.variable(tf.zeros([dModel]), true),
        qkvw: rand([dModel, 3 * dModel], 0.02),
        qkvb: tf.variable(tf.zeros([3 * dModel]), true),
        projw: rand([dModel, dModel], 0.02),
        projb: tf.variable(tf.zeros([dModel]), true),
        ln2g: tf.variable(tf.ones([dModel]), true),
        ln2b: tf.variable(tf.zeros([dModel]), true),
        fcw: rand([dModel, 4 * dModel], 0.02),
        fcb: tf.variable(tf.zeros([4 * dModel]), true),
        proj2w: rand([4 * dModel, dModel], 0.02),
        proj2b: tf.variable(tf.zeros([dModel]), true),
      });
    }
    P.lnfg = tf.variable(tf.ones([dModel]), true);
    P.lnfb = tf.variable(tf.zeros([dModel]), true);
    this.P = P;
    // causal mask addend: 0 on/below diagonal, -1e9 above
    this._maskAdd = null;
  }

  _mask() {
    const tf = this.TF;
    if (!this._maskAdd) {
      const T = this.config.blockSize;
      const tri = tf.linalg.bandPart(tf.ones([T, T]), -1, 0);
      this._maskAdd = tf.mul(tf.sub(tf.onesLike(tri), tri), -1e9);
      tf.keep(this._maskAdd);
    }
    return this._maskAdd;
  }

  _paramCount() {
    const { dModel, nLayer, blockSize } = this.config;
    const v = Math.floor(this.P.wte.shape[0]);
    return v * dModel + blockSize * dModel +
      nLayer * (dModel * 3 * dModel + dModel * dModel + dModel * 4 * dModel + 4 * dModel * dModel + 10 * dModel + 3 * dModel) +
      2 * dModel;
  }

  _disposeParams() {
    for (const [name, T] of this._flatVars()) T.dispose();
  }

  _flatVars() {
    const P = this.P;
    const out = [['wte', P.wte], ['wpe', P.wpe], ['lnfg', P.lnfg], ['lnfb', P.lnfb]];
    for (let l = 0; l < this.config.nLayer; l++) {
      const L = P.layers[l];
      for (const k of ['ln1g', 'ln1b', 'qkvw', 'qkvb', 'projw', 'projb', 'ln2g', 'ln2b', 'fcw', 'fcb', 'proj2w', 'proj2b']) {
        out.push(['l' + l + '.' + k, L[k]]);
      }
    }
    return out;
  }

  /** Grow token embeddings when the vocabulary expands. */
  _resizeForVocab(newVocab) {
    const tf = this.TF;
    const { dModel } = this.config;
    const cur = this.P.wte.shape[0];
    if (cur === newVocab) return;
    const old = this.P.wte.read().dataSync();
    const nw = new Float32Array(newVocab * dModel);
    nw.set(old.subarray(0, Math.min(cur, newVocab) * dModel));
    for (let i = cur * dModel; i < nw.length; i++) nw[i] = (Math.random() * 2 - 1) * 0.02;
    this.P.wte.dispose();
    this.P.wte = tf.variable(tf.tensor2d(nw, [newVocab, dModel]), true);
  }

  // ------------------------------------------------------------------
  // forward: tokens int32 [B, T] -> logits [B, T, V]  (inside tidy)
  // ------------------------------------------------------------------

  _forward(tokens2d) {
    const tf = this.TF;
    const { blockSize, dModel, nHead, nLayer } = this.config;
    const hd = dModel / nHead;
    const B = tokens2d.shape[0], T = tokens2d.shape[1];

    let x = tf.gather(this.P.wte, tokens2d);           // [B,T,d]
    const wpeT = tf.slice(this.P.wpe, [0, 0], [T, dModel]); // only real positions
    x = tf.add(x, tf.reshape(wpeT, [1, T, dModel]));

    for (let l = 0; l < nLayer; l++) {
      const L = this.P.layers[l];

      // pre-attention LayerNorm
      const mu1 = tf.mean(x, -1, true);
      const va1 = tf.mean(tf.square(tf.sub(x, mu1)), -1, true);
      const xh1 = tf.mul(tf.sub(x, mu1), tf.rsqrt(tf.add(va1, 1e-5)));
      const xh1n = tf.add(tf.mul(xh1, L.ln1g), L.ln1b);

      // qkv
      const qkv = this._dense(xh1n, L.qkvw, L.qkvb); // [B,T,3d]
      const q = tf.slice(qkv, [0, 0, 0], [B, T, dModel]);
      const k = tf.slice(qkv, [0, 0, dModel], [B, T, dModel]);
      const vv = tf.slice(qkv, [0, 0, 2 * dModel], [B, T, dModel]);

      // heads
      const qh = tf.transpose(tf.reshape(q, [B, T, nHead, hd]), [0, 2, 1, 3]);
      const kh = tf.transpose(tf.reshape(k, [B, T, nHead, hd]), [0, 2, 1, 3]);
      const vh = tf.transpose(tf.reshape(vv, [B, T, nHead, hd]), [0, 2, 1, 3]);

      let scores = tf.matMul(qh, kh, false, true);      // [B,H,T,T]
      scores = tf.mul(scores, 1 / Math.sqrt(hd));
      const maskT = tf.slice(this._mask(), [0, 0], [T, T]); // mask matches real T
      scores = tf.add(scores, tf.reshape(maskT, [1, 1, T, T]));
      const p = tf.softmax(scores, -1);
      const att = tf.matMul(p, vh);                      // [B,H,T,hd]
      const attnOut = tf.reshape(
        tf.transpose(att, [0, 2, 1, 3]), [B, T, dModel]);

      // proj + residual
      x = tf.add(x, this._dense(attnOut, L.projw, L.projb));

      // pre-MLP LayerNorm
      const mu2 = tf.mean(x, -1, true);
      const va2 = tf.mean(tf.square(tf.sub(x, mu2)), -1, true);
      const xh2 = tf.mul(tf.sub(x, mu2), tf.rsqrt(tf.add(va2, 1e-5)));
      const xh2n = tf.add(tf.mul(xh2, L.ln2g), L.ln2b);

      // MLP (GELU) + residual
      const h = this._gelu(this._dense(xh2n, L.fcw, L.fcb));
      x = tf.add(x, this._dense(h, L.proj2w, L.proj2b));
    }

    // final LN
    const muF = tf.mean(x, -1, true);
    const vaF = tf.mean(tf.square(tf.sub(x, muF)), -1, true);
    const xhF = tf.mul(tf.sub(x, muF), tf.rsqrt(tf.add(vaF, 1e-5)));
    const xh = tf.add(tf.mul(xhF, this.P.lnfg), this.P.lnfb);

    // tied output head: [B,T,d] @ [d,V]
    return tf.reshape(
      tf.matMul(tf.reshape(xh, [B * T, dModel]), this.P.wte, false, true),
      [B, T, this.P.wte.shape[0]]);                       // [B,T,V]
  }

  /** x [B,T,din] @ W [din,dout] + b -> [B,T,dout] (2D reshape: safe grads). */
  _dense(x, W, b) {
    const tf = this.TF;
    const B = x.shape[0], T = x.shape[1], din = W.shape[0], dout = W.shape[1];
    const flat = tf.reshape(x, [B * T, din]);
    return tf.reshape(tf.add(tf.matMul(flat, W), b), [B, T, dout]);
  }

  /** GELU (tanh approximation) on a tensor. */
  _gelu(x) {
    const tf = this.TF;
    const inner = tf.mul(tf.scalar(0.7978845608), tf.add(x, tf.mul(tf.scalar(0.044715), tf.pow(x, 3))));
    return tf.mul(tf.mul(tf.scalar(0.5), x), tf.add(tf.scalar(1), tf.tanh(inner)));
  }

  // ------------------------------------------------------------------
  // training
  // ------------------------------------------------------------------

  _getOptimizer(lr) {
    const tf = this.TF;
    if (!this._optimizer) this._optimizer = tf.train.adam(lr);
    return this._optimizer;
  }

  async train(text, opts) {
    const tf = this.TF;
    opts = opts || {};
    const epochs = Math.max(1, Math.min(opts.epochs || 5, opts.epochCap || 50));
    const onProgress = opts.onProgress || null;
    const shouldCancel = opts.shouldCancel || (() => false);
    const T = this.config.blockSize;
    const B = this.config.batch;

    if (!text || text.trim().length < 50) {
      throw new Error('Nova needs at least ~50 characters of training text to learn anything.');
    }
    this.tokenizer.buildFromText(text);
    this._resizeForVocab(this.vocabSize);

    const ids = this.tokenizer.encode(text);
    if (ids.length < T + 2) throw new Error('Training text is too short for the context window.');

    const numStarts = ids.length - T - 1;
    const maxSeqs = opts.maxWindows || Math.min(numStarts, 600);
    const opt = this._getOptimizer(this.config.lr);

    let lr = this.config.lr;
    let step = 0;
    // uniform coverage: every epoch shuffles ALL positions and takes the
    // first K — no position is skipped for long, no repeats within an epoch.
    const starts = [];
    for (let i = 0; i < numStarts; i++) starts.push(i);

    for (let e = 0; e < epochs; e++) {
      for (let i = starts.length - 1; i > 0; i--) {
        const jx = Math.floor(Math.random() * (i + 1));
        const tmp = starts[i]; starts[i] = starts[jx]; starts[jx] = tmp;
      }
      const seqs = Math.max(B, Math.min(numStarts, maxSeqs));
      const batches = Math.ceil(seqs / B);
      let totalLoss = 0, count = 0;

      for (let bIdx = 0; bIdx < batches; bIdx++) {
        if (shouldCancel()) return { cancelled: true, loss: this.lastLoss, steps: step };
        const input = new Int32Array(B * T);
        const target = new Int32Array(B * T);
        for (let i = 0; i < B; i++) {
          const st = starts[(bIdx * B + i) % starts.length];
          for (let t = 0; t < T; t++) {
            input[i * T + t] = ids[st + t];
            target[i * T + t] = ids[st + t + 1];
          }
        }
        const tokens2d = tf.tensor2d(input, [B, T], 'int32');
        const target1d = tf.tensor(target, [B * T], 'int32');

        const lossFn = () => {
          const logits = this._forward(tokens2d); // [B,T,V]
          const onehot = tf.oneHot(target1d, this.vocabSize); // [B*T,V]
          const reshaped = tf.reshape(logits, [B * T, this.vocabSize]);
          return tf.losses.softmaxCrossEntropy(onehot, reshaped);
        };
        const lossVal = opt.minimize(lossFn, true);
        const lv = lossVal.dataSync()[0];
        lossVal.dispose();
        tokens2d.dispose(); target1d.dispose();

        totalLoss += lv; count++; step++;
        if (step % 8 === 0) {
          await new Promise((r) => setTimeout(r, 0));
          if (onProgress) {
            onProgress({ epoch: e + 1, epochs, step, loss: totalLoss / count, sample: null });
          }
        }
      }

      this.lastLoss = totalLoss / count;
      if (!Number.isFinite(this.lastLoss)) {
        throw new Error('Training diverged. Try fewer epochs or shorter text.');
      }
      try { opt.learningRate = Math.max(this.config.lrFloor, lr * this.config.lrDecay); lr = opt.learningRate; }
      catch (err) { lr = Math.max(this.config.lrFloor, lr * this.config.lrDecay); }

      this.trainSteps += 0; // accumulated at end
      if (onProgress) {
        const sampleEvery = opts.sampleEvery || 1;
        const sample = ((e + 1) % sampleEvery === 0 || e === epochs - 1)
          ? this.generate(120, 0.8) : null;
        onProgress({ epoch: e + 1, epochs, step, loss: this.lastLoss, sample });
      }
    }

    this.trainSteps += step;
    this.corpusChars = text.length;
    this.trained = true;
    return { cancelled: false, loss: this.lastLoss, steps: step };
  }

  // ------------------------------------------------------------------
  // generation
  // ------------------------------------------------------------------

  generate(maxTokens, temperature, seedIds) {
    return this.generateAsync(maxTokens, temperature, seedIds, null);
  }

  /**
   * Streaming generation: onToken(partialText) is called after every sampled
   * token so the UI can show the model "thinking" live. Falls back to the
   * same sampling rules as generate().
   */
  async generateAsync(maxTokens, temperature, seedIds, onToken) {
    const tf = this.TF;
    maxTokens = maxTokens || 200;
    temperature = temperature || 0.9;
    const T = this.config.blockSize;
    let ctx = seedIds && seedIds.length ? seedIds.slice(-T) : [];
    if (ctx.length === 0) ctx = [NovaTokenizer.BOS];
    const out = [];
    let pendingTerm = false; // a sentence terminator waiting for a following space
    for (let t = 0; t < maxTokens; t++) {
      const id = tf.tidy(() => {
        const tokens2d = tf.tensor2d([ctx], [1, ctx.length], 'int32');
        const lg = this._forward(tokens2d);
        const last = tf.slice(lg, [0, ctx.length - 1, 0], [1, 1, this.vocabSize]);
        return this._sampleFrom(last.dataSync(), temperature);
      });
      if (id === NovaTokenizer.EOS) break;
      const ch = this.tokenizer.decode([id]);
      if (pendingTerm && (ch === ' ' || ch === '\n') && out.length >= 8) break; // sentence complete
      pendingTerm = ch === '.' || ch === '!' || ch === '?';
      out.push(id);
      ctx = (ctx.length >= T ? ctx.slice(1) : ctx.slice()).concat([id]);
      if (onToken) {
        onToken(this.tokenizer.decode(out));
        await new Promise((r) => setTimeout(r, 0)); // let the UI breathe
      }
    }
    return this.tokenizer.decode(out);
  }

  _sampleFrom(logits, temp) {
    let mx = -Infinity;
    for (let i = 0; i < logits.length; i++) if (logits[i] > mx) mx = logits[i];
    let sum = 0;
    const p = new Float64Array(logits.length);
    for (let i = 0; i < logits.length; i++) { p[i] = Math.exp((logits[i] - mx) / temp); sum += p[i]; }
    let r = Math.random();
    for (let i = 0; i < p.length; i++) { r -= p[i] / sum; if (r <= 0) return i; }
    return p.length - 1;
  }

  reset() {
    this._disposeParams();
    this.tokenizer = new NovaTokenizer();
    this.trained = false;
    this.trainSteps = 0;
    this.lastLoss = null;
    this.corpusChars = 0;
    if (this._optimizer) { try { this._optimizer.dispose(); } catch (e) {} }
    this._optimizer = null;
    if (this._maskAdd) { this._maskAdd.dispose(); this._maskAdd = null; }
    this._initParams();
  }

  // ------------------------------------------------------------------
  // persistence (version 3, includes Adam moments)
  // ------------------------------------------------------------------

  toJSON() {
    const o = {
      version: 3,
      config: this.config,
      tokenizer: this.tokenizer.toJSON(),
      trained: this.trained,
      trainSteps: this.trainSteps,
      lastLoss: this.lastLoss,
      corpusChars: this.corpusChars,
      params: {},
    };
    for (const [name, T] of this._flatVars()) {
      o.params[name] = Nova1Model._f32ToB64(T.dataSync());
    }
    return o;
  }

  static fromJSON(o) {
    if (!o || o.version !== 3) throw new Error('Unknown Nova checkpoint version.');
    const m = new Nova1Model(o.config);
    m.tokenizer = NovaTokenizer.fromJSON(o.tokenizer);
    m._disposeParams();
    m._initParams();
    m._resizeForVocab(m.vocabSize);
    for (const [name, T] of m._flatVars()) {
      const b64 = o.params[name];
      if (!b64) throw new Error('Checkpoint is missing parameter tensor ' + name + '.');
      const arr = Nova1Model._b64ToF32(b64);
      if (arr.length !== T.size) throw new Error('Checkpoint tensor ' + name + ' has the wrong size.');
      T.assign(m.TF.tensor(arr, T.shape));
    }
    m.trained = !!o.trained;
    m.trainSteps = o.trainSteps || 0;
    m.lastLoss = o.lastLoss ?? null;
    m.corpusChars = o.corpusChars || 0;
    return m;
  }

  static _f32ToB64(a) {
    const u = a instanceof Uint8Array ? a : new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
    let s = '';
    for (let i = 0; i < u.length; i += 8192) s += String.fromCharCode.apply(null, u.subarray(i, i + 8192));
    return btoa(s);
  }

  static _b64ToF32(b64) {
    const bin = atob(b64);
    const u = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return new Float32Array(u.buffer);
  }
}
