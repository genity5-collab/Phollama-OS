'use strict';
/**
 * Nova 1.0 — a tiny GPT, implemented from scratch in plain JavaScript.
 *
 * Architecture follows Karpathy's minGPT / microgpt (decoder-only transformer):
 *   token embedding (tied with the output head)
 *   + learned position embedding
 *   + nLayer blocks of: [LayerNorm -> causal self-attention -> residual,
 *                         LayerNorm -> GELU MLP -> residual]
 *   + final LayerNorm
 *   + next-character prediction over the whole vocabulary
 *
 * Training uses the Adam optimizer (Kingma & Ba), as in minGPT. The previous
 * Nova architecture (fixed-window neural probabilistic LM with plain SGD)
 * plateaued at a loss where speech stayed garbled; research pointed to
 * exactly this upgrade (github.com/karpathy/minGPT, github.com/jayyvk/
 * trainmyowngpt — a JS port proving models this size train in seconds).
 *
 * No libraries, no external APIs, no pretrained weights. The honest
 * consequence of a model this small: it learns the character patterns of the
 * text it is trained on and continues your messages accordingly. It is NOT a
 * ChatGPT-class assistant.
 */
class Nova1Model {
  constructor(cfg) {
    cfg = cfg || {};
    this.config = {
      blockSize: cfg.blockSize || 32,  // context length T in characters
      dModel: cfg.dModel || 32,        // embedding width
      nHead: cfg.nHead || 2,           // attention heads
      nLayer: cfg.nLayer || 2,         // transformer blocks
      lr: cfg.lr || 3e-3,             // Adam learning rate
      lrDecay: cfg.lrDecay || 0.97,    // per-epoch decay
      lrFloor: cfg.lrFloor || 3e-4,
      batch: cfg.batch || 12,
    };
    this.tokenizer = new NovaTokenizer();
    this.trained = false;
    this.trainSteps = 0;
    this.lastLoss = null;
    this.corpusChars = 0;
    this._initParams();
    this._resetOptimizer();
  }

  get vocabSize() { return this.tokenizer.vocabSize(); }

  _initParams() {
    const { blockSize, dModel, nLayer } = this.config;
    const v = Math.max(this.vocabSize, 8);
    const P = { layers: [] };
    P.wte = Nova1Model._rand(v * dModel, 0.02);          // token embeddings (tied output head)
    P.wpe = Nova1Model._rand(blockSize * dModel, 0.02);  // position embeddings
    for (let l = 0; l < nLayer; l++) {
      P.layers.push({
        ln1g: new Float32Array(dModel).fill(1), ln1b: new Float32Array(dModel),
        qkvw: Nova1Model._rand(dModel * 3 * dModel, 0.02), qkvb: new Float32Array(3 * dModel),
        projw: Nova1Model._rand(dModel * dModel, 0.02), projb: new Float32Array(dModel),
        ln2g: new Float32Array(dModel).fill(1), ln2b: new Float32Array(dModel),
        fcw: Nova1Model._rand(dModel * 4 * dModel, 0.02), fcb: new Float32Array(4 * dModel),
        proj2w: Nova1Model._rand(4 * dModel * dModel, 0.02), proj2b: new Float32Array(dModel),
      });
    }
    P.lnfg = new Float32Array(dModel).fill(1);
    P.lnfb = new Float32Array(dModel);
    this.P = P;
  }

  _resetOptimizer() { this._t = 0; this._adam = null; }

  static _rand(n, scale) {
    const a = new Float32Array(n);
    for (let i = 0; i < n; i++) a[i] = (Math.random() * 2 - 1) * scale;
    return a;
  }

  _paramCount() {
    const { dModel, nLayer } = this.config;
    const v = Math.floor(this.P.wte.length / dModel);
    return v * dModel + this.config.blockSize * dModel +
      nLayer * (dModel * 3 * dModel + dModel * dModel + dModel * 4 * dModel + 4 * dModel * dModel + 10 * dModel + 3 * dModel) +
      2 * dModel;
  }

  /** Grow the token-embedding/output table when the vocabulary expands. */
  _resizeForVocab(newVocab) {
    const { dModel } = this.config;
    const cur = Math.floor(this.P.wte.length / dModel);
    if (cur === newVocab) return;
    const nw = new Float32Array(newVocab * dModel);
    nw.set(this.P.wte.subarray(0, Math.min(cur, newVocab) * dModel));
    for (let i = cur * dModel; i < nw.length; i++) nw[i] = (Math.random() * 2 - 1) * 0.02;
    this.P.wte = nw;
  }

  // ------------------------------------------------------------------
  // layer norm (forward + backward)
  // ------------------------------------------------------------------

  _lnFwd(x, off, g, b, out) {
    const d = this.config.dModel;
    let mu = 0;
    for (let j = 0; j < d; j++) mu += x[off + j];
    mu /= d;
    let vs = 0;
    for (let j = 0; j < d; j++) { const dv = x[off + j] - mu; vs += dv * dv; }
    const rstd = 1 / Math.sqrt(vs / d + 1e-5);
    const xh = new Float32Array(d);
    for (let j = 0; j < d; j++) {
      xh[j] = (x[off + j] - mu) * rstd;
      out[j] = xh[j] * g[j] + b[j];
    }
    return { xhat: xh, rstd };
  }

  _lnBwd(dy, g, cache, dxAcc, dxOff, dg, db) {
    const d = this.config.dModel;
    const { xhat, rstd } = cache;
    const dxh = new Float32Array(d);
    for (let j = 0; j < d; j++) {
      dxh[j] = dy[j] * g[j];
      dg[j] += dy[j] * xhat[j];
      db[j] += dy[j];
    }
    let m1 = 0, m2 = 0;
    for (let j = 0; j < d; j++) { m1 += dxh[j]; m2 += dxh[j] * xhat[j]; }
    m1 /= d; m2 /= d;
    for (let j = 0; j < d; j++) {
      dxAcc[dxOff + j] += (dxh[j] - m1 - xhat[j] * m2) * rstd;
    }
  }

  _gelu(x) { return 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x))); }
  _geluGrad(x) {
    const c = 0.7978845608;
    const inner = c * (x + 0.044715 * x * x * x);
    const t = Math.tanh(inner);
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * c * (1 + 3 * 0.044715 * x * x);
  }

  // ------------------------------------------------------------------
  // forward for one sequence (no cache — training and generation both
  // recompute; at this scale full recompute per generated char is ~10ms)
  // ------------------------------------------------------------------

  _forward(tokens) {
    const { blockSize, dModel, nHead, nLayer } = this.config;
    const hd = dModel / nHead;
    const n = tokens.length;
    const scale = 1 / Math.sqrt(hd);

    const x = new Float32Array(n * dModel);
    for (let t = 0; t < n; t++) {
      const tokOff = (tokens[t] % Math.max(1, Math.floor(this.P.wte.length / dModel))) * dModel;
      const posOff = (t % blockSize) * dModel;
      for (let j = 0; j < dModel; j++) x[t * dModel + j] = this.P.wte[tokOff + j] + this.P.wpe[posOff + j];
    }

    const A = { tokens, n, xIn: [x], blocks: [] };

    let cur = x;
    for (let l = 0; l < nLayer; l++) {
      const L = this.P.layers[l];
      const B = { xIn: cur };
      A.xIn.push(cur);

      // --- pre-attention LN ---
      const xhat1 = new Float32Array(n * dModel);
      const ln1c = [];
      for (let t = 0; t < n; t++) {
        const y = new Float32Array(dModel);
        ln1c.push(this._lnFwd(cur, t * dModel, L.ln1g, L.ln1b, y));
        xhat1.set(y, t * dModel);
      }
      B.ln1c = ln1c; B.xhat1 = xhat1;

      // --- qkv ---
      const qkv = new Float32Array(n * 3 * dModel);
      for (let t = 0; t < n; t++) {
        for (let o = 0; o < 3 * dModel; o++) {
          let s = L.qkvb[o];
          for (let i = 0; i < dModel; i++) s += xhat1[t * dModel + i] * L.qkvw[i * 3 * dModel + o];
          qkv[t * 3 * dModel + o] = s;
        }
      }
      B.qkv = qkv;

      // --- causal attention per head ---
      const attnOut = new Float32Array(n * dModel);
      const heads = [];
      for (let h = 0; h < nHead; h++) {
        const Q = new Float32Array(n * hd), K = new Float32Array(n * hd), V = new Float32Array(n * hd);
        for (let t = 0; t < n; t++) {
          const base = t * 3 * dModel + h * hd;
          for (let j = 0; j < hd; j++) {
            Q[t * hd + j] = qkv[base + j];
            K[t * hd + j] = qkv[dModel + base + j];
            V[t * hd + j] = qkv[2 * dModel + base + j];
          }
        }
        const p = new Float32Array(n * n);
        for (let t = 0; t < n; t++) {
          const sc = new Float32Array(t + 1);
          let mx = -Infinity;
          for (let k = 0; k <= t; k++) {
            let s = 0;
            for (let j = 0; j < hd; j++) s += Q[t * hd + j] * K[k * hd + j];
            sc[k] = s * scale;
            if (sc[k] > mx) mx = sc[k];
          }
          let sum = 0;
          const ex = new Float32Array(t + 1);
          for (let k = 0; k <= t; k++) { ex[k] = Math.exp(sc[k] - mx); sum += ex[k]; }
          for (let k = 0; k <= t; k++) {
            const pk = ex[k] / sum;
            p[t * n + k] = pk;
            for (let j = 0; j < hd; j++) attnOut[t * dModel + h * hd + j] += pk * V[k * hd + j];
          }
        }
        heads.push({ Q, K, V, p });
      }
      B.heads = heads; B.attnOut = attnOut;

      // --- proj + residual ---
      const xMid = new Float32Array(n * dModel);
      for (let t = 0; t < n; t++) {
        for (let o = 0; o < dModel; o++) {
          let s = L.projb[o];
          for (let i = 0; i < dModel; i++) s += attnOut[t * dModel + i] * L.projw[i * dModel + o];
          xMid[t * dModel + o] = cur[t * dModel + o] + s;
        }
      }
      B.xMid = xMid;

      // --- pre-MLP LN ---
      const yhat = new Float32Array(n * dModel);
      const ln2c = [];
      for (let t = 0; t < n; t++) {
        const y = new Float32Array(dModel);
        ln2c.push(this._lnFwd(xMid, t * dModel, L.ln2g, L.ln2b, y));
        yhat.set(y, t * dModel);
      }
      B.ln2c = ln2c; B.yhat = yhat;

      // --- MLP ---
      const fch = new Float32Array(n * 4 * dModel);
      const gel = new Float32Array(n * 4 * dModel);
      for (let t = 0; t < n; t++) {
        for (let o = 0; o < 4 * dModel; o++) {
          let s = L.fcb[o];
          for (let i = 0; i < dModel; i++) s += yhat[t * dModel + i] * L.fcw[i * 4 * dModel + o];
          fch[t * 4 * dModel + o] = s;
          gel[t * 4 * dModel + o] = this._gelu(s);
        }
      }
      const xOut = new Float32Array(n * dModel);
      for (let t = 0; t < n; t++) {
        for (let o = 0; o < dModel; o++) {
          let s = L.proj2b[o];
          for (let i = 0; i < 4 * dModel; i++) s += gel[t * 4 * dModel + i] * L.proj2w[i * dModel + o];
          xOut[t * dModel + o] = xMid[t * dModel + o] + s;
        }
      }
      B.fch = fch; B.gel = gel; B.xOut = xOut;
      A.blocks.push(B);
      cur = xOut;
    }
    A.xFinal = cur;

    // --- final LN + tied logits ---
    const xh = new Float32Array(n * dModel);
    const lnfc = [];
    for (let t = 0; t < n; t++) {
      const y = new Float32Array(dModel);
      lnfc.push(this._lnFwd(cur, t * dModel, this.P.lnfg, this.P.lnfb, y));
      xh.set(y, t * dModel);
    }
    const v = this.vocabSize;
    const logits = new Float32Array(n * v);
    for (let t = 0; t < n; t++) {
      for (let vi = 0; vi < v; vi++) {
        let s = 0;
        const vo = vi * dModel;
        for (let j = 0; j < dModel; j++) s += xh[t * dModel + j] * this.P.wte[vo + j];
        logits[t * v + vi] = s;
      }
    }
    A.lnfc = lnfc; A.xh = xh; A.logits = logits;
    return A;
  }

  // ------------------------------------------------------------------
  // backward for one sequence, accumulating grads into G; returns loss sum
  // ------------------------------------------------------------------

  _backward(A, targets, G) {
    const { dModel, nHead, nLayer } = this.config;
    const hd = dModel / nHead;
    const n = A.n;
    const v = this.vocabSize;

    // softmax CE
    let lossSum = 0;
    const dlog = new Float32Array(n * v);
    for (let t = 0; t < n; t++) {
      const off = t * v;
      let mx = -Infinity;
      for (let i = 0; i < v; i++) if (A.logits[off + i] > mx) mx = A.logits[off + i];
      let sum = 0;
      for (let i = 0; i < v; i++) { const e = Math.exp(A.logits[off + i] - mx); dlog[off + i] = e; sum += e; }
      for (let i = 0; i < v; i++) dlog[off + i] /= sum;
      lossSum += -Math.log(Math.max(dlog[off + targets[t]], 1e-9));
      dlog[off + targets[t]] -= 1;
    }

    // tied head
    const dxh = new Float32Array(n * dModel);
    for (let t = 0; t < n; t++) {
      for (let vi = 0; vi < v; vi++) {
        const dl = dlog[t * v + vi];
        if (dl === 0) continue;
        const vo = vi * dModel;
        for (let j = 0; j < dModel; j++) {
          dxh[t * dModel + j] += dl * this.P.wte[vo + j];
          G.wte[vo + j] += dl * A.xh[t * dModel + j];
        }
      }
    }

    // final LN
    const dx = new Float32Array(n * dModel);
    for (let t = 0; t < n; t++) {
      this._lnBwd(dxh.subarray(t * dModel, (t + 1) * dModel), this.P.lnfg, A.lnfc[t], dx, t * dModel, G.lnfg, G.lnfb);
    }

    for (let l = nLayer - 1; l >= 0; l--) {
      const L = this.P.layers[l];
      const Lg = G.layers[l];
      const B = A.blocks[l];

      // ---- MLP branch (skip + branch add into dxMid) ----
      const dgel = new Float32Array(n * 4 * dModel);
      for (let t = 0; t < n; t++) {
        const dres = dx.subarray(t * dModel, (t + 1) * dModel);
        for (let i = 0; i < 4 * dModel; i++) {
          let s = 0;
          for (let o = 0; o < dModel; o++) {
            Lg.proj2w[i * dModel + o] += B.gel[t * 4 * dModel + i] * dres[o];
            s += L.proj2w[i * dModel + o] * dres[o];
          }
          dgel[t * 4 * dModel + i] = s;
        }
        for (let o = 0; o < dModel; o++) Lg.proj2b[o] += dres[o];
      }
      const dfc = new Float32Array(n * 4 * dModel);
      for (let i = 0; i < n * 4 * dModel; i++) dfc[i] = dgel[i] * this._geluGrad(B.fch[i]);

      const dyhat = new Float32Array(n * dModel);
      for (let t = 0; t < n; t++) {
        for (let i = 0; i < dModel; i++) {
          let s = 0;
          for (let o = 0; o < 4 * dModel; o++) {
            Lg.fcw[i * 4 * dModel + o] += B.yhat[t * dModel + i] * dfc[t * 4 * dModel + o];
            s += L.fcw[i * 4 * dModel + o] * dfc[t * 4 * dModel + o];
          }
          dyhat[t * dModel + i] = s;
        }
        for (let o = 0; o < 4 * dModel; o++) Lg.fcb[o] += dfc[t * 4 * dModel + o];
      }

      const dxMid = new Float32Array(n * dModel); // grad wrt xMid = skip(dx) + ln2 branch
      for (let t = 0; t < n; t++) {
        for (let j = 0; j < dModel; j++) dxMid[t * dModel + j] += dx[t * dModel + j];
        this._lnBwd(dyhat.subarray(t * dModel, (t + 1) * dModel), L.ln2g, B.ln2c[t], dxMid, t * dModel, Lg.ln2g, Lg.ln2b);
      }

      // ---- attention branch ----
      const dattn = new Float32Array(n * dModel);
      for (let t = 0; t < n; t++) {
        const dres = dxMid.subarray(t * dModel, (t + 1) * dModel);
        for (let i = 0; i < dModel; i++) {
          let s = 0;
          for (let o = 0; o < dModel; o++) {
            Lg.projw[i * dModel + o] += B.attnOut[t * dModel + i] * dres[o];
            s += L.projw[i * dModel + o] * dres[o];
          }
          dattn[t * dModel + i] = s;
        }
        for (let o = 0; o < dModel; o++) Lg.projb[o] += dres[o];
      }

      const dqkv = new Float32Array(n * 3 * dModel);
      for (let h = 0; h < nHead; h++) {
        const H = B.heads[h];
        const { Q, K, V, p } = H;
        const dQ = new Float32Array(n * hd), dK = new Float32Array(n * hd), dV = new Float32Array(n * hd);
        const scale = Math.sqrt(hd); // 1/scale folded below
        for (let t = 0; t < n; t++) {
          const dout = dattn.subarray(t * dModel + h * hd, t * dModel + (h + 1) * hd);
          let dotsum = 0;
          const dp = new Float32Array(t + 1);
          for (let k = 0; k <= t; k++) {
            let dotp = 0;
            for (let j = 0; j < hd; j++) {
              dotp += dout[j] * V[k * hd + j];
              dV[k * hd + j] += p[t * n + k] * dout[j];
            }
            dp[k] = dotp;
            dotsum += p[t * n + k] * dotp;
          }
          for (let k = 0; k <= t; k++) {
            const ds = p[t * n + k] * (dp[k] - dotsum) / scale;
            for (let j = 0; j < hd; j++) {
              dQ[t * hd + j] += ds * K[k * hd + j];
              dK[k * hd + j] += ds * Q[t * hd + j];
            }
          }
        }
        for (let t = 0; t < n; t++) {
          const base = t * 3 * dModel + h * hd;
          for (let j = 0; j < hd; j++) {
            dqkv[base + j] = dQ[t * hd + j];
            dqkv[dModel + base + j] = dK[t * hd + j];
            dqkv[2 * dModel + base + j] = dV[t * hd + j];
          }
        }
      }

      const dxhat = new Float32Array(n * dModel);
      for (let t = 0; t < n; t++) {
        for (let i = 0; i < dModel; i++) {
          let s = 0;
          for (let o = 0; o < 3 * dModel; o++) {
            Lg.qkvw[i * 3 * dModel + o] += B.xhat1[t * dModel + i] * dqkv[t * 3 * dModel + o];
            s += L.qkvw[i * 3 * dModel + o] * dqkv[t * 3 * dModel + o];
          }
          dxhat[t * dModel + i] = s;
        }
        for (let o = 0; o < 3 * dModel; o++) Lg.qkvb[o] += dqkv[t * 3 * dModel + o];
      }

      const dxIn = new Float32Array(n * dModel); // grad wrt layer input = skip(dxMid) + ln1 branch
      for (let t = 0; t < n; t++) {
        for (let j = 0; j < dModel; j++) dxIn[t * dModel + j] += dxMid[t * dModel + j];
        this._lnBwd(dxhat.subarray(t * dModel, (t + 1) * dModel), L.ln1g, B.ln1c[t], dxIn, t * dModel, Lg.ln1g, Lg.ln1b);
      }
      dx.set(dxIn);
    }

    // ---- embedding grads: x = wte[token] + wpe[pos] ----
    for (let t = 0; t < n; t++) {
      const tok = A.tokens[t];
      const pos = t % this.config.blockSize;
      for (let j = 0; j < dModel; j++) {
        const g = dx[t * dModel + j];
        G.wte[tok * dModel + j] += g;
        G.wpe[pos * dModel + j] += g;
      }
    }

    return lossSum;
  }

  _newGrads() {
    const { dModel, nLayer, blockSize } = this.config;
    const v = Math.floor(this.P.wte.length / dModel);
    const G = {
      wte: new Float32Array(v * dModel),
      wpe: new Float32Array(blockSize * dModel),
      layers: [], lnfg: new Float32Array(dModel), lnfb: new Float32Array(dModel),
    };
    for (let l = 0; l < nLayer; l++) {
      G.layers.push({
        ln1g: new Float32Array(dModel), ln1b: new Float32Array(dModel),
        qkvw: new Float32Array(dModel * 3 * dModel), qkvb: new Float32Array(3 * dModel),
        projw: new Float32Array(dModel * dModel), projb: new Float32Array(dModel),
        ln2g: new Float32Array(dModel), ln2b: new Float32Array(dModel),
        fcw: new Float32Array(dModel * 4 * dModel), fcb: new Float32Array(4 * dModel),
        proj2w: new Float32Array(4 * dModel * dModel), proj2b: new Float32Array(dModel),
      });
    }
    return G;
  }

  _flatTensors() {
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

  _gradTensor(G, name) {
    const [head, rest] = [name.split('.')[0], name.split('.')[1]];
    if (rest === undefined) return G[head];
    const l = parseInt(head.slice(1), 10);
    return G.layers[l][rest];
  }

  _zeroGrads(G) {
    const zero = (a) => a.fill(0);
    zero(G.wte); zero(G.wpe); zero(G.lnfg); zero(G.lnfb);
    for (const L of G.layers) for (const k of Object.keys(L)) zero(L[k]);
  }

  /** One Adam update step with the averaged batch gradient. */
  _adamStep(G, seqCount, lr) {
    if (!this._adam) {
      this._adam = {};
      for (const [name, T] of this._flatTensors()) {
        this._adam[name] = { m: new Float32Array(T.length), v: new Float32Array(T.length) };
      }
    }
    this._t++;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const bc1 = 1 - Math.pow(b1, this._t), bc2 = 1 - Math.pow(b2, this._t);
    for (const [name, T] of this._flatTensors()) {
      const gT = this._gradTensor(G, name);
      if (!gT || gT.length !== T.length) continue;
      const st = this._adam[name];
      const inv = 1 / seqCount;
      for (let i = 0; i < T.length; i++) {
        const g = gT[i] * inv;
        st.m[i] = b1 * st.m[i] + (1 - b1) * g;
        st.v[i] = b2 * st.v[i] + (1 - b2) * g * g;
        T[i] -= lr * (st.m[i] / bc1) / (Math.sqrt(st.v[i] / bc2) + eps);
      }
    }
  }

  /**
   * Trains on `text`. Real mini-batch Adam over random context windows —
   * every sequence teaches every position (the transformer advantage).
   * onProgress({epoch, epochs, step, loss, sample}) fires per epoch.
   */
  async train(text, opts) {
    opts = opts || {};
    const epochs = Math.max(1, Math.min(opts.epochs || 5, opts.epochCap || 50));
    const onProgress = opts.onProgress || null;
    const shouldCancel = opts.shouldCancel || (() => false);
    const T = this.config.blockSize;
    const batch = this.config.batch;

    if (!text || text.trim().length < 50) {
      throw new Error('Nova needs at least ~50 characters of training text to learn anything.');
    }
    this.tokenizer.buildFromText(text);
    this._resizeForVocab(this.vocabSize);

    const ids = this.tokenizer.encode(text);
    if (ids.length < T + 2) throw new Error('Training text is too short for the context window.');

    const numStarts = ids.length - T - 1;
    const maxSeqs = opts.maxWindows || 300; // sequences per epoch (browser default)
    const G = this._newGrads();

    let lr = this.config.lr;
    let step = 0;
    const reportEvery = Math.max(1, Math.floor(Math.max(1, Math.min(numStarts, maxSeqs)) / batch / 4));

    for (let e = 0; e < epochs; e++) {
      const seqs = Math.min(numStarts, maxSeqs);
      let totalLoss = 0, count = 0, seqInBatch = 0;
      // zero grads
      const Gz = this._newGrads();
      for (let s = 0; s < seqs; s++) {
        if (shouldCancel()) return { cancelled: true, loss: this.lastLoss, steps: step };
        const st = Math.floor(Math.random() * numStarts);
        const toks = ids.slice(st, st + T);
        const tgt = ids.slice(st + 1, st + T + 1);
        const A = this._forward(toks);
        totalLoss += this._backward(A, tgt, Gz);
        count++; seqInBatch++; step++;
        if (seqInBatch === batch || s === seqs - 1) {
          this._adamStep(Gz, seqInBatch, lr);
          this._zeroGrads(Gz); // grads are consumed by the step — reset
          seqInBatch = 0;
          if (step % (reportEvery * batch) < batch) {
            await new Promise((r) => setTimeout(r, 0));
            if (onProgress) {
              onProgress({ epoch: e + 1, epochs, step, loss: totalLoss / count / T, sample: null });
            }
          }
        }
      }
      this.lastLoss = totalLoss / count / T;
      if (!Number.isFinite(this.lastLoss)) {
        throw new Error('Training diverged. Try fewer epochs or shorter text.');
      }
      lr = Math.max(this.config.lrFloor, lr * this.config.lrDecay);
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
   * Samples a continuation from the model's ACTUAL next-token predictions.
   * Untrained weights still give real (if meaningless) predictions.
   */
  generate(maxTokens, temperature, seedIds) {
    maxTokens = maxTokens || 200;
    temperature = temperature || 0.9;
    const T = this.config.blockSize;
    let ctx = seedIds && seedIds.length ? seedIds.slice(-T) : [];
    if (ctx.length === 0) ctx = [NovaTokenizer.BOS];
    const out = [];
    for (let t = 0; t < maxTokens; t++) {
      const A = this._forward(ctx);
      const off = (A.n - 1) * this.vocabSize;
      const logits = A.logits.subarray(off, off + this.vocabSize);
      const id = this.sample(logits, temperature);
      if (id === NovaTokenizer.EOS) break;
      if (id !== NovaTokenizer.PAD && id !== NovaTokenizer.BOS) out.push(id);
      ctx = (ctx.length < T ? ctx.concat([id]) : ctx.slice(1).concat([id]));
      if (ctx.length === 1 && ctx[0] === NovaTokenizer.BOS && id === NovaTokenizer.PAD) break;
    }
    return this.tokenizer.decode(out);
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

  reset() {
    this.tokenizer = new NovaTokenizer();
    this.trained = false;
    this.trainSteps = 0;
    this.lastLoss = null;
    this.corpusChars = 0;
    this._initParams();
    this._resetOptimizer();
  }

  // ---------- persistence ----------

  toJSON() {
    const o = {
      version: 2,
      config: this.config,
      tokenizer: this.tokenizer.toJSON(),
      trained: this.trained,
      trainSteps: this.trainSteps,
      lastLoss: this.lastLoss,
      corpusChars: this.corpusChars,
      params: {},
    };
    for (const [name, T] of this._flatTensors()) o.params[name] = Nova1Model._f32ToB64(T);
    return o;
  }

  static fromJSON(o) {
    if (!o || o.version !== 2) throw new Error('Unknown Nova checkpoint version.');
    const m = new Nova1Model(o.config);
    m.tokenizer = NovaTokenizer.fromJSON(o.tokenizer);
    m._initParams();
    m._resizeForVocab(m.vocabSize);
    for (const [name, T] of m._flatTensors()) {
      const b64 = o.params[name];
      if (!b64) throw new Error('Checkpoint is missing parameter tensor ' + name + '.');
      const arr = Nova1Model._b64ToF32(b64);
      if (arr.length !== T.length) throw new Error('Checkpoint tensor ' + name + ' has the wrong size.');
      T.set(arr);
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
