#!/usr/bin/env node
'use strict';
/**
 * Nova training pipeline (agent-run, fast: TensorFlow.js backend).
 *
 *   MODEL=nova-1.5 node train/train.js          # train one model
 *   MODEL=nova-1.0 EPOCHS=200 node train/train.js
 *   CONTINUE=1 MODEL=... node train/train.js    # resume fine-tuning
 *
 * Each run writes js/base-checkpoints/<model-id>.js — the trained default
 * weights the site ships for that model — plus per-epoch resumable
 * checkpoints at train/last-checkpoint-<model>.js.
 */
process.env.TF_CPP_MIN_LOG_LEVEL = process.env.TF_CPP_MIN_LOG_LEVEL || '3';
globalThis.tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const src =
  fs.readFileSync(path.join(ROOT, 'js/tokenizer.js'), 'utf8') + '\n' +
  fs.readFileSync(path.join(ROOT, 'js/model.js'), 'utf8') +
  '\nglobalThis.NovaTokenizer = NovaTokenizer; globalThis.Nova1Model = Nova1Model;';
eval(src);

const MODELS = {
  'nova-1.0': { blockSize: 48, dModel: 48, nHead: 2, nLayer: 2, lr: 3e-3, lrDecay: 0.985, lrFloor: 3e-4, batch: 16 },
  'nova-1.5': { blockSize: 64, dModel: 56, nHead: 2, nLayer: 3, lr: 3e-3, lrDecay: 0.985, lrFloor: 3e-4, batch: 16 },
  'supernova-1.0': { blockSize: 80, dModel: 64, nHead: 4, nLayer: 3, lr: 3e-3, lrDecay: 0.985, lrFloor: 3e-4, batch: 16 },
};

const MODEL_ID = process.env.MODEL || 'nova-1.0';
if (!MODELS[MODEL_ID]) throw new Error('Unknown MODEL: ' + MODEL_ID + ' (use one of: ' + Object.keys(MODELS).join(', ') + ')');
const CONFIG = MODELS[MODEL_ID];
const EPOCHS = parseInt(process.env.EPOCHS || '100', 10);
const MAX_SEQS = parseInt(process.env.MAX_SEQS || process.env.MAX_WINDOWS || '400', 10);

(async () => {
  const corpus = fs.readFileSync(path.join(__dirname, 'corpus.txt'), 'utf8');
  console.log(`model ${MODEL_ID} | corpus ${corpus.length.toLocaleString()} chars | ${EPOCHS} epochs x ${MAX_SEQS} seqs`);

  let m;
  const resumePath = path.join(__dirname, `last-checkpoint-${MODEL_ID}.json`);
  const outFile = path.join(ROOT, 'js/base-checkpoints', MODEL_ID + '.js');
  let json = null;
  if (process.env.CONTINUE === '1') {
    if (fs.existsSync(resumePath)) {
      json = JSON.parse(fs.readFileSync(resumePath, 'utf8'));
      console.log('resuming from', resumePath);
    } else if (fs.existsSync(outFile)) {
      const s = fs.readFileSync(outFile, 'utf8');
      json = JSON.parse(s.slice(s.indexOf('] = ') + 4, s.lastIndexOf('}') + 1));
    }
  }
  if (json) {
    m = Nova1Model.fromJSON(json);
    console.log('continuing from checkpoint (loss ' + (m.lastLoss || 0).toFixed(4) + ')');
    if (process.env.LR) m.config.lr = parseFloat(process.env.LR);
  } else {
    m = new Nova1Model(CONFIG);
  }

  const t0 = Date.now();
  const res = await m.train(corpus, {
    epochs: EPOCHS,
    epochCap: 100000,
    maxWindows: MAX_SEQS,
    sampleEvery: 10,
    onProgress: (p) => {
      // checkpoint every epoch, log every 10th
      fs.writeFileSync(resumePath, JSON.stringify(m.toJSON()));
      if (p.epoch % 10 === 0 || p.epoch === p.epochs) {
        console.log(`epoch ${p.epoch}/${p.epochs} · loss ${p.loss.toFixed(4)}` +
          (p.sample ? ` · sample: ${JSON.stringify(p.sample.slice(0, 70))}` : ''));
      }
    },
  });
  console.log(`done: loss ${res.loss.toFixed(4)} after ${res.steps} steps in ${((Date.now() - t0) / 1000).toFixed(0)}s`);

  const out =
    `/* Nova base checkpoint: ${MODEL_ID} — trained by train/train.js (${new Date().toISOString()}). */\n` +
    `/* Loss ${res.loss.toFixed(4)}, ${res.steps} Adam steps. This repo's own model + pipeline, no external weights. */\n` +
    `window.NOVA_BASE_CHECKPOINTS = window.NOVA_BASE_CHECKPOINTS || {};\n` +
    `window.NOVA_BASE_CHECKPOINTS[${JSON.stringify(MODEL_ID)}] = ` + JSON.stringify(m.toJSON()) + ';\n';
  fs.mkdirSync(path.dirname(outFile), { recursive: true });
  fs.writeFileSync(outFile, out);
  if (fs.existsSync(resumePath)) fs.unlinkSync(resumePath);
  console.log('written', path.relative(ROOT, outFile), '(' + Math.round(out.length / 1024) + ' KB)');

  const m2 = Nova1Model.fromJSON(JSON.parse(out.slice(out.indexOf('] = ') + 4, out.lastIndexOf('}') + 1)));
  for (const seed of ['Hello', 'What is your name', 'What is a transformer', 'What can Supernova do']) {
    console.log('seed ' + JSON.stringify(seed) + ' -> ' + JSON.stringify(m2.generate(100, 0.62, m2.tokenizer.encode(seed).slice(0, -1)).trim()).slice(0, 120));
  }
})().catch((e) => { console.error('TRAIN FAILED', e); process.exit(1); });
