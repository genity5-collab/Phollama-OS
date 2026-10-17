# Nova training system (agent-run, fast)

Trains the **base weights** every visitor of the site loads, using
TensorFlow.js (tfjs-node, AVX-accelerated) — ~15-20x faster than the old
pure-JS engine. A full Nova 1.0 retrain takes about two minutes.

## The 3-step retrain loop (no manual training needed for users)

1. **Edit the lesson** — `train/corpus.txt` is plain text, one
   `Question\nAnswer` pair per two lines.
2. **Train one or all models**:

   ```
   node train/train.js                       # defaults to nova-1.0
   MODEL=nova-1.5 node train/train.js        # the improved model
   MODEL=supernova-1.0 node train/train.js   # the experimental big sibling
   EPOCHS=35 MAX_SEQS=2000 node train/train.js   # knobs
   CONTINUE=1 LR=1e-3 MODEL=nova-1.0 node train/train.js  # resume fine-tune
   ```

   Each run writes `js/base-checkpoints/<model-id>.js` (trained default
   weights shipped with the site) and keeps a resumable per-epoch checkpoint
   at `train/last-checkpoint-<model>.json`.

3. **Push** — GitHub Pages redeploys the site automatically.

## How the site uses checkpoints

- `js/base-checkpoints/*.js` register `window.NOVA_BASE_CHECKPOINTS['<id>']`
  (weights + vocabulary with stable token IDs + config + loss/steps).
- On first visit the app loads each model's base checkpoint, so Nova already
  talks with zero manual training. The UI labels this as **base training**.
- The **Train** button trains the selected model further, in the visitor's
  own browser (tfjs WebGL), saved under `nova.model.<id>.v3` in localStorage.
- A browser-saved checkpoint always wins over the shipped base. **Reset**
  returns a model to a fresh untrained state.

## Model configs (in train/train.js)

| Model | Context | dModel | Layers | Heads | ~Params |
|---|---|---|---|---|---|
| nova-1.0 | 48 | 48 | 2 | 2 | 64K |
| nova-1.5 | 64 | 56 | 3 | 2 | 122K |
| supernova-1.0 | 80 | 64 | 3 | 4 | ~190K |

## Honesty note

All weights are produced by this repo's own training script on this repo's own
corpus (plus SmolTalk-style everyday-conversation pairs; see
train/smoltalk-extra.txt). No external model weights are used anywhere, and no
external inference API is called — TensorFlow.js is a math engine, not an AI
provider. The models still only know their corpus; planned Supernova features
(image generation, tools) are not implemented and are labeled as planned.

## 2026-10-17: publish push

- nova-1.0 discontinued and removed from the site (registry, script tag,
  checkpoint). Stored selections fall back to nova-1.5.
- nova-1.5 and supernova-1.0 retrained on the 155-turn chat curriculum
  (chat-turn format, digit math incl. times tables, daily chat, meta).
