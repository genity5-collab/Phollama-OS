# Nova 1.0 training system (agent-run, not user-run)

This is the pipeline that produces the **base weights** every visitor of the
site loads. It runs outside the browser, so the agent can retrain Nova without
the owner having to do anything by hand.

## The 3-step retrain loop

1. **Edit the lesson** — `train/corpus.txt` is plain text (currently a
   conversational Q&A set). Add or change lines; more repetition = stronger
   recall; more variety = more flexible phrasing.
2. **Run training** — from the repo root:

   ```
   node train/train.js            # defaults: 55 epochs, 15k windows/epoch
   EPOCHS=80 MAX_WINDOWS=25000 node train/train.js   # heavier run
   ```

   This trains a fresh Nova 1.0 with real SGD (same `js/model.js` the browser
   uses) and overwrites `js/base-checkpoint.js` — the trained weights that
   `index.html` ships as the default model. The run prints loss per epoch and
   sanity-check generations at the end.

3. **Push** — `git add -A && git commit -m "Retrain Nova" && git push`.
   GitHub Pages redeploys the site automatically. No other steps.

## How the site uses the checkpoint

- `js/base-checkpoint.js` sets `window.NOVA_BASE_CHECKPOINT` (weights,
  vocabulary with stable token IDs, config, loss, step count, `base: true`).
- On first visit (nothing in localStorage) the app loads it, so Nova already
  talks. The UI honestly labels this as **base training**.
- The **Train** button still lets anyone train further on top of the base
  weights, in their own browser.
- A checkpoint saved in the visitor's own localStorage always wins over the
  base one. **Reset model** returns to a fresh untrained network.

## Model architecture (v2, researched)

`js/model.js` implements a minGPT/microgpt-style transformer: learned token +
position embeddings (output head tied to token embeddings), nLayer blocks of
pre-LN causal self-attention (nHead heads) + GELU MLP, final LayerNorm, Adam
optimizer with per-epoch LR decay. Backprop was verified with finite-difference
gradient checks. Research sources: karpathy/minGPT (architecture + Adam recipe)
and jayyvk/trainmyowngpt (JS-only browser training of the same design).

## Honesty note

The base weights are produced by this repo's own training script on this
repo's own corpus. They are not downloaded from any external AI provider, and
no external inference API is used anywhere. The model is still the same small
from-scratch network; it does not gain general knowledge, only the corpus
patterns.

## SmolTalk note (2026-10-17)

Nova's corpus gained extra everyday-conversation pairs inspired by Hugging
Face's SmolTalk chat data (`HuggingFaceTB/smol-smoltalk`, Apache-2.0). The
public dataset endpoint only exposes long instruction-following examples
(Python tasks, formatting rules), which do not fit Nova's tiny 48-character
context, so the added pairs follow SmolTalk's everyday-conversation style
instead of copying its long rows. Nova's weights remain 100% trained by this
repo's own pipeline — no external model weights are used anywhere.
