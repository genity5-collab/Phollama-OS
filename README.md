# Nova AI

A polished, mobile-first AI chatbot powered by the **Nova family** — small
language models built and trained **from scratch**, running entirely in the
browser. No external AI APIs, no API keys, no external pretrained weights, no
paid dependencies. [TensorFlow.js](https://github.com/tensorflow/tfjs) is used
purely as the tensor math engine (matrix ops + autodiff) — the architecture,
tokenizer, corpus and weights are Nova's own.

Project name: `nova`

## The Nova family

| Model | ID | Status | What it is |
|---|---|---|---|
| Nova 1.0 | `nova-1.0` | Initial model | Tiny transformer (~64K parameters) |
| Nova 1.5 | `nova-1.5` | Improved model | Larger context + depth (~120K parameters) |
| Nova Supernova 1.0 | `supernova-1.0` | Experimental | Largest Nova (~190K parameters). **Planned (not yet available): image generation, tools** |

All three ship with **base conversational training** (trained by this repo's
own pipeline, weights in `js/base-checkpoints/`), so chatting needs zero
manual training. The **Train** button still lets anyone train any model
further, locally, on their own text. Planned features are always labeled as
planned — never as available.

## What is implemented (honestly)

- **Chat interface** — conversation history, new chat, multiline composer
  (Enter to send, Shift+Enter for newline), loading indicator, error handling,
  formatted rendering of lists, bold text, inline code and code blocks. Every
  visible button works.
- **Model selector** — three real, selectable models with full details; nothing
  unavailable is shown as selectable.
- **Real trainable models** — minGPT/microgpt-style decoder-only transformers
  in `js/model.js`: tied token embeddings, learned position embeddings,
  pre-LN causal self-attention, GELU MLPs, Adam optimizer (moments included in
  checkpoints so resumed training continues smoothly). Backprop uses tfjs
  autodiff; everything else (tokenizer, training loop, sampling, pipeline) is
  this repo's own code.
- **Custom tokenizer** — character-level with stable token IDs (fixed specials
  `0-3`, printable ASCII `4-98`, newline `99`, corpus extras `100+`),
  serialized with each model so IDs never shift.
- **Persistence** — per-model weights + vocabulary + chat history in
  `localStorage`, with quota/unavailable-storage handling. Nothing leaves the
  device.
- **Training system** — `train/` pipeline (agent-run, outside the browser,
  ~15x faster than the old pure-JS engine): edit `train/corpus.txt`, run
  `MODEL=<id> node train/train.js`, push; GitHub Pages redeploys.
  Per-epoch resumable checkpoints; `CONTINUE=1` resumes fine-tuning.

## Capability statement

The Nova models are genuinely trainable but tiny. They learn the character
patterns and phrasing of their training corpus and continue your messages
accordingly. They do not have general knowledge, reasoning, or ChatGPT-level
intelligence, and the UI never claims otherwise. Untrained models say so and
show their genuine raw (meaningless) output. Nothing fakes AI replies.

## Running locally

```
cd nova
npm install           # tfjs-node for the training pipeline (site itself needs only the CDN)
node train/train.js   # optional: train models
python3 -m http.server 8000   # or any static server
```

Deploying to any static host (GitHub Pages, etc.) works out of the box; all
asset paths are relative and TensorFlow.js loads from the jsDelivr CDN.

## Project structure

```
nova/
  index.html                 — app shell
  css/style.css              — Nova dark theme, responsive/mobile-first
  js/tokenizer.js            — char tokenizer with stable token IDs
  js/model.js                — Nova transformer (tfjs engine, Adam, checkpoints v3)
  js/storage.js              — safe localStorage wrappers
  js/ui.js                   — chat UI, model selector, training flow
  js/base-checkpoints/       — trained base weights per model (shipped)
  train/corpus.txt           — the conversation corpus Nova trains on
  train/train.js             — fast training pipeline (MODEL=<id>)
  README.md
```

## GitHub

This repository is structured to be synchronized with the GitHub repository
named `nova`. No GitHub synchronization is claimed beyond what is actually
configured and tested here.
