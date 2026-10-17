# Nova AI — From Scratch

Two real language models built from zero.
Your architecture. Your data. Your weights.

| Model | Size | Good at |
|---|---|---|
| Nova Chat | 124M params | General conversation |
| Nova Code | 124M params | Coding help |

No pretrained weights. No external AI APIs.
PyTorch transformer, BPE tokenizer, trained on Colab.

---

## Quick start (3 steps)

### Step 1: Prepare data (run locally or in Colab)
```bash
pip install torch tokenizers datasets huggingface_hub numpy
python data/prepare.py
```

### Step 2: Train in Colab (free T4 GPU)
1. Open `notebooks/nova_train.ipynb` in GitHub
2. Click **Open in Colab**
3. Runtime → Change runtime type → **T4 GPU**
4. Add secret `HF_TOKEN` (click the 🔑 key icon)
5. Run all cells

### Step 3: Run locally on Windows
```bash
pip install torch tokenizers
python src/generate.py --model chat
python src/generate.py --model code
```

---

## File structure

```
nova-ai/
  src/
    config.py      — model sizes and training settings
    model.py       — transformer architecture (pure PyTorch)
    tokenizer.py   — BPE tokenizer
    train.py       — training loop
    generate.py    — chat interface
  data/
    prepare.py     — download and tokenize training data
  eval/
    test_chat.json — 20 chat test cases
    test_code.json — 20 code test cases
    evaluate.py    — scoring script
  notebooks/
    nova_train.ipynb — Colab training notebook
  checkpoints/
    nova-chat/best.pt   — best chat model weights
    nova-code/best.pt   — best code model weights
  models/
    nova_tokenizer.json — trained BPE tokenizer
```

---

## Training numbers to watch

```
Good:   train loss falling + val loss falling
OK:     train loss < val loss (small gap)
BAD:    val loss rising while train loss falls = overfitting, STOP
```

---

## Eval scores

Run before every deployment:
```bash
python eval/evaluate.py --model chat
python eval/evaluate.py --model code
```

- 75+  = deploy
- 50-74 = more training data needed
- <50  = do not deploy

---

## Honest capability statement

Nova Chat and Nova Code are 124M parameter models.
They learn the patterns in their training data.
They will make mistakes. They are not GPT-4.
They are real, trainable, improvable, and 100% yours.

Improving quality:
1. Add more training examples to data/chat_raw.txt
2. Re-run data/prepare.py
3. Fine-tune: python src/train.py --model chat --resume
4. Evaluate: python eval/evaluate.py --model chat
5. Deploy when score >= 75

---

## Credits

Architecture: GPT-2 (Radford et al., 2019)
Reference implementation: Karpathy nanoGPT
Training data: TinyStories, CodeAlpaca (open licensed)
Everything else: built from scratch for Nova
