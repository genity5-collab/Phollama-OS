# data/prepare.py
# Downloads and prepares training data for Nova Chat and Nova Code.
# Run this once before training.
# All data is open-licensed.

import os
import sys
import json
import random
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import train_tokenizer, NovaTokenizer

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────
# CHAT DATA
# ─────────────────────────────────────────────

CHAT_SYSTEM_PROMPT = """You are Nova, an honest and helpful AI assistant.
You give clear, correct answers.
If you do not know something, you say so.
You never make up facts."""

# Your own Nova corpus (already in the repo)
NOVA_CORPUS = "train/corpus.txt"

def load_nova_corpus():
    """Load existing Nova training data and convert to messages format."""
    if not os.path.exists(NOVA_CORPUS):
        return []
    examples = []
    text = open(NOVA_CORPUS).read()
    blocks = text.strip().split('\n\n')
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 2:
            user_line = next((l for l in lines if l.startswith('You: ')), None)
            nova_line = next((l for l in lines if l.startswith('Nova: ')), None)
            if user_line and nova_line:
                examples.append({
                    "messages": [
                        {"role": "system",    "content": CHAT_SYSTEM_PROMPT},
                        {"role": "user",      "content": user_line[5:].strip()},
                        {"role": "assistant", "content": nova_line[6:].strip()},
                    ]
                })
    print(f"Loaded {len(examples)} examples from Nova corpus")
    return examples


def download_chat_data():
    """
    Download open chat datasets from Hugging Face.
    Uses datasets library.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        return []

    examples = []

    # TinyStories: simple, clean English stories
    print("Loading TinyStories...")
    try:
        ds = load_dataset("roneneldan/TinyStories", split="train[:20000]")
        for item in ds:
            story = item.get("text", "").strip()
            if len(story) > 50:
                examples.append({
                    "messages": [
                        {"role": "system",    "content": CHAT_SYSTEM_PROMPT},
                        {"role": "user",      "content": "Tell me a short story."},
                        {"role": "assistant", "content": story[:500]},
                    ]
                })
        print(f"  TinyStories: {len(examples)} examples")
    except Exception as e:
        print(f"  TinyStories failed: {e}")

    return examples


# ─────────────────────────────────────────────
# CODE DATA
# ─────────────────────────────────────────────

CODE_SYSTEM_PROMPT = """You are Nova Code, a coding assistant.
You write clean, working code.
You explain what the code does.
If you are not sure, you say so."""

def download_code_data():
    """Download open code datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        return []

    examples = []

    # CodeAlpaca: instruction -> code pairs
    print("Loading CodeAlpaca...")
    try:
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train[:20022]")
        for item in ds:
            instruction = item.get("instruction", "").strip()
            output      = item.get("output", "").strip()
            if instruction and output and len(output) > 10:
                examples.append({
                    "messages": [
                        {"role": "system",    "content": CODE_SYSTEM_PROMPT},
                        {"role": "user",      "content": instruction},
                        {"role": "assistant", "content": output},
                    ]
                })
        print(f"  CodeAlpaca: {len(examples)} examples")
    except Exception as e:
        print(f"  CodeAlpaca failed: {e}")

    return examples


# ─────────────────────────────────────────────
# TOKENIZE AND SAVE
# ─────────────────────────────────────────────

def format_example(example):
    """Convert messages to training text."""
    text = ""
    for m in example["messages"]:
        role    = m["role"]
        content = m["content"]
        if role == "system":
            text += f"<|system|>{content}\n"
        elif role == "user":
            text += f"<|user|>{content}\n"
        elif role == "assistant":
            text += f"<|assistant|>{content}<eos>\n"
    return text


def tokenize_and_save(examples, tokenizer, out_train, out_val, val_fraction=0.1):
    """Tokenize all examples and save as binary files for fast training."""
    random.shuffle(examples)

    split = int(len(examples) * (1 - val_fraction))
    train_examples = examples[:split]
    val_examples   = examples[split:]

    def process(exs, path):
        all_ids = []
        for ex in exs:
            text = format_example(ex)
            ids  = tokenizer.encode(text, add_special_tokens=False)
            all_ids.extend(ids)
        arr = np.array(all_ids, dtype=np.uint16)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        arr.tofile(path)
        print(f"  Saved {len(exs)} examples, {len(all_ids):,} tokens -> {path}")

    process(train_examples, out_train)
    process(val_examples,   out_val)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("data",   exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("=" * 50)
    print("Step 1: Prepare raw text for tokenizer training")
    print("=" * 50)

    # Collect all raw text for tokenizer training
    raw_texts = []

    nova_examples = load_nova_corpus()
    chat_examples = download_chat_data()
    code_examples = download_code_data()

    all_examples = nova_examples + chat_examples + code_examples

    # Write raw text file for tokenizer training
    raw_path = "data/all_raw.txt"
    with open(raw_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(format_example(ex) + "\n")
    print(f"Raw text written: {raw_path}")

    print("\n" + "=" * 50)
    print("Step 2: Train BPE tokenizer")
    print("=" * 50)

    tok_path = "models/nova_tokenizer.json"
    if not os.path.exists(tok_path):
        train_tokenizer([raw_path], vocab_size=8000, save_path=tok_path)
    else:
        print(f"Tokenizer already exists: {tok_path}")

    tokenizer = NovaTokenizer(tok_path)

    print("\n" + "=" * 50)
    print("Step 3: Tokenize and save training data")
    print("=" * 50)

    # Chat data
    chat_all = nova_examples + chat_examples
    if chat_all:
        print(f"\nChat: {len(chat_all)} examples")
        tokenize_and_save(
            chat_all, tokenizer,
            "data/chat_train.bin",
            "data/chat_val.bin"
        )

    # Code data
    if code_examples:
        print(f"\nCode: {len(code_examples)} examples")
        tokenize_and_save(
            code_examples, tokenizer,
            "data/code_train.bin",
            "data/code_val.bin"
        )

    print("\n" + "=" * 50)
    print("Done! Ready to train.")
    print("=" * 50)
    print("Next: open notebooks/nova_train.ipynb in Colab")
