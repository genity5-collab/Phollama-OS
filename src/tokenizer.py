# src/tokenizer.py
# BPE tokenizer - trained on your data, saved with your model.
# Uses HuggingFace tokenizers library (fast, battle-tested BPE).
# Your vocab, your token IDs, nobody else's.

import os
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors


# Special tokens - fixed IDs, never change
SPECIAL_TOKENS = [
    "<pad>",    # 0 - padding
    "<unk>",    # 1 - unknown character
    "<bos>",    # 2 - beginning of sequence
    "<eos>",    # 3 - end of sequence
    "<|user|>",      # 4 - user turn
    "<|assistant|>", # 5 - assistant turn
    "<|system|>",    # 6 - system prompt
    "<|code|>",      # 7 - code block
    "<|endcode|>",   # 8 - end code block
]


def train_tokenizer(
    text_files,
    vocab_size=8000,
    save_path="models/nova_tokenizer.json"
):
    """
    Train a BPE tokenizer on your text files.
    Run this once before training the model.

    Args:
        text_files: list of paths to .txt files
        vocab_size: vocabulary size (8000 is good for small models)
        save_path: where to save the tokenizer
    """
    print(f"Training BPE tokenizer on {len(text_files)} files...")
    print(f"Vocabulary size: {vocab_size}")

    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # Pre-tokenizer: split on whitespace and punctuation
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # Trainer settings
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,          # token must appear at least twice
        show_progress=True,
    )

    # Train on your files
    tokenizer.train(text_files, trainer)

    # Add post-processor for special tokens
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> $B:1 <eos>:1",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    print(f"Tokenizer saved to: {save_path}")
    print(f"Actual vocab size: {tokenizer.get_vocab_size()}")

    # Verify: encode/decode roundtrip
    test = "Hello, I am Nova. print('hello world')"
    encoded = tokenizer.encode(test)
    decoded = tokenizer.decode(encoded.ids)
    print(f"Roundtrip test:")
    print(f"  Original: {test}")
    print(f"  Tokens:   {encoded.tokens[:10]}...")
    print(f"  Decoded:  {decoded}")

    return tokenizer


def load_tokenizer(path="models/nova_tokenizer.json"):
    """Load a saved tokenizer."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Tokenizer not found at {path}. "
            f"Run: python src/tokenizer.py to train it first."
        )
    tokenizer = Tokenizer.from_file(path)
    return tokenizer


class NovaTokenizer:
    """
    Wrapper around the HuggingFace tokenizer.
    Adds convenience methods for chat formatting.
    """

    def __init__(self, path="models/nova_tokenizer.json"):
        self.tokenizer = load_tokenizer(path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.pad_id  = self.tokenizer.token_to_id("<pad>")
        self.unk_id  = self.tokenizer.token_to_id("<unk>")
        self.bos_id  = self.tokenizer.token_to_id("<bos>")
        self.eos_id  = self.tokenizer.token_to_id("<eos>")
        self.user_id = self.tokenizer.token_to_id("<|user|>")
        self.asst_id = self.tokenizer.token_to_id("<|assistant|>")

    def encode(self, text, add_special_tokens=True):
        """Text -> list of token IDs."""
        encoded = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoded.ids

    def decode(self, ids, skip_special=True):
        """List of token IDs -> text."""
        if skip_special:
            special_ids = {self.pad_id, self.bos_id, self.eos_id,
                          self.user_id, self.asst_id}
            ids = [i for i in ids if i not in special_ids]
        return self.tokenizer.decode(ids)

    def encode_chat(self, messages):
        """
        Format a conversation and encode it.
        messages = [
            {"role": "system", "content": "You are Nova..."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        """
        text = ""
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                text += f"<|system|>{content}\n"
            elif role == "user":
                text += f"<|user|>{content}\n"
            elif role == "assistant":
                text += f"<|assistant|>{content}<eos>\n"
        return self.encode(text, add_special_tokens=False)

    def format_chat_prompt(self, messages):
        """
        Format messages as a string prompt for generation.
        Ends with <|assistant|> so model continues from there.
        """
        text = ""
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                text += f"<|system|>{content}\n"
            elif role == "user":
                text += f"<|user|>{content}\n"
            elif role == "assistant":
                text += f"<|assistant|>{content}<eos>\n"
        text += "<|assistant|>"
        return text

    def chars_per_token(self, sample_text):
        """Measure compression ratio (higher = more efficient)."""
        ids = self.encode(sample_text)
        return len(sample_text) / len(ids)


if __name__ == "__main__":
    import sys

    # If tokenizer already exists, test it
    if os.path.exists("models/nova_tokenizer.json"):
        print("Loading existing tokenizer...")
        tok = NovaTokenizer("models/nova_tokenizer.json")
        print(f"Vocab size: {tok.vocab_size}")

        tests = [
            "Hello, how are you?",
            "def hello_world():\n    print('Hello, world!')",
            "What is 2 + 2?",
        ]
        for t in tests:
            ids = tok.encode(t)
            back = tok.decode(ids)
            ratio = tok.chars_per_token(t)
            print(f"  '{t[:30]}' -> {len(ids)} tokens ({ratio:.1f} chars/token)")
            assert t.strip() in back or back.strip() in t, f"Roundtrip failed: {back}"
        print("All roundtrip tests passed.")

    else:
        # Train a demo tokenizer on corpus.txt if available
        corpus = "train/corpus.txt"
        if os.path.exists(corpus):
            print(f"Training tokenizer on {corpus}...")
            train_tokenizer([corpus], vocab_size=8000, save_path="models/nova_tokenizer.json")
        else:
            print("No corpus found. Create data/chat_raw.txt first.")
            print("Then run: python src/tokenizer.py")
