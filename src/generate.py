# src/generate.py
# Run your trained Nova model locally.
# Works on Windows CPU after training on Colab.

import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import NovaGPT
from src.tokenizer import NovaTokenizer
from src.config import NovaChatConfig, NovaCodeConfig


def load_model(model_type="chat"):
    """Load the best trained model."""
    config = NovaChatConfig if model_type == "chat" else NovaCodeConfig
    best   = os.path.join(config.out_dir, "best.pt")
    final  = os.path.join(config.out_dir, "final.pt")

    path = best if os.path.exists(best) else final
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained model found at {path}.\n"
            f"Train first: open notebooks/nova_train.ipynb in Colab."
        )

    model, _ = NovaGPT.load_checkpoint(path, config)
    model.eval()
    return model, config


def chat(model_type="chat"):
    """Interactive chat loop."""
    print(f"Loading Nova {model_type} model...")
    model, config = load_model(model_type)
    tokenizer = NovaTokenizer("models/nova_tokenizer.json")

    system = (
        "You are Nova, an honest and helpful AI assistant."
        if model_type == "chat"
        else "You are Nova Code, a coding assistant. Write clean, working code."
    )

    print(f"\nNova {model_type.title()} is ready.")
    print(f"Model: {config.model_name} ({model.count_params()/1e6:.1f}M params)")
    print("Type 'quit' to exit, 'reset' to start new conversation.\n")

    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            history = []
            print("Conversation reset.\n")
            continue

        # Build messages
        messages = [{"role": "system", "content": system}]
        messages += history
        messages.append({"role": "user", "content": user_input})

        # Encode prompt
        prompt = tokenizer.format_chat_prompt(messages)
        ids    = tokenizer.encode(prompt, add_special_tokens=False)
        idx    = torch.tensor([ids], dtype=torch.long)

        # Generate
        with torch.no_grad():
            out = model.generate(
                idx,
                max_new_tokens=200,
                temperature=0.8,
                top_k=40,
            )

        # Decode only the new tokens
        new_ids  = out[0][len(ids):].tolist()
        response = tokenizer.decode(new_ids)

        # Stop at end of assistant turn
        for stop in ["<|user|>", "<|system|>", "<eos>"]:
            if stop in response:
                response = response[:response.index(stop)]
        response = response.strip()

        if not response:
            response = "I'm not sure. Could you rephrase that?"

        print(f"Nova: {response}\n")

        # Add to history (keep last 6 turns to fit in context)
        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": response})
        if len(history) > 12:
            history = history[-12:]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="chat", choices=["chat", "code"],
                        help="Which model to run: chat or code")
    args = parser.parse_args()
    chat(args.model)
