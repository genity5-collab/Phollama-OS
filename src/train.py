# src/train.py
# Training loop for Nova GPT.
# Runs on Colab GPU or locally.
# Saves checkpoints, logs loss, evaluates on val set.

import os
import sys
import time
import math
import numpy as np
import torch
from contextlib import nullcontext

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import NovaGPT
from src.config import NovaChatConfig, NovaCodeConfig


def get_config(model_type="chat"):
    if model_type == "chat":
        return NovaChatConfig
    elif model_type == "code":
        return NovaCodeConfig
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'chat' or 'code'.")


def get_lr(step, config):
    """
    Cosine learning rate schedule with warmup.
    Warmup: linear increase for warmup_steps
    Then: cosine decay from lr to min_lr
    """
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    if step > config.max_steps:
        return config.min_lr
    decay = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def load_data(path, device):
    """Load binary token file."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Run: python data/prepare.py first."
        )
    data = np.fromfile(path, dtype=np.uint16)
    return torch.from_numpy(data.astype(np.int64)).to(device)


def get_batch(data, config, device):
    """Sample a random batch of sequences."""
    T  = config.block_size
    B  = config.batch_size
    ix = torch.randint(len(data) - T, (B,))
    x  = torch.stack([data[i:i + T] for i in ix])
    y  = torch.stack([data[i + 1:i + T + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, config, device, eval_batches=20):
    """Estimate loss on train and val sets without updating weights."""
    model.eval()
    losses = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        batch_losses = []
        for _ in range(eval_batches):
            x, y = get_batch(data, config, device)
            _, loss = model(x, y)
            batch_losses.append(loss.item())
        losses[split] = sum(batch_losses) / len(batch_losses)
    model.train()
    return losses


def train(model_type="chat", resume=False):
    """
    Main training function.
    model_type: "chat" or "code"
    resume: True to continue from last checkpoint
    """
    config = get_config(model_type)

    # Device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon GPU")
    else:
        device = "cpu"
        print("Using CPU (slow - recommend Colab for training)")

    print(f"\nTraining Nova {model_type} model")
    print(f"Max steps: {config.max_steps:,}")
    print(f"Batch size: {config.batch_size} x {config.grad_accum_steps} = {config.batch_size * config.grad_accum_steps} effective")

    # Load data
    print("\nLoading data...")
    train_data = load_data(config.train_data, device)
    val_data   = load_data(config.val_data,   device)
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens:   {len(val_data):,}")

    # Model
    os.makedirs(config.out_dir, exist_ok=True)
    checkpoint_path = os.path.join(config.out_dir, "checkpoint.pt")
    best_path       = os.path.join(config.out_dir, "best.pt")

    start_step = 0
    if resume and os.path.exists(checkpoint_path):
        print(f"\nResuming from {checkpoint_path}")
        model, ckpt = NovaGPT.load_checkpoint(checkpoint_path, config)
        start_step  = ckpt.get('step', 0)
    else:
        print("\nStarting fresh training")
        model = NovaGPT(config)

    model = model.to(device)

    # Optimizer: AdamW with weight decay
    # Separate params: embeddings and biases don't get weight decay
    decay_params    = [p for n, p in model.named_parameters()
                       if p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {'params': decay_params,    'weight_decay': 0.1},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=config.learning_rate, betas=(0.9, 0.95))

    if resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])

    # Mixed precision for GPU
    # (torch.cuda.GradScaler was removed in newer PyTorch -> torch.amp.GradScaler)
    if device == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda")
        except AttributeError:
            scaler = torch.cuda.amp.GradScaler()
        ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        scaler = None
        ctx = nullcontext()

    # Training loop
    print(f"\nStarting training from step {start_step}...")
    print("-" * 60)

    best_val_loss = float('inf')
    bad_evals = 0
    patience = getattr(config, 'patience', 5)
    t0 = time.time()

    for step in range(start_step, config.max_steps):
        # Update learning rate
        lr = get_lr(step, config)
        for group in optimizer.param_groups:
            group['lr'] = lr

        # Evaluation
        if step % config.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, config, device)
            dt = time.time() - t0
            print(
                f"step {step:5d}/{config.max_steps} | "
                f"train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | "
                f"lr {lr:.2e} | "
                f"{dt:.0f}s elapsed"
            )
            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                model.save_checkpoint(best_path, optimizer, step, losses['val'])
                print(f"  New best val loss: {best_val_loss:.4f} -> saved best.pt")
                bad_evals = 0
            else:
                bad_evals += 1
                if bad_evals >= patience:
                    print(f"  Val loss has not improved for {patience} evals -> early stop (overfitting).")
                    break

        # Gradient accumulation
        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(config.grad_accum_steps):
            x, y = get_batch(train_data, config, device)
            with ctx:
                _, loss = model(x, y)
                loss = loss / config.grad_accum_steps
            loss_accum += loss.item()
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        # Gradient clipping
        if scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Save checkpoint
        if step % config.save_interval == 0 and step > 0:
            model.save_checkpoint(checkpoint_path, optimizer, step, loss_accum)

    # Final save
    final_path = os.path.join(config.out_dir, "final.pt")
    model.save_checkpoint(final_path, optimizer, config.max_steps, best_val_loss)
    print(f"\nTraining complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved:   {final_path}")
    print(f"Best model:    {best_path}")

    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="chat", choices=["chat", "code"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args.model, args.resume)
