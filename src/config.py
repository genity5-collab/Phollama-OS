# src/config.py
# All Nova model settings in one place.
# Change here, affects everything.

class NovaChatConfig:
    """Nova Chat 124M - general conversation model."""
    model_name  = "nova-chat-124m"
    model_type  = "chat"
    vocab_size  = 8000
    block_size  = 512
    n_layer     = 12
    n_head      = 8
    n_embd      = 512
    dropout     = 0.1
    bias        = False

    # Training
    batch_size          = 32
    grad_accum_steps    = 4
    learning_rate       = 3e-4
    min_lr              = 3e-5
    warmup_steps        = 200
    max_steps           = 10000
    eval_interval       = 500
    save_interval       = 1000
    grad_clip           = 1.0

    # Data
    train_data  = "data/chat_train.bin"
    val_data    = "data/chat_val.bin"
    raw_data    = "data/chat_raw.txt"

    # Output
    out_dir     = "checkpoints/nova-chat"
    hf_repo     = "your-username/nova-chat-124m"  # CHANGE THIS


class NovaCodeConfig:
    """Nova Code 124M - coding assistant model."""
    model_name  = "nova-code-124m"
    model_type  = "code"
    vocab_size  = 8000
    block_size  = 512
    n_layer     = 12
    n_head      = 8
    n_embd      = 512
    dropout     = 0.1
    bias        = False

    # Training
    batch_size          = 32
    grad_accum_steps    = 4
    learning_rate       = 3e-4
    min_lr              = 3e-5
    warmup_steps        = 200
    max_steps           = 10000
    eval_interval       = 500
    save_interval       = 1000
    grad_clip           = 1.0

    # Data
    train_data  = "data/code_train.bin"
    val_data    = "data/code_val.bin"
    raw_data    = "data/code_raw.txt"

    # Output
    out_dir     = "checkpoints/nova-code"
    hf_repo     = "your-username/nova-code-124m"  # CHANGE THIS


def count_params(config):
    """Calculate total parameter count for a config."""
    V = config.vocab_size
    T = config.block_size
    d = config.n_embd
    L = config.n_layer
    params = (
        V * d +           # token embeddings
        T * d +           # position embeddings
        L * (
            d * 3 * d +   # QKV
            d * d +       # output proj
            d * 4 * d +   # MLP fc
            4 * d * d +   # MLP proj
            4 * d         # layer norms
        ) +
        2 * d             # final layer norm
    )
    return params


if __name__ == "__main__":
    for cfg in [NovaChatConfig, NovaCodeConfig]:
        p = count_params(cfg)
        size_mb = (p * 2) / 1e6  # fp16 = 2 bytes per param
        print(f"{cfg.model_name}: {p/1e6:.1f}M params | {size_mb:.0f}MB in fp16")
