# src/model.py
# Nova GPT - decoder-only transformer.
# Pure PyTorch. No external AI libraries.
# Architecture: same as GPT-2. Weights are 100% yours.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.
    Each token can only attend to previous tokens (causal mask).
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head   = config.n_head
        self.n_embd   = config.n_embd
        self.head_size = config.n_embd // config.n_head
        self.dropout  = config.dropout

        # Q, K, V projections in one matrix
        self.c_attn  = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj  = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # Causal mask: lower triangle only
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()

        # Compute Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_size)
        att = (q @ k.transpose(-2, -1)) * scale

        # Apply causal mask
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # Aggregate values
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    """
    Feed-forward network inside each transformer block.
    Expands to 4x the embedding size then projects back.
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class Block(nn.Module):
    """
    One transformer block.
    Pre-norm: LayerNorm before attention and MLP (more stable).
    Residual connections around both.
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # attention + residual
        x = x + self.mlp(self.ln_2(x))   # MLP + residual
        return x


class NovaGPT(nn.Module):
    """
    Nova GPT: your own language model.
    Decoder-only transformer, GPT-2 architecture.
    Train it, own it, ship it.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),  # token embeddings
            wpe  = nn.Embedding(config.block_size, config.n_embd),  # position embeddings
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),   # final norm
        ))

        # Output head: tied weights with token embeddings (saves ~4M params)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights (GPT-2 style)
        self.apply(self._init_weights)
        # Scale residual projections (important for stability)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        total = self.count_params()
        print(f"Nova {config.model_name}: {total/1e6:.1f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, \
            f"Sequence {T} exceeds block size {self.config.block_size}"

        # Token + position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(
            self.transformer.wte(idx) + self.transformer.wpe(pos)
        )

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final layer norm
        x = self.transformer.ln_f(x)

        if targets is not None:
            # Training: compute loss over all positions
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
            return logits, loss
        else:
            # Inference: only last token (faster)
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        """
        Generate tokens autoregressively.
        temperature: higher = more random, lower = more focused
        top_k: only sample from top k most likely tokens
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context if too long
            idx_cond = idx if idx.size(1) <= self.config.block_size \
                       else idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_tok), dim=1)

        return idx

    def save_checkpoint(self, path, optimizer=None, step=0, loss=0.0):
        """Save model + optimizer state for resumable training."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'model_state': self.state_dict(),
            # vars(cls) is a mappingproxy that torch.save cannot pickle -> plain dict
            'config_dict': {
                k: getattr(self.config, k) for k in dir(self.config)
                if not k.startswith('_') and isinstance(getattr(self.config, k), (int, float, str, bool))
            },
            'step': step,
            'loss': loss,
        }
        if optimizer is not None:
            checkpoint['optimizer_state'] = optimizer.state_dict()
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path} (step {step}, loss {loss:.4f})")

    @classmethod
    def load_checkpoint(cls, path, config):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model = cls(config)
        model.load_state_dict(checkpoint['model_state'])
        print(f"Loaded checkpoint: {path} (step {checkpoint.get('step',0)}, loss {checkpoint.get('loss',0):.4f})")
        return model, checkpoint


if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from config import NovaChatConfig, NovaCodeConfig

    print("Testing Nova models...")
    for cfg in [NovaChatConfig, NovaCodeConfig]:
        model = NovaGPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        t = torch.randint(0, cfg.vocab_size, (2, 16))
        logits, loss = model(x, t)
        print(f"  logits: {logits.shape} | loss: {loss.item():.4f}")
        # Test generation
        seed = torch.randint(0, cfg.vocab_size, (1, 4))
        out  = model.generate(seed, max_new_tokens=8)
        print(f"  generate: {seed.shape} -> {out.shape}")
        print(f"  {cfg.model_name} OK")
    print("All tests passed.")
