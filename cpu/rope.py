import torch
import torch.nn as nn

class NymphRoPE(nn.Module):
    """
    Rotary Positional Embedding (RoPE) for the Nymph family.
    Standardized for a 2048 context length and Eager mode execution.
    Conforms to HF-style attribute naming and precision requirements.
    """
    def __init__(self, config):
        super().__init__()
        # Dynamic extraction from config to allow model family scaling
        self.head_dim = config.get("head_dim", 64)
        self.theta_base = config.get("rope_theta", 10000.0)
        self.max_seq_len = config.get("max_position_embeddings", 2048)
        
        # Precompute frequencies
        inv_freq = 1.0 / (self.theta_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(self.max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Set persistent=True to ensure buffers move with model.to(device)
        self.register_buffer("cos", emb.cos(), persistent=True)
        self.register_buffer("sin", emb.sin(), persistent=True)

    def forward(self, x):
        """
        Applies rotary embeddings to input tensor x.
        x shape: (Batch, Heads, SeqLen, HeadDim)
        """
        batch_size, num_heads, seq_len, head_dim = x.shape
        
        # Slice and reshape to (1, 1, SeqLen, HeadDim)
        # We cast to x.dtype (bfloat16) to ensure the pipeline doesn't break on type mismatch
        cos = self.cos[:seq_len, :].view(1, 1, seq_len, head_dim).to(dtype=x.dtype)
        sin = self.sin[:seq_len, :].view(1, 1, seq_len, head_dim).to(dtype=x.dtype)
        
        # Rotate half logic: [x1, x2] -> [-x2, x1]
        x1 = x[..., : head_dim // 2]
        x2 = x[..., head_dim // 2 :]
        rotated_half_x = torch.cat((-x2, x1), dim=-1)
        
        return (x * cos) + (rotated_half_x * sin)
