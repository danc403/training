import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from trainer.rope import NymphRoPE

# Aliasing for code compatibility
import torch.nn as te

# Compatibility Check: Native RMSNorm was added in Torch 2.1.
# On CPU, we ensure the custom implementation is robust for float32.
if not hasattr(nn, "RMSNorm"):
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
        def forward(self, x):
            # Explicitly cast to float32 for the mean square to avoid overflow if using higher dtypes elsewhere
            norm_x = torch.mean(x.pow(2), dim=-1, keepdim=True)
            x_normed = x * torch.rsqrt(norm_x + self.eps)
            return self.weight * x_normed
    nn.RMSNorm = RMSNorm
    te.RMSNorm = RMSNorm

class NymphBlock(nn.Module):
    """
    Standardized Transformer Layer for the Nymph family.
    Integrates Grouped Query Attention (GQA), QK-Norm, and RoPE.
    Optimized for CPU performance on AVX2-capable hardware.
    """
    def __init__(self, config, rope_module, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.rope = rope_module
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = config.get("head_dim", 64)
        self.group_size = self.num_heads // self.num_kv_heads
        
        # Norms
        self.input_layernorm = nn.RMSNorm(self.hidden_size, eps=config["rms_norm_eps"])
        self.post_attention_layernorm = nn.RMSNorm(self.hidden_size, eps=config["rms_norm_eps"])
        
        # Attention Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        
        # Residual projection
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.o_proj.residual_scale_flag = True
        
        # QK-Norm: Critical for stability in Muon-trained models
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config["rms_norm_eps"])
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config["rms_norm_eps"])
        
        # Gated MLP (SwiGLU)
        self.gate_proj = nn.Linear(self.hidden_size, config["intermediate_size"], bias=False)
        self.up_proj = nn.Linear(self.hidden_size, config["intermediate_size"], bias=False)
        
        # Down projection
        self.down_proj = nn.Linear(config["intermediate_size"], self.hidden_size, bias=False)
        self.down_proj.residual_scale_flag = True

    def forward(self, x, past_key_value=None, mask=None):
        residual = x
        x = self.input_layernorm(x)
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        batch_size, seq_len, _ = x.shape
        
        # Reshape for multi-head attention: (B, S, H, D) -> (B, H, S, D)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Apply Rotary Positional Embeddings
        q = self.rope(q)
        k = self.rope(k)
        
        # KV Caching logic
        if past_key_value is not None:
            prev_k, prev_v = past_key_value
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
        
        new_past_key_value = (k, v)
        
        # GQA expansion
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)
        
        # Native SDPA: Optimized on CPU via Intel MKL/OpenMP
        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=mask, 
            is_causal=(mask is None and past_key_value is None)
        )
        
        # Restore shape
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        x = self.o_proj(attn_output)
        x = x + residual
        
        # MLP Block
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        x = x + residual
        
        return x, new_past_key_value

class NymphModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rope = NymphRoPE(config)
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        
        self.layers = nn.ModuleList([
            NymphBlock(config, self.rope, i) for i in range(config["num_hidden_layers"])
        ])
        
        self.norm = nn.RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        
        if config.get("tie_word_embeddings", False):
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if hasattr(module, 'weight') and module.weight is not None:
            if isinstance(module, (nn.Linear, nn.Embedding)):
                std = 0.02
                if hasattr(module, 'residual_scale_flag'):
                    std *= (2 * self.config["num_hidden_layers"]) ** -0.5
                
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if hasattr(module, 'bias') and module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids, past_key_values=None, mask=None):
        x = self.embed_tokens(input_ids)
        
        new_past_key_values = []
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            
            # Checkpoint logic: helpful if RAM was tight, but we keep it modular.
            # On 24GB RAM Latitude for Sprite, this is likely overkill.
            if self.training and past_kv is None and self.config.get("use_checkpoint", False):
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                x, kv = checkpoint(create_custom_forward(layer), x, past_kv, mask, use_reentrant=False)
            else:
                x, kv = layer(x, past_key_value=past_kv, mask=mask)
                
            new_past_key_values.append(kv)
            
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits, new_past_key_values
