import json
import os
import torch

# Model Family Configurations
MODEL_CONFIGS = {
    "sprite": {
        "vocab_size": 24000,
        "context_length": 2048,
        "emb_dim": 384,
        "n_heads": 6,
        "n_layers": 6,
        "hidden_dim": 2048,
        "head_dim": 64,
        "qk_norm": True,
        "n_kv_groups": 3,
        "rope_base": 10000.0,
        "dtype": torch.float32, # Changed from bfloat16 for AVX2 compatibility
        "device": "cpu",        # Explicitly setting CPU for the Latitude 5480 run
    },
    "nymph": {
        "vocab_size": 24000,
        "context_length": 2048,
        "emb_dim": 512,
        "n_heads": 8,
        "n_layers": 12,
        "hidden_dim": 2048,
        "head_dim": 64,
        "qk_norm": True,
        "n_kv_groups": 4,
        "rope_base": 10000.0,
        "dtype": torch.float32, # CPU native performance
        "device": "cpu",
    },
    "dragonfly": {
        "vocab_size": 24000,
        "context_length": 2048,
        "emb_dim": 512,
        "n_heads": 8,
        "n_layers": 16,
        "hidden_dim": 2048,
        "head_dim": 64,
        "qk_norm": True,
        "n_kv_groups": 4,
        "rope_base": 10000.0,
        "dtype": torch.float32,
        "device": "cpu",
    },
    "wyrm": {
        "vocab_size": 24000,
        "context_length": 2048,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 16,
        "hidden_dim": 2304,
        "head_dim": 64,
        "qk_norm": True,
        "n_kv_groups": 4,
        "rope_base": 10000.0,
        "dtype": torch.float32,
        "device": "cpu",
    }
}

class UnifiedConfig:
    def __init__(self, config_path=None, checkpoint_path=None, model_type=None):
        self.raw = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.raw = json.load(f)
        elif model_type and model_type.lower() in MODEL_CONFIGS:
            self.raw = {"config": MODEL_CONFIGS[model_type.lower()]}
        
        self.data = self.raw.get("config", self.raw)
        self.norm_config = self._normalize()
        
        # Pull device and dtype for global access in training loop
        self.device = self.data.get("device", "cpu")
        self.dtype = self.data.get("dtype", torch.float32)
        
        self.checkpoint = None
        if checkpoint_path and os.path.exists(checkpoint_path):
            # map_location="cpu" is mandatory for this branch
            self.checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    def _normalize(self):
        config = {
            "vocab_size": self.data.get("vocab_size", 24000),
            "hidden_size": self.data.get("emb_dim") or self.data.get("hidden_size"),
            "num_hidden_layers": self.data.get("n_layers") or self.data.get("num_hidden_layers"),
            "num_attention_heads": self.data.get("n_heads") or self.data.get("num_attention_heads"),
            "num_key_value_heads": self.data.get("n_kv_groups") or self.data.get("num_key_value_heads"),
            "rms_norm_eps": self.data.get("norm_eps") or self.data.get("rms_norm_eps", 1e-6),
            "rope_theta": self.data.get("rope_base") or self.data.get("rope_theta", 10000.0),
            "tie_word_embeddings": self.data.get("tie_word_embeddings", True),
            "max_position_embeddings": self.data.get("context_length") or self.data.get("max_position_embeddings", 2048)
        }

        if "hidden_dim" in self.data:
            config["intermediate_size"] = self.data["hidden_dim"]
        elif "intermediate_size" not in self.data:
            hidden = config["hidden_size"]
            multiple_of = 256
            intermediate = int(2 * (hidden * 4 / 3)) 
            intermediate = ((intermediate + multiple_of - 1) // multiple_of) * multiple_of
            config["intermediate_size"] = intermediate
        else:
            config["intermediate_size"] = self.data["intermediate_size"]

        return config

    def _get_weight_map(self):
        """
        Maps current iDragonfly HF-Standard keys to Qwen2 production keys.
        This now matches the naming in our updated model.py.
        """
        return {
            "embed_tokens.weight": "model.embed_tokens.weight",
            "norm.weight": "model.norm.weight",
            "lm_head.weight": "lm_head.weight",
            "layers.{i}.q_proj.weight": "model.layers.{i}.self_attn.q_proj.weight",
            "layers.{i}.k_proj.weight": "model.layers.{i}.self_attn.k_proj.weight",
            "layers.{i}.v_proj.weight": "model.layers.{i}.self_attn.v_proj.weight",
            "layers.{i}.o_proj.weight": "model.layers.{i}.self_attn.o_proj.weight",
            "layers.{i}.q_norm.weight": "model.layers.{i}.self_attn.q_norm.weight",
            "layers.{i}.k_norm.weight": "model.layers.{i}.self_attn.k_norm.weight",
            "layers.{i}.gate_proj.weight": "model.layers.{i}.mlp.gate_proj.weight",
            "layers.{i}.down_proj.weight": "model.layers.{i}.mlp.down_proj.weight",
            "layers.{i}.up_proj.weight": "model.layers.{i}.mlp.up_proj.weight",
            "layers.{i}.input_layernorm.weight": "model.layers.{i}.input_layernorm.weight",
            "layers.{i}.post_attention_layernorm.weight": "model.layers.{i}.post_attention_layernorm.weight",
        }

    def get_state_dict(self, map_to_qwen=False):
        if self.checkpoint is None:
            return None
        
        sd = self.checkpoint.get("model", self.checkpoint)
        if not map_to_qwen:
            return sd

        new_sd = {}
        mapping = self._get_weight_map()
        num_layers = self.norm_config["num_hidden_layers"]

        for k, v in sd.items():
            mapped_key = k
            for internal_key, qwen_key in mapping.items():
                if "{i}" in internal_key:
                    matched = False
                    for i in range(num_layers):
                        if k == internal_key.format(i=i):
                            mapped_key = qwen_key.format(i=i)
                            matched = True
                            break
                    if matched: break
                elif k == internal_key:
                    mapped_key = qwen_key
                    break
            new_sd[mapped_key] = v
            
        return new_sd

    def save_production_assets(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        hf_config = {
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "hidden_act": "silu",
            **self.norm_config
        }
        with open(os.path.join(output_dir, "config.json"), 'w') as f:
            json.dump(hf_config, f, indent=2)
            
        state_dict = self.get_state_dict(map_to_qwen=True)
        if state_dict is not None:
            torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
