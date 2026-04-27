import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import AutoTokenizer
from config import UnifiedConfig
from model import NymphModel

class NymphEngine:
    def __init__(self, model_dir, tokenizer_path=None, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        config_file = os.path.join(model_dir, "config.json")
        weights_file = os.path.join(model_dir, "pytorch_model.bin")
        self.manager = UnifiedConfig(config_file, weights_file)
        self.config = self.manager.norm_config
        
        t_path = tokenizer_path if tokenizer_path else model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(t_path, trust_remote_code=True)
        
        # Importing the core architecture
        self.model = NymphModel(self.config)
        
        sd = self.manager.get_state_dict()
        if sd:
            self.model.load_state_dict(sd, strict=False)
            
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def generate(self, prompt, max_new=50, temperature=1.0):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        past_key_values = None
        
        for _ in range(max_new):
            model_in = input_ids if past_key_values is None else input_ids[:, -1:]
            logits, past_key_values = self.model(model_in, past_key_values=past_key_values)
            
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat((input_ids, next_tok), dim=1)
            if next_tok.item() in self.tokenizer.all_special_ids:
                break
                
        return self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
