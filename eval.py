#!/usr/bin/env python3

import os
import sys
import torch
import json
import glob
import re
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

# Import existing project files
from trainer.model import NymphModel
from trainer.config import UnifiedConfig

# EVALUATION CONFIGURATION
CHECKPOINT_DIR = "./checkpoints"
TOKENIZER_PATH = "./tokenizer/tokenizer.json"
DEFAULT_OUTPUT_FILE = "model.eval.log"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if "--cpu" in sys.argv:
    DEVICE = "cpu"

# Handle optional model argument
TARGET_MODEL = None
if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
    TARGET_MODEL = sys.argv[1]
    OUTPUT_FILE = f"{TARGET_MODEL}.eval.log"
else:
    OUTPUT_FILE = DEFAULT_OUTPUT_FILE

# Updated questions to match shard.py marker format: <|user|>Prompt\n\n<|assistant|>
# Injected context-aware prompts to verify sandbox retrieval performance
EVAL_QUESTIONS = [
    "<|user|>The night sky was streaked with a brilliant\n\n<|assistant|>",
    "<|user|>Laugh and the world laughs with you\n\n<|assistant|>",
    "<|user|>Tell me about the Pokemon Pikachu.\n\n<|assistant|>",
    "<|user|>List the base stats for the Pokemon Abomasnow.\n\n<|assistant|>",
    "<|user|>What is the capital of France?\n\n<|assistant|>",
    "<|user|>Write a poem about a fucking cat.\n\n<|assistant|>",
    "<|user|>What is the Earth's only natural satellite?\n\n<|assistant|>",
    "<|user|>If I have two apples and buy two more, how many do I have?\n\n<|assistant|>",
    "<|user|>Explain the concept of a gravitational singularity.\n\n<|assistant|>",
    "<|user|>Who wrote the book Children of Time?\n\n<|assistant|>",
    "<|context_start|>The solar array consists of four 300 watt panels producing 1200 watts total.<|context_end|><|user|>How much total power does the solar array produce?\n\n<|assistant|>",
    "<|context_start|>The iDragonfly suite uses a custom TTS engine with GGML math functions to reduce latency.<|context_end|><|user|>What does the iDragonfly suite use for its TTS engine?\n\n<|assistant|>"
]

SAMPLING_PARAMS = {
    "temperature": 0.0,
    "top_p": 0.9,
    "top_k": 40,
    "rep_penalty": 1.35,
    "max_new_tokens": 50
}

def apply_repetition_penalty(logits, tokens, penalty):
    if penalty == 1.0:
        return logits
    score = torch.gather(logits, 1, tokens)
    score = torch.where(score > 0, score / penalty, score * penalty)
    logits.scatter_(1, tokens, score)
    return logits

def sample(logits, temperature, top_p, top_k):
    logits = logits / max(temperature, 1e-5)
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float('-inf')
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def run_eval():
    print(f"--- iDragonfly Automated Eval | Device: {DEVICE} ---")
    
    if not os.path.exists(CHECKPOINT_DIR):
        print("Checkpoint directory not found.")
        return

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
    
    # MANUAL STOP ID: Explicitly set to ID 8 (<|end_of_text|>) 
    # to match shard.py and handle null post_processor issues.
    STOP_ID = 8
    
    # Filter families based on TARGET_MODEL argument
    if TARGET_MODEL:
        if os.path.isdir(os.path.join(CHECKPOINT_DIR, TARGET_MODEL)):
            families = [TARGET_MODEL]
        else:
            print(f"Error: Model directory '{TARGET_MODEL}' not found in {CHECKPOINT_DIR}.")
            return
    else:
        families = [d for d in os.listdir(CHECKPOINT_DIR) if os.path.isdir(os.path.join(CHECKPOINT_DIR, d))]
    
    with open(OUTPUT_FILE, 'a') as f_out:
        for family in families:
            model_dir = os.path.join(CHECKPOINT_DIR, family)
            config_path = os.path.join(model_dir, "config.json")
            unified_conf = UnifiedConfig(config_path=config_path if os.path.exists(config_path) else None, model_type=family)
            
            model = NymphModel(unified_conf.norm_config).to(DEVICE)
            
            # Find regular step checkpoints
            checkpoints = glob.glob(os.path.join(model_dir, "*.pt"))
            
            # Also look in the /final/ subdirectory for the last .pt file
            final_dir = os.path.join(model_dir, "final")
            if os.path.exists(final_dir):
                final_checkpoints = glob.glob(os.path.join(final_dir, "*.pt"))
                checkpoints.extend(final_checkpoints)
            
            # Sort by step number
            checkpoints.sort(key=lambda x: int(re.search(r'step_(\d+)', x).group(1)) if re.search(r'step_(\d+)', x) else 0)
            
            for cp_path in checkpoints:
                is_final = "final" in cp_path
                step_match = re.search(r'step_(\d+)', cp_path)
                step_num = step_match.group(1) if step_match else "unknown"
                
                step_name = f"step_{step_num}_final" if is_final else f"step_{step_num}"
                eval_id = f"{family}_{step_name}"
                
                print(f"Testing {eval_id}...")
                
                raw_sd = torch.load(cp_path, map_location=DEVICE, weights_only=True)
                state_dict = raw_sd.get("model", raw_sd)
                new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict, strict=True)
                model.eval()
                
                eval_entry = {"eval_id": eval_id, "results": []}
                
                for q in EVAL_QUESTIONS:
                    ids = tokenizer.encode(q)
                    input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)
                    
                    with torch.no_grad():
                        for _ in range(SAMPLING_PARAMS["max_new_tokens"]):
                            logits, _ = model(input_ids)
                            next_token_logits = logits[:, -1, :].clone()
                            next_token_logits = apply_repetition_penalty(next_token_logits, input_ids, SAMPLING_PARAMS["rep_penalty"])
                            next_tok = sample(next_token_logits, SAMPLING_PARAMS["temperature"], SAMPLING_PARAMS["top_p"], SAMPLING_PARAMS["top_k"])
                            
                            input_ids = torch.cat((input_ids, next_tok), dim=1)
                            
                            # Explicitly check for our trained STOP_ID
                            if next_tok.item() == STOP_ID:
                                break
                    
                    response = tokenizer.decode(input_ids[0][len(ids):]).strip()
                    eval_entry["results"].append({"question": q, "answer": response})
                
                f_out.write(json.dumps(eval_entry) + "\n")
                f_out.flush()

    print(f"Done. Results written to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_eval()
