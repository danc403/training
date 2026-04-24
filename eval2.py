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
DEFAULT_OUTPUT_FILE = "context_rag_eval.log"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if "--cpu" in sys.argv:
    DEVICE = "cpu"

TARGET_MODEL = None
if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
    TARGET_MODEL = sys.argv[1]
    OUTPUT_FILE = f"{TARGET_MODEL}.rag.eval.log"
else:
    OUTPUT_FILE = DEFAULT_OUTPUT_FILE

# TIERED RAG QUESTIONS
# Aligned to use <|context_start|> and <|context_end|> to match shard.py logic.
EVAL_QUESTIONS = [
    {
        "tier": "Sprite",
        "context": "The sky appears blue because of Rayleigh scattering. Short-wave blue light is scattered more than long-wave red light by the atmosphere.",
        "prompt": "What scientific effect makes the sky appear blue?"
    },
    {
        "tier": "Wyrm",
        "context": "The sky appears blue because of Rayleigh scattering. Short-wave blue light is scattered more than long-wave red light by the atmosphere.",
        "prompt": "Explain why the sky isn't red during the day."
    },
    {
        "tier": "Sprite",
        "context": "A 48V battery bank typically consists of four 12V batteries in series. Hybrid inverters manage the DC-to-AC conversion while simultaneously charging the bank from solar arrays.",
        "prompt": "How many 12V batteries are needed for a 48V bank?"
    },
    {
        "tier": "Nymph",
        "context": "A 48V battery bank typically consists of four 12V batteries in series. Hybrid inverters manage the DC-to-AC conversion while simultaneously charging the bank from solar arrays.",
        "prompt": "How do I connect my 12V batteries to a 48V hybrid system?"
    },
    {
        "tier": "Sprite",
        "context": "Lugia is a Psychic/Flying type Pokemon known as the Guardian of the Seas. It resides in the Whirl Islands.",
        "prompt": "What is the elemental type of Lugia?"
    },
    {
        "tier": "Nymph",
        "context": "Lugia is a Psychic/Flying type Pokemon known as the Guardian of the Seas. It resides in the Whirl Islands.",
        "prompt": "Where can I find the master of the legendary birds?"
    },
    {
        "tier": "Sprite",
        "context": "Total Virtual Memory = Physical RAM + Swap Space.",
        "prompt": "What two components make up Total Virtual Memory?"
    },
    {
        "tier": "Wyrm",
        "context": "Total Virtual Memory = Physical RAM + Swap Space.",
        "prompt": "I have 32GB of RAM and 8GB of swap. What is my total memory capacity?"
    },
    {
        "tier": "Sprite",
        "context": "Floor joists for a 10-foot span in a tiny home should typically be 2x6 or 2x8 depending on the species and grade of lumber used, spaced 16 inches on center.",
        "prompt": "What is the typical spacing for floor joists in a tiny home?"
    },
    {
        "tier": "Nymph",
        "context": "Floor joists for a 10-foot span in a tiny home should typically be 2x6 or 2x8 depending on the species and grade of lumber used, spaced 16 inches on center.",
        "prompt": "What lumber should I buy for my 10x24 floor frame?"
    },
    {
        "tier": "Sprite",
        "context": "International Residential Code (IRC) requires a minimum of 1/4 unit vertical for every 12 units horizontal for adequate roof drainage.",
        "prompt": "What is the minimum vertical rise required for 12 units of horizontal run?"
    },
    {
        "tier": "Wyrm",
        "context": "The Second Law of Thermodynamics states that entropy always increases, leading eventually to the heat death of the universe.",
        "prompt": "Write a short prose snippet about the end of all energy."
    }
]

SAMPLING_PARAMS = {
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 40,
    "rep_penalty": 1.3,
    "max_new_tokens": 80
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
    print(f"--- iDragonfly Tiered Context Eval | Device: {DEVICE} ---")
    
    if not os.path.exists(CHECKPOINT_DIR):
        print("Checkpoint directory not found.")
        return

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
    STOP_ID = 8 
    
    if TARGET_MODEL:
        families = [TARGET_MODEL] if os.path.isdir(os.path.join(CHECKPOINT_DIR, TARGET_MODEL)) else []
    else:
        families = [d for d in os.listdir(CHECKPOINT_DIR) if os.path.isdir(os.path.join(CHECKPOINT_DIR, d))]
    
    with open(OUTPUT_FILE, 'a') as f_out:
        for family in families:
            model_dir = os.path.join(CHECKPOINT_DIR, family)
            config_path = os.path.join(model_dir, "config.json")
            unified_conf = UnifiedConfig(config_path=config_path if os.path.exists(config_path) else None, model_type=family)
            model = NymphModel(unified_conf.norm_config).to(DEVICE)
            
            checkpoints = glob.glob(os.path.join(model_dir, "*.pt"))
            final_dir = os.path.join(model_dir, "final")
            if os.path.exists(final_dir):
                checkpoints.extend(glob.glob(os.path.join(final_dir, "*.pt")))
            
            checkpoints.sort(key=lambda x: int(re.search(r'step_(\d+)', x).group(1)) if re.search(r'step_(\d+)', x) else 0)
            
            for cp_path in checkpoints:
                step_match = re.search(r'step_(\d+)', cp_path)
                step_num = step_match.group(1) if step_match else "unknown"
                eval_id = f"{family}_rag_step_{step_num}"
                
                print(f"Testing {eval_id}...")
                raw_sd = torch.load(cp_path, map_location=DEVICE, weights_only=True)
                state_dict = raw_sd.get("model", raw_sd)
                new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict, strict=True)
                model.eval()
                
                eval_entry = {"eval_id": eval_id, "results": []}
                
                for item in EVAL_QUESTIONS:
                    # Repaired format to use context markers and correct newlines
                    formatted_q = f"<|context_start|>{item['context']}<|context_end|><|user|>{item['prompt']}\n\n<|assistant|>"
                    
                    ids = tokenizer.encode(formatted_q)
                    input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)
                    
                    with torch.no_grad():
                        for _ in range(SAMPLING_PARAMS["max_new_tokens"]):
                            logits, _ = model(input_ids)
                            next_token_logits = logits[:, -1, :].clone()
                            next_token_logits = apply_repetition_penalty(next_token_logits, input_ids, SAMPLING_PARAMS["rep_penalty"])
                            next_tok = sample(next_token_logits, SAMPLING_PARAMS["temperature"], SAMPLING_PARAMS["top_p"], SAMPLING_PARAMS["top_k"])
                            input_ids = torch.cat((input_ids, next_tok), dim=1)
                            if next_tok.item() == STOP_ID:
                                break
                    
                    response = tokenizer.decode(input_ids[0][len(ids):]).strip()
                    eval_entry["results"].append({
                        "tier": item["tier"],
                        "question": formatted_q, 
                        "answer": response
                    })
                
                f_out.write(json.dumps(eval_entry) + "\n")
                f_out.flush()

    print(f"Done. Results written to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_eval()
