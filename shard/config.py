import os
import glob
import numpy as np

CONTEXT_SIZE = 2048
STRIDE = 256
SHARD_SIZE = 10000000
TOKEN_TYPE = np.uint16
MASK_TYPE = np.uint8
TOKEN_PATH = "./tokenizer/tokenizer.json"
EOS_TOKEN_STR = "<|end_of_text|>"
PAD_TOKEN_ID = 3
# Dedicated string for the intermediate sharding boundary
INTERMEDIATE_SEP_STR = "<|eos|>" 
MAX_OPEN_BUCKETS = 100
BASE_MASK_CHANCE = 0.15

INCLUDE_SUMMARIES = True
INCLUDE_THINKING = True
INCLUDE_INTRODUCTIONS = True

TEXT_MIN_TOKENS = 375
TEXT_MAX_TOKENS = 400
POEM_MIN_TOKENS = 75
POEM_MAX_TOKENS = 650
BOOK_MIN_TOKENS = 2500
KNOWLEDGE_MIN_TOKENS = 25
WIKI_MAX_TOKENS = 1000

PATH_WEIGHT_OVERRIDES = {
    "solar.jsonl": 3,
    "factbook.jsonl": 2,
    "global.jsonl": 2,
    "favorites.jsonl": 2
}

USER_OPEN = "<|user|>"
ASSISTANT_OPEN = "<|assistant|>"
CONTEXT_OPEN = "<|context_start|>"
CONTEXT_CLOSE = "<|context_end|>"
THOUGHT_OPEN = "<think>"
THOUGHT_CLOSE = "</think>"
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESP_OPEN = "<tool_response>"
TOOL_RESP_CLOSE = "</tool_response>"

# --- Estimation Heuristics ---
# BPE Factor: Average characters per token. 
# 3.5 is calibrated for mixed JSONL/Code/Prose.
BPE_ESTIMATE_FACTOR = 3.5

def estimate_dataset_stats(file_input):
    """
    Calculates rough heuristics for JSONL datasets.
    Handles UTF-8 with error skipping for robustness.
    """
    if isinstance(file_input, str):
        files = [file_input]
    else:
        files = file_input

    total_rows = 0
    total_bytes = 0

    for fpath in files:
        if not os.path.exists(fpath):
            continue
            
        total_bytes += os.path.getsize(fpath)
        # Handle unicode and prevent choking on bad bytes
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip():
                    total_rows += 1

    # Heuristic: Total tokens approx = total characters / BPE factor
    total_tokens = int(total_bytes / BPE_ESTIMATE_FACTOR)
    
    avg_tokens_per_row = 0
    if total_rows > 0:
        avg_tokens_per_row = int(total_tokens / total_rows)

    return {
        "total_rows": total_rows,
        "total_tokens": total_tokens,
        "avg_tokens_per_row": avg_tokens_per_row,
        "total_files": len(files)
    }

def get_sep_id(tokenizer):
    """Utility to get the ID for the configured separator string."""
    tid = tokenizer.token_to_id(INTERMEDIATE_SEP_STR)
    return tid
