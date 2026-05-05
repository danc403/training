import json
import os
import numpy as np
import random
from tokenizers import Tokenizer

# --- Configuration ---
CONTEXT_SIZE = 2048
STRIDE = 256
SHARD_SIZE = 10000000          
TOKEN_TYPE = np.uint16
MASK_TYPE = np.uint8
TOKEN_PATH = "./tokenizer/tokenizer.json"
EOS_TOKEN_STR = "<|end_of_text|>"
PAD_TOKEN_ID = 3 
BASE_MASK_CHANCE = 0.15
MAX_OPEN_BUCKETS = 50

# --- INJECTION & SOURCE VARIABLES ---
INCLUDE_SUMMARIES = True     
INCLUDE_INTRODUCTIONS = True 
INCLUDE_OSS = True           # Toggle for datasets/text/oss.jsonl
BOOK_MIN_TOKENS = 2500

# --- PROMPT VARIANTS ---
BOOK_PROMPTS = [
    "How does the book {title} begin?",
    "Start a book titled {title}:",
    "Give me the opening lines of {title}:"
]

SUMMARY_PROMPTS = [
    "Summarize the work titled {title}:",
    "Provide a summary of {title}:",
    "Give me an overview of {title} by {author}:",
    "What is {title} about?",
    "Summarize {title} by {author}:"
]

# --- MARKERS ---
USER_OPEN = "<|user|>"
ASSISTANT_OPEN = "<|assistant|>"

def get_summary_prompt(title, author):
    valid_templates = SUMMARY_PROMPTS if author else [p for p in SUMMARY_PROMPTS if "{author}" not in p]
    return random.choice(valid_templates).format(title=title, author=author if author else "")

def get_book_prompt(title):
    return random.choice(BOOK_PROMPTS).format(title=title)

def apply_random_base_mask(tokens, chance):
    return [0 if random.random() < chance else 1 for _ in range(len(tokens))]

def run_text_sharding(is_tune_mode=False):
    output_dir = "instruct" if is_tune_mode else "base"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    tokenizer = Tokenizer.from_file(TOKEN_PATH)
    
    def get_token_id(s):
        tid = tokenizer.token_to_id(s)
        return tid if tid is not None else tokenizer.encode(s, add_special_tokens=False).ids[0]

    eos_id = get_token_id(EOS_TOKEN_STR)
    user_id = get_token_id(USER_OPEN)
    asst_id = get_token_id(ASSISTANT_OPEN)
    newline_id = get_token_id("\n")

    shard_idx, tokens_in_shard = 0, 0
    f_d = open(os.path.join(output_dir, f"text_shard_{shard_idx}_data.bin"), "wb")
    f_m = open(os.path.join(output_dir, f"text_shard_{shard_idx}_mask.bin"), "wb")

    def write_window(t_list, m_list):
        nonlocal shard_idx, tokens_in_shard, f_d, f_m
        if tokens_in_shard >= SHARD_SIZE:
            f_d.close(); f_m.close(); shard_idx += 1
            f_d = open(os.path.join(output_dir, f"text_shard_{shard_idx}_data.bin"), "wb")
            f_m = open(os.path.join(output_dir, f"text_shard_{shard_idx}_mask.bin"), "wb")
            tokens_in_shard = 0
        np.array(t_list, dtype=TOKEN_TYPE).tofile(f_d)
        np.array(m_list, dtype=MASK_TYPE).tofile(f_m)
        tokens_in_shard += CONTEXT_SIZE

    active_buckets = []

    def pack_sequence(t_seq, m_seq):
        if not t_seq: return
        if t_seq[-1] != eos_id:
            t_seq.append(eos_id); m_seq.append(1)

        if len(t_seq) > CONTEXT_SIZE:
            pos = 0
            while pos + CONTEXT_SIZE <= len(t_seq):
                write_window(t_seq[pos:pos+CONTEXT_SIZE], m_seq[pos:pos+CONTEXT_SIZE])
                pos += (CONTEXT_SIZE - STRIDE)
            return

        for b in active_buckets:
            if len(b['t']) + len(t_seq) <= CONTEXT_SIZE:
                if len(b['t']) > 0: m_seq[0] = 0
                b['t'].extend(t_seq); b['m'].extend(m_seq)
                return
        if len(active_buckets) < MAX_OPEN_BUCKETS:
            active_buckets.append({'t': list(t_seq), 'm': list(m_seq)})
            return
        fullest = max(range(len(active_buckets)), key=lambda idx: len(active_buckets[idx]['t']))
        b = active_buckets[fullest]
        pad_len = CONTEXT_SIZE - len(b['t'])
        write_window(b['t'] + [PAD_TOKEN_ID]*pad_len, b['m'] + [0]*pad_len)
        active_buckets[fullest] = {'t': list(t_seq), 'm': list(m_seq)}

    # Define the list of files to process
    target_files = ["./datasets/text/noss.jsonl"]
    if INCLUDE_OSS:
        target_files.append("./datasets/text/oss.jsonl")

    for file_path in target_files:
        if not os.path.exists(file_path):
            print(f"Skipping missing file: {file_path}")
            continue
            
        print(f"Processing: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                tks = tokenizer.encode(data["text"], add_special_tokens=False).ids
                title = data.get("title")
                author = data.get("author", "")

                # 1. SUMMARY INJECTION (Instruct)
                if INCLUDE_SUMMARIES and data.get("summary") and title:
                    p_str = get_summary_prompt(title, author)
                    t_q = tokenizer.encode(p_str, add_special_tokens=False).ids
                    t_ans = tokenizer.encode(data["summary"], add_special_tokens=False).ids
                    s_tks = [user_id] + t_q + [newline_id, newline_id, asst_id] + t_ans + [eos_id]
                    s_msk = [0] * (len(t_q) + 4) + [1] * (len(t_ans) + 1)
                    pack_sequence(s_tks, s_msk)

                # 2. INTRODUCTION INJECTION (Instruct)
                if INCLUDE_INTRODUCTIONS and title and len(tks) >= BOOK_MIN_TOKENS:
                    p_str = get_book_prompt(title)
                    t_q = tokenizer.encode(p_str, add_special_tokens=False).ids
                    # Opening snippet
                    r_tks = [user_id] + t_q + [newline_id, newline_id, asst_id] + tks[:2000] + [eos_id]
                    r_msk = [0] * (len(t_q) + 4) + [1] * (min(len(tks), 2000) + 1)
                    pack_sequence(r_tks, r_msk)

                # 3. BASE SLIDING WINDOW (Full Text)
                if not is_tune_mode:
                    msk = apply_random_base_mask(tks, BASE_MASK_CHANCE)
                    pack_sequence(tks + [eos_id], msk + [1])

    # Final Flush
    for b in active_buckets:
        pad_len = CONTEXT_SIZE - len(b['t'])
        write_window(b['t'] + [PAD_TOKEN_ID]*pad_len, b['m'] + [0]*pad_len)
    f_d.close(); f_m.close()

if __name__ == "__main__":
    run_text_sharding(is_tune_mode=False) 
    run_text_sharding(is_tune_mode=True)
