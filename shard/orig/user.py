import json
import os
import numpy as np
import random
import glob
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

# --- INJECTION VARIABLES ---
INCLUDE_SUMMARIES = True     
INCLUDE_INTRODUCTIONS = True 
BOOK_MIN_TOKENS = 2000

# --- MARKERS ---
USER_OPEN = "<|user|>"
ASSISTANT_OPEN = "<|assistant|>"
THOUGHT_OPEN = "<|thought|>"
THOUGHT_CLOSE = "</thought>"
CONTEXT_OPEN = "<|context_start|>"
CONTEXT_CLOSE = "<|context_end|>"

# --- PROMPT VARIANTS ---
SUMMARY_PROMPTS = [
    "Summarize the following: {title}",
    "Provide a summary of {title} by {author}:",
    "Give me an overview of {title}:"
]

def get_summary_prompt(title, author):
    valid = SUMMARY_PROMPTS if author else [p for p in SUMMARY_PROMPTS if "{author}" not in p]
    return random.choice(valid).format(title=title, author=author if author else "")

def apply_random_base_mask(length, chance):
    return [0 if random.random() < chance else 1 for _ in range(length)]

def get_row_capabilities(data):
    caps = []
    if "prompt" in data and "response" in data: caps.append("INSTRUCT")
    if "context" in data: caps.append("HAS_CONTEXT")
    if "thought" in data: caps.append("HAS_THOUGHT")
    if "text" in data: caps.append("RAW_TEXT")
    if all(k in data for k in ["mask_pre", "mask_target"]): caps.append("DETERMINISTIC")
    return caps

def run_user_data_sharding(is_tune_mode=False):
    output_dir = "instruct" if is_tune_mode else "base"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    tokenizer = Tokenizer.from_file(TOKEN_PATH)
    def get_token_id(s):
        tid = tokenizer.token_to_id(s)
        return tid if tid is not None else tokenizer.encode(s, add_special_tokens=False).ids[0]

    eos_id = get_token_id(EOS_TOKEN_STR)
    user_id = get_token_id(USER_OPEN)
    asst_id = get_token_id(ASSISTANT_OPEN)
    th_o_id = get_token_id(THOUGHT_OPEN)
    th_c_id = get_token_id(THOUGHT_CLOSE)
    ctx_o_id = get_token_id(CONTEXT_OPEN)
    ctx_c_id = get_token_id(CONTEXT_CLOSE)
    newline_id = get_token_id("\n")

    shard_idx, tokens_in_shard = 0, 0
    f_d = open(os.path.join(output_dir, f"user_shard_{shard_idx}_data.bin"), "wb")
    f_m = open(os.path.join(output_dir, f"user_shard_{shard_idx}_mask.bin"), "wb")

    def write_window(t_list, m_list):
        nonlocal shard_idx, tokens_in_shard, f_d, f_m
        if tokens_in_shard >= SHARD_SIZE:
            f_d.close(); f_m.close(); shard_idx += 1
            f_d = open(os.path.join(output_dir, f"user_shard_{shard_idx}_data.bin"), "wb")
            f_m = open(os.path.join(output_dir, f"user_shard_{shard_idx}_mask.bin"), "wb")
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

    user_files = glob.glob("./datasets/user_data/*.jsonl")
    
    for fpath in user_files:
        print(f"Routing User Data: {fpath}")
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                caps = get_row_capabilities(data)
                
                # --- ROUTE 1: INSTRUCT (with Context, Thought, or Deterministic) ---
                if "INSTRUCT" in caps:
                    tks, msk = [], []
                    
                    # Add Context if present
                    if "HAS_CONTEXT" in caps:
                        c_enc = tokenizer.encode(data["context"], add_special_tokens=False)
                        c_ids = list(c_enc.ids)
                        c_msk = [1] * len(c_ids)
                        
                        if "DETERMINISTIC" in caps:
                            pre, tar = data["mask_pre"], data["mask_target"]
                            idx_pre = data["context"].find(pre)
                            idx_tar = data["context"].find(tar, idx_pre + len(pre))
                            if idx_pre != -1 and idx_tar != -1:
                                t_start, t_end = idx_pre, idx_tar + len(tar)
                                for i in range(len(c_ids)):
                                    s, e = c_enc.offsets[i]
                                    if not (e <= t_start or s >= t_end): c_msk[i] = 0
                        
                        tks += [ctx_o_id] + c_ids + [ctx_c_id]
                        msk += [1] + c_msk + [1]

                    # User Prompt
                    p_ids = tokenizer.encode(f"{USER_OPEN}{data['prompt']}\n\n", add_special_tokens=False).ids
                    tks += p_ids
                    msk += [0] * len(p_ids)

                    # Thought Block (if present)
                    tks += [asst_id]
                    msk += [0]
                    if "HAS_THOUGHT" in caps:
                        th_ids = tokenizer.encode(data["thought"], add_special_tokens=False).ids
                        tks += [th_o_id] + th_ids + [th_c_id]
                        msk += [1] * (len(th_ids) + 2) # Thoughts are often masked 0 in some protocols, but we keep 1 for reasoning reinforcement

                    # Response
                    r_ids = tokenizer.encode(data["response"], add_special_tokens=False).ids
                    tks += r_ids
                    msk += [1] * len(r_ids)
                    pack_sequence(tks, msk)

                # --- ROUTE 2: RAW TEXT (Articles, Stories, Logs) ---
                if "RAW_TEXT" in caps:
                    full_tks = tokenizer.encode(data["text"], add_special_tokens=False).ids
                    
                    if is_tune_mode:
                        # Optional Summary Injection
                        if INCLUDE_SUMMARIES and data.get("summary") and data.get("title"):
                            p_str = get_summary_prompt(data["title"], data.get("author", ""))
                            t_q = tokenizer.encode(p_str, add_special_tokens=False).ids
                            t_ans = tokenizer.encode(data["summary"], add_special_tokens=False).ids
                            s_tks = [user_id] + t_q + [newline_id, newline_id, asst_id] + t_ans
                            s_msk = [0] * (len(t_q) + 5) + [1] * len(t_ans)
                            pack_sequence(s_tks, s_msk)
                    else:
                        # Standard sliding window for base training
                        base_msk = apply_random_base_mask(len(full_tks), BASE_MASK_CHANCE)
                        pack_sequence(full_tks, base_msk)

    # Final Flush
    for b in active_buckets:
        pad_len = CONTEXT_SIZE - len(b['t'])
        write_window(b['t'] + [PAD_TOKEN_ID]*pad_len, b['m'] + [0]*pad_len)
    f_d.close(); f_m.close()

if __name__ == "__main__":
    run_user_data_sharding(is_tune_mode=False)
    run_user_data_sharding(is_tune_mode=True)
