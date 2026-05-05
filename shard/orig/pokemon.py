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
MAX_OPEN_BUCKETS = 50
BASE_MASK_CHANCE = 0.15

# --- INJECTION TICKET ---
INCLUDE_THINKING = True 

# --- Paths ---
INPUT_DIR = "./datasets/pokemon/"
INPUT_FILES = glob.glob(os.path.join(INPUT_DIR, "*.jsonl"))

# --- MARKERS ---
USER_OPEN = "<|user|>"
ASSISTANT_OPEN = "<|assistant|>"
CONTEXT_OPEN = "<|context_start|>"
CONTEXT_CLOSE = "<|context_end|>"
THOUGHT_OPEN = "<think>"
THOUGHT_CLOSE = "</think>"

def apply_random_base_mask(tokens, chance):
    return [0 if random.random() < chance else 1 for _ in range(len(tokens))]

def run_pokemon_sharding(is_tune_mode):
    output_dir = "instruct" if is_tune_mode else "base"
    prefix = "pokemon"
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    tokenizer = Tokenizer.from_file(TOKEN_PATH)
    
    def get_token_id(s):
        tid = tokenizer.token_to_id(s)
        return tid if tid is not None else tokenizer.encode(s, add_special_tokens=False).ids[0]

    eos_id = get_token_id(EOS_TOKEN_STR)
    user_id = get_token_id(USER_OPEN)
    asst_id = get_token_id(ASSISTANT_OPEN)
    ctx_o_id = get_token_id(CONTEXT_OPEN)
    ctx_c_id = get_token_id(CONTEXT_CLOSE)
    th_o_id = get_token_id(THOUGHT_OPEN)
    th_c_id = get_token_id(THOUGHT_CLOSE)

    # --- PRE-SCAN LOGIC (From original shard.py) ---
    file_metadata = {}
    for fpath in INPUT_FILES:
        row_count, total_chars = 0, 0
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                row_count += 1
                total_chars += len(line)
        if row_count > 0:
            est_chunks = max(1, (total_chars // 4) // CONTEXT_SIZE)
            file_metadata[fpath] = {
                "handle": open(fpath, 'r', encoding='utf-8'),
                "accumulator": 0.0,
                "step_value": 1.0 / row_count # Simple round-robin for this module
            }

    shard_idx, tokens_in_shard = 0, 0
    f_d = open(os.path.join(output_dir, f"{prefix}_shard_{shard_idx}_data.bin"), "wb")
    f_m = open(os.path.join(output_dir, f"{prefix}_shard_{shard_idx}_mask.bin"), "wb")

    def write_window(t_list, m_list):
        nonlocal shard_idx, tokens_in_shard, f_d, f_m
        if tokens_in_shard >= SHARD_SIZE:
            f_d.close(); f_m.close(); shard_idx += 1
            f_d = open(os.path.join(output_dir, f"{prefix}_shard_{shard_idx}_data.bin"), "wb")
            f_m = open(os.path.join(output_dir, f"{prefix}_shard_{shard_idx}_mask.bin"), "wb")
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

    # --- MAIN PROCESSING LOOP ---
    active_files = list(file_metadata.keys())
    while active_files:
        for fpath in list(active_files):
            line = file_metadata[fpath]["handle"].readline()
            if not line:
                file_metadata[fpath]["handle"].close()
                active_files.remove(fpath)
                continue
            
            data = json.loads(line)
            tks, msk = [], []

            # 1. Handle Context (with Deterministic Masking support)
            ctx = data.get("context", "")
            if ctx:
                ctx_enc = tokenizer.encode(ctx, add_special_tokens=False)
                ctx_tks = list(ctx_enc.ids)
                ctx_msk = [1] * len(ctx_tks)

                # Deterministic logic from shard.py
                if is_tune_mode and all(k in data for k in ["mask_pre", "mask_target"]):
                    idx_pre = ctx.find(data["mask_pre"])
                    idx_target = ctx.find(data["mask_target"], idx_pre + len(data["mask_pre"]))
                    if idx_pre != -1 and idx_target != -1:
                        t_start, t_end = idx_pre, idx_target + len(data["mask_target"])
                        for i in range(len(ctx_tks)):
                            s, e = ctx_enc.offsets[i]
                            if not (e <= t_start or s >= t_end):
                                ctx_msk[i] = 0
                
                tks = [ctx_o_id] + ctx_tks + [ctx_c_id]
                msk = [1] + ctx_msk + [1]

            # 2. Instruction / User Block
            prompt = data.get("prompt", "")
            p_tks = tokenizer.encode(prompt, add_special_tokens=False).ids
            tks += [user_id] + p_tks
            msk += [0] * (len(p_tks) + 1) if is_tune_mode else [1] * (len(p_tks) + 1)

            # 3. Thinking Block
            thought = data.get("thought") or data.get("thinking")
            if INCLUDE_THINKING and thought:
                th_tks = tokenizer.encode(thought, add_special_tokens=False).ids
                tks += [th_o_id] + th_tks + [th_c_id]
                msk += [1] * (len(th_tks) + 2) if is_tune_mode else [1] * (len(th_tks) + 2)

            # 4. Response Block
            resp = data.get("response", "")
            r_tks = tokenizer.encode(resp, add_special_tokens=False).ids
            tks += [asst_id] + r_tks + [eos_id]
            msk += [1] * (len(r_tks) + 2)

            # Base Mode Random Masking Override
            if not is_tune_mode:
                msk = apply_random_base_mask(tks, BASE_MASK_CHANCE)
                msk[-1] = 1

            pack_sequence(tks, msk)

    # Final Flush
    for b in active_buckets:
        if len(b['t']) > 0:
            pad_len = CONTEXT_SIZE - len(b['t'])
            write_window(b['t'] + [PAD_TOKEN_ID]*pad_len, b['m'] + [0]*pad_len)
    f_d.close(); f_m.close()

if __name__ == "__main__":
    run_pokemon_sharding(is_tune_mode=False)
    run_pokemon_sharding(is_tune_mode=True)
