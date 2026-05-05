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

# --- Dataset Weights ---
# Solar and Factbook are high-signal; giving them higher weights for the math engine.
PATH_WEIGHT_OVERRIDES = {
    "solar.jsonl": 3,
    "factbook.jsonl": 2,
    "global.jsonl": 2
}

# --- Markers ---
USER_OPEN = "<|user|>"
ASSISTANT_OPEN = "<|assistant|>"
CONTEXT_OPEN = "<|context_start|>"
CONTEXT_CLOSE = "<|context_end|>"

def apply_random_base_mask(mask, chance):
    return [0 if (m == 1 and random.random() < chance) else m for m in mask]

def run_factbook_sharding(is_tune_mode):
    output_dir = "instruct" if is_tune_mode else "base"
    prefix = "fact_solar"
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

    # Recursive glob to handle nested factbook structures and specific solar path
    fb_files = glob.glob("./datasets/factbook/**/*.jsonl", recursive=True)
    solar_files = glob.glob("./datasets/solar/*.jsonl")
    all_target_files = fb_files + solar_files

    file_metadata = {}
    print(f"PRE-SCANNING FACTBOOK/SOLAR FOR {output_dir.upper()}...")
    
    for fpath in all_target_files:
        fpath_clean = fpath.replace("\\", "/")
        row_count, total_chars = 0, 0
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                row_count += 1
                total_chars += len(line)
        
        if row_count > 0:
            est_tokens = (total_chars // 4)
            est_chunks = max(1, est_tokens // CONTEXT_SIZE)
            
            weight = 1
            for key, w in PATH_WEIGHT_OVERRIDES.items():
                if key in fpath_clean:
                    weight = w

            file_metadata[fpath_clean] = {
                "rows": row_count,
                "est_chunks": est_chunks,
                "handle": open(fpath, 'r', encoding='utf-8'),
                "weight": weight,
                "accumulator": 0.0
            }

    # Deterministic round-robin math
    total_weighted_chunks = sum(m["est_chunks"] * m["weight"] for m in file_metadata.values())
    for fpath in file_metadata:
        meta = file_metadata[fpath]
        meta["step_value"] = (meta["est_chunks"] * meta["weight"]) / total_weighted_chunks

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

    # Processing loop
    active_files = list(file_metadata.keys())
    while active_files:
        for f in active_files:
            file_metadata[f]["accumulator"] += file_metadata[f]["step_value"]
        
        target_fpath = max(active_files, key=lambda f: file_metadata[f]["accumulator"])
        meta = file_metadata[target_fpath]
        meta["accumulator"] -= 1.0
        
        line = meta["handle"].readline()
        if not line:
            meta["handle"].close()
            active_files.remove(target_fpath)
            continue
            
        data = json.loads(line)
        prompt, resp, ctx = data.get("prompt"), data.get("response"), data.get("context", "")

        tks, msk = [], []
        if ctx:
            ctx_enc = tokenizer.encode(ctx, add_special_tokens=False)
            ctx_tks = list(ctx_enc.ids)
            ctx_msk = [1] * len(ctx_tks)
            
            # Deterministic masking for data tables/stats
            if all(k in data for k in ["mask_pre", "mask_target"]):
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

        inst_str = f"{USER_OPEN}{prompt}\n\n{ASSISTANT_OPEN}{resp}{EOS_TOKEN_STR}"
        inst_enc = tokenizer.encode(inst_str, add_special_tokens=False)
        inst_tks = list(inst_enc.ids)
        
        if is_tune_mode:
            inst_msk = [1] * len(inst_tks)
            asst_idx = inst_str.find(ASSISTANT_OPEN)
            if asst_idx != -1:
                limit = asst_idx + len(ASSISTANT_OPEN)
                for i in range(len(inst_tks)):
                    s, e = inst_enc.offsets[i]
                    if s < limit: inst_msk[i] = 0
        else:
            inst_msk = apply_random_base_mask([1] * len(inst_tks), BASE_MASK_CHANCE)

        tks.extend(inst_tks)
        msk.extend(inst_msk)
        msk[-1] = 1 # Always learn EOS
        
        pack_sequence(tks, msk)

    # Flush
    for b in active_buckets:
        if len(b['t']) > 0:
            pad_len = CONTEXT_SIZE - len(b['t'])
            write_window(b['t'] + [PAD_TOKEN_ID]*pad_len, b['m'] + [0]*pad_len)
    f_d.close(); f_m.close()

if __name__ == "__main__":
    run_factbook_sharding(is_tune_mode=False)
    run_factbook_sharding(is_tune_mode=True)
