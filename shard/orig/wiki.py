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
REPORT_FILE = "wiki_report.log"
EOS_TOKEN_STR = "<|end_of_text|>"

# --- BUCKET SYSTEM CONFIG ---
MAX_OPEN_BUCKETS = 100 # Increased for high-volume wiki entries
PAD_TOKEN_ID = 3 

# --- MASKING CONFIGURATION ---
BASE_MASK_CHANCE = 0.15

# --- FILTER THRESHOLDS ---
KNOWLEDGE_MIN_TOKENS = 25    
WIKI_MAX_TOKENS = 1000        

# --- TOKENIZER MARKERS ---
USER_OPEN = "<|user|>"
ASSISTANT_OPEN = "<|assistant|>"
EOS_TOKEN_STR = "<|end_of_text|>"

def apply_random_base_mask(mask, chance):
    return [0 if (m == 1 and random.random() < chance) else m for m in mask]

def run_wiki_sharding(is_tune_mode):
    output_dir = "instruct" if is_tune_mode else "base"
    prefix = "wiki"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    tokenizer = Tokenizer.from_file(TOKEN_PATH)
    def get_token_id(s):
        tid = tokenizer.token_to_id(s)
        return tid if tid is not None else tokenizer.encode(s, add_special_tokens=False).ids[0]

    eos_id = get_token_id(EOS_TOKEN_STR)
    asst_id = get_token_id(ASSISTANT_OPEN)
    
    # Targeting wiki-specific files
    all_dataset_files = glob.glob("./datasets/**/wiki.jsonl", recursive=True)
    
    file_metadata = {}
    print(f"\nPRE-SCANNING WIKI DATASETS FOR {output_dir.upper()}...")
    
    for fpath in all_dataset_files:
        fpath_clean = fpath.replace("\\", "/")
        row_count, total_chars = 0, 0
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                row_count += 1
                total_chars += len(line)
        
        if row_count > 0:
            est_tokens = (total_chars // 4)
            est_chunks = max(1, est_tokens // CONTEXT_SIZE)
            file_metadata[fpath_clean] = {
                "rows": row_count,
                "est_chunks": est_chunks,
                "handle": open(fpath, 'r', encoding='utf-8'),
                "weight": 1,
                "accumulator": 0.0
            }

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

        # Wiki entries should fit in CONTEXT_SIZE, so no sliding window needed
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
        raw_text = data.get("text", "")
        t_enc = tokenizer.encode(raw_text, add_special_tokens=False)
        tks = list(t_enc.ids)

        # Knowledge-specific filtering
        if KNOWLEDGE_MIN_TOKENS <= len(tks) <= WIKI_MAX_TOKENS:
            if is_tune_mode:
                # In tune mode, Wiki entries are treated as background knowledge or summaries
                # We mask everything but the last 80% to force recall if relevant
                msk = [1] * len(tks)
                for i in range(len(tks) // 5): msk[i] = 0
            else:
                msk = apply_random_base_mask([1] * len(tks), BASE_MASK_CHANCE)
            
            pack_sequence(tks, msk)

    for b in active_buckets:
        if len(b['t']) > 0:
            pad_len = CONTEXT_SIZE - len(b['t'])
            write_window(b['t'] + [PAD_TOKEN_ID]*pad_len, b['m'] + [0]*pad_len)
    
    f_d.close(); f_m.close()
    print(f"COMPLETED {output_dir.upper()} WIKI SHARDING.")

if __name__ == "__main__":
    run_wiki_sharding(is_tune_mode=False)
    run_wiki_sharding(is_tune_mode=True)
