import json
import os
import numpy as np
import random
import glob
from tokenizers import Tokenizer

# --- Configuration Integration ---
try:
    import shard.config as c
except ImportError:
    import config as c

def apply_random_base_mask(mask, chance):
    return [0 if (m == 1 and random.random() < chance) else m for m in mask]

def save_intermediate_packet(group, tokens, masks, sep_id):
    """
    Standardized sink for wiki packets.
    """
    os.makedirs("./temp", exist_ok=True)
    d_path = f"./temp/{group}_data.tmp"
    m_path = f"./temp/{group}_mask.tmp"
    
    d_chunk = np.array(list(tokens) + [sep_id], dtype=c.TOKEN_TYPE)
    m_chunk = np.array(list(masks) + [0], dtype=c.MASK_TYPE)
    
    with open(d_path, "ab") as fd, open(m_path, "ab") as fm:
        fd.write(d_chunk.tobytes())
        fm.write(m_chunk.tobytes())

def run_wiki_sharding(is_tune_mode):
    group_name = "wiki_instruct" if is_tune_mode else "wiki_base"
    
    tokenizer = Tokenizer.from_file(c.TOKEN_PATH)
    sep_id = c.get_sep_id(tokenizer)
    
    def get_token_id(s):
        tid = tokenizer.token_to_id(s)
        return tid if tid is not None else tokenizer.encode(s, add_special_tokens=False).ids[0]

    eos_id = get_token_id(c.EOS_TOKEN_STR)
    
    all_dataset_files = glob.glob("./datasets/**/wiki.jsonl", recursive=True)
    
    file_metadata = {}
    print(f"\nPRE-SCANNING WIKI DATASETS FOR {group_name.upper()}...")
    
    for fpath in all_dataset_files:
        fpath_clean = fpath.replace("\\", "/")
        row_count, total_chars = 0, 0
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                row_count += 1
                total_chars += len(line)
        
        if row_count > 0:
            est_tokens = (total_chars // 4)
            est_chunks = max(1, est_tokens // c.CONTEXT_SIZE)
            file_metadata[fpath_clean] = {
                "rows": row_count,
                "est_chunks": est_chunks,
                "handle": open(fpath, 'r', encoding='utf-8'),
                "weight": 1,
                "accumulator": 0.0
            }

    total_weighted_chunks = sum(m["est_chunks"] * m["weight"] for m in file_metadata.values())
    if total_weighted_chunks == 0: return

    for fpath in file_metadata:
        meta = file_metadata[fpath]
        meta["step_value"] = (meta["est_chunks"] * meta["weight"]) / total_weighted_chunks

    def process_sequence(t_seq, m_seq):
        if not t_seq: return
        if t_seq[-1] != eos_id:
            t_seq.append(eos_id); m_seq.append(1)
        
        # Standard Wiki Striding
        if len(t_seq) > c.CONTEXT_SIZE:
            pos = 0
            while pos + c.CONTEXT_SIZE <= len(t_seq):
                save_intermediate_packet(group_name, t_seq[pos:pos+c.CONTEXT_SIZE], m_seq[pos:pos+c.CONTEXT_SIZE], sep_id)
                pos += (c.CONTEXT_SIZE - c.STRIDE)
            return
            
        save_intermediate_packet(group_name, t_seq, m_seq, sep_id)

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
        
        # Validation to ensure no empty rows are processed
        if not raw_text or str(raw_text).strip().lower() == "none":
            continue

        tks = tokenizer.encode(raw_text, add_special_tokens=False).ids

        if c.KNOWLEDGE_MIN_TOKENS <= len(tks) <= c.WIKI_MAX_TOKENS:
            if is_tune_mode:
                # 20% Prefix Masking logic
                msk = [1] * len(tks)
                mask_boundary = len(tks) // 5
                for i in range(mask_boundary): msk[i] = 0
                process_sequence(tks, msk)
            else:
                msk = apply_random_base_mask([1] * len(tks), c.BASE_MASK_CHANCE)
                process_sequence(tks, msk)

    print(f"COMPLETED {group_name.upper()} WIKI DUMP.")

if __name__ == "__main__":
    for f in ["wiki_base_data.tmp", "wiki_base_mask.tmp", "wiki_instruct_data.tmp", "wiki_instruct_mask.tmp"]:
        target = f"./temp/{f}"
        if os.path.exists(target): os.remove(target)

    run_wiki_sharding(is_tune_mode=False)
    #run_wiki_sharding(is_tune_mode=True)
