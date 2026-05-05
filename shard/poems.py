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

# --- PROMPT VARIANTS ---
POEM_PROMPTS = [
    "Recite the poem titled {title}:",
    "Tell me the poem {title}:",
    "Provide the text for the poem {title}:"
]

def get_poem_prompt(title):
    return random.choice(POEM_PROMPTS).format(title=title)

def apply_random_base_mask(mask, chance):
    return [0 if (m == 1 and random.random() < chance) else m for m in mask]

def save_intermediate_packet(group, tokens, masks, sep_id):
    """
    Standardized sink for poem packets.
    """
    os.makedirs("./temp", exist_ok=True)
    d_path = f"./temp/{group}_data.tmp"
    m_path = f"./temp/{group}_mask.tmp"
    
    d_chunk = np.array(list(tokens) + [sep_id], dtype=c.TOKEN_TYPE)
    m_chunk = np.array(list(masks) + [0], dtype=c.MASK_TYPE)
    
    with open(d_path, "ab") as fd, open(m_path, "ab") as fm:
        fd.write(d_chunk.tobytes())
        fm.write(m_chunk.tobytes())

def run_poem_sharding(is_tune_mode):
    group_name = "poems_instruct" if is_tune_mode else "poems_base"
    
    tokenizer = Tokenizer.from_file(c.TOKEN_PATH)
    sep_id = c.get_sep_id(tokenizer)
    
    def get_token_id(s):
        tid = tokenizer.token_to_id(s)
        return tid if tid is not None else tokenizer.encode(s, add_special_tokens=False).ids[0]

    eos_id = get_token_id(c.EOS_TOKEN_STR)
    user_id = get_token_id(c.USER_OPEN)
    asst_id = get_token_id(c.ASSISTANT_OPEN)
    newline_id = get_token_id("\n")
    
    # Specific targeted files
    targets = ["poems.jsonl", "favorites.jsonl"]
    all_dataset_files = []
    for t in targets:
        all_dataset_files.extend(glob.glob(f"./datasets/**/{t}", recursive=True))

    file_metadata = {}
    print(f"\nPRE-SCANNING POEM DATASETS FOR {group_name.upper()}...")
    
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
            
            weight = 1
            if hasattr(c, "PATH_WEIGHT_OVERRIDES"):
                for key, w in c.PATH_WEIGHT_OVERRIDES.items():
                    if key in fpath_clean: weight = w

            file_metadata[fpath_clean] = {
                "rows": row_count,
                "est_chunks": est_chunks,
                "handle": open(fpath, 'r', encoding='utf-8'),
                "weight": weight,
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
        # Poems are almost always short, so we dump them as single packets
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
        text_content = data.get("text", "")
        title = data.get("title", "Untitled")
        
        tks_raw = tokenizer.encode(text_content, add_special_tokens=False).ids

        # Ensure content is present to avoid empty user/assistant rows
        has_text = text_content and str(text_content).strip().lower() != "none"

        if has_text and c.POEM_MIN_TOKENS <= len(tks_raw) <= c.POEM_MAX_TOKENS:
            # Respect weights by writing multiple packets if requested
            for _ in range(meta["weight"]):
                if is_tune_mode:
                    p_str = get_poem_prompt(title)
                    t_q = tokenizer.encode(p_str, add_special_tokens=False).ids
                    
                    full_tks = [user_id] + t_q + [newline_id, newline_id, asst_id] + tks_raw
                    # SFT Masking: Mask prompt, keep response
                    full_msk = [0] * (len(t_q) + 4) + [1] * (len(tks_raw) + 1)
                    process_sequence(full_tks, full_msk)
                else:
                    # Pre-training style: Randomly mask sequence
                    msk = apply_random_base_mask([1] * len(tks_raw), c.BASE_MASK_CHANCE)
                    process_sequence(tks_raw, msk)

    print(f"COMPLETED {group_name.upper()} INTERMEDIATE DUMP.")

if __name__ == "__main__":
    # Clear old temp files for the poem group
    for f in ["poems_base_data.tmp", "poems_base_mask.tmp", "poems_instruct_data.tmp", "poems_instruct_mask.tmp"]:
        target = f"./temp/{f}"
        if os.path.exists(target): os.remove(target)

    run_poem_sharding(is_tune_mode=False)
    run_poem_sharding(is_tune_mode=True)
