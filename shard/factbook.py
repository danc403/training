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
    Standardized sink for factbook and solar packets.
    """
    os.makedirs("./temp", exist_ok=True)
    d_path = f"./temp/{group}_data.tmp"
    m_path = f"./temp/{group}_mask.tmp"
    
    d_chunk = np.array(list(tokens) + [sep_id], dtype=c.TOKEN_TYPE)
    m_chunk = np.array(list(masks) + [0], dtype=c.MASK_TYPE)
    
    with open(d_path, "ab") as fd, open(m_path, "ab") as fm:
        fd.write(d_chunk.tobytes())
        fm.write(m_chunk.tobytes())

def run_factbook_sharding(is_tune_mode):
    group_name = "fact_solar_instruct" if is_tune_mode else "fact_solar_base"
    
    tokenizer = Tokenizer.from_file(c.TOKEN_PATH)
    sep_id = c.get_sep_id(tokenizer)
    
    def get_token_id(s):
        tid = tokenizer.token_to_id(s)
        return tid if tid is not None else tokenizer.encode(s, add_special_tokens=False).ids[0]

    eos_id = get_token_id(c.EOS_TOKEN_STR)
    user_id = get_token_id(c.USER_OPEN)
    asst_id = get_token_id(c.ASSISTANT_OPEN)
    ctx_o_id = get_token_id(c.CONTEXT_OPEN)
    ctx_c_id = get_token_id(c.CONTEXT_CLOSE)

    fb_files = glob.glob("./datasets/factbook/**/*.jsonl", recursive=True)
    solar_files = glob.glob("./datasets/solar/*.jsonl")
    all_target_files = fb_files + solar_files

    file_metadata = {}
    print(f"PRE-SCANNING FACTBOOK/SOLAR FOR {group_name.upper()}...")
    
    for fpath in all_target_files:
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
        prompt, resp, ctx = data.get("prompt"), data.get("response"), data.get("context", "")

        # Validation logic to skip empty or None user/assistant values
        has_prompt = prompt and str(prompt).strip().lower() != "none"
        has_resp = resp and str(resp).strip().lower() != "none"
        
        if not (has_prompt and has_resp):
            continue

        tks, msk = [], []
        if ctx:
            ctx_enc = tokenizer.encode(ctx, add_special_tokens=False)
            ctx_tks = list(ctx_enc.ids)
            ctx_msk = [1] * len(ctx_tks)
            
            # --- MAINTAINED DETERMINISTIC MASKING LOGIC ---
            if all(k in data for k in ["mask_pre", "mask_target"]):
                idx_pre = ctx.find(data["mask_pre"])
                if idx_pre != -1:
                    idx_target = ctx.find(data["mask_target"], idx_pre + len(data["mask_pre"]))
                    if idx_target != -1:
                        t_start, t_end = idx_pre, idx_target + len(data["mask_target"])
                        for i in range(len(ctx_tks)):
                            s, e = ctx_enc.offsets[i]
                            if not (e <= t_start or s >= t_end):
                                ctx_msk[i] = 0
            
            tks = [ctx_o_id] + ctx_tks + [ctx_c_id]
            msk = [1] + ctx_msk + [1]

        inst_str = f"{c.USER_OPEN}{prompt}\n\n{c.ASSISTANT_OPEN}{resp}{c.EOS_TOKEN_STR}"
        inst_enc = tokenizer.encode(inst_str, add_special_tokens=False)
        inst_tks = list(inst_enc.ids)
        
        if is_tune_mode:
            inst_msk = [1] * len(inst_tks)
            asst_idx = inst_str.find(c.ASSISTANT_OPEN)
            if asst_idx != -1:
                limit = asst_idx + len(c.ASSISTANT_OPEN)
                for i in range(len(inst_tks)):
                    s, e = inst_enc.offsets[i]
                    if s < limit: inst_msk[i] = 0
        else:
            inst_msk = apply_random_base_mask([1] * len(inst_tks), c.BASE_MASK_CHANCE)

        tks.extend(inst_tks)
        msk.extend(inst_msk)
        if msk: msk[-1] = 1 
        
        process_sequence(tks, msk)

    print(f"COMPLETED {group_name.upper()} FACTBOOK/SOLAR DUMP.")

if __name__ == "__main__":
    for f in ["fact_solar_base_data.tmp", "fact_solar_base_mask.tmp", "fact_solar_instruct_data.tmp", "fact_solar_instruct_mask.tmp"]:
        target = f"./temp/{f}"
        if os.path.exists(target): os.remove(target)

    run_factbook_sharding(is_tune_mode=False)
    run_factbook_sharding(is_tune_mode=True)
