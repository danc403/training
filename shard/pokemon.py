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
    Standardized sink for pokemon packets.
    """
    os.makedirs("./temp", exist_ok=True)
    d_path = f"./temp/{group}_data.tmp"
    m_path = f"./temp/{group}_mask.tmp"
    
    d_chunk = np.array(list(tokens) + [sep_id], dtype=c.TOKEN_TYPE)
    m_chunk = np.array(list(masks) + [0], dtype=c.MASK_TYPE)
    
    with open(d_path, "ab") as fd, open(m_path, "ab") as fm:
        fd.write(d_chunk.tobytes())
        fm.write(m_chunk.tobytes())

def run_pokemon_sharding(is_tune_mode):
    group_name = "pokemon_instruct" if is_tune_mode else "pokemon_base"
    
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
    th_o_id = get_token_id(c.THOUGHT_OPEN)
    th_c_id = get_token_id(c.THOUGHT_CLOSE)

    input_dir = "./datasets/pokemon/"
    input_files = glob.glob(os.path.join(input_dir, "*.jsonl"))

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

    print(f"SHARDING POKEMON DATA FOR {group_name.upper()}...")
    for fpath in input_files:
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                
                # Content Validation to prevent empty user/assistant rows
                prompt = data.get("prompt", "")
                resp = data.get("response", "")
                
                has_prompt = prompt and str(prompt).strip().lower() != "none"
                has_resp = resp and str(resp).strip().lower() != "none"
                
                if not (has_prompt and has_resp):
                    continue

                tks, msk = [], []

                # 1. Context Handling (Deterministic Stats Masking)
                ctx = data.get("context", "")
                if ctx:
                    ctx_enc = tokenizer.encode(ctx, add_special_tokens=False)
                    ctx_tks = list(ctx_enc.ids)
                    ctx_msk = [1] * len(ctx_tks)
                    ctx_offsets = ctx_enc.offsets

                    if is_tune_mode and all(k in data for k in ["mask_pre", "mask_target"]):
                        idx_pre = ctx.find(data["mask_pre"])
                        # Secure search for target after the prefix
                        if idx_pre != -1:
                            idx_target = ctx.find(data["mask_target"], idx_pre + len(data["mask_pre"]))
                            if idx_target != -1:
                                t_start, t_end = idx_pre, idx_target + len(data["mask_target"])
                                # Guarded loop to match token offsets to token list
                                for i in range(len(ctx_tks)):
                                    if i < len(ctx_offsets):
                                        s, e = ctx_offsets[i]
                                        if not (e <= t_start or s >= t_end):
                                            ctx_msk[i] = 0
                    
                    tks = [ctx_o_id] + ctx_tks + [ctx_c_id]
                    msk = [1] + ctx_msk + [1]

                # 2. User Prompt
                p_tks = tokenizer.encode(prompt, add_special_tokens=False).ids
                tks += [user_id] + list(p_tks)
                msk += [0] * (len(p_tks) + 1) if is_tune_mode else [1] * (len(p_tks) + 1)

                # 3. Reasoning / Thought Injection
                thought = data.get("thought") or data.get("thinking")
                if c.INCLUDE_THINKING and thought:
                    th_tks = list(tokenizer.encode(thought, add_special_tokens=False).ids)
                    tks += [th_o_id] + th_tks + [th_c_id]
                    msk += [1] * (len(th_tks) + 2) if is_tune_mode else [1] * (len(th_tks) + 2)

                # 4. Final Assistant Response
                r_tks = list(tokenizer.encode(resp, add_special_tokens=False).ids)
                tks += [asst_id] + r_tks
                msk += [1] * (len(r_tks) + 1)

                # Final Global Random Masking for Base Mode
                if not is_tune_mode:
                    msk = apply_random_base_mask(msk, c.BASE_MASK_CHANCE)
                
                process_sequence(tks, msk)

    print(f"COMPLETED {group_name.upper()} POKEMON DUMP.")

if __name__ == "__main__":
    for f in ["pokemon_base_data.tmp", "pokemon_base_mask.tmp", "pokemon_instruct_data.tmp", "pokemon_instruct_mask.tmp"]:
        target = f"./temp/{f}"
        if os.path.exists(target): os.remove(target)

    run_pokemon_sharding(is_tune_mode=False)
    run_pokemon_sharding(is_tune_mode=True)
