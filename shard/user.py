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

def save_intermediate_packet(group, tokens, masks, sep_id):
    """
    Standardized sink for user packets.
    """
    os.makedirs("./temp", exist_ok=True)
    d_path = f"./temp/{group}_data.tmp"
    m_path = f"./temp/{group}_mask.tmp"
    
    d_chunk = np.array(list(tokens) + [sep_id], dtype=c.TOKEN_TYPE)
    m_chunk = np.array(list(masks) + [0], dtype=c.MASK_TYPE)
    
    with open(d_path, "ab") as fd, open(m_path, "ab") as fm:
        fd.write(d_chunk.tobytes())
        fm.write(m_chunk.tobytes())

def run_user_data_sharding(is_tune_mode=False):
    group_name = "user_instruct" if is_tune_mode else "user_base"
    
    tokenizer = Tokenizer.from_file(c.TOKEN_PATH)
    sep_id = c.get_sep_id(tokenizer)
    
    def get_token_id(s):
        tid = tokenizer.token_to_id(s)
        return tid if tid is not None else tokenizer.encode(s, add_special_tokens=False).ids[0]

    eos_id = get_token_id(c.EOS_TOKEN_STR)
    user_id = get_token_id(c.USER_OPEN)
    asst_id = get_token_id(c.ASSISTANT_OPEN)
    th_o_id = get_token_id(c.THOUGHT_OPEN)
    th_c_id = get_token_id(c.THOUGHT_CLOSE)
    ctx_o_id = get_token_id(c.CONTEXT_OPEN)
    ctx_c_id = get_token_id(c.CONTEXT_CLOSE)
    newline_id = get_token_id("\n")

    def process_sequence(t_seq, m_seq):
        if not t_seq: return
        if t_seq[-1] != eos_id:
            t_seq.append(eos_id); m_seq.append(1)

        # Handle Long Text with Striding
        if len(t_seq) > c.CONTEXT_SIZE:
            pos = 0
            while pos + c.CONTEXT_SIZE <= len(t_seq):
                save_intermediate_packet(group_name, t_seq[pos:pos+c.CONTEXT_SIZE], m_seq[pos:pos+c.CONTEXT_SIZE], sep_id)
                pos += (c.CONTEXT_SIZE - c.STRIDE)
            return
        
        # Standard Packet
        save_intermediate_packet(group_name, t_seq, m_seq, sep_id)

    user_files = glob.glob("./datasets/user_data/*.jsonl")
    
    for fpath in user_files:
        print(f"Routing User Data: {fpath}")
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                caps = get_row_capabilities(data)
                
                if "INSTRUCT" in caps:
                    # Content Validation: Skip empty/none prompt or response
                    prompt_val = data.get("prompt")
                    resp_val = data.get("response")
                    
                    has_prompt = prompt_val and str(prompt_val).strip().lower() != "none"
                    has_resp = resp_val and str(resp_val).strip().lower() != "none"
                    
                    if not (has_prompt and has_resp):
                        continue

                    tks, msk = [], []
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

                    p_ids = tokenizer.encode(f"{c.USER_OPEN}{data['prompt']}\n\n", add_special_tokens=False).ids
                    tks += p_ids
                    msk += [0] * len(p_ids)

                    tks += [asst_id]
                    msk += [0]
                    if "HAS_THOUGHT" in caps:
                        th_ids = tokenizer.encode(data["thought"], add_special_tokens=False).ids
                        tks += [th_o_id] + th_ids + [th_c_id]
                        msk += [1] * (len(th_ids) + 2)

                    r_ids = tokenizer.encode(data["response"], add_special_tokens=False).ids
                    tks += r_ids
                    msk += [1] * len(r_ids)
                    
                    if not is_tune_mode:
                        msk = apply_random_base_mask(len(tks), c.BASE_MASK_CHANCE)
                    
                    process_sequence(tks, msk)

                if "RAW_TEXT" in caps:
                    text_val = data.get("text")
                    if not text_val or str(text_val).strip().lower() == "none":
                        continue

                    full_tks = tokenizer.encode(text_val, add_special_tokens=False).ids
                    if is_tune_mode:
                        if c.INCLUDE_SUMMARIES and data.get("summary") and data.get("title"):
                            p_str = get_summary_prompt(data["title"], data.get("author", ""))
                            t_q = tokenizer.encode(p_str, add_special_tokens=False).ids
                            t_ans = tokenizer.encode(data["summary"], add_special_tokens=False).ids
                            s_tks = [user_id] + t_q + [newline_id, newline_id, asst_id] + t_ans
                            s_msk = [0] * (len(t_q) + 4) + [1] * len(t_ans)
                            process_sequence(s_tks, s_msk)
                    else:
                        base_msk = apply_random_base_mask(len(full_tks), c.BASE_MASK_CHANCE)
                        process_sequence(full_tks, base_msk)

if __name__ == "__main__":
    # Clear temp files for this specific group to prevent stale data mixing
    for f in ["user_base_data.tmp", "user_base_mask.tmp", "user_instruct_data.tmp", "user_instruct_mask.tmp"]:
        target = f"./temp/{f}"
        if os.path.exists(target): os.remove(target)
        
    run_user_data_sharding(is_tune_mode=False)
    run_user_data_sharding(is_tune_mode=True)
