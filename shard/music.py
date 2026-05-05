import json
import os
import numpy as np
import random
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
    Standardized sink for music packets.
    """
    os.makedirs("./temp", exist_ok=True)
    d_path = f"./temp/{group}_data.tmp"
    m_path = f"./temp/{group}_mask.tmp"
    
    d_chunk = np.array(list(tokens) + [sep_id], dtype=c.TOKEN_TYPE)
    m_chunk = np.array(list(masks) + [0], dtype=c.MASK_TYPE)
    
    with open(d_path, "ab") as fd, open(m_path, "ab") as fm:
        fd.write(d_chunk.tobytes())
        fm.write(m_chunk.tobytes())

def run_music_sharding(is_tune_mode):
    group_name = "music_instruct" if is_tune_mode else "music_base"
    
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

    input_file = "./datasets/music/music_instruct.jsonl"
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return

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

    print(f"SHARDING MUSIC DATA FOR {group_name.upper()}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            
            # Content Validation: Ensure prompt and response are not empty or None
            user_text = data.get("prompt", "")
            resp_text = data.get("response", "")
            
            has_prompt = user_text and str(user_text).strip().lower() != "none"
            has_resp = resp_text and str(resp_text).strip().lower() != "none"
            
            if not (has_prompt and has_resp):
                continue
            
            ctx_text = data.get("context", "")
            ctx_enc = tokenizer.encode(ctx_text, add_special_tokens=False)
            ctx_tks = list(ctx_enc.ids)
            
            user_tks = tokenizer.encode(user_text, add_special_tokens=False).ids
            resp_tks = tokenizer.encode(resp_text, add_special_tokens=False).ids

            # Assemble Sequence
            tks = [ctx_o_id] + ctx_tks + [ctx_c_id]
            tks += [user_id] + list(user_tks)
            
            thought_text = data.get("thought") or data.get("thinking")
            th_tks = []
            if c.INCLUDE_THINKING and thought_text:
                th_tks = list(tokenizer.encode(thought_text, add_special_tokens=False).ids)
                tks += [th_o_id] + th_tks + [th_c_id]
            
            tks += [asst_id] + list(resp_tks)

            if is_tune_mode:
                # Mask Context and User but keep tags for structure
                msk = [1] + ([0] * len(ctx_tks)) + [1] 
                msk += [0] * (len(user_tks) + 1)
                
                if c.INCLUDE_THINKING and thought_text:
                    msk += [1] + ([1] * len(th_tks)) + [1]
                
                msk += [1] + ([1] * len(resp_tks))
            else:
                msk = apply_random_base_mask([1] * len(tks), c.BASE_MASK_CHANCE)
            
            if msk: msk[-1] = 1 # Always learn EOS
            process_sequence(tks, msk)

    print(f"COMPLETED {group_name.upper()} MUSIC DUMP.")

if __name__ == "__main__":
    for f in ["music_base_data.tmp", "music_base_mask.tmp", "music_instruct_data.tmp", "music_instruct_mask.tmp"]:
        target = f"./temp/{f}"
        if os.path.exists(target): os.remove(target)

    run_music_sharding(is_tune_mode=False) 
    run_music_sharding(is_tune_mode=True)
