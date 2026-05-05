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
    Standardized sink for tool-use packets.
    """
    os.makedirs("./temp", exist_ok=True)
    d_path = f"./temp/{group}_data.tmp"
    m_path = f"./temp/{group}_mask.tmp"
    
    # Explicitly ensure masks are uint8 and clipped to 0-1 range to prevent OOB
    d_chunk = np.array(list(tokens) + [sep_id], dtype=c.TOKEN_TYPE)
    m_chunk = np.array([min(1, max(0, int(m))) for m in masks] + [0], dtype=c.MASK_TYPE)
    
    with open(d_path, "ab") as fd, open(m_path, "ab") as fm:
        fd.write(d_chunk.tobytes())
        fm.write(m_chunk.tobytes())

def run_tool_sharding(is_tune_mode):
    group_name = "tools_instruct" if is_tune_mode else "tools_base"
    
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
    tc_o_id = get_token_id(c.TOOL_CALL_OPEN)
    tc_c_id = get_token_id(c.TOOL_CALL_CLOSE)
    tr_o_id = get_token_id(c.TOOL_RESP_OPEN)
    tr_c_id = get_token_id(c.TOOL_RESP_CLOSE)

    tool_data_dir = "./datasets/tools/"
    tool_files = glob.glob(os.path.join(tool_data_dir, "*.jsonl"))

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

    print(f"SHARDING TOOL DATA FOR {group_name.upper()}...")
    for file_path in tool_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                
                # Content Validation: Ensure prompt and final answer are not empty or None
                prompt = data.get("prompt", "")
                final = data.get("answer") or data.get("final_response", "")
                
                has_prompt = prompt and str(prompt).strip().lower() != "none"
                has_final = final and str(final).strip().lower() != "none"
                
                if not (has_prompt and has_final):
                    continue
                
                tks, msk = [], []
                
                # 1. User Block
                u_tks = list(tokenizer.encode(prompt, add_special_tokens=False).ids)
                tks += [user_id] + u_tks
                msk += [0] * (len(u_tks) + 1) 

                # 2. Reasoning Block
                thought = data.get("thought") or data.get("thinking")
                if c.INCLUDE_THINKING and thought:
                    th_tks = list(tokenizer.encode(thought, add_special_tokens=False).ids)
                    tks += [th_o_id] + th_tks + [th_c_id]
                    msk += [1] * (len(th_tks) + 2)
                
                # 3. Call and Environmental Feedback
                call = data.get("call") or data.get("tool_call")
                if call:
                    c_tks = list(tokenizer.encode(call, add_special_tokens=False).ids)
                    tks += [tc_o_id] + c_tks + [tc_c_id]
                    msk += [1] * (len(c_tks) + 2) 
                    
                    resp = data.get("response") or data.get("tool_response")
                    if resp:
                        r_tks = list(tokenizer.encode(resp, add_special_tokens=False).ids)
                        tks += [tr_o_id] + r_tks + [tr_c_id]
                        # Mask environmental response from training loss
                        msk += [1] + ([0] * len(r_tks)) + [1]

                # 4. Final Answer
                f_tks = list(tokenizer.encode(final, add_special_tokens=False).ids)
                tks += [asst_id] + f_tks
                msk += [1] * (len(f_tks) + 1)

                if not is_tune_mode:
                    # In base mode, we re-mask the whole sequence
                    msk = apply_random_base_mask(msk, c.BASE_MASK_CHANCE)
                
                process_sequence(tks, msk)

    print(f"COMPLETED {group_name.upper()} TOOL DUMP.")

if __name__ == "__main__":
    for f in ["tools_base_data.tmp", "tools_base_mask.tmp", "tools_instruct_data.tmp", "tools_instruct_mask.tmp"]:
        target = f"./temp/{f}"
        if os.path.exists(target): os.remove(target)

    run_tool_sharding(is_tune_mode=False) 
    run_tool_sharding(is_tune_mode=True)
