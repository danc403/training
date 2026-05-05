import json
import os
import numpy as np
import random
import glob
from tokenizers import Tokenizer

# --- Configuration ---
CONTEXT_SIZE = 2048
SHARD_SIZE = 10000000          
TOKEN_TYPE = np.uint16
MASK_TYPE = np.uint8
TOKEN_PATH = "./tokenizer/tokenizer.json"
EOS_TOKEN_STR = "<|end_of_text|>"
PAD_TOKEN_ID = 3 
BASE_MASK_CHANCE = 0.15
MAX_OPEN_BUCKETS = 100 # Increased for diverse tool snippets

# --- INJECTION TICKET ---
INCLUDE_THINKING = True 

# --- Paths ---
TOOL_DATA_DIR = "./datasets/tools/"
# We target all .jsonl files in the tool directory
TOOL_FILES = glob.glob(os.path.join(TOOL_DATA_DIR, "*.jsonl"))

# --- MARKERS ---
USER_OPEN = "<|user|>"
ASSISTANT_OPEN = "<|assistant|>"
THOUGHT_OPEN = "<think>"
THOUGHT_CLOSE = "</think>"
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESP_OPEN = "<tool_response>"
TOOL_RESP_CLOSE = "</tool_response>"

def apply_random_base_mask(tokens, chance):
    return [0 if random.random() < chance else 1 for _ in range(len(tokens))]

def run_tool_sharding(is_tune_mode):
    output_dir = "instruct" if is_tune_mode else "base"
    prefix = "tools"
    
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
    
    tokenizer = Tokenizer.from_file(TOKEN_PATH)
    
    def get_token_id(s):
        tid = tokenizer.token_to_id(s)
        return tid if tid is not None else tokenizer.encode(s, add_special_tokens=False).ids[0]

    eos_id = get_token_id(EOS_TOKEN_STR)
    user_id = get_token_id(USER_OPEN)
    asst_id = get_token_id(ASSISTANT_OPEN)
    th_o_id = get_token_id(THOUGHT_OPEN)
    th_c_id = get_token_id(THOUGHT_CLOSE)
    tc_o_id = get_token_id(TOOL_CALL_OPEN)
    tc_c_id = get_token_id(TOOL_CALL_CLOSE)
    tr_o_id = get_token_id(TOOL_RESP_OPEN)
    tr_c_id = get_token_id(TOOL_RESP_CLOSE)

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
            # Tool calls shouldn't really exceed context, but safety first
            pos = 0
            while pos + CONTEXT_SIZE <= len(t_seq):
                write_window(t_seq[pos:pos+CONTEXT_SIZE], m_seq[pos:pos+CONTEXT_SIZE])
                pos += (CONTEXT_SIZE - 256)
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

    print(f"Sharding {prefix.upper()} - Mode: {output_dir.upper()}")

    for file_path in TOOL_FILES:
        print(f" Processing: {os.path.basename(file_path)}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                
                # Assemble the Sequence
                tks, msk = [], []
                
                # 1. User Prompt
                u_tks = tokenizer.encode(data["prompt"], add_special_tokens=False).ids
                tks += [user_id] + u_tks
                msk += [0] * (len(u_tks) + 1) # Mask User Tag + Prompt

                # 2. Thinking Block
                thought = data.get("thought") or data.get("thinking")
                if INCLUDE_THINKING and thought:
                    th_tks = tokenizer.encode(thought, add_special_tokens=False).ids
                    tks += [th_o_id] + th_tks + [th_c_id]
                    # In Instruct mode, we learn the thinking process
                    msk += [1] + ([1] * len(th_tks)) + [1]
                
                # 3. Tool Call and Response (The Trajectory)
                call = data.get("call") or data.get("tool_call")
                if call:
                    c_tks = tokenizer.encode(call, add_special_tokens=False).ids
                    tks += [tc_o_id] + c_tks + [tc_c_id]
                    msk += [1] + ([1] * len(c_tks)) + [1] # Learn the call
                    
                    resp = data.get("response") or data.get("tool_response")
                    if resp:
                        r_tks = tokenizer.encode(resp, add_special_tokens=False).ids
                        tks += [tr_o_id] + r_tks + [tr_c_id]
                        # Mask the response from the server (Model didn't generate it)
                        msk += [1] + ([0] * len(r_tks)) + [1]

                # 4. Final Answer (Assistant)
                final = data.get("answer") or data.get("final_response")
                if final:
                    f_tks = tokenizer.encode(final, add_special_tokens=False).ids
                    tks += [asst_id] + f_tks + [eos_id]
                    msk += [1] + ([1] * len(f_tks)) + [1]

                # Base Mode Override
                if not is_tune_mode:
                    msk = apply_random_base_mask(tks, BASE_MASK_CHANCE)
                    msk[-1] = 1 # Keep EOS
                
                pack_sequence(tks, msk)

    # Final Flush
    for b in active_buckets:
        if len(b['t']) > 0:
            pad_len = CONTEXT_SIZE - len(b['t'])
            write_window(b['t'] + [PAD_TOKEN_ID]*pad_len, b['m'] + [0]*pad_len)

    f_d.close(); f_m.close()

if __name__ == "__main__":
    run_tool_sharding(is_tune_mode=False) 
    run_tool_sharding(is_tune_mode=True)
