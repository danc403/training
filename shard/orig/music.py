import json
import os
import numpy as np
import random
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
MAX_OPEN_BUCKETS = 50

# --- INJECTION TICKET ---
INCLUDE_THINKING = True  # Set to False to skip 'thought' or 'thinking' keys

# --- Paths ---
INPUT_FILE = "./datasets/music/music_instruct.jsonl"

# --- MARKERS ---
USER_OPEN = "<|user|>"
ASSISTANT_OPEN = "<|assistant|>"
CONTEXT_OPEN = "<|context_start|>"
CONTEXT_CLOSE = "<|context_end|>"
THOUGHT_OPEN = "<think>"
THOUGHT_CLOSE = "</think>"

def apply_random_base_mask(tokens, chance):
    return [0 if random.random() < chance else 1 for _ in range(len(tokens))]

def run_music_sharding(is_tune_mode):
    output_dir = "instruct" if is_tune_mode else "base"
    prefix = "music"
    
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
    
    tokenizer = Tokenizer.from_file(TOKEN_PATH)
    
    def get_token_id(s):
        tid = tokenizer.token_to_id(s)
        return tid if tid is not None else tokenizer.encode(s, add_special_tokens=False).ids[0]

    eos_id = get_token_id(EOS_TOKEN_STR)
    user_id = get_token_id(USER_OPEN)
    asst_id = get_token_id(ASSISTANT_OPEN)
    ctx_o_id = get_token_id(CONTEXT_OPEN)
    ctx_c_id = get_token_id(CONTEXT_CLOSE)
    th_o_id = get_token_id(THOUGHT_OPEN)
    th_c_id = get_token_id(THOUGHT_CLOSE)

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

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            
            # Tokenize Context, User, and Response
            ctx_tks = tokenizer.encode(data.get("context", ""), add_special_tokens=False).ids
            user_tks = tokenizer.encode(data["prompt"], add_special_tokens=False).ids
            resp_tks = tokenizer.encode(data["response"], add_special_tokens=False).ids

            # Assemble Tokens with Thinking Check
            tks = [ctx_o_id] + ctx_tks + [ctx_c_id]
            tks += [user_id] + user_tks
            
            # Check for thinking/thought keys
            thought_text = data.get("thought") or data.get("thinking")
            if INCLUDE_THINKING and thought_text:
                th_tks = tokenizer.encode(thought_text, add_special_tokens=False).ids
                tks += [th_o_id] + th_tks + [th_c_id]
            
            tks += [asst_id] + resp_tks + [eos_id]

            if is_tune_mode:
                # Masking: Everything is 0 EXCEPT Thought and Assistant Response
                msk = [1] + ([0] * len(ctx_tks)) + [1] # Context wrapper tags only
                msk += [0] * (len(user_tks) + 1)       # Mask User tag and Prompt
                
                if INCLUDE_THINKING and thought_text:
                    msk += [1] + ([1] * len(th_tks)) + [1] # Learn Thought process
                
                msk += [1] + ([1] * len(resp_tks)) + [1] # Learn Assistant Response + EOS
            else:
                # Base Mode: 15% random noise
                msk = apply_random_base_mask(tks, BASE_MASK_CHANCE)
            
            # Final Safety: Always learn the end
            msk[-1] = 1
            pack_sequence(tks, msk)

    for b in active_buckets:
        if len(b['t']) > 0:
            pad_len = CONTEXT_SIZE - len(b['t'])
            write_window(b['t'] + [PAD_TOKEN_ID]*pad_len, b['m'] + [0]*pad_len)
    f_d.close(); f_m.close()

if __name__ == "__main__":
    run_music_sharding(is_tune_mode=False) 
    run_music_sharding(is_tune_mode=True)
