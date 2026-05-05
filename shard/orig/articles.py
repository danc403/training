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
REPORT_FILE = "articles_report.log"
EOS_TOKEN_STR = "<|end_of_text|>"

# --- BUCKET SYSTEM CONFIG ---
MAX_OPEN_BUCKETS = 50  
PAD_TOKEN_ID = 3 

# --- MASKING CONFIGURATION ---
BASE_MASK_CHANCE = 0.15

# --- FILTER THRESHOLDS ---
TEXT_MIN_TOKENS = 375
TEXT_MAX_TOKENS = 400

# --- INJECTION VARIABLES ---
INCLUDE_SUMMARIES = True     

# --- PROMPT VARIANTS ---
SUMMARY_PROMPTS = [
    "Summarize the work titled {title}:",
    "Provide a summary of {title}:",
    "Give me an overview of {title} by {author}:",
    "What is {title} about?",
    "Summarize {title} by {author}:"
]

# --- TOKENIZER MARKERS ---
USER_OPEN = "<|user|>"
ASSISTANT_OPEN = "<|assistant|>"

def get_summary_prompt(title, author):
    if author and author.strip():
        valid_templates = SUMMARY_PROMPTS
    else:
        valid_templates = [p for p in SUMMARY_PROMPTS if "{author}" not in p]
    template = random.choice(valid_templates)
    return template.format(title=title, author=author if author else "")

def apply_random_base_mask(mask, chance):
    return [0 if (m == 1 and random.random() < chance) else m for m in mask]

def run_article_sharding(is_tune_mode):
    output_dir = "instruct" if is_tune_mode else "base"
    prefix = "articles"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    tokenizer = Tokenizer.from_file(TOKEN_PATH)
    def get_token_id(s):
        tid = tokenizer.token_to_id(s)
        return tid if tid is not None else tokenizer.encode(s, add_special_tokens=False).ids[0]

    eos_id = get_token_id(EOS_TOKEN_STR)
    user_id = get_token_id(USER_OPEN)
    asst_id = get_token_id(ASSISTANT_OPEN)
    newline_id = get_token_id("\n")
    
    # Target specific article path
    article_files = glob.glob("./datasets/articles/articles.jsonl", recursive=True)
    if not article_files:
        print("No article files found in ./datasets/articles/")
        return

    file_metadata = {}
    for fpath in article_files:
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
        target_fpath = active_files[0] 
        meta = file_metadata[target_fpath]
        
        line = meta["handle"].readline()
        if not line:
            meta["handle"].close()
            active_files.remove(target_fpath)
            continue
            
        data = json.loads(line)
        text = data.get("text", "")
        title = data.get("title", "Untitled")
        author = data.get("author", "")
        summary = data.get("summary", "")
        
        t_enc = tokenizer.encode(text, add_special_tokens=False)
        tks_raw = list(t_enc.ids)

        # Filter strictly for the 375-400 token sweet spot
        if TEXT_MIN_TOKENS <= len(tks_raw) <= TEXT_MAX_TOKENS:
            if is_tune_mode:
                # 1. Direct Text Injection (Learning the content)
                msk_base = apply_random_base_mask([1] * len(tks_raw), BASE_MASK_CHANCE)
                pack_sequence(tks_raw, msk_base)
                
                # 2. Summary Injection (Learning the condensation)
                if INCLUDE_SUMMARIES and summary:
                    p_str = get_summary_prompt(title, author)
                    t_q = tokenizer.encode(p_str, add_special_tokens=False).ids
                    t_ans = tokenizer.encode(summary, add_special_tokens=False).ids
                    
                    s_tks = [user_id] + t_q + [newline_id, newline_id, asst_id] + t_ans
                    # Mask prompt and assistant tags
                    s_msk = [0] * (len(t_q) + 5) + [1] * len(t_ans)
                    pack_sequence(s_tks, s_msk)
            else:
                # Base mode: Pure text streaming
                msk = apply_random_base_mask([1] * len(tks_raw), BASE_MASK_CHANCE)
                pack_sequence(tks_raw, msk)

    for b in active_buckets:
        if len(b['t']) > 0:
            pad_len = CONTEXT_SIZE - len(b['t'])
            write_window(b['t'] + [PAD_TOKEN_ID]*pad_len, b['m'] + [0]*pad_len)
    
    f_d.close(); f_m.close()
    print(f"COMPLETED {output_dir.upper()} ARTICLE SHARDING.")

if __name__ == "__main__":
    run_article_sharding(is_tune_mode=False)
    run_article_sharding(is_tune_mode=True)
