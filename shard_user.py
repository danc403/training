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
REPORT_FILE = "report.log"

# --- BUCKET SYSTEM CONFIG ---
MAX_OPEN_BUCKETS = 50
FLUSH_THRESHOLD = 0.90

# --- MASKING CONFIGURATION ---
BASE_MASK_CHANCE = 0.15
POEM_INSTRUCT_PROB = 0.20 # Probability of inclusion in Instruct mode
MASK_DETERMINISTIC_RESPONSE = False

# --- INJECTION VARIABLES ---
INCLUDE_SUMMARIES = True

# --- PROMPT VARIANTS ---
POEM_PROMPTS = [
    "Recite the poem titled {title}:",
    "Tell me the poem {title}:",
    "Provide the text for the poem {title}:"
]

BOOK_PROMPTS = [
    "How does the book {title} begin?",
    "Start a book titled {title}:",
    "Give me the opening lines of {title}:"
]

SUMMARY_PROMPTS = [
    "Summarize the work titled {title}:",
    "Provide a summary of {title}:",
    "Give me an overview of {title} by {author}:",
    "What is {title} about?",
    "Summarize {title} by {author}:"
]

# --- CORE LOGIC FUNCTIONS ---

def get_poem_prompt(title):
    return random.choice(POEM_PROMPTS).format(title=title)

def get_book_prompt(title):
    return random.choice(BOOK_PROMPTS).format(title=title)

# Generate a random summary prompt from predefined templates
def get_summary_prompt(title, author):
    if author and author.strip():
        valid_templates = SUMMARY_PROMPTS
    else:
        valid_templates = [p for p in SUMMARY_PROMPTS if "{author}" not in p]
    template = random.choice(valid_templates)
    return template.format(title=title, author=author if author else "")

# Determine the processing capabilities of a data row based on existing keys
def get_row_capabilities(data):
    caps = []
    if "prompt" in data and "response" in data:
        caps.append("INSTRUCT")
        if "context" in data: caps.append("HAS_CONTEXT")
    
    if all(k in data for k in ["mask_pre", "mask_target", "mask_post", "context"]):
        caps.append("DETERMINISTIC")
        
    if "text" in data and data["text"] is not None:
        if isinstance(data["text"], str) and len(data["text"].strip()) > 0:
            caps.append("RAW_TEXT")
            
    if data.get("title") and str(data["title"]).strip():
        caps.append("HAS_TITLE") 
        if data.get("author") and str(data["author"]).strip(): 
            caps.append("HAS_AUTHOR")
            
    if data.get("summary") and str(data["summary"]).strip(): 
        caps.append("HAS_SUMMARY")
    
    # Identify poem rows
    if data.get("source_table") == "poems":
        caps.append("HAS_POEM")
        
    return caps

# Create mask based on marker IDs to hide prompt and reveal response
def get_multi_mask(tokens, marker_ids):
    mask = np.ones(len(tokens), dtype=np.uint8)
    if not marker_ids: return mask.tolist()
    n = len(marker_ids)
    mask_start = 0
    found_any = False
    for i in range(len(tokens) - n + 1):
        if tokens[i:i+n] == marker_ids:
            mask[mask_start : i + n] = 0
            mask_start = i + n
            found_any = True
    return mask.tolist() if found_any else ([1] * len(tokens))

# Apply random base mask for unsupervised training
def apply_random_base_mask(mask, chance):
    return [0 if (m == 1 and random.random() < chance) else m for m in mask]

# Find ID of a special token based on vocabulary keys
def find_special_token(tokenizer, keys, default_str):
    vocab = tokenizer.get_vocab()
    for key in keys:
        if key in vocab:
            return vocab[key]
    
    for token_str, token_id in vocab.items():
        token_lower = token_str.lower()
        if any(k.lower() in token_lower for k in keys if len(k) > 2):
            return token_id
            
    try:
        return tokenizer.encode(default_str, add_special_tokens=False).ids[0]
    except:
        return 0

# Main execution loop for dataset sharding
def run_sharding(is_tune_mode):
    output_dir = "instruct" if is_tune_mode else "base"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    tokenizer = Tokenizer.from_file(TOKEN_PATH)
    
    # Retrieve special tokens as IDs from the loaded tokenizer
    eos_token_id = find_special_token(tokenizer, ["<|end_of_text|>", "</s>", "<|im_end|>", "<|endoftext|>"], "<|end_of_text|>")
    pad_token_id = find_special_token(tokenizer, ["<|pad|>", "<pad>", "<|padding|>"], "<|pad|>")
    user_id = find_special_token(tokenizer, ["<|user|>", "<|im_start|>", "[INST]"], "<|user|>")
    asst_id = find_special_token(tokenizer, ["<|assistant|>", "<|im_start|>", "[/INST]"], "<|assistant|>")
    ctx_o_id = find_special_token(tokenizer, ["<|context_start|>", "<context>"], "<|context_start|>")
    ctx_c_id = find_special_token(tokenizer, ["<|context_end|>", "</context>"], "<|context_end|>")
    
    try:
        newline_id = tokenizer.encode("\n", add_special_tokens=False).ids[0]
    except:
        newline_id = find_special_token(tokenizer, ["\n"], "\n")
        
    assistant_marker_ids = [asst_id]
    all_dataset_files = sorted(glob.glob("./datasets/user_data/*.jsonl"))

    # Metadata tracking for balanced dataset consumption
    file_metadata = {}
    print(f"\nPRE-SCANNING DATASETS FOR {output_dir.upper()}...")
    
    for fpath in all_dataset_files:
        fpath_clean = fpath.replace("\\", "/")
        row_count, total_chars = 0, 0
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    row_count += 1
                    total_chars += len(line)
        
        if row_count > 0:
            est_tokens = (total_chars // 4)
            est_chunks = max(1, est_tokens // CONTEXT_SIZE)
            file_metadata[fpath_clean] = {
                "rows": row_count, "est_chunks": est_chunks,
                "handle": open(fpath, 'r', encoding='utf-8'),
                "weight": 1, "accumulator": 0.0
            }

    if not file_metadata: return

    total_weighted_chunks = sum(m["est_chunks"] * m["weight"] for m in file_metadata.values())
    for fpath in file_metadata:
        file_metadata[fpath]["step_value"] = (file_metadata[fpath]["est_chunks"] * file_metadata[fpath]["weight"]) / total_weighted_chunks

    stats = {"total_raw_docs": 0, "kept_docs": 0, "total_tokens": 0, "file_breakdown": {f: {"raw": 0, "kept": 0} for f in file_metadata}}

    shard_idx, tokens_in_shard = 0, 0
    f_d = open(os.path.join(output_dir, f"shard_{shard_idx}_data.bin"), "wb")
    f_m = open(os.path.join(output_dir, f"shard_{shard_idx}_mask.bin"), "wb")

    # Flush current window to binary shards
    def write_window(t_list, m_list):
        nonlocal shard_idx, tokens_in_shard, f_d, f_m
        
        if len(t_list) > CONTEXT_SIZE:
            t_list = t_list[:CONTEXT_SIZE]
            m_list = m_list[:CONTEXT_SIZE]
        
        pad_len = CONTEXT_SIZE - len(t_list)
        t_final = t_list + [pad_token_id] * pad_len
        m_final = m_list + [0] * pad_len
        
        if tokens_in_shard >= SHARD_SIZE:
            f_d.close(); f_m.close(); shard_idx += 1
            f_d = open(os.path.join(output_dir, f"shard_{shard_idx}_data.bin"), "wb")
            f_m = open(os.path.join(output_dir, f"shard_{shard_idx}_mask.bin"), "wb")
            tokens_in_shard = 0
            
        np.array(t_final, dtype=TOKEN_TYPE).tofile(f_d)
        np.array(m_final, dtype=MASK_TYPE).tofile(f_m)
        tokens_in_shard += CONTEXT_SIZE; stats["total_tokens"] += CONTEXT_SIZE

    active_buckets = [] 

    # Handle sequence packing and windowing
    def pack_sequence(t_seq, m_seq):
        if not t_seq: return
        
        if len(t_seq) > CONTEXT_SIZE:
            pos = 0
            while pos + CONTEXT_SIZE <= len(t_seq):
                write_window(t_seq[pos:pos+CONTEXT_SIZE], m_seq[pos:pos+CONTEXT_SIZE])
                pos += (CONTEXT_SIZE - STRIDE)
            t_seq, m_seq = t_seq[pos:], m_seq[pos:]
            if not t_seq: return

        for b in active_buckets:
            needs_sep = 1 if (len(b['t']) > 0 and b['t'][-1] != eos_token_id) else 0
            if len(b['t']) + needs_sep + len(t_seq) <= CONTEXT_SIZE:
                if needs_sep:
                    b['t'].append(eos_token_id); b['m'].append(1)
                b['t'].extend(t_seq); b['m'].extend(m_seq)
                return

        if len(active_buckets) >= MAX_OPEN_BUCKETS:
            active_buckets.sort(key=lambda x: len(x['t']), reverse=True)
            b = active_buckets.pop(0)
            write_window(b['t'], b['m'])
            
        active_buckets.append({'t': list(t_seq), 'm': list(m_seq)})

    # Main processing loop
    active_files = list(file_metadata.keys())
    while active_files:
        for f in active_files: file_metadata[f]["accumulator"] += file_metadata[f]["step_value"]
        target_fpath = max(active_files, key=lambda f: file_metadata[f]["accumulator"])
        meta = file_metadata[target_fpath]
        meta["accumulator"] -= 1.0  
        line = meta["handle"].readline()
        
        if not line:
            meta["handle"].close(); active_files.remove(target_fpath); continue
        if not line.strip(): continue

        data = json.loads(line)
        stats["total_raw_docs"] += 1; stats["file_breakdown"][target_fpath]["raw"] += 1
        caps = get_row_capabilities(data)
        
        for _ in range(meta["weight"]):
            row_candidates = []
            
            # --- INSTRUCT MODE PATH ---
            if "INSTRUCT" in caps:
                prompt, resp, ctx = data["prompt"], data["response"], data.get("context", "")
                if "HAS_CONTEXT" in caps and ctx and str(ctx).strip():
                    ctx_enc = tokenizer.encode(str(ctx), add_special_tokens=False)
                    ctx_tks = list(ctx_enc.ids); ctx_msk = [1] * len(ctx_tks)
                    if "DETERMINISTIC" in caps:
                        idx_pre = str(ctx).find(data["mask_pre"])
                        idx_target = str(ctx).find(data["mask_target"], idx_pre + len(data["mask_pre"]))
                        if idx_pre != -1 and idx_target != -1:
                            t_start, t_end = idx_pre, idx_target + len(data["mask_target"])
                            for i in range(len(ctx_tks)):
                                s, e = ctx_enc.offsets[i]
                                if not (e <= t_start or s >= t_end): ctx_msk[i] = 0
                    tks, msk = [ctx_o_id] + ctx_tks + [ctx_c_id], [1] + ctx_msk + [1]
                else: tks, msk = [], []
                
                p_tks = tokenizer.encode(prompt, add_special_tokens=False).ids
                r_tks = tokenizer.encode(resp, add_special_tokens=False).ids
                inst_tks = [user_id] + p_tks + [newline_id, newline_id, asst_id] + r_tks + [eos_token_id]
                inst_msk = [1] * len(inst_tks)
                
                if is_tune_mode:
                    try:
                        asst_idx = inst_tks.index(asst_id)
                        for i in range(len(inst_msk)):
                            if i <= asst_idx: inst_msk[i] = 0
                    except ValueError: pass
                else:
                    if "DETERMINISTIC" not in caps: inst_msk = apply_random_base_mask([1] * len(inst_tks), BASE_MASK_CHANCE)
                tks.extend(inst_tks); msk.extend(inst_msk); row_candidates.append((tks, msk))

            # --- POEM PATH ---
            if "HAS_POEM" in caps:
                if not is_tune_mode:
                    # BASE mode: Include all poems as raw text
                    tks = tokenizer.encode(data["text"], add_special_tokens=False).ids
                    msk = apply_random_base_mask([1] * (len(tks) + 1), BASE_MASK_CHANCE)
                    row_candidates.append((tks + [eos_token_id], msk))
                elif is_tune_mode and random.random() < POEM_INSTRUCT_PROB:
                    # INSTRUCT mode: Conditional random sampling
                    p_str = get_poem_prompt(data.get("title"))
                    t_q = tokenizer.encode(p_str, add_special_tokens=False).ids
                    t_ans = tokenizer.encode(data["text"], add_special_tokens=False).ids
                    p_tks = [user_id] + t_q + [newline_id, newline_id, asst_id] + t_ans + [eos_token_id]
                    row_candidates.append((p_tks, get_multi_mask(p_tks, assistant_marker_ids)))

            # --- SUMMARY/TEXT PATH ---
            if "RAW_TEXT" not in caps and is_tune_mode and INCLUDE_SUMMARIES and "HAS_SUMMARY" in caps and "HAS_TITLE" in caps:
                p_str = get_summary_prompt(data.get("title"), data.get("author", ""))
                t_q = tokenizer.encode(p_str, add_special_tokens=False).ids
                t_ans = tokenizer.encode(data["summary"], add_special_tokens=False).ids
                s_tks = [user_id] + t_q + [newline_id, newline_id, asst_id] + t_ans + [eos_token_id]
                row_candidates.append((s_tks, get_multi_mask(s_tks, assistant_marker_ids)))

            if "RAW_TEXT" in caps and "HAS_POEM" not in caps:
                tks = tokenizer.encode(data["text"], add_special_tokens=False).ids
                if is_tune_mode:
                    if INCLUDE_SUMMARIES and "HAS_SUMMARY" in caps and "HAS_TITLE" in caps:
                        p_str = get_summary_prompt(data.get("title"), data.get("author", ""))
                        t_q = tokenizer.encode(p_str, add_special_tokens=False).ids
                        t_ans = tokenizer.encode(data["summary"], add_special_tokens=False).ids
                        s_tks = [user_id] + t_q + [newline_id, newline_id, asst_id] + t_ans + [eos_token_id]
                        row_candidates.append((s_tks, get_multi_mask(s_tks, assistant_marker_ids)))
                else:
                    msk = apply_random_base_mask([1]*(len(tks)+1), BASE_MASK_CHANCE)
                    row_candidates.append((tks + [eos_token_id], msk))

            for c_t, c_m in row_candidates:
                stats["kept_docs"] += 1; stats["file_breakdown"][target_fpath]["kept"] += 1
                pack_sequence(list(c_t), list(c_m))

    for b in active_buckets:
        if len(b['t']) > 0:
            write_window(b['t'], b['m'])

    f_d.close(); f_m.close()
    full_report = f"=== SHARDING REPORT: {output_dir.upper()} ===\nTotal Tokens: {stats['total_tokens']}\n"
    for f, s in stats["file_breakdown"].items():
        if s['kept'] > 0: full_report += f" - {f}: {s['kept']} kept / {s['raw']} raw\n"
    print(full_report)
    with open(REPORT_FILE, "a") as rf: rf.write(full_report + "\n")

if __name__ == "__main__":
    if os.path.exists(REPORT_FILE): os.remove(REPORT_FILE)
    run_sharding(is_tune_mode=False)
    run_sharding(is_tune_mode=True)
