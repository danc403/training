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
EOS_TOKEN_STR = "<|end_of_text|>"
WIKI_ENTRIES_PER_CLUSTER = 3

# --- BUCKET SYSTEM CONFIG ---
MAX_OPEN_BUCKETS = 50  
FLUSH_THRESHOLD = 0.90  
PAD_TOKEN_ID = 3 

# --- MASKING CONFIGURATION ---
BASE_MASK_CHANCE = 0.15
MASK_DETERMINISTIC_RESPONSE = False 

# --- FILTER THRESHOLDS ---
KNOWLEDGE_MIN_TOKENS = 25    
WIKI_MAX_TOKENS = 1000        
FACTBOOK_MIN_TOKENS = 5      
FACTBOOK_MAX_TOKENS = 2000   
TEXT_MIN_TOKENS = 375
TEXT_MAX_TOKENS = 400
POEM_MIN_TOKENS = 75          
POEM_MAX_TOKENS = 650  
BOOK_MIN_TOKENS = 2500

# --- DATASET WEIGHTING & OVERRIDES ---
PATH_WEIGHT_OVERRIDES = {
    "favorites.jsonl": 2,
    "solar.jsonl": 2,
    "pokemon": 1
}

# --- INJECTION VARIABLES ---
INCLUDE_SUMMARIES = True     
INCLUDE_INTRODUCTIONS = True 
BOOK_SNIPPETS_PER_ENTRY = 1  
POEM_INJECTION_PROBABILITY = 0.01  

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

def get_summary_prompt(title, author):
    if author and author.strip():
        valid_templates = SUMMARY_PROMPTS
    else:
        valid_templates = [p for p in SUMMARY_PROMPTS if "{author}" not in p]
    template = random.choice(valid_templates)
    return template.format(title=title, author=author if author else "")

def get_poem_prompt(title):
    return random.choice(POEM_PROMPTS).format(title=title)

def get_book_prompt(title):
    return random.choice(BOOK_PROMPTS).format(title=title)

def get_row_capabilities(data):
    caps = []
    if "prompt" in data and "response" in data:
        caps.append("INSTRUCT")
        if "context" in data: caps.append("HAS_CONTEXT")
    
    if all(k in data for k in ["mask_pre", "mask_target", "mask_post", "context"]):
        caps.append("DETERMINISTIC")
        
    if "text" in data:
        caps.append("RAW_TEXT")
        if data.get("title"):
            caps.append("HAS_TITLE") 
            if data.get("author"): caps.append("HAS_AUTHOR")
        if data.get("summary"): caps.append("HAS_SUMMARY")
    return caps

def validate_filters(full_path, t_count):
    path_lower = full_path.replace("\\", "/").lower()
    
    if "datasets/articles/" in path_lower:
        return TEXT_MIN_TOKENS <= t_count <= TEXT_MAX_TOKENS
    
    if "datasets/factbook/" in path_lower:
        return FACTBOOK_MIN_TOKENS <= t_count <= FACTBOOK_MAX_TOKENS
    
    if "poems.jsonl" in path_lower:
        return POEM_MIN_TOKENS <= t_count <= POEM_MAX_TOKENS
        
    if "books.jsonl" in path_lower:
        return t_count >= BOOK_MIN_TOKENS

    if "wiki.jsonl" in path_lower:
        return KNOWLEDGE_MIN_TOKENS <= t_count <= WIKI_MAX_TOKENS

    return t_count > 0

# --- TOKENIZER MARKERS ---
USER_OPEN = "<|user|>"
ASSISTANT_OPEN = "<|assistant|>"
CONTEXT_OPEN = "<|context_start|>"
CONTEXT_CLOSE = "<|context_end|>"
EOS_TOKEN_STR = "<|end_of_text|>"

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

def apply_random_base_mask(mask, chance):
    return [0 if (m == 1 and random.random() < chance) else m for m in mask]

def run_sharding(is_tune_mode):
    output_dir = "instruct" if is_tune_mode else "base"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    tokenizer = Tokenizer.from_file(TOKEN_PATH)
    
    def get_token_id(s):
        tid = tokenizer.token_to_id(s)
        if tid is None: tid = tokenizer.encode(s, add_special_tokens=False).ids[0]
        return tid

    eos_token_id = get_token_id(EOS_TOKEN_STR)
    user_id = get_token_id(USER_OPEN)
    asst_id = get_token_id(ASSISTANT_OPEN)
    ctx_o_id = get_token_id(CONTEXT_OPEN)
    ctx_c_id = get_token_id(CONTEXT_CLOSE)
    newline_id = get_token_id("\n")
    assistant_marker_ids = [asst_id]
    
    all_dataset_files = glob.glob("./datasets/**/*.jsonl", recursive=True)

    # --- MATH-BASED PRE-SCAN ---
    file_metadata = {}
    print(f"\nPRE-SCANNING DATASETS FOR {output_dir.upper()}...")
    
    for fpath in all_dataset_files:
        fpath_clean = fpath.replace("\\", "/")
        if is_tune_mode:
            if not any(x in fpath_clean for x in ["instruct", "factbook", "solar", "text", "user_data"]):
                continue
        
        row_count = 0
        total_chars = 0
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                row_count += 1
                total_chars += len(line)
        
        if row_count > 0:
            # Deterministic chunk estimation:
            # (Total Chars / 4) to estimate tokens, then divide by Context Size to get chunk volume.
            est_tokens = (total_chars // 4)
            est_chunks = max(1, est_tokens // CONTEXT_SIZE)
            
            file_metadata[fpath_clean] = {
                "rows": row_count,
                "est_chunks": est_chunks,
                "handle": open(fpath, 'r', encoding='utf-8'),
                "weight": 1,
                "accumulator": 0.0
            }
            for key, w in PATH_WEIGHT_OVERRIDES.items():
                if key in fpath_clean:
                    file_metadata[fpath_clean]["weight"] = w

    # Calculate global ratio for deterministic round-robin
    total_weighted_chunks = sum(m["est_chunks"] * m["weight"] for m in file_metadata.values())
    for fpath in file_metadata:
        meta = file_metadata[fpath]
        meta["step_value"] = (meta["est_chunks"] * meta["weight"]) / total_weighted_chunks

    stats = {"total_raw_docs": 0, "kept_docs": 0, "total_tokens": 0, "file_breakdown": {}}
    for fpath in file_metadata:
        stats["file_breakdown"][fpath] = {"raw": 0, "kept": 0}

    # --- OUTPUT HANDLERS ---
    shard_idx, tokens_in_shard = 0, 0
    f_d = open(os.path.join(output_dir, f"shard_{shard_idx}_data.bin"), "wb")
    f_m = open(os.path.join(output_dir, f"shard_{shard_idx}_mask.bin"), "wb")

    def write_window(t_list, m_list):
        nonlocal shard_idx, tokens_in_shard, f_d, f_m
        if tokens_in_shard >= SHARD_SIZE:
            f_d.close(); f_m.close(); shard_idx += 1
            f_d = open(os.path.join(output_dir, f"shard_{shard_idx}_data.bin"), "wb")
            f_m = open(os.path.join(output_dir, f"shard_{shard_idx}_mask.bin"), "wb")
            tokens_in_shard = 0
        np.array(t_list, dtype=TOKEN_TYPE).tofile(f_d)
        np.array(m_list, dtype=MASK_TYPE).tofile(f_m)
        tokens_in_shard += CONTEXT_SIZE; stats["total_tokens"] += CONTEXT_SIZE

    active_buckets = [] 

    def pack_sequence(t_seq, m_seq):
        if not t_seq: return
        if t_seq[-1] != eos_token_id:
            t_seq.append(eos_token_id); m_seq.append(1)

        # 1. Immediate sharding for oversized items (Sliding Window)
        if len(t_seq) > CONTEXT_SIZE:
            pos = 0
            while pos + CONTEXT_SIZE <= len(t_seq):
                write_window(t_seq[pos:pos+CONTEXT_SIZE], m_seq[pos:pos+CONTEXT_SIZE])
                pos += (CONTEXT_SIZE - STRIDE)
            if pos < len(t_seq):
                t_seq, m_seq = t_seq[pos:], m_seq[pos:]
            else:
                return

        # 2. Check existing buckets for a fit
        for b in active_buckets:
            if len(b['t']) + len(t_seq) <= CONTEXT_SIZE:
                if len(b['t']) > 0: m_seq[0] = 0
                b['t'].extend(t_seq); b['m'].extend(m_seq)
                return

        # 3. New bucket if capacity exists
        if len(active_buckets) < MAX_OPEN_BUCKETS:
            active_buckets.append({'t': list(t_seq), 'm': list(m_seq)})
            return

        # 4. Mandatory Flush
        fullest = max(range(len(active_buckets)), key=lambda idx: len(active_buckets[idx]['t']))
        b = active_buckets[fullest]
        pad_len = CONTEXT_SIZE - len(b['t'])
        write_window(b['t'] + [PAD_TOKEN_ID]*pad_len, b['m'] + [0]*pad_len)
        active_buckets[fullest] = {'t': list(t_seq), 'm': list(m_seq)}

    # --- DETERMINISTIC MATH PROCESSING ENGINE ---
    print(f"STARTING MATH-DRIVEN INTERLEAVED STREAMING...")
    
    active_files = list(file_metadata.keys())
    while active_files:
        # Update accumulators based on fixed math ratios
        for f in active_files:
            file_metadata[f]["accumulator"] += file_metadata[f]["step_value"]
        
        # Select the file that has crossed the unit threshold or is highest
        target_fpath = max(active_files, key=lambda f: file_metadata[f]["accumulator"])
        meta = file_metadata[target_fpath]
        meta["accumulator"] -= 1.0  # Subtract one unit of "debt"
        
        line = meta["handle"].readline()
        if not line:
            meta["handle"].close()
            active_files.remove(target_fpath)
            continue
            
        data = json.loads(line)
        stats["total_raw_docs"] += 1
        stats["file_breakdown"][target_fpath]["raw"] += 1
        
        caps = get_row_capabilities(data)
        
        # Row-level weight loops preserved for override logic
        for _ in range(meta["weight"]):
            row_candidates = []
            
            # Capability: INSTRUCT
            if "INSTRUCT" in caps:
                prompt, resp, ctx = data["prompt"], data["response"], data.get("context", "")
                
                # 1. Handle Context in ISOLATION to prevent tag bleed
                if "HAS_CONTEXT" in caps:
                    ctx_enc = tokenizer.encode(ctx, add_special_tokens=False)
                    ctx_tks = list(ctx_enc.ids)
                    ctx_msk = [1] * len(ctx_tks)
                    
                    if "DETERMINISTIC" in caps:
                        # Find mask range within the ISOLATED context string
                        idx_pre = ctx.find(data["mask_pre"])
                        idx_target = ctx.find(data["mask_target"], idx_pre + len(data["mask_pre"]))
                        
                        if idx_pre != -1 and idx_target != -1:
                            t_start = idx_pre
                            t_end = idx_target + len(data["mask_target"])
                            
                            # Map to the ISOLATED context tokens
                            for i in range(len(ctx_tks)):
                                s, e = ctx_enc.offsets[i]
                                if not (e <= t_start or s >= t_end):
                                    ctx_msk[i] = 0
                    
                    # Wrap the masked context in structural tags
                    # Structural tags are ALWAYS unmasked (1)
                    tks = [ctx_o_id] + ctx_tks + [ctx_c_id]
                    msk = [1] + ctx_msk + [1]
                else:
                    tks, msk = [], []

                # 2. Append the Instruction/Response block
                inst_str = f"{USER_OPEN}{prompt}\n\n{ASSISTANT_OPEN}{resp}{EOS_TOKEN_STR}"
                inst_enc = tokenizer.encode(inst_str, add_special_tokens=False)
                
                inst_tks = list(inst_enc.ids)
                inst_msk = [1] * len(inst_tks)

                # 3. Apply standard Instruct masking (mask everything before assistant)
                if is_tune_mode:
                    asst_marker_idx = inst_str.find(ASSISTANT_OPEN)
                    if asst_marker_idx != -1:
                        limit = asst_marker_idx + len(ASSISTANT_OPEN)
                        for i in range(len(inst_tks)):
                            s, e = inst_enc.offsets[i]
                            if s < limit:
                                inst_msk[i] = 0
                else:
                    # Apply random noise ONLY if NOT deterministic row
                    if "DETERMINISTIC" not in caps: 
                        inst_msk = apply_random_base_mask([1] * len(inst_tks), BASE_MASK_CHANCE)
                
                # Concatenate the masked context and the masked instruction
                tks.extend(inst_tks)
                msk.extend(inst_msk)
                
                # Ensure EOS is always learned
                msk[-1] = 1 
                row_candidates.append((tks, msk))

            # Capability: RAW_TEXT
            if "RAW_TEXT" in caps:
                if is_tune_mode and not any(k in caps for k in ["HAS_SUMMARY", "HAS_TITLE"]):
                    pass
                else:
                    tks = tokenizer.encode(data["text"], add_special_tokens=False).ids
                    if validate_filters(target_fpath, len(tks)):
                        title, author = data.get("title"), data.get("author", "")
                        
                        if is_tune_mode:
                            if INCLUDE_SUMMARIES and "HAS_SUMMARY" in caps and "HAS_TITLE" in caps:
                                p_str = get_summary_prompt(title, author)
                                t_q = tokenizer.encode(p_str, add_special_tokens=False).ids
                                t_ans = tokenizer.encode(data["summary"], add_special_tokens=False).ids
                                s_tks = [user_id] + t_q + [newline_id, newline_id, asst_id] + t_ans + [eos_token_id]
                                row_candidates.append((s_tks, get_multi_mask(s_tks, assistant_marker_ids)))

                            if INCLUDE_INTRODUCTIONS and "HAS_TITLE" in caps:
                                t_len = len(tks)
                                if POEM_MIN_TOKENS <= t_len <= POEM_MAX_TOKENS:
                                    p_str = get_poem_prompt(title)
                                    t_q = tokenizer.encode(p_str, add_special_tokens=False).ids
                                    r_tks = [user_id] + t_q + [newline_id, newline_id, asst_id] + tks + [eos_token_id]
                                    row_candidates.append((r_tks, get_multi_mask(r_tks, assistant_marker_ids)))
                                elif t_len >= BOOK_MIN_TOKENS:
                                    p_str = get_book_prompt(title)
                                    t_q = tokenizer.encode(p_str, add_special_tokens=False).ids
                                    r_tks = [user_id] + t_q + [newline_id, newline_id, asst_id] + tks[:2000] + [eos_token_id]
                                    row_candidates.append((r_tks, get_multi_mask(r_tks, assistant_marker_ids)))
                                    for _ in range(BOOK_SNIPPETS_PER_ENTRY):
                                        start = random.randint(0, max(0, t_len - CONTEXT_SIZE))
                                        row_candidates.append((tks[start:start+CONTEXT_SIZE], [1]*CONTEXT_SIZE))

                        if not is_tune_mode:
                            msk = apply_random_base_mask([1]*(len(tks)+1), BASE_MASK_CHANCE)
                            row_candidates.append((tks + [eos_token_id], msk))

            for c_t, c_m in row_candidates:
                stats["kept_docs"] += 1
                stats["file_breakdown"][target_fpath]["kept"] += 1
                pack_sequence(list(c_t), list(c_m))

    # Flush remaining buckets
    for b in active_buckets:
        if len(b['t']) > 0:
            pad_len = CONTEXT_SIZE - len(b['t'])
            write_window(b['t'] + [PAD_TOKEN_ID]*pad_len, b['m'] + [0]*pad_len)

    f_d.close(); f_m.close()
    
    full_report = f"=== SHARDING REPORT: {output_dir.upper()} ===\nTotal Tokens: {stats['total_tokens']}\n"
    for f, s in stats["file_breakdown"].items():
        if s['kept'] > 0:
            full_report += f" - {f}: {s['kept']} kept / {s['raw']} raw\n"
    print(full_report)
    with open(REPORT_FILE, "a") as rf: rf.write(full_report + "\n")

if __name__ == "__main__":
    if os.path.exists(REPORT_FILE): os.remove(REPORT_FILE)
    run_sharding(is_tune_mode=False)
    run_sharding(is_tune_mode=True)
