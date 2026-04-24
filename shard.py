import json
import os
import numpy as np
import random
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
MAX_OPEN_BUCKETS = 5
FLUSH_THRESHOLD = 0.90  # Flush if bucket is 90% full and we have too many buckets
# PAD is at index 3 in our train_tokenizer list: [<s>, </s>, <unk>, <pad>, ...]
PAD_TOKEN_ID = 3 

# --- MASKING CONFIGURATION ---
BASE_MASK_CHANCE = 0.15
# Toggle to determine if the mask_target should also be masked in the response string
MASK_DETERMINISTIC_RESPONSE = False 

# --- FILTER THRESHOLDS ---
# These only apply to the wiki.jsonl
KNOWLEDGE_MIN_TOKENS = 25    
WIKI_MAX_TOKENS = 1000        

FACTBOOK_MIN_TOKENS = 5     
FACTBOOK_MAX_TOKENS = 2000   
# These two only apply to articles.jsonl.
TEXT_MIN_TOKENS = 375
TEXT_MAX_TOKENS = 400

POEM_MIN_TOKENS = 75          
POEM_MAX_TOKENS = 650  
BOOK_MIN_TOKENS = 2500

# --- INJECTION VARIABLES ---
INCLUDE_SUMMARIES = True     # If True, processes "summary" keys into Instruct pairs
INCLUDE_INTRODUCTIONS = True # If True, processes "book/poem" content into Instruct pairs
BOOK_SNIPPETS_PER_ENTRY = 1  # this only applies to the instruct set.
POEM_INJECTION_PROBABILITY = 0.01  # % chance to include raw poem text in Instruct

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

# --- TOKENIZER MARKERS ---
# We are moving context to the dedicated context tokens to sandbox the novels.
USER_OPEN = "<|user|>"
ASSISTANT_OPEN = "<|assistant|>"
CONTEXT_OPEN = "<|context_start|>"
CONTEXT_CLOSE = "<|context_end|>"

# Preserved for future iterations
THOUGHT_OPEN = "<think>"
THOUGHT_CLOSE = "</think>"

def get_multi_mask(tokens, marker_ids):
    mask = np.ones(len(tokens), dtype=np.uint8)
    if not marker_ids:
        return mask.tolist()
    n = len(marker_ids)
    mask_start = 0
    found_any = False
    for i in range(len(tokens) - n + 1):
        if tokens[i:i+n] == marker_ids:
            # Traditional masking (for instruct): mask everything up to the assistant marker
            mask[mask_start : i + n] = 0
            mask_start = i + n
            found_any = True
    return mask.tolist() if found_any else ([1] * len(tokens))

def apply_random_base_mask(mask, chance):
    return [0 if (m == 1 and random.random() < chance) else m for m in mask]

def run_sharding(is_tune_mode):
    output_dir = "instruct" if is_tune_mode else "base"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    tokenizer = Tokenizer.from_file(TOKEN_PATH)
    
    # --- MANUAL ID ACQUISITION ---
    # This ensures markers are treated as atomic tokens, not strings to be parsed
    def get_token_id(s):
        tid = tokenizer.token_to_id(s)
        if tid is None:
            # Fallback for special characters if not in vocab explicitly
            tid = tokenizer.encode(s, add_special_tokens=False).ids[0]
        return tid

    eos_token_id = get_token_id(EOS_TOKEN_STR)
    user_id = get_token_id(USER_OPEN)
    asst_id = get_token_id(ASSISTANT_OPEN)
    ctx_o_id = get_token_id(CONTEXT_OPEN)
    ctx_c_id = get_token_id(CONTEXT_CLOSE)
    
    # Kept for reference but context takes priority in current shard logic
    thought_o_id = get_token_id(THOUGHT_OPEN)
    thought_c_id = get_token_id(THOUGHT_CLOSE)
    
    newline_id = get_token_id("\n")
    
    # We use this specifically for the masking function search
    assistant_marker_ids = [asst_id]
    
    if is_tune_mode:
        active_configs = {
            "factbook.jsonl": {"weight": 1},
            "global.jsonl": {"weight": 1},
            "solar.jsonl": {"weight": 1},
            "pokemon_instruct.jsonl": {"weight": 1},
            "text.jsonl": {"weight": 1}
        }
    else:
        active_configs = {
            "pokemon_instruct.jsonl": {"weight": 1},
            "factbook.jsonl": {"weight": 1},
            "global.jsonl": {"weight": 1},
            "solar.jsonl": {"weight": 2},
            "wiki.jsonl": {"weight": 1},
            "text.jsonl": {"weight": 1},
            "articles.jsonl": {"weight": 1}
        }

    # Pools for interleaving
    long_pool = [] 
    short_pool = []
    
    wiki_cluster_tokens, wiki_cluster_masks, wiki_count = [], [], 0
    stats = {"total_raw_docs": 0, "kept_docs": 0, "total_tokens": 0, "file_breakdown": {}}

    print(f"\nPROCESSING: {output_dir.upper()}")
    
    for fname, config in active_configs.items():
        if not os.path.exists(fname): continue
        weight = config["weight"]
        stats["file_breakdown"][fname] = {"raw": 0, "kept": 0}
        
        with open(fname, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                stats["total_raw_docs"] += 1
                stats["file_breakdown"][fname]["raw"] += 1
                row_candidates = []

                if "prompt" in data and "response" in data:
                    prompt = data["prompt"]
                    resp = data["response"]
                    ctx = data.get("context", "")
                    
                    # 1. Assemble the full sequence string for offset tracking
                    if ctx:
                        full_str = f"{CONTEXT_OPEN}{ctx}{CONTEXT_CLOSE}{USER_OPEN}{prompt}\n\n{ASSISTANT_OPEN}{resp}{EOS_TOKEN_STR}"
                    else:
                        full_str = f"{USER_OPEN}{prompt}\n\n{ASSISTANT_OPEN}{resp}{EOS_TOKEN_STR}"
                    
                    # 2. Tokenize the entire assembled string
                    encoding = tokenizer.encode(full_str, add_special_tokens=False)
                    tks = encoding.ids
                    msk = [1] * len(tks)

                    # 3. Deterministic Masking Logic (Sequential Find)
                    m_pre = data.get("mask_pre")
                    m_target = data.get("mask_target")
                    m_post = data.get("mask_post")

                    deterministic = False
                    # Use "is not None" to allow empty strings as valid anchors
                    if m_pre is not None and m_target is not None and m_post is not None:
                        idx_pre = full_str.find(m_pre)
                        if idx_pre != -1:
                            idx_target = full_str.find(m_target, idx_pre + len(m_pre))
                            if idx_target != -1:
                                idx_post = full_str.find(m_post, idx_target + len(m_target))
                                if idx_post != -1:
                                    deterministic = True
                                    # APPLY DOUBLE MASK CHANGE: Mask from start of Pre to end of Target
                                    target_start_char = idx_pre
                                    target_end_char = idx_target + len(m_target)
                                    
                                    # Map character indices to tokens
                                    for i in range(len(tks)):
                                        start, end = encoding.offsets[i]
                                        # If token overlaps target character range, mask it
                                        if not (end <= target_start_char or start >= target_end_char):
                                            msk[i] = 0

                    # 4. Apply Global Masking Rules
                    if is_tune_mode:
                        # Instruct side: mask everything up to and including the Assistant marker
                        # Note: Deterministic context masking is preserved within the 0s
                        asst_marker_start = full_str.find(ASSISTANT_OPEN)
                        if asst_marker_start != -1:
                            limit = asst_marker_start + len(ASSISTANT_OPEN)
                            for i in range(len(tks)):
                                start, end = encoding.offsets[i]
                                if start < limit:
                                    msk[i] = 0
                    else:
                        # BASE SIDE LOGIC: Skip random noise if deterministic keys were applied
                        if not deterministic:
                            msk = apply_random_base_mask(msk, BASE_MASK_CHANCE)
                    
                    msk[-1] = 1 # Always learn EOS
                    row_candidates.append((tks, msk))

                elif "text" in data:
                    if is_tune_mode and fname in ["factbook.jsonl", "global.jsonl", "pokemon_instruct.jsonl"]:
                        continue
                        
                    raw_text = data["text"]
                    tks = tokenizer.encode(raw_text, add_special_tokens=False).ids
                    t_count = len(tks)

                    # Threshold Filters
                    keep = True
                    if fname == "wiki.jsonl" and not (KNOWLEDGE_MIN_TOKENS <= t_count <= WIKI_MAX_TOKENS): keep = False
                    if fname in ["factbook.jsonl", "global.jsonl"] and not (FACTBOOK_MIN_TOKENS <= t_count <= FACTBOOK_MAX_TOKENS): keep = False
                    if fname == "articles.jsonl" and not (TEXT_MIN_TOKENS <= t_count <= TEXT_MAX_TOKENS): keep = False
                    if fname == "text.jsonl" and not (POEM_MIN_TOKENS <= t_count <= POEM_MAX_TOKENS or t_count >= BOOK_MIN_TOKENS): keep = False

                    if keep:
                        title = data.get("title")
                        author = data.get("author", "")
                        
                        if is_tune_mode and fname == "text.jsonl":
                            if data.get("summary"):
                                t_q = tokenizer.encode(random.choice(SUMMARY_PROMPTS).format(title=title, author=author), add_special_tokens=False).ids
                                t_ans = tokenizer.encode(data["summary"], add_special_tokens=False).ids
                                s_tks = [user_id] + t_q + [newline_id, newline_id, asst_id] + t_ans + [eos_token_id]
                                row_candidates.append((s_tks, get_multi_mask(s_tks, assistant_marker_ids)))
                            
                            if POEM_MIN_TOKENS <= t_count <= POEM_MAX_TOKENS:
                                t_q = tokenizer.encode(random.choice(POEM_PROMPTS).format(title=title), add_special_tokens=False).ids
                                r_tks = [user_id] + t_q + [newline_id, newline_id, asst_id] + tks + [eos_token_id]
                                row_candidates.append((r_tks, get_multi_mask(r_tks, assistant_marker_ids)))
                                
                                if random.random() < POEM_INJECTION_PROBABILITY:
                                    tks_copy = list(tks)
                                    tks_copy.append(eos_token_id)
                                    row_candidates.append((tks_copy, [1]*len(tks_copy)))

                            elif t_count >= BOOK_MIN_TOKENS:
                                t_q = tokenizer.encode(random.choice(BOOK_PROMPTS).format(title=title), add_special_tokens=False).ids
                                r_tks = [user_id] + t_q + [newline_id, newline_id, asst_id] + tks[:2000] + [eos_token_id]
                                row_candidates.append((r_tks, get_multi_mask(r_tks, assistant_marker_ids)))
                                
                                for _ in range(BOOK_SNIPPETS_PER_ENTRY):
                                    start_idx = random.randint(0, max(0, t_count - CONTEXT_SIZE))
                                    snippet_tks = tks[start_idx : start_idx + CONTEXT_SIZE]
                                    row_candidates.append((snippet_tks, [1]*len(snippet_tks)))

                        if not is_tune_mode:
                            tks_copy = list(tks)
                            tks_copy.append(eos_token_id)
                            row_candidates.append((tks_copy, apply_random_base_mask([1]*len(tks_copy), BASE_MASK_CHANCE)))

                for c_t, c_m in row_candidates:
                    if not c_t: continue
                    stats["kept_docs"] += 1
                    stats["file_breakdown"][fname]["kept"] += 1
                    
                    if c_t[-1] != eos_token_id:
                        c_t.append(eos_token_id)
                        c_m.append(1)

                    if fname == "wiki.jsonl" and not is_tune_mode:
                        wiki_cluster_tokens.extend(c_t); wiki_cluster_masks.extend(c_m)
                        wiki_count += 1
                        if wiki_count >= WIKI_ENTRIES_PER_CLUSTER:
                            for _ in range(weight): short_pool.append((list(wiki_cluster_tokens), list(wiki_cluster_masks)))
                            wiki_cluster_tokens, wiki_cluster_masks, wiki_count = [], [], 0
                    else:
                        if len(c_t) >= BOOK_MIN_TOKENS:
                            for _ in range(weight): long_pool.append((list(c_t), list(c_m)))
                        else:
                            for _ in range(weight): short_pool.append((list(c_t), list(c_m)))

    if wiki_cluster_tokens:
        short_pool.append((list(wiki_cluster_tokens), list(wiki_cluster_masks)))

    random.shuffle(long_pool)
    random.shuffle(short_pool)
    
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
        for b in active_buckets:
            if len(b['t']) + len(t_seq) <= CONTEXT_SIZE:
                if len(b['t']) > 0: m_seq[0] = 0
                b['t'].extend(t_seq)
                b['m'].extend(m_seq)
                return
        if len(active_buckets) < MAX_OPEN_BUCKETS:
            active_buckets.append({'t': list(t_seq), 'm': list(m_seq)})
        else:
            for i, b in enumerate(active_buckets):
                if len(b['t']) >= (CONTEXT_SIZE * FLUSH_THRESHOLD):
                    pad_len = CONTEXT_SIZE - len(b['t'])
                    write_window(b['t'] + [PAD_TOKEN_ID]*pad_len, b['m'] + [0]*pad_len)
                    active_buckets[i] = {'t': list(t_seq), 'm': list(m_seq)}
                    return
            fullest = max(range(len(active_buckets)), key=lambda idx: len(active_buckets[idx]['t']))
            b = active_buckets[fullest]
            pad_len = CONTEXT_SIZE - len(b['t'])
            write_window(b['t'] + [PAD_TOKEN_ID]*pad_len, b['m'] + [0]*pad_len)
            active_buckets[fullest] = {'t': list(t_seq), 'm': list(m_seq)}

    for l_t, l_m in long_pool:
        pos = 0
        while pos + CONTEXT_SIZE <= len(l_t):
            write_window(l_t[pos:pos+CONTEXT_SIZE], l_m[pos:pos+CONTEXT_SIZE])
            pos += (CONTEXT_SIZE - STRIDE)
        if pos < len(l_t):
            pack_sequence(l_t[pos:], l_m[pos:])

    for s_t, s_m in short_pool:
        pack_sequence(s_t, s_m)

    for b in active_buckets:
        if len(b['t']) > 0:
            pad_len = CONTEXT_SIZE - len(b['t'])
            write_window(b['t'] + [PAD_TOKEN_ID]*pad_len, b['m'] + [0]*pad_len)

    f_d.close(); f_m.close()
    
    full_report = f"=== SHARDING REPORT: {output_dir.upper()} ===\nTotal Tokens: {stats['total_tokens']}\n"
    for f, s in stats["file_breakdown"].items():
        full_report += f" - {f}: {s['kept']} kept / {s['raw']} raw\n"
    print(full_report)
    with open(REPORT_FILE, "a") as rf: rf.write(full_report + "\n")

if __name__ == "__main__":
    if os.path.exists(REPORT_FILE): os.remove(REPORT_FILE)
    run_sharding(is_tune_mode=False)
    run_sharding(is_tune_mode=True)
