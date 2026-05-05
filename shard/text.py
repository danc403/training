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

# --- PROMPT VARIANTS ---
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

def get_summary_prompt(title, author):
    valid_templates = SUMMARY_PROMPTS if author else [p for p in SUMMARY_PROMPTS if "{author}" not in p]
    return random.choice(valid_templates).format(title=title, author=author if author else "")

def get_book_prompt(title):
    return random.choice(BOOK_PROMPTS).format(title=title)

def apply_random_base_mask(tokens, chance):
    return [0 if random.random() < chance else 1 for _ in range(len(tokens))]

def save_intermediate_packet(group, tokens, masks, sep_id):
    """
    Appends processed packets to temp binaries with the sentinel.
    """
    os.makedirs("./temp", exist_ok=True)
    d_path = f"./temp/{group}_data.tmp"
    m_path = f"./temp/{group}_mask.tmp"
    
    d_chunk = np.array(list(tokens) + [sep_id], dtype=c.TOKEN_TYPE)
    m_chunk = np.array(list(masks) + [0], dtype=c.MASK_TYPE)
    
    with open(d_path, "ab") as fd, open(m_path, "ab") as fm:
        fd.write(d_chunk.tobytes())
        fm.write(m_chunk.tobytes())

def run_text_sharding(is_tune_mode=False):
    group_name = "text_instruct" if is_tune_mode else "text_base"
    
    tokenizer = Tokenizer.from_file(c.TOKEN_PATH)
    sep_id = c.get_sep_id(tokenizer)
    
    def get_token_id(s):
        tid = tokenizer.token_to_id(s)
        return tid if tid is not None else tokenizer.encode(s, add_special_tokens=False).ids[0]

    eos_id = get_token_id(c.EOS_TOKEN_STR)
    user_id = get_token_id(c.USER_OPEN)
    asst_id = get_token_id(c.ASSISTANT_OPEN)
    newline_id = get_token_id("\n")

    def process_sequence(t_seq, m_seq):
        """
        Handles the actual windowing/striding logic.
        Writes finalized packets to the intermediate sink.
        """
        if not t_seq: return
        if t_seq[-1] != eos_id:
            t_seq.append(eos_id); m_seq.append(1)

        # Long Form: Stride through windows
        if len(t_seq) > c.CONTEXT_SIZE:
            pos = 0
            while pos + c.CONTEXT_SIZE <= len(t_seq):
                save_intermediate_packet(group_name, t_seq[pos:pos+c.CONTEXT_SIZE], m_seq[pos:pos+c.CONTEXT_SIZE], sep_id)
                pos += (c.CONTEXT_SIZE - c.STRIDE)
            return
        
        # Short Form: Write as single packet
        save_intermediate_packet(group_name, t_seq, m_seq, sep_id)

    # Define the list of files to process
    target_files = ["./datasets/text/noss.jsonl"]
    if getattr(c, "INCLUDE_OSS", True): 
        target_files.append("./datasets/text/oss.jsonl")

    for file_path in target_files:
        if not os.path.exists(file_path):
            print(f"Skipping: {file_path}")
            continue
            
        print(f"Processing Text Group: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                text_content = data.get("text", "")
                summary_content = data.get("summary", "")
                title = data.get("title")
                author = data.get("author", "")

                tks = tokenizer.encode(text_content, add_special_tokens=False).ids

                # --- 1. SUMMARY INJECTION (Instruct/Base) ---
                if c.INCLUDE_SUMMARIES and title:
                    # Content check to prevent empty/None sequences
                    has_summary = summary_content and str(summary_content).strip().lower() != "none"
                    
                    if has_summary:
                        p_str = get_summary_prompt(title, author)
                        t_q = tokenizer.encode(p_str, add_special_tokens=False).ids
                        t_ans = tokenizer.encode(summary_content, add_special_tokens=False).ids
                        
                        s_tks = [user_id] + t_q + [newline_id, newline_id, asst_id] + t_ans + [eos_id]
                        
                        if is_tune_mode:
                            # SFT Masking: Mask prompt, keep response
                            s_msk = [0] * (len(t_q) + 4) + [1] * (len(t_ans) + 1)
                            process_sequence(s_tks, s_msk)
                        else:
                            # Base Masking: Randomly mask whole sequence
                            s_msk = apply_random_base_mask(s_tks, c.BASE_MASK_CHANCE)
                            process_sequence(s_tks, s_msk)

                # --- 2. INTRODUCTION INJECTION (Instruct/Base) ---
                if c.INCLUDE_INTRODUCTIONS and title and len(tks) >= c.BOOK_MIN_TOKENS:
                    # Content check for the text body used in intro
                    has_intro_text = text_content and str(text_content).strip().lower() != "none"
                    
                    if has_intro_text:
                        p_str = get_book_prompt(title)
                        t_q = tokenizer.encode(p_str, add_special_tokens=False).ids
                        
                        # Snippet capped to fit single context window comfortably
                        intro_chunk = tks[:2000]
                        r_tks = [user_id] + t_q + [newline_id, newline_id, asst_id] + intro_chunk + [eos_id]
                        
                        if is_tune_mode:
                            # SFT Masking: Mask prompt, keep response
                            r_msk = [0] * (len(t_q) + 4) + [1] * (len(intro_chunk) + 1)
                            process_sequence(r_tks, r_msk)
                        else:
                            # Base Masking: Randomly mask whole sequence
                            r_msk = apply_random_base_mask(r_tks, c.BASE_MASK_CHANCE)
                            process_sequence(r_tks, r_msk)

                # --- 3. BASE SLIDING WINDOW (Full Text) ---
                # Only processed if NOT in tune mode
                if not is_tune_mode:
                    has_text = text_content and str(text_content).strip().lower() != "none"
                    if has_text:
                        msk = apply_random_base_mask(tks, c.BASE_MASK_CHANCE)
                        process_sequence(tks + [eos_id], msk + [1])

if __name__ == "__main__":
    # Clean old temp files for this group if starting fresh
    for f in ["text_base_data.tmp", "text_base_mask.tmp", "text_instruct_data.tmp", "text_instruct_mask.tmp"]:
        if os.path.exists(f"./temp/{f}"): os.remove(f"./temp/{f}")
        
    run_text_sharding(is_tune_mode=False) 
    run_text_sharding(is_tune_mode=True)
