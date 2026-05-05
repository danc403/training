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
    "Summarize the work titled {title}:",
    "Provide a summary of {title}:",
    "Give me an overview of {title} by {author}:",
    "What is {title} about?",
    "Summarize {title} by {author}:"
]

def get_summary_prompt(title, author):
    if author and author.strip():
        valid_templates = SUMMARY_PROMPTS
    else:
        valid_templates = [p for p in SUMMARY_PROMPTS if "{author}" not in p]
    template = random.choice(valid_templates)
    return template.format(title=title, author=author if author else "")

def apply_random_base_mask(mask, chance):
    return [0 if (m == 1 and random.random() < chance) else m for m in mask]

def save_intermediate_packet(group, tokens, masks, sep_id):
    """
    Standardized sink for article packets.
    """
    os.makedirs("./temp", exist_ok=True)
    d_path = f"./temp/{group}_data.tmp"
    m_path = f"./temp/{group}_mask.tmp"
    
    d_chunk = np.array(list(tokens) + [sep_id], dtype=c.TOKEN_TYPE)
    m_chunk = np.array(list(masks) + [0], dtype=c.MASK_TYPE)
    
    with open(d_path, "ab") as fd, open(m_path, "ab") as fm:
        fd.write(d_chunk.tobytes())
        fm.write(m_chunk.tobytes())

def run_article_sharding(is_tune_mode):
    group_name = "articles_instruct" if is_tune_mode else "articles_base"
    
    tokenizer = Tokenizer.from_file(c.TOKEN_PATH)
    sep_id = c.get_sep_id(tokenizer)
    
    def get_token_id(s):
        tid = tokenizer.token_to_id(s)
        return tid if tid is not None else tokenizer.encode(s, add_special_tokens=False).ids[0]

    eos_id = get_token_id(c.EOS_TOKEN_STR)
    user_id = get_token_id(c.USER_OPEN)
    asst_id = get_token_id(c.ASSISTANT_OPEN)
    newline_id = get_token_id("\n")
    
    article_files = glob.glob("./datasets/articles/articles.jsonl", recursive=True)
    if not article_files:
        print("No article files found in ./datasets/articles/")
        return

    def process_sequence(t_seq, m_seq):
        if not t_seq: return
        if t_seq[-1] != eos_id:
            t_seq.append(eos_id); m_seq.append(1)
        
        # Article striding logic
        if len(t_seq) > c.CONTEXT_SIZE:
            pos = 0
            while pos + c.CONTEXT_SIZE <= len(t_seq):
                save_intermediate_packet(group_name, t_seq[pos:pos+c.CONTEXT_SIZE], m_seq[pos:pos+c.CONTEXT_SIZE], sep_id)
                pos += (c.CONTEXT_SIZE - c.STRIDE)
            return
            
        save_intermediate_packet(group_name, t_seq, m_seq, sep_id)

    for fpath in article_files:
        print(f"Processing Articles: {fpath}")
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                text = data.get("text", "")
                title = data.get("title", "Untitled")
                author = data.get("author", "")
                summary = data.get("summary", "")
                
                tks_raw = tokenizer.encode(text, add_special_tokens=False).ids

                # Respect thresholds from config
                if c.TEXT_MIN_TOKENS <= len(tks_raw) <= c.TEXT_MAX_TOKENS:
                    # --- 1. BASE TEXT PROCESSING ---
                    # Only process raw text if we are NOT in Tune Mode.
                    if not is_tune_mode:
                        msk_base = apply_random_base_mask([1] * len(tks_raw), c.BASE_MASK_CHANCE)
                        process_sequence(tks_raw, msk_base)
                        
                    # --- 2. INSTRUCT PAIR PROCESSING ---
                    # Process summaries in both modes (if configured).
                    if c.INCLUDE_SUMMARIES:
                        # Validation to prevent empty/None sequences in the shards.
                        has_content = summary and str(summary).strip().lower() != "none"
                        
                        if has_content:
                            p_str = get_summary_prompt(title, author)
                            t_q = tokenizer.encode(p_str, add_special_tokens=False).ids
                            t_ans = tokenizer.encode(summary, add_special_tokens=False).ids
                            
                            s_tks = [user_id] + t_q + [newline_id, newline_id, asst_id] + t_ans
                            
                            if is_tune_mode:
                                # Standard SFT masking (Mask prompt, keep response)
                                s_msk = [0] * (len(t_q) + 4) + [1] * (len(t_ans) + 1)
                            else:
                                # Pre-training style masking (Randomly mask entire sequence)
                                s_msk = apply_random_base_mask([1] * len(s_tks), c.BASE_MASK_CHANCE)
                                
                            process_sequence(s_tks, s_msk)

    print(f"COMPLETED {group_name.upper()} ARTICLE DUMP.")

if __name__ == "__main__":
    # Clean old temp files
    for f in ["articles_base_data.tmp", "articles_base_mask.tmp", "articles_instruct_data.tmp", "articles_instruct_mask.tmp"]:
        target = f"./temp/{f}"
        if os.path.exists(target): os.remove(target)

    run_article_sharding(is_tune_mode=False)
    run_article_sharding(is_tune_mode=True)
