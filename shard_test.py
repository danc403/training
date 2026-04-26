import os
import random
import sys
import numpy as np
from tokenizers import Tokenizer

# --- Configuration ---
CONTEXT_SIZE = 2048
TOKEN_TYPE = np.uint16
MASK_TYPE = np.uint8
TOKEN_PATH = "./tokenizer/tokenizer.json"
DATA_DIRS = ["base", "instruct"]

# Set output encoding to UTF-8 to prevent redirection errors
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def inspect_shards():
    if not os.path.exists(TOKEN_PATH):
        print(f"Error: Tokenizer not found at {TOKEN_PATH}")
        return

    tokenizer = Tokenizer.from_file(TOKEN_PATH)

    for d_dir in DATA_DIRS:
        print(f"\n" + "="*50)
        print(f"INSPECTING DIRECTORY: {d_dir.upper()}")
        print("="*50)

        if not os.path.exists(d_dir):
            continue

        all_files = os.listdir(d_dir)
        data_shards = [f for f in all_files if f.endswith("_data.bin")]
        
        if not data_shards:
            continue

        shard_name = random.choice(data_shards)
        mask_name = shard_name.replace("_data.bin", "_mask.bin")
        data_path = os.path.join(d_dir, shard_name)
        mask_path = os.path.join(d_dir, mask_name)

        if not os.path.exists(mask_path):
            continue

        file_size_bytes = os.path.getsize(data_path)
        total_tokens = file_size_bytes // np.dtype(TOKEN_TYPE).itemsize
        num_windows = total_tokens // CONTEXT_SIZE

        if num_windows < 1:
            continue

        selected_window = random.randint(0, num_windows - 1)
        start_token_idx = selected_window * CONTEXT_SIZE

        with open(data_path, "rb") as f_d, open(mask_path, "rb") as f_m:
            f_d.seek(start_token_idx * np.dtype(TOKEN_TYPE).itemsize)
            f_m.seek(start_token_idx * np.dtype(MASK_TYPE).itemsize)

            chunk_tokens = np.fromfile(f_d, dtype=TOKEN_TYPE, count=CONTEXT_SIZE)
            chunk_masks = np.fromfile(f_m, dtype=MASK_TYPE, count=CONTEXT_SIZE)

        print(f"\n--- Shard: {shard_name} | Window: {selected_window} ---")
        print(f"Mask Summary: {int(np.sum(chunk_masks))} active / {CONTEXT_SIZE} total")
        print("-" * 30)

        # Build output string for efficient printing and redirection
        output_buffer = []
        for i in range(len(chunk_tokens)):
            # Handle potential token ID out of range or special cases
            try:
                token_text = tokenizer.decode([int(chunk_tokens[i])], skip_special_tokens=False)
            except:
                token_text = f" [ERR:{chunk_tokens[i]}] "

            if chunk_masks[i] == 0:
                # Wrap masked tokens in brackets to visualize the ignored parts
                output_buffer.append(f"[[{token_text}]]")
            else:
                output_buffer.append(token_text)

        print("".join(output_buffer))
        print("-" * 30)

if __name__ == "__main__":
    inspect_shards()
