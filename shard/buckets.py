import os
import numpy as np
import glob
import random
from tokenizers import Tokenizer 

try:
    import shard.config as c
except ImportError:
    import config as c

def run_dynamic_bucket_interleave(is_tune_mode):
    mode_suffix = "instruct" if is_tune_mode else "base"
    output_dir = mode_suffix
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- STARTING {mode_suffix.upper()} INTERLEAVE ---")
    
    tokenizer = Tokenizer.from_file(c.TOKEN_PATH)
    SEP_ID = int(c.get_sep_id(tokenizer))
    
    data_files = sorted(glob.glob(f"./temp/*_{mode_suffix}_data.tmp"))
    if not data_files: return

    # 1. Setup Weighted Pools based on Sequence Counts
    pools = []
    total_global_sequences = 0
    d_dtype = np.dtype(c.TOKEN_TYPE)
    m_dtype = np.dtype(c.MASK_TYPE)

    for d_path in data_files:
        m_path = d_path.replace("_data.tmp", "_mask.tmp")
        
        sequence_count = 0
        with open(d_path, "rb") as f_scan:
            while True:
                chunk = f_scan.read(d_dtype.itemsize * 4096)
                if not chunk:
                    break
                data_chunk = np.frombuffer(chunk, dtype=c.TOKEN_TYPE)
                sequence_count += np.count_nonzero(data_chunk == SEP_ID)

        total_global_sequences += sequence_count
        pools.append({
            "name": os.path.basename(d_path),
            "d_handle": open(d_path, "rb"),
            "m_handle": open(m_path, "rb"),
            "accumulator": 0.0,
            "exhausted": False,
            "total_sequences": sequence_count,
            "sequences_pulled": 0 
        })

    for p in pools:
        p["step_value"] = p["total_sequences"] / total_global_sequences if total_global_sequences > 0 else 0

    # 2. Shard & Bucket State
    open_buckets = [] # Starts empty, will grow as needed
    shard_idx = 0
    tokens_in_shard = 0
    prefix = f"ares_{mode_suffix}"
    
    # 10% Threshold for padding/closing
    CLOSE_THRESHOLD = int(c.CONTEXT_SIZE * 0.90)
    
    f_d = open(os.path.join(output_dir, f"{prefix}_shard_{shard_idx}_data.bin"), "wb")
    f_m = open(os.path.join(output_dir, f"{prefix}_shard_{shard_idx}_mask.bin"), "wb")

    def flush_to_shard(bucket):
        nonlocal tokens_in_shard, shard_idx, f_d, f_m
        
        # Determine exact padding needed for alignment
        current_len = bucket["data"].shape[0]
        pad_len = c.CONTEXT_SIZE - current_len
        
        final_d = np.concatenate([bucket["data"], np.full(pad_len, c.PAD_TOKEN_ID, dtype=c.TOKEN_TYPE)])
        final_m = np.concatenate([bucket["mask"], np.zeros(pad_len, dtype=c.MASK_TYPE)])
        
        f_d.write(final_d.tobytes())
        f_m.write(final_m.tobytes())
        f_d.flush(); f_m.flush()
        
        tokens_in_shard += c.CONTEXT_SIZE
        print(".", end="", flush=True)

        if tokens_in_shard >= c.SHARD_SIZE:
            f_d.close(); f_m.close()
            shard_idx += 1
            f_d = open(os.path.join(output_dir, f"{prefix}_shard_{shard_idx}_data.bin"), "wb")
            f_m = open(os.path.join(output_dir, f"{prefix}_shard_{shard_idx}_mask.bin"), "wb")
            tokens_in_shard = 0
            print(f"\n[Rotation] Shard {shard_idx}")

    # 3. Extraction Loop
    try:
        active_pools = [p for p in pools if p["total_sequences"] > 0]
        while active_pools:
            for p in active_pools:
                p["accumulator"] += p["step_value"]
            
            target = max(active_pools, key=lambda x: x["accumulator"])
            target["accumulator"] -= 1.0
            
            seq_tokens, seq_masks = [], []
            while True:
                b_d = target["d_handle"].read(d_dtype.itemsize)
                b_m = target["m_handle"].read(m_dtype.itemsize)
                
                if not b_d:
                    target["exhausted"] = True
                    break
                    
                t_id = np.frombuffer(b_d, dtype=c.TOKEN_TYPE)[0]
                m_val = np.frombuffer(b_m, dtype=c.MASK_TYPE)[0]
                
                if int(t_id) == SEP_ID:
                    target["sequences_pulled"] += 1
                    break 
                
                seq_tokens.append(t_id)
                seq_masks.append(m_val)

            if target["sequences_pulled"] >= target["total_sequences"] or target["exhausted"]:
                target["d_handle"].close(); target["m_handle"].close()
                active_pools.remove(target)

            if seq_tokens:
                s_d = np.array(seq_tokens, dtype=c.TOKEN_TYPE)
                s_m = np.array(seq_masks, dtype=c.MASK_TYPE)
                s_len = s_d.shape[0]

                # Truncate sequences that exceed context on their own
                if s_len > c.CONTEXT_SIZE:
                    s_d = s_d[:c.CONTEXT_SIZE]
                    s_m = s_m[:c.CONTEXT_SIZE]
                    s_len = c.CONTEXT_SIZE

                placed = False
                # Always start checking from the first open bucket
                for i in range(len(open_buckets)):
                    b = open_buckets[i]
                    if b["count"] + s_len <= c.CONTEXT_SIZE:
                        b["data"] = np.concatenate([b["data"], s_d])
                        b["mask"] = np.concatenate([b["mask"], s_m])
                        b["count"] = b["data"].shape[0]
                        placed = True
                        break
                    else:
                        # If it doesn't fit AND bucket is >90% full, close it now
                        if b["count"] >= CLOSE_THRESHOLD:
                            flush_to_shard(b)
                            open_buckets.pop(i)
                            # We popped an element, so don't increment index or we skip one
                            # However, in this logic, we exit the loop once placed or new bucket added
                            break 

                if not placed:
                    # Start a new bucket if existing ones couldn't take the sequence
                    open_buckets.append({"data": s_d, "mask": s_m, "count": s_len})

        # 4. Final Merge & Close
        print(f"\nFinalizing {len(open_buckets)} residual buckets...")
        
        # Attempt to merge remaining buckets if they fit together
        while len(open_buckets) > 1:
            merged = False
            for i in range(len(open_buckets)):
                for j in range(i + 1, len(open_buckets)):
                    if open_buckets[i]["count"] + open_buckets[j]["count"] <= c.CONTEXT_SIZE:
                        open_buckets[i]["data"] = np.concatenate([open_buckets[i]["data"], open_buckets[j]["data"]])
                        open_buckets[i]["mask"] = np.concatenate([open_buckets[i]["mask"], open_buckets[j]["mask"]])
                        open_buckets[i]["count"] = open_buckets[i]["data"].shape[0]
                        open_buckets.pop(j)
                        merged = True
                        break
                if merged: break
            if not merged: break

        for b in open_buckets:
            flush_to_shard(b)

    finally:
        if not f_d.closed: f_d.close()
        if not f_m.closed: f_m.close()
        for p in pools:
            try:
                p["d_handle"].close(); p["m_handle"].close()
            except: pass

    print(f"\nCOMPLETED: {shard_idx + 1} shards created.")

if __name__ == "__main__":
    run_dynamic_bucket_interleave(is_tune_mode=False)
    run_dynamic_bucket_interleave(is_tune_mode=True)
