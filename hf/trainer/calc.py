import os
import re
import json

# Configuration
LOG_FILES = ["sprite.log", "nymph.log", "dragonfly.log", "checkpoints/wyrm0/wyrm.log"]
OUTPUT_REPORT = "audit.json"
MAX_HISTORY = 3

def format_lr(lr_str):
    """Converts scientific notation to decimal and trims trailing zeros."""
    try:
        decimal_str = "{:.10f}".format(float(lr_str))
        return decimal_str.rstrip('0').rstrip('.')
    except (ValueError, TypeError):
        return "0"

def parse_logs_sequential():
    summary = {}

    for log_file in LOG_FILES:
        if not os.path.exists(log_file):
            continue
            
        model_name = os.path.basename(log_file).replace(".log", "")
        if "wyrm" in log_file and "wyrm0" in log_file:
            model_name = "wyrm0"
        
        # State variables
        last_metrics = None
        last_memory = {"allocated": "N/A", "reserved": "N/A"}
        checkpoint_history = []

        with open(log_file, 'r') as f:
            for line in f:
                # 1. Look for memory summary line
                # Format: --- Step 1000 Memory: 2.88GB allocated, 21.09GB reserved ---
                mem_match = re.search(r"--- Step \d+ Memory: ([\d\.]+)GB allocated, ([\d\.]+)GB reserved ---", line)
                if mem_match:
                    last_memory = {
                        "allocated": f"{mem_match.group(1)}GB",
                        "reserved": f"{mem_match.group(2)}GB"
                    }
                    continue

                # 2. Look for metrics line
                m_match = re.search(r"Step (\d+)\. Loss ([\d\.]+), LR ([\d\.eE\-\+]+)\.", line)
                if m_match:
                    last_metrics = {
                        "step": int(m_match.group(1)),
                        "loss": float(m_match.group(2)),
                        "lr": format_lr(m_match.group(3)),
                        "memory": last_memory.copy()
                    }
                    continue

                # 3. Look for the saved confirmation
                if last_metrics and "Saved:" in line:
                    s_match = re.search(r"step_(\d+)\.pt", line)
                    if s_match and int(s_match.group(1)) == last_metrics["step"]:
                        checkpoint_history.append({
                            "step": last_metrics["step"],
                            "loss": last_metrics["loss"],
                            "lr": last_metrics["lr"],
                            "allocated": last_metrics["memory"]["allocated"],
                            "reserved": last_metrics["memory"]["reserved"],
                            "path": f"./checkpoints/{model_name}/step_{last_metrics['step']}.pt"
                        })
                        if len(checkpoint_history) > MAX_HISTORY:
                            checkpoint_history.pop(0)

        summary[model_name] = checkpoint_history

    with open(OUTPUT_REPORT, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print("\n=== SEQUENTIAL LOG AUDIT (Last 3 Saves) ===")
    for model, history in summary.items():
        print(f"\nMODEL: {model.upper()}")
        if not history:
            print("  No confirmed 'Saved:' entries found.")
            continue
            
        for i, entry in enumerate(reversed(history)):
            label = "LATEST" if i == 0 else f"PREV {i}"
            print(f"  [{label}] Step: {entry['step']:<6} | Loss: {entry['loss']:.4f} | LR: {entry['lr']}")
            print(f"           Memory: {entry['allocated']} alloc / {entry['reserved']} res")

if __name__ == "__main__":
    parse_logs_sequential()
