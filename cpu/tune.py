import os
import sys
import argparse
import time
import json
import torch
import torch.nn as nn

# Import project modules
from trainer.config import UnifiedConfig
from trainer.model import NymphModel
from trainer.optim import MuonAdamW
from trainer.data_loader import get_dataloader

def check_memory(step):
    """Prints a terse memory snapshot to avoid screen reader clutter."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"--- Step {step} Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved ---")
        sys.stdout.flush()

def calculate_mfu(model, tokens_per_sec):
    """
    Dynamic MFU calculation across Ampere (30-series), 
    Ada (40-series), and Blackwell (50-series).
    Values represent Dense BF16/FP16 Tensor TFLOPS.
    """
    device_name = torch.cuda.get_device_name(0).upper()
    
    # Generation-specific TFLOPS mapping (Dense half-precision)
    gpu_peaks = {
        # 30-Series (Ampere)
        "3060": 51.2e12,
        "3070": 81.3e12,
        "3080": 119.0e12,
        "3090": 142.0e12,
        # 40-Series (Ada Lovelace)
        "4060": 15.1e12,
        "4070": 29.0e12,
        "4080": 97.5e12,
        "4090": 165.2e12,
        # 50-Series (Blackwell)
        "5070": 125.0e12,
        "5080": 180.0e12,
        "5090": 209.5e12
    }
    
    # Default to 3090 if no match found
    peak_flops = 142.0e12 
    for key, val in gpu_peaks.items():
        if key in device_name:
            peak_flops = val
            break

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    flops_per_token = 6 * params 
    current_flops = tokens_per_sec * flops_per_token
    return (current_flops / peak_flops) * 100

def main():
    parser = argparse.ArgumentParser(description="iDragonfly Nymph Training Orchestrator (Pure Eager)")
    
    # Model and Path Arguments
    parser.add_argument("--model_name", type=str, required=True, help="Specific IDG model: sprite, nymph, dragonfly, or wyrm")
    parser.add_argument("--config_path", type=str, default=None, help="Optional path to external json config file")
    parser.add_argument("--data_path", type=str, default="./data", help="Directory containing binary shards")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints", help="Base directory for checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to a specific .pt file to resume from")
    
    # Training Hyperparameters
    parser.add_argument("--lr", type=float, default=6e-4, help="Peak learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Micro-batch size per device")
    parser.add_argument("--total_batch_size", type=int, default=262144, help="Target global batch size in tokens")
    parser.add_argument("--max_steps", type=int, default=10000, help="Total training steps")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Learning rate warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for optimizer")
    
    # Reporting and Saving Control
    parser.add_argument("--log_interval", type=int, default=1, help="How often to print metrics")
    parser.add_argument("--save_interval", type=int, default=500, help="How often to save a checkpoint")
    
    # Hardware and Logic Control
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--freeze", action="store_true", help="Enable legacy bottom-half layer and embedding freezing")
    parser.add_argument("--no_opt", action="store_true", help="Do not load optimizer state from checkpoint")
    
    args = parser.parse_args()

    # 1. Setup Environment
    device = args.device
    model_ckpt_path = os.path.join(args.ckpt_dir, args.model_name)
    os.makedirs(model_ckpt_path, exist_ok=True)
    
    # 2. Load Configuration and Model
    mgr = UnifiedConfig(config_path=args.config_path, checkpoint_path=args.resume, model_type=args.model_name)
    config = mgr.norm_config
    model = NymphModel(config)
    
    start_step = 0
    total_tokens_seen = 0
    model.to(device=device, dtype=torch.bfloat16)
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint_data = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint_data.get("model", checkpoint_data), strict=True)
        start_step = checkpoint_data.get("step", 0)
        total_tokens_seen = checkpoint_data.get("total_tokens", 0)
        print(f"Successfully loaded weights. Resume step: {start_step}")
        sys.stdout.flush()
    
    # 3. Data Loading
    ctx_len = config["max_position_embeddings"]
    train_loader = get_dataloader(args.data_path, ctx_len, args.batch_size)
    
    # 4. Optimizer Configuration
    tokens_per_sample = ctx_len - 1
    micro_batch_tokens = args.batch_size * tokens_per_sample
    grad_accum_steps = max(1, args.total_batch_size // micro_batch_tokens)

    # --- Start Layer Freezing Logic ---
    layer_names = []
    for name, _ in model.named_parameters():
        if "layers" in name:
            parts = name.split(".")
            layer_idx = parts[parts.index("layers") + 1]
            if layer_idx not in layer_names:
                layer_names.append(layer_idx)
    
    num_layers = len(layer_names)
    freeze_until = num_layers // 2
    
    if args.freeze:
        print(f"FREEZE ENABLED: Detected {num_layers} layers. Freezing first {freeze_until} layers and embeddings.")
    else:
        print(f"FREEZE DISABLED: Training all {num_layers} layers and embeddings.")
        
    muon_params = []
    adamw_params = []
    for name, p in model.named_parameters():
        should_train = True
        
        # Only apply freezing logic if the --freeze flag is passed
        if args.freeze:
            # Freeze embeddings to prevent concept drift
            if "embed_tokens" in name:
                should_train = False
            
            # Freeze the bottom half of the transformer layers
            if "layers" in name:
                parts = name.split(".")
                layer_idx = int(parts[parts.index("layers") + 1])
                if layer_idx < freeze_until:
                    should_train = False
        
        if not should_train:
            p.requires_grad = False
            continue

        if p.ndim == 2 and "embed_tokens" not in name and "lm_head" not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    param_groups = [
        {"params": muon_params, "kind": "muon", "lr": args.lr, "weight_decay": args.weight_decay, "momentum": 0.95, "ns_steps": 5, "beta2": 0.999},
        {"params": adamw_params, "kind": "adamw", "lr": args.lr, "weight_decay": args.weight_decay, "betas": (0.9, 0.95), "eps": 1e-8}
    ]
    
    optimizer = MuonAdamW(param_groups)
    
    if args.resume and "optimizer" in checkpoint_data:
        # Check if the number of parameter groups in the checkpoint matches our setup
        if args.no_opt:
             print("Optimizer state found in checkpoint, but --no_opt is set. Initializing fresh optimizer.")
        elif len(checkpoint_data["optimizer"]["param_groups"]) == len(optimizer.param_groups):
            try:
                optimizer.load_state_dict(checkpoint_data["optimizer"])
                print("Successfully restored optimizer state.")
            except:
                print("Optimizer state mismatch on internal tensors. Initializing fresh optimizer.")
        else:
            print("Optimizer group count mismatch (Check --freeze setting). Initializing fresh optimizer.")
    
    # 5. Training Loop
    model.train()
    step = start_step
    session_step = 0
    effective_max = args.max_steps - start_step
    data_iter = iter(train_loader)
    
    print(f"--- IDG START: {args.model_name} | Accumulation: {grad_accum_steps} ---")
    sys.stdout.flush()
    
    while step < args.max_steps:
        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0.0
        
        for micro_step in range(grad_accum_steps):
            try:
                x, y, mask = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y, mask = next(data_iter)
                
            x, y, mask = x.to(device, non_blocking=True), y.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            
            # Pure PyTorch Autocast for production stability
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                # FORWARD PASS
                logits, _ = model(input_ids=x)
                
                # Ensure memory is contiguous for flattening
                shift_logits = logits.contiguous()
                shift_labels = y.contiguous()
                shift_mask = mask.contiguous()

                v_size = shift_logits.shape[-1]
                
                # Robust flattening for CrossEntropy
                flat_logits = shift_logits.reshape(-1, v_size)
                flat_labels = shift_labels.reshape(-1)
                flat_mask = shift_mask.reshape(-1)
                
                # Numerical Stability: Cast to FP32 for softmax/log operations
                flat_logits = flat_logits.to(torch.float32)

                # Diagnostic Check for Logit Explosion
                if step == 0 and micro_step == 0:
                    l_max = flat_logits.max().item()
                    t_max = flat_labels.max().item()
                    if t_max >= v_size or abs(l_max) > 20.0:
                        print(f"DIAGNOSTIC: Vocab {v_size} | Target {t_max} | Logit Peak {l_max:.2f}")
                        sys.stdout.flush()

                # Calculate loss in float32
                loss_raw = nn.functional.cross_entropy(flat_logits, flat_labels, reduction='none')
                
                # Weighted Masking with Clamping for stability
                mask_sum = flat_mask.sum()
                if mask_sum < 1.0:
                    # If micro-batch is effectively empty/all masked, produce zero-gradient loss
                    loss = (logits.sum() * 0.0)
                else:
                    loss = (loss_raw * flat_mask).sum() / (mask_sum + 1e-8)
                
                loss = loss / grad_accum_steps
                
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"ABORT: Numerical Instability (NaN/Inf) at Step {step}")
                sys.exit(1)

            loss.backward()
            loss_accum += loss.detach().item()

        # If loss is still huge after model initialization fixes, abort
        if step == 0 and loss_accum > 16.0:
            print(f"CRITICAL: Initial Loss {loss_accum:.2f} is extreme. Check model.py initialization.")
            sys.exit(1)
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Iterative Scheduler Logic
        if session_step < args.warmup_steps:
            curr_lr = args.lr * (session_step + 1) / args.warmup_steps
        else:
            progress = (session_step - args.warmup_steps) / max(1, effective_max - args.warmup_steps)
            curr_lr = args.lr * max(0.0, 1.0 - progress)
            
        for g in optimizer.param_groups:
            g['lr'] = curr_lr
            
        optimizer.step()
        
        # Reporting
        t1 = time.time()
        dt = t1 - t0
        tokens_in_step = grad_accum_steps * micro_batch_tokens
        total_tokens_seen += tokens_in_step
        tps = tokens_in_step / dt
        mfu = calculate_mfu(model, tps)
        
        if step % args.log_interval == 0:
            active_lr = optimizer.param_groups[0]['lr']
            print(f"STEP {step} | Loss: {loss_accum:.4f} | LR: {active_lr:.2e} | TPS: {int(tps)} | MFU: {mfu:.1f}% | Tokens: {total_tokens_seen}")
            sys.stdout.flush()
            if step % 100 == 0:
                check_memory(step)
            
        if step % args.save_interval == 0 and step > start_step:
            checkpoint_path = os.path.join(model_ckpt_path, f"step_{step}.pt")
            torch.save({
                "step": step,
                "total_tokens": total_tokens_seen,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config_raw": mgr.raw
            }, checkpoint_path)
            mgr.save_production_assets(model_ckpt_path)
            print(f"--- Saved Checkpoint: {checkpoint_path} ---")
            sys.stdout.flush()
            
        step += 1
        session_step += 1

    final_dir = os.path.join(model_ckpt_path, "final")
    os.makedirs(final_dir, exist_ok=True)
    mgr.save_production_assets(final_dir)
    
    # Save the standard production bin file
    torch.save(model.state_dict(), os.path.join(final_dir, "pytorch_instr_model.bin"))
    
    # Save the full stateful checkpoint to allow seamless future resumes
    torch.save({
        "step": step,
        "total_tokens": total_tokens_seen,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config_raw": mgr.raw
    }, os.path.join(final_dir, "final_instr_checkpoint.pt"))
    
    print(f"Training complete. Assets ready in {final_dir}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
