import os
import sys
import argparse
import time
import json
import torch
import torch.nn as nn
import psutil

# Import project modules
from trainer.config import UnifiedConfig
from trainer.model import NymphModel
from trainer.optim import MuonAdamW
from trainer.data_loader import get_dataloader
from trainer.controller import LossController

def check_memory(step):
    """Prints a terse memory snapshot of system RAM for CPU training."""
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / 1024**3
    print(f"--- Step {step} System Memory: {mem_gb:.2f}GB used ---")
    sys.stdout.flush()

def calculate_mfu(model, tokens_per_sec):
    """
    Throughput tracking relative to i5-7300HQ estimated peak.
    Replaces GPU-specific MFU for local CPU hardware.
    """
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Estimated peak TFLOPS for i5-7300HQ is roughly 0.2
    peak_flops = 0.2e12 
    flops_per_token = 6 * params 
    current_flops = tokens_per_sec * flops_per_token
    return (current_flops / peak_flops) * 100

def main():
    parser = argparse.ArgumentParser(description="iDragonfly Nymph Training Orchestrator (CPU Optimized)")
    
    parser.add_argument("--model_name", type=str, required=True, help="Specific IDG model: sprite, nymph, dragonfly, or wyrm")
    parser.add_argument("--config_path", type=str, default=None, help="Optional path to external json config file")
    parser.add_argument("--data_path", type=str, default="./data", help="Directory containing binary shards")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints", help="Base directory for checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to a specific .pt file to resume from")
    
    parser.add_argument("--lr", type=float, default=6e-4, help="Peak learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Micro-batch size (reduced for CPU L3 cache)")
    parser.add_argument("--total_batch_size", type=int, default=16384, help="Target global batch size in tokens")
    parser.add_argument("--max_steps", type=int, default=10000, help="Total training steps")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Learning rate warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for optimizer")
    
    parser.add_argument("--log_interval", type=int, default=1, help="How often to print metrics")
    parser.add_argument("--save_interval", type=int, default=500, help="How often to save a checkpoint")
    parser.add_argument("--use_loss_controller", action="store_true", help="Enable the Shock-and-Recovery LossController")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda)")
    
    args = parser.parse_args()

    device = args.device
    model_ckpt_path = os.path.join(args.ckpt_dir, args.model_name)
    os.makedirs(model_ckpt_path, exist_ok=True)
    
    mgr = UnifiedConfig(config_path=args.config_path, checkpoint_path=args.resume, model_type=args.model_name)
    config = mgr.norm_config
    model = NymphModel(config)
    
    start_step = 0
    total_tokens_seen = 0
    
    # Use FP32 for CPU to avoid bfloat16 emulation overhead
    model.to(device=device, dtype=torch.float32)
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint_data = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint_data.get("model", checkpoint_data), strict=True)
        start_step = checkpoint_data.get("step", 0)
        total_tokens_seen = checkpoint_data.get("total_tokens", 0)
        print(f"Successfully loaded weights. Resume step: {start_step}")
        sys.stdout.flush()
    
    ctx_len = config["max_position_embeddings"]
    train_loader = get_dataloader(args.data_path, ctx_len, args.batch_size)
    
    tokens_per_sample = ctx_len - 1
    micro_batch_tokens = args.batch_size * tokens_per_sample
    grad_accum_steps = max(1, args.total_batch_size // micro_batch_tokens)
        
    muon_params = []
    adamw_params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
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
        optimizer.load_state_dict(checkpoint_data["optimizer"])

    loss_manager = LossController(optimizer=optimizer, save_interval=args.save_interval)
    
    model.train()
    step = start_step
    data_iter = iter(train_loader)
    
    print(f"--- IDG CPU START: {args.model_name} | Accumulation: {grad_accum_steps} ---")
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
            
            # No autocast for CPU; Native FP32 provides best throughput on i5-7300HQ
            logits, _ = model(input_ids=x)
            
            shift_logits = logits.contiguous()
            shift_labels = y.contiguous()
            shift_mask = mask.contiguous()

            v_size = shift_logits.shape[-1]
            flat_logits = shift_logits.reshape(-1, v_size)
            flat_labels = shift_labels.reshape(-1)
            flat_mask = shift_mask.reshape(-1)
            
            loss_raw = nn.functional.cross_entropy(flat_logits, flat_labels, reduction='none')
            
            mask_sum = flat_mask.sum()
            if mask_sum < 1e-6:
                loss = (loss_raw * 0.0).sum()
            else:
                loss = (loss_raw * flat_mask).sum() / (mask_sum + 1e-8)
            
            loss = loss / grad_accum_steps
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"ABORT: Numerical Instability (NaN/Inf) at Step {step}")
                sys.exit(1)

            loss.backward()
            loss_accum += loss.detach().item()

        if step == 0 and loss_accum > 16.0:
            print(f"CRITICAL: Initial Loss {loss_accum:.2f} is extreme. Check model.py initialization.")
            sys.exit(1)
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Learning Rate Schedule
        if args.use_loss_controller and loss_manager.in_warmup:
             pass 
        else:
            if step < args.warmup_steps:
                curr_lr = args.lr * (step + 1) / args.warmup_steps
            else:
                progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
                curr_lr = args.lr * max(0.0, 1.0 - progress)
            for g in optimizer.param_groups:
                g['lr'] = curr_lr
            
            if args.use_loss_controller:
                loss_manager.sync_baseline()
            
        optimizer.step()

        if args.use_loss_controller and step % args.log_interval == 0:
            loss_manager.step(loss_accum, step)
        
        t1 = time.time()
        dt = t1 - t0
        tokens_in_step = grad_accum_steps * micro_batch_tokens
        total_tokens_seen += tokens_in_step
        tps = tokens_in_step / dt
        mfu = calculate_mfu(model, tps)
        
        if step % args.log_interval == 0:
            active_lr = optimizer.param_groups[0]['lr']
            print(f"STEP {step} | Loss: {loss_accum:.4f} | LR: {active_lr:.2e} | TPS: {int(tps)} | Efficiency: {mfu:.1f}% | Tokens: {total_tokens_seen}")
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

    final_dir = os.path.join(model_ckpt_path, "final")
    os.makedirs(final_dir, exist_ok=True)
    mgr.save_production_assets(final_dir)
    torch.save(model.state_dict(), os.path.join(final_dir, "pytorch_model.bin"))
    
    torch.save({
        "step": step,
        "total_tokens": total_tokens_seen,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config_raw": mgr.raw
    }, os.path.join(final_dir, "final_checkpoint.pt"))
    
    print(f"Training complete. Assets ready in {final_dir}")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
