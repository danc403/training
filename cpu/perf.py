import time

class PerformanceTracker:
    def __init__(self, cfg, model_type, gpu_name=None):
        self.cfg = cfg
        self.model_type = model_type
        self.last_time = time.time()
        self.last_step = 0
        self.gpu_database = {
            "h100": 989e12,
            "a100": 312e12,
            "rtx4090": 330e12,
            "rtx3090": 142e12,
            "rtx4080super": 210e12,
            "rtx4060": 15.11e12,
            "default": 330e12
        }
        selected_gpu = gpu_name.lower().replace(" ", "") if gpu_name else "default"
        self.peak_flops = self.gpu_database.get(selected_gpu, self.gpu_database["default"])
        
        # Aligned with UnifiedConfig normalization in config.py
        self.L = cfg["num_hidden_layers"]
        self.E = cfg["hidden_size"]
        self.S = cfg["max_position_embeddings"]

    def log_metrics(self, step, total_batch_tokens, loss, lr):
        current_time = time.time()
        elapsed = current_time - self.last_time
        if elapsed <= 0: return
        steps_passed = step - self.last_step
        tokens_processed = steps_passed * total_batch_tokens
        tokens_per_second = tokens_processed / elapsed
        
        # Flops calculation: 12 * layers * (embedding dimension squared)
        params_estimate = 12 * self.L * (self.E ** 2)
        # 6 * parameters per token for the forward and backward pass
        flops_per_token = 6 * params_estimate
        achieved_flops = tokens_per_second * flops_per_token
        mfu = achieved_flops / self.peak_flops
        
        print(f"Step {step}. Loss {loss:.4f}, LR {lr:.2e}. TOKS {int(tokens_per_second)}, MFU {mfu * 100:.2f} percent")
        self.last_time = current_time
        self.last_step = step
