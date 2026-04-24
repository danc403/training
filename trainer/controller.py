import torch

class LossController:
    """
    Monitors training loss and intervenes if the model stagnates.
    Implements a 'Shock and Recovery' mechanism with a grace period for checkpoints.
    Includes a re-verification gate to cancel shocks if natural recovery occurs.
    """
    def __init__(self, optimizer, stagnation_window=300, delta_threshold=0.0001, 
                 warmup_steps=200, patience=3, save_interval=500, grace_period=100):
        self.optimizer = optimizer
        self.stagnation_window = stagnation_window 
        self.delta_threshold = delta_threshold     
        self.patience = patience                   
        self.warmup_steps = warmup_steps           
        
        # Checkpoint Awareness
        self.save_interval = save_interval
        self.grace_period = grace_period
        
        # --- Internal State Trackers ---
        self.loss_history = []
        self.patience_counter = 0
        self.in_warmup = False
        self.warmup_counter = 0
        
        # Re-verification logic
        self.pending_kick = False
        self.pre_grace_min_loss = float('inf')
        
        # We track the baseline LRs from the scheduler separately
        self.base_lrs = [g['lr'] for g in self.optimizer.param_groups]
        self.active_multiplier = 1.0

    def step(self, current_loss, global_step):
        """
        Main entry point for the loss controller per training step.
        """
        # --- OOM PROTECTION GATE ---
        # Ensure loss is a float, not a tensor with a grad_fn attached.
        if torch.is_tensor(current_loss):
            current_loss = current_loss.detach().item()

        # 1. Rolling History
        self.loss_history.append(current_loss)
        if len(self.loss_history) > self.stagnation_window:
            self.loss_history.pop(0)

        # 2. Warmup / Recovery Logic
        if self.in_warmup:
            self.warmup_counter += 1
            
            # Linear ramp from 0 to (Base_LR * active_multiplier)
            for i, g in enumerate(self.optimizer.param_groups):
                target = self.base_lrs[i] * self.active_multiplier
                new_lr = (self.warmup_counter / self.warmup_steps) * target
                g['lr'] = new_lr
            
            if self.warmup_counter >= self.warmup_steps:
                self.in_warmup = False
                self.warmup_counter = 0
            return

        # 3. Deferred Shock Re-Verification
        if self.pending_kick:
            # If we just passed the checkpoint (e.g., step 501)
            if global_step % self.save_interval == 1 or global_step % self.save_interval == 0:
                # Check if current loss is lower than the best seen before/during the grace period
                if current_loss < self.pre_grace_min_loss:
                    print("--- [LOSS CONTROLLER] Natural Recovery Detected. Cancelling Deferred Shock. ---")
                    self.pending_kick = False
                    self.patience_counter = 0
                    self.loss_history.clear()
                else:
                    # Stagnation persisted after save, execute now
                    print("--- [LOSS CONTROLLER] Stagnation Persists Post-Save. Executing Kick. ---")
                    # Calculate a fresh delta for the kick severity
                    half = self.stagnation_window // 2
                    avg_early = sum(self.loss_history[:half]) / half
                    avg_late = sum(self.loss_history[half:]) / half
                    delta = avg_early - avg_late
                    self.pending_kick = False
                    self._trigger_kick(global_step, delta)
                return

        # 4. Stagnation Detection
        if len(self.loss_history) == self.stagnation_window:
            half = self.stagnation_window // 2
            avg_early = sum(self.loss_history[:half]) / half
            avg_late = sum(self.loss_history[half:]) / half
            delta = avg_early - avg_late
            
            if delta < self.delta_threshold:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    # Check for save interval proximity
                    steps_until_save = self.save_interval - (global_step % self.save_interval)
                    if steps_until_save <= self.grace_period and steps_until_save > 0:
                        if not self.pending_kick:
                            print("--- [LOSS CONTROLLER] Plateau Detected. Deferring shock for Checkpoint (Grace Period). ---")
                            self.pending_kick = True
                            self.pre_grace_min_loss = min(self.loss_history)
                    else:
                        self._trigger_kick(global_step, delta)
            else:
                # Success: reset patience and return multiplier to 1.0
                self.patience_counter = 0
                self.active_multiplier = 1.0
                self.pending_kick = False

    def _trigger_kick(self, step, delta):
        """
        Executes the 'Shock'. Sets a temporary boost multiplier.
        Removed optimizer.state.clear() to prevent training instability.
        """
        if delta <= 0:
            self.active_multiplier = 2.0
            status = "CRITICAL (Zero/Negative Slope)"
        elif delta < (self.delta_threshold / 2):
            self.active_multiplier = 1.5
            status = "STAGNANT (Severe Plateau)"
        else:
            self.active_multiplier = 1.2
            status = "SLOW (Minor Plateau)"

        print("\n--- [LOSS CONTROLLER] " + str(status) + " TRIGGERED AT STEP " + str(step) + " ---")
        
        # 1. Log New Targets
        for i, g in enumerate(self.optimizer.param_groups):
            kind = g.get('kind', 'group')
            target_lr = self.base_lrs[i] * self.active_multiplier
            print("--- " + str(kind) + ": Shock Multiplier " + str(self.active_multiplier) + "x. Target: " + "{:.2e}".format(target_lr))
        
        # 2. Reset Control Flags
        self.in_warmup = True
        self.warmup_counter = 0
        self.patience_counter = 0
        self.loss_history.clear()

    def sync_baseline(self):
        """
        Call this in your main loop BEFORE calling LossController.step().
        This ensures the controller knows what the scheduler INTENDS the LR to be.
        """
        if not self.in_warmup:
            # Captures the current LR from the scheduler as the new baseline
            self.base_lrs = [g['lr'] for g in self.optimizer.param_groups]
