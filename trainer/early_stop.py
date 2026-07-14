import torch

class EarlyStopping:
    """
    Monitors training loss for stagnation.
    Integrates with LossController to ensure we do not stop during recovery phases.
    """
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def check(self, current_loss, loss_manager):
        """
        Evaluates the current loss. Returns True if training should stop.
        loss_manager is a reference to the active LossController.
        """
        # Audit Gate: Do not interfere with LossController interventions
        if loss_manager.in_warmup:
            return False

        # If we see improvement beyond the threshold, reset patience
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            # Stagnation detected
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        
        return False
