import backend
xp = backend.xp

class CrossEntropyLoss:
    def __init__(self, ignore_index=None):
        self.ignore_index = ignore_index
        self.logits = None
        self.targets = None
        self.probs = None

    def forward(self, logits, targets):
        """
        logits: [batch, seq_len, vocab_size]
        targets: [batch, seq_len] (indices)
        """
        self.logits = logits
        self.targets = targets
        
        # Flatten logits and targets
        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = targets.reshape(-1)
        
        # Mask ignored index
        if self.ignore_index is not None:
            mask = targets_flat != self.ignore_index
            logits_flat = logits_flat[mask]
            targets_flat = targets_flat[mask]
        
        # Numerical stability for softmax
        exps = xp.exp(logits_flat - xp.max(logits_flat, axis=1, keepdims=True))
        probs = exps / xp.sum(exps, axis=1, keepdims=True)
        self.probs = probs
        
        # Select probabilities of target classes
        batch_size = probs.shape[0]
        
        # Advanced indexing in CuPy works similarly to NumPy
        correct_logprobs = -xp.log(probs[xp.arange(batch_size), targets_flat] + 1e-9)
        
        loss = xp.mean(correct_logprobs)
        return loss

    def backward(self):
        """
        Returns gradient dL/dlogits
        """
        logits_flat = self.logits.reshape(-1, self.logits.shape[-1])
        targets_flat = self.targets.reshape(-1)
        
        grad = xp.zeros_like(logits_flat)
        
        if self.ignore_index is not None:
            mask = targets_flat != self.ignore_index
            valid_indices = xp.where(mask)[0]
            
            probs = self.probs
            targets_masked = targets_flat[mask]
            
            grad_masked = probs.copy()
            grad_masked[xp.arange(len(targets_masked)), targets_masked] -= 1
            
            # Scale by 1/N for mean reduction
            grad_masked /= len(targets_masked)
            
            grad[valid_indices] = grad_masked
            
        else:
            probs = self.probs
            grad = probs.copy()
            grad[xp.arange(len(targets_flat)), targets_flat] -= 1
            grad /= len(targets_flat)
            
        return grad.reshape(self.logits.shape)
