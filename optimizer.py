import backend
xp = backend.xp

class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        
        self.m = {}
        self.v = {}
        self.t = 0
        
        # Initialize moments
        for i, (obj, param_name, grad_name) in enumerate(self.params):
            param = getattr(obj, param_name)
            self.m[i] = xp.zeros_like(param)
            self.v[i] = xp.zeros_like(param)

    def step(self):
        self.t += 1
        
        for i, (obj, param_name, grad_name) in enumerate(self.params):
            param = getattr(obj, param_name)
            grad = getattr(obj, grad_name)
            
            if grad is None:
                continue
                
            # Update moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameter
            param_update = -self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)
            
            # Apply update
            new_param = param + param_update
            setattr(obj, param_name, new_param)

    def zero_grad(self):
        for obj, param_name, grad_name in self.params:
            setattr(obj, grad_name, None)
