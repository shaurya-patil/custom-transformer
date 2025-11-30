import os
import numpy as np

# Default to NumPy
xp = np
use_gpu = False

def init_backend():
    global xp, use_gpu
    backend_name = os.environ.get('TRANSFORMER_BACKEND', 'numpy')
    
    if backend_name == 'cupy':
        try:
            import cupy as cp
            xp = cp
            use_gpu = True
            print("Backend: CuPy (GPU)")
        except ImportError:
            print("Warning: CuPy not found. Falling back to NumPy (CPU).")
            xp = np
            use_gpu = False
    else:
        xp = np
        use_gpu = False
        print("Backend: NumPy (CPU)")

# Initialize on import
init_backend()

def to_numpy(array):
    if use_gpu:
        return xp.asnumpy(array)
    return np.array(array)

def to_tensor(array):
    return xp.array(array)

class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def backward(self, grad):
        raise NotImplementedError
        
    def parameters(self):
        return []
