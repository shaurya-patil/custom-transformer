import backend
xp = backend.xp
from backend import Module

class Linear(Module):
    def __init__(self, in_features, out_features):
        limit = xp.sqrt(6 / (in_features + out_features))
        self.W = xp.random.uniform(-limit, limit, (in_features, out_features))
        self.b = xp.zeros(out_features)
        
        # Gradients
        self.dW = None
        self.db = None
        # Cache
        self.x = None

    def forward(self, x):
        self.x = x
        return xp.dot(x, self.W) + self.b

    def backward(self, grad):
        # grad: [batch, ..., out_features]
        # x: [batch, ..., in_features]
        
        x_reshaped = self.x.reshape(-1, self.x.shape[-1])
        grad_reshaped = grad.reshape(-1, grad.shape[-1])
        
        self.dW = xp.dot(x_reshaped.T, grad_reshaped)
        self.db = xp.sum(grad_reshaped, axis=0)
        
        # dx = grad @ W.T
        dx = xp.dot(grad, self.W.T)
        return dx
        
    def parameters(self):
        return [(self, 'W', 'dW'), (self, 'b', 'db')]

class LayerNorm(Module):
    def __init__(self, features, eps=1e-6):
        self.gamma = xp.ones(features)
        self.beta = xp.zeros(features)
        self.eps = eps
        
        self.dgamma = None
        self.dbeta = None
        self.cache = None

    def forward(self, x):
        mean = xp.mean(x, axis=-1, keepdims=True)
        var = xp.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / xp.sqrt(var + self.eps)
        
        self.cache = (x, x_norm, mean, var)
        return self.gamma * x_norm + self.beta

    def backward(self, grad):
        x, x_norm, mean, var = self.cache
        N = x.shape[-1]
        
        self.dgamma = xp.sum(grad * x_norm, axis=(0, 1)) # Assuming [batch, seq, dim]
        self.dbeta = xp.sum(grad, axis=(0, 1))
        
        dx_norm = grad * self.gamma
        
        std_inv = 1. / xp.sqrt(var + self.eps)
        
        dx = (1. / N) * std_inv * (
            N * dx_norm - 
            xp.sum(dx_norm, axis=-1, keepdims=True) * x_norm - 
            xp.sum(dx_norm, axis=-1, keepdims=True)
        )
        return dx
        
    def parameters(self):
        return [(self, 'gamma', 'dgamma'), (self, 'beta', 'dbeta')]

class GELU(Module):
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return 0.5 * x * (1 + xp.tanh(xp.sqrt(2 / xp.pi) * (x + 0.044715 * xp.power(x, 3))))

    def backward(self, grad):
        x = self.input
        cdf = 0.5 * (1 + xp.tanh(xp.sqrt(2 / xp.pi) * (x + 0.044715 * xp.power(x, 3))))
        pdf = 0.5 * x * (1 - xp.tanh(xp.sqrt(2 / xp.pi) * (x + 0.044715 * xp.power(x, 3)))**2) * \
              (xp.sqrt(2 / xp.pi) * (1 + 3 * 0.044715 * xp.power(x, 2)))
        return grad * (cdf + pdf)

class Softmax(Module):
    def __init__(self, axis=-1):
        self.axis = axis
        self.output = None

    def forward(self, x):
        e_x = xp.exp(x - xp.max(x, axis=self.axis, keepdims=True))
        self.output = e_x / xp.sum(e_x, axis=self.axis, keepdims=True)
        return self.output

    def backward(self, grad):
        sum_grad_y = xp.sum(grad * self.output, axis=self.axis, keepdims=True)
        return self.output * (grad - sum_grad_y)

class Embedding(Module):
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = xp.random.randn(vocab_size, d_model)
        self.dW = None
        self.x = None

    def forward(self, x):
        self.x = x
        return self.weight[x]

    def backward(self, grad):
        self.dW = xp.zeros_like(self.weight)
        
        if backend.use_gpu:
             xp.scatter_add(self.dW, self.x, grad)
        else:
             xp.add.at(self.dW, self.x, grad)
        
        return None 
        
    def parameters(self):
        return [(self, 'weight', 'dW')]

class PositionalEncoding(Module):
    def __init__(self, d_model, max_len=5000):
        self.pe = xp.zeros((max_len, d_model))
        position = xp.arange(0, max_len).reshape(-1, 1)
        div_term = xp.exp(xp.arange(0, d_model, 2) * -(xp.log(10000.0) / d_model))
        
        self.pe[:, 0::2] = xp.sin(position * div_term)
        self.pe[:, 1::2] = xp.cos(position * div_term)
        
        self.pe = self.pe[xp.newaxis, ...] # [1, max_len, d_model]

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :]

    def backward(self, grad):
        return grad
