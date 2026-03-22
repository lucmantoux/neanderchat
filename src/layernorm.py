import numpy as np

class LayerNorm:
    def __init__(self, d, eps=1e-5):
        self.d = d
        self.gamma = np.ones(d)       # (d,)
        self.beta = np.zeros(d)       # (d,)
        self.eps = eps

    def forward(self, x):
        # x: (d, s)  — each column is one token's embedding
        self.x = x                                            # (d, s)  saved for backward
        mean = np.mean(x, axis=0, keepdims=True)              # (d, s) → (1, s)
        var = np.var(x, axis=0, keepdims=True)                # (d, s) → (1, s)
        self.x_hat = (x - mean) / np.sqrt(var + self.eps)     # (d, s) - (1, s) → (d, s)  /  (1, s) → (d, s)
        return self.gamma[:, None] * self.x_hat + self.beta[:, None]  # (d, 1) * (d, s) → (d, s)  +  (d, 1) → (d, s)

    def backward(self, d_out): # d_out: (d, s) , xhat: (d, s)
        self.d_gamma = np.sum(d_out * self.x_hat, axis=1) # dimension (d,)  gradient of loss w.r.t. gamma
        self.d_beta = np.sum(d_out, axis=1) # Dimensions are (d,)  gradient of loss w.r.t. beta
        self.d_x_hat = self.gamma[:, None] * d_out # (d, 1) * (d, s) → (d, s)  gradient of loss w.r.t. x_hat
        var = np.var(self.x, axis=0, keepdims=True) # (d, s) → (1, s)  variance of input
        std_inv = 1 / np.sqrt(var + self.eps) # (1, s)  inverse of standard deviation
        c1 = self.d_x_hat.mean(axis=0, keepdims=True)              # (1, s)
        c2 = (self.d_x_hat * self.x_hat).mean(axis=0, keepdims=True)  # (1, s)
        d_x = (self.d_x_hat - c1 - self.x_hat * c2) * std_inv      # (d, s)
        return d_x # (d, s)
    
    def update(self, lr):
        self.gamma -= lr * self.d_gamma
        self.beta -= lr * self.d_beta
        