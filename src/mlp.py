import numpy as np
from .utils import gelu, gelu_derivative

class MLP:
    def __init__(self, d, p, n_layers=1):
        self.d = d
        self.p = p
        self.W_up = np.random.randn(p, d) * 0.02             # (p, d)
        self.b_up = np.zeros(p)                               # (p,)
        residual_scale = 0.02 / np.sqrt(2 * n_layers)
        self.W_down = np.random.randn(d, p) * residual_scale  # (d, p)  GPT-2 residual scaling
        self.b_down = np.zeros(d)                             # (d,)

    def forward(self, x):
        # x: (d, s)
        self.x = x                                            # (d, s)  saved for backward
        self.h_pre = self.W_up @ x + self.b_up[:, None]      # (p, d) @ (d, s) → (p, s)  +  (p, 1) → (p, s)
        self.h = gelu(self.h_pre)                             # (p, s) → (p, s)  element-wise
        out = self.W_down @ self.h + self.b_down[:, None]     # (d, p) @ (p, s) → (d, s)  +  (d, 1) → (d, s)
        return out + x                                        # (d, s) + (d, s) → (d, s)  residual

    def backward(self, d_out):
        # Downward stage
        self.d_h = self.W_down.T @ d_out          # (p, d) @ (d, s) → (p, s)
        self.d_W_down = d_out @ self.h.T          # (d, s) @ (s, p) → (d, p)
        self.d_b_down = d_out.sum(axis=1)   # (d, s) → (d,)
        self.d_h_pre = self.d_h * gelu_derivative(self.h_pre)  # (p, s) element-wise
        # Upward stage
        self.d_x = self.W_up.T @ self.d_h_pre    # (d, p) @ (p, s) → (d, s)
        self.d_W_up = self.d_h_pre @ self.x.T          # (p, s) @ (s, d) → (p, d)
        self.d_b_up = self.d_h_pre.sum(axis=1)   # (p, s) → (p,)
        return self.d_x + d_out   # MLP path + residual path 
    
    def update(self, lr):
        self.W_up -= lr * self.d_W_up
        self.b_up -= lr * self.d_b_up
        self.W_down -= lr * self.d_W_down
        self.b_down -= lr * self.d_b_down