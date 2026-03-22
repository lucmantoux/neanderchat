import numpy as np
from .utils import softmax,softmax_backward

class AttentionHead:
    def __init__(self, d, q, n_layers=1):
        self.W_Q = np.random.randn(q, d) * 0.02              # (q, d)
        self.b_Q = np.zeros(q)                                # (q,)
        self.W_K = np.random.randn(q, d) * 0.02              # (q, d)
        self.b_K = np.zeros(q)                                # (q,)
        self.W_V_down = np.random.randn(q, d) * 0.02         # (q, d)
        self.b_V_down = np.zeros(q)                           # (q,)
        residual_scale = 0.02 / np.sqrt(2 * n_layers)
        self.W_V_up = np.random.randn(d, q) * residual_scale # (d, q)  GPT-2 residual scaling
        self.b_V_up = np.zeros(d)                             # (d,)
        self.d = d
        self.q = q
        
    def forward(self, x):
        # x: (d, s)
        self.x = x                                            # (d, s)  saved for backward
        self.Q = self.W_Q @ x + self.b_Q[:, None]            # (q, d) @ (d, s) → (q, s)  +  (q, 1) → (q, s)
        self.K = self.W_K @ x + self.b_K[:, None]            # (q, d) @ (d, s) → (q, s)  +  (q, 1) → (q, s)
        self.KQ = self.Q.T @ self.K                           # (s, q) @ (q, s) → (s, s)
        self.scores = self.KQ / np.sqrt(self.q)               # (s, s) / scalar → (s, s)
        seq_len = x.shape[1]                                  # scalar s
        mask = np.triu(np.ones((seq_len, seq_len)) * -np.inf, k=1)  # (s, s)  upper-triangle = -inf
        self.scores += mask                                   # (s, s) + (s, s) → (s, s)
        self.A = softmax(self.scores, axis=-1)                # (s, s) → (s, s)  each row sums to 1
        self.V_low = self.W_V_down @ x + self.b_V_down[:, None]  # (q, d) @ (d, s) → (q, s)  +  (q, 1) → (q, s)
        self.V = self.W_V_up @ self.V_low + self.b_V_up[:, None] # (d, q) @ (q, s) → (d, s)  +  (d, 1) → (d, s)
        output = self.V @ self.A.T                            # (d, s) @ (s, s) → (d, s)
        return output    # (d, s)
    
    def backward(self, d_out):
        # d_out:(d,s)
        self.d_V = d_out @ self.A  # (d, s) @ (s, s) → (d, s)
        self.d_W_V_up = self.d_V @ self.V_low.T         # (d, s) @ (s, q) → (d, q)
        self.d_b_V_up = self.d_V.sum(axis=1)                           # (d, s) → (d,)
        self.d_V_low = self.W_V_up.T @ self.d_V  # (q, d) @ (d, s) → (q, s)
        self.d_W_V_down = self.d_V_low @ self.x.T         # (q, s) @ (s, d) → (q, d)
        self.d_b_V_down = self.d_V_low.sum(axis=1)                           # (q, s) → (q,)
        d_A = (self.V.T @ d_out).T            
        self.d_scores = softmax_backward(self.A, d_A) 
        self.d_KQ = self.d_scores / np.sqrt(self.q)  # (s, s) / scalar → (s, s)
        self.d_Q = self.K @ self.d_KQ.T      
        self.d_K = self.Q @ self.d_KQ        
        self.d_W_Q = self.d_Q @ self.x.T    # (q, s) @ (s, d) → (q, d)
        self.d_W_K = self.d_K @ self.x.T    # (q, s) @ (s, d) → (q, d)  
        self.d_b_Q = self.d_Q.sum(axis=1)  # (q, s) → (q,)
        self.d_b_K = self.d_K.sum(axis=1)  # (q, s) → (q,)
        self.d_x = self.W_Q.T @ self.d_Q + self.W_K.T @ self.d_K + self.W_V_down.T @ self.d_V_low  # (d, q) @ (q, s) + (d, q) @ (q, s) + (d, q) @ (q, s) → (d, s)
        return self.d_x  # (d, s)
 
    def update(self, lr):
        self.W_Q -= lr * self.d_W_Q
        self.b_Q -= lr * self.d_b_Q
        self.W_K -= lr * self.d_W_K
        self.b_K -= lr * self.d_b_K
        self.W_V_down -= lr * self.d_W_V_down
        self.b_V_down -= lr * self.d_b_V_down
        self.W_V_up -= lr * self.d_W_V_up
        self.b_V_up -= lr * self.d_b_V_up


class MultiHeadAttention:
    def __init__(self, d, q, H, n_layers=1):
        self.d = d
        self.q = q
        self.H = H
        self.heads = [AttentionHead(d, q, n_layers) for _ in range(H)]
        self.b_out = np.zeros(d)                              # (d,)

    def forward(self, x):
        # x: (d, s)
        head_outputs = [head.forward(x) for head in self.heads]  # H × (d, s)
        output = sum(head_outputs) + self.b_out[:, None]      # (d, s) + (d, s) + ... + (d, 1) → (d, s)
        return output                                         # (d, s)

    def backward(self, d_out):
        # d_out: (d, s)
        d_out_per_head = d_out
        d_x_total = sum(head.backward(d_out_per_head) for head in self.heads)  # H × (d, s) → (d, s)
        self.d_b_out = d_out.sum(axis=1)  # (d, s) → (d,)
        return d_x_total  # (d, s)
    
    def update(self, lr):
        for head in self.heads:
            head.update(lr)
        self.b_out -= lr * self.d_b_out