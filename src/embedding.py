import numpy as np
from .utils import softmax

class Embedding:
    def __init__(self, vocab_size, d, c):
        self.vocab_size = vocab_size
        self.d = d
        self.c = c
        self.W_E = np.random.randn(d, vocab_size) * 0.02     # (d, V)
        self.W_pos = np.random.randn(d, c) * 0.02            # (d, c)
        
    def forward(self, token_ids):
        # token_ids: (s,)  — integer indices
        self.token_ids = token_ids                            # (s,)  saved for backward
        seq_len = token_ids.shape[0]                          # scalar s
        embedded = self.W_E[:, token_ids]                     # (d, V)[:, s] → (d, s)  column lookup
        return embedded + self.W_pos[:, :seq_len]             # (d, s) + (d, s) → (d, s)
          
    def backward(self, d_embedded):
        # d_embedded: (d, s)
        seq_len = self.token_ids.shape[0]                          # scalar s
        self.d_W_pos = np.zeros_like(self.W_pos)          # (d, c)
        self.d_W_pos[:, :seq_len] = d_embedded             # first s columns get the gradient           # (d, s)[:, s] → (d,)  sum over sequence length
        self.d_W_E = np.zeros_like(self.W_E)                           # (d, V)
        np.add.at(self.d_W_E, (np.s_[:], self.token_ids), d_embedded)          # add d_embedded.T to rows of d_W_E at indices token_ids
        return self.d_W_E, self.d_W_pos    # (d, V), (d,)
    
    def update(self, lr):
        self.W_E -= lr * self.d_W_E
        self.W_pos -= lr * self.d_W_pos      
    
class Unembedding:
    def __init__(self, d, vocab_size):
        self.vocab_size = vocab_size
        self.d = d
        self.W_U = np.random.randn(vocab_size, d) * 0.02     # (V, d)
        self.b_U = np.zeros(vocab_size)                       # (V,)

    def forward(self, x):
        # x: (d, s)
        self.x = x                                            # (d, s)  saved for backward
        logits = self.W_U @ x + self.b_U[:, None]            # (V, d) @ (d, s) → (V, s)  +  (V, 1) → (V, s)
        return logits                                         # (V, s)

    def backward(self, d_logits):
        # d_logits: (V, s)
        self.d_W_U = d_logits @ self.x.T                     # (V, s) @ (s, d) → (V, d)
        self.d_b_U = d_logits.sum(axis=1)                    # (V, s) → (V,)
        self.d_x = self.W_U.T @ d_logits                        # (d, V) @ (V, s) → (d, s)
        return self.d_x  # (d, s)
    
    def update(self, lr):
        self.W_U -= lr * self.d_W_U
        self.b_U -= lr * self.d_b_U