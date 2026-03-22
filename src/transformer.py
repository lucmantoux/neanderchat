import numpy as np
from .layernorm import LayerNorm
from .attention import MultiHeadAttention
from .mlp import MLP
from .embedding import Embedding, Unembedding
from .utils import softmax

class TransformerBlock:
    def __init__(self, d, q, H, p, n_layers=1):
        self.d = d
        self.q = q
        self.H = H
        self.p = p
        self.ln_attn = LayerNorm(d)
        self.attn = MultiHeadAttention(d, q, H, n_layers)
        self.ln_mlp = LayerNorm(d)
        self.mlp = MLP(d, p, n_layers)

    def forward(self, x):
        # x: (d, s)
        x_norm = self.ln_attn.forward(x)                     # (d, s) → (d, s)
        x = x + self.attn.forward(x_norm)                    # (d, s) + (d, s) → (d, s)  residual around attention
        x_norm = self.ln_mlp.forward(x)                      # (d, s) → (d, s)
        x = self.mlp.forward(x_norm)                         # (d, s) → (d, s)  MLP does residual internally
        return x       # (d, s)
    
    def backward(self, d_out):
        d_mlp = self.mlp.backward(d_out)                    # (d, s) — includes MLP residual
        d_ln_mlp = self.ln_mlp.backward(d_mlp)              # (d, s)
        d_attn = self.attn.backward(d_ln_mlp)               # (d, s) — path through attention
        d_ln_attn = self.ln_attn.backward(d_attn)           # (d, s)
        return d_ln_attn + d_ln_mlp                          # attention path + residual path

    def update(self, lr):
        self.ln_attn.update(lr)
        self.attn.update(lr)
        self.ln_mlp.update(lr)
        self.mlp.update(lr)

class NeanderChat:
    """
    Full language model: embedding -> N transformer blocks -> unembedding.
    """

    def __init__(self, vocab_size, embedding_dim, sequence_length, key_dim, num_heads, proj_dim, n_layers):
        self.vocab_size = vocab_size
        self.d = embedding_dim
        self.s = sequence_length
        self.q = key_dim
        self.H = num_heads
        self.p = proj_dim
        self.n_layers = n_layers
        self.embedding = Embedding(vocab_size, self.d, self.s)
        self.blocks = [TransformerBlock(self.d, self.q, self.H, self.p, self.n_layers) for _ in range(self.n_layers)]
        self.final_ln = LayerNorm(self.d)
        self.unembedding = Unembedding(self.d, vocab_size)

    def forward(self, token_ids):
        # token_ids: (s,)
        x = self.embedding.forward(token_ids)                 # (s,) → (d, s)
        for block in self.blocks:
            x = block.forward(x)                              # (d, s) → (d, s)
        x = self.final_ln.forward(x)                          # (d, s) → (d, s)
        logits = self.unembedding.forward(x)                  # (d, s) → (V, s)
        return logits                                         # (V, s)

    def generate(self, token_ids, max_new_tokens=50, temperature=1.0):
        # token_ids: (s,)
        for _ in range(max_new_tokens):
            logits = self.forward(token_ids)                  # (V, s)
            last_token_logits = logits[:, -1] / temperature   # (V, s)[:, -1] → (V,) / scalar → (V,)
            probs = softmax(last_token_logits)                # (V,) → (V,)
            next_token_id = np.random.choice(self.vocab_size, p=probs)  # scalar
            token_ids = np.append(token_ids, next_token_id)   # (s,) → (s+1,)
            if len(token_ids) > self.s:
                token_ids = token_ids[-self.s:]               # truncate to (s,)
        return token_ids  # (s',)
    
    def backward(self, d_logits):
        # d_logits: (V, s)
        d_x = self.unembedding.backward(d_logits)          # (d, s)
        d_x = self.final_ln.backward(d_x)                  # (d, s)
        for block in reversed(self.blocks):
            d_x = block.backward(d_x)                      # (d, s)
        d_W_E, d_W_pos = self.embedding.backward(d_x)     # (d, V), (d,)
        return d_W_E, d_W_pos  # gradients for embedding parameters
    
    def clip_gradients(self, max_norm=1.0):
        grads = []
        for layer in self._all_layers():
            for name, val in vars(layer).items():
                if name.startswith('d_') and isinstance(val, np.ndarray):
                    grads.append(val)
        if not grads:
            return
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-8)
            for g in grads:
                g *= scale

    def _all_layers(self):
        yield self.embedding
        yield self.unembedding
        yield self.final_ln
        for block in self.blocks:
            yield block.ln_attn
            yield block.ln_mlp
            yield block.mlp
            yield block.attn
            for head in block.attn.heads:
                yield head

    def update(self, lr):
        self.embedding.update(lr)
        for block in self.blocks:
            block.update(lr)
        self.final_ln.update(lr)
        self.unembedding.update(lr)

    def save_weights(self, path):
        data = {
            '_vocab_size': np.array(self.vocab_size),
            '_d': np.array(self.d), '_c': np.array(self.s),
            '_q': np.array(self.q), '_H': np.array(self.H),
            '_p': np.array(self.p), '_n_layers': np.array(self.n_layers),
            'emb_W_E': self.embedding.W_E, 'emb_W_pos': self.embedding.W_pos,
            'final_ln_gamma': self.final_ln.gamma, 'final_ln_beta': self.final_ln.beta,
            'unemb_W_U': self.unembedding.W_U, 'unemb_b_U': self.unembedding.b_U,
        }
        for i, blk in enumerate(self.blocks):
            data[f'b{i}_ln_attn_g'] = blk.ln_attn.gamma
            data[f'b{i}_ln_attn_b'] = blk.ln_attn.beta
            data[f'b{i}_ln_mlp_g'] = blk.ln_mlp.gamma
            data[f'b{i}_ln_mlp_b'] = blk.ln_mlp.beta
            data[f'b{i}_attn_b_out'] = blk.attn.b_out
            data[f'b{i}_mlp_W_up'] = blk.mlp.W_up
            data[f'b{i}_mlp_b_up'] = blk.mlp.b_up
            data[f'b{i}_mlp_W_down'] = blk.mlp.W_down
            data[f'b{i}_mlp_b_down'] = blk.mlp.b_down
            for j, h in enumerate(blk.attn.heads):
                p = f'b{i}_h{j}'
                data.update({
                    f'{p}_W_Q': h.W_Q, f'{p}_b_Q': h.b_Q,
                    f'{p}_W_K': h.W_K, f'{p}_b_K': h.b_K,
                    f'{p}_W_Vd': h.W_V_down, f'{p}_b_Vd': h.b_V_down,
                    f'{p}_W_Vu': h.W_V_up, f'{p}_b_Vu': h.b_V_up,
                })
        np.savez(path, **data)

    def load_weights(self, path):
        d = np.load(path)
        self.embedding.W_E = d['emb_W_E']
        self.embedding.W_pos = d['emb_W_pos']
        self.final_ln.gamma = d['final_ln_gamma']
        self.final_ln.beta = d['final_ln_beta']
        self.unembedding.W_U = d['unemb_W_U']
        self.unembedding.b_U = d['unemb_b_U']
        for i, blk in enumerate(self.blocks):
            blk.ln_attn.gamma = d[f'b{i}_ln_attn_g']
            blk.ln_attn.beta = d[f'b{i}_ln_attn_b']
            blk.ln_mlp.gamma = d[f'b{i}_ln_mlp_g']
            blk.ln_mlp.beta = d[f'b{i}_ln_mlp_b']
            blk.attn.b_out = d[f'b{i}_attn_b_out']
            blk.mlp.W_up = d[f'b{i}_mlp_W_up']
            blk.mlp.b_up = d[f'b{i}_mlp_b_up']
            blk.mlp.W_down = d[f'b{i}_mlp_W_down']
            blk.mlp.b_down = d[f'b{i}_mlp_b_down']
            for j, h in enumerate(blk.attn.heads):
                p = f'b{i}_h{j}'
                h.W_Q = d[f'{p}_W_Q']; h.b_Q = d[f'{p}_b_Q']
                h.W_K = d[f'{p}_W_K']; h.b_K = d[f'{p}_b_K']
                h.W_V_down = d[f'{p}_W_Vd']; h.b_V_down = d[f'{p}_b_Vd']
                h.W_V_up = d[f'{p}_W_Vu']; h.b_V_up = d[f'{p}_b_Vu']
