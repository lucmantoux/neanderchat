import torch
from .embedding import Embedding, Unembedding
from .layernorm import LayerNorm
from .attention import MultiHeadAttention
from .mlp import MLP
from .utils import softmax


class TransformerBlock:
    def __init__(self, d, q, H, p, n_layers, device):
        self.ln_attn = LayerNorm(d, device)
        self.attn = MultiHeadAttention(d, q, H, n_layers, device)
        self.ln_mlp = LayerNorm(d, device)
        self.mlp = MLP(d, p, n_layers, device)

    def forward(self, x):
        x = x + self.attn.forward(self.ln_attn.forward(x))
        x = self.mlp.forward(self.ln_mlp.forward(x))
        return x

    def parameters(self):
        return self.ln_attn.parameters() + self.attn.parameters() + self.ln_mlp.parameters() + self.mlp.parameters()


class NeanderChat:
    def __init__(self, vocab_size, d, c, q, H, p, n_layers, device):
        self.vocab_size = vocab_size
        self.d, self.c, self.q, self.H, self.p, self.n_layers = d, c, q, H, p, n_layers
        self.device = device
        self.embedding = Embedding(vocab_size, d, c, device)
        self.blocks = [TransformerBlock(d, q, H, p, n_layers, device) for _ in range(n_layers)]
        self.final_ln = LayerNorm(d, device)
        self.unembedding = Unembedding(d, vocab_size, device)

    def forward(self, token_ids):
        x = self.embedding.forward(token_ids)
        for b in self.blocks:
            x = b.forward(x)
        x = self.final_ln.forward(x)
        return self.unembedding.forward(x)

    def parameters(self):
        p = []
        p.extend(self.embedding.parameters())
        for b in self.blocks:
            p.extend(b.parameters())
        p.extend(self.final_ln.parameters())
        p.extend(self.unembedding.parameters())
        return p

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def step(self, lr):
        with torch.no_grad():
            for p in self.parameters():
                if p.grad is not None:
                    p -= lr * p.grad

    def clip_gradients(self, max_norm=1.0):
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        if not grads:
            return
        total_norm = torch.sqrt(sum((g * g).sum() for g in grads))
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-8)
            for g in grads:
                g.mul_(scale)

    def generate(self, token_ids, max_new_tokens=50, temperature=1.0):
        ids = token_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(ids[-self.c:])
            probs = softmax(logits[:, -1] / temperature, dim=0)
            nxt = torch.multinomial(probs, 1)
            ids = torch.cat([ids, nxt])
        return ids

    def save_weights(self, path):
        data = {
            "config": {
                "vocab_size": self.vocab_size, "d": self.d, "c": self.c,
                "q": self.q, "H": self.H, "p": self.p, "n_layers": self.n_layers,
            },
            "params": [p.detach().cpu() for p in self.parameters()],
        }
        torch.save(data, path)

    @classmethod
    def load_from_file(cls, path, device):
        data = torch.load(path, map_location=device)
        cfg = data["config"]
        m = cls(cfg["vocab_size"], cfg["d"], cfg["c"], cfg["q"], cfg["H"], cfg["p"], cfg["n_layers"], device)
        with torch.no_grad():
            for p, saved in zip(m.parameters(), data["params"]):
                p.copy_(saved.to(device))
        return m
