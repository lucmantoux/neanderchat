import torch


class Embedding:
    def __init__(self, vocab_size, d, c, device):
        self.W_E = torch.randn(d, vocab_size, device=device) * 0.02
        self.W_E.requires_grad_(True)
        self.W_pos = torch.randn(d, c, device=device) * 0.02
        self.W_pos.requires_grad_(True)

    def forward(self, token_ids):
        s = token_ids.shape[0]
        return self.W_E[:, token_ids] + self.W_pos[:, :s]

    def parameters(self):
        return [self.W_E, self.W_pos]


class Unembedding:
    def __init__(self, d, vocab_size, device):
        self.W_U = torch.randn(vocab_size, d, device=device) * 0.02
        self.W_U.requires_grad_(True)
        self.b_U = torch.zeros(vocab_size, device=device, requires_grad=True)

    def forward(self, x):
        return self.W_U @ x + self.b_U[:, None]

    def parameters(self):
        return [self.W_U, self.b_U]
