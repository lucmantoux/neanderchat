import torch
from .utils import softmax


class AttentionHead:
    def __init__(self, d, q, n_layers, device):
        residual_scale = 0.02 / (2 * n_layers) ** 0.5
        self.W_Q = (torch.randn(q, d, device=device) * 0.02).requires_grad_()
        self.b_Q = torch.zeros(q, device=device, requires_grad=True)
        self.W_K = (torch.randn(q, d, device=device) * 0.02).requires_grad_()
        self.b_K = torch.zeros(q, device=device, requires_grad=True)
        self.W_V_down = (torch.randn(q, d, device=device) * 0.02).requires_grad_()
        self.b_V_down = torch.zeros(q, device=device, requires_grad=True)
        self.W_V_up = (torch.randn(d, q, device=device) * residual_scale).requires_grad_()
        self.b_V_up = torch.zeros(d, device=device, requires_grad=True)
        self.q = q
        self.device = device

    def forward(self, x):
        s = x.shape[1]
        Q = self.W_Q @ x + self.b_Q[:, None]
        K = self.W_K @ x + self.b_K[:, None]
        scores = (Q.transpose(0, 1) @ K) / (self.q ** 0.5)
        mask = torch.triu(torch.full((s, s), float("-inf"), device=self.device), diagonal=1)
        A = softmax(scores + mask, dim=-1)
        V_low = self.W_V_down @ x + self.b_V_down[:, None]
        V = self.W_V_up @ V_low + self.b_V_up[:, None]
        return V @ A.transpose(0, 1)

    def parameters(self):
        return [
            self.W_Q, self.b_Q, self.W_K, self.b_K,
            self.W_V_down, self.b_V_down, self.W_V_up, self.b_V_up
        ]


class MultiHeadAttention:
    def __init__(self, d, q, H, n_layers, device):
        self.heads = [AttentionHead(d, q, n_layers, device) for _ in range(H)]
        self.b_out = torch.zeros(d, device=device, requires_grad=True)

    def forward(self, x):
        out = sum(h.forward(x) for h in self.heads)
        return out + self.b_out[:, None]

    def parameters(self):
        p = [self.b_out]
        for h in self.heads:
            p.extend(h.parameters())
        return p
