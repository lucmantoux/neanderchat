import torch
from .utils import gelu


class MLP:
    def __init__(self, d, p, n_layers, device):
        residual_scale = 0.02 / (2 * n_layers) ** 0.5
        self.W_up = (torch.randn(p, d, device=device) * 0.02).requires_grad_()
        self.b_up = torch.zeros(p, device=device, requires_grad=True)
        self.W_down = (torch.randn(d, p, device=device) * residual_scale).requires_grad_()
        self.b_down = torch.zeros(d, device=device, requires_grad=True)

    def forward(self, x):
        h = gelu(self.W_up @ x + self.b_up[:, None])
        out = self.W_down @ h + self.b_down[:, None]
        return out + x

    def parameters(self):
        return [self.W_up, self.b_up, self.W_down, self.b_down]
