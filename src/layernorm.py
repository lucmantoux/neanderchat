import torch


class LayerNorm:
    def __init__(self, d, device, eps=1e-5):
        self.gamma = torch.ones(d, device=device, requires_grad=True)
        self.beta = torch.zeros(d, device=device, requires_grad=True)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma[:, None] * x_hat + self.beta[:, None]

    def parameters(self):
        return [self.gamma, self.beta]
