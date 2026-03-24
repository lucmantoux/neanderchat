import torch


def softmax(x, dim=-1):
    return torch.softmax(x, dim=dim)


def gelu(x):
    return torch.nn.functional.gelu(x)
