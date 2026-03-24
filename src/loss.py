import torch


def cross_entropy_loss(logits, targets):
    # logits: (V, s), targets: (s,)
    return torch.nn.functional.cross_entropy(logits.transpose(0, 1), targets)
