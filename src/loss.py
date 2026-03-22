"""
loss.py — Cross-entropy loss for NeanderChat.
"""

import numpy as np
from .utils import softmax


def cross_entropy_loss(logits, targets):
    # logits: (V, s)   targets: (s,)
    probs = softmax(logits, axis=0)                           # (V, s) → (V, s)  softmax over vocab axis
    seq_len = targets.shape[0]                                # scalar s
    probs_safe = np.clip(probs, 1e-12, 1.0)                  # prevent log(0) → -inf → NaN
    loss = -np.log(probs_safe[targets, np.arange(seq_len)]).mean()
    d_logits = probs.copy()                                   # (V, s)
    d_logits[targets, np.arange(seq_len)] -= 1                # (V, s)  subtract 1 at correct classes
    d_logits /= seq_len                                       # (V, s)  average over positions
    return loss, d_logits                                     # scalar, (V, s)
