"""
train.py — Training loop for NeanderChat.

WHAT THIS DOES:
1. Load data, build tokenizer and model.
2. For each epoch:
   a. Slide a window of size (c+1) across the token sequence.
      - Input:  tokens[i : i+c]     (the context)
      - Target: tokens[i+1 : i+c+1] (shifted by one — the next word at each position)
   b. Forward pass → logits.
   c. Compute cross-entropy loss.
   d. Backward pass → gradients in every layer.
   e. Update all weights with SGD: W -= lr * d_W

WHAT YOU NEED TO IMPLEMENT:
- The training loop itself.
- A function that collects all parameters and applies SGD updates.
- Print loss every N steps to see if it's decreasing.

TIPS:
- Start with lr = 0.01, adjust if loss explodes or doesn't move.
- The loss for random weights should be around -log(1/vocab_size) ≈ log(192) ≈ 5.26
- If it goes below 3.0, the model is learning something.



Uses DEFAULTS from parameters.py (d=64, c=16, etc.).


"""

import os
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from parameters import DEFAULTS
from src.tokenizer import WordTokenizer
from src.transformer import NeanderChat
from src.loss import cross_entropy_loss


def train(defaults=DEFAULTS, num_epochs=10, lr=0.001, print_every=100):
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "train.txt")
    raw_text = open(data_path, "r").read()
    tokenizer = WordTokenizer(raw_text)
    tokens = tokenizer.encode(raw_text)
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")

    d = defaults["d"]
    c = defaults.get("c", defaults.get("s", 16))
    q = defaults["q"]
    H = defaults["H"]
    p = defaults["p"]
    n_layers = defaults["n_layers"]

    model = NeanderChat(vocab_size, d, c, q, H, p, n_layers)
    num_tokens = len(tokens)
    num_windows = num_tokens - c

    for epoch in range(num_epochs):
        running_loss = 0.0
        batch_count = 0
        for step in range(num_windows):
            input_ids = tokens[step : step + c]
            target_ids = tokens[step + 1 : step + c + 1]

            logits = model.forward(input_ids)                    # (V, s)
            loss, d_logits = cross_entropy_loss(logits, target_ids)

            running_loss += loss
            batch_count += 1

            model.backward(d_logits)
            model.clip_gradients(max_norm=1.0)
            model.update(lr)

            if not np.isfinite(loss):
                print(f"  *** NaN/Inf loss at epoch {epoch+1} step {step+1} — stopping.")
                return model

            if (step + 1) % print_every == 0:
                avg_loss = running_loss / batch_count
                print(f"Epoch {epoch+1}, Step {step+1}/{num_windows}, Avg Loss: {avg_loss:.4f}")
                running_loss = 0.0
                batch_count = 0

    return model
            

if __name__ == "__main__":
    train()