#!/usr/bin/env python3
"""NeanderChat — low-level PyTorch training script."""

import os
import sys
import time
import torch

from src.tokenizer import WordTokenizer
from src.transformer import NeanderChat
from src.loss import cross_entropy_loss


device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Parameters (edit here)
# Model
D = 200 # embedding dimension
C = 30 # context length
Q = 128 # query dimension
H = 6 # number of heads
P = 300 # MLP dimension
N_LAYERS = 6 # number of layers

# Training
LR = 0.5*1e-3
EPOCHS = 2
STEPS = 10000
LOG_EVERY = 10
WARMUP = 10
LOG_HOUR = 3600

# Generation
TEMP = 0.7
GEN_LEN = 60
SEED = "the cat sat"

# Files
NAME = "neander"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "train.txt")
LOG_PATH = os.path.join(os.path.dirname(__file__), "log.txt")


def fmt_eta(sec):
    sec = int(max(0, sec))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


if not os.path.exists(DATA_PATH):
    print(f"Missing data file: {DATA_PATH}")
    sys.exit(1)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

tok = WordTokenizer(text)
tokens = torch.tensor(tok.encode(text), dtype=torch.long, device=device)
V = tok.vocab_size
max_start = tokens.shape[0] - C - 1
if max_start < 1:
    print("Not enough tokens for context length C.")
    sys.exit(1)

model = NeanderChat(V, D, C, Q, H, P, N_LAYERS, device=device)
n_params = sum(p.numel() for p in model.parameters())
total_steps = EPOCHS * STEPS

print(f"Device: {device}")
print(f"Parameters: {n_params:,}")
print(f"Planned: {EPOCHS} epochs x {STEPS} steps = {total_steps:,}")

t_w = time.time()
for _ in range(WARMUP):
    i = torch.randint(0, max_start + 1, (1,), device=device).item()
    x = tokens[i:i + C]
    y = tokens[i + 1:i + C + 1]
    model.zero_grad()
    loss = cross_entropy_loss(model.forward(x), y)
    loss.backward()
    model.clip_gradients(1.0)
    model.step(LR)
sec_per_step = (time.time() - t_w) / WARMUP
print(f"Warmup {WARMUP} steps: {sec_per_step*1000:.1f} ms/step")
print(f"Estimated full run: ~{fmt_eta(sec_per_step * total_steps)}")
print("Type yes to start training: ", end="", flush=True)
if input().strip().lower() != "yes":
    print("Aborted.")
    sys.exit(0)

# Re-init to avoid warmup affecting final run
model = NeanderChat(V, D, C, Q, H, P, N_LAYERS, device=device)
t0 = time.time()
next_hour = t0 + LOG_HOUR
loss_hour = 0.0
n_hour = 0

for epoch in range(1, EPOCHS + 1):
    running_loss = 0.0
    running_count = 0
    for step in range(1, STEPS + 1):
        i = torch.randint(0, max_start + 1, (1,), device=device).item()
        x = tokens[i:i + C]
        y = tokens[i + 1:i + C + 1]

        model.zero_grad()
        loss = cross_entropy_loss(model.forward(x), y)
        if not torch.isfinite(loss):
            print(f"NaN/Inf at epoch {epoch} step {step}")
            sys.exit(1)
        loss.backward()
        model.clip_gradients(1.0)
        model.step(LR)

        lv = float(loss.item())
        running_loss += lv
        running_count += 1
        loss_hour += lv
        n_hour += 1

        now = time.time()
        if now >= next_hour:
            avg_hour = loss_hour / max(n_hour, 1)
            gstep = (epoch - 1) * STEPS + step
            line = (
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} | elapsed {fmt_eta(now - t0)} | "
                f"epoch {epoch}/{EPOCHS} step {step}/{STEPS} global {gstep} | "
                f"avg_loss_hour {avg_hour:.6f}\n"
            )
            with open(LOG_PATH, "a", encoding="utf-8") as lf:
                lf.write(line)
            loss_hour, n_hour = 0.0, 0
            next_hour = now + LOG_HOUR

        if step % LOG_EVERY == 0:
            avg = running_loss / running_count
            done = (epoch - 1) * STEPS + step
            eta = (time.time() - t0) / done * (total_steps - done)
            print(f"E{epoch} {step:>7}/{STEPS} | loss {avg:.4f} | ETA {fmt_eta(eta)}")
            running_loss, running_count = 0.0, 0

model.save_weights(f"{NAME}.pt")
print(f"Saved -> {NAME}.pt")

seed_ids = torch.tensor(tok.encode(SEED), dtype=torch.long, device=device)
out_ids = model.generate(seed_ids, max_new_tokens=GEN_LEN, temperature=TEMP)
print("\n>", tok.decode(out_ids.tolist()), "\n")
