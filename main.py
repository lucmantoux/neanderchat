#!/usr/bin/env python3
"""NeanderChat — train and generate, pure Python + NumPy."""

import os, sys, time
import numpy as np

from src.tokenizer import WordTokenizer
from src.transformer import NeanderChat
from src.loss import cross_entropy_loss
from src.utils import softmax

# ═══════════════════════════════════════════════════════════
#  PARAMETERS — edit these
# ═══════════════════════════════════════════════════════════
D         = 500       # embedding dimension
C         = 60        # context length
Q         = 128       # key / query dim per head
H         = 8         # attention heads
P         = 600       # MLP projection dim
N_LAYERS  = 8         # transformer blocks

LR        = 0.001     # learning rate
EPOCHS    = 10         # training epochs
STEPS     = 2000000      # steps per epoch
LOG_EVERY = 50        # print every N steps

TEMP      = 0.8       # generation temperature
GEN_LEN   = 60        # tokens to generate
SEED      = "the cat sat"

NAME      = "neander"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "train.txt")
LOG_PATH  = os.path.join(os.path.dirname(__file__), "log.txt")
WARMUP    = 10        # steps to time before asking confirmation
LOG_HOUR  = 3600      # append to log.txt every N seconds (1 hour)

# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════

def fmt_eta(sec):
    sec = int(max(0, sec))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def train_step(m, toks, ms, lr):
    i = np.random.randint(0, ms + 1)
    x, y = toks[i : i + C], toks[i + 1 : i + C + 1]
    loss, dL = cross_entropy_loss(m.forward(x), y)
    if not np.isfinite(loss):
        return float("nan")
    m.backward(dL)
    m.clip_gradients(1.0)
    m.update(lr)
    return loss


# ═══════════════════════════════════════════════════════════
#  SETUP
# ═══════════════════════════════════════════════════════════
with open(DATA_PATH) as f:
    text = f.read()

tok    = WordTokenizer(text)
tokens = tok.encode(text)
V      = tok.vocab_size

model  = NeanderChat(V, D, C, Q, H, P, N_LAYERS)

# ── uncomment to load a saved model instead of training ──
# model.load_weights(f"{NAME}.npz"); print(f"Loaded {NAME}.npz")

n_params = sum(v.size for l in model._all_layers()
               for k, v in vars(l).items()
               if isinstance(v, np.ndarray)
               and (k.startswith(("W_", "b_")) or k in ("gamma", "beta")))

total_steps = EPOCHS * STEPS
max_start = len(tokens) - C - 1
if max_start < 1:
    print("Not enough tokens for context length C."); sys.exit(1)

print("─" * 56)
print(f"  Parameters:  {n_params:,}")
print(f"  Vocab:       {V}     Corpus tokens: {len(tokens):,}")
print(f"  Planned:     {EPOCHS} epochs × {STEPS} steps = {total_steps:,} steps")
print("─" * 56)
print(f"\n  Warmup: {WARMUP} steps (timing only, model reset after)…")

t_w = time.time()
for _ in range(WARMUP):
    train_step(model, tokens, max_start, LR)
warmup_sec = time.time() - t_w
sec_per_step = warmup_sec / WARMUP
est_total = sec_per_step * total_steps

print(f"  → {sec_per_step*1000:.1f} ms/step   estimated full run: ~{fmt_eta(est_total)}")
print("\n  Type yes and Enter to start training, anything else to abort: ", end="", flush=True)
if input().strip().lower() != "yes":
    print("Aborted.")
    sys.exit(0)

model = NeanderChat(V, D, C, Q, H, P, N_LAYERS)
print("\n  Training…\n")

# ═══════════════════════════════════════════════════════════
#  TRAIN
# ═══════════════════════════════════════════════════════════
t0 = time.time()
next_log = t0 + LOG_HOUR
loss_h, n_h = 0.0, 0

for epoch in range(1, EPOCHS + 1):
    rl, bc = 0.0, 0
    for step in range(1, STEPS + 1):
        loss = train_step(model, tokens, max_start, LR)
        if not np.isfinite(loss):
            print(f"  *** NaN at epoch {epoch} step {step}"); break

        rl += loss; bc += 1
        loss_h += loss
        n_h += 1

        now = time.time()
        if now >= next_log:
            gs = (epoch - 1) * STEPS + step
            avg_h = loss_h / n_h if n_h else 0.0
            line = (
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"elapsed {fmt_eta(now - t0)} | "
                f"epoch {epoch}/{EPOCHS} step {step}/{STEPS} global {gs} | "
                f"avg_loss_hour {avg_h:.6f}\n"
            )
            with open(LOG_PATH, "a", encoding="utf-8") as lf:
                lf.write(line)
            print(f"  [log] {LOG_PATH} ← hourly snapshot")
            loss_h, n_h = 0.0, 0
            next_log = now + LOG_HOUR

        if step % LOG_EVERY == 0:
            avg   = rl / bc
            done  = (epoch - 1) * STEPS + step
            total = EPOCHS * STEPS
            eta   = (time.time() - t0) / done * (total - done)
            print(f"  E{epoch} {step:>5}/{STEPS} | loss {avg:.4f} | ETA {fmt_eta(eta)}")
            rl, bc = 0.0, 0

model.save_weights(f"{NAME}.npz")
elapsed = time.time() - t0
m, s = divmod(int(elapsed), 60)
print(f"\nSaved → {NAME}.npz  ({m}m{s:02d}s total)")

# ═══════════════════════════════════════════════════════════
#  GENERATE
# ═══════════════════════════════════════════════════════════
ids = tok.encode(SEED)
for _ in range(GEN_LEN):
    logits = model.forward(ids[-C:])
    probs  = softmax(logits[:, -1] / TEMP)
    ids    = np.append(ids, np.random.choice(V, p=probs))

print(f"\n> {tok.decode(ids)}\n")
