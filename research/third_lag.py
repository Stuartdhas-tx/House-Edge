"""TRAIN-only: within the dominant XOR cell, does another lag's win (or a big payout) shift P(win)?"""
import numpy as np
from common import load_split
from house_edge.history import machine_payouts
from house_edge.rules import PAIRS
q, h, labels, cond = load_split("train")
rows = []
for qid, (qq, S) in machine_payouts(q, h).items():
    if cond[qid] != "stable":
        continue
    W = {t: int(v > 0) for t, v in S.items()}
    ticks = [t for t in S if all(t - k in S for k in range(1, 9))]
    best = None
    for a, b in PAIRS:
        x = np.array([W[t - a] ^ W[t - b] for t in ticks]); y = np.array([W[t] for t in ticks])
        d = abs(y[x == 1].mean() - y[x == 0].mean())
        if best is None or d > best[0]: best = (d, a, b)
    _, a, b = best
    x = np.array([W[t - a] ^ W[t - b] for t in ticks]); y = np.array([W[t] for t in ticks])
    for k in range(1, 9):
        if k in (a, b): continue
        z = np.array([W[t - k] for t in ticks]); big = np.array([S[t - k] >= 5 for t in ticks])
        for cell in (0, 1):
            m = x == cell
            if (m & (z == 1)).sum() >= 5 and (m & (z == 0)).sum() >= 5:
                rows.append((qq.machine_type, k, cell, y[m & (z == 1)].mean() - y[m & (z == 0)].mean(),
                             y[m & big].mean() - y[m & ~big].mean() if (m & big).sum() >= 3 else np.nan))
import pandas as pd
R = pd.DataFrame(rows, columns=["type", "k", "cell", "d_win", "d_big"])
print("mean within-cell effect of lag-k win on P(win), stable train machines (signed, then |.|):")
print(R.groupby("k").d_win.agg(["mean", lambda v: v.abs().mean(), "count"]).round(3).T)
print("\nby type, signed:"); print(R.pivot_table(index="type", columns="k", values="d_win").round(3))
print("\nwithin-cell effect of lag-k payout >= 5 (vs not), signed:"); print(R.groupby("k").d_big.mean().round(3).to_dict())
