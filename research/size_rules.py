"""TRAIN-only: is the winning-size regime driven by lagged payout SIZES (not just wins)?

Per machine, among winning ticks with lags 1..8 observed, compare the per-win size log-likelihood ratio
(train EM components) across lag features; max effect vs a within-machine permutation null.
"""
import json, random
import numpy as np
from common import load_split
from house_edge.history import machine_payouts
from house_edge.rules import PAIRS

comp = json.load(open("../runs/exp4/model.json"))["size_components"]
q, h, labels, cond = load_split("train")
rng = np.random.default_rng(1)
rows, top = [], {}
signed = {}
for qid, (qq, S) in machine_payouts(q, h).items():
    A, B, _ = comp[qq.machine_type]
    llr = {1: np.log(A[0] / B[0]), 2: np.log(A[1] / B[1]), 5: np.log(A[2] / B[2]), 20: np.log(A[3] / B[3])}
    U = [u for u in S if S[u] > 0 and all(u - k in S for k in range(1, 9))]
    if len(U) < 40:
        continue
    L = np.array([llr[S[u]] for u in U])
    feats = {}
    for k in range(1, 9):
        feats[f"win{k}"] = np.array([S[u - k] > 0 for u in U])
        feats[f"ge2_{k}"] = np.array([S[u - k] >= 2 for u in U])
        feats[f"ge5_{k}"] = np.array([S[u - k] >= 5 for u in U])
        feats[f"one{k}"] = np.array([S[u - k] == 1 for u in U])
    for a, b in PAIRS:
        feats[f"xwin{a}{b}"] = feats[f"win{a}"] ^ feats[f"win{b}"]
        feats[f"xge2_{a}{b}"] = feats[f"ge2_{a}"] ^ feats[f"ge2_{b}"]
    def max_eff(Lv):
        best, arg = 0.0, None
        for name, f in feats.items():
            if f.sum() >= 8 and (~f).sum() >= 8:
                e = abs(Lv[f].mean() - Lv[~f].mean())
                if e > best:
                    best, arg = e, name
        return best, arg
    obs, arg = max_eff(L)
    null = [max_eff(rng.permutation(L))[0] for _ in range(20)]
    rows.append((cond[qid], obs, np.mean(null), np.percentile(null, 95), arg))
    for name, f in feats.items():
        if f.sum() >= 8 and (~f).sum() >= 8:
            signed.setdefault(name, []).append(L[f].mean() - L[~f].mean())
obs = np.array([r[1] for r in rows]); nm = np.array([r[2] for r in rows]); n95 = np.array([r[3] for r in rows])
print(f"machines tested: {len(rows)}")
print(f"max lag-feature effect on size LLR: observed median {np.median(obs):.3f} vs permutation-null median {np.median(nm):.3f}")
print(f"share of machines above their own null 95th pct: {(obs > n95).mean():.3f} (expected ~0.05 if no effect)")
from collections import Counter
print("most common top features:", Counter(r[4] for r in rows).most_common(6))
s = {k: (np.mean(v), np.std(v) / np.sqrt(len(v))) for k, v in signed.items()}
print("largest pooled signed effects (mean ± se):", [(k, round(m, 3), round(e, 3)) for k, (m, e) in sorted(s.items(), key=lambda kv: -abs(kv[1][0] / kv[1][1]))[:6]])
