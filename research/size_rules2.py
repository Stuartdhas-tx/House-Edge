"""TRAIN-only characterisation: which lag feature families drive the size regime, and how sharply."""
import json
from collections import Counter
import numpy as np
from common import load_split
from house_edge.history import machine_payouts

comp = json.load(open("../runs/exp4/model.json"))["size_components"]
q, h, labels, cond = load_split("train")
rng = np.random.default_rng(2)
fam_counts, lag_counts, cells = Counter(), Counter(), {}
n_sig = n_all = 0
for qid, (qq, S) in machine_payouts(q, h).items():
    A, B, _ = comp[qq.machine_type]
    lv = {1: np.log(A[0] / B[0]), 2: np.log(A[1] / B[1]), 5: np.log(A[2] / B[2]), 20: np.log(A[3] / B[3])}
    U = [u for u in S if S[u] > 0 and all(u - k in S for k in range(1, 9))]
    if len(U) < 40:
        continue
    n_all += 1
    L = np.array([lv[S[u]] for u in U]); size = np.array([S[u] for u in U])
    feats = {}
    for k in range(1, 9):
        for fam, fn in (("win", lambda v: v > 0), ("ge2", lambda v: v >= 2), ("ge5", lambda v: v >= 5), ("eq1", lambda v: v == 1)):
            feats[(fam, k)] = np.array([fn(S[u - k]) for u in U])
    def max_eff(Lv):
        return max(((abs(Lv[f].mean() - Lv[~f].mean()), key) for key, f in feats.items() if f.sum() >= 8 and (~f).sum() >= 8), default=(0, None))
    obs, key = max_eff(L)
    null95 = np.percentile([max_eff(rng.permutation(L))[0] for _ in range(20)], 95)
    if obs > null95 and key:
        n_sig += 1
        fam_counts[key[0]] += 1; lag_counts[key[1]] += 1
        f = feats[key]
        d1 = [np.mean(size[f] == v) for v in (1, 2, 5, 20)]; d0 = [np.mean(size[~f] == v) for v in (1, 2, 5, 20)]
        hi, lo = (d1, d0) if d1[0] > d0[0] else (d0, d1)
        cells.setdefault(qq.machine_type, []).append((hi, lo))
print(f"significant machines {n_sig}/{n_all}")
print("top feature family:", fam_counts.most_common(), "\ntop feature lag:", sorted(lag_counts.items()))
for t, v in sorted(cells.items()):
    hi = np.mean([x[0] for x in v], 0); lo = np.mean([x[1] for x in v], 0)
    print(f"{t:9s} n={len(v):3d}  size dist in 'more 1x' cell {np.round(hi,3).tolist()}  vs other cell {np.round(lo,3).tolist()}")
