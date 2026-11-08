"""R1 (TRAIN only): structure of the payout-size rule.
(a) Does the size-rule lag relate to the machine's win-rule pair (a, b)?
(b) Detection rate vs number of wins (is non-detection just low power?)
(c) In drift/shared episodes, does the size rule flip along with the win rule?
"""
import json
from collections import Counter
import numpy as np
from common import load_split
from house_edge.history import machine_payouts
from house_edge.rules import PAIRS

comp = json.load(open("../runs/exp4/model.json"))["size_components"]
import sys
q, h, labels, cond = load_split(sys.argv[1] if len(sys.argv) > 1 else "train")
rng = np.random.default_rng(3)
rows = []
for qid, (qq, S) in machine_payouts(q, h).items():
    if cond[qid] == "short_history":
        continue
    W = {t: int(v > 0) for t, v in S.items()}
    T = [t for t in S if all(t - k in S for k in range(1, 9))]
    # dominant win pair from full history
    best = None
    for a, b in PAIRS:
        x = np.array([W[t - a] ^ W[t - b] for t in T]); y = np.array([W[t] for t in T])
        if min(x.sum(), (1 - x).sum()) < 5:
            continue
        d = abs(y[x == 1].mean() - y[x == 0].mean())
        if best is None or d > best[0]:
            best = (d, a, b)
    A, B, _ = comp[qq.machine_type]
    lv = {1: np.log(A[0] / B[0]), 2: np.log(A[1] / B[1]), 5: np.log(A[2] / B[2]), 20: np.log(A[3] / B[3])}
    U = [u for u in T if S[u] > 0]
    L = np.array([lv[S[u]] for u in U])
    eff = {}
    for k in range(1, 9):
        f = np.array([S[u - k] >= 2 for u in U])
        if f.sum() >= 6 and (~f).sum() >= 6:
            eff[k] = (L[f].mean() - L[~f].mean())
    if not eff:
        continue
    kmax = max(eff, key=lambda k: abs(eff[k]))
    # permutation null for the max over 8 lags
    null = []
    for _ in range(30):
        Lp = rng.permutation(L)
        null.append(max(abs(Lp[np.array([S[u - k] >= 2 for u in U])].mean() - Lp[~np.array([S[u - k] >= 2 for u in U])].mean()) for k in eff))
    sig = abs(eff[kmax]) > np.percentile(null, 95)
    # time split: effect of the max lag in first vs second half of history
    mid = np.median(U)
    halves = []
    for part in (np.array(U) < mid, np.array(U) >= mid):
        f = np.array([S[u - kmax] >= 2 for u in U])
        if (f & part).sum() >= 4 and (~f & part).sum() >= 4:
            halves.append(L[f & part].mean() - L[~f & part].mean())
    rows.append(dict(cond=cond[qid], type=qq.machine_type, a=best[1], b=best[2], win_contrast=best[0], kmax=kmax,
                     sig=sig, nwins=len(U), eff=eff[kmax], halves=halves))
n = len(rows)
sig = [r for r in rows if r["sig"]]
print(f"(a) machines {n}; significant size rule {len(sig)} ({len(sig)/n:.2f})")
clear = [r for r in sig if r["win_contrast"] > 0.5]
rel = Counter("k==a (first win lag)" if r["kmax"] == r["a"] else "k==b (second win lag)" if r["kmax"] == r["b"] else "other" for r in clear)
print(f"    among significant machines with a clear win rule (n={len(clear)}): size lag vs win pair: {dict(rel)}")
print("    (chance if unrelated: k==a ~1/8, k==b ~1/8)")
print("    win pair -> size lag table:", sorted(Counter((r['a'], r['b'], r['kmax']) for r in clear).items(), key=lambda x: -x[1])[:8])
print("\n(b) detection rate by number of observed wins:")
for lo, hi in ((0, 60), (60, 90), (90, 120), (120, 400)):
    g = [r for r in rows if lo <= r["nwins"] < hi]
    if g:
        print(f"    wins {lo:3d}-{hi:3d}: n={len(g):3d} detected {np.mean([r['sig'] for r in g]):.2f}")
print("\n(c) significant machines: does the size effect keep its sign between history halves?")
for c in ("stable", "drift", "shared_change"):
    g = [r for r in sig if r["cond"] == c and len(r["halves"]) == 2]
    if g:
        flips = np.mean([np.sign(r["halves"][0]) != np.sign(r["halves"][1]) for r in g])
        print(f"    {c:14s} n={len(g):3d} sign flips between halves: {flips:.2f}")
