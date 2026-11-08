"""R3 (TRAIN only): is a machine's win rule (pair, orientation) predictable from type, signal or episode-mates?"""
from collections import Counter
import numpy as np
from common import load_split
from house_edge.history import machine_payouts
from house_edge.rules import PAIRS

q, h, labels, cond = load_split("train")
rows = []
for qid, (qq, S) in machine_payouts(q, h).items():
    if cond[qid] != "stable":  # clean, unflipped rules
        continue
    W = {t: int(v > 0) for t, v in S.items()}
    T = [t for t in S if all(t - k in S for k in range(1, 9))]
    best = None
    for a, b in PAIRS:
        x = np.array([W[t - a] ^ W[t - b] for t in T]); y = np.array([W[t] for t in T])
        if min(x.sum(), (1 - x).sum()) < 5:
            continue
        r1, r0 = y[x == 1].mean(), y[x == 0].mean()
        if best is None or abs(r1 - r0) > best[0]:
            best = (abs(r1 - r0), (a, b), int(r1 > r0), max(r1, r0), min(r1, r0))
    if best and best[0] > 0.5:
        rows.append(dict(ep=qq.episode_id, type=qq.machine_type, signal=qq.signal, pair=best[1], xor=best[2], hi=best[3], lo=best[4]))
print(f"stable train machines with a clear rule: {len(rows)}")
print("pair share overall:", dict(Counter(r["pair"] for r in rows).most_common(5)))
for key in ("type", "signal"):
    print(f"\npair share by {key}:")
    for v in sorted({r[key] for r in rows}):
        g = [r for r in rows if r[key] == v]
        c = Counter(r["pair"] for r in g)
        print(f"   {v!s:9s} n={len(g):3d} " + " ".join(f"{p}:{c[p]/len(g):.2f}" for p in ((1, 3), (2, 5), (4, 7))))
    print(f"XOR (differ->win) share by {key}:", {v: round(np.mean([r['xor'] for r in rows if r[key] == v]), 3) for v in sorted({r[key] for r in rows})})
print("\nhigh/low cell levels by type x signal (mean):")
for t in ("balanced", "jackpot", "steady"):
    print("  ", t, {s: (round(np.mean([r['hi'] for r in rows if r['type'] == t and r['signal'] == s]), 3),
                         round(np.mean([r['lo'] for r in rows if r['type'] == t and r['signal'] == s]), 3)) for s in (-1, 0, 1)})
# episode-mates: do machines in the same episode share pairs / orientations more than chance?
eps = {}
for r in rows:
    eps.setdefault(r["ep"], []).append(r)
same_pair = same_or = tot = 0
for g in eps.values():
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            tot += 1; same_pair += g[i]["pair"] == g[j]["pair"]; same_or += g[i]["xor"] == g[j]["xor"]
pc = Counter(r["pair"] for r in rows); base_pair = sum((v / len(rows)) ** 2 for v in pc.values())
xr = np.mean([r["xor"] for r in rows]); base_or = xr ** 2 + (1 - xr) ** 2
print(f"\nwithin-episode pairs of machines: same pair {same_pair/tot:.3f} (chance {base_pair:.3f}); same orientation {same_or/tot:.3f} (chance {base_or:.3f})")
