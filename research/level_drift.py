"""DIAGNOSTIC (validation look #9 cont.): do cell levels drift? For each validation machine, take its clearest win pair
over the whole history (sign-agnostic contrast), find the cell it is in at the cutoff, and compare the empirical win
rate of THAT cell over the last N ticks with the true w. If short windows beat the full history, levels drift."""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from house_edge.history import machine_payouts  # noqa: E402
from house_edge.rules import PAIRS  # noqa: E402
from slot_forecasting.io import load_inputs, read_rows, read_targets  # noqa: E402

P = ROOT / "public/validation"
q, h = load_inputs(P)
M = machine_payouts(q, h)
T = {t.query_id: 1 - t.probabilities[0] for t in read_targets(P / "targets.csv")}
C = {r["query_id"]: r["condition"] for r in read_rows(P / "conditions.csv", ("query_id", "condition"))}
WINDOWS = (24, 48, 96, 160, 10_000)
err = defaultdict(lambda: defaultdict(list))
for qid, (qq, S) in M.items():
    W = {u: int(v > 0) for u, v in S.items()}
    ticks = sorted(u for u in W if all(u - k in W for k in range(1, 9)))
    if len(ticks) < 60:
        continue
    best = None
    for a, b in PAIRS:
        x = np.array([W[u - a] ^ W[u - b] for u in ticks]); y = np.array([W[u] for u in ticks])
        if min(x.sum(), (1 - x).sum()) >= 5:
            con = abs(y[x == 1].mean() - y[x == 0].mean())
            if best is None or con > best[0]:
                best = (con, a, b)
    c = qq.cutoff
    if best is None or best[0] < 0.5 or (c - best[1]) not in W or (c - best[2]) not in W:
        continue
    _, a, b = best
    cell = W[c - a] ^ W[c - b]
    for n in WINDOWS:
        sel = [W[u] for u in ticks if u >= c - n and (W[u - a] ^ W[u - b]) == cell]
        if len(sel) >= 5:
            for key in (C[qid], f"{qq.machine_type}"):
                err[key][n].append((np.mean(sel) - T[qid]) ** 2)
for key in ("stable", "drift", "shared_change", "short_history", "balanced", "jackpot", "steady"):
    row = err[key]
    common = min(len(row[n]) for n in WINDOWS) if all(row[n] for n in WINDOWS) else 0
    print(f"{key:14s} " + "  ".join(f"last{n if n < 10_000 else 'ALL'}: rmse {np.sqrt(np.mean(row[n])):.4f} (n={len(row[n])})" for n in WINDOWS))
