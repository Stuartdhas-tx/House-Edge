"""HISTORIES ONLY (train + validation inputs; no targets; condition labels used to keep stable machines): are a machine's high- and low-cell win rates correlated
beyond the type x signal means? Binomial noise is independent across cells, so any correlation of the empirical
rates is correlation of the true levels; if strong, a joint (machine-level) prior could pool the two cells."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from house_edge.history import machine_payouts  # noqa: E402
from house_edge.rules import PAIRS  # noqa: E402
from slot_forecasting.io import load_inputs, read_rows  # noqa: E402

for split in ("train", "validation"):
    q, h = load_inputs(ROOT / "public" / split)
    cond = {r['query_id']: r['condition'] for r in read_rows(ROOT / 'public' / split / 'conditions.csv', ('query_id', 'condition'))}
    rows = []
    for qid, (qq, S) in machine_payouts(q, h).items():
        if cond[qid] != 'stable':  # analysis only: rule flips swap hi/lo between halves
            continue
        W = {u: int(v > 0) for u, v in S.items()}
        ticks = [u for u in W if all(u - k in W for k in range(1, 9))]
        if len(ticks) < 120:
            continue
        mid = sorted(ticks)[len(ticks) // 2]
        best = None
        for a, b in PAIRS:
            x = np.array([W[u - a] ^ W[u - b] for u in ticks]); y = np.array([W[u] for u in ticks])
            if min(x.sum(), (1 - x).sum()) >= 10:
                r1, r0 = y[x == 1].mean(), y[x == 0].mean()
                if best is None or abs(r1 - r0) > best[0]:
                    best = (abs(r1 - r0), a, b)
        if best is None or best[0] < 0.4:
            continue
        _, a, b = best
        # split-half: identify hi/lo cell on the first half, measure both cell rates on the second half
        x = {u: W[u - a] ^ W[u - b] for u in ticks}
        first = [u for u in ticks if u < mid]; second = [u for u in ticks if u >= mid]
        r = {c: np.mean([W[u] for u in first if x[u] == c] or [np.nan]) for c in (0, 1)}
        hi_c = 1 if r[1] > r[0] else 0
        hi = np.mean([W[u] for u in second if x[u] == hi_c] or [np.nan]); lo = np.mean([W[u] for u in second if x[u] != hi_c] or [np.nan])
        rows.append((qq.machine_type, qq.signal, hi, lo))
    for t in ("balanced", "jackpot", "steady"):
        sel = [(s, hi, lo) for tt, s, hi, lo in rows if tt == t and not np.isnan(hi) and not np.isnan(lo)]
        if len(sel) < 10:
            continue
        s, hi, lo = map(np.array, zip(*sel))
        # remove type x signal means before correlating
        for v in (-1, 0, 1):
            m = s == v
            hi[m] -= hi[m].mean(); lo[m] -= lo[m].mean()
        print(f"{split:10s} {t:8s} n={len(sel):3d} corr(hi, lo | type, signal) = {np.corrcoef(hi, lo)[0, 1]:+.2f}  "
              f"sd hi {hi.std():.3f} sd lo {lo.std():.3f}")
