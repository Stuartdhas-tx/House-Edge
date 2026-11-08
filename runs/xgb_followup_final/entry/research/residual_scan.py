"""DIAGNOSTIC (validation look #9 cont.): does anything observable at the cutoff explain EXP-6's win-probability
residual logit(true w) - logit(our w)? Per type OLS on lagged wins / big payouts / signal / recent win rate."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from house_edge.history import machine_payouts  # noqa: E402
from slot_forecasting.io import load_inputs, read_rows, read_submission, read_targets  # noqa: E402

P = ROOT / "public/validation"
queries, history = load_inputs(P)
M = machine_payouts(queries, history)
T = {t.query_id: np.array(t.probabilities) for t in read_targets(P / "targets.csv")}
C = {r["query_id"]: r["condition"] for r in read_rows(P / "conditions.csv", ("query_id", "condition"))}
pred = read_submission(ROOT / "runs/exp6/validation_predictions.csv", set(M))
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
names = [f"win{k}" for k in range(1, 9)] + [f"big{k}" for k in range(1, 9)] + [f"j20_{k}" for k in range(1, 9)] + \
        ["sig-1", "sig+1", "recent16-all", "n_obs/100", "cutoff/100"]
for t in ("balanced", "jackpot", "steady"):
    X, y, hi = [], [], []
    for qid, (q, S) in M.items():
        if q.machine_type != t:
            continue
        c = q.cutoff
        f = [(S[c - k] > 0) if c - k in S else 0.5 for k in range(1, 9)]
        f += [(S[c - k] >= 2) if c - k in S else 0.0 for k in range(1, 9)]
        f += [(S[c - k] == 20) if c - k in S else 0.0 for k in range(1, 9)]
        rec = [S[u] > 0 for u in S if u >= c - 16]
        allw = [v > 0 for v in S.values()]
        f += [q.signal == -1, q.signal == 1, (np.mean(rec) if rec else 0) - np.mean(allw), len(S) / 100, c / 100]
        X.append([1.0] + [float(v) for v in f]); y.append(lg(1 - T[qid][0]) - lg(1 - pred[qid][0])); hi.append(1 - T[qid][0] > 0.3)
    X, y = np.array(X), np.array(y)
    for label, m in (("all", np.ones(len(y), bool)), ("hi-cell", np.array(hi)), ("lo-cell", ~np.array(hi))):
        b, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        r = y[m] - X[m] @ b
        cov = np.linalg.pinv(X[m].T @ X[m]) * r.var(ddof=X.shape[1])
        tstat = b / np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        top = sorted(zip(names, b[1:], tstat[1:]), key=lambda z: -abs(z[2]))[:5]
        print(f"{t:8s} {label:7s} n={m.sum():3d} resid sd {y[m].std():.3f} R2 {1 - r.var() / y[m].var():.2f} | "
              + "  ".join(f"{n} {bb:+.2f}(t={tt:+.1f})" for n, bb, tt in top))
