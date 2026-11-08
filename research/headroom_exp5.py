"""DIAGNOSTIC (validation look #9; nothing is fit or selected here). Where does EXP-5/EXP-6 error live, and what do
the simulator's true probabilities look like (w = 1 - p_0 and size = p_{1,2,5,20} / w)?"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slot_forecasting.io import read_queries, read_rows, read_submission, read_targets  # noqa: E402

P = ROOT / "public/validation"
Q = {q.query_id: q for q in read_queries(P / "queries.csv")}
ids = sorted(Q)
T = {t.query_id: np.array(t.probabilities) for t in read_targets(P / "targets.csv")}
C = {r["query_id"]: r["condition"] for r in read_rows(P / "conditions.csv", ("query_id", "condition"))}
REF = {k: np.array(v) for k, v in read_submission(P / "reference.csv", set(ids)).items()}
CONDS = ("stable", "short_history", "drift", "shared_change")


def points(pred):
    ep = defaultdict(lambda: [None, [], []])
    for k in ids:
        e = ep[Q[k].episode_id]; e[0] = C[k]
        e[1].append(((pred[k] - T[k]) ** 2).sum()); e[2].append(((REF[k] - T[k]) ** 2).sum())
    r = {c: np.mean([np.mean(v[1]) for v in ep.values() if v[0] == c]) / np.mean([np.mean(v[2]) for v in ep.values() if v[0] == c]) for c in CONDS}
    return 60 / (1 + np.mean(list(r.values()))), r


def ws(p):
    return 1 - p[0], p[1:] / max(1 - p[0], 1e-12)


for run in ("exp5", "exp6"):
    pred = {k: np.array(v) for k, v in read_submission(ROOT / f"runs/{run}/validation_predictions.csv", set(ids)).items()}
    rows = {}
    rows["as is"] = points(pred)
    rows["true w, our size"] = points({k: np.r_[1 - ws(T[k])[0], ws(T[k])[0] * ws(pred[k])[1]] for k in ids})
    rows["our w, true size"] = points({k: np.r_[1 - ws(pred[k])[0], ws(pred[k])[0] * ws(T[k])[1]] for k in ids})
    print(f"\n{run}")
    for name, (pts, r) in rows.items():
        print(f"  {name:18s} {pts:6.3f}  " + " ".join(f"{c[:6]} {r[c]:.4f}" for c in CONDS))
    # squared-error split by component and condition
    for c in CONDS:
        ks = [k for k in ids if C[k] == c]
        ew = np.mean([(ws(pred[k])[0] - ws(T[k])[0]) ** 2 for k in ks])
        print(f"  {c:14s} mean |w err| {np.mean([abs(ws(pred[k])[0] - ws(T[k])[0]) for k in ks]):.4f}  (w err)^2 {ew:.5f}")

print("\nTRUE w by type (quantiles 5/25/50/75/95) and most common rounded values:")
for t in ("balanced", "jackpot", "steady"):
    w = np.array([ws(T[k])[0] for k in ids if Q[k].machine_type == t])
    print(f"  {t:8s} n={len(w)} q={np.round(np.quantile(w, [.05, .25, .5, .75, .95]), 3).tolist()} top={Counter(np.round(w, 2)).most_common(8)}")
print("\nTRUE size vectors by type: distinct (rounded 3dp) count and most common:")
for t in ("balanced", "jackpot", "steady"):
    s = [tuple(np.round(ws(T[k])[1], 3)) for k in ids if Q[k].machine_type == t]
    cnt = Counter(s)
    print(f"  {t:8s} distinct={len(cnt)} top={cnt.most_common(4)}")

print("\nEXP-6 w error: share of squared error from big misses (|err|>0.25) vs level error, by condition")
pred = {k: np.array(v) for k, v in read_submission(ROOT / "runs/exp6/validation_predictions.csv", set(ids)).items()}
for c in CONDS:
    e = np.array([ws(pred[k])[0] - ws(T[k])[0] for k in ids if C[k] == c])
    big = np.abs(e) > 0.25
    print(f"  {c:14s} big misses {big.sum():3d}/{len(e)} carry {(e[big] ** 2).sum() / (e ** 2).sum():.2f} of sq err; "
          f"small-err RMS {np.sqrt((e[~big] ** 2).mean()):.4f}")
print("\nTRUE w within cell vs signal (cell from true w: hi if w>0.3 for balanced/steady, >0.2 jackpot):")
for t in ("balanced", "jackpot", "steady"):
    for cell in ("hi", "lo"):
        out = []
        for s in (-1, 0, 1):
            w = np.array([ws(T[k])[0] for k in ids if Q[k].machine_type == t and Q[k].signal == s])
            thr = 0.2 if t == "jackpot" else 0.3
            w = w[w > thr] if cell == "hi" else w[w <= thr]
            out.append(f"s={s:+d} n={len(w):3d} mean {w.mean():.3f} sd {w.std():.3f}")
        print(f"  {t:8s} {cell}: " + " | ".join(out))
