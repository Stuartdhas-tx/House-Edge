"""DIAGNOSTIC (validation look #6; nothing is fit or selected here).

1) Where does EXP-4's remaining validation error live: win probability w or payout-size distribution s?
2) How large must a validation difference be to exceed episode-sampling noise?
3) How well have train-CV Brier gains predicted realized validation gains so far?
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slot_forecasting.io import read_queries, read_rows, read_submission, read_targets

P = ROOT / "public/validation"
queries = {q.query_id: q for q in read_queries(P / "queries.csv")}
ids = set(queries)
targets = {t.query_id: t for t in read_targets(P / "targets.csv")}
cond = {r["query_id"]: r["condition"] for r in read_rows(P / "conditions.csv", ("query_id", "condition"))}
ref = read_submission(P / "reference.csv", ids)
runs = {r: read_submission(ROOT / f"runs/{r}/validation_predictions.csv", ids) for r in ("exp1", "exp2", "exp3", "exp4")}
model = json.loads((ROOT / "runs/exp4/model.json").read_text())
CONDS = ("stable", "short_history", "drift", "shared_change")


def sq(p, q):
    return sum((a - b) ** 2 for a, b in zip(p, q))


def per_episode(pred):
    ep = {}
    for qid, t in targets.items():
        e = ep.setdefault(t.episode_id, [cond[qid], [], []])
        e[1].append(sq(pred[qid], t.probabilities)); e[2].append(sq(ref[qid], t.probabilities))
    return {k: (v[0], sum(v[1]) / len(v[1]), sum(v[2]) / len(v[2])) for k, v in ep.items()}


def points(pred, episodes=None):
    pe = per_episode(pred)
    eps = episodes if episodes is not None else list(pe)
    ratios = {}
    for c in CONDS:
        sel = [pe[e] for e in eps if pe[e][0] == c]
        ratios[c] = (sum(x[1] for x in sel) / len(sel)) / (sum(x[2] for x in sel) / len(sel))
    R = sum(ratios.values()) / 4
    return 60 / (1 + R), ratios


def split(p):
    w = 1 - p[0]
    return w, [x / w for x in p[1:]] if w > 0 else [0.25] * 4


def combine(w, s):
    return [1 - w] + [w * x for x in s]


mix = {t: [pi * a + (1 - pi) * b for a, b in zip(A, B)] for t, (A, B, pi) in model["size_components"].items()}
ours = runs["exp4"]
variants = {
    "EXP-4 as submitted": ours,
    "our w + TRUE size": {k: combine(split(ours[k])[0], split(targets[k].probabilities)[1]) for k in ids},
    "TRUE w + our size": {k: combine(split(targets[k].probabilities)[0], split(ours[k])[1]) for k in ids},
    "TRUE w + blind type-mix size": {k: combine(split(targets[k].probabilities)[0], mix[queries[k].machine_type]) for k in ids},
    "our w + blind type-mix size": {k: combine(split(ours[k])[0], mix[queries[k].machine_type]) for k in ids},
}
print("1) Error decomposition (official formula, all 96 validation episodes)")
for name, pred in variants.items():
    pts, r = points(pred)
    print(f"   {name:30s} {pts:7.3f}   " + "  ".join(f"{c[:6]} {r[c]:.4f}" for c in CONDS))

# per-query share of EXP-4 squared error coming from p_0 (win/lose) vs payout sizes
e0 = sum((ours[k][0] - targets[k].probabilities[0]) ** 2 for k in ids)
etot = sum(sq(ours[k], targets[k].probabilities) for k in ids)
print(f"   share of EXP-4 squared error in p_0 column: {e0/etot:.2f}")

print("\n2) Episode-bootstrap noise (resample episodes within condition, paired across models)")
pe_eps = {}
for qid, t in targets.items():
    pe_eps.setdefault(cond[qid], set()).add(t.episode_id)
pe_eps = {c: sorted(v) for c, v in pe_eps.items()}
rng = random.Random(314159)
pairs = [("exp3", "exp4"), ("exp2", "exp3"), ("exp1", "exp2")]
draws = {p: [] for p in pairs}
single = []
cache = {r: per_episode(runs[r]) for r in runs}


def pts_from(pe, eps):
    rs = []
    for c in CONDS:
        sel = [pe[e] for e in eps if pe[e][0] == c]
        rs.append((sum(x[1] for x in sel) / len(sel)) / (sum(x[2] for x in sel) / len(sel)))
    return 60 / (1 + sum(rs) / 4)


for _ in range(3000):
    eps = [rng.choice(pe_eps[c]) for c in CONDS for _ in pe_eps[c]]
    single.append(pts_from(cache["exp4"], eps))
    for a, b in pairs:
        draws[(a, b)].append(pts_from(cache[b], eps) - pts_from(cache[a], eps))
single.sort()
print(f"   EXP-4 points 95% interval: {single[75]:.2f} – {single[2924]:.2f}")
for (a, b), d in draws.items():
    d.sort()
    print(f"   {b} minus {a}: mean {sum(d)/len(d):+.3f} pts, 95% interval {d[75]:+.3f} … {d[2924]:+.3f}, P(gain>0) {sum(x > 0 for x in d)/len(d):.3f}")

print("\n3) Did train-CV predict validation gains? (train OOF Brier -> points via 60/(1+R) slope at R~0.1, ref err ~0.19)")
cv = json.loads((ROOT / "runs/final_cv_report.json").read_text())
cv4 = json.loads((ROOT / "runs/exp4_cv/cv.json").read_text())
oof = {"exp1": cv["exp1"]["oof"]["mean"], "exp2": cv["exp2"]["oof"]["mean"], "exp3": cv["exp3"]["oof"]["mean"], "exp4": cv4["oof_calibrated"]["mean"]}
val = {r: points(runs[r])[0] for r in runs}
for a, b in (("exp1", "exp2"), ("exp2", "exp3"), ("exp3", "exp4"), ("exp1", "exp4")):
    d_brier = oof[b] - oof[a]
    pred_pts = -d_brier / 0.19 * 60 / (1.1 ** 2)
    print(f"   {a}->{b}: train-CV dBrier {d_brier:+.5f} ~ predicted {pred_pts:+.2f} pts | realized validation {val[b]-val[a]:+.2f} pts")
