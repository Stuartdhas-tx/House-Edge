"""EXP-5 stdlib model vs the numpy pipeline used in research/cv_exp5.py (same formulas, copied to avoid re-running
the CV on import). Checks cell-structured EM components (train) and full 5-class predictions (validation inputs).
Run: python tests/test_equivalence_exp5.py runs/exp5/model.json"""
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from common import TYPES, Snapshots, load_split  # noqa: E402
from cv_change import ChangeTables, log_offset_prior  # noqa: E402
from cv_oriented import expand_prior, oriented_tables  # noqa: E402
from cv_pair_rule import five  # noqa: E402
from house_edge.registry import load_model  # noqa: E402
from house_edge.rules import N_RULES, PAIRS  # noqa: E402
from house_edge.size_rule import first_lag  # noqa: E402

SIZE_IDX = {1: 0, 2: 1, 5: 2, 20: 3}
model = load_model(sys.argv[1])


def groups(S, t, a):
    g = np.zeros((2, 4))
    for u, v in S.items():
        if u < t and v > 0 and (u - a) in S:
            g[int(S[u - a] >= 2), SIZE_IDX[v]] += 1
    return g


def em_cells(G, A, B, iters=100):
    A, B, pi = np.array(A, float), np.array(B, float), 0.5
    for _ in range(iters):
        l1 = np.log(pi) + G[:, 1] @ np.log(A) + G[:, 0] @ np.log(B)
        l0 = np.log(1 - pi) + G[:, 0] @ np.log(A) + G[:, 1] @ np.log(B)
        r = 1 / (1 + np.exp(np.clip(l0 - l1, -700, 700)))
        A = (r[:, None] * G[:, 1] + (1 - r)[:, None] * G[:, 0]).sum(0) + 0.5; A /= A.sum()
        B = (r[:, None] * G[:, 0] + (1 - r)[:, None] * G[:, 1]).sum(0) + 0.5; B /= B.sum()
        pi = float(np.clip(r.mean(), 0.05, 0.95))
    if A[0] < B[0]:
        A, B = B, A
    return A, B


# 1) EM components on all train
tq, th, tl, tc = load_split("train")
train = Snapshots(tq, th, tl, tc, k_back=1)
d_comp = 0.0
for t in TYPES:
    G = []
    for qid, (q, S) in train.machines.items():
        if q.machine_type == t:
            a = first_lag(S, q.cutoff)
            if a is not None:
                G.append(groups(S, q.cutoff, a))
    A, B = em_cells(np.array(G), *model.size_components[t][:2])
    d_comp = max(d_comp, np.abs(A - model.rule_components[t][0]).max(), np.abs(B - model.rule_components[t][1]).max())

# 2) predictions on validation inputs
vq, vh, vl, vc = load_split("validation")
val = Snapshots(vq, vh, vl, vc, k_back=1)
mu = np.array([model.win_rate[t] for t in TYPES]); lv = np.array([model.levels[t] for t in TYPES])
lpd = log_offset_prior(model.no_change_prior, len(model.offsets))
LM, PR = oriented_tables(ChangeTables(val), val, mu[val.mtype], model.alpha, lv[val.mtype], model.orient_strength)
_, group = np.unique(val.episode, return_inverse=True)
scores = expand_prior(np.array(model.log_prior))[None, None, :] + model.beta * LM
Lm = logsumexp(scores, axis=2); post = np.exp(scores - Lm[:, :, None]); condp = (post * PR).sum(2)
indep = lpd[None, :] + Lm
Gn = group.max() + 1
sums = np.zeros((Gn, Lm.shape[1])); np.add.at(sums, group, Lm)
shared = lpd[None, :] + sums
sig = model.shared_prior
log_zs = np.log(sig) + logsumexp(shared, axis=1)
log_zi = np.log(1 - sig) + np.bincount(group, weights=logsumexp(indep, axis=1), minlength=Gn)
p_sh = 1 / (1 + np.exp(np.minimum(700, log_zi - log_zs)))
pd = p_sh[group][:, None] * np.exp(shared - logsumexp(shared, axis=1, keepdims=True))[group] \
    + (1 - p_sh[group])[:, None] * np.exp(indep - logsumexp(indep, axis=1, keepdims=True))
w = (pd * condp).sum(1)
marg = np.einsum("sd,sdh->sh", pd, post)
rule_of = np.concatenate([[0], np.arange(1, N_RULES), np.arange(1, N_RULES)])
first_of_h = np.array([0] * 9 + [a for a, _ in PAIRS])[rule_of]
masses = np.stack([marg[:, first_of_h == a].sum(1) for a in range(8)], 1)
z = np.log(np.clip(w, 1e-6, 1 - 1e-6) / (1 - np.clip(w, 1e-6, 1 - 1e-6)))
icpt = np.array([model.calibration_intercepts[t] for t in TYPES])
w_cal = 1 / (1 + np.exp(-(model.calibration_slope * z + icpt[val.mtype])))
size = np.zeros((len(w), 4))
for i, (qid, _, mt, _, t, _) in enumerate(val.rows):
    S = val.machines[qid][1]
    A, B = map(np.asarray, model.rule_components[mt])
    pa = 0.5 * masses[i, 0]
    for a in range(1, 8):
        if (t - a) not in S:
            pa += 0.5 * masses[i, a]; continue
        g = groups(S, t, a)
        p1 = 1 / (1 + np.exp(-np.clip((g[1] - g[0]) @ (np.log(A) - np.log(B)), -700, 700)))
        pa += masses[i, a] * (p1 if S[t - a] >= 2 else 1 - p1)
    size[i] = pa * A + (1 - pa) * B
pred = model.predict(vq, vh)
d_p = np.abs(five(w_cal, size) - np.array([pred[q] for q in val.qid])).max()
print(f"max |EM component diff|={d_comp:.2e}  max |probability diff| on validation={d_p:.2e}")
assert d_comp < 1e-9 and d_p < 1e-9, "research path diverges from shipped model"
print("EQUIVALENCE PASSED")
