"""EXP-4 stdlib model vs numpy CV pipeline: hi/lo levels, EB rule prior, full 5-class predictions. Run: python tests/test_equivalence_exp4.py model.json"""
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from common import TYPES, Snapshots, load_split  # noqa: E402
from cv_change import ChangeTables, log_offset_prior, win_probs  # noqa: E402
from cv_oriented import expand_prior, hi_lo_levels, oriented_tables  # noqa: E402
from cv_pair_rule import five  # noqa: E402
from house_edge.registry import load_model  # noqa: E402
from house_edge.rules import N_RULES  # noqa: E402

model = load_model(sys.argv[1])
mu = np.array([model.win_rate[t] for t in TYPES])
lpd = log_offset_prior(model.no_change_prior, len(model.offsets))
tq, th, tl, tc = load_split("train")
train = Snapshots(tq, th, tl, tc, k_back=1)
hl = hi_lo_levels(train, sorted(train.machines))
d_lv = max(abs(hl[i][j] - model.levels[t][j]) for i, t in enumerate(TYPES) for j in (0, 1))
LMo, _ = oriented_tables(ChangeTables(train), train, mu[train.mtype], model.alpha, hl[train.mtype], model.orient_strength)
rule_of = np.concatenate([[0], np.arange(1, N_RULES), np.arange(1, N_RULES)])
per_h = logsumexp(lpd[None, :, None] + model.beta * LMo, axis=1)
prior = np.full(N_RULES, 1.0 / N_RULES)
for _ in range(model.eb_iterations):
    s = expand_prior(np.log(prior))[None, :] + per_h
    post = np.exp(s - logsumexp(s, axis=1, keepdims=True))
    mass = model.prior_smoothing + np.array([post[:, rule_of == r].sum() for r in range(N_RULES)])
    prior = mass / mass.sum()
d_prior = np.abs(np.log(prior) - np.array(model.log_prior)).max()

vq, vh, vl, vc = load_split("validation")
val = Snapshots(vq, vh, vl, vc, k_back=1)
lv = np.array([model.levels[t] for t in TYPES])
LM, PR = oriented_tables(ChangeTables(val), val, mu[val.mtype], model.alpha, lv[val.mtype], model.orient_strength)
_, group = np.unique(val.episode, return_inverse=True)
w, _ = win_probs(LM, PR, expand_prior(np.array(model.log_prior)), model.beta, lpd, model.shared_prior, group)
z = np.log(np.clip(w, 1e-6, 1 - 1e-6) / (1 - np.clip(w, 1e-6, 1 - 1e-6)))
icpt = np.array([model.calibration_intercepts[t] for t in TYPES])
w_cal = 1 / (1 + np.exp(-(model.calibration_slope * z + icpt[val.mtype])))
size = np.zeros((len(w), 4))
for ti, t in enumerate(TYPES):
    A, B, pi = map(np.asarray, model.size_components[t])
    m = val.mtype == ti
    pa = 1 / (1 + np.exp(-(model.regime_temper * (val.size_counts[m] @ (np.log(A) - np.log(B))) + np.log(pi / (1 - pi)))))
    size[m] = pa[:, None] * A + (1 - pa)[:, None] * B
pred = model.predict(vq, vh)
d_p = np.abs(five(w_cal, size) - np.array([pred[q] for q in val.qid])).max()
print(f"max |level diff|={d_lv:.2e}  max |log prior diff|={d_prior:.2e}  max |probability diff| on validation={d_p:.2e}")
assert d_lv < 1e-9 and d_prior < 1e-9 and d_p < 1e-9, "research path diverges from shipped model"
print("EQUIVALENCE PASSED")
