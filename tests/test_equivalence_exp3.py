"""EXP-3 stdlib model vs the numpy CV pipeline (EM components and full 5-class predictions). Run: python tests/test_equivalence_exp3.py model.json"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from common import TYPES, Snapshots, load_split  # noqa: E402
from cv_change import ChangeTables, log_offset_prior, win_probs  # noqa: E402
from cv_exp3 import em_sizes  # noqa: E402
from cv_pair_rule import five, machine_stats  # noqa: E402
from house_edge.registry import load_model  # noqa: E402

model = load_model(sys.argv[1])
tq, th, tl, tc = load_split("train")
train = Snapshots(tq, th, tl, tc, k_back=1)
stats = machine_stats(train)
d_comp = 0.0
for ti, t in enumerate(TYPES):
    counts = np.array([stats[q][3] for q in sorted(stats) if stats[q][0] == ti], float)
    A, B, pi = em_sizes(counts)
    a2, b2, pi2 = model.size_components[t]
    d_comp = max(d_comp, np.abs(A - a2).max(), np.abs(B - b2).max(), abs(pi - pi2))

vq, vh, vl, vc = load_split("validation")
val = Snapshots(vq, vh, vl, vc, k_back=1)
mu = np.array([model.win_rate[t] for t in TYPES])
lpd = log_offset_prior(model.no_change_prior, len(model.offsets))
LM, PR = ChangeTables(val).build(mu[val.mtype], model.alpha)
_, group = np.unique(val.episode, return_inverse=True)
w, _ = win_probs(LM, PR, np.array(model.log_prior), model.beta, lpd, model.shared_prior, group)
z = np.log(np.clip(w, 1e-6, 1 - 1e-6) / (1 - np.clip(w, 1e-6, 1 - 1e-6)))
icpt = np.array([model.calibration_intercepts[t] for t in TYPES])
w_cal = 1 / (1 + np.exp(-(model.calibration_slope * z + icpt[val.mtype])))
size = np.zeros((len(w), 4))
for ti, t in enumerate(TYPES):
    A, B, pi = map(np.asarray, model.size_components[t])
    m = val.mtype == ti
    pa = 1 / (1 + np.exp(-(model.regime_temper * (val.size_counts[m] @ (np.log(A) - np.log(B))) + np.log(pi / (1 - pi)))))
    size[m] = pa[:, None] * A + (1 - pa)[:, None] * B
P_np = five(w_cal, size)
pred = model.predict(vq, vh)
P_std = np.array([pred[q] for q in val.qid])
d_p = np.abs(P_np - P_std).max()
print(f"max |EM component diff|={d_comp:.2e}  max |probability diff| on validation={d_p:.2e}")
assert d_comp < 1e-9 and d_p < 1e-9, "research path diverges from shipped model"
print("EQUIVALENCE PASSED")
