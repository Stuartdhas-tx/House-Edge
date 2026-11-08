"""The numpy CV path for EXP-2 must match the shipped stdlib ChangePointForecaster. Run: python tests/test_equivalence_change.py model.json"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from common import Snapshots, load_split  # noqa: E402
from cv_change import ChangeTables, eb_prior_change, log_offset_prior, win_probs  # noqa: E402
from house_edge.registry import load_model  # noqa: E402

model = load_model(sys.argv[1])
types = ("balanced", "jackpot", "steady")
mu = np.array([model.win_rate[t] for t in types])
lpd = log_offset_prior(model.no_change_prior, len(model.offsets))

tq, th, tl, tc = load_split("train")
train = Snapshots(tq, th, tl, tc, k_back=1)
LM, _ = ChangeTables(train).build(mu[train.mtype], model.alpha)
lp = eb_prior_change(LM, model.beta, lpd, model.prior_smoothing, model.eb_iterations)
d_prior = np.abs(lp - np.array(model.log_prior)).max()

vq, vh, vl, vc = load_split("validation")
val = Snapshots(vq, vh, vl, vc, k_back=1)
LM, PR = ChangeTables(val).build(mu[val.mtype], model.alpha)
_, group = np.unique(val.episode, return_inverse=True)
w_np, _ = win_probs(LM, PR, np.array(model.log_prior), model.beta, lpd, model.shared_prior, group)
pred = model.predict(vq, vh)
w_std = np.array([1.0 - pred[qid][0] for qid in val.qid])
d_w = np.abs(np.clip(w_np, 1e-6, 1 - 1e-6) - w_std).max()
print(f"max |log prior diff|={d_prior:.2e}  max |w diff| on validation={d_w:.2e}")
assert d_prior < 1e-9 and d_w < 1e-9, "research path diverges from shipped model"
print("EQUIVALENCE PASSED")
