"""The numpy research path used for CV must match the shipped stdlib model. Run: python tests/test_equivalence.py"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from common import Snapshots, load_split  # noqa: E402
from cv_pair_rule import eb_prior, rule_arrays, softmax_rows  # noqa: E402
from house_edge.registry import load_model  # noqa: E402

model = load_model(sys.argv[1] if len(sys.argv) > 1 else ROOT / "runs/exp1/model.json")
tq, th, tl, tc = load_split("train")
train = Snapshots(tq, th, tl, tc, k_back=1)
types = ("balanced", "jackpot", "steady")
mu = np.array([model.win_rate[t] for t in types])
N, K = train.counts(model.half_life)
logml, _ = rule_arrays(N, K, train.cell_index, mu[train.mtype], model.alpha)
lp = eb_prior(logml, model.beta, model.prior_smoothing, model.eb_iterations)
d_prior = np.abs(lp - np.array(model.log_prior)).max()

vq, vh, vl, vc = load_split("validation")
val = Snapshots(vq, vh, vl, vc, k_back=1)
N, K = val.counts(model.half_life)
logml, pred = rule_arrays(N, K, val.cell_index, mu[val.mtype], model.alpha)
w_np = (softmax_rows(np.array(model.log_prior)[None, :] + model.beta * logml) * pred).sum(1)
w_std = np.array([model.win_probability(val.machines[qid][1], val.machines[qid][0].cutoff, val.machines[qid][0].machine_type)[0]
                  for qid in val.qid])
d_w = np.abs(w_np - w_std).max()
print(f"max |log prior diff|={d_prior:.2e}  max |w diff| on validation={d_w:.2e}")
assert d_prior < 1e-9 and d_w < 1e-9, "research path diverges from shipped model"
print("EQUIVALENCE PASSED")
