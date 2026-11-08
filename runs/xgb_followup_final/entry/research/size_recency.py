"""TRAIN-only test: do recency-weighted winning sizes predict the next win's size better than full history?

Holds EXP-2's out-of-fold win probabilities fixed and varies only the size model. Scored on all pseudo-queries
and on real cutoff rows; per-fold paired differences against the EXP-3 size model (regime, lambda=0.75, all history).
"""
import json
from pathlib import Path

import numpy as np

from common import TYPES, Snapshots, brier, episode_cv_folds, load_split
from cv_change import ChangeTables, eb_prior_change, log_offset_prior, win_probs
from cv_exp3 import em_sizes
from cv_pair_rule import agg_weights, five, machine_stats, type_priors
from house_edge.history import SIZES

EXP2 = json.loads((Path(__file__).resolve().parents[1] / "configs/exp2.json").read_text())
q, h, labels, cond = load_split("train")
snaps = Snapshots(q, h, labels, cond, k_back=16)
tables = ChangeTables(snaps)
fold = np.array([episode_cv_folds(q, cond)[e] for e in snaps.episode])
_, group = np.unique([f"{e}|{t}" for e, t in zip(snaps.episode, snaps.tick)], return_inverse=True)
stats = machine_stats(snaps)
lpd = log_offset_prior(EXP2["no_change_prior"], len(tables.D))
W = {f: agg_weights(snaps, fold == f) for f in range(5)}
WR = {f: agg_weights(snaps, (fold == f) & snaps.real) for f in range(5)}
w_set, comps = {}, {}
for f in range(5):
    qids = sorted({snaps.rows[i][0] for i in np.flatnonzero(fold != f)})
    mu, _ = type_priors(stats, qids)
    LM, PR = tables.build(mu[snaps.mtype], EXP2["alpha"])
    lp = eb_prior_change(LM[(fold != f) & snaps.real], EXP2["beta"], lpd)
    w_set[f], _ = win_probs(LM, PR, lp, EXP2["beta"], lpd, EXP2["shared_prior"], group)
    comps[f] = {ti: em_sizes(np.array([stats[x][3] for x in qids if stats[x][0] == ti], float)) for ti in range(3)}


def weighted_size_counts(hl):
    out = np.zeros((len(snaps.rows), 4))
    for i, (qid, _, _, _, t, _) in enumerate(snaps.rows):
        for u, v in snaps.machines[qid][1].items():
            if u < t and v > 0:
                out[i, SIZES.index(v)] += 1.0 if hl is None else 2.0 ** (-(t - 1 - u) / hl)
    return out


def loss(f, sc, lam):
    size = np.zeros((len(fold), 4))
    for ti in range(3):
        A, B, pi = comps[f][ti]
        m = snaps.mtype == ti
        pa = 1 / (1 + np.exp(-(lam * (sc[m] @ (np.log(A) - np.log(B))) + np.log(pi / (1 - pi)))))
        size[m] = pa[:, None] * A + (1 - pa)[:, None] * B
    b = brier(five(w_set[f], size), snaps)
    return W[f] @ b, WR[f] @ b


ref = {f: loss(f, weighted_size_counts(None), 0.75) for f in range(5)}
print("reference (all history, lambda .75): all", round(np.mean([ref[f][0] for f in range(5)]), 5), "real", round(np.mean([ref[f][1] for f in range(5)]), 5))
for hl in (16, 32, 64, 128):
    sc = weighted_size_counts(hl)
    for lam in (0.75, 1.0, 1.5, 2.0):
        r = {f: loss(f, sc, lam) for f in range(5)}
        da = np.array([r[f][0] - ref[f][0] for f in range(5)])
        dr = np.array([r[f][1] - ref[f][1] for f in range(5)])
        print(f"half-life {hl:4d} lambda {lam:4.2f}: d_all {da.mean():+.5f} (se {da.std(ddof=1)/5**.5:.5f}, folds better {int((da<0).sum())}/5) | d_real {dr.mean():+.5f} (se {dr.std(ddof=1)/5**.5:.5f})")
