"""EXP-4 (TRAIN only): oriented change-point model + EXP-3 regime sizes + calibration.

Outputs: honest nested-CV OOF losses (calibration fit on inner out-of-fold predictions), in-sample losses,
and the calibration parameters for the final config (fit on all-train OOF win probabilities).
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from common import CONDITIONS, TYPES, Snapshots, brier, condition_loss, episode_cv_folds, load_split, write_json
from cv_change import ChangeTables, log_offset_prior, win_probs
from cv_exp3 import em_sizes
from cv_oriented import expand_prior, hi_lo_levels, oriented_tables
from cv_pair_rule import agg_weights, five, machine_stats, type_priors
from house_edge.rules import N_RULES

ROOT = Path(__file__).resolve().parents[1]
C2 = json.loads((ROOT / "configs/exp2.json").read_text())
C3 = json.loads((ROOT / "configs/exp3.json").read_text())
STRENGTH, BETA, ALPHA_NULL = 40.0, 1.0, 0.5

q, h, labels, cond = load_split("train")
snaps = Snapshots(q, h, labels, cond, k_back=16)
tables = ChangeTables(snaps)
fold = np.array([episode_cv_folds(q, cond)[e] for e in snaps.episode])
_, group = np.unique([f"{e}|{t}" for e, t in zip(snaps.episode, snaps.tick)], return_inverse=True)
stats = machine_stats(snaps)
lpd = log_offset_prior(C2["no_change_prior"], len(tables.D))
sc = snaps.size_counts
rule_of = np.concatenate([[0], np.arange(1, N_RULES), np.arange(1, N_RULES)])


def eb_oriented(LMo_real, beta=BETA, iterations=20):
    per_h = logsumexp(lpd[None, :, None] + beta * LMo_real, axis=1)
    prior = np.full(N_RULES, 1.0 / N_RULES)
    for _ in range(iterations):
        s = expand_prior(np.log(prior))[None, :] + per_h
        post = np.exp(s - logsumexp(s, axis=1, keepdims=True))
        mass = 1.0 + np.array([post[:, rule_of == r].sum() for r in range(N_RULES)])
        prior = mass / mass.sum()
    return np.log(prior)


cache = {}


def fit_w(train_mask):
    key = hashlib.sha256(np.packbits(train_mask).tobytes()).hexdigest()  # full-mask key (no collisions)
    if key in cache:
        return cache[key]
    qids = sorted({snaps.rows[i][0] for i in np.flatnonzero(train_mask)})
    mu, _ = type_priors(stats, qids)
    hl = hi_lo_levels(snaps, qids)
    LMo, PRo = oriented_tables(tables, snaps, mu[snaps.mtype], ALPHA_NULL, hl[snaps.mtype], STRENGTH)
    lp = eb_oriented(LMo[train_mask & snaps.real])
    w, _ = win_probs(LMo, PRo, expand_prior(lp), BETA, lpd, C2["shared_prior"], group)
    size = np.zeros((len(fold), 4))
    for ti in range(3):
        A, B, pi = em_sizes(np.array([stats[x][3] for x in qids if stats[x][0] == ti], float))
        m = snaps.mtype == ti
        pa = 1 / (1 + np.exp(-(C3["regime_temper"] * (sc[m] @ (np.log(A) - np.log(B))) + np.log(pi / (1 - pi)))))
        size[m] = pa[:, None] * A + (1 - pa)[:, None] * B
    cache[key] = (w, size, lp, hl)
    return cache[key]


def calibrate(w, th):
    w = np.clip(w, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(th[0] * np.log(w / (1 - w)) + th[1 + snaps.mtype])))


def fit_theta(w_oof, size, mask):
    Wm = agg_weights(snaps, mask)
    return minimize(lambda th: float(Wm @ brier(five(calibrate(w_oof, th), size), snaps)), np.array([1.0, 0, 0, 0]),
                    method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 2000}).x


t0 = time.perf_counter()
every = np.ones(len(fold), bool)
w_oof = np.zeros(len(fold))
per_fold = {}
for f in range(5):
    per_fold[f] = fit_w(fold != f)
    w_oof[fold == f] = per_fold[f][0][fold == f]
print(f"outer folds done {time.perf_counter()-t0:.0f}s", flush=True)
oof_unc, oof_cal = np.zeros(len(fold)), np.zeros(len(fold))
thetas = []
for f in range(5):
    inner = np.zeros(len(fold))
    for g in range(5):
        if g != f:
            inner[fold == g] = fit_w((fold != f) & (fold != g))[0][fold == g]
    w, size, _, _ = per_fold[f]
    th = fit_theta(inner, size, fold != f)
    thetas.append(th.tolist())
    oof_unc[fold == f] = brier(five(w, size), snaps)[fold == f]
    oof_cal[fold == f] = brier(five(calibrate(w, th), size), snaps)[fold == f]
    print(f"fold {f} nested calibration done {time.perf_counter()-t0:.0f}s", flush=True)
w_all, size_all, lp_all, hl_all = fit_w(every)
theta_all = fit_theta(w_oof, size_all, every)
in_cal = brier(five(calibrate(w_all, theta_all), size_all), snaps)
exp3 = json.loads((ROOT / "runs/final_cv_report.json").read_text())["exp3"]
report = {"strength": STRENGTH, "beta": BETA, "alpha_null": ALPHA_NULL,
          "oof_uncalibrated": condition_loss(oof_unc, snaps, every), "oof_calibrated": condition_loss(oof_cal, snaps, every),
          "in_sample_calibrated": condition_loss(in_cal, snaps, every),
          "oof_per_fold": [condition_loss(oof_cal, snaps, fold == f)["mean"] for f in range(5)],
          "exp3_oof_per_fold": exp3["oof_per_fold"], "exp3_oof": exp3["oof"],
          "oof_real_rows": condition_loss(oof_cal, snaps, snaps.real)["mean"], "exp3_real_rows": exp3["oof_real_rows"],
          "theta_by_fold": thetas, "theta_all_train": theta_all.tolist(),
          "hi_lo_all_train": {TYPES[i]: hl_all[i].tolist() for i in range(3)}, "seconds": time.perf_counter() - t0}
report["delta_vs_exp3_per_fold"] = (np.array(report["oof_per_fold"]) - np.array(exp3["oof_per_fold"])).tolist()
write_json(ROOT / "runs/exp4_cv/cv.json", report)
print(json.dumps(report, indent=1))
