"""Consolidated TRAIN cross-validation report for the final configs of EXP-1/2/3 (+ pooled_recent).

Out-of-fold: every fitted quantity (type base rates, size priors, EB rule prior, EM size components,
calibration) is refit without the held-out episodes. In-sample: fit on all train, scored on train.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from common import CONDITIONS, TYPES, Snapshots, brier, condition_loss, episode_cv_folds, load_split, pooled_recent_snapshots, write_json
from cv_change import ChangeTables, eb_prior_change, log_offset_prior, win_probs
from cv_exp3 import em_sizes
from cv_pair_rule import agg_weights, eb_prior, five, machine_stats, rule_arrays, softmax_rows, type_priors

ROOT = Path(__file__).resolve().parents[1]
C1, C2, C3 = (json.loads((ROOT / f"configs/exp{i}.json").read_text()) for i in (1, 2, 3))
q, h, labels, cond = load_split("train")
snaps = Snapshots(q, h, labels, cond, k_back=16)
tables = ChangeTables(snaps)
fold = np.array([episode_cv_folds(q, cond)[e] for e in snaps.episode])
_, group = np.unique([f"{e}|{t}" for e, t in zip(snaps.episode, snaps.tick)], return_inverse=True)
stats = machine_stats(snaps)
sc = snaps.size_counts
every = np.ones(len(fold), bool)
lpd = log_offset_prior(C2["no_change_prior"], len(tables.D))


def calibrate(w, theta):
    w = np.clip(w, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(theta[0] * np.log(w / (1 - w)) + theta[1 + snaps.mtype])))


def fit_predict(train_mask):
    """All three models fitted on `train_mask` episodes; returns per-row Brier for every row."""
    qids = sorted({snaps.rows[i][0] for i in np.flatnonzero(train_mask)})
    mu, size_p = type_priors(stats, qids)
    real = train_mask & snaps.real
    out = {}
    # EXP-1
    N, K = snaps.counts(C1["half_life"])
    logml, pred = rule_arrays(N, K, snaps.cell_index, mu[snaps.mtype], C1["alpha"])
    lp = eb_prior(logml[real], C1["beta"])
    w1 = (softmax_rows(lp[None, :] + C1["beta"] * logml) * pred).sum(1)
    k1 = C1["size_strength"]
    out["exp1"] = brier(five(w1, (k1 * size_p[snaps.mtype] + sc) / (k1 + sc.sum(1))[:, None]), snaps)
    # EXP-2
    LM, PR = tables.build(mu[snaps.mtype], C2["alpha"])
    lp2 = eb_prior_change(LM[real], C2["beta"], lpd)
    w2, _ = win_probs(LM, PR, lp2, C2["beta"], lpd, C2["shared_prior"], group)
    k2 = C2["size_strength"]
    out["exp2"] = brier(five(w2, (k2 * size_p[snaps.mtype] + sc) / (k2 + sc.sum(1))[:, None]), snaps)
    # EXP-3: regime sizes (EM on training machines) + calibration fit on training rows' OOF-free w2
    size3 = np.zeros((len(fold), 4))
    for ti in range(3):
        A, B, pi = em_sizes(np.array([stats[x][3] for x in qids if stats[x][0] == ti], float))
        m = snaps.mtype == ti
        pa = 1 / (1 + np.exp(-(C3["regime_temper"] * (sc[m] @ (np.log(A) - np.log(B))) + np.log(pi / (1 - pi)))))
        size3[m] = pa[:, None] * A + (1 - pa)[:, None] * B
    out["_w2"], out["_size3"] = w2, size3
    return out


full = fit_predict(every)
folds = {f: fit_predict(fold != f) for f in range(5)}
# calibration needs w predicted WITHOUT the rows it is fit on: inner OOF w2 over the training folds
w2_oof = np.zeros(len(fold))
for f in range(5):
    w2_oof[fold == f] = folds[f]["_w2"][fold == f]
for f in range(5):
    trn = fold != f
    Wtr = agg_weights(snaps, trn)
    # inner OOF within training folds: rows of fold g (g != f) predicted by a model fit without g and f
    inner = np.zeros(len(fold))
    for g in range(5):
        if g != f:
            inner[fold == g] = fit_predict((fold != f) & (fold != g))["_w2"][fold == g]
    S = folds[f]["_size3"]
    theta = minimize(lambda th: float(Wtr @ brier(five(calibrate(inner, th), S), snaps)), np.array([1.0, 0, 0, 0]),
                     method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 2000}).x
    folds[f]["exp3"] = brier(five(calibrate(folds[f]["_w2"], theta), S), snaps)
theta_all = np.array([C3["calibration_slope"]] + [C3["calibration_intercepts"][t] for t in TYPES])
full["exp3"] = brier(five(calibrate(full["_w2"], theta_all), full["_size3"]), snaps)
base = brier(pooled_recent_snapshots(q, h, labels, snaps), snaps)

report = {}
for name in ("pooled_recent", "exp1", "exp2", "exp3"):
    oof = base.copy() if name == "pooled_recent" else np.zeros(len(fold))
    if name != "pooled_recent":
        for f in range(5):
            oof[fold == f] = folds[f][name][fold == f]
    per_fold = [condition_loss(oof, snaps, fold == f)["mean"] for f in range(5)]
    ins = base if name == "pooled_recent" else full[name]
    report[name] = {"oof": condition_loss(oof, snaps, every), "in_sample": condition_loss(ins, snaps, every),
                    "oof_per_fold": per_fold, "oof_fold_sd": float(np.std(per_fold, ddof=1)),
                    "oof_real_rows": condition_loss(oof, snaps, snaps.real)["mean"]}
    report[name]["gap_oof_minus_in_sample"] = report[name]["oof"]["mean"] - report[name]["in_sample"]["mean"]
for a, b in (("exp2", "exp1"), ("exp3", "exp2")):
    d = np.array(report[a]["oof_per_fold"]) - np.array(report[b]["oof_per_fold"])
    report[f"{a}_minus_{b}_per_fold"] = d.tolist()
write_json(ROOT / "runs/final_cv_report.json", report)
print(f"{'model':14s} {'OOF mean':>9s} {'in-sample':>9s} {'gap':>8s} {'fold sd':>8s} {'real rows':>9s} | OOF by condition (stable/short/drift/shared)")
for name in ("pooled_recent", "exp1", "exp2", "exp3"):
    r = report[name]
    print(f"{name:14s} {r['oof']['mean']:9.5f} {r['in_sample']['mean']:9.5f} {r['gap_oof_minus_in_sample']:+8.5f} {r['oof_fold_sd']:8.5f} {r['oof_real_rows']:9.5f} | "
          + " / ".join(f"{r['oof'][c]:.4f}" for c in CONDITIONS) + "   folds " + str(np.round(r["oof_per_fold"], 4).tolist()))
for k in ("exp2_minus_exp1_per_fold", "exp3_minus_exp2_per_fold"):
    print(k, np.round(report[k], 5).tolist())
