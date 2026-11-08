"""EXP-3 CV (TRAIN only): payout-size regime model + win-probability calibration on top of EXP-2.

Uses EXP-2's out-of-fold win probabilities (EB prior and base rates refit without the held-out fold).
Size models are scored both on all pseudo-queries and on REAL cutoff rows only (train labels.csv),
because history ticks may share a regime with their neighbours more than the cutoff does.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from common import CONDITIONS, TYPES, Snapshots, brier, condition_loss, episode_cv_folds, load_split, write_json
from cv_change import ChangeTables, eb_prior_change, log_offset_prior, win_probs
from cv_pair_rule import agg_weights, five, machine_stats, type_priors

EXP2 = json.loads((Path(__file__).resolve().parents[1] / "configs/exp2.json").read_text())


def em_sizes(counts, iterations=200):
    """Two-component multinomial mixture with one latent regime per machine. counts: (M, 4)."""
    tot = counts.sum(0) + 1.0
    m = tot / tot.sum()
    A = m * np.array([1.3, 0.9, 0.8, 0.8]); A /= A.sum()
    B = m * np.array([0.7, 1.1, 1.2, 1.2]); B /= B.sum()
    pi = 0.5
    for _ in range(iterations):
        la = np.log(pi) + counts @ np.log(A)
        lb = np.log(1 - pi) + counts @ np.log(B)
        r = 1.0 / (1.0 + np.exp(np.clip(lb - la, -700, 700)))
        A = (r[:, None] * counts).sum(0) + 0.5; A /= A.sum()
        B = ((1 - r)[:, None] * counts).sum(0) + 0.5; B /= B.sum()
        pi = float(np.clip(r.mean(), 1e-3, 1 - 1e-3))
    if A[0] < B[0]:
        A, B, pi = B, A, 1 - pi
    return A, B, pi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/exp3_cv")
    out = Path(p.parse_args().out)
    if out.exists():
        raise SystemExit("Choose a new output folder.")
    t0 = time.perf_counter()
    q, h, labels, cond = load_split("train")
    snaps = Snapshots(q, h, labels, cond, k_back=16)
    tables = ChangeTables(snaps)
    fold_of_ep = episode_cv_folds(q, cond)
    fold = np.array([fold_of_ep[e] for e in snaps.episode])
    _, group = np.unique([f"{e}|{t}" for e, t in zip(snaps.episode, snaps.tick)], return_inverse=True)
    stats = machine_stats(snaps)
    lpd = log_offset_prior(EXP2["no_change_prior"], len(tables.D))
    settings = list(range(5)) + ["all"]
    train_of = {f: (fold != f) if f != "all" else np.ones(len(fold), bool) for f in settings}
    W = {f: agg_weights(snaps, fold == f) for f in range(5)}
    W_real = {f: agg_weights(snaps, (fold == f) & snaps.real) for f in range(5)}
    # EXP-2 win probabilities: OOF (fold f rows from the fold-f fit) and per-setting full arrays
    w_set, size_p, qids_of = {}, {}, {}
    for f in settings:
        qids = sorted({snaps.rows[i][0] for i in np.flatnonzero(train_of[f])})
        mu, size_p[f] = type_priors(stats, qids)
        qids_of[f] = qids
        LM, PR = tables.build(mu[snaps.mtype], EXP2["alpha"])
        lp = eb_prior_change(LM[train_of[f] & snaps.real], EXP2["beta"], lpd)
        w_set[f], _ = win_probs(LM, PR, lp, EXP2["beta"], lpd, EXP2["shared_prior"], group)
        del LM, PR
    w_oof = np.zeros(len(fold))
    for f in range(5):
        w_oof[fold == f] = w_set[f][fold == f]
    print(f"EXP-2 win probabilities ready {time.perf_counter()-t0:.0f}s", flush=True)

    sc = snaps.size_counts
    results = {}

    def evaluate(name, P_by_fold):
        loss_all = [float(W[f] @ brier(P_by_fold[f], snaps)) for f in range(5)]
        loss_real = [float(W_real[f] @ brier(P_by_fold[f], snaps)) for f in range(5)]
        results[name] = {"all": float(np.mean(loss_all)), "real": float(np.mean(loss_real)),
                         "real_folds": loss_real, "all_folds": loss_all}

    # ---- size model S0: Dirichlet shrinkage toward the type distribution ----
    for kappa in (25, 50, 100, 200, 400, 1000, 1e6):
        evaluate(f"dirichlet_k{kappa:g}", {f: five(w_set[f], (kappa * size_p[f][snaps.mtype] + sc) / (kappa + sc.sum(1))[:, None]) for f in range(5)})
    # ---- size model S1: two-regime mixture (EM on training-fold machines), tempered posterior ----
    comps = {}
    for f in settings:
        comps[f] = {}
        for ti, t in enumerate(TYPES):
            counts = np.array([stats[qid][3] for qid in qids_of[f] if stats[qid][0] == ti], float)
            comps[f][ti] = em_sizes(counts)
    for lam in (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
        P_by = {}
        for f in range(5):
            size = np.zeros((len(fold), 4))
            for ti in range(3):
                A, B, pi = comps[f][ti]
                m = snaps.mtype == ti
                llr = sc[m] @ (np.log(A) - np.log(B))
                pa = 1.0 / (1.0 + np.exp(-(lam * llr + np.log(pi / (1 - pi)))))
                size[m] = pa[:, None] * A + (1 - pa)[:, None] * B
            P_by[f] = five(w_set[f], size)
        evaluate(f"regime_lam{lam:g}", P_by)
    best_size = min(results, key=lambda k: (results[k]["real"], k))
    print(f"size models done {time.perf_counter()-t0:.0f}s", flush=True)

    # ---- calibration of w on top of the best real-row size model: logit(w') = g*logit(w) + d_type ----
    def size_for(name, f):
        if name.startswith("dirichlet"):
            kappa = float(name.split("_k")[1])
            return (kappa * size_p[f][snaps.mtype] + sc) / (kappa + sc.sum(1))[:, None]
        lam = float(name.split("lam")[1])
        size = np.zeros((len(fold), 4))
        for ti in range(3):
            A, B, pi = comps[f][ti]
            m = snaps.mtype == ti
            pa = 1.0 / (1.0 + np.exp(-(lam * (sc[m] @ (np.log(A) - np.log(B))) + np.log(pi / (1 - pi)))))
            size[m] = pa[:, None] * A + (1 - pa)[:, None] * B
        return size

    def calibrate(w, theta):
        z = np.log(np.clip(w, 1e-6, 1 - 1e-6) / (1 - np.clip(w, 1e-6, 1 - 1e-6)))
        return 1.0 / (1.0 + np.exp(-(theta[0] * z + theta[1 + snaps.mtype])))

    cal_variants = {}
    for size_name in ("dirichlet_k50", "regime_lam0.75", "regime_lam1"):
        cal = {"theta_by_fold": [], "held_out": [], "uncalibrated": [], "held_out_real": [], "uncal_real": []}
        for f in range(5):
            trn = fold != f
            S_f = size_for(size_name, f)
            Wtr = agg_weights(snaps, trn)

            def obj(theta):
                return float(Wtr @ brier(five(calibrate(w_oof, theta), S_f), snaps))
            theta = minimize(obj, np.array([1.0, 0.0, 0.0, 0.0]), method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 2000}).x
            cal["theta_by_fold"].append(theta.tolist())
            cal["held_out"].append(float(W[f] @ brier(five(calibrate(w_set[f], theta), S_f), snaps)))
            cal["uncalibrated"].append(float(W[f] @ brier(five(w_set[f], S_f), snaps)))
            cal["held_out_real"].append(float(W_real[f] @ brier(five(calibrate(w_set[f], theta), S_f), snaps)))
            cal["uncal_real"].append(float(W_real[f] @ brier(five(w_set[f], S_f), snaps)))
        S_all = size_for(size_name, "all")
        cal["theta_all_train"] = minimize(lambda th: float(agg_weights(snaps, np.ones(len(fold), bool)) @ brier(five(calibrate(w_oof, th), S_all), snaps)),
                                          np.array([1.0, 0.0, 0.0, 0.0]), method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 2000}).x.tolist()
        cal_variants[size_name] = cal
        print(f"{size_name:15s} calibrated held-out {np.round(cal['held_out'],5).tolist()} mean {np.mean(cal['held_out']):.5f} | uncal {np.mean(cal['uncalibrated']):.5f} | real rows cal {np.mean(cal['held_out_real']):.5f} uncal {np.mean(cal['uncal_real']):.5f} | theta {np.round(cal['theta_all_train'],4).tolist()}")
    cal = cal_variants["dirichlet_k50"]
    theta_all = np.array(cal["theta_all_train"])
    report = {"size_models": results, "best_size_by_real_rows": best_size,
              "em_components_all_train": {TYPES[ti]: {"A": comps["all"][ti][0].round(4).tolist(), "B": comps["all"][ti][1].round(4).tolist(),
                                                      "pi_A": round(comps["all"][ti][2], 4)} for ti in range(3)},
              "calibration": cal_variants,
              "seconds": time.perf_counter() - t0}
    write_json(out / "cv.json", report)
    for k, v in sorted(results.items(), key=lambda kv: kv[1]["real"]):
        print(f"{k:18s} all={v['all']:.5f}  real-cutoff rows={v['real']:.5f}")
    print("EM components (all train):", json.dumps(report["em_components_all_train"]))
    print("done")

if __name__ == "__main__":
    main()
