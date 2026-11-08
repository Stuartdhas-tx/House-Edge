"""Fixed joint-size hypotheses, TRAIN only, with honest nested calibration.

Run: python research/cv_joint_size.py --out runs/<fresh> [--uncalibrated-only]
No validation data are loaded. Episode bootstrap conditions on fitted OOF models;
it does not account for adaptive architecture selection or refitting uncertainty.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from common import CONDITIONS, brier, condition_loss, episode_cv_folds, load_split
from cv_pair_rule import agg_weights
from joint_numpy import ROOT, VARIANTS, JointNumpyHarness


def loss_summary(loss, snaps, folds):
    every = np.ones(len(loss), bool)
    return {
        "all": condition_loss(loss, snaps, every),
        "real": condition_loss(loss, snaps, snaps.real),
        "fold_all": [condition_loss(loss, snaps, folds == f)["mean"] for f in range(5)],
        "fold_real": [condition_loss(loss, snaps, (folds == f) & snaps.real)["mean"] for f in range(5)],
    }


def paired_summary(delta, snaps, folds, seed=314159):
    out = loss_summary(delta, snaps, folds)
    rng = np.random.default_rng(seed)
    for name, mask in (("all", np.ones(len(delta), bool)), ("real", snaps.real)):
        condition_draws = []
        for c in CONDITIONS:
            m = mask & (snaps.condition == c)
            episodes = np.unique(snaps.episode[m])
            values = np.array([delta[m & (snaps.episode == e)].mean() for e in episodes])
            draws = rng.integers(len(values), size=(5000, len(values)))
            condition_draws.append(values[draws].mean(axis=1))
        out[f"bootstrap95_{name}"] = np.quantile(np.mean(condition_draws, axis=0), [0.025, 0.975]).tolist()
    return out


def fit_calibration(harness, prediction, mask):
    weights = agg_weights(harness.snaps, mask)
    # Both w and size in prediction must be OOF within the specified mask.
    objective = lambda theta: float(weights @ brier(harness.probabilities(prediction, theta), harness.snaps))
    result = minimize(objective, np.array([1.0, 0, 0, 0]), method="Nelder-Mead",
                      options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 2000})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Calibration did not converge: {result.message}")
    return result.x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--uncalibrated-only", action="store_true")
    parser.add_argument("--k-back", type=int, default=16)
    parser.add_argument("--include-missing-mixture", action="store_true",
                        help="Add joint_all: retain all wins, with a .5 mixture for unavailable lag values.")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("Choose a fresh output directory; refusing to overwrite results.")
    if args.k_back < 1:
        raise SystemExit("--k-back must be positive.")
    out.mkdir(parents=True)
    start = time.perf_counter()
    q, history, labels, conditions = load_split("train")
    harness = JointNumpyHarness(q, history, labels, conditions, k_back=args.k_back,
                                include_missing_mixture=args.include_missing_mixture)
    variants = harness.variants
    snaps = harness.snaps
    fold_of = episode_cv_folds(q, conditions)
    folds = np.array([fold_of[e] for e in snaps.episode])
    n = len(folds)
    every = np.ones(n, bool)
    raw_oof = {name: (np.zeros(n), np.zeros((n, 4))) for name in variants}
    outer = {}
    for f in range(5):
        outer[f] = harness.fit_predict(folds != f)
        held = folds == f
        for name in variants:
            raw_oof[name][0][held] = outer[f][name][0][held]
            raw_oof[name][1][held] = outer[f][name][1][held]
        print(f"outer fold {f}: {time.perf_counter() - start:.1f}s", flush=True)

    predictions = {f"{name}_uncal": harness.probabilities(raw_oof[name]) for name in variants}
    report = {
        "design": "Fixed variants; TRAIN only; whole-episode five-fold CV; raw episode/condition Brier.",
        "variants": list(variants),
        "missing_mixture_assumption": ".5 big-payout probability for unavailable historical lag values" if args.include_missing_mixture else None,
        "uncalibrated_only": args.uncalibrated_only,
        "k_back": args.k_back,
        "rows": n,
        "real_rows": int(snaps.real.sum()),
        "episodes": len(set(snaps.episode)),
        "config": harness.config,
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__), ROOT / "research/joint_numpy.py")},
        "input_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted((ROOT / "public/train").glob("*.csv"))},
        "fold_by_episode": fold_of,
    }
    if not args.uncalibrated_only:
        calibrated = {name: np.zeros((n, 5)) for name in variants}
        theta_by_fold = {name: [] for name in variants}
        for f in range(5):
            # Each inner model excludes both the outer and inner episodes.
            # Cache keys include the full mask; only compact predictions persist.
            inner = {name: (np.zeros(n), np.full((n, 4), 0.25)) for name in variants}
            for g in range(5):
                if g == f:
                    continue
                fit = harness.fit_predict((folds != f) & (folds != g))
                held_inner = folds == g
                for name in variants:
                    inner[name][0][held_inner] = fit[name][0][held_inner]
                    inner[name][1][held_inner] = fit[name][1][held_inner]
            for name in variants:
                theta = fit_calibration(harness, inner[name], folds != f)
                theta_by_fold[name].append(theta.tolist())
                calibrated[name][folds == f] = harness.probabilities(outer[f][name], theta)[folds == f]
            print(f"nested calibration fold {f}: {time.perf_counter() - start:.1f}s", flush=True)
        report["theta_by_fold"] = theta_by_fold
        # Deployment calibration uses global OOF win AND OOF conditional size.
        report["theta_all_train"] = {name: fit_calibration(harness, raw_oof[name], every).tolist()
                                     for name in variants}
        predictions.update({f"{name}_cal": calibrated[name] for name in variants})

    losses = {name: brier(p, snaps) for name, p in predictions.items()}
    report["results"] = {name: loss_summary(loss, snaps, folds) for name, loss in losses.items()}
    report["paired_deltas"] = {}
    for suffix in ("uncal", "cal"):
        if f"exp5_{suffix}" not in losses:
            continue
        for name in variants[1:]:
            key = f"{name}_{suffix}_minus_exp5_{suffix}"
            report["paired_deltas"][key] = paired_summary(losses[f"{name}_{suffix}"] - losses[f"exp5_{suffix}"], snaps, folds)
        # These contrasts isolate the two changes introduced after joint mixing.
        for a, b in (("common_mix", "joint_mix"), ("joint_size", "common_mix")):
            key = f"{a}_{suffix}_minus_{b}_{suffix}"
            report["paired_deltas"][key] = paired_summary(losses[f"{a}_{suffix}"] - losses[f"{b}_{suffix}"], snaps, folds)
    report["seconds"] = time.perf_counter() - start
    (out / "cv.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    names = list(losses)
    with (out / "oof_losses.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "episode_id", "tick", "real_cutoff", "condition", "fold"] + names)
        for i, row in enumerate(snaps.rows):
            writer.writerow([row[0], row[1], row[4], int(row[5]), row[3], int(folds[i])]
                            + [float(losses[name][i]) for name in names])
    np.savez_compressed(out / "oof_predictions.npz", query_id=snaps.qid, episode_id=snaps.episode,
                        tick=snaps.tick, real=snaps.real, fold=folds, outcome=snaps.y,
                        **predictions,
                        **{f"{name}_raw_win": raw_oof[name][0] for name in variants},
                        **{f"{name}_raw_size": raw_oof[name][1] for name in variants})
    for name, result in report["results"].items():
        print(f"{name:18s} all {result['all']['mean']:.8f} real {result['real']['mean']:.8f}")
    for name, result in report["paired_deltas"].items():
        print(f"{name}: real {result['real']['mean']:+.8f}, CI95 {result['bootstrap95_real']}")
    print(f"Completed in {report['seconds']:.1f}s; results: {out}")


if __name__ == "__main__":
    main()
