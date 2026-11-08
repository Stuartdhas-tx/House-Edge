"""Fixed TRAIN-only XGBoost benchmark; see XGB_BENCHMARK_PLAN.md.

No validation inputs, early stopping, or parameter search. Complete episodes
stay together. EXP-5 calibration cross-fits both win and size predictions;
the optional correction uses raw inner-OOF probabilities as margins/features.
This evaluates fixed candidates, not an unbiased adaptive family selector.
"""
import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import xgboost as xgb

from common import brier, episode_cv_folds, load_split
from cv_joint_size import fit_calibration, loss_summary, paired_summary
from cv_pair_rule import agg_weights
from joint_numpy import ROOT, JointNumpyHarness

SEED = 20261107
BASE_PARAMETERS = {
    "objective": "multi:softprob", "num_class": 5,
    "booster": "gbtree", "tree_method": "hist", "device": "cpu",
    "eta": 0.03, "min_child_weight": 20, "lambda": 20, "alpha": 1,
    "subsample": 0.8, "colsample_bytree": 0.8, "max_bin": 128,
    "seed": SEED, "nthread": 2, "verbosity": 1,
}
STANDALONE = {
    "xgb_depth3": {"parameters": {**BASE_PARAMETERS, "max_depth": 3}, "rounds": 300},
    "xgb_depth5": {"parameters": {**BASE_PARAMETERS, "max_depth": 5}, "rounds": 300},
}
CORRECTION = {"parameters": {**BASE_PARAMETERS, "max_depth": 2,
                              "min_child_weight": 40, "lambda": 50, "alpha": 2}, "rounds": 100}
BLEND_WEIGHTS = (0.10, 0.25)
PROBABILITY_NAMES = ("exp5_p_0", "exp5_p_1", "exp5_p_2", "exp5_p_5", "exp5_p_20")
CORE_SOURCES = (
    "research/cv_xgb.py", "research/xgb_features.py", "research/joint_numpy.py",
    "research/cv_joint_size.py", "research/common.py", "research/cv_pair_rule.py",
    "research/cv_change.py", "research/cv_exp3.py", "research/cv_oriented.py",
    "house_edge/rules.py", "house_edge/size_rule.py", "house_edge/history.py",
    "configs/exp5.json", "XGB_BENCHMARK_PLAN.md",
)


def normalized_training_weights(snaps, mask):
    """Mean-one weights preserve the prescribed Hessian/regularization scale."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        raise ValueError("Cannot fit an empty training partition.")
    weights = agg_weights(snaps, mask)[mask]
    weights *= len(weights) / weights.sum()
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Training weights must be finite and positive.")
    return weights


def validate_probabilities(probabilities, n_rows):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (probabilities.shape != (n_rows, 5) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()):
        raise ValueError("Expected finite N-by-5 class probabilities in [0, 1].")
    totals = probabilities.sum(axis=1, keepdims=True)
    if not np.allclose(totals, 1, rtol=0, atol=2e-6):
        raise ValueError("Class probabilities do not sum to one.")
    # Native predictions use float32; normalize once in float64 for scoring.
    return probabilities / totals


def fit_predict_booster(train_x, train_y, weights, predict_x, feature_names, *,
                        parameters, rounds, train_base=None, predict_base=None,
                        model_path=None):
    """Fit one fixed model; prediction labels are deliberately not accepted."""
    if (train_base is None) != (predict_base is None):
        raise ValueError("Correction margins are required at both fit and prediction time.")
    train_margin = predict_margin = None
    if train_base is not None:
        train_base = validate_probabilities(train_base, len(train_x))
        predict_base = validate_probabilities(predict_base, len(predict_x))
        # For softmax, log(p) is a valid raw multiclass margin. The common
        # additive constant is immaterial. Clip only to avoid log(0).
        train_margin = np.log(np.clip(train_base, 1e-12, 1.0))
        predict_margin = np.log(np.clip(predict_base, 1e-12, 1.0))
    train = xgb.DMatrix(train_x, label=np.asarray(train_y, dtype=np.int32),
                       weight=weights, base_margin=train_margin,
                       feature_names=list(feature_names), missing=np.nan, nthread=2)
    predict = xgb.DMatrix(predict_x, base_margin=predict_margin,
                         feature_names=list(feature_names), missing=np.nan, nthread=2)
    booster = xgb.train(dict(parameters), train, num_boost_round=rounds, verbose_eval=False)
    probabilities = validate_probabilities(booster.predict(predict, strict_shape=True), len(predict_x))
    if model_path is not None:
        booster.save_model(model_path)
    return probabilities


def nested_exp5_inputs(harness, folds, outer_fold):
    """OOF w/size for outer training; every fit excludes outer and inner folds."""
    n = len(folds)
    inner = (np.zeros(n), np.full((n, 4), 0.25))
    filled = np.zeros(n, bool)
    for g in range(5):
        if g == outer_fold:
            continue
        mask = (folds != outer_fold) & (folds != g)
        predicted = harness.fit_predict(mask)["exp5"]
        rows = folds == g
        inner[0][rows], inner[1][rows] = predicted[0][rows], predicted[1][rows]
        filled[rows] = True
    if not np.array_equal(filled, folds != outer_fold):
        raise AssertionError("Inner OOF predictions must cover exactly outer-training rows.")
    return inner


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--include-correction", action="store_true")
    parser.add_argument("--save-fold-models", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit("Choose a fresh output directory; refusing to overwrite results.")
    # Import is delayed so the low-level margin API can be checked on toy data.
    from xgb_features import build_features

    start = time.perf_counter()
    out.mkdir(parents=True)
    q, history, labels, conditions = load_split("train")
    harness = JointNumpyHarness(q, history, labels, conditions, k_back=16)
    snaps = harness.snaps
    feature_start = time.perf_counter()
    features, names = build_features(snaps)
    features = np.asarray(features, dtype=np.float32)
    names = list(names)
    if (features.shape != (len(snaps.rows), len(names)) or len(set(names)) != len(names)
            or np.isinf(features).any()):
        raise ValueError("Feature matrix shape/names are invalid, or contains infinity.")
    feature_seconds = time.perf_counter() - feature_start
    fold_of = episode_cv_folds(q, conditions)
    folds = np.array([fold_of[e] for e in snaps.episode])
    n = len(folds)
    variants = ["exp5_uncal", "exp5_cal"]
    for name in STANDALONE:
        variants.append(name)
        variants.extend(f"{name}_blend{round(100 * w):02d}" for w in BLEND_WEIGHTS)
    if args.include_correction:
        variants.append("xgb_correction")
    predictions = {name: np.zeros((n, 5)) for name in variants}
    exp5_oof = (np.zeros(n), np.zeros((n, 4)))
    theta_by_fold = []
    fold_seconds = []
    manifest = {
        "design": "Fixed exploratory XGB family comparison; no outer-fold tuning or validation data.",
        "selection_caveat": "Selecting the best reported candidate is not an unbiased estimate of adaptive model selection.",
        "bootstrap_caveat": "Paired within-condition episode bootstrap conditions on fitted OOF models; excludes refitting and prior search uncertainty.",
        "training_objective": "XGBoost multiclass log loss; mean-one condition/episode weights; no class reweighting.",
        "evaluation_objective": "Real-cutoff five-class Brier primary, all snapshots secondary; no official reference-error normalization.",
        "correction_base": "Uncalibrated EXP-5; inner OOF within outer training for both probability features and log base margins.",
        "blend_base": "Fresh nested-calibrated EXP-5, fitted without the outer episodes.",
        "variants": variants,
        "parameters": {**STANDALONE, **({"xgb_correction": CORRECTION} if args.include_correction else {})},
        "blend_weights": list(BLEND_WEIGHTS),
        "versions": {"python": platform.python_version(), "xgboost": xgb.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__},
        "exp5_config": harness.config,
        "feature_names": names,
        "correction_added_feature_names": list(PROBABILITY_NAMES) if args.include_correction else [],
        "feature_count": len(names), "feature_seconds": feature_seconds,
        "rows": n, "real_rows": int(snaps.real.sum()), "episodes": len(set(snaps.episode)),
        "fold_by_episode": fold_of,
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in CORE_SOURCES},
        "input_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted((ROOT / "public/train").glob("*.csv"))},
        "screening_rule": "Real-cutoff delta below zero, paired 95% bootstrap interval entirely below zero, no condition delta above .002; exploratory screening only.",
    }
    write_json(out / "manifest.json", manifest)
    print(f"Features: {features.shape}, {feature_seconds:.1f}s; {len(variants)} fixed reported variants", flush=True)

    for f in range(5):
        fold_start = time.perf_counter()
        train, held = folds != f, folds == f
        if set(snaps.episode[train]) & set(snaps.episode[held]):
            raise AssertionError("Outer training and prediction episodes overlap.")
        base = harness.fit_predict(train)["exp5"]
        inner = nested_exp5_inputs(harness, folds, f)
        theta = fit_calibration(harness, inner, train)
        theta_by_fold.append(theta.tolist())
        raw_base = harness.probabilities(base)
        calibrated_base = harness.probabilities(base, theta)
        predictions["exp5_uncal"][held] = raw_base[held]
        predictions["exp5_cal"][held] = calibrated_base[held]
        exp5_oof[0][held], exp5_oof[1][held] = base[0][held], base[1][held]
        weights = normalized_training_weights(snaps, train)
        for name, spec in STANDALONE.items():
            model_path = out / f"fold{f}_{name}.json" if args.save_fold_models else None
            candidate = fit_predict_booster(features[train], snaps.y[train], weights,
                                            features[held], names, parameters=spec["parameters"],
                                            rounds=spec["rounds"], model_path=model_path)
            predictions[name][held] = candidate
            for w in BLEND_WEIGHTS:
                blended = (1 - w) * calibrated_base[held] + w * candidate
                predictions[f"{name}_blend{round(100 * w):02d}"][held] = validate_probabilities(blended, int(held.sum()))
        if args.include_correction:
            inner_probabilities = harness.probabilities(inner)[train]
            train_features = np.column_stack([features[train], inner_probabilities]).astype(np.float32)
            held_features = np.column_stack([features[held], raw_base[held]]).astype(np.float32)
            model_path = out / f"fold{f}_xgb_correction.json" if args.save_fold_models else None
            predictions["xgb_correction"][held] = fit_predict_booster(
                train_features, snaps.y[train], weights, held_features, names + list(PROBABILITY_NAMES),
                parameters=CORRECTION["parameters"], rounds=CORRECTION["rounds"],
                train_base=inner_probabilities, predict_base=raw_base[held], model_path=model_path)
        fold_seconds.append(time.perf_counter() - fold_start)
        print(f"outer fold {f}: {fold_seconds[-1]:.1f}s; total {time.perf_counter() - start:.1f}s", flush=True)

    # Every OOF probability row must have been filled exactly once.
    for name in predictions:
        predictions[name] = validate_probabilities(predictions[name], n)
    losses = {name: brier(p, snaps) for name, p in predictions.items()}
    report = {**manifest,
              "exp5_theta_by_fold": theta_by_fold,
              "exp5_theta_all_train": fit_calibration(harness, exp5_oof, np.ones(n, bool)).tolist(),
              "results": {name: loss_summary(loss, snaps, folds) for name, loss in losses.items()},
              "paired_deltas": {}, "screen_pass": {}, "fold_seconds": fold_seconds}
    for name in variants:
        if name == "exp5_cal":
            continue
        comparison = paired_summary(losses[name] - losses["exp5_cal"], snaps, folds)
        report["paired_deltas"][f"{name}_minus_exp5_cal"] = comparison
        if name not in ("exp5_uncal", "exp5_cal"):
            report["screen_pass"][name] = bool(
                comparison["real"]["mean"] < 0 and comparison["bootstrap95_real"][1] < 0
                and max(v for c, v in comparison["real"].items() if c != "mean") <= 0.002)
    if args.include_correction:
        report["paired_deltas"]["xgb_correction_minus_exp5_uncal"] = paired_summary(
            losses["xgb_correction"] - losses["exp5_uncal"], snaps, folds)
    report["seconds"] = time.perf_counter() - start
    write_json(out / "cv.json", report)
    with (out / "oof_losses.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "episode_id", "tick", "real_cutoff", "condition", "fold"] + variants)
        for i, row in enumerate(snaps.rows):
            writer.writerow([row[0], row[1], row[4], int(row[5]), row[3], int(folds[i])]
                            + [float(losses[name][i]) for name in variants])
    np.savez_compressed(out / "oof_predictions.npz", query_id=snaps.qid,
                        episode_id=snaps.episode, tick=snaps.tick, real=snaps.real,
                        fold=folds, outcome=snaps.y, **predictions)
    for name, result in report["results"].items():
        print(f"{name:24s} all {result['all']['mean']:.8f} real {result['real']['mean']:.8f}")
    for name, result in report["paired_deltas"].items():
        print(f"{name}: real {result['real']['mean']:+.8f}, CI95 {result['bootstrap95_real']}")
    print(f"Screening outcomes: {report['screen_pass']}")
    print(f"Completed in {report['seconds']:.1f}s; results: {out}")


if __name__ == "__main__":
    main()
