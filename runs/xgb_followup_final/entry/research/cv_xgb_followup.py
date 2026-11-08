"""Finite TRAIN-only calibration/feature ablation in XGB_FOLLOWUP_PLAN.md.

Three predetermined partitions reuse the same episodes. Aggregation averages
paired losses, never probabilities, and bootstraps episode clusters only.
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

from common import CONDITIONS, TYPES, brier, condition_loss, episode_cv_folds, load_split
from cv_joint_size import fit_calibration, loss_summary, paired_summary
from cv_pair_rule import agg_weights
from cv_xgb import (CORE_SOURCES, CORRECTION, PROBABILITY_NAMES, fit_predict_booster,
                    nested_exp5_inputs, normalized_training_weights, validate_probabilities, write_json)
from followup_calibration import BOUNDS, PENALTY, fit_calibrator, predict_probabilities
from joint_numpy import ROOT, JointNumpyHarness
from xgb_features import build_features

PARTITIONS = (("original", ""), ("followup-a", "followup-a|"), ("followup-b", "followup-b|"))
CANDIDATES = ("type_size_cal", "compact_xgb", "full_xgb", "half_full")
VARIANTS = ("exp5_uncal", "exp5_cal") + CANDIDATES
TIE_ORDER = ("type_size_cal", "compact_xgb", "half_full", "full_xgb")
EXTRA_SOURCES = ("research/cv_xgb_followup.py", "research/followup_calibration.py", "XGB_FOLLOWUP_PLAN.md")


def partition_folds(queries, conditions, salt=""):
    """Original hash rule or a fixed salt, always balanced within condition."""
    if not salt:
        return episode_cv_folds(queries, conditions)
    episodes = {q.episode_id: conditions[q.query_id] for q in queries}
    result = {}
    for condition in CONDITIONS:
        ordered = sorted((e for e, c in episodes.items() if c == condition),
                         key=lambda e: hashlib.sha256((salt + e).encode()).hexdigest())
        result.update({episode: i % 5 for i, episode in enumerate(ordered)})
    return result


def checked_booster(train_x, y, weights, test_x, names, train_base, test_base, path):
    """Fit the fixed correction and verify its saved artifact on held-out inputs."""
    probabilities = fit_predict_booster(
        train_x, y, weights, test_x, names, parameters=CORRECTION["parameters"],
        rounds=CORRECTION["rounds"], train_base=train_base, predict_base=test_base, model_path=path)
    booster = xgb.Booster(model_file=path)
    booster.set_param({"nthread": 2})
    base = validate_probabilities(test_base, len(test_x))
    matrix = xgb.DMatrix(test_x, base_margin=np.log(np.clip(base, 1e-12, 1)),
                         feature_names=list(names), missing=np.nan, nthread=2)
    reloaded = validate_probabilities(booster.predict(matrix, strict_shape=True), len(test_x))
    difference = float(np.abs(probabilities - reloaded).max())
    if difference > 1e-10:
        raise AssertionError(f"Saved correction does not reproduce predictions: {path}, {difference}")
    return probabilities, difference


def compare_original(out, reference, predictions, snaps, folds, report):
    """Replication check against the already-completed first benchmark."""
    names = {"exp5_uncal": "exp5_uncal", "exp5_cal": "exp5_cal", "full_xgb": "xgb_correction"}
    previous = json.loads((reference / "cv.json").read_text())
    with np.load(reference / "oof_predictions.npz") as saved:
        for key, value in (("query_id", snaps.qid), ("episode_id", snaps.episode),
                           ("tick", snaps.tick), ("real", snaps.real), ("fold", folds)):
            if not np.array_equal(value, saved[key]):
                raise AssertionError(f"Original benchmark metadata differs: {key}")
        differences = {name: float(np.abs(predictions[name] - saved[old]).max()) for name, old in names.items()}
    summary_equal = {name: report["results"][name] == previous["results"][old] for name, old in names.items()}
    check = {"reference": str(reference), "max_probability_difference": differences,
             "result_summaries_exact_equal": summary_equal,
             "exp5_calibration_exact_equal": report["exp5_theta_by_fold"] == previous["exp5_theta_by_fold"]}
    write_json(out / "replication_check.json", check)
    if max(differences.values()) > 1e-10:
        raise AssertionError(f"Original controls/full correction failed replication: {check}")
    return check


def run_partition(harness, features, names, folds, out, *, reference=None):
    out.mkdir()
    start = time.perf_counter()
    snaps, n = harness.snaps, len(folds)
    predictions = {name: np.zeros((n, 5)) for name in VARIANTS}
    raw_oof = (np.zeros(n), np.zeros((n, 4)))
    type_features = np.eye(3, dtype=np.float32)[snaps.mtype]
    compact_names = [f"type_{t}" for t in TYPES] + list(PROBABILITY_NAMES)
    full_names = names + list(PROBABILITY_NAMES)
    parameters, control_parameters, timings, reload_checks = [], [], [], []
    for f in range(5):
        fold_start = time.perf_counter()
        train, held = folds != f, folds == f
        if set(snaps.episode[train]) & set(snaps.episode[held]):
            raise AssertionError("Outer partitions split an episode.")
        base = harness.fit_predict(train)["exp5"]
        inner = nested_exp5_inputs(harness, folds, f)
        raw = harness.probabilities(base)
        inner_probabilities = harness.probabilities(inner)[train]
        control_theta = fit_calibration(harness, inner, train)
        control_parameters.append(control_theta.tolist())
        calibrated = harness.probabilities(base, control_theta)
        predictions["exp5_uncal"][held], predictions["exp5_cal"][held] = raw[held], calibrated[held]
        raw_oof[0][held], raw_oof[1][held] = base[0][held], base[1][held]
        theta, info = fit_calibrator(inner[0][train], inner[1][train], snaps.mtype[train],
                                    snaps.y[train], agg_weights(snaps, train)[train])
        parameters.append(info)
        predictions["type_size_cal"][held] = predict_probabilities(base[0][held], base[1][held], snaps.mtype[held], theta)
        write_json(out / f"fold{f}_calibration.json", {"exp5_theta": control_theta.tolist(), "type_size": info})
        weights = normalized_training_weights(snaps, train)
        checks = {}
        for candidate, x, feature_names in (("compact_xgb", type_features, compact_names),
                                            ("full_xgb", features, full_names)):
            train_x = np.column_stack([x[train], inner_probabilities]).astype(np.float32)
            test_x = np.column_stack([x[held], raw[held]]).astype(np.float32)
            predictions[candidate][held], checks[candidate] = checked_booster(
                train_x, snaps.y[train], weights, test_x, feature_names, inner_probabilities,
                raw[held], out / f"fold{f}_{candidate}.json")
        predictions["half_full"][held] = 0.5 * calibrated[held] + 0.5 * predictions["full_xgb"][held]
        reload_checks.append(checks)
        timings.append(time.perf_counter() - fold_start)
        print(f"{out.name} fold {f}: {timings[-1]:.1f}s; calibration success={info['success']}, bounds={info['active_bounds']}", flush=True)
    predictions = {name: validate_probabilities(p, n) for name, p in predictions.items()}
    losses = {name: brier(p, snaps) for name, p in predictions.items()}
    report = {"results": {name: loss_summary(loss, snaps, folds) for name, loss in losses.items()},
              "exp5_theta_by_fold": control_parameters, "type_size_calibration_by_fold": parameters,
              "saved_model_reload_max_difference_by_fold": reload_checks,
              "fold_seconds": timings, "seconds": time.perf_counter() - start}
    if reference is not None:
        report["replication_check"] = compare_original(out, reference, predictions, snaps, folds, report)
    write_json(out / "cv.json", report)
    np.savez_compressed(out / "oof_predictions.npz", query_id=snaps.qid, episode_id=snaps.episode,
                        tick=snaps.tick, real=snaps.real, fold=folds, outcome=snaps.y,
                        exp5_raw_win=raw_oof[0], exp5_raw_size=raw_oof[1], **predictions)
    with (out / "oof_losses.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "episode_id", "tick", "real_cutoff", "condition", "fold"] + list(VARIANTS))
        for i, row in enumerate(snaps.rows):
            writer.writerow([row[0], row[1], row[4], int(row[5]), row[3], int(folds[i])]
                            + [float(losses[name][i]) for name in VARIANTS])
    print(f"{out.name} real Brier: " + ", ".join(f"{name}={report['results'][name]['real']['mean']:.8f}" for name in VARIANTS), flush=True)
    return report, losses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reference", default=str(ROOT / "runs/xgb_cv"))
    args = parser.parse_args()
    out, reference = Path(args.out), Path(args.reference)
    if out.exists():
        raise SystemExit("Choose a fresh output directory; refusing to overwrite results.")
    if not (reference / "cv.json").is_file() or not (reference / "oof_predictions.npz").is_file():
        raise SystemExit("The original benchmark reference is required for replication checks.")
    start = time.perf_counter()
    out.mkdir(parents=True)
    q, history, labels, conditions = load_split("train")
    harness = JointNumpyHarness(q, history, labels, conditions, k_back=16)
    snaps = harness.snaps
    features, names = build_features(snaps)
    features, names = np.asarray(features, np.float32), list(names)
    if features.shape != (len(snaps.rows), len(names)) or np.isinf(features).any() or len(set(names)) != len(names):
        raise ValueError("Invalid feature shape, names, or infinite feature values.")
    partitions = {name: partition_folds(q, conditions, salt) for name, salt in PARTITIONS}
    source_paths = tuple(dict.fromkeys(CORE_SOURCES + EXTRA_SOURCES))
    manifest = {
        "design": "XGB_FOLLOWUP_PLAN.md: four fixed postprocessors, three predefined partitions of the same TRAIN episodes.",
        "primary_endpoint": "Actual-cutoff episode/condition mean Brier; full-minus-compact is the feature-information diagnostic.",
        "uncertainty": "Repeat-average paired losses, bootstrap episode clusters within condition. Repetitions are not independent; no model-selection or refitting uncertainty.",
        "variants": list(VARIANTS), "partitions": partitions,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "xgboost": xgb.__version__},
        "xgb_correction": CORRECTION, "type_size_penalty": PENALTY, "type_size_bounds": BOUNDS,
        "full_feature_names": names + list(PROBABILITY_NAMES),
        "compact_feature_names": [f"type_{t}" for t in TYPES] + list(PROBABILITY_NAMES),
        "exp5_config": harness.config, "rows": len(snaps.rows), "real_rows": int(snaps.real.sum()),
        "episodes": len(set(snaps.episode)), "reference": str(reference),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_paths},
        "input_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted((ROOT / "public/train").glob("*.csv"))},
    }
    write_json(out / "manifest.json", manifest)
    reports, losses_by_repeat = {}, {}
    for name, _ in PARTITIONS:
        folds = np.array([partitions[name][e] for e in snaps.episode])
        reports[name], losses_by_repeat[name] = run_partition(
            harness, features, names, folds, out / name, reference=reference if name == "original" else None)
    every = np.ones(len(snaps.rows), bool)
    original_folds = np.array([partitions["original"][e] for e in snaps.episode])
    mean_losses = {name: np.mean([losses[name] for losses in losses_by_repeat.values()], axis=0) for name in VARIANTS}
    results = {name: {"all": condition_loss(loss, snaps, every), "real": condition_loss(loss, snaps, snaps.real)}
               for name, loss in mean_losses.items()}
    contrasts = [(name, "exp5_cal") for name in CANDIDATES] + [("full_xgb", "compact_xgb")]
    paired = {}
    for candidate, control in contrasts:
        summary = paired_summary(mean_losses[candidate] - mean_losses[control], snaps, original_folds)
        # Fold summaries from one arbitrary partition do not describe repeated CV.
        summary.pop("fold_all")
        summary.pop("fold_real")
        summary["real_delta_by_partition"] = {
            repeat: condition_loss(loss[candidate] - loss[control], snaps, snaps.real)["mean"]
            for repeat, loss in losses_by_repeat.items()}
        paired[f"{candidate}_minus_{control}"] = summary
    screening = {}
    for name in CANDIDATES:
        summary = paired[f"{name}_minus_exp5_cal"]
        condition_deltas = [summary["real"][c] for c in CONDITIONS]
        fitting_ok = (name != "type_size_cal" or all(
            info["success"] for report in reports.values() for info in report["type_size_calibration_by_fold"]))
        screening[name] = {
            "better_in_every_partition": all(delta < 0 for delta in summary["real_delta_by_partition"].values()),
            "conditions_improved": sum(delta < 0 for delta in condition_deltas),
            "max_condition_harm": max(condition_deltas), "fitting_ok": fitting_ok,
        }
        s = screening[name]
        s["eligible"] = bool(s["better_in_every_partition"] and s["conditions_improved"] >= 3
                             and s["max_condition_harm"] <= 0.002 and fitting_ok)
    eligible = [name for name in TIE_ORDER if screening[name]["eligible"]]
    selected = None
    if eligible:
        best_loss = min(results[name]["real"]["mean"] for name in eligible)
        selected = next(name for name in TIE_ORDER if name in eligible and results[name]["real"]["mean"] - best_loss < 1e-8)
    summary = {**manifest, "results": results, "paired_deltas": paired, "screening": screening,
               "selected_for_one_validation_comparison": selected,
               "selection_caveat": "Exploratory screening of an adaptively chosen family, not statistical confirmation. Feature/deployment verification remains required; no upload is automatic.",
               "partition_summaries": {name: report["results"] for name, report in reports.items()},
               "original_replication": reports["original"]["replication_check"],
               "seconds": time.perf_counter() - start}
    write_json(out / "cv.json", summary)
    # Only losses are averaged. Every prediction remains in its partition file.
    np.savez_compressed(out / "repeat_mean_losses.npz", query_id=snaps.qid,
                        episode_id=snaps.episode, tick=snaps.tick, real=snaps.real, **mean_losses)
    for name, result in results.items():
        print(f"repeat mean {name:15s}: real {result['real']['mean']:.8f}, all {result['all']['mean']:.8f}")
    for name, contrast in paired.items():
        print(f"{name}: real {contrast['real']['mean']:+.8f}, CI95 {contrast['bootstrap95_real']}")
    print(f"Exploratory screening: {screening}; selected={selected}")
    print(f"Completed in {summary['seconds']:.1f}s; results: {out}")


if __name__ == "__main__":
    main()
