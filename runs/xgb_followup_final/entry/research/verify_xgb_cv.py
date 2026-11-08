"""Replay saved TRAIN-only XGBoost fold models without any retraining.

Run after cv_xgb.py has written both cv.json and oof_predictions.npz:
  python research/verify_xgb_cv.py --run runs/xgb_cv --reference runs/joint_all_cv

This checks research OOF artifacts, not an organizer prediction ZIP. It never
loads validation/test data and never fits or tunes any model.
"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from common import ROOT, Snapshots, episode_cv_folds, load_split
from xgb_features import build_features


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probabilities(values, n):
    """Mirror the runner's documented float64 normalization, without fitting."""
    p = np.asarray(values, dtype=np.float64)
    if p.shape != (n, 5) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Expected finite N-by-5 probabilities in [0,1].")
    totals = p.sum(axis=1, keepdims=True)
    if not np.allclose(totals, 1, rtol=0, atol=2e-6):
        raise ValueError("Probability rows do not sum to one within 2e-6.")
    return p / totals


def compare(actual, expected, tolerance):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} versus {expected.shape}.")
    maximum = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    return {"max_absolute_difference": maximum, "exact_equal": bool(np.array_equal(actual, expected)),
            "passed": bool(np.isfinite(maximum) and maximum <= tolerance)}


def _check(check, message):
    if not check:
        raise ValueError(message)


def verify(run, reference=None, tolerance=1e-12):
    run = Path(run)
    required = ("manifest.json", "cv.json", "oof_predictions.npz")
    missing = [name for name in required if not (run / name).is_file()]
    if missing:
        raise ValueError("CV run is incomplete; missing " + ", ".join(missing))
    start = time.perf_counter()
    manifest = json.loads((run / "manifest.json").read_text())
    cv = json.loads((run / "cv.json").read_text())
    report = {
        "status": "running", "run": str(run.resolve()), "tolerance": tolerance,
        "scope": "TRAIN-only replay of saved fold boosters and OOF predictions; no fitting or tuning.",
        "organizer_zip_checked": False,
        "organizer_zip_note": "This is not an organizer CSV/ZIP reproduction or packaging check.",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "xgboost": xgb.__version__},
        "artifact_sha256": {name: sha256(run / name) for name in required},
        "verifier_sha256": sha256(__file__),
    }
    try:
        _check(xgb.__version__ == manifest["versions"]["xgboost"] == "3.4.1",
               "Replay requires the recorded XGBoost 3.4.1 version.")
        for field in ("source_sha256", "input_sha256", "feature_names", "fold_by_episode", "variants", "parameters"):
            _check(manifest[field] == cv[field], f"Manifest and CV report disagree on {field}.")
        source_checks = {}
        for name, expected in manifest["source_sha256"].items():
            current = sha256(ROOT / name)
            source_checks[name] = {"expected": expected, "current": current, "matched": current == expected}
        report["source_hashes"] = source_checks
        _check(all(row["matched"] for row in source_checks.values()), "Source files changed since the benchmark manifest.")
        current_inputs = {path.name: sha256(path) for path in sorted((ROOT / "public/train").glob("*.csv"))}
        report["input_sha256"] = current_inputs
        _check(current_inputs == manifest["input_sha256"], "TRAIN CSV input hashes differ from the benchmark manifest.")

        q, history, labels, conditions = load_split("train")
        snaps = Snapshots(q, history, labels, conditions, k_back=16)
        features, names = build_features(snaps)
        _check(names == manifest["feature_names"], "Rebuilt feature names/order differ from the manifest.")
        _check(features.shape == (manifest["rows"], manifest["feature_count"]), "Rebuilt feature shape differs.")
        report["features"] = {"shape": list(features.shape), "dtype": str(features.dtype),
                              "sha256_c_order_bytes": hashlib.sha256(features.tobytes(order="C")).hexdigest(),
                              "names_matched": True}
        fold_of = episode_cv_folds(q, conditions)
        _check(fold_of == manifest["fold_by_episode"], "Rebuilt episode folds differ from the manifest.")
        folds = np.array([fold_of[e] for e in snaps.episode])
        saved = np.load(run / "oof_predictions.npz", allow_pickle=False)
        metadata = {"query_id": snaps.qid, "episode_id": snaps.episode, "tick": snaps.tick,
                    "real": snaps.real, "fold": folds, "outcome": snaps.y}
        report["row_alignment"] = {name: bool(np.array_equal(saved[name], expected))
                                   for name, expected in metadata.items()}
        _check(all(report["row_alignment"].values()), "Saved OOF IDs/ticks/real flags/folds/outcomes do not align with TRAIN rows.")
        n = len(snaps.rows)
        report["probabilities_valid"] = {}
        for name in manifest["variants"]:
            probabilities(saved[name], n)
            report["probabilities_valid"][name] = True
        standalone = [name for name in manifest["parameters"] if name != "xgb_correction"]
        replay = {name: np.zeros((n, 5)) for name in manifest["parameters"]}
        model_checks = {}
        for fold in range(5):
            held = folds == fold
            _check(held.any(), f"Fold {fold} has no held-out rows.")
            for name in manifest["parameters"]:
                path = run / f"fold{fold}_{name}.json"
                _check(path.is_file(), f"Missing saved fold model: {path}.")
                correction = name == "xgb_correction"
                feature_names = names + (manifest["correction_added_feature_names"] if correction else [])
                values = features[held]
                margin = None
                if correction:
                    # The runner appends raw EXP-5 probabilities before its
                    # margin-only normalization. Saved raw OOF differs at most
                    # by the runner's final float64 row normalization.
                    raw_base = np.asarray(saved["exp5_uncal"][held], dtype=np.float64)
                    values = np.column_stack([values, raw_base]).astype(np.float32)
                    margin = np.log(np.clip(probabilities(raw_base, int(held.sum())), 1e-12, 1.0))
                booster = xgb.Booster(params={"nthread": 2})
                booster.load_model(path)
                _check(booster.feature_names == feature_names, f"Saved feature names differ in {path.name}.")
                _check(booster.num_boosted_rounds() == manifest["parameters"][name]["rounds"],
                       f"Boosted round count differs in {path.name}.")
                matrix = xgb.DMatrix(values, feature_names=feature_names, base_margin=margin,
                                     missing=np.nan, nthread=2)
                predicted = probabilities(booster.predict(matrix, strict_shape=True), int(held.sum()))
                replay[name][held] = predicted
                check = compare(predicted, saved[name][held], tolerance)
                model_checks[path.name] = {"sha256": sha256(path), "rows": int(held.sum()),
                                          "rounds": booster.num_boosted_rounds(), **check}
                _check(check["passed"], f"Saved booster {path.name} does not reproduce its held-out predictions.")
            print(f"Replayed fold {fold}: {len(manifest['parameters'])} saved models", flush=True)
        report["fold_models"] = model_checks
        report["oof_replay"] = {name: compare(prediction, saved[name], tolerance) for name, prediction in replay.items()}
        report["blend_replay"] = {}
        for name in standalone:
            for weight in manifest["blend_weights"]:
                key = f"{name}_blend{round(100 * weight):02d}"
                predicted = probabilities((1 - weight) * saved["exp5_cal"] + weight * replay[name], n)
                check = compare(predicted, saved[key], tolerance)
                report["blend_replay"][key] = check
                _check(check["passed"], f"Saved fixed blend {key} was not reproduced.")
        # Reapply stored outer-fold calibration without refitting it.
        calibrated = np.zeros((n, 5))
        raw = probabilities(saved["exp5_uncal"], n)
        win = 1 - raw[:, 0]
        _check((win > 0).all(), "Cannot reconstruct conditional size from zero-win baseline rows.")
        size = raw[:, 1:] / win[:, None]
        for fold, theta in enumerate(cv["exp5_theta_by_fold"]):
            held = folds == fold
            w = np.clip(win[held], 1e-6, 1 - 1e-6)
            z = theta[0] * np.log(w / (1 - w)) + np.asarray(theta)[1 + snaps.mtype[held]]
            w = 1 / (1 + np.exp(-z))
            calibrated[held] = np.column_stack([1 - w, w[:, None] * size[held]])
        report["exp5_calibration_replay"] = compare(probabilities(calibrated, n), saved["exp5_cal"], tolerance)
        _check(report["exp5_calibration_replay"]["passed"], "Saved fold calibration does not reproduce exp5_cal.")
        report["exp5_refit_performed"] = False
        if reference is not None:
            reference = Path(reference)
            reference_path = reference / "oof_predictions.npz" if reference.is_dir() else reference
            prior = np.load(reference_path, allow_pickle=False)
            keys = list(zip(prior["query_id"].tolist(), prior["tick"].tolist()))
            _check(len(keys) == len(set(keys)), "Reference has duplicate (query_id,tick) keys.")
            lookup = {key: i for i, key in enumerate(keys)}
            positions = np.array([lookup[(str(qid), int(tick))] for qid, tick in zip(snaps.qid, snaps.tick)])
            _check(len(positions) == len(keys), "Reference contains a different snapshot set.")
            for name in metadata:
                _check(np.array_equal(prior[name][positions], saved[name]), f"Reference metadata differs: {name}.")
            comparisons = {name: compare(prior[name][positions], saved[name], tolerance)
                           for name in ("exp5_uncal", "exp5_cal")}
            report["reference"] = {"path": str(reference_path.resolve()), "sha256": sha256(reference_path),
                                   "metadata_aligned": True, "baseline_comparisons": comparisons}
            _check(all(row["passed"] for row in comparisons.values()), "Baseline OOF probabilities differ from the reference run.")
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        report["seconds"] = time.perf_counter() - start
        (run / "verification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--reference", help="Optional prior joint CV directory or OOF NPZ to compare EXP-5 baselines.")
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    if not np.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("--tolerance must be finite and positive.")
    report = verify(args.run, args.reference, args.tolerance)
    maximum = max(row["max_absolute_difference"] for row in report["fold_models"].values())
    print(f"OOF REPLAY PASSED: {len(report['fold_models'])} models, maximum difference {maximum:.3g}; no ZIP check.")


if __name__ == "__main__":
    main()
