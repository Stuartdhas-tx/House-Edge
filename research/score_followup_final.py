"""One logged official validation comparison of the already-frozen finalist.

Requires matching successful ZIP reproduction records. Refuses repeat scoring
into the same run. Does not train, tune, modify the entry, or upload anything.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from common import PUBLIC, ROOT, score_validation_csv
from slot_forecasting.io import read_rows, read_submission, read_targets
from slot_forecasting.rounds import score_round
from slot_forecasting.schema import CONDITIONS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    destination = run / "validation_confirmation.json"
    if destination.exists():
        raise SystemExit("This frozen candidate has already been scored here; no repeat look.")
    manifest = json.loads((run / "run.json").read_text())
    for name, expected in manifest["artifact_sha256"].items():
        if digest(run / name) != expected:
            raise SystemExit(f"Frozen candidate artifact changed: {name}")
    checks = json.loads((run / "reproduction_checks.json").read_text())
    if not checks["passed"] or len(checks["reproduction_runs"]) != 2:
        raise SystemExit("Both intended interpreter reproduction checks must pass first.")
    for check in checks["reproduction_runs"]:
        if (check["max_abs_probability_diff"] > 1e-8
                or check["csv_sha256"] != digest(run / "submission.csv")
                or check["zip_sha256"] != digest(run / "entry.zip")):
            raise SystemExit("Reproduction checks do not match this CSV/ZIP pair.")
    candidate_path = run / "validation_predictions.csv"
    baseline_path = ROOT / "runs/exp5/validation_predictions.csv"
    candidate = score_round(candidate_path, PUBLIC)
    baseline = score_round(baseline_path, PUBLIC)
    targets = read_targets(PUBLIC / "validation/targets.csv")
    ids = {t.query_id for t in targets}
    forecasts = [read_submission(path, ids) for path in (
        candidate_path, baseline_path, PUBLIC / "validation/reference.csv")]
    groups = {r["query_id"]: r["condition"] for r in read_rows(
        PUBLIC / "validation/conditions.csv", ("query_id", "condition"))}
    errors, episode_conditions = {}, {}
    for target in targets:
        episode = target.episode_id
        episode_conditions[episode] = groups[target.query_id]
        truth = np.asarray(target.probabilities)
        errors.setdefault(episode, []).append([
            float(((np.asarray(p[target.query_id]) - truth) ** 2).sum()) for p in forecasts])
    means = {episode: np.asarray(values).mean(axis=0) for episode, values in errors.items()}
    rng = np.random.default_rng(314159)
    resampled_ratios = []
    for condition in CONDITIONS:
        values = np.array([means[e] for e in sorted(means) if episode_conditions[e] == condition])
        indices = rng.integers(len(values), size=(5000, len(values)))
        samples = values[indices].mean(axis=1)
        resampled_ratios.append(samples[:, :2] / np.maximum(samples[:, 2:], 1e-9))
    ratios = np.mean(resampled_ratios, axis=0)
    points = candidate["max_points"] / (1 + ratios)
    bootstrap_gain = points[:, 0] - points[:, 1]
    gain = candidate["technical_points"] - baseline["technical_points"]
    condition_changes = {c: candidate["by_condition"][c]["loss_ratio"]
                         - baseline["by_condition"][c]["loss_ratio"] for c in CONDITIONS}
    report = {
        "design": "One frozen candidate versus the existing EXP-5 entry, after TRAIN-only selection and reproduction.",
        "audit_caveat": "Previously inspected public validation; historical tuning/audit halves are not independent. No settings changed after this comparison.",
        "candidate_kind": manifest["selected"], "candidate": candidate, "baseline": baseline,
        "gain_points": gain, "paired_episode_bootstrap95_gain": np.quantile(bootstrap_gain, [.025, .975]).tolist(),
        "bootstrap_method": "5000 paired resamples of complete episodes within each condition; candidate, baseline and reference errors resampled together; seed314159.",
        "condition_ratio_changes": condition_changes,
        "historical_halves_candidate": score_validation_csv(candidate_path),
        "historical_halves_baseline": score_validation_csv(baseline_path),
        "promotion_rule": "Point gain >=0.10 on the local60-point scale, no condition ratio deterioration >.005, plus matching passed reproduction.",
        "passes_promotion_rule": bool(gain >= .10 and max(condition_changes.values()) <= .005),
        "validation_csv_sha256": digest(candidate_path), "baseline_validation_sha256": digest(baseline_path),
        "test_csv_sha256": digest(run / "submission.csv"), "zip_sha256": digest(run / "entry.zip"),
        "validation_input_sha256": {p.name: digest(p) for p in sorted((PUBLIC / "validation").glob("*.csv"))},
        "uploaded": False,
    }
    with destination.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    manifest["validation_scored"] = True
    manifest["reproduction_verified"] = True
    (run / "run.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "candidate_kind", "gain_points", "paired_episode_bootstrap95_gain", "condition_ratio_changes",
        "historical_halves_candidate", "passes_promotion_rule", "uploaded")}, indent=2))


if __name__ == "__main__":
    main()
