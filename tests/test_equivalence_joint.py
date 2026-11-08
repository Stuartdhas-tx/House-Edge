"""Compare the shipped joint predictor with the independent NumPy CV path.

Uses TRAIN query histories at their actual cutoffs and already-saved model
parameters; it does not fit anything or read validation targets.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from common import TYPES, load_split
from house_edge.registry import load_model
from joint_numpy import JointNumpyHarness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--include-missing-mixture", action="store_true")
    args = parser.parse_args()
    model = load_model(args.model)
    data = json.loads(Path(args.model).read_text())
    queries, history, labels, conditions = load_split("train")
    harness = JointNumpyHarness(queries, history, labels, conditions, config=data, k_back=1,
                               include_missing_mixture=args.include_missing_mixture)
    theta = [model.calibration_slope] + [model.calibration_intercepts[t] for t in TYPES]
    variants = harness.predict_model(data)
    errors = {}
    cases = [("exp5", (False, False, False, False)),
             ("joint_mix", (False, False, True, False)),
             ("common_mix", (False, True, True, False)),
             ("joint_size", (True, True, True, False))]
    if args.include_missing_mixture:
        cases.append(("joint_all", (True, False, True, True)))
    for name, flags in cases:
        (model.joint_size_evidence, model.joint_size_common,
         model.joint_average, model.joint_size_missing_mixture) = flags
        stdlib = model.predict(queries, history)
        reference = harness.probabilities(variants[name], theta)
        error = float(np.abs(reference - np.array([stdlib[qid] for qid in harness.snaps.qid])).max())
        errors[name] = error
        if error >= 1e-9:
            raise AssertionError(f"NumPy/shipped mismatch for {name}: {error}")
    print(json.dumps({"train_queries": len(queries), "max_probability_diff_by_variant": errors}, indent=2))
    print("JOINT EQUIVALENCE PASSED")


if __name__ == "__main__":
    main()
