"""Fit a shipped (stdlib) forecaster on TRAIN, save model.json, write validation/test CSVs, score validation.

Each invocation that scores validation is one logged look at validation targets.
"""
import argparse
import json
import platform
import time
from pathlib import Path

from common import PUBLIC, ROOT, load_split, score_validation_csv, write_json
from slot_forecasting.io import write_submission

from house_edge.registry import build_model, load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="JSON with {'model': name, ...hyperparameters}")
    p.add_argument("--out", required=True)
    p.add_argument("--no-score", action="store_true", help="skip the validation look")
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit("Choose a new output folder.")
    config = json.loads(Path(a.config).read_text())
    t0 = time.perf_counter()
    tq, th, tl, _ = load_split("train")
    model = build_model(config).fit(tq, th, tl)
    fit_s = time.perf_counter() - t0
    out.mkdir(parents=True)
    model.save(out / "model.json")
    model = load_model(out / "model.json")  # predict from the saved artifact, as the entry will
    timings = {"fit": fit_s}
    for split, name in (("validation", "validation_predictions.csv"), ("test", "submission.csv")):
        q, h, _, _ = load_split(split)
        t1 = time.perf_counter()
        write_submission(out / name, model.predict(q, h))
        timings[f"predict_{split}"] = time.perf_counter() - t1
    report = {"config": config, "seconds": timings, "python": platform.python_version(),
              "train_only_fit": True}
    if not a.no_score:
        report["validation"] = score_validation_csv(out / "validation_predictions.csv")
    write_json(out / "run.json", report)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
