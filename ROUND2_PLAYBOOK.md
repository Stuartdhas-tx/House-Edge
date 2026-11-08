# Round two (Adaptation): 90-minute playbook

**Confirmed organizer announcement supplied by the team:** Main closes **Sunday, November 8, 2026 at 10:30 AM CST**. Adaptation runs **10:30 AM–12:00 noon CST**. Main resubmissions are allowed; the latest response received before its deadline counts. This supersedes the earlier overnight schedule.

The announcement changes the schedule, not the scoring weights. The current kit is **Main 60 / Adaptation 40**; inspect the new kit's metadata and organizer instructions on release. Do not change organizer metadata or scoring code to reconcile an older 80/20 guide.

The log records Main EXP-5 uploaded Saturday around 19:20. Keep the exact submitted files and confirmation; a later Main replacement needs its own matching CSV, verified ZIP, and confirmation.

| Time, Sunday CST | Action and stopping rule |
|---|---|
| Before 10:30 | Finish any Main replacement with upload margin. Prepare Python 3.13, NumPy/SciPy for research, and these commands. |
| 10:30–10:40 | Extract Adaptation, check compatibility, fit and package the fixed frequency baseline without scoring validation. |
| By 10:45 | Upload the verified baseline pair under Adaptation as a fallback. Keep confirmation. |
| 10:40–11:05 | Run corrected TRAIN cross-validation and refit EXP-5. Timebox any TRAIN mechanism checks. |
| 11:05–11:20 | Freeze one candidate using TRAIN results; perform one validation confirmation batch against the baseline. No tuning from that batch. |
| By 11:30 | Finish candidate checks and fresh-process reproduction. If unfinished or unsuccessful, retain the verified baseline. |
| By 11:40 | Upload the chosen verified pair and keep confirmation. |
| 11:40–12:00 | Resolve upload problems only. Do not start another model search. |

The earlier clean-kit copy/refit/package dry run took 2.5 minutes on Main. Corrected five-variant nested CV later took about 107 seconds on Main. These are measured component timings, not a rehearsal or runtime guarantee for this whole Adaptation workflow.

The revised no-validation-score fallback path was also rehearsed on Main data in `runs/adaptation_fallback_rehearsal/`: fit, test CSV, format check, ZIP build, and fresh-process Python 3.13.9 reproduction passed in 3.54 seconds, with byte-identical probabilities. This verifies that component's commands; those Main query IDs are not an Adaptation entry, and nothing was uploaded.

## A. Inspect the new kit and prepare the fallback

Extract into a fresh folder. From the Main kit:

```sh
python tools/adapt_setup.py "../adaptation-kit"
cd "../adaptation-kit"
python - <<'PY'
from pathlib import Path
from slot_forecasting.rounds import round_metadata
metadata = round_metadata(Path("public"))
assert metadata["round"] == "adaptation", metadata
print(metadata)
PY
```

`adapt_setup.py` copies our code and leaves organizer files untouched. It refuses overwrites and checks known machine types and payouts. It currently copies before profiling, so a failed profile may leave copied code: investigate that failure rather than repeatedly running setup. A successful profile does not establish that temporal rules are unchanged. If metadata fails the new kit's own checker, resolve the kit/version mismatch; do not force Main metadata onto it.

Set the verification interpreter once, then use the same shell below. Check this installed local path before release:

```sh
ADAPT_PY313="$HOME/.local/share/uv/python/cpython-3.13.9-macos-aarch64-none/bin/python3.13"
"$ADAPT_PY313" --version
python -c 'import numpy, scipy; print(numpy.__version__, scipy.__version__)'
```

Build a fixed `pooled_recent` fallback directly. This avoids `run_starter.py`, which automatically scores six baselines on validation. This code reads no validation targets; it writes validation predictions for the single later confirmation batch.

```sh
python - <<'PY'
import hashlib
import json
import platform
from pathlib import Path
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.io import load_inputs, read_labels, write_submission

out = Path("runs/a_baseline")
out.mkdir(parents=True, exist_ok=False)
queries, history = load_inputs(Path("public/train"))
labels = read_labels(Path("public/train/labels.csv"), queries)
model = FrequencyForecaster(method="pooled_recent").fit(queries, history, labels)
model.save(out / "baseline.json")
for split, filename in (("validation", "validation_predictions.csv"), ("test", "submission.csv")):
    q, h = load_inputs(Path("public") / split)
    write_submission(out / filename, model.predict(q, h))
report = {
    "settings": {"method": "pooled_recent", "prior_strength": 20.0, "half_life": 24.0},
    "python": platform.python_version(),
    "selection": "Fixed fallback; fitted on Adaptation TRAIN; no validation scoring.",
    "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(Path("public/train").glob("*.csv"))},
}
(out / "run.json").write_text(json.dumps(report, indent=2) + "\n")
PY
python check_submission.py runs/a_baseline/submission.csv
python build_entry.py --starter runs/a_baseline --out zips/a_baseline.zip
python tools/verify_entry.py zips/a_baseline.zip runs/a_baseline/submission.csv --python "$ADAPT_PY313"
```

If every command succeeds and reproduction passes, upload **`runs/a_baseline/submission.csv` + `zips/a_baseline.zip`** under Adaptation by 10:45. This remains the default fallback if later work fails or runs past its timebox. No upload is automatic; click Submit and retain confirmation.

## B. Select and calibrate on TRAIN only

Use the corrected nested runner, which excludes the outer fold from every inner model and uses out-of-fold predictions for both win and size calibration inputs:

```sh
python research/cv_joint_size.py --out runs/a_nested_cv
```

For this short round, the default candidate is EXP-5 with calibration selected on new TRAIN data. The runner also reports fixed joint-inference controls; their Main results did not justify adoption. Do not promote a new joint variant during this workflow merely because its mean is slightly better. Its assumptions and checks are documented separately.

Primary comparison: actual query cutoffs (`results.*.real`), with condition results and fold variation. Pseudo-query rows are supporting evidence. These are realized-outcome Brier scores, not official technical points; they omit official inverse-reference-error weights. An inconsistent fold count alone does not establish that a mechanism broke.

The legacy `cv_exp4.py`, `cv_exp5.py`, and `cv_exp6.py` calibration reports are **not independently honest outer-fold estimates**: calibration can indirectly depend on the held-out outer fold. Do not use them as the decisive independent comparison. `cv_joint_size.py` is corrected for the fixed EXP-5/joint variants, not arbitrary model changes; it currently requires the original contrast threshold of 0.5.

Create the candidate config from the corrected report. Compare calibrated versus uncalibrated EXP-5 on TRAIN real cutoffs; use identity calibration if calibration does not improve that endpoint. Selected final coefficients come from global out-of-fold TRAIN predictions.

```sh
python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("runs/a_nested_cv/cv.json").read_text())
cal = report["results"]["exp5_cal"]["real"]["mean"]
uncal = report["results"]["exp5_uncal"]["real"]["mean"]
theta = report["theta_all_train"]["exp5"] if cal < uncal else [1.0, 0.0, 0.0, 0.0]
config = dict(report["config"])
config["model"] = "size_rule"
config["calibration_slope"] = theta[0]
config["calibration_intercepts"] = dict(zip(("balanced", "jackpot", "steady"), theta[1:]))
out = Path("configs/a_exp5_selected.json")
with out.open("x") as handle:
    json.dump(config, handle, indent=2)
    handle.write("\n")
print({"real_brier_cal": cal, "real_brier_uncal": uncal, "selected_theta": theta})
PY
python research/run_model.py --config configs/a_exp5_selected.json --out runs/a_exp5_selected --no-score
python check_submission.py runs/a_exp5_selected/submission.csv
```

Do **not** pass this report directly to `tools/config_from_cv.py`: that helper expects a single `theta_all_train` list, while this report stores a dictionary by variant. The explicit extraction above matches the current schema.

Optional TRAIN mechanism diagnostics, only if there is time before the 11:05 freeze:

```sh
python research/run_model.py --config configs/exp4.json --out runs/exp4 --no-score
(cd research && python r1_size_structure.py && python r3_rule_priors.py)
```

These scripts require `runs/exp4/model.json`; its calibration is irrelevant to their history diagnostics. New win pairs within lags 1–8 are covered. A size lag moving from first to second win lag, a different payout partition, or absent clear-rule training examples is a structural warning. Retain the verified fallback if the candidate cannot be assessed reliably in time. Do not run validation-target headroom/residual scripts to choose an architecture.

## C. One confirmation batch, then freeze

After the candidate and config are frozen, compare the baseline and that one candidate on fresh validation:

```sh
python check_submission.py runs/a_baseline/validation_predictions.csv --split validation
python check_submission.py runs/a_exp5_selected/validation_predictions.csv --split validation
```

Record both scores, condition ratios, and the decision in a new Adaptation `EXPERIMENT_LOG.md`. This is one confirmation batch with two forecasts. Later reuse of those targets would not remain independent.

Adopt EXP-5 only if it beats the fallback on official validation points, has no unexplained severe condition regression, and completes all checks below. Otherwise keep the uploaded fallback. Do not tune settings against this batch or start another architecture search.

## D. Package and reproduce by 11:30; upload by 11:40

For an adopted EXP-5 candidate:

```sh
python tests/test_leakage.py runs/a_exp5_selected/model.json
python tests/test_equivalence_exp5.py runs/a_exp5_selected/model.json
python tools/make_entry.py --run runs/a_exp5_selected --name a_exp5_selected --model-text \
  'EXP-5 structure. Global model parameters refitted on Adaptation TRAIN; structural hyperparameters carried from Main. Calibration selected using corrected episode-grouped nested TRAIN CV, with final coefficients from global OOF win and size predictions, or identity calibration if that performed better. Exact values are in model.json. One validation confirmation batch was used after freezing this candidate.'
python tools/verify_entry.py zips/a_exp5_selected.zip runs/a_exp5_selected/submission.csv
python tools/verify_entry.py zips/a_exp5_selected.zip runs/a_exp5_selected/submission.csv --python "$ADAPT_PY313"
python check_submission.py runs/a_exp5_selected/submission.csv
```

`make_entry.py` copies current source and uses the entry name for the documented config path, so keep `a_exp5_selected` consistent. Its README asserts reproduction on the training interpreter and Python 3.13.9; that is supported only after verification actually passes. If a different interpreter is used or entry text changes, correct the README, rebuild with `build_entry.py --source`, and verify the exact rebuilt ZIP.

Upload **`runs/a_exp5_selected/submission.csv` + `zips/a_exp5_selected.zip`** through the announced Adaptation channel by 11:40. Keep the exact pair, local hashes, and confirmation. If organizers use a portal with receipt/reproduction status, follow its announced checks; a Google Form confirmation establishes receipt, not completed reproduction. The latest received response can replace a good earlier entry, so never upload an unchecked replacement.

All output names must be fresh. If CV, fitting, compatibility, verification, or upload fails, preserve the verified baseline and its confirmation. The noon deadline leaves no overnight research stage.
