# Focused follow-up: a modest improvement, ready for resubmission

2026-11-07. **Recommend the frozen EXP-5 + full XGBoost correction candidate as a Main replacement.** It passes the practical promotion rule recorded before this follow-up's results. The currently recorded upload remains EXP-5; **nothing was uploaded in this work**.

- New Main CSV: `runs/xgb_followup_final/submission.csv`.
- Matching code ZIP: `runs/xgb_followup_final/entry.zip` (323,548 bytes).
- Keep the existing `runs/exp5/submission.csv` + `zips/exp5.zip` as the verified local fallback.
- Main's confirmed deadline is Sunday November 8, **10:30 AM CST**. The latest received Form response counts.

## What improved

The correction adds a small, regularized tree model to EXP-5's probability estimates, using strictly past rule, payout-size and peer-history features. It does not replace the probabilistic base model. The fitted trees have depth 2, 100 boosting rounds, and the same fixed settings used in the initial benchmark.

The earlier correction failed its strict confidence-interval screen. The user then asked to continue improvement with overfitting controls. A **new finite plan** was written in `XGB_FOLLOWUP_PLAN.md` before new results: compare calibration-only controls and a fixed half blend, rerun three predefined episode partitions, freeze one eligible candidate, and make at most one official validation confirmation. The earlier experiment's result was not rewritten or retrospectively declared successful.

## Training evaluation and selection

There are 96 training episodes, 768 actual cutoffs, and 12,288 rows including pseudo-cutoffs. All fits and postprocessing use whole-episode exclusion; both base win and size predictions for postprocessor training are inner-OOF within each outer training fold. Signal is not propagated backward to pseudo-queries, missing ticks are unknown, and peer features use only earlier observations in the same episode.

Primary endpoint: actual-cutoff five-class Brier, lower is better. These are realized-outcome surrogate errors without official reference normalization, not contest points.

| Model | Original partition | Partition A | Partition B | Repeat mean |
|---|---:|---:|---:|---:|
| EXP-5 control | 0.34854627 | 0.34852804 | 0.34865508 | 0.34857646 |
| Win + type/size calibration | 0.34845079 | 0.34852573 | 0.34863026 | 0.34853560 |
| Compact XGB calibration | 0.34834387 | 0.34818002 | 0.34923510 | 0.34858633 |
| Full XGB correction | 0.34738791 | 0.34810399 | 0.34784097 | 0.34777762 |
| Half correction / half EXP-5 | 0.34774893 | 0.34807627 | 0.34801849 | 0.34794790 |

The full correction improved in all three partitions. Its repeat-mean change versus EXP-5 is **-0.00079884**, with episode-bootstrap interval **[-0.00270825, +0.00111787]**. The interval still includes zero. Three of four conditions improved on average; stable worsened by 0.001254, below the prospective 0.002 harm limit. The compact calibration model did not improve consistently; calibration-only gains were much smaller. Full-minus-compact is also uncertain, so do not claim definitive feature attribution.

These partitions reuse the **same 96 episodes**. They are split-sensitivity checks, not independent replications. The bootstrap averages each episode's paired loss differences across repetitions before resampling episode clusters within condition; it does not count 288 repetitions as independent and does not average probabilities into a new ensemble. It also does not account for the history of architecture selection or refitting uncertainty.

The original partition exactly reproduces the earlier EXP-5 and full-correction OOF arrays and calibration. All 15 low-dimensional calibrations converged without active bounds. Saved native models passed reload checks. Full correction was selected by the prespecified mean-loss rule, before any new validation scoring.

## One frozen official validation comparison

The postprocessor was then refitted once on the original partition's global OOF base predictions. EXP-5 was fitted on all TRAIN data with identity calibration. No validation targets trained parameters. The authoritative validation and test CSVs were generated with the packaged standard-library inference code, and fresh-process reproduction passed before scoring.

| Model | Official local validation points |
|---|---:|
| Existing EXP-5 | 57.475361 / 60 |
| EXP-5 + XGB correction | **57.626713 / 60** |
| Change | **+0.151352** |

Paired episode-bootstrap 95% interval for the point gain: **[-0.025975, +0.347498]**. The gain is modest and uncertain, not proof of superior hidden-test performance or absence of overfitting.

| Condition | EXP-5 ratio | New ratio | Change |
|---|---:|---:|---:|
| stable | 0.030193 | 0.027029 | -0.003163 |
| short_history | 0.089073 | 0.080764 | -0.008309 |
| drift | 0.030315 | 0.028643 | -0.001673 |
| shared_change | 0.026122 | 0.028300 | +0.002178 |

The prespecified practical gate was gain at least 0.10 local points, no condition ratio worsening above 0.005, and verified reproduction. The observed gain is 0.151352 and the largest ratio deterioration is 0.002178 in shared change, so the candidate passes. No settings or blend weights were changed after this confirmation.

Historical tuning/audit points: new **57.6084 / 57.6571**, existing EXP-5 **57.4242 / 57.5412**. Both halves improved, but they have been examined repeatedly and are **not pristine audits**. One additional validation comparison batch was made; no other follow-up candidate was scored on validation.

The 60-point scale is the local organizer kit's actual checker. Its 60/40 metadata and scorer were not modified to match externally pasted 80/20 text. Within this Main comparison, the same official metric and data were used for both models.

## Submission reproduction and runtime

- **43 focused tests pass** across original XGB feature/CV checks, new calibration math, portable features/trees, and the final wrapper. Portable-tree and wrapper tests also pass under Python 3.13 with site packages disabled.
- Calibration analytic gradient agrees with finite differences; the portable calibration map agrees with the NumPy reference to 3.45e-15 on synthetic cases.
- All 768 real TRAIN rows' 1,259 portable features match NumPy exactly in float32. Future guards cover own/peer histories; the wrapper requires the kit's shared-cutoff episode contract and rejects future observations.
- Portable tree evaluation was compared with the initial five correction models over 12,288 TRAIN OOF rows: all 6,144,000 leaf decisions and all raw margins matched native XGBoost; maximum probability difference was 3.72e-8 from softmax arithmetic. This is a separate native-equivalence check. The final CSV is generated with the **portable** runtime, so native predictions are not the reproduction reference.
- The **exact final ZIP** recreates the **exact test CSV byte-for-byte** on Python **3.12.12 and 3.13.9**, with maximum difference **0.0**. No third-party package, network, GPU or retraining is needed at inference. This is a local check, not an organizer-side Linux receipt.
- The test CSV passes the organizer format checker for all **1,536 queries**. Existing baseline leakage tests pass: 21 truncation/shuffle checks plus exact lags, missingness, determinism and future guards.
- Follow-up TRAIN CV took **357.3 s**. Final postprocessor fit: **6.92 s**; base fit: **7.08 s**; validation prediction: **7.71 s**; test prediction: **14.52 s**. Full fit/predict/package preparation: **37.47 s**.
- All 37 original organizer manifest files and both original EXP-5 submission artifacts are unchanged.

SHA-256:

- CSV: `0fd84198a9a520e265a89887542fa11279348369b847420cad4b393bfac73afa`.
- ZIP: `26fc295873d1bcf8952c43c67a9fac4dfecb20716d1a8d1c796c3767bfbed615`.

## Files and commands

Design: `XGB_FOLLOWUP_PLAN.md`. Research: `research/followup_calibration.py`, `research/cv_xgb_followup.py`, `research/finalize_xgb_followup.py`, `research/score_followup_final.py`. Runtime: `house_edge/xgb_features_portable.py`, `house_edge/xgb_portable.py`, `house_edge/followup_predict.py`. The original experiment's source files remain unchanged.

Detailed artifacts: `runs/xgb_followup/cv.json`, per-partition OOF/fold files, and `runs/xgb_followup_final/` with `freeze.json`, `run.json`, `reproduction_checks.json`, `validation_confirmation.json`, and the exact `entry/` source tree.

Commands actually executed include:

```sh
python tests/test_followup_calibration.py
python tests/test_xgb_features.py
python tests/test_xgb_cv.py
python tests/test_xgb_features_portable.py
python tests/test_xgb_portable.py
python tests/test_followup_predict.py
python research/cv_xgb_followup.py --out runs/xgb_followup --reference runs/xgb_cv
python research/finalize_xgb_followup.py --cv runs/xgb_followup --out runs/xgb_followup_final
python check_submission.py runs/xgb_followup_final/submission.csv
python tests/test_leakage.py runs/xgb_followup_final/entry/base_model.json
python tools/verify_entry.py runs/xgb_followup_final/entry.zip runs/xgb_followup_final/submission.csv
python tools/verify_entry.py runs/xgb_followup_final/entry.zip runs/xgb_followup_final/submission.csv --python /Users/stuarts/.local/share/uv/python/cpython-3.13.9-macos-aarch64-none/bin/python3.13
python research/score_followup_final.py --run runs/xgb_followup_final
```

Use fresh output directories for fitting; the scorer deliberately refuses another confirmation into this run. The deployment ZIP includes checkpoints, prediction source, training source/settings, and README; it contains no forecast CSV, private data or credentials.

## Next evaluation

The highest-value next check is the frozen correction procedure versus refitted EXP-5 on the fresh Adaptation data, with training-only selection and a final validation confirmation. The three Main partitions do not replace that fresh evaluation. Keep the existing 90-minute fallback workflow ready and reserve upload time; do not simply reuse Main's CSV or checkpoint against changed query IDs and mechanisms.

Method references motivating the calibration control: [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html) and [Kull et al.](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8ca01ea920679a0fe3728441494041b9-Abstract.html). The win/conditional-size parameterization and practical promotion thresholds are our prospective design choices.
