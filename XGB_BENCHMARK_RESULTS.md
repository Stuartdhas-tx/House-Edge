# XGBoost benchmark: results and decision

2026-11-07, following the organizer's deadline extension. **Keep EXP-5 as the Main entry.** Standalone XGBoost was worse, fixed blends were inconclusive, and a small XGB correction improved mean training CV but failed the predefined uncertainty gate. No new validation target score was computed and nothing was uploaded.

The confirmed schedule is Main through **Sunday November 8, 10:30 AM CST**, then **90 minutes of Adaptation until noon**. Latest Main response counts. `ROUND2_PLAYBOOK.md` has the revised operational plan. Organizer files and local 60/40 scoring metadata were not modified.

## What was tested

The fixed plan is `XGB_BENCHMARK_PLAN.md`. Two standalone multiclass models (depth 3 and 5, 300 rounds), each blended at fixed 10% and 25% weights with EXP-5, plus one depth-2, 100-round correction model. No early stopping on held-out folds or parameter sweep followed the results.

Features include exact payout/win/big lags 1–8, fixed history windows, coverage and missingness, local rule-conditioned counts and rates, size-conditioned counts, machine type, available current signal, and earlier peer histories. There are **1,259 features**, no learned global feature transform, and no ID or condition inputs. The pseudo-query signal is missing, not copied backward from the real query.

Five whole-episode folds cover **96 episodes**, **768 actual cutoffs**, and **12,288 rows** including earlier pseudo-cutoffs. Real-cutoff Brier is primary. Conditions/episodes have balanced weights. XGBoost trains log loss and is evaluated with Brier. The outcome labels are noisy; these numbers omit official reference-error normalization and are **not contest points**.

The EXP-5 control uses corrected nested calibration. Both win and conditional size predictions used to train calibration exclude the outer held-out fold. Correction training uses uncalibrated EXP-5 inner-OOF probabilities for both appended features and log-probability base margins; every upstream fit excludes the outer and inner episodes. The outer prediction uses EXP-5 fitted only on outer training episodes. No global OOF stacking leakage is used.

## Results

Lower Brier is better. Positive changes are worse.

| Model | Real-cutoff Brier | Change vs EXP-5 | All-snapshot Brier |
|---|---:|---:|---:|
| EXP-5, corrected nested calibration | 0.34854627 | +0.00000000 | 0.36842761 |
| Standalone XGB, depth 3 | 0.36137424 | +0.01282797 | 0.38060866 |
| Standalone XGB, depth 5 | 0.36215042 | +0.01360415 | 0.38105979 |
| 90% EXP-5 / 10% depth-3 XGB | 0.34859161 | +0.00004534 | 0.36825569 |
| 90% EXP-5 / 10% depth-5 XGB | 0.34855099 | +0.00000472 | 0.36824675 |
| 75% EXP-5 / 25% depth-3 XGB | 0.34917523 | +0.00062895 | 0.36857698 |
| 75% EXP-5 / 25% depth-5 XGB | 0.34912294 | +0.00057666 | 0.36857715 |
| EXP-5 plus XGB correction | 0.34738791 | -0.00115837 | 0.36800372 |

The correction model's primary difference is **-0.00115837**, with paired within-condition episode-bootstrap 95% interval **[-0.00330849, +0.00099590]**. It improved 4/5 folds. Condition changes (negative is better):

| Condition | Correction minus EXP-5 |
|---|---:|
| Stable | +0.00041379 |
| Short history | -0.00169314 |
| Drift | -0.00292416 |
| Shared change | -0.00042996 |

This is a promising correction idea, not an established improvement. The interval includes zero. The predeclared screen required an interval entirely below zero and no condition regression above 0.002; **none of the seven XGB candidates passed**. Do not turn a favorable mean into a claim that XGB has beaten EXP-5. Conversely, these results do not establish that every possible XGBoost feature set or configuration is worse.

The uncertainty intervals condition on these fitted OOF predictions. They do not account for refitting, earlier human model selection, or selection among all tested candidates. Main validation and its historical halves have already been examined; neither is a pristine audit. No additional validation look was spent on this inconclusive comparison.

## Verification, runtime, and submission status

- **12 focused tests passed**: eight feature invariants plus four tests for nested correction exclusions, balanced weights, zero-tree base-margin identity, and invalid probabilities. Feature tests cover future payouts on own/peer machines, exact missing lags, pseudo-query signal masking, ID/condition invariance, and determinism.
- **All 15 saved fold models replay their held-out predictions**; maximum probability difference **2.22e-16**. Fixed blends and stored EXP-5 calibration also replay successfully. This verifies research artifacts, not an organizer XGB ZIP.
- Both EXP-5 control OOF arrays match the previous corrected `runs/joint_all_cv` reference within the verifier's 1e-12 tolerance. Row/fold alignment and source/input hashes match.
- Training feature construction: **1.37 seconds**, roughly 59 MiB. Complete benchmark: **353.5 seconds** (5.9 minutes), CPU with two XGB threads.
- Research environment: Python 3.12.12, XGBoost 3.4.1, NumPy 2.4.6, SciPy 1.18.0.
- All **37/37 organizer files** retain their manifest hashes. Existing EXP-5 CSV and ZIP are unchanged.
- **No XGB submission CSV/ZIP was produced or uploaded.** Native XGB dependency packaging and Python 3.13/Linux reproduction remain unverified. A saved research fold model is not a deployable final entry.

Existing EXP-5 official local validation remains **57.475361/60**, with stable / short / drift / shared loss ratios **0.030193 / 0.089073 / 0.030315 / 0.026122**. Historical tuning/audit scores **57.4242 / 57.5412** are descriptive, not independent confirmation. Hidden test performance remains unknown.

Current Main artifacts:

- `runs/exp5/submission.csv` — SHA-256 `286311e7c5aaa4947a886ae4ec13cdc4d7a7f8506daa9d427d0f792ea78ac699`.
- `zips/exp5.zip` — SHA-256 `f94dd6980f47bfa31fd93ca3cf8a1962168b8c01274388bcf6fde38d5bc1dc4c`.

## Files and executed commands

Implementation: `research/xgb_features.py`, `research/cv_xgb.py`. Tests: `tests/test_xgb_features.py`, `tests/test_xgb_cv.py`. Replay: `research/verify_xgb_cv.py`. Results and 15 saved fold models: `runs/xgb_cv/` (`cv.json`, `manifest.json`, `oof_predictions.npz`, `oof_losses.csv`, `verification.json`).

```sh
python tests/test_xgb_features.py
python tests/test_xgb_cv.py
python research/cv_xgb.py --out runs/xgb_cv --include-correction --save-fold-models
python research/verify_xgb_cv.py --run runs/xgb_cv --reference runs/joint_all_cv
```

Use a fresh output directory for a repeat. Do not rerun with adjusted settings until a new hypothesis and decision rule are recorded; this benchmark is complete.

## Adaptation preparation

The updated playbook targets a verified fallback by 10:45, training-only selection before one fresh validation confirmation, completed final artifacts by 11:30, and upload by 11:40. The new baseline fit/package/reproduce path was rehearsed on **Main data only** in `runs/adaptation_fallback_rehearsal/`. It passed the format checker and fresh-process Python 3.13.9 reproduction byte-for-byte in 3.54 seconds, without reading validation targets. Those rehearsal files are not an Adaptation entry.

The next highest-value evaluation is refitting the frozen EXP-5 procedure on the fresh Adaptation training episodes and checking it once on untouched Adaptation validation. The correction model remains a candidate for a separately specified future experiment, not a replacement selected from this inconclusive result.

Technical references used for the implementation: [XGBoost probability and tree parameters](https://xgboost.readthedocs.io/en/stable/parameter.html), [base margins for stacking](https://xgboost.readthedocs.io/en/stable/tutorials/intercept.html), [model save/load](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html).
