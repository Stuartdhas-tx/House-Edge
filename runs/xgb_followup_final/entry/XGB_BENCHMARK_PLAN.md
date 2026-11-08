# XGBoost comparison — fixed plan before results

Recorded November 7, 2026, after the organizer announcement supplied by the user: Main now closes Sunday November 8 at 10:30 AM CST; Adaptation runs 10:30 AM–noon (90 minutes). Latest Main Form response counts. This announcement changes the schedule, not the local kit's scoring code or metadata.

Keep `runs/exp5/submission.csv` and `zips/exp5.zip` as the working Main entry. Do not overwrite or upload an experimental candidate automatically.

## Question and fixed candidates

Can a different model family recover structure missed by EXP-5?

1. Standalone five-class XGBoost, depth 3.
2. The same standalone model, depth 5.
3. Fixed probability blends with each standalone model: 90% EXP-5 / 10% XGB and 75% EXP-5 / 25% XGB.
4. One XGB correction model using log probabilities from uncalibrated EXP-5 as its base margin, depth 2. Its training base predictions and corresponding probability features must be inner out-of-fold within each outer training split.

Standalone settings: 300 rounds, learning rate 0.03, minimum child weight 20, L2 20, L1 1, row and column subsampling 0.8, histogram training with 128 bins, CPU with 2 threads. Correction: 100 rounds, depth 2, minimum child weight 40, L2 50, L1 2; otherwise the same settings. Seed 20261107. Objective `multi:softprob`, five classes. No outer-fold early stopping and no parameter sweep after seeing results.

XGBoost fits multiclass log loss; evaluation uses five-class Brier. Training weights balance conditions and episodes, normalized to mean one within each training mask so minimum child weight and regularization operate at their intended scale. No class reweighting changes the probability target.

## Features and data flow

Strictly past machine/episode histories → deterministic features → XGB → five payout probabilities. No machine IDs, episode IDs or condition labels as features. Conditions are permitted only for balancing evaluation and weights.

Features: machine type, current signal only at real query cutoffs (unknown on earlier pseudo-queries), exact tick lags 1–8, missingness, history coverage, payout frequencies over several fixed windows, candidate-rule conditional win frequencies and supports, lag-conditioned payout sizes, and earlier peer observations within the same episode. Missing ticks are unknown. Every feature at tick t excludes all observations at or after t, including on peers. No globally fitted feature transformation.

Correction: inner episode folds fit EXP-5 → inner OOF probabilities become correction training margins/features → fit correction on outer training episodes → predict outer held-out episodes using EXP-5 fitted only on outer training episodes. Using a global OOF table for correction training would leak outer episodes and is forbidden.

## Evaluation

- Train data only: 96 whole episodes, the existing five condition-balanced folds; 768 actual query cutoffs are primary, 12,288 rows with 16 cutoffs per machine are secondary.
- Compare with uncalibrated EXP-5 and the corrected, nested-calibration EXP-5 control. Blends use that calibrated control. Correction uses uncalibrated EXP-5 margins throughout to avoid a further calibration stacking layer.
- Every upstream EXP-5 fit and calibration excludes the outer episodes. No validation targets are read during this comparison.
- Report episode/condition-balanced outcome Brier, all five fold results, four condition differences, and paired within-condition episode-bootstrap intervals. These are surrogate errors, not official technical points; the reference-error normalization is absent.
- Report every fixed candidate. The comparison is exploratory across model families; choosing the best outer-CV result is not an unbiased estimate of the whole selection process. Bootstrap intervals condition on the fitted predictions and do not account for earlier adaptive architecture research.
- Advance at most one candidate to an additional, explicitly logged validation confirmation only if actual-cutoff Brier improves, its paired 95% interval is entirely below zero, and no condition worsens by more than 0.002 Brier. This is a conservative screening rule, not proof against selection bias. Otherwise retain EXP-5 without another validation look.
- A validation confirmation would use the official scorer and paired episode differences; the public validation set has already been examined, so it is not a fresh audit. A replacement additionally needs fresh-process reproduction and valid dependencies before consideration.

## Checks, cost and failure modes

Feature tests must cover future invariance on every machine, exact-lag missingness, pseudo-query signal masking, ID/condition invariance, and determinism. Check valid five-class probabilities, fold boundaries, baseline agreement with the corrected previous run, and save source/input hashes, feature names, versions and parameters.

Expected runtime: several minutes on CPU; measure actual runtime. XGBoost 3.4.1, NumPy 2.4.6, SciPy 1.18.0 and scikit-learn 1.9.0 are installed locally. Native dependency packaging is not yet verified. Do not claim an XGB entry is submission-ready from a CV result alone.

Possible failures: too few independent episodes, noisy outcome labels, machine-specific rules that generic trees cannot infer reliably, signal sparsity in pseudo-queries, overconfident boosting, and correction-stage distribution changes between OOF and full-fit base predictions. A negative result applies to these fixed features/configurations, not every possible boosted model.

Primary references: [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html), [base margins and stacking](https://xgboost.readthedocs.io/en/stable/tutorials/intercept.html), and [saved model format](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html).
