# Experiment log: House Edge, Main round

Conventions:
- **E3** = episode-grouped 5-fold CV on TRAIN pseudo-queries (16 cutoffs per machine, 12,288 rows). The loss is realized 5-class Brier, aggregated like the official metric. Lower is better. Differences in E3 estimate differences in official error.
- **VAL** = official `check_submission.py --split validation` on all 96 validation episodes, plus the kit's tuning and audit halves (48 episodes each).
- Every VAL scoring is a "look" at validation targets and is numbered here.
- Python 3.12.12 (macOS arm64). No seeds are needed: all models are deterministic. Bootstrap seed 314159.

| # | When (CT) | Run | Command | Settings | E3 Brier (OOF) | VAL points (full / tuning / audit) | VAL ratios stable / short / drift / shared | Runtime | Decision |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 14:07 | `baseline-pooled-recent/` | `python run_starter.py --method pooled_recent --out baseline-pooled-recent` | kit defaults | 0.5588 | 30.826 / — / — | 0.958 / 1.001 / 0.946 / 0.880 | 3 s | **Fallback entry** (`baseline-pooled-recent-entry.zip`). Checker valid. |
| look 1 | 14:08 | `runs/probe_hl30/` | `python research/probe.py` | DESIGN.md diagnostic: 3 hard-coded pairs, size vectors from validation targets | n/a | 49.988 / 50.382 / 49.563 | 0.096 / 0.123 / 0.361 / 0.222 | <1 s | Reference only. Not eligible: it uses validation-derived constants. |
| — | 14:09 | `runs/exp1_cv/` | `python research/cv_pair_rule.py` | grid h{8..∞} α{1..16} β{.25..1} κ{2..1e6} | 0.3916 | — | — | 22 s | Optimum on the grid edge (α=1, β=1), so widened the grid |
| — | 14:10 | `runs/exp1_cv_wide/` | same, wider grid | h{6..∞} α{.25..4} β{.75..3} κ{25..200} → **h=12, α=0.5, β=2, κ=50** | **0.3872** (Δ vs pooled_recent −0.171, CI95 [−0.180, −0.164]) | — | — | 22 s | Interior optimum, flat surface. Accepted. |
| look 2 | 14:11 | `runs/exp1/` | `python research/run_model.py --config configs/exp1.json --out runs/exp1` | EXP-1 fit on train only (stdlib) | — | **53.537 / 53.410 / 53.693** | 0.113 / 0.143 / 0.113 / 0.115 | fit 1.1 s, test predict 1.2 s | **Accepted.** Leak tests and equivalence pass. Entry `zips/exp1.zip` reproduces byte-identically on Python 3.12 and 3.13. |

| — | 14:16 | `runs/exp2_cv/` | `python research/cv_change.py` | α{.5,1,2} β{.5,1,1.5} ρ(no change){.5,.8,.95} σ(shared){0,.5} → **α=.5, β=1, ρ=.5, σ=.5** | **0.3829** (EXP-1 0.3872). Pooling alone adds only −0.0001 | — | — | 137 s | Accepted |
| look 3 | 14:19 | `runs/exp2/` | `python research/run_model.py --config configs/exp2.json --out runs/exp2` | EXP-2 change-point + episode pooling | — | **54.469 / 54.614 / 54.323** | 0.080 / 0.133 / 0.100 / 0.093 | fit 5.2 s, test 4.2 s | **Accepted**: all four conditions better than EXP-1. Leak tests (including cross-machine) and equivalence pass. `zips/exp2.zip` verified. |
| — | 14:22 | `runs/exp3_cv`, `runs/exp3_cv_b` | `python research/cv_exp3.py` | Size: Dirichlet κ vs two-regime EM (λ); calibration slope + type intercepts | regime λ=.75 + calibration **0.38151** vs EXP-2 0.38293 (5/5 folds) | — | — | 25 s | Calibration: 5/5 folds. Regime size: −.0007 on all rows (5/5) but +.0005 ± .0007 on real-cutoff rows, which is noise |
| look 4 | 14:24 | `runs/exp3/` | `... --config configs/exp3.json --out runs/exp3` | EXP-2 + regime size (λ .75) + calibration (slope 1.127) | nested OOF 0.38142 | **54.559 / 54.744 / 54.371** | 0.079 / 0.133 / 0.096 / 0.091 | fit 5.2 s, test 3.7 s | +0.13 on tuning, **below the DESIGN gate of +0.3**. Kept as a component because train CV is 5/5 folds and better in all conditions. Superseded by EXP-4. |
| — | 14:25 | — | `python research/size_recency.py` | recency-weighted size counts (h 16–128) | worse on real rows at every h | — | — | 20 s | **Rejected**: the history size regime is static per machine |
| — | 14:26 | — | `python research/third_lag.py` | third-lag / big-payout effects inside the dominant XOR cell (stable train machines) | signed effect ≈ 0.00 at every lag | — | — | 5 s | **Rejected**: no residual structure, so logistic regression and ensembles are skipped |
| — | 14:30 | `runs/final_cv_report.json` | `python research/final_cv_report.py` | consolidated nested train CV of EXP-1/2/3 | see table below | — | — | 79 s | — |
| — | 14:33 | — | `python research/cv_oriented.py 20,40,80,160 0.75,1.0,1.5` | oriented hierarchical cell priors, strength s, β | s=40, β=1: **0.37761** vs EXP-2 0.38279 (5/5 folds, all conditions) | — | — | 150 s | Interior optimum (s 20–40 flat; 80/160 worse) |
| — | 14:36 | `runs/exp4_cv/` | `python research/cv_exp4.py` | oriented + regime size + nested calibration | **0.37656** vs EXP-3 0.38142 (5/5 folds) | — | — | 92 s | **Bug caught and fixed**: the cache key collided across equal-size folds. Results are from the corrected rerun. |
| look 5 | 14:39 | `runs/exp4/` | `... --config configs/exp4.json --out runs/exp4` | EXP-4 = oriented priors (s=40) + EXP-3 size + calibration (slope 1.097) | nested OOF 0.37656 | **55.111 / 55.202 / 55.024** | 0.073 / 0.124 / 0.082 / 0.076 | fit 6.8 s, test 11 s | **FINAL.** +0.46 on tuning (passes the +0.3 gate), all conditions better. Leak tests and equivalence pass; byte-identical reproduction on 3.12 and 3.13; retrain identical. |
| look 6 | 18:47 | — | `python research/headroom.py` | **DIAGNOSTIC ONLY** (nothing fit or selected): error decomposition, episode-bootstrap noise, train-CV calibration | — | true size + our w 57.78; true w + our size 57.08; our w + blind size 54.78 | — | 10 s | Size headroom ≈ +2.7 pts, w headroom ≈ +2.0 pts. Validation resolves paired differences of about ±0.5 pts only. Train-CV all-rows over-predicts validation gains (realized ≈ 58% of predicted); real-cutoff rows are closer. |

## Consolidated train cross-validation (E3; nested; lower is better)

| model | OOF mean | in-sample | gap | per-fold OOF | stable / short / drift / shared (OOF) | real-cutoff rows |
|---|---|---|---|---|---|---|
| pooled_recent | 0.55837 | — | — | .5747 .5280 .5665 .5567 .5677 | .5470 / .5585 / .5738 / .5541 | 0.56261 |
| EXP-1 | 0.38694 | 0.38686 | +0.00008 | .3952 .3709 .3958 .3817 .3922 | .3682 / .4014 / .4016 / .3766 | 0.36795 |
| EXP-2 | 0.38279 | 0.38278 | +0.00001 | .3932 .3676 .3922 .3754 .3863 | .3632 / .4002 / .3960 / .3718 | 0.36515 |
| EXP-3 | 0.38142 | 0.38115 | +0.00027 | .3916 .3665 .3920 .3733 .3843 | .3623 / .4000 / .3938 / .3696 | 0.36370 |
| **EXP-4** | **0.37656** | 0.37630 | +0.00026 | .3862 .3616 .3861 .3690 .3808 | .3611 / .3899 / .3884 / .3669 | **0.36064** |

Paired per-fold deltas:
- EXP-2 − EXP-1: −.0020 −.0033 −.0036 −.0063 −.0059
- EXP-3 − EXP-2: −.0016 −.0011 −.0002 −.0021 −.0020
- EXP-4 − EXP-3: −.0054 −.0050 −.0059 −.0043 −.0035

All are 5/5 negative. Fold-to-fold sd of the OOF level is about 0.011, which is fold difficulty: pooled_recent's is 0.018.

## Final submission (Main)

- **CSV:** `runs/exp4/submission.csv` (SHA-256 `f07a7bd1…b1fb`), valid for 1,536 test queries.
- **ZIP:** `zips/exp4.zip` (SHA-256 `24657878…d458`). Source folder: `entries/exp4/`.
- **Saved parameters:** `runs/exp4/model.json`. **Config:** `configs/exp4.json`. **Validation predictions:** `runs/exp4/validation_predictions.csv`.
- **Reproduce:**
  ```
  python research/run_model.py --config configs/exp4.json --out <new dir>
  python tools/verify_entry.py zips/exp4.zip runs/exp4/submission.csv [--python python3.13]
  ```
- **Fallback chain**, all verified: `baseline-pooled-recent-entry.zip`, `zips/exp1.zip`, `zips/exp2.zip`, `zips/exp3.zip`. Each ZIP pairs with its `runs/<name>/submission.csv`.

## EXP-1 details

- **E3 nested CV.** Hyperparameters were chosen on 4 folds and scored on the 5th. Held-out Brier per fold: .3952 / .3713 / .3967 / .3817 / .3926. Mean .3875, sd .0091.
- **Held-out minus in-fold gap per fold:** +.010 / −.020 / +.012 / −.007 / +.007. The mean is about 0, so there is no overfitting signal.
- **In-sample (all train) vs OOF by condition:** .3681/.3682 stable, .4012/.4014 short, .4015/.4016 drift, .3766/.3766 shared.
- **Selections across folds were stable:** h=12, β=2.0 every time; α ∈ {0.25, 0.5}; κ ∈ {50, 100}.
- **Test CSV:** `runs/exp1/submission.csv`, SHA-256 `198a8d66…3464`. **ZIP:** `zips/exp1.zip`, SHA-256 `88ebfe13…1d66`.

## Research after EXP-4 (18:28–18:54 CT, from file timestamps; train data and validation HISTORIES only, no targets)

| Script | Question | Finding | Status |
|---|---|---|---|
| `research/size_rules.py`, `size_rules2.py` | Do lagged payout sizes drive the size regime? | 28% of machines significant vs 5% expected; top features: payout ≥ 2 at lags 1, 2, 4; direction varies by machine | Confirmed |
| `research/r1_size_structure.py` | Which lag, how many machines, stable over time? | Size lag = **first lag of the win pair** for 69/78 clear machines (88%; chance 12.5%). Detection rises with #wins (18% → 66%), so non-detection is power. No sign flips between halves, including drift/shared. | Confirmed |
| `research/r1_replicate.py validation` | Replicates on independent data? | Validation histories: 77/89 (87%); detection 19% → 77%; flips ≤ 2% | **Replicated** |
| `research/r2_size_prototype.py` | Value partition and achievable gain | "Big" = payout ≥ 2 (0, 1 vs 2, 5, 20: −.086 vs +.086). Prototype with EXP-4 win probabilities fixed: Brier −0.0078 all rows, **−0.0120 real-cutoff rows, 5/5 folds on both**. Cell-structured EM on TRAIN recovers the two size vectors per type to about ±0.01. | Next: EXP-5 |
| `research/r3_rule_priors.py` | Is the win rule predictable from type, signal or episode-mates? | Pair uniform over (1,3)/(2,5)/(4,7) by type and signal; no within-episode sharing; orientation ≈ 50% (jackpot 39% XOR). **Cell levels rise monotonically with signal** (steady low cell .095/.137/.207). | Candidate: EXP-6 (small) |

## EXP-5: size regime tied to the win rule (18:56–19:02 CT)

| # | When | Run | Command | Settings | Train CV (OOF, 5-fold, paired vs EXP-4) | VAL points (full / tuning / audit) | VAL ratios stable / short / drift / shared | Runtime | Decision |
|---|---|---|---|---|---|---|---|---|---|
| — | 18:59 | `runs/exp5_cv/` | `python research/cv_exp5.py` | No new tuned hyperparameters. Size lag = win pair's first lag; "big" = payout ≥ 2; per-machine orientation; cell-structured EM; calibration refit on OOF rows | all rows 0.36842 vs 0.37658 (Δ −0.0081, **5/5 folds**); real-cutoff rows 0.34840 vs 0.36057 (Δ −0.0124, **5/5**); better in every condition; in-sample 0.36817 (gap +0.00025) | — | — | 42 s | Passes train gates |
| look 7 | 19:01 | `runs/exp5/` | `python research/run_model.py --config configs/exp5.json --out runs/exp5` | EXP-5 | — | **57.475 / 57.424 / 57.541** | **0.030 / 0.089 / 0.030 / 0.026** | fit 7.5 s, test 11.4 s | **FINAL CANDIDATE.** Paired bootstrap vs EXP-4: **+2.36 pts [+2.02, +2.71], P(gain) = 1.00**. Equivalence (4.7e-14) and leakage tests pass; byte-identical on 3.12 and 3.13; retrain identical; kit files unmodified. |

**Final Main entry (replaces EXP-4):**
- CSV: `runs/exp5/submission.csv` (SHA-256 `286311e7…c699`)
- ZIP: `zips/exp5.zip` (SHA-256 `f94dd698…dc4c`)
- Fallback: `zips/exp4.zip` with `runs/exp4/submission.csv`

## Post-upload audit and round-two prep (19:15–19:30 CT)

- **Audit (no new models, no selection):**
  - All saved validation CSVs were re-scored with `check_submission.py`, and every logged number matched.
  - An independent paired episode bootstrap of EXP-5 vs EXP-4 gave +2.36 [+2.03, +2.70]. EXP-5 had lower error in 24/24 stable, 21/24 short, 24/24 drift and 23/24 shared episodes.
  - `zips/exp5.zip` was re-verified byte-identical on Python 3.12.12 and 3.13.9. Leakage and equivalence tests pass, and all 37 kit files are unchanged.
- **Main upload:** the team uploaded EXP-5 at about 19:20 CT.
- **Round-two tooling added:**
  - `tools/adapt_setup.py`: copies our code into the new kit, profiles the data, and blocks on unknown types or payouts.
  - `tools/config_from_cv.py`: CV calibration → config. It regenerates `configs/exp5.json` and `configs/exp4.json` exactly from their cv.json files.
  - `tools/make_entry.py`: the README now uses the round from metadata.
  - `ROUND2_PLAYBOOK.md`: the step-by-step plan.
- **Dry run** on a clean organizer-only copy of the kit:
  - Baseline → EXP-5 refit → entry → verify → cv_exp5 recalibration → refit → leak test → r1/r3 took 2.5 min.
  - The output was byte-identical to the uploaded CSV.
  - The EXP-4 fallback chain (`final_cv_report.py` → `cv_exp4.py` → `config_from_cv.py`) was also run.

## EXP-6: hi/lo cell levels per (type, signal) (19:26–19:33 CT)

| # | When | Run | Command | Settings | Train CV (paired vs EXP-5) | VAL points (full / tuning / audit) | VAL ratios stable / short / drift / shared | Runtime | Decision |
|---|---|---|---|---|---|---|---|---|---|
| — | 19:27 | `runs/exp6_cv/` | `cd research && python cv_exp6.py` | Oriented hi/lo levels per (type, query signal), fit on training machines with that signal. Signal levels apply on real-cutoff rows only; earlier pseudo-query rows keep type levels (signal is never attached to them). No new tuned hyperparameters; calibration refit on OOF rows | real-cutoff rows 0.34764 vs 0.34840 (Δ −0.00071, **4/5 folds**); all rows Δ −0.00004 (4/5); real-row shared_change slightly worse (.3019 vs .3000) | — | — | 85 s | Small gain, below the 5/5 bar. Built for one confirmation look. |
| look 8 | 19:31 | `runs/exp6/` | `python research/run_model.py --config configs/exp6.json --out runs/exp6 --no-score`, then `check_submission.py` | EXP-6 | — | **57.549 / 57.419 / 57.702** (EXP-5: 57.475 / 57.424 / 57.541) | 0.0285 / 0.0887 / 0.0278 / 0.0253 | fit 8.0 s, test 11.3 s | Paired episode bootstrap vs EXP-5: **+0.07 [−0.01, +0.15]**, P(gain) 0.965. Tuning half −0.005, **fails the +0.3 gate**. Equivalence passes (4.9e-14) and leakage tests pass. `zips/exp6.zip` (`e82bd923…4ecc`) with `runs/exp6/submission.csv` (`50a3ef47…315a`) reproduces byte-identically on 3.12 and 3.13. The `query_tables` refactor leaves EXP-5 test and validation CSVs byte-identical. |

## Research pass after EXP-6 (19:49–19:58 CT)

| # | Script | Data used | Finding | Status |
|---|---|---|---|---|
| look 9 (diagnostic) | `research/headroom_exp5.py` | validation targets; nothing fit or selected | With **true w** and our sizes, EXP-5 would score **59.66**; with our w and **true sizes**, only 57.77. So about 2.2 of the 2.5 missing points are win probability. True size vectors are exactly 2 per type (the EXP-5 size structure is right). Wrong-rule misses are rare (0–7 per condition) except in short_history (7 misses carry 56% of its w error). Elsewhere the error is level error (RMS 0.056–0.064). | Decisive diagnosis |
| look 9 (cont.) | `research/residual_scan.py` | validation targets | No observable cutoff feature explains the w residual (lagged wins, big payouts, 20× payouts, recent rate: all \|t\| < 2.5) except **jackpot low cell, signal −1** (t = −5.9): EXP-6's training estimate is 0.037 against a true value of about 0.009. | Small absolute effect |
| look 9 (cont.) | `research/level_drift.py` | validation targets | In stable machines the full-history cell rate predicts the true level best (RMSE 0.064, vs 0.083 for the last 24 ticks). **Levels do not drift**, so the error is estimation noise. | No recency fix |
| — | `research/cv_exp7_grid.py 0.5,0.3 40,80` | train CV only | Contrast threshold × prior strength on EXP-6: real-row OOF 0.34764 (0.5/40, which equals EXP-6), 0.34816 (0.5/80), 0.34763 (0.3/40), 0.34790 (0.3/80). **Flat.** | Rejected: config knobs exhausted |
| — | `research/level_correlation.py` | train and validation histories only (condition labels used to keep stable machines) | Split-half: hi and lo cell levels correlate **positively** for balanced (+0.31 train / +0.47 val) and steady (+0.17 / +0.37). Jackpot is unreliable because hi/lo are misidentified. A first pass on all conditions gave about −0.7, an artifact of rule flips. | Candidate for round two (≈ +0.1 pt, needs a joint prior) |

Main decision is unchanged: no new technique is shippable before 21:00. EXP-6 remains the best measured model (+0.07 over EXP-5, within noise).


## Joint win-and-size inference (2026-11-07, train-only experiment)

Implemented `house_edge/joint_size.py` with joint likelihood and predictive averaging; full details and executed commands are in `JOINT_INFERENCE_RESULTS.md`. No new validation look.

| Variant | Corrected nested real-cutoff Brier | Decision |
|---|---:|---|
| EXP-5, corrected nested calibration | 0.34854627 | Control |
| Joint averaging only | 0.34855986 | No adoption evidence |
| Joint averaging, common events | 0.34932137 | No adoption evidence |
| Common-event joint likelihood | 0.34951704 | No adoption evidence |
| All-win joint likelihood, missing-lag mixture | 0.34895715 | No adoption evidence |

The all-win/missing-mixture follow-up has paired ΔBrier +0.00041088, CI95 [-0.00012271, +0.00130084]. Most deterioration is short history. The initial common-event restriction lost useful size observations. Both win and size predictions are cross-fitted inside every outer calibration fold.

Checks: 15 unit tests pass on Python 3.12/3.13; all-five-variant NumPy equivalence max 1.074e-13; existing leakage tests pass; experimental test CSV valid; all 37 organizer files unchanged. Final nested CV 106.8 s; experimental fit 8.0 s/test prediction 12.4 s. No new ZIP/reproduction or upload because training evidence does not support replacement. Keep EXP-5.

## Round-two one-command rehearsal (21:50–21:56 CT)

- **Command:** `python tools/adapt_setup.py <fresh organizer-only kit copy> --allow-round main`, then `python tools/round2_run.py --allow-round main` inside that kit.
- **Total time:** 5 min 49 s.
- **Timeline:**
  - Baseline ready at 4 s.
  - First model entry (`a_exp5_maincfg`) ready at 56 s.
  - `a_exp5_recal` ready at 158 s.
  - `a_exp6` ready at 317 s.
- **Output check:** the rebuilt CSVs are byte-identical to the uploaded EXP-5 (`286311e7…c699`) and to EXP-6 (`50a3ef47…315a`). Every entry passed the leakage and equivalence tests and a fresh reproduction check on Python 3.13.9.
- **Pick rule** (fixed in the script's docstring): EXP-6 was better in only 4/5 folds, so it was not picked. **PICK = a_exp5_recal**, as expected.
- **Schedule** (confirmed by organizers): Main closes Sun 10:30 CT (resubmission allowed); round two runs 10:30–12:00 CT.


## XGBoost benchmark after the deadline extension (2026-11-07)

Fixed plan: `XGB_BENCHMARK_PLAN.md`; full results and executed commands: `XGB_BENCHMARK_RESULTS.md`. TRAIN only; no new validation look. Main now closes Sunday 10:30 AM CST; Adaptation ends noon.

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

Correction primary delta -0.00115837, CI95 [-0.00330849, +0.00099590], 4/5 folds improved. All seven candidates fail the predefined uncertainty screen. Keep EXP-5; correction is promising but inconclusive.

12 tests pass. 15 saved fold models replay within 2.22e-16; baseline agrees with prior corrected CV. Runtime 353.5 s. No XGB ZIP, validation score, or upload. Native dependency packaging is unverified. All 37 organizer files and EXP-5 artifacts are unchanged. Revised fallback workflow rehearsed on Main data: valid format, Python 3.13.9 byte-identical reproduction, 3.54 s; no Adaptation entry or upload.


## Focused XGB follow-up and one validation confirmation (2026-11-07)

User requested continued improvement with overfitting controls. Prospective finite plan: `XGB_FOLLOWUP_PLAN.md`; full report: `XGB_FOLLOWUP_RESULTS.md`. Original experiment/gate preserved.

| Model | Original partition | Partition A | Partition B | Repeat mean |
|---|---:|---:|---:|---:|
| EXP-5 control | 0.34854627 | 0.34852804 | 0.34865508 | 0.34857646 |
| Win + type/size calibration | 0.34845079 | 0.34852573 | 0.34863026 | 0.34853560 |
| Compact XGB calibration | 0.34834387 | 0.34818002 | 0.34923510 | 0.34858633 |
| Full XGB correction | 0.34738791 | 0.34810399 | 0.34784097 | 0.34777762 |
| Half correction / half EXP-5 | 0.34774893 | 0.34807627 | 0.34801849 | 0.34794790 |

Selected full correction by the fixed exploratory rule. Repeat-mean real Brier delta -0.00079884, CI95 [-0.00270825, +0.00111787]; same 96 episodes, not independent repeats.

One frozen validation batch: **57.626713/60**, gain **+0.151352** over EXP-5; paired point interval [-0.025975, +0.347498]. Historical tuning/audit 57.6084/57.6571, both previously examined. Passes practical gate; gain remains uncertain.

| Condition | EXP-5 ratio | New ratio | Change |
|---|---:|---:|---:|
| stable | 0.030193 | 0.027029 | -0.003163 |
| short_history | 0.089073 | 0.080764 | -0.008309 |
| drift | 0.030315 | 0.028643 | -0.001673 |
| shared_change | 0.026122 | 0.028300 | +0.002178 |

**Recommended replacement, not uploaded:** `runs/xgb_followup_final/submission.csv` + `runs/xgb_followup_final/entry.zip`. Both original EXP-5 artifacts retained. Final ZIP reproduces byte-for-byte on Python 3.12/3.13, standard library only; format and leakage checks pass. 43 focused tests pass. Follow-up CV: 357.3 s; final fit/predict/package: 37.47 s; test inference: 14.52 s. All 37 organizer files are unchanged.
CSV SHA-256 `0fd84198a9a520e265a89887542fa11279348369b847420cad4b393bfac73afa`; ZIP `26fc295873d1bcf8952c43c67a9fac4dfecb20716d1a8d1c796c3767bfbed615`. Latest received Main response before Sunday10:30AM CST counts; no automatic upload was performed.

## EXP-5 ZIP with team name (Sun 2026-11-08, about 00:30 CT)

- `entries/exp5_razor/`: a copy of `entries/exp5/`. Only the README changed: the title adds "(Team Razor, Round 1)", and two lines read "Team name: Razor" and "Round: 1 (main)". Code, `model.json` and the forecast are unchanged.
- `python build_entry.py --source entries/exp5_razor --out zips/exp5_razor.zip`, SHA-256 `58ade032…62e0cb`.
- `tools/verify_entry.py` gives byte-identical reproduction of `runs/exp5/submission.csv` (`286311e7…c699`) on Python 3.12.12 and 3.13.9. `check_submission.py` reports the format as valid.
- The uploaded `zips/exp5.zip` (`f94dd698…dc4c`) and `entries/exp5/` are kept unchanged. Nothing was uploaded.
