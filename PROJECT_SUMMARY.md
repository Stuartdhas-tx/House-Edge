# House Edge: what we did, what we found, and where everything is

**Current status update, November 7, 2026:** The recommended Main replacement is now **EXP-5 + XGBoost correction**, with **57.626713/60** local validation points, versus EXP-5's 57.475361. Exact files: `runs/xgb_followup_final/submission.csv` and `runs/xgb_followup_final/entry.zip`. They pass format/leakage checks and reproduce byte-for-byte on Python 3.12/3.13 using only the standard library. **They have not been uploaded**; the recorded upload remains EXP-5 (`runs/exp5/submission.csv`, `zips/exp5.zip`). The gain is modest and uncertain (paired point interval -0.026 to +0.347). See `XGB_FOLLOWUP_RESULTS.md` for the prospective selection rule, full results and limitations.

The organizer announcement supplied by the user moves Main's deadline to **Sunday November 8 at 10:30 AM CST**, with Adaptation running **10:30 AM–noon**. The latest Main Form response counts. See `ROUND2_PLAYBOOK.md` for the revised workflow.

**Evaluation correction:** This is a historical summary. Its statements that validation was used only for confirmation and that all calibration was independently cross-fitted are too strong. Repeated validation diagnostics informed model development, and the earlier calibration CV indirectly reused outer-fold information. The joint-inference experiment introduced corrected nested calibration for both win and size predictions; its comparable EXP-5 real-cutoff Brier is **0.34854627**. See `JOINT_INFERENCE_RESULTS.md`. Neither previously examined validation half is a pristine audit, and the amount of selection optimism is unknown. The reported **57.475361/60** is a measured local validation score, not a hidden-test prediction. No independent XGBoost result existed when the original summary below was written.

**First XGBoost benchmark:** Fixed standalone models scored 0.361374 / 0.362150 real-cutoff training-CV Brier versus EXP-5's 0.348546. Small blends were inconclusive. A correctly cross-fitted XGB correction reached **0.347388**, but its paired improvement interval crossed zero; it did not pass that experiment's predefined screen. The later finite follow-up and one frozen validation confirmation are documented separately in `XGB_FOLLOWUP_RESULTS.md`; the first result has not been retrospectively reclassified.

TAMU Datathon 2026 · Main round · written 2026-11-07, 19:10 CT.
Companion files: `DESIGN.md` (the original plan) and `EXPERIMENT_LOG.md` (every run, with commands and timestamps).

---

## 1. One-paragraph summary

We went from the organizer baseline (**30.83 / 60** on validation, `pooled_recent`) to **57.48 / 60** (EXP-5). The steps were:
1. Analyse the data to find how the simulator actually works.
2. Encode that mechanism directly in a small Bayesian model written in plain Python.
3. Select every setting with cross-validation on the training data, using validation only to confirm.

The final model needs no third-party libraries, predicts the test set in about 11 seconds, and reproduces byte-for-byte in a fresh process on Python 3.12 and 3.13.

| Model | What it adds | Validation points (full / tuning half / audit half) |
|---|---|---|
| `pooled_recent` (organizer baseline) | recency-weighted payout frequencies | 30.83 |
| EXP-1 | pair-rule win model with rule averaging | 53.54 / 53.41 / 53.69 |
| EXP-2 | + change points and shared changes across an episode | 54.47 / 54.61 / 54.32 |
| EXP-3 | + payout-size regimes and calibration | 54.56 / 54.74 / 54.37 |
| EXP-4 | + oriented hierarchical (high/low) cell priors | 55.11 / 55.20 / 55.02 |
| **EXP-5 (final)** | + payout-size rule tied to the win rule | **57.48 / 57.42 / 57.54** |

---

## 2. The problem in plain terms

- Each query asks for the probabilities of the next spin paying 0×, 1×, 2×, 5× or 20× for one machine, at a given tick.
- The score compares our five probabilities with the simulator's **true** probabilities, not with the one outcome that happened.
- Squared error is averaged per episode, then per condition. Each of the four conditions is divided by the error of a frozen baseline:
  - `stable`
  - `short_history`
  - `drift`
  - `shared_change`
- `points = 60 / (1 + R)`, where R is the average of the four ratios. Matching the baseline gives 30; perfect gives 60.
- **Rules we respected throughout:**
  - Only ticks strictly before the cutoff are used, on every machine, including the episode's other machines.
  - Missing ticks are "unknown", never zero.
  - Condition labels and validation targets are never model inputs.
  - Episodes are kept whole when splitting data.

---

## 3. What we discovered about the simulator

All of these were measured on the data, not assumed.

1. **The answer has two parts:** `P(win)` × `which win size`. Each machine type has exactly two possible win-size distributions.
2. **Win chance follows an XOR rule on two earlier spins.** A machine wins mostly when its results 2 and 5 ticks ago differ, or mostly when they match. Which one depends on the machine. In this round the pairs are (1,3), (2,5) and (4,7).
3. **The conditions differ in timing:**
   - `stable`: the rule never changes.
   - `drift`: the rule flips at a machine-specific time.
   - `shared_change`: the rule flips at almost the same tick on all 8 machines in an episode.
   - `short_history`: identifiable from history coverage alone, which is legitimate to use.
4. **Machines don't directly influence each other.** Neighbours help only by revealing when a shared change happened.
5. **Payout size follows its own rule (the EXP-5 breakthrough).**
   - The size regime at tick t is set by whether the payout at the *first lag* of the machine's win pair was ≥ 2×. For a (2,5) machine that means "was the payout 2 ticks ago big?".
   - The direction is fixed per machine and never flips.
   - Found on training data (88% of clear machines) and replicated on validation **histories** (87%) without reading validation answers.

---

## 4. The final model (EXP-5), in steps

Code: `house_edge/` (Python standard library only). Saved parameters: `runs/exp5/model.json`.

1. **Win chance: rule averaging.** We consider 37 candidate rules: none, 8 single lags, and 28 XOR pairs of exact lags 1–8. Each is weighted by how well it explains the machine's history; we never pick just one.
2. **Change points.** Each rule is also scored as "changed d ticks ago" (d = 8–200) or "no change", with separate statistics before and after the change. Only the post-change part predicts.
3. **Episode pooling.** We compare "all 8 machines changed at the same moment" against "each changed on its own", using only observed histories.
4. **Hierarchical priors.** Each rule has a "high" cell and a "low" cell, pulled toward typical levels for the machine type. Those levels are learned from training machines, e.g. steady 0.93 / 0.16. This mainly helps short histories.
5. **Rule popularity.** How common each rule is gets learned from training histories (empirical Bayes).
6. **Calibration.** Four numbers: a slope of 1.094 plus a per-type offset, fitted on out-of-fold training predictions.
7. **Payout size.** Use the win model's belief about which rule the machine has, take that rule's first lag, and check whether that payout was ≥ 2×. The machine's own past wins tell us which direction applies. The two size distributions per type are learned by EM on training histories.
8. **Output:** `[1 − w, w × size distribution]`.

Main settings, all chosen by training-data cross-validation:
- `alpha = 0.5`
- `beta = 1.0`
- P(no change) = 0.5
- shared-change prior = 0.5
- prior strength = 40

---

## 5. How we avoided overfitting

- **Selection on training data only.**
  - We used 5-fold cross-validation grouped by whole episodes, on training "pseudo-queries": each training machine predicted at its last 16 ticks, with every machine truncated at that tick.
  - The realized-outcome Brier score is a fair stand-in for the official error, because differences in Brier equal differences in official error on average.
  - Everything fitted was refit without the held-out fold. That includes base rates, rule priors, levels, size distributions and calibration.
- **Every adopted step won in 5/5 folds and in every condition.** The generalization gap (out-of-fold minus in-sample) never exceeded 0.0003.

| Model | Train CV out-of-fold Brier (lower is better) | Real-cutoff rows only |
|---|---|---|
| pooled_recent | 0.5584 | 0.5626 |
| EXP-1 | 0.3869 | 0.3680 |
| EXP-2 | 0.3828 | 0.3652 |
| EXP-3 | 0.3814 | 0.3637 |
| EXP-4 | 0.3766 | 0.3606 |
| **EXP-5** | **0.3684** | **0.3484** |

- **Validation was for confirmation only.**
  - We made 7 looks in total. One was a reproduction of the design-phase probe, one was purely diagnostic, and five scored models.
  - No setting was tuned on validation.
  - The tuning and audit halves always agreed within about 0.4 points.
- **Noise awareness.** Paired episode-bootstrap of validation gains:

  | Step | Gain | 95% interval |
  |---|---|---|
  | EXP-1 → 2 | +0.93 | [+0.52, +1.46] |
  | EXP-2 → 3 | +0.09 | [−0.05, +0.22] (noise-level) |
  | EXP-3 → 4 | +0.55 | [−0.10, +1.12] |
  | **EXP-4 → 5** | **+2.36** | **[+2.02, +2.71]** |

- **Honest caveats.**
  - Validation answers informed the early design analysis, and 7 looks were made, so 57.5 is slightly optimistic.
  - Episode sampling alone suggests a test score of roughly 57–58.
  - Training CV tends to overstate gains on all rows. Real-cutoff rows track validation better.
  - EXP-3 missed our own pre-set gate (+0.3 points on the tuning half). It was kept on training-CV evidence and later superseded.

---

## 6. Tests and verification (all passing for EXP-5)

| Check | Result |
|---|---|
| Leakage tests (`tests/test_leakage.py`): future ticks ignored, random future payouts on every machine in the episode ignored, exact-lag reading, missing ≠ zero, kit rejects future rows, deterministic | pass |
| Shipped stdlib model = numpy research code (`tests/test_equivalence_exp5.py`) | max difference 4.7e-14 |
| Fresh-process reproduction from the ZIP in a read-only folder (`tools/verify_entry.py`) | byte-identical on Python 3.12.12 and 3.13.9 |
| Retraining from scratch | byte-identical model and CSVs |
| Organizer checker (`check_submission.py`) | valid (validation 768, test 1,536 queries) |
| Organizer ZIP contract | pass |
| Organizer files unchanged (`SHA256.json`) | 37/37 unchanged |
| Runtime | fit 7.5 s, test prediction 11.4 s |

---

## 7. Ideas tested and rejected (with evidence)

- **Recency-weighted payout sizes:** worse at every half-life on real-cutoff rows.
- **A third lag or big-payout effect on win chance inside the XOR cells:** effect ≈ 0, so logistic regression and ensembles were not pursued.
- **Predicting a machine's rule pair from type, signal or episode-mates:** no relationship.
- **Transformers, GBMs, neural nets:**
  - The mechanism is already captured explicitly.
  - The remaining error is mostly data-limited (short histories).
  - They would add dependency and reproduction risk.
- **Deferred to round two:** signal-dependent cell levels (expected +0.1 to +0.4) and a jackpot orientation prior (tiny).

---

## 8. Bugs found and fixed along the way

1. **Probe size vectors summed to 0.9999** and were rejected by the kit's strict writer. Fixed by normalizing rows (reference model only).
2. **The CV grid stored about 90M floats.** Refactored to store aggregated losses only; same results.
3. **Cache-key collision in `research/cv_exp4.py`.** Folds of equal size could share a cached fit, which is potential leakage. Caught because the runtime was suspiciously short. Fixed with a full-mask hash and rerun, so all reported numbers come from the corrected version.
4. **`OrientedForecaster.fit` indexed a tuple by mistake.** Crashed immediately; fixed before any results.
5. **`research/cv_oriented.py` parsed command-line arguments at import time.** Moved into `main()`.
6. **An EXP-4 entry README briefly had an empty model description.** The entry was rebuilt and re-verified.
7. **Wrong timestamps in `EXPERIMENT_LOG.md`** for the post-EXP-4 research. Corrected from file modification times.

---

## 9. Where everything is

| Path | What |
|---|---|
| `DESIGN.md` | Original technical design and data analysis |
| `EXPERIMENT_LOG.md` | Every run: command, settings, CV, validation, runtime, decision |
| `house_edge/` | **Shipped model code** (stdlib only): `history.py`, `rules.py`, `pair_rule.py` (EXP-1), `change_point.py` (EXP-2), `calibrated.py` (EXP-3), `oriented.py` (EXP-4), `size_rule.py` (EXP-5), `registry.py` |
| `research/` | Research only (numpy/scipy): CV for each experiment (`cv_*.py`), `run_model.py`, `probe.py`, `final_cv_report.py`, `headroom.py`, size-rule research (`size_rules*.py`, `r1_*.py`, `r2_*.py`, `r3_*.py`), rejected-idea tests (`size_recency.py`, `third_lag.py`) |
| `tests/` | Leakage test, plus a numpy-vs-stdlib equivalence test per model |
| `tools/` | `make_entry.py` (assemble entry + ZIP), `verify_entry.py` (fresh-process reproduction check) |
| `configs/` | Final settings for each experiment (`exp1.json` … `exp5.json`) |
| `runs/<exp>/` | `model.json`, `validation_predictions.csv`, `submission.csv`, `run.json`, `validation_scores.json` (`*_cv` folders hold CV reports; `*_retrain_check` folders prove retraining is deterministic) |
| `entries/<exp>/` | Submission folders: `README.md`, `reproduce.json`, `predict.py`, `model.json`, code |
| `zips/<exp>.zip` | Upload-ready code ZIPs |
| `baseline-pooled-recent/`, `baseline-pooled-recent-entry.zip` | Organizer-baseline fallback |
| `outputs/`, `entry.zip` | Original kit `pooled` run, untouched |

---

## 10. Submission status and how to reproduce

**Final Main entry: EXP-5**

| File | Path | SHA-256 |
|---|---|---|
| CSV | `runs/exp5/submission.csv` | `286311e7c5aaa4947a886ae4ec13cdc4d7a7f8506daa9d427d0f792ea78ac699` |
| ZIP | `zips/exp5.zip` | `f94dd6980f47bfa31fd93ca3cf8a1962168b8c01274388bcf6fde38d5bc1dc4c` |

- **Already uploaded by the team:** EXP-4 (`zips/exp4.zip` + `runs/exp4/submission.csv`).
- **To do:** upload the EXP-5 pair before 21:00 CT and check the receipt shows the EXP-5 CSV hash above.
- **Verified fallbacks:** `zips/exp1.zip` to `zips/exp4.zip` (each with its `runs/<exp>/submission.csv`), and `baseline-pooled-recent-entry.zip`.

```sh
# retrain from scratch (about 30 s) and write validation/test predictions
python research/run_model.py --config configs/exp5.json --out runs/<new-folder>

# what the organizers run, from inside the extracted ZIP
python predict.py --data /data --out /output/submission.csv

# checks
python check_submission.py runs/exp5/submission.csv
python check_submission.py runs/exp5/validation_predictions.csv --split validation
python tests/test_leakage.py runs/exp5/model.json
python tests/test_equivalence_exp5.py runs/exp5/model.json
python tools/verify_entry.py zips/exp5.zip runs/exp5/submission.csv [--python <path to python3.13>]
```

---

## 11. Round two (opens 21:00 CT, due 11:00 Sunday)

The organizers say some patterns change. The model relearns its rule popularity, levels, size distributions and calibration from the new training data, but the rule *families* are fixed in code. Plan:

1. Extract the new kit into a separate folder and confirm `public/metadata.json` says `adaptation`.
2. Run the `pooled_recent` baseline, then refit EXP-5 on the new training data. Upload a first valid entry early.
3. Rerun the research scripts on the new data (`research/r1_size_structure.py`, `research/r3_rule_priors.py`, plus the pair scan in `DESIGN.md`) to see whether the lag pairs, the size rule or the change behaviour moved. Extend the candidate rules if needed.
4. Re-tune with the CV scripts (`research/cv_exp5.py` etc.), confirm once on the new validation set, then package and verify with `tools/verify_entry.py`.
