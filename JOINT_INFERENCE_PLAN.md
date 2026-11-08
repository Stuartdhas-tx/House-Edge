**Joint win-and-size inference experiment — 2026-11-07**

The user requested implementation of the first recommendation in `TECHNIQUE_RESEARCH.md`. This experiment uses new `joint_size` names because `exp7` is already used by a separate grid experiment.

The hypothesis is that conditional payout sizes can identify an uncertain win-rule pair, particularly in short histories. The new predictor also averages complete hypothesis-specific probabilities, preserving the dependence between win probability and payout size.

**Fixed comparisons, chosen before results**

| Name | Size observations | Size evidence updates rule posterior | Output mixture |
|---|---|---|---|
| exp5 | Existing per-lag eligible wins | No | Product of means |
| joint_mix | Existing per-lag eligible wins | No | Mean of products |
| common_mix | Same complete-window wins for every lag | No | Mean of products |
| joint_size | Same complete-window wins for every lag | Yes | Mean of products |

All variants keep EXP-5's global fitting procedure and hyperparameters. The joint model adds each lag's orientation-marginalized size log-likelihood to the win evidence. Null/single rules use the equal size-component mixture on the same outcomes. The size term is included once, outside the win-likelihood temperature. Comparisons with differently sized likelihood event sets are forbidden.

The posterior over shared versus independent change times is recomputed after including size evidence. The positive-payout vector averages `posterior * win_probability * conditional_size`; any final win calibration rescales its total mass while retaining its conditional size distribution.

**Evaluation**

1. Run an uncalibrated five-fold episode-held-out comparison on training data, with actual query cutoffs primary and 16-cutoff snapshots secondary. Report every variant, per-condition deltas and paired episode-bootstrap intervals.
2. Evaluate low-dimensional calibration with nested episode splits, cross-fitting BOTH win and size predictions. No outer episode may influence the calibration-training predictions. Use the existing all-snapshot calibration objective to isolate the inference change rather than add another search.
3. Use the training evidence to decide whether `joint_size` merits a single official validation confirmation. Require an actual-cutoff improvement with reasonably consistent fold/condition evidence; an inconclusive result does not justify a stronger claim or a replacement upload.
4. If confirmed, generate a separate saved model, CSV and ZIP, then run numerical equivalence, joint-path leakage, probability validity and fresh-process reproduction checks. Report full/tuning/audit validation scores and condition ratios. Existing validation halves are historical splits, not pristine holdouts.

Raw training Brier lacks the official inverse-reference-error normalization; report it as Brier, never as technical points. Bootstrap intervals condition on fitted cross-validation predictions and do not cover all prior model selection.

**Implementation and ownership**

- `house_edge/joint_size.py`: standard-library predictor and persistence; existing models remain available.
- `research/cv_joint_size.py`: train-only vectorized comparison and proper nested calibration.
- `tests/test_joint_size.py`: independent enumeration, missing/future history, joint episode pooling and serialization checks.
- Root integration: model registry, a new config, research/validation log, and packaging if supported.

The inference addition is small compared with the current win-rule table construction: seven lag-specific two-orientation likelihoods per machine. Full CV is more expensive because it now nests calibration correctly. Runtime is to be measured.

Potential failures are weak size information, discarded observations under the common-window control, overconfident posteriors under imperfect size components, and existing calibration compensating for the older approximation. The controls distinguish these effects. No hyperparameter sweep or validation-target-driven repair is planned for this experiment.

**One follow-up specified after the training pilot**

The first uncalibrated comparison found actual-cutoff losses 0.34922522 (EXP-5), 0.34923389 (joint averaging only), 0.34994108 (common-window averaging), and 0.35017860 (common-window joint evidence). The common-window restriction was costly, especially in short histories. These results motivated one additional fixed candidate, before any new validation scoring:

`joint_all` uses every observed winning payout for every candidate rule. When a candidate's lag is observed, use its usual orientation-conditioned component. When the lag is missing, use the neutral mixture `(A+B)/2` for that outcome under either orientation. Add the missing-outcome log-likelihood to both orientation likelihoods. Consequently, orientation inference retains all of EXP-5's eligible size observations while rule comparisons explain the same full response set.

This is equivalent to marginalizing the unavailable binary size cell with a 50/50 prior, conditional on the observed missingness pattern. It is an explicit approximation; the true chance of a big historical payout need not be 50%. Missing ticks are never assigned zero. It is not the full missing-history filter described in the research report.

Run this candidate with optional `--include-missing-mixture` in fresh output folders. Keep the original four-variant run and report both results, including the negative first result. Compare uncalibrated performance first, and only then run nested calibration if the follow-up merits it. Do not tune a missing-cell probability against validation.
