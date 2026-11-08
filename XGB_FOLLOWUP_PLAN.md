# Focused correction follow-up — fixed before new results

2026-11-07. The user explicitly asked to continue improving the model while controlling overfitting. EXP-5 remains the working submission. The earlier XGB correction reduced mean real-cutoff TRAIN CV Brier from 0.348546 to 0.347388, but its interval crossed zero and it failed that experiment's predeclared gate. That result and decision remain unchanged.

## Question

Does the correction improve on simple recalibration, and does its average gain persist under predefined changes to the episode partition? This is a finite follow-up motivated by already-seen TRAIN results, not a new independent data set.

The saved trees use many rule, size and peer-history features, but training split importance does not demonstrate that these features cause held-out gains. A direct ablation is needed.

## Fixed models

Use the same EXP-5 upstream fitting, history features, training rows, balanced weights, XGB correction parameters and seed as the first benchmark. No depth, regularization or feature search.

1. **Type/size calibration:** extend the existing four win-calibration parameters with three conditional-size logit offsets per machine type (nine extra parameters, 13 total). The fourth size offset is the reference zero. Penalize the centered four-offset vector per type by `0.001 * sum(centered_offset**2)`, added to the weighted mean five-class Brier loss. This avoids favoring a particular reference class. Fit by L-BFGS-B from identity, with win slope in [0.25,4] and remaining coefficients in [-4,4]. Record convergence and any active bounds.
2. **Compact XGB calibration control:** the same depth-2, 100-round correction, using only the three type indicators and five raw EXP-5 probabilities. No historical summary or signal inputs.
3. **Full XGB correction:** exactly the previous correction, with all 1,259 history features plus five raw probabilities. This is a replication/control on the original partition, not a new hyperparameter trial.
4. **Half correction:** fixed 50% calibrated EXP-5 + 50% full correction probabilities. No fitted blending weight.

All fitted post-processors use raw EXP-5 predictions cross-fitted within the outer training split. Both win and size inputs exclude outer and inner episodes. The original four-parameter calibration is fitted from those same inner predictions and serves as the comparison baseline. No extra layer is fitted on outer-held-out predictions.

## Three fixed episode partitions

- Original: `common.episode_cv_folds`, sorting the SHA-256 of the episode ID within each condition, then assigning round-robin to five folds.
- Follow-up A: the same process using SHA-256 of `followup-a|` plus the episode ID.
- Follow-up B: the same process using SHA-256 of `followup-b|` plus the episode ID.

Report all three. These are sensitivity checks on the same 96 episodes, not three independent replications. Primary endpoint is actual-cutoff Brier; pseudo-query results are secondary. The official reference-error normalization is absent.

For uncertainty, average each episode's paired **loss differences** across the three repetitions, then bootstrap the 96 episode clusters within condition. Do not treat 288 episode-repetitions as independent, and do not average probabilities into an unplanned ensemble. Report full-correction minus compact-control as the primary mechanism diagnostic, alongside each candidate versus calibrated EXP-5.

## Decision fixed before the follow-up

This follow-up uses an explicitly exploratory screening rule, rather than treating the earlier 95% rule as a prohibition on research. It does not retroactively make the earlier experiment pass.

Allow at most one frozen candidate into one further official validation comparison if:

- its actual-cutoff mean Brier is below calibrated EXP-5 in all three partitions;
- its repeat-averaged mean improves in at least three of four conditions;
- no repeat-averaged condition worsens by more than 0.002 Brier;
- fitting, feature leakage, prediction validity and saved-model checks pass.

Among eligible candidates choose the lowest repeat-averaged primary loss; ties within 1e-8 favor, in order, type/size calibration, compact correction, half correction, full correction. Report the uncertainty interval even if it crosses zero. This is evidence to investigate, not a declaration of statistical certainty. If none qualifies, stop this finite batch and retain EXP-5; do not add seeds or tune until a result passes.

The selected candidate, if any, is trained once on all TRAIN data, with its post-processor fitted on global OOF base predictions from the **original partition only**. This keeps deployment fitting consistent with the pipeline evaluated per partition; the other two partitions are sensitivity checks, not an extra training ensemble.

Freeze that candidate and evaluate its validation predictions once against the existing EXP-5 predictions, using official technical points, per-condition ratios, and paired episode differences. The public validation set is already development-influenced; do not claim it is independent. Do not tune from that comparison. Recommend a replacement only if the validation point gain is at least 0.10 on this kit's 60-point scale, no condition ratio worsens by more than 0.005, and matching test CSV/code ZIP pass fresh-process reproduction. An interval crossing zero still means the gain remains uncertain. No upload is automatic.

## Implementation, checks and budget

New research files only: `research/cv_xgb_followup.py`, `research/followup_calibration.py`; new focused tests. Keep original benchmark code and outputs immutable. Source/input hashes, versions, parameters, calibration coefficients and fold models must be saved.

Validate the calibration analytic gradient against finite differences, identity behavior, normalized probabilities, type indexing, and nested exclusion masks. The original partition's baseline and full correction must reproduce the previous benchmark. Repeat existing feature tests. Review saved artifacts and verify any proposed deployment with the actual inference implementation before scoring validation.

Expected CPU runtime: roughly 6–10 minutes; measure it. If native XGBoost deployment cannot be made reproducible within the organizer contract, keep the standard-library EXP-5 submission. Do not confuse a research model with a packaged entry.

Calibrating multiclass outputs is supported by [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html) and [Kull et al.](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8ca01ea920679a0fe3728441494041b9-Abstract.html). The specific win/conditional-size map and fixed regularization above are our proposed experiment, not a claimed result from those papers.
