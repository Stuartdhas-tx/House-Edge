**Joint win-and-size inference: implementation and results**

2026-11-07. Decision: **keep EXP-5; the tested joint-inference variants did not show a reliable improvement.** These are training cross-validation results, not new official validation or hidden-test scores.

The implementation lets payout-size likelihood update the posterior over win rules, and averages complete hypothesis-specific payout probabilities. It preserves EXP-5's fitted global parameters. The new model is registered as `joint_size` and uses Python's standard library at inference.

**Comparison**

Five episode-grouped outer folds cover 96 training episodes and 768 real query cutoffs. An additional 15 pseudo-cutoffs per machine give 12,288 total rows. Calibration is fitted inside each outer training split from inner out-of-fold predictions of BOTH win probabilities and conditional sizes. All upstream fits exclude both the outer and inner episodes. Actual-cutoff five-class Brier was the primary endpoint; lower is better.

| Variant | Uncalibrated real Brier | Nested calibrated real Brier | Calibrated change vs EXP-5 |
|---|---:|---:|---:|
| EXP-5, corrected nested calibration | 0.34922522 | 0.34854627 | +0.00000000 |
| Joint averaging only | 0.34923389 | 0.34855986 | +0.00001359 |
| Joint averaging, common events | 0.34994108 | 0.34932137 | +0.00077510 |
| Common-event joint likelihood | 0.35017860 | 0.34951704 | +0.00097076 |
| All-win joint likelihood, missing-lag mixture | 0.34960245 | 0.34895715 | +0.00041088 |

The all-win follow-up's calibrated difference is +0.00041088, with a paired within-condition episode-bootstrap 95% interval of [-0.00012271, +0.00130084]. It improved three folds and worsened two; the point estimate is worse overall and the interval includes zero. Its short-history difference is +0.00164895. Its all-row difference is +0.00002786. There is no evidence supporting a replacement submission.

The corrected nested baseline differs from the earlier published EXP-5 CV number because the previous calibration procedure reused an OOF table indirectly influenced by the outer held-out fold. This new comparison evaluates every candidate consistently. It does not change the existing EXP-5 artifact or its previously measured official validation score.

**What the controls reveal**

- Joint averaging alone changes very little. This corrects a mathematical approximation but did not produce a useful measured gain.
- Using only common complete windows loses useful size observations, especially in short histories.
- The initial common-event joint likelihood remained worse after calibration.
- A single follow-up kept every observed winning payout. Missing candidate lags use `(A+B)/2`, contributing to both orientation likelihoods. This preserves the old orientation estimates while comparing likelihoods on the same response set. It recovered part of the loss but still did not outperform EXP-5.

The follow-up assumes a neutral 50/50 latent size cell when a lag is unavailable. It is a documented approximation, not full causal missing-history filtering. The follow-up was specified after the first training results; the original four fixed variants and their negative result are retained. No missing-cell prior or other hyperparameter was tuned to validation.

**Verification and runtime**

- `tests/test_joint_size.py`: all 15 tests pass on Python 3.12.12 and 3.13.9. Tests include independent enumeration of joint/shared/independent posteriors and missing cells; joint-path future invariance on every machine; missingness; identical size-component neutrality; persistence; and invalid configuration rejection.
- `tests/test_equivalence_joint.py`: all five variants match the independent NumPy reference over 768 training queries; maximum probability difference is 1.074e-13.
- `tests/test_leakage.py`: all 21 existing truncation/shuffle checks, missingness, determinism and future guards pass for the saved common-window model. The new unit tests additionally cover the actual joint pooling path.
- Fresh full training gives exactly the same global win rates, cell levels, rule prior, and size components as EXP-5. This isolates the inference change.
- The original EXP-5 uncalibrated CV summaries reproduce within 5.55e-17. Adding the optional fifth variant leaves all four existing OOF arrays, summaries and calibration parameters exactly unchanged. See `runs/joint_all_cv/reproduction_checks.json`.
- The saved uncalibrated common-window model fits in 8.0 seconds and predicts the full test set in 12.4 seconds. Its experimental test CSV passes the organizer format checker for 1,536 queries. This format check is not an endorsement of its predictive quality.
- The final five-variant nested CV took 106.8 seconds.
- All 37 organizer manifest files remain unchanged, and the adopted EXP-5 CSV retains SHA-256 `286311e7c5aaa4947a886ae4ec13cdc4d7a7f8506daa9d427d0f792ea78ac699`.

No new validation-target score was computed, and no replacement entry was packaged or uploaded. The submitted EXP-5 remains the recommendation. Serialization is checked, but a new joint candidate ZIP was not subjected to fresh-process reproduction because the candidate failed the training comparison.

**Files and reproduction**

- Model: `house_edge/joint_size.py`; registry addition in `house_edge/registry.py`.
- Configurations: `configs/joint_size_uncal.json`, `configs/joint_size.json`, `configs/joint_all.json`. All are experimental; the latter two contain global OOF calibration from the corresponding new CV report.
- Evaluation: `research/joint_numpy.py`, `research/cv_joint_size.py`.
- Tests: `tests/test_joint_size.py`, `tests/test_equivalence_joint.py`.
- Plan and follow-up disclosure: `JOINT_INFERENCE_PLAN.md`.
- Initial results: `runs/joint_size_pilot/`, `runs/joint_size_cv/`.
- Follow-up results: `runs/joint_all_pilot/`, `runs/joint_all_cv/`.
- Saved experimental model and predictions: `runs/joint_size_uncal/`.

Commands actually executed, from the kit root:

```sh
python research/cv_joint_size.py --out runs/joint_size_pilot --uncalibrated-only
python research/cv_joint_size.py --out runs/joint_size_cv
python research/cv_joint_size.py --out runs/joint_all_pilot --uncalibrated-only --include-missing-mixture
python research/cv_joint_size.py --out runs/joint_all_cv --include-missing-mixture
python research/run_model.py --config configs/joint_size_uncal.json --out runs/joint_size_uncal --no-score
python tests/test_joint_size.py
/Users/stuarts/.local/share/uv/python/cpython-3.13.9-macos-aarch64-none/bin/python3.13 tests/test_joint_size.py
python tests/test_equivalence_joint.py runs/joint_size_uncal/model.json --include-missing-mixture
python tests/test_leakage.py runs/joint_size_uncal/model.json
python check_submission.py runs/joint_size_uncal/submission.csv
```

Use fresh `--out` names when repeating. The corrected CV runner is reusable for these fixed hypotheses, not a general replacement for every earlier experiment; its fitting path deliberately uses the original contrast threshold of 0.5.

**Limits and next experiment**

Raw Brier differences estimate probability-error differences in expectation, but omit the official inverse-reference-error weights. Episode-bootstrap intervals condition on fitted OOF predictions and do not cover all earlier model selection. Training outcomes are noisy, so this is failure to establish an improvement, not proof that all possible joint models are inferior.

Existing official EXP-5 validation: **57.475361/60**; stable/short/drift/shared ratios **0.030193 / 0.089073 / 0.030315 / 0.026122**. The historical tuning/audit scores are **57.4242 / 57.5412**; those halves have been examined repeatedly and are not pristine.

The next highest-value experiment is partial pooling of signal-specific high/low cell estimates toward type-level priors, using the corrected nested evaluation. That targets level estimation without discarding observations or expanding the rule family.
