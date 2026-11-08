# House Edge — TAMU Datathon 2026 | Claude Code instructions

## Mission
Build the strongest **valid, reproducible probabilistic forecaster** for the House Edge slot-machine challenge. Maximize the organizer's official validation **technical points**, while avoiding leakage, overfitting, and submission failures. This is a time-limited competition: keep a working fallback at all times.

## Source of truth
- Read `README.md`, `RULES.md`, `DATA.md`, `EXPERIMENTS.md`, `public/metadata.json`, and relevant source before changing code.
- This supplied kit is **Main 60 points / Adaptation 40 points**; consult `public/metadata.json` for the active round. Some externally copied guides have different weights or submission instructions. Prefer the active organizer kit and latest verified organizer announcement.
- The organizer's scripts in `slot_forecasting/contest.py`, `scoring.py`, `submission_contract.py`, and `check_submission.py` define the local validation and output contract. **Do not modify these to improve reported scores.**
- The goal is estimating the true five-class **probability vector**, NOT choosing a bet, maximizing winnings, or predicting only the most likely class.

## Problem and inputs
- Payout order is exactly `[0, 1, 2, 5, 20]`; output probabilities must be finite, in `[0,1]`, and sum to 1.
- Each query contains `query_id`, `episode_id`, `machine_id`, `machine_type`, `signal`, `cutoff`.
- Historical rows contain `episode_id`, `machine_id`, `tick`, `payout`.
- Use only observations with `tick < cutoff` for **every machine**, including machines in the same episode.
- A missing tick is **unknown**, never a zero payout. For lag `k`, use the observation at exactly `cutoff-k`, not simply the kth most recent row.
- `signal` is an observed categorical feature; do not assume `+1` means higher payouts or invent a known signal value at earlier training-history ticks.
- `machine_id` is scoped to `episode_id`; cross-machine features may use only other machines **in the same episode** and only prior ticks.
- Test has no true probabilities, sampled next-outcome labels, or condition labels. Never use hidden files, infer private seed values, or search for organizer secrets.

## Evaluation: non-negotiable
- Use official **validation technical points** and four condition loss ratios, not the old generic score, simple accuracy, expected payout error, or realized-outcome Brier score.
- The validation target is the simulator's actual distribution (`public/validation/targets.csv`), not its one sampled outcome.
- Official objective: per-query sum of squared differences, averaged within episodes and conditions; each of four conditions has equal weight, normalized to its frozen reference; round points = `max_points/(1+mean_loss_ratio)`.
- Conditions: `stable`, `short_history`, `drift`, `shared_change`. These labels may be used for evaluation but **not** as inference inputs or a model gate unless the condition can genuinely be inferred from allowed observable features.
- Split training and validation by whole `episode_id`; don't mix related machines across folds.
- Keep a truly untouched audit set when selecting models. After tuning against it, stop claiming it is independent; create another group-wise evaluation or disclose the limitation.
- Always report validation technical points, per-condition ratios, runtime, and whether data leakage checks passed. A small numerical lift can be noise.

## Known baseline (from user's actual local run)
- `global` 26.730/60; `context` 29.685/60; `empirical` 29.918/60.
- `recent` 30.743/60; `pooled` 30.000/60; **`pooled_recent` 30.826/60**.
- `run_starter.py` saves `pooled` by default even if a different candidate scored higher.
- Preserve the baseline CSV/ZIP. Never overwrite the best reproducible entry while experimenting.

## Repository map
- `run_starter.py`, `run_experiments.py`: organizer baseline and experiment entrypoints.
- `slot_forecasting/baseline.py`: frequency forecasting baselines.
- `slot_forecasting/transition.py`: lag-conditioned transition model.
- `slot_forecasting/experiments.py`: 12-candidate experiment suite with tuning/audit episode split.
- `slot_forecasting/io.py`, `schema.py`: data loading and output validation.
- `slot_forecasting/contest.py`, `scoring.py`: official scoring logic; leave unchanged.
- `build_entry.py`, `check_submission.py`: packaging and submission validation.
- `public/{train,validation,test}/`: organizer data; **read-only**.

## First-run commands (from kit root)
```bash
python run_starter.py --method pooled_recent --out baseline-pooled-recent
python check_submission.py baseline-pooled-recent/validation_predictions.csv --split validation
python check_submission.py baseline-pooled-recent/submission.csv
python build_entry.py --starter baseline-pooled-recent --out baseline-pooled-recent-entry.zip
python run_experiments.py --out experiments-01
python check_submission.py experiments-01/validation_predictions.csv --split validation
```
Use **new output directories** for each run. Do not assume `run_experiments.py` creates a directly reproducible upload ZIP: custom entries need their own `predict.py`, `reproduce.json`, and saved artifacts.

## Modeling strategy (hypotheses, not established improvements)
1. Verify baselines and scoring; profile data and establish trustworthy episode-level folds.
2. Test leakage-safe features: exact lags 1–8; payouts and categories; recent/long-window frequencies; exponential weighting; missingness; type and signal; machine-specific shrinkage; regime-change indicators; neighboring machines' **past** results.
3. Try cheap interpretable enhancements first: tuned recency windows, hierarchical smoothing, transition models, calibrated mixtures of frequencies.
4. Then compare regularized multinomial models and CPU-feasible boosted trees **if** available packages can run offline in the organizer environment.
5. Consider compact sequence models only if simpler models plateau and enough time remains for verification.
6. Blend complementary, well-calibrated models using out-of-fold predictions; avoid tuning ensemble weights against the untouched audit set.
7. Test whether improvements hold in *all four conditions*, not just aggregate score.

### Expanded examples: handle with care
- Earlier history rows can yield extra supervised training examples, but use only features that were observable **before that earlier tick**.
- Do not attach the query-time `signal` or future machine/episode summaries to synthetic earlier-tick examples unless the feature was actually observable at that earlier time.
- Account for within-episode correlation, noisy single-spin labels, missing ticks, and weighting so long histories do not dominate.
- A model trained on realized payout labels is not automatically optimal for a score comparing true conditional probability distributions.

## Development and team workflow
- When asked for design, write a **short, actionable `DESIGN.md` before implementation**: hypothesis ranking, data-flow diagram or module interfaces, evaluation protocol, estimated runtime, expected failure modes, and packaging plan. Label measured facts vs speculative ideas.
- Work in small, independently testable steps; run the checker after every candidate and keep `EXPERIMENT_LOG.md` with commands, seed, settings, score, condition breakdown, runtime, and decision.
- For a four-person team, prefer distinct Git branches/workstreams: features; models; evaluation and blending; submission/testing. Agree on a stable `fit(...)` / `predict(...)` interface and file ownership before concurrent edits.
- Don't edit teammates' work or merge conflicting code without checking changes. Avoid giant rewrites and unrelated refactoring of the organizer kit.
- Use deterministic seeds and record dependency versions. Avoid downloading data or dependencies at inference time.
- Keep code simple and memory-efficient; target CPU reproducibility within the organizer's runtime and size constraints. If extra libraries are required, verify an approved environment or packaged compatible Linux wheels.
- Never include team tokens, credentials, .env files, unrelated datasets, or private information in ZIPs or Git.
- Don't claim to have run a command, improved a score, passed reproduction, or uploaded an entry unless it actually happened.

## Submission contract
- CSV columns **exactly**: `query_id,p_0,p_1,p_2,p_5,p_20` with every current-round test query exactly once.
- Package a custom code folder with `README.md`, `reproduce.json`, `predict.py`, its source modules, and all model artifacts required offline. `predict.py` must accept `--data` and `--out`.
- Typical `reproduce.json`: `{"image":"python","command":["python","predict.py","--data","/data","--out","/output/submission.csv"]}`. Follow the current kit's precise requirements.
- Build custom ZIP using `python build_entry.py --source my-entry --out entry-custom.zip`; keep the forecast CSV separate from the ZIP.
- Verify a clean, fresh-process reproduction of the exact uploaded CSV, with per-probability tolerance <= `1e-8` after normalization where required. Verify no absolute paths, external network access, or unbundled dependencies.
- Main and Adaptation have different query IDs. On adaptation, read the fresh kit and rerun baseline/evaluation before changing methods. Don't reuse Main `submission.csv` unchanged.
- Organizer deadline in this kit: Main **Saturday Nov 7, 2026 9 PM CT**, Adaptation **Sunday Nov 8, 2026 11 AM CT**. Upload early using the organizer's announced forms; a local `VALID FORMAT` message is **not** an upload receipt or code-reproduction pass.

## Required final response after a coding task
State: what changed; commands actually executed; validation technical points and per-condition ratios; audit methodology/results; runtime; reproduction check; file paths; remaining risks; and the next single highest-value experiment. If something was not tested, say so plainly.
