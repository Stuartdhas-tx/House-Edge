# House Edge entry: exp2

- Round: main · Season ID: 5fffdc4d0c700bbf61c3f911
- Forecast file: submitted separately as `submission.csv` (SHA-256 `d698a96f84af0d7d020a79446ab69064827ddeb401347674f15f46b370b6ed1d`)
- Environment: `python` image, **Python standard library only** (no requirements.txt). Tested locally on Python 3.12.12.
- Hardware: any CPU; prediction takes a few seconds and well under 1 GiB RAM.

## Recreate the forecast

Run from this folder:

```sh
python predict.py --data /data --out /output/submission.csv
```

`predict.py` loads the saved parameters in `model.json` and predicts `/data/test` deterministically (no randomness,
no network, no training at prediction time). It writes only to `--out`.

## Model

EXP-2 change-aware Bayesian pair-rule forecaster. P(win) averages over 37 lag rules (null, 8 single lags, 28 XOR pairs of exact lags 1-8) and over change offsets d in {8,12,...,200} ticks plus 'no change': each (rule, offset) has separate Beta-Bernoulli cells before/after the change (shrunk to type base rates, alpha=0.5), evidence = Beta-binomial marginal likelihood (beta=1), rule prior learned by empirical Bayes on training histories, P(no change)=0.5. Within each episode a shared-change hypothesis (one offset for all 8 machines) is compared with independent offsets (prior 0.5) using only observed histories. Output = [1-w, w*size], size = type training size distribution shrunk (strength 50) toward the machine's own winning sizes.

Settings: `model.json` (hyperparameters, type base rates, size priors, learned rule prior). Hyperparameters were
chosen by episode-grouped 5-fold cross-validation on TRAIN pseudo-queries; all parameters were fit on the main-round
training split only. Validation targets were used for scoring, never for fitting.

Training command (from the team repository): `python research/run_model.py --config configs/exp2.json --out runs/exp2`

Seeds: none needed (deterministic). `slot_forecasting/{__init__,io,schema}.py` are unmodified copies of the organizer kit.
