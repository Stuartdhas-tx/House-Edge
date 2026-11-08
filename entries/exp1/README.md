# House Edge entry: exp1

- Round: main · Season ID: 5fffdc4d0c700bbf61c3f911
- Forecast file: submitted separately as `submission.csv` (SHA-256 `198a8d662d7d006f6fcdc1e8fae3581db25748766516b41486ef3e3064443464`)
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

EXP-1 Bayesian pair-rule forecaster: P(win) is a posterior average over 37 lag rules (null, 8 single lags, 28 XOR pairs of exact lags 1-8) with Beta-Bernoulli cells shrunk to type base rates, recency-weighted (half-life 12) Beta-binomial evidence tempered by beta=2, and an empirical-Bayes rule prior learned from training histories. Output = [1-w, w*size], size = type training size distribution shrunk (strength 50) toward the machine's own winning sizes.

Settings: `model.json` (hyperparameters, type base rates, size priors, learned rule prior). Hyperparameters were
chosen by episode-grouped 5-fold cross-validation on TRAIN pseudo-queries; all parameters were fit on the main-round
training split only. Validation targets were used for scoring, never for fitting.

Training command (from the team repository): `python research/run_model.py --config configs/exp1.json --out runs/exp1`

Seeds: none needed (deterministic). `slot_forecasting/{__init__,io,schema}.py` are unmodified copies of the organizer kit.
