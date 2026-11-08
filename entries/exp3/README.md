# House Edge entry: exp3

- Round: main · Season ID: 5fffdc4d0c700bbf61c3f911
- Forecast file: submitted separately as `submission.csv` (SHA-256 `2c68d625380dcfc109a090c264daf5c0c078b8db95bf8addeefe1e3f9fc8b74c`)
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

EXP-3 = EXP-2 change-aware Bayesian pair-rule forecaster + payout-size regime model + win-probability calibration.

- P(win): posterior average over 37 lag rules (null rule, 8 single lags, 28 XOR pairs of exact lags 1-8) and over change offsets d in {8,12,...,200} ticks plus 'no change'. Each (rule, offset) has separate Beta-Bernoulli cells before/after the change, shrunk to the machine type's base win rate (alpha=0.5); evidence = Beta-binomial marginal likelihood (beta=1); rule prior learned by empirical Bayes on training histories; P(no change)=0.5.
- Episode pooling: a shared-change hypothesis (one offset for all machines in the episode) is compared with independent offsets (prior 0.5), using only observed histories before the cutoff.
- Calibration: logit(w') = 1.1274*logit(w) + intercept[type] (balanced +0.164, jackpot +0.053, steady -0.154), fit on out-of-fold TRAIN pseudo-query predictions.
- Size: per type, two winning-size distributions learned by EM on TRAIN histories (one latent regime per machine); a machine's regime posterior uses its own winning sizes with tempered log-likelihood ratio (0.75).
- Output = [1-w', w' * size]. Missing ticks are unknown (never zero); only ticks before the cutoff are read.

Settings: `model.json` (hyperparameters, type base rates, size priors, learned rule prior). Hyperparameters were
chosen by episode-grouped 5-fold cross-validation on TRAIN pseudo-queries; all parameters were fit on the main-round
training split only. Validation targets were used for scoring, never for fitting.

Training command (from the team repository): `python research/run_model.py --config configs/exp3.json --out runs/exp3`

Seeds: none needed (deterministic). `slot_forecasting/{__init__,io,schema}.py` are unmodified copies of the organizer kit.
