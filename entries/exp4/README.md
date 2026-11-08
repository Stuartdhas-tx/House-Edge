# House Edge entry: exp4

- Round: main · Season ID: 5fffdc4d0c700bbf61c3f911
- Forecast file: submitted separately as `submission.csv` (SHA-256 `f07a7bd179ff30fae7676da3fb90e729e75285d77477db267271ae2818c6b1fb`)
- Environment: `python` image, **Python standard library only** (no requirements.txt). Fresh-process reproduction verified byte-identical on Python 3.12.12 and 3.13.9 (macOS arm64).
- Hardware: any CPU, single core; measured test prediction time 11 s, well under 1 GiB RAM.

## Recreate the forecast

Run from this folder:

```sh
python predict.py --data /data --out /output/submission.csv
```

`predict.py` loads the saved parameters in `model.json` and predicts `/data/test` deterministically (no randomness,
no network, no training at prediction time). It writes only to `--out`.

## Model

EXP-4: change-aware Bayesian pair-rule forecaster with oriented hierarchical priors, payout-size regimes and calibration. All parameters were fit on the main-round TRAIN split only.

- Rules: P(win) is a posterior average over 37 lag rules (null, 8 single lags, 28 XOR pairs of exact lags 1-8) and over change offsets d in {8,12,...,200} ticks plus 'no change' (prior 0.5). Missing ticks are unknown, never zero; only ticks before the cutoff are read.
- Oriented hierarchical priors: every non-null rule has an orientation (prior 1/2). Its high cell has prior Beta(40*hi, 40*(1-hi)), its low cell Beta(40*lo, 40*(1-lo)), with per-type levels hi/lo learned from training machines with a clear full-history rule: balanced .842/.060, jackpot .669/.021, steady .934/.159. The null rule uses Beta(0.5*mu, 0.5*(1-mu)) with mu = type base win rate.
- Changes: separate cells before/after a change, pre-change orientation marginalised (a change may flip the rule), post-change cells predict. Evidence = Beta-binomial marginal likelihood (beta=1); rule prior by empirical Bayes on training histories.
- Episode pooling: a shared change offset for all machines in an episode vs independent offsets (prior 0.5), inferred from observed histories only.
- Calibration: logit(w') = 1.0970*logit(w) + intercept[type] (balanced +0.022, jackpot -0.210, steady +0.045), fit on nested out-of-fold TRAIN pseudo-query predictions.
- Size: per type, two winning-size distributions learned by EM on TRAIN histories; machine regime posterior from its own winning sizes (tempered LLR 0.75).
- Output = [1-w', w' * size].

Settings: `model.json` (hyperparameters, type base rates, size priors, learned rule prior). Hyperparameters were
chosen by episode-grouped 5-fold cross-validation on TRAIN pseudo-queries; all parameters were fit on the main-round
training split only. Validation targets were used for scoring, never for fitting.

Training command (from the team repository): `python research/run_model.py --config configs/exp4.json --out runs/exp4`

Seeds: none needed (deterministic). `slot_forecasting/{__init__,io,schema}.py` are unmodified copies of the organizer kit.
