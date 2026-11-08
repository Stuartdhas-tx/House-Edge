# House Edge entry: exp6

- Round: main · Season ID: 5fffdc4d0c700bbf61c3f911
- Forecast file: submitted separately as `submission.csv` (SHA-256 `50a3ef479cf4d19a3c699e6b4b632c630de1174df49007284de0081e6789315a`)
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

EXP-6: EXP-5 (change-aware Bayesian pair-rule forecaster plus a payout-size regime tied to the win rule) with high/low cell levels per machine type and query signal. All parameters were fit on the main-round TRAIN split only.

- Win probability w: posterior average over 37 lag rules (null, 8 single lags, 28 XOR pairs of exact lags 1-8) and over change offsets d in {8,12,...,200} ticks plus 'no change' (prior 0.5). Oriented hierarchical cell priors: high cell Beta(40*hi, 40*(1-hi)), low cell Beta(40*lo, 40*(1-lo)), with levels per (type, query signal) learned from training machines with a clear rule and that query signal. Pre-change orientation is marginalised. Evidence is the Beta-binomial marginal likelihood (beta=1); the rule prior is learned by empirical Bayes on training histories. Episode pooling compares a shared change offset with independent offsets (prior 0.5), using only observed histories.
- Calibration: logit(w') = 1.0938*logit(w) + intercept[type] (balanced +0.034, jackpot -0.205, steady +0.037), fit on out-of-fold TRAIN pseudo-query predictions.
- Payout size: a machine's winning-size regime at tick t is set by whether the payout at the FIRST lag a of its win pair (a, b) was >= 2, with a machine-specific orientation. Orientation posterior comes from the machine's own past wins split by that cell. The win model's posterior mass is grouped by first lag; null and single-lag mass uses an even regime mix. The two size vectors per type are learned by EM on TRAIN histories with this cell structure.
- Output = [1-w', w' * size]. Missing ticks are unknown (never zero); only ticks before the cutoff are read on every machine.

Settings: `model.json` (hyperparameters, type base rates, size priors, learned rule prior). Hyperparameters were
chosen by episode-grouped 5-fold cross-validation on TRAIN pseudo-queries; all parameters were fit on the main-round
training split only. Validation targets were used for scoring, never for fitting.

Training command (from the team repository): `python research/run_model.py --config configs/exp6.json --out runs/exp6`

Seeds: none needed (deterministic). `slot_forecasting/{__init__,io,schema}.py` are unmodified copies of the organizer kit.
