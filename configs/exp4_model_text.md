EXP-4: change-aware Bayesian pair-rule forecaster with oriented hierarchical priors, payout-size regimes and calibration. All parameters were fit on the main-round TRAIN split only.

- Rules: P(win) is a posterior average over 37 lag rules (null, 8 single lags, 28 XOR pairs of exact lags 1-8) and over change offsets d in {8,12,...,200} ticks plus 'no change' (prior 0.5). Missing ticks are unknown, never zero; only ticks before the cutoff are read.
- Oriented hierarchical priors: every non-null rule has an orientation (prior 1/2). Its high cell has prior Beta(40*hi, 40*(1-hi)), its low cell Beta(40*lo, 40*(1-lo)), with per-type levels hi/lo learned from training machines with a clear full-history rule: balanced .842/.060, jackpot .669/.021, steady .934/.159. The null rule uses Beta(0.5*mu, 0.5*(1-mu)) with mu = type base win rate.
- Changes: separate cells before/after a change, pre-change orientation marginalised (a change may flip the rule), post-change cells predict. Evidence = Beta-binomial marginal likelihood (beta=1); rule prior by empirical Bayes on training histories.
- Episode pooling: a shared change offset for all machines in an episode vs independent offsets (prior 0.5), inferred from observed histories only.
- Calibration: logit(w') = 1.0970*logit(w) + intercept[type] (balanced +0.022, jackpot -0.210, steady +0.045), fit on nested out-of-fold TRAIN pseudo-query predictions.
- Size: per type, two winning-size distributions learned by EM on TRAIN histories; machine regime posterior from its own winning sizes (tempered LLR 0.75).
- Output = [1-w', w' * size].
