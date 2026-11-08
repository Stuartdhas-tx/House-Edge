"""EXP-1: Bayesian pair-rule forecaster.

P(win) is a posterior average over 37 candidate rules (null, 8 single lags, 28 XOR pairs). Each rule
has two Beta-Bernoulli cells shrunk to the machine type's base win rate; rule evidence is the
recency-weighted Beta-binomial marginal likelihood, tempered by `beta`, against an empirical-Bayes
rule prior learned from training histories. Output = [1-w, w * size], where size is the type's
training size distribution shrunk toward the machine's own winning sizes.
"""

import json
import math
from pathlib import Path

from .history import SIZES, machine_payouts, size_counts, win_map
from .rules import N_RULES, rule_cells, training_events, weighted_counts

TYPES = ("balanced", "jackpot", "steady")
EPSILON = 1e-6


def log_marginals(n, k, a0, b0):
    """Beta-binomial log marginal likelihood of each rule's cell counts (empty cells contribute 0)."""
    const = math.lgamma(a0 + b0) - math.lgamma(a0) - math.lgamma(b0)
    out = []
    for r in range(N_RULES):
        total = 0.0
        for i in (2 * r, 2 * r + 1):
            if n[i] > 0:
                total += const + math.lgamma(a0 + k[i]) + math.lgamma(b0 + n[i] - k[i]) - math.lgamma(a0 + b0 + n[i])
        out.append(total)
    return out


def predictives(n, k, cells, a0, b0):
    """Posterior-mean win probability of each rule's current cell (pooled cells if the cell is unknown)."""
    out = []
    for r, c in enumerate(cells):
        if c is None:
            nn, kk = n[2 * r] + n[2 * r + 1], k[2 * r] + k[2 * r + 1]
        else:
            nn, kk = n[2 * r + c], k[2 * r + c]
        out.append((a0 + kk) / (a0 + b0 + nn))
    return out


def posterior(log_prior, log_ml, beta):
    scores = [p + beta * m for p, m in zip(log_prior, log_ml)]
    top = max(scores)
    weights = [math.exp(s - top) for s in scores]
    total = math.fsum(weights)
    return [w / total for w in weights]


def five_class(w, size):
    w = min(1.0 - EPSILON, max(EPSILON, w))
    probs = [1.0 - w] + [w * s for s in size]
    total = math.fsum(probs)
    return tuple(p / total for p in probs)


class PairRuleForecaster:
    def __init__(self, half_life=32.0, alpha=4.0, beta=1.0, size_strength=50.0,
                 prior_smoothing=1.0, eb_iterations=20):
        self.half_life = None if half_life is None else float(half_life)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.size_strength = float(size_strength)
        self.prior_smoothing = float(prior_smoothing)
        self.eb_iterations = int(eb_iterations)
        self.win_rate = None
        self.size_prior = None
        self.log_prior = None

    # ---- training -------------------------------------------------------------------------
    def fit(self, queries, history, labels=None):
        """Type base rates and size priors from observed history; rule prior by empirical Bayes."""
        machines = machine_payouts(queries, history)
        wins = {t: [0, 0] for t in TYPES}
        sizes = {t: [0.0] * len(SIZES) for t in TYPES}
        for query, payouts in machines.values():
            for value in payouts.values():
                wins[query.machine_type][0] += value > 0
                wins[query.machine_type][1] += 1
            for i, c in enumerate(size_counts(payouts, query.cutoff)):
                sizes[query.machine_type][i] += c
        self.win_rate = {t: (wins[t][0] + 1.0) / (wins[t][1] + 2.0) for t in TYPES}
        self.size_prior = {t: [(c + 0.5) / (math.fsum(sizes[t]) + 2.0) for c in sizes[t]] for t in TYPES}
        evidence = []
        for query, payouts in machines.values():
            n, k = weighted_counts(training_events(payouts), query.cutoff, self.half_life)
            a0, b0 = self._cell_prior(query.machine_type)
            evidence.append(log_marginals(n, k, a0, b0))
        self.log_prior = self.empirical_bayes_prior(evidence)
        return self

    def empirical_bayes_prior(self, evidence):
        prior = [1.0 / N_RULES] * N_RULES
        for _ in range(self.eb_iterations):
            log_prior = [math.log(p) for p in prior]
            mass = [self.prior_smoothing] * N_RULES
            for log_ml in evidence:
                for r, p in enumerate(posterior(log_prior, log_ml, self.beta)):
                    mass[r] += p
            total = math.fsum(mass)
            prior = [m / total for m in mass]
        return [math.log(p) for p in prior]

    # ---- inference ------------------------------------------------------------------------
    def _cell_prior(self, machine_type):
        mu = self.win_rate[machine_type]
        return self.alpha * mu, self.alpha * (1.0 - mu)

    def win_probability(self, payouts, tick, machine_type):
        n, k = weighted_counts(training_events(payouts), tick, self.half_life)
        a0, b0 = self._cell_prior(machine_type)
        post = posterior(self.log_prior, log_marginals(n, k, a0, b0), self.beta)
        preds = predictives(n, k, rule_cells(win_map(payouts), tick), a0, b0)
        return math.fsum(p * q for p, q in zip(post, preds)), post

    def size_distribution(self, payouts, tick, machine_type):
        counts = size_counts(payouts, tick)
        prior = self.size_prior[machine_type]
        total = self.size_strength + sum(counts)
        return [(self.size_strength * p + c) / total for p, c in zip(prior, counts)]

    def predict(self, queries, history):
        if self.log_prior is None:
            raise ValueError("Fit or load the model before predicting.")
        result = {}
        for query_id, (query, payouts) in machine_payouts(queries, history).items():
            w, _ = self.win_probability(payouts, query.cutoff, query.machine_type)
            result[query_id] = five_class(w, self.size_distribution(payouts, query.cutoff, query.machine_type))
        return result

    # ---- persistence ----------------------------------------------------------------------
    def to_dict(self):
        return {"model": "pair_rule", "version": 1, "half_life": self.half_life, "alpha": self.alpha,
                "beta": self.beta, "size_strength": self.size_strength,
                "prior_smoothing": self.prior_smoothing, "eb_iterations": self.eb_iterations,
                "win_rate": self.win_rate, "size_prior": self.size_prior, "log_prior": self.log_prior}

    @classmethod
    def from_dict(cls, d):
        if d.get("model") != "pair_rule" or d.get("version") != 1:
            raise ValueError("Unsupported model file.")
        model = cls(d["half_life"], d["alpha"], d["beta"], d["size_strength"], d["prior_smoothing"], d["eb_iterations"])
        model.win_rate, model.size_prior, model.log_prior = d["win_rate"], d["size_prior"], d["log_prior"]
        if len(model.log_prior) != N_RULES or set(model.win_rate) != set(TYPES):
            raise ValueError("Malformed model file.")
        return model

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
