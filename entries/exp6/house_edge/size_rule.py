"""EXP-5: EXP-4 win model + payout-size regime tied to the win rule.

Finding (train, replicated on validation histories): a machine's winning-size regime at tick t is set by whether
the payout at the FIRST lag `a` of its win-rule pair (a, b) was >= 2, with a machine-specific orientation that does
not flip over time. Size here = sum over the win model's posterior mass of pair rules grouped by first lag a of
P(regime A | payout(t-a), orientation posterior from the machine's own past wins); null/single-lag rule mass uses an
even regime mix. The two size vectors per type are learned by EM on TRAIN histories with this cell structure.
"""

import math

from .history import machine_payouts
from .oriented import OrientedForecaster
from .pair_rule import TYPES, five_class
from .rules import N_RULES, PAIRS

SIZE_INDEX = {1: 0, 2: 1, 5: 2, 20: 3}
FIRST_PAIR_RULE = 1 + 8  # rule indices: 0 null, 1-8 single lags, 9-36 pairs (PAIRS order)


def first_lag(payouts, tick):
    """First lag of the most consistent win pair before `tick` (sign-agnostic over two halves); EM grouping only."""
    wins = {u: int(v > 0) for u, v in payouts.items() if u < tick}
    ticks = [u for u in wins if all(u - k in wins for k in range(1, 9))]
    if len(ticks) < 16:
        return None
    mid = ticks[len(ticks) // 2]
    parts = ([u for u in ticks if u < mid], [u for u in ticks if u >= mid])
    best = None
    for a, b in PAIRS:
        score = 0.0
        for part in parts:
            x = [wins[u - a] ^ wins[u - b] for u in part]
            n1 = sum(x)
            n0 = len(x) - n1
            if min(n1, n0) >= 3:
                y = [wins[u] for u in part]
                score += abs(sum(v for c, v in zip(x, y) if c) / n1 - sum(v for c, v in zip(x, y) if not c) / n0)
        if best is None or score > best[0]:
            best = (score, a)
    return best[1]


def size_groups(payouts, tick, lag):
    """Winning-size counts before `tick`, split by whether the payout `lag` ticks earlier was >= 2."""
    g = [[0.0] * 4, [0.0] * 4]
    for u, v in payouts.items():
        if u < tick and v > 0 and (u - lag) in payouts:
            g[int(payouts[u - lag] >= 2)][SIZE_INDEX[v]] += 1.0
    return g


def em_cells(groups, A, B, iterations=100):
    """Two size vectors with a latent per-machine orientation (cell1->A or cell1->B). Mirror of research em_cells."""
    A, B, pi = list(A), list(B), 0.5
    for _ in range(iterations):
        la, lb = [math.log(x) for x in A], [math.log(x) for x in B]
        r = []
        for g0, g1 in groups:
            l1 = math.log(pi) + math.fsum(c * x for c, x in zip(g1, la)) + math.fsum(c * x for c, x in zip(g0, lb))
            l0 = math.log(1 - pi) + math.fsum(c * x for c, x in zip(g0, la)) + math.fsum(c * x for c, x in zip(g1, lb))
            r.append(1.0 / (1.0 + math.exp(max(-700.0, min(700.0, l0 - l1)))))
        A = [math.fsum(ri * g1[j] + (1 - ri) * g0[j] for ri, (g0, g1) in zip(r, groups)) + 0.5 for j in range(4)]
        A = [x / math.fsum(A) for x in A]
        B = [math.fsum(ri * g0[j] + (1 - ri) * g1[j] for ri, (g0, g1) in zip(r, groups)) + 0.5 for j in range(4)]
        B = [x / math.fsum(B) for x in B]
        pi = min(0.95, max(0.05, math.fsum(r) / len(r)))
    if A[0] < B[0]:
        A, B = B, A
    return A, B


class SizeRuleForecaster(OrientedForecaster):
    def __init__(self, *args, size_em_iterations=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.size_em_iterations = int(size_em_iterations)
        self.rule_components = None

    def fit(self, queries, history, labels=None):
        super().fit(queries, history, labels)  # EXP-4 win model; static EM components seed the cell EM
        groups = {t: [] for t in TYPES}
        for q, payouts in machine_payouts(queries, history).values():
            a = first_lag(payouts, q.cutoff)
            if a is not None:
                groups[q.machine_type].append(size_groups(payouts, q.cutoff, a))
        self.rule_components = {t: [list(v) for v in em_cells(groups[t], *self.size_components[t][:2], self.size_em_iterations)]
                                for t in TYPES}
        return self

    def first_lag_masses(self, marginal):
        """Posterior mass per first lag 1..7 (index 0: null and single-lag rules)."""
        masses = [0.0] * 8
        for h, m in enumerate(marginal):
            r = self.rule_of(h)
            masses[PAIRS[r - FIRST_PAIR_RULE][0] if r >= FIRST_PAIR_RULE else 0] += m
        return masses

    def rule_size(self, payouts, tick, machine_type, marginal):
        A, B = self.rule_components[machine_type]
        diff = [math.log(x) - math.log(y) for x, y in zip(A, B)]
        masses = self.first_lag_masses(marginal)
        pa = 0.5 * masses[0]
        for lag in range(1, 8):
            if masses[lag] <= 0.0:
                continue
            cur = payouts.get(tick - lag)
            if cur is None:
                pa += 0.5 * masses[lag]
                continue
            g0, g1 = size_groups(payouts, tick, lag)
            lo = math.fsum((c1 - c0) * d for c0, c1, d in zip(g0, g1, diff))
            p1 = 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, lo))))  # P(cell 1 -> regime A)
            pa += masses[lag] * (p1 if cur >= 2 else 1.0 - p1)
        return [pa * x + (1.0 - pa) * y for x, y in zip(A, B)]

    def predict(self, queries, history):
        if self.log_prior is None:
            raise ValueError("Fit or load the model before predicting.")
        machines = machine_payouts(queries, history)
        episodes = {}
        for qid, (q, _) in machines.items():
            episodes.setdefault((q.episode_id, q.cutoff), []).append(qid)
        result = {}
        for key in sorted(episodes):
            qids = sorted(episodes[key])
            tables = [self.query_tables(*machines[qid]) for qid in qids]
            ws, _, marginals = self.episode_win_probabilities(tables, with_marginals=True)
            for qid, w, marginal in zip(qids, ws, marginals):
                q, payouts = machines[qid]
                result[qid] = five_class(self.calibrate(w, q.machine_type),
                                         self.rule_size(payouts, q.cutoff, q.machine_type, marginal))
        return result

    def predict_machine(self, payouts, tick, machine_type):
        ws, _, marginals = self.episode_win_probabilities([self.machine_tables(payouts, tick, machine_type)], with_marginals=True)
        return (self.calibrate(ws[0], machine_type), *self.rule_size(payouts, tick, machine_type, marginals[0]))

    def to_dict(self):
        d = super().to_dict()
        d.update(model="size_rule", size_em_iterations=self.size_em_iterations, rule_components=self.rule_components)
        return d

    @classmethod
    def from_dict(cls, d):
        if d.get("model") != "size_rule" or d.get("version") != 1:
            raise ValueError("Unsupported model file.")
        model = cls(d["alpha"], d["beta"], d["no_change_prior"], d["shared_prior"], d["size_strength"],
                    d["prior_smoothing"], d["eb_iterations"], d["d_min"], d["d_max"], d["d_step"], d["min_pre"],
                    d["calibration_slope"], d["calibration_intercepts"], d["size_model"], d["regime_temper"],
                    d["em_iterations"], d["orient_strength"], d["contrast_threshold"],
                    size_em_iterations=d["size_em_iterations"])
        model.win_rate, model.size_prior, model.log_prior = d["win_rate"], d["size_prior"], d["log_prior"]
        model.size_components, model.levels, model.rule_components = d["size_components"], d["levels"], d["rule_components"]
        if len(model.log_prior) != N_RULES or set(model.rule_components) != set(TYPES):
            raise ValueError("Malformed model file.")
        return model
