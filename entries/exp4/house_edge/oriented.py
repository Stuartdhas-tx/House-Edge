"""EXP-4: oriented hierarchical cell priors for the change-aware rule model (plus EXP-3 sizes/calibration).

Every non-null rule has an orientation o (prior 1/2): cell o is the 'high' cell with prior
Beta(s*hi[type], s*(1-hi[type])) and the other cell is 'low' with Beta(s*lo[type], ...). The pre-change
orientation is marginalised (a change may flip it); the post-change orientation predicts. hi/lo per type are
moment estimates from training machines whose clearest full-history XOR rule has contrast > 0.5 (an
observable criterion; no condition labels). Hypothesis columns: [null] + [(r, o=0)] + [(r, o=1)], r = 1..36.
"""

import math

from .calibrated import CalibratedChangePointForecaster, em_sizes
from .change_point import logsumexp, softmax
from .history import machine_payouts, size_counts, win_map
from .pair_rule import TYPES, PairRuleForecaster
from .rules import N_RULES, PAIRS, rule_cells, training_events

LOG_HALF = math.log(0.5)


def cell_log_ml(n, k, a, b):
    if n <= 0:
        return 0.0
    return (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
            + math.lgamma(a + k) + math.lgamma(b + n - k) - math.lgamma(a + b + n))


def logaddexp(x, y):
    top = max(x, y)
    return top + math.log1p(math.exp(-abs(x - y)))


def hi_lo_levels(machines, threshold=0.5, min_cell=5):
    """machines: iterable of (machine_type, payouts). Mean (high, low) cell win rate per type."""
    acc = {t: [] for t in TYPES}
    for machine_type, payouts in machines:
        wins = win_map(payouts)
        ticks = [t for t in payouts if all(t - k in payouts for k in range(1, 9))]
        best = None
        for a, b in PAIRS:
            x = [wins[t - a] ^ wins[t - b] for t in ticks]
            y = [wins[t] for t in ticks]
            n1 = sum(x)
            n0 = len(x) - n1
            if n1 < min_cell or n0 < min_cell:
                continue
            r1 = sum(v for c, v in zip(x, y) if c) / n1
            r0 = sum(v for c, v in zip(x, y) if not c) / n0
            if best is None or abs(r1 - r0) > best[0]:
                best = (abs(r1 - r0), max(r1, r0), min(r1, r0))
        if best and best[0] > threshold:
            acc[machine_type].append(best[1:])
    if any(not acc[t] for t in TYPES):
        raise ValueError("Every machine type needs at least one training machine with a clear rule.")
    return {t: [math.fsum(v[0] for v in acc[t]) / len(acc[t]), math.fsum(v[1] for v in acc[t]) / len(acc[t])] for t in TYPES}


class OrientedForecaster(CalibratedChangePointForecaster):
    def __init__(self, alpha=0.5, beta=1.0, no_change_prior=0.5, shared_prior=0.5, size_strength=50.0,
                 prior_smoothing=1.0, eb_iterations=20, d_min=8, d_max=200, d_step=4, min_pre=8,
                 calibration_slope=1.0, calibration_intercepts=None, size_model="regime", regime_temper=0.75,
                 em_iterations=200, orient_strength=40.0, contrast_threshold=0.5):
        super().__init__(alpha, beta, no_change_prior, shared_prior, size_strength, prior_smoothing, eb_iterations,
                         d_min, d_max, d_step, min_pre, calibration_slope, calibration_intercepts, size_model,
                         regime_temper, em_iterations)
        self.orient_strength = float(orient_strength)
        self.contrast_threshold = float(contrast_threshold)
        self.levels = None

    # hypothesis h -> rule
    @staticmethod
    def rule_of(h):
        return 0 if h == 0 else 1 + (h - 1) % (N_RULES - 1)

    def hypothesis_log_prior(self):
        return [self.log_prior[0]] + self.log_prior[1:] + self.log_prior[1:]

    def _tables_for(self, n_post, k_post, pre, cells_now, machine_type):
        hi, lo = self.levels[machine_type]
        s = self.orient_strength
        prior = {True: (s * hi, s * (1 - hi)), False: (s * lo, s * (1 - lo))}
        a0, b0 = self._cell_prior(machine_type)
        ml0 = cell_log_ml(n_post[0], k_post[0], a0, b0)
        if pre is not None:
            ml0 += cell_log_ml(pre[0][0], pre[1][0], a0, b0)
        ml, pr = [ml0], [(a0 + k_post[0]) / (a0 + b0 + n_post[0])]
        cols = {0: ([], []), 1: ([], [])}
        for o in (0, 1):
            for r in range(1, N_RULES):
                post = math.fsum(cell_log_ml(n_post[2 * r + c], k_post[2 * r + c], *prior[c == o]) for c in (0, 1))
                cur = cells_now[r]
                if cur is None:
                    p = 0.5 * math.fsum((prior[c == o][0] + k_post[2 * r + c]) / (sum(prior[c == o]) + n_post[2 * r + c]) for c in (0, 1))
                else:
                    a, b = prior[cur == o]
                    p = (a + k_post[2 * r + cur]) / (a + b + n_post[2 * r + cur])
                cols[o][0].append(post)
                cols[o][1].append(p)
        pre_marg = [0.0] * (N_RULES - 1)
        if pre is not None:
            for i, r in enumerate(range(1, N_RULES)):
                pre_o = [math.fsum(cell_log_ml(pre[0][2 * r + c], pre[1][2 * r + c], *prior[c == o]) for c in (0, 1)) for o in (0, 1)]
                pre_marg[i] = logaddexp(pre_o[0], pre_o[1]) + LOG_HALF
        for o in (0, 1):
            ml.extend(post + LOG_HALF + pm for post, pm in zip(cols[o][0], pre_marg))
            pr.extend(cols[o][1])
        return ml, pr

    def machine_tables(self, payouts, tick, machine_type):
        events = training_events(payouts)
        cells_now = rule_cells(win_map(payouts), tick)
        bounds = sorted({tick - d for d in self.offsets} | {tick})
        snaps, n, k, count, j = {}, [0.0] * (2 * N_RULES), [0.0] * (2 * N_RULES), 0, 0
        for x in bounds:
            while j < len(events) and events[j][0] < x:
                _, win, indices = events[j]
                for i in indices:
                    n[i] += 1.0
                if win:
                    for i in indices:
                        k[i] += 1.0
                count += 1
                j += 1
            snaps[x] = (list(n), list(k), count)
        n_all, k_all, _ = snaps[tick]
        none = self._tables_for(n_all, k_all, None, cells_now, machine_type)
        logml, preds = [none[0]], [none[1]]
        for d in self.offsets:
            n_pre, k_pre, c_pre = snaps[tick - d]
            if c_pre < self.min_pre:
                logml.append(none[0])
                preds.append(none[1])
                continue
            n_post = [x - y for x, y in zip(n_all, n_pre)]
            k_post = [x - y for x, y in zip(k_all, k_pre)]
            ml, pr = self._tables_for(n_post, k_post, (n_pre, k_pre), cells_now, machine_type)
            logml.append(ml)
            preds.append(pr)
        return logml, preds

    def fit(self, queries, history, labels=None):
        PairRuleForecaster.fit(self, queries, history, labels)  # type base rates and Dirichlet size priors
        machines = machine_payouts(queries, history)
        self.levels = hi_lo_levels(((q.machine_type, p) for q, p in machines.values()), self.contrast_threshold)
        by_type = {t: [] for t in TYPES}
        for q, payouts in machines.values():
            by_type[q.machine_type].append([float(c) for c in size_counts(payouts, q.cutoff)])
        self.size_components = {t: list(em_sizes(by_type[t], self.em_iterations)) for t in TYPES}
        log_p_d = self.log_offset_prior()
        per_h_all = []
        for q, payouts in machines.values():
            table = self.machine_tables(payouts, q.cutoff, q.machine_type)[0]
            per_h_all.append([logsumexp([lp + self.beta * row[h] for lp, row in zip(log_p_d, table)]) for h in range(len(table[0]))])
        n_h = len(per_h_all[0])
        prior = [1.0 / N_RULES] * N_RULES
        for _ in range(self.eb_iterations):
            lp = [math.log(p) for p in prior]
            expanded = [lp[0]] + lp[1:] + lp[1:]
            mass = [self.prior_smoothing] * N_RULES
            for per_h in per_h_all:
                for h, p in enumerate(softmax([a + b for a, b in zip(expanded, per_h)])):
                    mass[self.rule_of(h)] += p
            total = math.fsum(mass)
            prior = [m / total for m in mass]
        self.log_prior = [math.log(p) for p in prior]
        assert n_h == 1 + 2 * (N_RULES - 1)
        return self

    def to_dict(self):
        d = super().to_dict()
        d.update(model="oriented", orient_strength=self.orient_strength, contrast_threshold=self.contrast_threshold,
                 levels=self.levels)
        return d

    @classmethod
    def from_dict(cls, d):
        if d.get("model") != "oriented" or d.get("version") != 1:
            raise ValueError("Unsupported model file.")
        model = cls(d["alpha"], d["beta"], d["no_change_prior"], d["shared_prior"], d["size_strength"],
                    d["prior_smoothing"], d["eb_iterations"], d["d_min"], d["d_max"], d["d_step"], d["min_pre"],
                    d["calibration_slope"], d["calibration_intercepts"], d["size_model"], d["regime_temper"],
                    d["em_iterations"], d["orient_strength"], d["contrast_threshold"])
        model.win_rate, model.size_prior, model.log_prior = d["win_rate"], d["size_prior"], d["log_prior"]
        model.size_components, model.levels = d["size_components"], d["levels"]
        if len(model.log_prior) != N_RULES or set(model.levels) != set(TYPES):
            raise ValueError("Malformed model file.")
        return model
