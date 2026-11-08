"""EXP-2: change-aware pair-rule forecaster with episode-level shared-change pooling.

For each machine, every rule r is scored under "no change" (all observed events) and under a change d ticks
before the prediction tick (separate Beta-Bernoulli cells before/after tick-d; the post-change cells predict).
A change whose pre-segment has fewer than `min_pre` events is indistinguishable from no change and reuses it.

Within an episode (machines share the prediction tick), two observable hypotheses are compared:
  shared:      one change offset d for all machines   Z_s = sum_d P(d) prod_m L_m(d)
  independent: each machine has its own d             Z_i = prod_m sum_d P(d) L_m(d)
with L_m(d) = sum_r prior_r ML_m(r, d)^beta. Condition labels are never used.
"""

import math

from .history import machine_payouts, win_map
from .pair_rule import PairRuleForecaster, five_class, log_marginals, predictives
from .rules import N_RULES, rule_cells, training_events


def logsumexp(values):
    top = max(values)
    return top + math.log(math.fsum(math.exp(v - top) for v in values))


def softmax(values):
    top = max(values)
    e = [math.exp(v - top) for v in values]
    total = math.fsum(e)
    return [x / total for x in e]


class ChangePointForecaster(PairRuleForecaster):
    def __init__(self, alpha=1.0, beta=1.0, no_change_prior=0.9, shared_prior=0.5, size_strength=50.0,
                 prior_smoothing=1.0, eb_iterations=20, d_min=8, d_max=200, d_step=4, min_pre=8):
        super().__init__(None, alpha, beta, size_strength, prior_smoothing, eb_iterations)
        self.no_change_prior = float(no_change_prior)
        self.shared_prior = float(shared_prior)
        self.d_min, self.d_max, self.d_step, self.min_pre = int(d_min), int(d_max), int(d_step), int(min_pre)

    @property
    def offsets(self):
        return list(range(self.d_min, self.d_max + 1, self.d_step))

    def log_offset_prior(self):
        n = len(self.offsets)
        return [math.log(self.no_change_prior)] + [math.log((1.0 - self.no_change_prior) / n)] * n

    # ---- per-machine evidence tables ------------------------------------------------------
    def machine_tables(self, payouts, tick, machine_type):
        """log ML and post-change predictive per (offset index, rule); index 0 = no change."""
        a0, b0 = self._cell_prior(machine_type)
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
        none_ml = log_marginals(n_all, k_all, a0, b0)
        none_pred = predictives(n_all, k_all, cells_now, a0, b0)
        logml, preds = [none_ml], [none_pred]
        for d in self.offsets:
            n_pre, k_pre, c_pre = snaps[tick - d]
            if c_pre < self.min_pre:
                logml.append(none_ml)
                preds.append(none_pred)
                continue
            n_post = [x - y for x, y in zip(n_all, n_pre)]
            k_post = [x - y for x, y in zip(k_all, k_pre)]
            pre_ml = log_marginals(n_pre, k_pre, a0, b0)
            post_ml = log_marginals(n_post, k_post, a0, b0)
            logml.append([p + q for p, q in zip(pre_ml, post_ml)])
            preds.append(predictives(n_post, k_post, cells_now, a0, b0))
        return logml, preds

    # ---- training -------------------------------------------------------------------------
    def fit(self, queries, history, labels=None):
        super().fit(queries, history, labels)  # type base rates, size priors (its rule prior is replaced)
        log_p_d = self.log_offset_prior()
        evidence = [self.machine_tables(payouts, q.cutoff, q.machine_type)[0]
                    for q, payouts in machine_payouts(queries, history).values()]
        prior = [1.0 / N_RULES] * N_RULES
        for _ in range(self.eb_iterations):
            log_prior = [math.log(p) for p in prior]
            mass = [self.prior_smoothing] * N_RULES
            for table in evidence:
                per_rule = [logsumexp([lp + self.beta * row[r] for lp, row in zip(log_p_d, table)]) for r in range(N_RULES)]
                for r, p in enumerate(softmax([a + b for a, b in zip(log_prior, per_rule)])):
                    mass[r] += p
            total = math.fsum(mass)
            prior = [m / total for m in mass]
        self.log_prior = [math.log(p) for p in prior]
        return self

    # ---- inference ------------------------------------------------------------------------
    def episode_win_probabilities(self, tables):
        """tables: list of (logml, preds) for machines sharing an episode and prediction tick."""
        log_p_d = self.log_offset_prior()
        n_off = len(log_p_d)
        L, cond_pred = [], []
        for logml, preds in tables:
            Lm, pm = [], []
            for d in range(n_off):
                scores = [lp + self.beta * m for lp, m in zip(self.log_prior, logml[d])]
                Lm.append(logsumexp(scores))
                post = softmax(scores)
                pm.append(math.fsum(p * q for p, q in zip(post, preds[d])))
            L.append(Lm)
            cond_pred.append(pm)
        indep = [[lp + l for lp, l in zip(log_p_d, Lm)] for Lm in L]
        shared = [log_p_d[d] + math.fsum(Lm[d] for Lm in L) for d in range(n_off)]
        if self.shared_prior > 0 and len(tables) > 1:
            log_zs = math.log(self.shared_prior) + logsumexp(shared)
            log_zi = (math.log(1.0 - self.shared_prior) + math.fsum(logsumexp(row) for row in indep)
                      if self.shared_prior < 1 else -math.inf)
            p_shared = 1.0 / (1.0 + math.exp(min(700.0, log_zi - log_zs)))
        else:
            p_shared = 0.0
        shared_post = softmax(shared)
        result = []
        for row, pm in zip(indep, cond_pred):
            pd = [p_shared * s + (1.0 - p_shared) * i for s, i in zip(shared_post, softmax(row))]
            result.append(math.fsum(a * b for a, b in zip(pd, pm)))
        return result, p_shared

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
            tables = [self.machine_tables(machines[qid][1], machines[qid][0].cutoff, machines[qid][0].machine_type) for qid in qids]
            ws, _ = self.episode_win_probabilities(tables)
            for qid, w in zip(qids, ws):
                q, payouts = machines[qid]
                result[qid] = five_class(w, self.size_distribution(payouts, q.cutoff, q.machine_type))
        return result

    def predict_machine(self, payouts, tick, machine_type):
        """Single-machine (independent-hypothesis) prediction, used by leakage tests."""
        ws, _ = self.episode_win_probabilities([self.machine_tables(payouts, tick, machine_type)])
        return (ws[0], *self.size_distribution(payouts, tick, machine_type))

    # ---- persistence ----------------------------------------------------------------------
    def to_dict(self):
        d = super().to_dict()
        d.update(model="change_point", no_change_prior=self.no_change_prior, shared_prior=self.shared_prior,
                 d_min=self.d_min, d_max=self.d_max, d_step=self.d_step, min_pre=self.min_pre)
        return d

    @classmethod
    def from_dict(cls, d):
        if d.get("model") != "change_point" or d.get("version") != 1:
            raise ValueError("Unsupported model file.")
        model = cls(d["alpha"], d["beta"], d["no_change_prior"], d["shared_prior"], d["size_strength"],
                    d["prior_smoothing"], d["eb_iterations"], d["d_min"], d["d_max"], d["d_step"], d["min_pre"])
        model.win_rate, model.size_prior, model.log_prior = d["win_rate"], d["size_prior"], d["log_prior"]
        if len(model.log_prior) != N_RULES:
            raise ValueError("Malformed model file.")
        return model
