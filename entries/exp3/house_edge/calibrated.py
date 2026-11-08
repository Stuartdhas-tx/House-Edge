"""EXP-3: EXP-2 plus a two-regime payout-size model and win-probability calibration.

Size: per machine type, a two-component multinomial mixture over winning payout sizes with one latent
regime per training machine, fit by EM on TRAINING histories. A machine's regime posterior uses its own
winning sizes with a tempered log-likelihood ratio (`regime_temper`).
Calibration: logit(w') = slope * logit(w) + intercept[type]; slope/intercepts were fit on out-of-fold
TRAIN pseudo-query predictions (research/cv_exp3.py) and are stored in the config.
"""

import math

from .change_point import ChangePointForecaster
from .history import SIZES, machine_payouts, size_counts
from .pair_rule import EPSILON, TYPES, five_class
from .rules import N_RULES


def em_sizes(counts, iterations=200):
    """Mirror of research/cv_exp3.em_sizes (checked by tests/test_equivalence_exp3.py)."""
    tot = [sum(c[j] for c in counts) + 1.0 for j in range(4)]
    m = [x / math.fsum(tot) for x in tot]
    A = [x * f for x, f in zip(m, (1.3, 0.9, 0.8, 0.8))]
    A = [x / math.fsum(A) for x in A]
    B = [x * f for x, f in zip(m, (0.7, 1.1, 1.2, 1.2))]
    B = [x / math.fsum(B) for x in B]
    pi = 0.5
    for _ in range(iterations):
        la_w = [math.log(x) for x in A]
        lb_w = [math.log(x) for x in B]
        r = []
        for c in counts:
            la = math.log(pi) + math.fsum(x * y for x, y in zip(c, la_w))
            lb = math.log(1 - pi) + math.fsum(x * y for x, y in zip(c, lb_w))
            r.append(1.0 / (1.0 + math.exp(max(-700.0, min(700.0, lb - la)))))
        A = [math.fsum(ri * c[j] for ri, c in zip(r, counts)) + 0.5 for j in range(4)]
        A = [x / math.fsum(A) for x in A]
        B = [math.fsum((1 - ri) * c[j] for ri, c in zip(r, counts)) + 0.5 for j in range(4)]
        B = [x / math.fsum(B) for x in B]
        pi = min(1 - 1e-3, max(1e-3, math.fsum(r) / len(r)))
    if A[0] < B[0]:
        A, B, pi = B, A, 1 - pi
    return A, B, pi


class CalibratedChangePointForecaster(ChangePointForecaster):
    def __init__(self, alpha=0.5, beta=1.0, no_change_prior=0.5, shared_prior=0.5, size_strength=50.0,
                 prior_smoothing=1.0, eb_iterations=20, d_min=8, d_max=200, d_step=4, min_pre=8,
                 calibration_slope=1.0, calibration_intercepts=None, size_model="regime", regime_temper=0.75,
                 em_iterations=200):
        super().__init__(alpha, beta, no_change_prior, shared_prior, size_strength, prior_smoothing,
                         eb_iterations, d_min, d_max, d_step, min_pre)
        self.calibration_slope = float(calibration_slope)
        self.calibration_intercepts = {t: float((calibration_intercepts or {}).get(t, 0.0)) for t in TYPES}
        if size_model not in ("regime", "dirichlet"):
            raise ValueError("size_model must be 'regime' or 'dirichlet'.")
        self.size_model = size_model
        self.regime_temper = float(regime_temper)
        self.em_iterations = int(em_iterations)
        self.size_components = None

    def fit(self, queries, history, labels=None):
        super().fit(queries, history, labels)
        by_type = {t: [] for t in TYPES}
        for q, payouts in machine_payouts(queries, history).values():
            by_type[q.machine_type].append([float(c) for c in size_counts(payouts, q.cutoff)])
        self.size_components = {t: list(em_sizes(by_type[t], self.em_iterations)) for t in TYPES}
        return self

    def size_distribution(self, payouts, tick, machine_type):
        if self.size_model == "dirichlet":
            return super().size_distribution(payouts, tick, machine_type)
        A, B, pi = self.size_components[machine_type]
        counts = size_counts(payouts, tick)
        llr = math.fsum(c * (math.log(a) - math.log(b)) for c, a, b in zip(counts, A, B))
        z = self.regime_temper * llr + math.log(pi / (1 - pi))
        pa = 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, z))))
        return [pa * a + (1 - pa) * b for a, b in zip(A, B)]

    def calibrate(self, w, machine_type):
        w = min(1.0 - EPSILON, max(EPSILON, w))
        z = self.calibration_slope * math.log(w / (1.0 - w)) + self.calibration_intercepts[machine_type]
        return 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, z))))

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
                result[qid] = five_class(self.calibrate(w, q.machine_type), self.size_distribution(payouts, q.cutoff, q.machine_type))
        return result

    def predict_machine(self, payouts, tick, machine_type):
        ws, _ = self.episode_win_probabilities([self.machine_tables(payouts, tick, machine_type)])
        return (self.calibrate(ws[0], machine_type), *self.size_distribution(payouts, tick, machine_type))

    def to_dict(self):
        d = super().to_dict()
        d.update(model="calibrated_change_point", calibration_slope=self.calibration_slope,
                 calibration_intercepts=self.calibration_intercepts, size_model=self.size_model,
                 regime_temper=self.regime_temper, em_iterations=self.em_iterations,
                 size_components=self.size_components)
        return d

    @classmethod
    def from_dict(cls, d):
        if d.get("model") != "calibrated_change_point" or d.get("version") != 1:
            raise ValueError("Unsupported model file.")
        model = cls(d["alpha"], d["beta"], d["no_change_prior"], d["shared_prior"], d["size_strength"],
                    d["prior_smoothing"], d["eb_iterations"], d["d_min"], d["d_max"], d["d_step"], d["min_pre"],
                    d["calibration_slope"], d["calibration_intercepts"], d["size_model"], d["regime_temper"],
                    d["em_iterations"])
        model.win_rate, model.size_prior, model.log_prior = d["win_rate"], d["size_prior"], d["log_prior"]
        model.size_components = d["size_components"]
        if len(model.log_prior) != N_RULES or set(model.size_components) != set(TYPES):
            raise ValueError("Malformed model file.")
        return model
