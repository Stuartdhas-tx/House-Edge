"""Joint rule/size inference on top of the unchanged EXP-5 fitted parameters.

Size evidence can update the win-rule posterior, and winning payouts can be
averaged jointly with win probabilities.  The two switches are separate so
their effects can be compared with the original product of posterior means.
Evidence uses a common outcome set: either complete-window events, or all
observed wins with an explicit neutral mixture for unavailable lag values.
"""

import math

from .change_point import logsumexp, softmax
from .history import machine_payouts
from .pair_rule import five_class
from .rules import MAX_LAG, PAIRS, training_events
from .size_rule import FIRST_PAIR_RULE, SIZE_INDEX, SizeRuleForecaster, size_groups

LOG_HALF = math.log(0.5)


def size_evidence(payouts, tick, A, B, common=True, complete_missing=False):
    """Return (log evidence, predictive size), each indexed by first lag.

    Index 0 represents null/single-lag rules with the fixed even regime mix;
    indices 1..MAX_LAG-1 represent a pair's first lag and latent orientation.
    With ``common=True``, every likelihood explains the same winning events
    whose entire lag window is observed.  Otherwise per-lag available events
    reproduce EXP-5's orientation estimate.  With ``complete_missing=True``,
    missing-lag outcomes contribute the even regime mixture to both orientation
    likelihoods, so every candidate explains all observed wins.  This assumes
    an independent 50/50 latent lag category; it is not a causal history filter.
    Without either option the likelihoods must NOT be used to compare rules
    with different missing histories.

    Future payouts are ignored, including when this helper is used directly.
    """
    if common and complete_missing:
        raise ValueError("common and complete_missing are alternative size evidence modes.")
    past = {u: v for u, v in payouts.items() if u < tick}
    if common:
        eligible = [u for u, win, _ in training_events(past) if win]
    else:
        eligible = [u for u, value in past.items() if value > 0]
    counts = [0.0] * len(SIZE_INDEX)
    for u in eligible:
        counts[SIZE_INDEX[past[u]]] += 1.0
    mix = [(a + b) * 0.5 for a, b in zip(A, B)]
    la, lb = [math.log(a) for a in A], [math.log(b) for b in B]
    diff = [a - b for a, b in zip(la, lb)]
    log_mix = [math.log(p) for p in mix]
    evidence = [math.fsum(c * p for c, p in zip(counts, log_mix))]
    predictions = [mix]
    for lag in range(1, MAX_LAG):
        if common:
            g0, g1 = [[0.0] * len(SIZE_INDEX) for _ in range(2)]
            groups = (g0, g1)
            for u in eligible:
                groups[int(past[u - lag] >= 2)][SIZE_INDEX[past[u]]] += 1.0
        else:
            g0, g1 = size_groups(past, tick, lag)
        log1 = LOG_HALF + math.fsum(c * p for c, p in zip(g1, la)) + math.fsum(c * p for c, p in zip(g0, lb))
        log0 = LOG_HALF + math.fsum(c * p for c, p in zip(g0, la)) + math.fsum(c * p for c, p in zip(g1, lb))
        if complete_missing:
            missing_ml = math.fsum((c - c0 - c1) * p for c, c0, c1, p in zip(counts, g0, g1, log_mix))
            log1 += missing_ml
            log0 += missing_ml
        evidence.append(logsumexp([log0, log1]))
        log_odds = math.fsum((c1 - c0) * d for c0, c1, d in zip(g0, g1, diff))
        p1 = 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, log_odds))))
        current = past.get(tick - lag)
        pa = 0.5 if current is None else (p1 if current >= 2 else 1.0 - p1)
        predictions.append([pa * a + (1.0 - pa) * b for a, b in zip(A, B)])
    return evidence, predictions


class JointSizeForecaster(SizeRuleForecaster):
    """EXP-5 fitting with optional joint evidence and joint predictive averaging."""

    def __init__(self, *args, joint_size_evidence=True, joint_size_common=True,
                 joint_average=True, joint_size_missing_mixture=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.joint_size_evidence = joint_size_evidence
        self.joint_size_common = joint_size_common
        self.joint_average = joint_average
        self.joint_size_missing_mixture = joint_size_missing_mixture
        self._validate_joint_options()

    def _validate_joint_options(self):
        for name in ("joint_size_evidence", "joint_size_common", "joint_average", "joint_size_missing_mixture"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")
        if self.joint_size_common and self.joint_size_missing_mixture:
            raise ValueError("joint_size_common and joint_size_missing_mixture are alternative evidence modes.")
        if self.joint_size_evidence and not (self.joint_size_common or self.joint_size_missing_mixture):
            raise ValueError("joint_size_evidence requires a common outcome set: joint_size_common or joint_size_missing_mixture.")

    def hypothesis_first_lag(self, h):
        rule = self.rule_of(h)
        return PAIRS[rule - FIRST_PAIR_RULE][0] if rule >= FIRST_PAIR_RULE else 0

    def machine_size_evidence(self, payouts, tick, machine_type):
        return size_evidence(payouts, tick, *self.rule_components[machine_type],
                             common=self.joint_size_common, complete_missing=self.joint_size_missing_mixture)

    def joint_episode_probabilities(self, tables, size_tables):
        """Return (raw wins, conditional four-size vectors, P(shared change)).

        ``tables`` contains EXP-5 (win log-marginals, win predictives), each
        shaped [offset][hypothesis].  ``size_tables`` contains the tuples from
        ``size_evidence`` for the same machines, in the same order.
        """
        self._validate_joint_options()
        if len(tables) != len(size_tables):
            raise ValueError("Win and size tables must contain the same machines.")
        if not tables:
            return [], [], 0.0
        log_p_d = self.log_offset_prior()
        n_off = len(log_p_d)
        hyp_prior = self.hypothesis_log_prior()
        first_lags = [self.hypothesis_first_lag(h) for h in range(len(hyp_prior))]
        likelihoods, conditional_wins, posts = [], [], []
        for (logml, preds), (size_ml, _) in zip(tables, size_tables):
            evidence = [size_ml[lag] if self.joint_size_evidence else 0.0 for lag in first_lags]
            lm, pm, ph = [], [], []
            for d in range(n_off):
                # beta tempers only the original win likelihood, not size evidence.
                scores = [lp + self.beta * ml + se for lp, ml, se in zip(hyp_prior, logml[d], evidence)]
                lm.append(logsumexp(scores))
                post = softmax(scores)
                pm.append(math.fsum(p * w for p, w in zip(post, preds[d])))
                ph.append(post)
            likelihoods.append(lm)
            conditional_wins.append(pm)
            posts.append(ph)
        independent = [[lp + lm for lp, lm in zip(log_p_d, row)] for row in likelihoods]
        shared = [log_p_d[d] + math.fsum(row[d] for row in likelihoods) for d in range(n_off)]
        if self.shared_prior > 0 and len(tables) > 1:
            log_zs = math.log(self.shared_prior) + logsumexp(shared)
            log_zi = (math.log(1.0 - self.shared_prior) + math.fsum(logsumexp(row) for row in independent)
                      if self.shared_prior < 1 else -math.inf)
            p_shared = 1.0 / (1.0 + math.exp(min(700.0, log_zi - log_zs)))
        else:
            p_shared = 0.0
        shared_post = softmax(shared)
        wins, sizes = [], []
        for row, pm, ph, (_, preds), (_, size_preds) in zip(
                independent, conditional_wins, posts, tables, size_tables):
            pd = [p_shared * s + (1.0 - p_shared) * i for s, i in zip(shared_post, softmax(row))]
            w = math.fsum(p * v for p, v in zip(pd, pm))
            lag_terms = [[] for _ in range(MAX_LAG)]
            for d in range(n_off):
                for h, lag in enumerate(first_lags):
                    mass = pd[d] * ph[d][h]
                    if self.joint_average:
                        mass *= preds[d][h]
                    lag_terms[lag].append(mass)
            masses = [math.fsum(terms) for terms in lag_terms]
            denominator = w if self.joint_average else math.fsum(masses)
            if denominator <= 0.0:
                # Only possible for externally supplied degenerate win tables.
                size = list(size_preds[0])
            else:
                size = [math.fsum(m * s[j] for m, s in zip(masses, size_preds)) / denominator
                        for j in range(len(SIZE_INDEX))]
            wins.append(w)
            sizes.append(size)
        return wins, sizes, p_shared

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
            size_tables = [self.machine_size_evidence(p, q.cutoff, q.machine_type)
                           for q, p in (machines[qid] for qid in qids)]
            ws, sizes, _ = self.joint_episode_probabilities(tables, size_tables)
            for qid, w, size in zip(qids, ws, sizes):
                q, _ = machines[qid]
                result[qid] = five_class(self.calibrate(w, q.machine_type), size)
        return result

    def predict_machine(self, payouts, tick, machine_type):
        tables = [self.machine_tables(payouts, tick, machine_type)]
        size_tables = [self.machine_size_evidence(payouts, tick, machine_type)]
        ws, sizes, _ = self.joint_episode_probabilities(tables, size_tables)
        return (self.calibrate(ws[0], machine_type), *sizes[0])

    def to_dict(self):
        d = super().to_dict()
        d.update(model="joint_size", joint_size_evidence=self.joint_size_evidence,
                 joint_size_common=self.joint_size_common, joint_average=self.joint_average,
                 joint_size_missing_mixture=self.joint_size_missing_mixture)
        return d

    @classmethod
    def from_dict(cls, d):
        if d.get("model") != "joint_size" or d.get("version") != 1:
            raise ValueError("Unsupported model file.")
        model = super().from_dict(dict(d, model="size_rule"))
        for name in ("joint_size_evidence", "joint_size_common", "joint_average"):
            setattr(model, name, d.get(name, True))
        model.joint_size_missing_mixture = d.get("joint_size_missing_mixture", False)
        model._validate_joint_options()
        return model
