"""Import-safe NumPy reference for four fixed EXP-5/joint-size variants.

Only the caller chooses data or writes files. Global parameters retain EXP-5's
fitting procedure; size likelihood changes posterior inference, never EB fitting.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp

from common import TYPES, Snapshots
from cv_change import ChangeTables, log_offset_prior
from cv_exp3 import em_sizes
from cv_oriented import expand_prior, hi_lo_levels, oriented_tables
from cv_pair_rule import five, machine_stats, type_priors
from house_edge.rules import N_RULES, PAIRS
from house_edge.size_rule import first_lag

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("exp5", "joint_mix", "common_mix", "joint_size")
SIZE_INDEX = {1: 0, 2: 1, 5: 2, 20: 3}
RULE_OF = np.concatenate([[0], np.arange(1, N_RULES), np.arange(1, N_RULES)])
FIRST_OF_H = np.array([0] * 9 + [a for a, _ in PAIRS])[RULE_OF]


def em_cells(groups, a, b, iterations=100):
    """Same cell EM and pseudocounts as EXP-5, without importing its runner."""
    a, b, pi = np.array(a, float), np.array(b, float), 0.5
    for _ in range(iterations):
        l1 = np.log(pi) + groups[:, 1] @ np.log(a) + groups[:, 0] @ np.log(b)
        l0 = np.log(1 - pi) + groups[:, 0] @ np.log(a) + groups[:, 1] @ np.log(b)
        r = 1 / (1 + np.exp(np.clip(l0 - l1, -700, 700)))
        a = (r[:, None] * groups[:, 1] + (1 - r)[:, None] * groups[:, 0]).sum(0) + 0.5
        b = (r[:, None] * groups[:, 0] + (1 - r)[:, None] * groups[:, 1]).sum(0) + 0.5
        a /= a.sum()
        b /= b.sum()
        pi = float(np.clip(r.mean(), 0.05, 0.95))
    return (a, b) if a[0] >= b[0] else (b, a)


def groups_before(payouts, tick, lag):
    groups = np.zeros((2, 4))
    for u, value in payouts.items():
        if u < tick and value > 0 and u - lag in payouts:
            groups[int(payouts[u - lag] >= 2), SIZE_INDEX[value]] += 1
    return groups


def calibrated_win(w, theta, machine_type):
    w = np.clip(w, 1e-6, 1 - 1e-6)
    theta = np.asarray(theta)
    return expit(theta[0] * np.log(w / (1 - w)) + theta[1 + machine_type])


class JointNumpyHarness:
    """Fit on episode-complete row masks; predict every snapshot from its past.

    Public APIs for equivalence checks:
      predict_model(model_dict) -> {variant: (raw_win, conditional_size)}
      probabilities((raw_win, conditional_size), theta=None) -> N x 5
    Use k_back=1 for the supplied query cutoffs; row order is self.snaps.qid.
    """

    def __init__(self, queries, history, labels, conditions, *, config=None, k_back=16,
                 include_missing_mixture=False):
        self.config = dict(config or json.loads((ROOT / "configs/exp5.json").read_text()))
        self.include_missing_mixture = bool(include_missing_mixture)
        self.variants = VARIANTS + (("joint_all",) if self.include_missing_mixture else ())
        self.snaps = Snapshots(queries, history, labels, conditions, k_back=k_back)
        self.tables = ChangeTables(self.snaps, **{k: self.config[k] for k in ("d_min", "d_max", "d_step", "min_pre")})
        self.stats = machine_stats(self.snaps)
        self.lpd = log_offset_prior(self.config["no_change_prior"], len(self.tables.D))
        _, self.group = np.unique(
            [f"{e}|{t}" for e, t in zip(self.snaps.episode, self.snaps.tick)], return_inverse=True)
        self.cache = {}
        self._precompute_size_counts()

    def _precompute_size_counts(self):
        s = self.snaps
        self.all_groups = np.zeros((len(s.rows), 8, 2, 4))
        self.common_groups = np.zeros_like(self.all_groups)
        self.current = np.full((len(s.rows), 8), -1, dtype=np.int8)
        by_machine = {}
        for i, row in enumerate(s.rows):
            by_machine.setdefault(row[0], []).append(i)
        self.fit_groups = {}
        for qid, (q, payouts) in s.machines.items():
            lag = first_lag(payouts, q.cutoff)
            self.fit_groups[qid] = (TYPES.index(q.machine_type),
                                   None if lag is None else groups_before(payouts, q.cutoff, lag))
            # The common set is exactly training_events, with wins selected.
            common_ticks = {u for u, win, _ in s.events[qid] if win}
            observed = sorted(payouts)
            ag, cg = np.zeros((8, 2, 4)), np.zeros((8, 2, 4))
            position = 0
            for i in sorted(by_machine.get(qid, []), key=lambda x: s.tick[x]):
                tick = s.tick[i]
                while position < len(observed) and observed[position] < tick:
                    u = observed[position]
                    value = payouts[u]
                    if value > 0:
                        j = SIZE_INDEX[value]
                        for a in range(1, 8):
                            if u - a in payouts:
                                cell = int(payouts[u - a] >= 2)
                                ag[a, cell, j] += 1
                                if u in common_ticks:
                                    cg[a, cell, j] += 1
                    position += 1
                self.all_groups[i], self.common_groups[i] = ag, cg
                for a in range(1, 8):
                    if tick - a in payouts:
                        self.current[i, a] = int(payouts[tick - a] >= 2)
        # Comparing evidence on different response sets would bias rule weights.
        totals = self.common_groups[:, 1:].sum(axis=(2, 3))
        if not np.array_equal(totals, np.broadcast_to(totals[:, :1], totals.shape)):
            raise AssertionError("Common size evidence must use identical events for every lag.")

    def _eb_prior(self, lm, mask):
        c = self.config
        per_h = logsumexp(self.lpd[None, :, None] + c["beta"] * lm[mask & self.snaps.real], axis=1)
        prior = np.full(N_RULES, 1.0 / N_RULES)
        for _ in range(c["eb_iterations"]):
            scores = expand_prior(np.log(prior))[None, :] + per_h
            post = np.exp(scores - logsumexp(scores, axis=1, keepdims=True))
            mass = c["prior_smoothing"] + np.array([post[:, RULE_OF == r].sum() for r in range(N_RULES)])
            prior = mass / mass.sum()
        return np.log(prior)

    def _tables(self, mu, levels):
        s, c = self.snaps, self.config
        return oriented_tables(self.tables, s, mu[s.mtype], c["alpha"],
                               levels[s.mtype], c["orient_strength"])

    def _offset_posterior(self, log_marginal):
        group, lpd, sigma = self.group, self.lpd, self.config["shared_prior"]
        indep = lpd[None, :] + log_marginal
        n_group = group.max() + 1
        sums = np.zeros((n_group, log_marginal.shape[1]))
        np.add.at(sums, group, log_marginal)
        shared = lpd[None, :] + sums
        if sigma <= 0:
            p_shared = np.zeros(n_group)
        elif sigma >= 1:
            p_shared = np.ones(n_group)
        else:
            log_zs = np.log(sigma) + logsumexp(shared, axis=1)
            log_zi = np.log(1 - sigma) + np.bincount(
                group, weights=logsumexp(indep, axis=1), minlength=n_group)
            p_shared = expit(log_zs - log_zi)
        p_shared[np.bincount(group, minlength=n_group) <= 1] = 0
        return (p_shared[group, None] * np.exp(shared - logsumexp(shared, axis=1, keepdims=True))[group]
                + (1 - p_shared[group, None]) * np.exp(indep - logsumexp(indep, axis=1, keepdims=True)))

    def posterior_summaries(self, lm, pr, log_prior, evidence=None, chunk=256):
        """Return w, rule masses, and joint P(win, first lag) without full posterior storage."""
        n = len(self.snaps.rows)
        lp = expand_prior(log_prior)
        log_marginal = np.empty(lm.shape[:2])

        def scores(lo, hi):
            out = self.config["beta"] * lm[lo:hi] + lp[None, None, :]
            if evidence is not None:
                out += evidence[lo:hi, FIRST_OF_H][:, None, :]
            return out

        for lo in range(0, n, chunk):
            hi = min(n, lo + chunk)
            log_marginal[lo:hi] = logsumexp(scores(lo, hi), axis=2)
        offsets = self._offset_posterior(log_marginal)
        masses, win_masses = np.zeros((n, 8)), np.zeros((n, 8))
        for lo in range(0, n, chunk):
            hi = min(n, lo + chunk)
            post = scores(lo, hi)
            post -= log_marginal[lo:hi, :, None]
            np.exp(post, out=post)
            post *= offsets[lo:hi, :, None]
            marginal = post.sum(axis=1)
            joint = (post * pr[lo:hi]).sum(axis=1)
            for a in range(8):
                masses[lo:hi, a] = marginal[:, FIRST_OF_H == a].sum(axis=1)
                win_masses[lo:hi, a] = joint[:, FIRST_OF_H == a].sum(axis=1)
        if not np.allclose(masses.sum(axis=1), 1, rtol=0, atol=2e-11):
            raise AssertionError("Posterior hypothesis mass is not normalized.")
        return win_masses.sum(axis=1), masses, win_masses

    def _size_terms(self, components):
        s = self.snaps
        all_pa, common_pa = np.full((len(s.rows), 8), 0.5), np.full((len(s.rows), 8), 0.5)
        evidence = np.zeros_like(common_pa)
        completed_evidence = np.zeros_like(common_pa) if self.include_missing_mixture else None
        a_row, b_row = np.empty((len(s.rows), 4)), np.empty((len(s.rows), 4))
        for ti, (a, b) in components.items():
            m = s.mtype == ti
            la, lb = np.log(a), np.log(b)
            a_row[m], b_row[m] = a, b
            for groups, output in ((self.all_groups, all_pa), (self.common_groups, common_pa)):
                g = groups[m]
                orientation = expit((g[:, :, 1] - g[:, :, 0]) @ (la - lb))
                current = self.current[m]
                output[m] = np.where(current < 0, 0.5, np.where(current == 1, orientation, 1 - orientation))
                output[m, 0] = 0.5
            g = self.common_groups[m]
            l1 = g[:, :, 1] @ la + g[:, :, 0] @ lb
            l0 = g[:, :, 0] @ la + g[:, :, 1] @ lb
            evidence[m] = np.logaddexp(l1, l0) - np.log(2.0)
            # Null and single-lag hypotheses explain the same response set.
            evidence[m, 0] = g[:, 1].sum(axis=1) @ np.log((a + b) / 2)
            if self.include_missing_mixture:
                # Keep every observed winning response. If its size-driving lag
                # is missing, approximate its latent big/small probability by .5.
                # The missing term is orientation-independent, so all_pa stays
                # exactly the EXP-5 posterior from the available-lag responses.
                g = self.all_groups[m]
                missing = s.size_counts[m, None, :] - g.sum(axis=2)
                if (missing < 0).any():
                    raise AssertionError("Known-lag responses cannot exceed all observed wins.")
                l1 = g[:, :, 1] @ la + g[:, :, 0] @ lb
                l0 = g[:, :, 0] @ la + g[:, :, 1] @ lb
                mix_log = np.log((a + b) / 2)
                completed_evidence[m] = np.logaddexp(l1, l0) - np.log(2.0) + missing @ mix_log
                completed_evidence[m, 0] = s.size_counts[m] @ mix_log
        return all_pa, common_pa, evidence, completed_evidence, a_row, b_row

    def _predict_tables(self, lm, pr, log_prior, components):
        all_pa, common_pa, evidence, completed_evidence, a, b = self._size_terms(components)
        w, masses, win_masses = self.posterior_summaries(lm, pr, log_prior)
        w_joint, _, win_masses_joint = self.posterior_summaries(lm, pr, log_prior, evidence)

        def size(weights, orientation):
            p_a = (weights * orientation).sum(axis=1)
            return p_a[:, None] * a + (1 - p_a[:, None]) * b

        # Conditional size must weight each hypothesis by its chance of a win.
        conditional = win_masses / w[:, None]
        conditional_joint = win_masses_joint / w_joint[:, None]
        result = {
            "exp5": (w, size(masses, all_pa)),
            "joint_mix": (w, size(conditional, all_pa)),
            "common_mix": (w, size(conditional, common_pa)),
            "joint_size": (w_joint, size(conditional_joint, common_pa)),
        }
        if self.include_missing_mixture:
            w_all, _, win_masses_all = self.posterior_summaries(lm, pr, log_prior, completed_evidence)
            result["joint_all"] = (w_all, size(win_masses_all / w_all[:, None], all_pa))
        for name, prediction in result.items():
            p = self.probabilities(prediction)
            if not (np.isfinite(p).all() and (p >= 0).all()
                    and np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-12)):
                raise AssertionError(f"Invalid probability vector in {name}.")
        return result

    def fit_predict(self, train_mask):
        """Fit EXP-5 globals without held-out episodes, then evaluate all variants."""
        if self.config.get("contrast_threshold", 0.5) != 0.5:
            raise ValueError("The reused EXP-5 hi_lo_levels estimator fixes contrast_threshold=0.5.")
        train_mask = np.asarray(train_mask, dtype=bool)
        if train_mask.shape != (len(self.snaps.rows),):
            raise ValueError("Training mask has the wrong shape.")
        for episode in np.unique(self.snaps.episode):
            values = train_mask[self.snaps.episode == episode]
            if values.any() and not values.all():
                raise ValueError("Training masks must keep entire episodes together.")
        key = hashlib.sha256(np.packbits(train_mask).tobytes()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        qids = sorted(set(self.snaps.qid[train_mask]))
        mu, _ = type_priors(self.stats, qids)
        levels = hi_lo_levels(self.snaps, qids)
        lm, pr = self._tables(mu, levels)
        log_prior = self._eb_prior(lm, train_mask)
        components = {}
        for ti in range(3):
            static = em_sizes(np.array([self.stats[x][3] for x in qids if self.stats[x][0] == ti], float),
                              self.config["em_iterations"])
            groups = np.array([self.fit_groups[x][1] for x in qids
                               if self.fit_groups[x][0] == ti and self.fit_groups[x][1] is not None])
            components[ti] = em_cells(groups, static[0], static[1], self.config["size_em_iterations"])
        self.cache[key] = self._predict_tables(lm, pr, log_prior, components)
        return self.cache[key]

    def predict_model(self, model):
        """Prediction-only equivalence API; model is a saved EXP-5-family dict."""
        mu = np.array([model["win_rate"][t] for t in TYPES])
        levels = np.array([model["levels"][t] for t in TYPES])
        components = {i: tuple(np.asarray(x) for x in model["rule_components"][t]) for i, t in enumerate(TYPES)}
        lm, pr = self._tables(mu, levels)
        return self._predict_tables(lm, pr, np.asarray(model["log_prior"]), components)

    def probabilities(self, prediction, theta=None):
        w, size = prediction
        if theta is not None:
            w = calibrated_win(w, theta, self.snaps.mtype)
        return five(w, size)
