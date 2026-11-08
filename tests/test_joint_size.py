"""Independent probability and leakage checks for joint rule/size inference.

Run: python tests/test_joint_size.py
Only synthetic histories and the saved training-fitted EXP-5 parameters are used.
"""

import itertools
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from house_edge.joint_size import JointSizeForecaster, size_evidence
from house_edge.registry import load_model
from house_edge.size_rule import SizeRuleForecaster
from slot_forecasting.schema import Play, Query, ValidationError


class TwoRuleForecaster(JointSizeForecaster):
    """Two known hypotheses with first lags 1 and 2, for exact enumeration."""

    toy_prior = (0.4, 0.6)

    def hypothesis_log_prior(self):
        return [math.log(p) for p in self.toy_prior]

    @staticmethod
    def rule_of(h):
        # Rule 9 is pair (1, 2); rule 16 is pair (2, 3).
        return (9, 16)[h]


def enumerate_posterior(model, tables, size_tables):
    """Enumerate all latent states in ordinary probability space.

    Unlike production inference, this does not first marginalize rules or
    factor shared/independent evidence. Tiny toy inputs avoid underflow.
    """
    machine_count = len(tables)
    offset_prior = [model.no_change_prior, 1.0 - model.no_change_prior]
    rule_prior = model.toy_prior
    total = shared_total = 0.0
    win_totals = [0.0] * machine_count
    size_totals = [[0.0] * 4 for _ in tables]
    for shared, mixture_prior in ((False, 1.0 - model.shared_prior),
                                  (True, model.shared_prior)):
        for offsets in itertools.product(range(2), repeat=machine_count):
            if shared and len(set(offsets)) != 1:
                continue
            offset_weight = (offset_prior[offsets[0]] if shared else
                             math.prod(offset_prior[d] for d in offsets))
            for hypotheses in itertools.product(range(2), repeat=machine_count):
                weight = mixture_prior * offset_weight
                for m, (d, h) in enumerate(zip(offsets, hypotheses)):
                    weight *= rule_prior[h] * math.exp(tables[m][0][d][h]) ** model.beta
                    if model.joint_size_evidence:
                        weight *= math.exp(size_tables[m][0][h + 1])
                total += weight
                shared_total += weight if shared else 0.0
                for m, (d, h) in enumerate(zip(offsets, hypotheses)):
                    win = tables[m][1][d][h]
                    win_totals[m] += weight * win
                    for j, size in enumerate(size_tables[m][1][h + 1]):
                        size_totals[m][j] += weight * size * (win if model.joint_average else 1.0)
    wins = [value / total for value in win_totals]
    sizes = [[value / (win_totals[m] if model.joint_average else total) for value in row]
             for m, row in enumerate(size_totals)]
    return wins, sizes, shared_total / total


def toy_size_table(evidence, first, second):
    return ([0.0, *map(math.log, evidence), *([0.0] * 5)],
            [[0.25] * 4, list(first), list(second), *([[0.25] * 4] * 5)])


def enumerate_missing_cells(payouts, tick, A, B, lag):
    """Brute-force both orientations and every unobserved historical cell."""
    past = {u: v for u, v in payouts.items() if u < tick}
    wins = [(u, (1, 2, 5, 20).index(v)) for u, v in past.items() if v > 0]
    missing = [u for u, _ in wins if u - lag not in past]
    evidence = 0.0
    predictive = [0.0] * 4
    for orientation in (0, 1):
        for bits in itertools.product((0, 1), repeat=len(missing)):
            hidden_cells = dict(zip(missing, bits))
            weight = 0.5 ** (1 + len(missing))
            for u, category in wins:
                cell = int(past[u - lag] >= 2) if u - lag in past else hidden_cells[u]
                weight *= (A if cell == orientation else B)[category]
            evidence += weight
            if tick - lag in past:
                cell = int(past[tick - lag] >= 2)
                size = A if cell == orientation else B
            else:
                size = [(a + b) / 2 for a, b in zip(A, B)]
            for j, value in enumerate(size):
                predictive[j] += weight * value
    return evidence, [value / evidence for value in predictive]


def fitted_fixture(**options):
    """Use existing fitted parameters, with only three offsets for fast tests."""
    data = json.loads((ROOT / "runs/exp5/model.json").read_text())
    data.update(model="joint_size", d_min=8, d_max=16, d_step=8,
                joint_size_evidence=True, joint_size_common=True, joint_average=True)
    data.update(options)
    return JointSizeForecaster.from_dict(data)


def synthetic_histories():
    rng = random.Random(104729)
    return [{t: rng.choice((0, 0, 0, 1, 2, 5, 20)) for t in range(64)} for _ in range(2)]


class JointSizeTests(unittest.TestCase):
    def assert_vector_close(self, actual, expected, places=12):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, places=places)

    def test_joint_average_retains_covariance_without_size_evidence(self):
        model = TwoRuleForecaster(d_min=8, d_max=8, joint_size_evidence=False)
        model.toy_prior = (0.5, 0.5)
        table = ([[0.0, 0.0]] * 2, [[0.9, 0.1]] * 2)
        size_table = toy_size_table((1.0, 1.0), (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0))
        wins, sizes, shared = model.joint_episode_probabilities([table], [size_table])
        self.assertAlmostEqual(wins[0], 0.5)
        self.assert_vector_close(sizes[0], (0.9, 0.1, 0.0, 0.0))
        self.assert_vector_close([wins[0] * s for s in sizes[0]], (0.45, 0.05, 0.0, 0.0))
        self.assertEqual(shared, 0.0)
        model.joint_average = False
        product_wins, product_sizes, _ = model.joint_episode_probabilities([table], [size_table])
        self.assert_vector_close(product_wins, wins)
        self.assert_vector_close(product_sizes[0], (0.5, 0.5, 0.0, 0.0))

    def test_size_evidence_updates_rule_weights(self):
        model = TwoRuleForecaster(d_min=8, d_max=8, beta=1.7)
        model.toy_prior = (0.5, 0.5)
        table = ([[0.0, 0.0]] * 2, [[0.9, 0.1]] * 2)
        size_table = toy_size_table((0.9, 0.1), (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0))
        wins, sizes, _ = model.joint_episode_probabilities([table], [size_table])
        self.assertAlmostEqual(wins[0], 0.82)
        self.assert_vector_close([wins[0] * s for s in sizes[0]], (0.81, 0.01, 0.0, 0.0))

    def test_shared_and_independent_mixtures_match_brute_force(self):
        tables = [
            ([[math.log(x) for x in row] for row in ((0.8, 0.2), (0.1, 0.9))],
             [[0.9, 0.2], [0.3, 0.8]]),
            ([[math.log(x) for x in row] for row in ((0.3, 0.6), (0.7, 0.1))],
             [[0.4, 0.7], [0.8, 0.15]]),
        ]
        size_tables = [
            toy_size_table((0.7, 0.2), (0.6, 0.2, 0.1, 0.1), (0.1, 0.2, 0.3, 0.4)),
            toy_size_table((0.25, 0.8), (0.2, 0.5, 0.2, 0.1), (0.5, 0.1, 0.1, 0.3)),
        ]
        for shared, evidence, joint in itertools.product((0.0, 0.37, 1.0), (False, True), (False, True)):
            with self.subTest(shared=shared, evidence=evidence, joint=joint):
                model = TwoRuleForecaster(d_min=8, d_max=8, no_change_prior=0.65,
                                          shared_prior=shared, beta=1.3,
                                          joint_size_evidence=evidence, joint_average=joint)
                actual = model.joint_episode_probabilities(tables, size_tables)
                expected = enumerate_posterior(model, tables, size_tables)
                self.assert_vector_close(actual[0], expected[0])
                for a, b in zip(actual[1], expected[1]):
                    self.assert_vector_close(a, b)
                    self.assertAlmostEqual(sum(a), 1.0)
                self.assertAlmostEqual(actual[2], expected[2], places=12)

    def test_common_window_likelihood_matches_two_orientation_calculation(self):
        A, B = (0.6, 0.2, 0.1, 0.1), (0.1, 0.2, 0.3, 0.4)
        payouts = dict.fromkeys(range(12), 0)
        payouts.update({0: 20, 8: 2, 9: 5})
        evidence, _ = size_evidence(payouts, 12, A, B)
        mix = [(a + b) / 2 for a, b in zip(A, B)]
        # Only events 8 and 9 have full windows. Lag 1 places them in
        # different cells; all other candidate first lags put both in cell 0.
        self.assertAlmostEqual(math.exp(evidence[0]), mix[1] * mix[2])
        self.assertAlmostEqual(math.exp(evidence[1]), (B[1] * A[2] + A[1] * B[2]) / 2)
        for lag in range(2, 8):
            self.assertAlmostEqual(math.exp(evidence[lag]), (A[1] * A[2] + B[1] * B[2]) / 2)
        # The isolated winning event at 0 is not size evidence for any rule.
        changed = dict(payouts)
        changed[0] = 0
        self.assertEqual(size_evidence(changed, 12, A, B), size_evidence(payouts, 12, A, B))
        # Removing tick 0 excludes event 8 for *every* rule, leaving event 9.
        del payouts[0]
        evidence, _ = size_evidence(payouts, 12, A, B)
        self.assert_vector_close(evidence, [math.log(mix[2])] * 8)

    def test_missing_current_lag_is_unknown_and_future_is_ignored(self):
        A, B = (0.6, 0.2, 0.1, 0.1), (0.1, 0.2, 0.3, 0.4)
        payouts = dict.fromkeys(range(12), 0)
        payouts.update({8: 2, 9: 5})
        del payouts[10]  # current lag 2 is unknown, not the previous row
        expected = size_evidence(payouts, 12, A, B)
        self.assert_vector_close(expected[1][2], [(a + b) / 2 for a, b in zip(A, B)])
        zeroed = dict(payouts)
        zeroed[10] = 0
        self.assertNotEqual(expected[1][2], size_evidence(zeroed, 12, A, B)[1][2])
        future = dict(payouts)
        future.update({12: 20, 13: 1, 100: 5})
        self.assertEqual(size_evidence(future, 12, A, B), expected)

    def test_no_complete_window_provides_no_likelihood_advantage(self):
        A, B = (0.6, 0.2, 0.1, 0.1), (0.1, 0.2, 0.3, 0.4)
        evidence, sizes = size_evidence({0: 1, 1: 2, 3: 5, 9: 20}, 10, A, B)
        self.assert_vector_close(evidence, [0.0] * 8)
        for size in sizes:
            self.assert_vector_close(size, [(a + b) / 2 for a, b in zip(A, B)])

    def test_missing_mixture_matches_explicit_hidden_cell_enumeration(self):
        A, B = (0.6, 0.2, 0.1, 0.1), (0.1, 0.2, 0.3, 0.4)
        payouts = {0: 1, 1: 2, 3: 0, 4: 5, 6: 20, 7: 1, 10: 2}
        evidence, sizes = size_evidence(payouts, 11, A, B, common=False, complete_missing=True)
        mix = [(a + b) / 2 for a, b in zip(A, B)]
        expected_null = math.prod(mix[(1, 2, 5, 20).index(v)] for v in payouts.values() if v > 0)
        self.assertAlmostEqual(math.exp(evidence[0]), expected_null, places=14)
        for lag in range(1, 8):
            with self.subTest(lag=lag):
                expected_evidence, expected_size = enumerate_missing_cells(payouts, 11, A, B, lag)
                self.assertAlmostEqual(evidence[lag], math.log(expected_evidence), places=12)
                self.assert_vector_close(sizes[lag], expected_size)

    def test_missing_counts_add_neutral_evidence_without_changing_orientation(self):
        A, B = (0.6, 0.2, 0.1, 0.1), (0.1, 0.2, 0.3, 0.4)
        payouts = {0: 1, 1: 2, 3: 0, 4: 5, 6: 20, 7: 1, 10: 2}
        old_evidence, old_sizes = size_evidence(payouts, 11, A, B, common=False)
        evidence, sizes = size_evidence(payouts, 11, A, B, common=False, complete_missing=True)
        self.assertEqual(sizes, old_sizes)
        self.assertEqual(evidence[0], old_evidence[0])
        for lag in range(1, 8):
            correction = math.fsum(math.log((A[j] + B[j]) / 2)
                                   for u, v in payouts.items() if v > 0 and u - lag not in payouts
                                   for j in [(1, 2, 5, 20).index(v)])
            self.assertAlmostEqual(evidence[lag] - old_evidence[lag], correction, places=12)
        future = dict(payouts)
        future.update({11: 20, 12: 1, 99: 2})
        self.assertEqual(size_evidence(future, 11, A, B, common=False, complete_missing=True),
                         (evidence, sizes))

    def test_fully_missing_cells_explain_every_win_without_favoring_a_lag(self):
        A, B = (0.6, 0.2, 0.1, 0.1), (0.1, 0.2, 0.3, 0.4)
        evidence, sizes = size_evidence({0: 1, 20: 5, 40: 20}, 41, A, B,
                                        common=False, complete_missing=True)
        mix = [(a + b) / 2 for a, b in zip(A, B)]
        expected = math.log(mix[0] * mix[2] * mix[3])
        self.assert_vector_close(evidence, [expected] * 8)
        for size in sizes:
            self.assert_vector_close(size, mix)

    def test_identical_size_components_leave_win_and_shared_posterior_unchanged(self):
        tables = [([[-0.2, -0.8], [-0.6, -0.1]], [[0.9, 0.1], [0.2, 0.8]]),
                  ([[-0.4, -0.3], [-0.1, -0.9]], [[0.7, 0.2], [0.4, 0.6]])]
        A = (0.1, 0.2, 0.3, 0.4)
        histories = [{u: v for u, v in p.items() if u % 11 != 3} for p in synthetic_histories()]
        for missing_mixture in (False, True):
            with self.subTest(missing_mixture=missing_mixture):
                model = TwoRuleForecaster(d_min=8, d_max=8, shared_prior=0.7,
                                          joint_size_common=not missing_mixture,
                                          joint_size_missing_mixture=missing_mixture)
                size_tables = [size_evidence(p, 40, A, A, common=not missing_mixture,
                                             complete_missing=missing_mixture) for p in histories]
                wins, sizes, shared = model.joint_episode_probabilities(tables, size_tables)
                old_wins, old_shared = model.episode_win_probabilities(tables)
                self.assert_vector_close(wins, old_wins)
                self.assertAlmostEqual(shared, old_shared, places=12)
                for size in sizes:
                    self.assert_vector_close(size, A)

    def test_joint_episode_path_ignores_future_on_every_machine(self):
        histories = [{u: v for u, v in p.items() if u % 11 != 3} for p in synthetic_histories()]
        tick = 40
        def forecast(model, maps):
            tables = [model.machine_tables(p, tick, t) for p, t in zip(maps, ("balanced", "steady"))]
            size_tables = [model.machine_size_evidence(p, tick, t) for p, t in zip(maps, ("balanced", "steady"))]
            return model.joint_episode_probabilities(tables, size_tables)

        truncated = [{u: v for u, v in p.items() if u < tick} for p in histories]
        noisy = [dict(p) for p in truncated]
        rng = random.Random(271828)
        for p in noisy:
            p.update({u: rng.choice((0, 1, 2, 5, 20)) for u in range(tick, 100)})
        for missing_mixture in (False, True):
            with self.subTest(missing_mixture=missing_mixture):
                model = fitted_fixture(joint_size_common=not missing_mixture,
                                       joint_size_missing_mixture=missing_mixture)
                self.assertEqual(forecast(model, truncated), forecast(model, histories))
                self.assertEqual(forecast(model, truncated), forecast(model, noisy))
                for p, short in zip(histories, truncated):
                    self.assertEqual(model.predict_machine(p, tick, "balanced"),
                                     model.predict_machine(short, tick, "balanced"))

    def test_all_switches_off_reproduce_exp5_control(self):
        model = fitted_fixture(joint_size_evidence=False, joint_size_common=False, joint_average=False)
        original = SizeRuleForecaster.from_dict(dict(model.to_dict(), model="size_rule"))
        for payouts in synthetic_histories():
            self.assert_vector_close(model.predict_machine(payouts, 40, "balanced"),
                                     original.predict_machine(payouts, 40, "balanced"))

    def test_serialization_preserves_predictions_and_flags(self):
        queries = [Query("q0", "episode", "m0", "balanced", 0, 64),
                   Query("q1", "episode", "m1", "steady", 1, 64)]
        history = [Play("episode", f"m{i}", tick, value)
                   for i, payouts in enumerate(synthetic_histories()) for tick, value in payouts.items()]
        for missing_mixture in (False, True):
            with self.subTest(missing_mixture=missing_mixture):
                model = fitted_fixture(joint_size_evidence=missing_mixture, joint_size_common=False,
                                       joint_average=True, joint_size_missing_mixture=missing_mixture)
                expected = model.predict(queries, history)
                with tempfile.TemporaryDirectory(prefix="joint-size-test-") as temporary:
                    path = Path(temporary) / "model.json"
                    model.save(path)
                    restored = load_model(path)
                    self.assertIsInstance(restored, JointSizeForecaster)
                    self.assertEqual(restored.to_dict(), model.to_dict())
                    self.assertEqual(restored.predict(queries, history), expected)
                for probabilities in expected.values():
                    self.assertAlmostEqual(sum(probabilities), 1.0)
                    self.assertTrue(all(0.0 <= p <= 1.0 for p in probabilities))
                with self.assertRaises(ValidationError):
                    model.predict(queries, history + [Play("episode", "m0", 64, 20)])

    def test_pre_followup_checkpoint_defaults_missing_mixture_to_false(self):
        model = fitted_fixture()
        data = model.to_dict()
        del data["joint_size_missing_mixture"]
        restored = JointSizeForecaster.from_dict(data)
        self.assertFalse(restored.joint_size_missing_mixture)
        self.assertEqual(restored.to_dict(), model.to_dict())

    def test_incoherent_or_nonboolean_options_are_rejected(self):
        with self.assertRaises(ValueError):
            JointSizeForecaster(joint_size_common=False, joint_size_evidence=True)
        for name in ("joint_size_evidence", "joint_size_common", "joint_average", "joint_size_missing_mixture"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                JointSizeForecaster(**{name: 1})
        with self.assertRaises(ValueError):
            fitted_fixture(joint_size_common=False, joint_size_evidence=True)
        with self.assertRaises(ValueError):
            JointSizeForecaster(joint_size_common=True, joint_size_missing_mixture=True)
        with self.assertRaises(ValueError):
            size_evidence({}, 10, (0.25,) * 4, (0.25,) * 4, common=True, complete_missing=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
