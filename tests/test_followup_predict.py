"""Synthetic frozen-postprocessor checks; no training or validation targets."""

from dataclasses import replace
import math
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from house_edge.followup_predict import PROBABILITY_NAMES, TYPES, predict_followup
from house_edge.xgb_features_portable import build_row, feature_names
from house_edge.xgb_portable import PortableXGBoost, float32
from slot_forecasting.schema import Play, Query, ValidationError


RAW = [.6, .04, .08, .12, .16]
IDENTITY = [1.0] + [0.0] * 12


class FakeBase:
    def __init__(self, predictions=None):
        self.predictions = predictions
        self.calibration_slope = 1.0
        self.calibration_intercepts = dict.fromkeys(TYPES, 0.0)
        self.calls = []

    def predict(self, queries, history):
        self.calls.append((queries, history))
        return ({q.query_id: list(RAW) for q in queries}
                if self.predictions is None else self.predictions)


def toy_trees(names, split_feature):
    """Five hand-built class trees, with one split affecting loss probability."""
    trees = []
    for output in range(5):
        split = output == 0
        n = 3 if split else 1
        trees.append({
            "id": output,
            "tree_param": {"num_nodes": str(n), "num_deleted": "0", "num_feature": str(len(names)),
                           "size_leaf_vector": "1"},
            "left_children": [1, -1, -1] if split else [-1],
            "right_children": [2, -1, -1] if split else [-1],
            "split_indices": [split_feature] * n,
            "split_conditions": [.5, .75, -.5] if split else [0.0],
            "default_left": [1] * n, "split_type": [0] * n,
            "base_weights": [0.0] * n,
            "parents": [2147483647, 0, 0] if split else [2147483647],
        })
    return {"learner": {
        "learner_model_param": {"num_class": "5", "num_target": "1", "num_feature": str(len(names))},
        "objective": {"name": "multi:softprob", "softmax_multiclass_param": {"num_class": "5"}},
        "feature_names": names,
        "gradient_booster": {"name": "gbtree", "model": {
            "gbtree_model_param": {"num_parallel_tree": "1", "num_trees": "5"},
            "trees": trees, "tree_info": list(range(5)), "iteration_indptr": [0, 5]}}}}


class RecordingTrees:
    def __init__(self, kind="compact_xgb", split_feature=1):
        names = [f"type_{t}" for t in TYPES] if kind == "compact_xgb" else feature_names()
        names = names + list(PROBABILITY_NAMES)
        self.predictor = PortableXGBoost(toy_trees(names, split_feature))
        self.num_features = self.predictor.num_features
        self.feature_names = self.predictor.feature_names
        self.calls = []

    def predict_row(self, features, base_prob):
        self.calls.append((list(features), list(base_prob)))
        return self.predictor.predict_row(features, base_prob)


def queries():
    return [Query(f"q{i}", "episode", f"m{i}", kind, i - 1, 16) for i, kind in enumerate(TYPES)]


def sigmoid_odds(win, slope, intercept):
    odds = (win / (1 - win)) ** slope * math.exp(intercept)
    return odds / (1 + odds)


class FollowupTests(unittest.TestCase):
    def assert_probabilities(self, result):
        for row in result.values():
            self.assertIsInstance(row, list)
            self.assertEqual(len(row), 5)
            self.assertTrue(all(math.isfinite(p) and 0 <= p <= 1 for p in row))
            self.assertAlmostEqual(math.fsum(row), 1.0, places=15)

    def assert_rows_equal(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            if math.isnan(b):
                self.assertTrue(math.isnan(a))
            else:
                self.assertEqual(a, b)

    def test_type_size_map_matches_analytical_odds_and_reference_offsets(self):
        theta = [2.0, math.log(3), math.log(.5), math.log(2)]
        factors = ([2, 3, 4, 1], [1, .5, 2, 1], [3, 1, .25, 1])
        theta += [math.log(factor) for row in factors for factor in row[:3]]
        output = predict_followup(FakeBase(), {"kind": "type_size_cal", "theta": theta}, queries(), [])
        for i, query in enumerate(queries()):
            win = sigmoid_odds(.4, theta[0], theta[1 + i])
            sizes = [a * b for a, b in zip([.1, .2, .3, .4], factors[i])]
            expected = [1 - win] + [win * s / sum(sizes) for s in sizes]
            for actual, wanted in zip(output[query.query_id], expected):
                self.assertAlmostEqual(actual, wanted, places=15)
        self.assert_probabilities(output)

    def test_identity_with_generators_predicts_base_batch_once(self):
        base = FakeBase()
        # Alternate forecasters may not expose calibration attributes.
        del base.calibration_slope, base.calibration_intercepts
        q = queries()
        history = [Play("episode", "m0", 0, 1)]
        output = predict_followup(base, {"kind": "type_size_cal", "theta": IDENTITY},
                                  (row for row in q), (row for row in history))
        self.assertEqual(base.calls, [(tuple(q), tuple(history))])
        for row in output.values():
            for actual, expected in zip(row, RAW):
                self.assertAlmostEqual(actual, expected, places=15)
        self.assert_probabilities(output)

    def test_compact_tree_receives_type_indicators_and_float32_raw_features(self):
        q = queries()
        raw = [.6123456789, .0123456789, .0753086422, .123456789, .176543211]
        self.assertAlmostEqual(sum(raw), 1.0)
        base = FakeBase({query.query_id: raw for query in q})
        booster = RecordingTrees()
        result = predict_followup(base, {"kind": "compact_xgb"}, q, [], booster)
        self.assertEqual(len(base.calls), 1)
        self.assertEqual(len(booster.calls), len(q))
        for i, ((row, margins), query) in enumerate(zip(booster.calls, q)):
            self.assertEqual(row, [float(i == j) for j in range(3)] + [float32(p) for p in raw])
            self.assertEqual(margins, raw)
            self.assertEqual(result[query.query_id], booster.predictor.predict_row(row, raw))
        self.assertGreater(result["q0"][0], result["q1"][0])
        self.assertEqual(result["q0"], result["q2"])
        self.assert_probabilities(result)

    def test_full_features_scope_peers_by_episode_and_current_cutoff(self):
        names = feature_names()
        q = [Query("a", "e", "self", "balanced", 1, 8),
             Query("b", "e", "peer", "jackpot", -1, 8),
             Query("c", "other", "peer", "steady", 0, 16)]
        own = {0: 1, 2: 20, 6: 2}
        peer = {t: 0 for t in range(8)}
        remote = {t: 20 for t in range(16)}
        histories = [own, peer, remote]
        history = [Play(query.episode_id, query.machine_id, t, value)
                   for query, payouts in zip(q, histories) for t, value in payouts.items()]
        booster = RecordingTrees("full_xgb", names.index("peer_all_win_rate"))
        result = predict_followup(FakeBase(), {"kind": "full_xgb"}, q, history, booster)
        first_row = booster.calls[0][0]
        self.assert_rows_equal(first_row, build_row(q[0], own, [peer]) + [float32(p) for p in RAW])
        self.assertEqual(first_row[names.index("peer_all_machine_n")], 1.0)
        self.assertEqual(first_row[names.index("peer_all_n")], 8.0)
        self.assertEqual(first_row[names.index("peer_all_win_rate")], 0.0)
        self.assertEqual(first_row[names.index("lag2_payout")], 2.0)
        self.assertEqual(booster.calls[2][0][names.index("peer_all_machine_n")], 0.0)
        # Remote-episode observations, including its legal later ticks, cannot
        # enter a's peer summary despite a reused machine identifier.
        changed = [replace(play, payout=0) if play.episode_id == "other"
                   else play for play in history]
        other_booster = RecordingTrees("full_xgb", names.index("peer_all_win_rate"))
        other = predict_followup(FakeBase(), {"kind": "full_xgb"}, q, changed, other_booster)
        self.assert_rows_equal(other_booster.calls[0][0], first_row)
        self.assertEqual(other["a"], result["a"])
        self.assert_probabilities(result)

    def test_mixed_cutoffs_within_episode_are_rejected_before_base_prediction(self):
        q = queries()
        q[1] = replace(q[1], cutoff=q[1].cutoff + 1)
        base = FakeBase()
        with self.assertRaisesRegex(ValueError, "shared cutoff"):
            predict_followup(base, {"kind": "type_size_cal", "theta": IDENTITY}, q, [])
        self.assertEqual(base.calls, [])

    def test_half_full_is_exact_half_of_trees_and_four_parameter_exp5_map(self):
        q = queries()
        theta = [1.7, -.25, .15, .45]
        booster = RecordingTrees("full_xgb")
        result = predict_followup(FakeBase(), {"kind": "half_full", "baseline_theta": theta}, q, [], booster)
        for i, query in enumerate(q):
            row, raw = booster.calls[i]
            tree = booster.predictor.predict_row(row, raw)
            win = sigmoid_odds(.4, theta[0], theta[1 + i])
            baseline = [1 - win] + [win * p for p in [.1, .2, .3, .4]]
            expected = [.5 * a + .5 * b for a, b in zip(tree, baseline)]
            for actual, wanted in zip(result[query.query_id], expected):
                self.assertAlmostEqual(actual, wanted, places=15)
        self.assert_probabilities(result)

    def test_calibration_extremes_stay_finite_and_baseline_keeps_exp5_clamp(self):
        q = queries()
        theta = [1.0, 1000.0, -1000.0, 1000.0] + [1000.0, -1000.0, 0.0] * 3
        result = predict_followup(FakeBase(), {"kind": "type_size_cal", "theta": theta}, q, [])
        self.assertEqual(result["q0"], [0.0, 1.0, 0.0, 0.0, 0.0])
        self.assertEqual(result["q1"], [1.0, 0.0, 0.0, 0.0, 0.0])
        self.assert_probabilities(result)
        booster = RecordingTrees("half_full")
        half = predict_followup(FakeBase(), {"kind": "half_full", "baseline_theta": theta[:4]}, q, [], booster)
        for i, query in enumerate(q):
            tree = booster.predictor.predict_row(*booster.calls[i])
            expected_loss = 1e-6 if i != 1 else 1 - 1e-6
            self.assertAlmostEqual(half[query.query_id][0], .5 * (tree[0] + expected_loss), places=15)

    def test_every_kind_rejects_exposed_future_on_self_or_any_peer(self):
        q = queries()
        configs = [{"kind": "type_size_cal", "theta": IDENTITY}, {"kind": "compact_xgb"},
                   {"kind": "full_xgb"}, {"kind": "half_full", "baseline_theta": IDENTITY[:4]}]
        for config in configs:
            for machine in q:
                for tick in (machine.cutoff, machine.cutoff + 3):
                    base = FakeBase()
                    booster = None if config["kind"] == "type_size_cal" else RecordingTrees(config["kind"])
                    with self.subTest(kind=config["kind"], machine=machine.machine_id, tick=tick):
                        with self.assertRaisesRegex(ValidationError, "Future data leak"):
                            predict_followup(base, config, q,
                                             [Play(machine.episode_id, machine.machine_id, tick, 20)], booster)
                        self.assertEqual(base.calls, [])

    def test_identity_guard_rejects_double_calibration_without_mutating_base(self):
        for attr, value in [("calibration_slope", 1.01), ("calibration_slope", math.nan),
                            ("calibration_intercepts", {**dict.fromkeys(TYPES, 0), "steady": .01}),
                            ("calibration_intercepts", {"balanced": 0}), ("calibration_intercepts", None)]:
            base = FakeBase()
            setattr(base, attr, value)
            with self.subTest(attribute=attr, value=value), self.assertRaisesRegex(ValueError, "base calibration"):
                predict_followup(base, {"kind": "type_size_cal", "theta": IDENTITY}, queries(), [])
            self.assertEqual(base.calls, [])
            self.assertIs(getattr(base, attr), value)

    def test_bad_config_or_booster_schema_fails_before_prediction(self):
        configs = [{}, {"kind": "unknown"}, {"kind": "type_size_cal"},
                   {"kind": "type_size_cal", "theta": IDENTITY[:-1]},
                   {"kind": "type_size_cal", "theta": [math.nan] + IDENTITY[1:]},
                   {"kind": "type_size_cal", "theta": [True] + IDENTITY[1:]},
                   {"kind": "half_full", "baseline_theta": [1, 0, 0]}, {"kind": "full_xgb"}]
        for config in configs:
            base = FakeBase()
            with self.subTest(config=config), self.assertRaises(ValueError):
                predict_followup(base, config, queries(), [])
            self.assertEqual(base.calls, [])
        for mismatch in ("count", "order"):
            base, booster = FakeBase(), RecordingTrees()
            if mismatch == "count":
                booster.num_features += 1
            else:
                booster.feature_names = tuple(reversed(booster.feature_names))
            with self.subTest(mismatch=mismatch), self.assertRaises(ValueError):
                predict_followup(base, {"kind": "compact_xgb"}, queries(), [], booster)
            self.assertEqual(base.calls, [])

    def test_bad_base_predictions_and_unknown_types_are_rejected(self):
        q = queries()[:1]
        invalid = [{}, {"q0": RAW, "extra": RAW}, {"q0": [0] * 5}, {"q0": [math.nan] + RAW[1:]},
                   {"q0": [-.1, .2, .3, .3, .3]}, {"q0": [.25] * 4}, {"q0": [1, 0, 0, 0, 0]}]
        for rows in invalid:
            with self.subTest(predictions=rows), self.assertRaises(ValueError):
                predict_followup(FakeBase(rows), {"kind": "type_size_cal", "theta": IDENTITY}, q, [])
        base = FakeBase()
        with self.assertRaisesRegex(ValueError, "Unknown machine type"):
            predict_followup(base, {"kind": "type_size_cal", "theta": IDENTITY},
                             [replace(q[0], machine_type="unknown")], [])
        self.assertEqual(base.calls, [])

    def test_runtime_import_and_prediction_work_without_site_packages(self):
        code = """
from house_edge.followup_predict import predict_followup
from slot_forecasting.schema import Query
class Base:
    def predict(self, queries, history):
        return {q.query_id: [.6, .04, .08, .12, .16] for q in queries}
q = Query('q', 'e', 'm', 'balanced', 0, 0)
out = predict_followup(Base(), {'kind': 'type_size_cal', 'theta': [1]+[0]*12}, [q], [])
assert abs(sum(out['q']) - 1) < 1e-15
import sys
assert not any(m.split('.')[0] in {'numpy', 'scipy', 'xgboost'} for m in sys.modules)
"""
        run = subprocess.run([sys.executable, "-S", "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)


if __name__ == "__main__":
    unittest.main()
