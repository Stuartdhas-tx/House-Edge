"""Hand-built numeric tree checks; runtime under test is standard library only."""

import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from house_edge.xgb_portable import PortableXGBoost, float32  # noqa: E402


def tree(index, split=False):
    n = 3 if split else 1
    return {"id": index, "tree_param": {"num_nodes": str(n), "num_deleted": "0", "num_feature": "2", "size_leaf_vector": "1"},
            "left_children": [1, -1, -1] if split else [-1],
            "right_children": [2, -1, -1] if split else [-1],
            "split_indices": [0] * n,
            "split_conditions": [1.0, .25, -.5] if split else [0.0],
            "default_left": [1] * n, "split_type": [0] * n,
            "base_weights": [99.0] * n, "parents": [2147483647, 0, 0] if split else [2147483647],
            "categories": [], "categories_nodes": [], "categories_segments": [], "categories_sizes": []}


def model(with_trees=True):
    trees = [tree(i, split=i == 0) for i in range(5)] if with_trees else []
    return {"version": [3, 4, 1], "learner": {
        "learner_model_param": {"num_class": "5", "num_target": "1", "num_feature": "2", "base_score": "[9,8,7,6,5]"},
        "objective": {"name": "multi:softprob", "softmax_multiclass_param": {"num_class": "5"}},
        "feature_names": ["a", "b"], "feature_types": [],
        "gradient_booster": {"name": "gbtree", "model": {
            "gbtree_model_param": {"num_parallel_tree": "1", "num_trees": str(len(trees))},
            "trees": trees, "tree_info": list(range(5)) if with_trees else [],
            "iteration_indptr": [0, 5] if with_trees else [0],
            "cats": {"enc": [], "feature_segments": [], "sorted_idx": []}}}}}


class PortableTests(unittest.TestCase):
    def test_numeric_missing_default_and_exact_threshold(self):
        predictor = PortableXGBoost(model())
        self.assertEqual(predictor.leaf_indices_row([.9, 0])[0], 1)
        self.assertEqual(predictor.leaf_indices_row([1., 0])[0], 2)
        self.assertEqual(predictor.leaf_indices_row([math.nan, 0])[0], 1)
        self.assertEqual(predictor.leaf_indices_row([None, 0])[0], 1)
        raw = model()
        raw["learner"]["gradient_booster"]["model"]["trees"][0]["default_left"][0] = 0
        self.assertEqual(PortableXGBoost(raw).leaf_indices_row([math.nan, 0])[0], 2)

    def test_features_round_to_float32_before_branching(self):
        value = 1 - 2 ** -26
        self.assertLess(value, 1)
        self.assertEqual(float32(value), 1)
        self.assertEqual(PortableXGBoost(model()).leaf_indices_row([value, 0])[0], 2)

    def test_leaf_split_conditions_override_base_weights(self):
        predictor = PortableXGBoost(model())
        base = [.2] * 5
        margins = predictor.predict_margin_row([0, 0], base)
        self.assertEqual(margins[0], float32(float32(math.log(.2)) + .25))
        self.assertEqual(margins[1], float32(math.log(.2)))
        probabilities = predictor.predict_row([0, 0], base)
        self.assertGreater(probabilities[0], probabilities[1])
        self.assertAlmostEqual(sum(probabilities), 1, places=15)

    def test_explicit_margin_identity_ignores_saved_base_score(self):
        predictor = PortableXGBoost(model(False))
        base = [.7, .1, .1, .08, .02]
        for actual, expected in zip(predictor.predict_row([0, 0], base), base):
            self.assertLess(abs(actual - expected), 1e-7)
        clipped = predictor.predict_row([0, 0], [1, 0, 0, 0, 0])
        self.assertTrue(all(math.isfinite(p) and p > 0 for p in clipped))
        self.assertAlmostEqual(sum(clipped), 1, places=15)

    def test_each_leaf_addition_rounds_to_float32(self):
        raw = model()
        forest = raw["learner"]["gradient_booster"]["model"]
        forest["trees"] = [tree(i) for i in range(20)]
        forest["tree_info"] = list(range(5)) * 4
        forest["iteration_indptr"] = [0, 5, 10, 15, 20]
        forest["gbtree_model_param"]["num_trees"] = "20"
        for i in (0, 5, 10, 15):
            forest["trees"][i]["split_conditions"][0] = 2 ** -25
        base_margin = float32(math.log(.2))
        predicted = PortableXGBoost(raw).predict_margin_row([0, 0], [.2] * 5)
        self.assertEqual(predicted[0], base_margin)
        self.assertNotEqual(float32(base_margin + 4 * 2 ** -25), base_margin)

    def test_file_loading_and_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            path.write_text(json.dumps(model()))
            predictor = PortableXGBoost.load(path)
            base = [.2] * 5
            expected = [predictor.predict_row(row, base) for row in ([0, 0], [2, 0])]
            self.assertEqual(predictor.predict([[0, 0], [2, 0]], [base, base]), expected)

    def test_input_rejection(self):
        predictor = PortableXGBoost(model())
        for features, base in [([0], [.2] * 5), ([math.inf, 0], [.2] * 5), ([0, 0], [.2] * 4),
                               ([0, 0], [0] * 5), ([0, 0], [math.nan, .25, .25, .25, .25])]:
            with self.subTest(features=features, base=base), self.assertRaises(ValueError):
                predictor.predict_row(features, base)
        with self.assertRaises(ValueError):
            predictor.predict([[0, 0]], [])

    def test_malformed_and_unsupported_models(self):
        mutators = [
            lambda m: m["learner"]["objective"].update(name="multi:softmax"),
            lambda m: m["learner"]["learner_model_param"].update(num_class="4"),
            lambda m: m["learner"]["gradient_booster"].update(name="dart"),
            lambda m: m["learner"].update(feature_types=["c", "float"]),
            lambda m: m["learner"]["gradient_booster"]["model"]["cats"].update(enc=[1]),
            lambda m: m["learner"]["gradient_booster"]["model"]["tree_info"].__setitem__(0, 5),
            lambda m: m["learner"]["gradient_booster"]["model"]["trees"][0]["tree_param"].update(size_leaf_vector="5"),
            lambda m: m["learner"]["gradient_booster"]["model"]["trees"][0]["split_type"].__setitem__(0, 1),
            lambda m: m["learner"]["gradient_booster"]["model"]["trees"][0]["left_children"].__setitem__(0, 0),
            lambda m: m["learner"]["gradient_booster"]["model"]["trees"][0]["right_children"].__setitem__(0, 99),
            lambda m: m["learner"]["gradient_booster"]["model"]["trees"][0]["split_conditions"].pop(),
            lambda m: m["learner"]["gradient_booster"]["model"]["trees"][0]["split_conditions"].__setitem__(0, math.nan),
            lambda m: m["learner"]["gradient_booster"]["model"]["trees"][0]["default_left"].__setitem__(0, 2),
        ]
        for index, mutate in enumerate(mutators):
            raw = copy.deepcopy(model())
            mutate(raw)
            with self.subTest(index=index), self.assertRaises(ValueError):
                PortableXGBoost(raw)
        with self.assertRaises(ValueError):
            PortableXGBoost({})


if __name__ == "__main__":
    unittest.main()
