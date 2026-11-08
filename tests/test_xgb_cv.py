"""Focused checks for correction cross-fitting, weights, and probability margins."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "research")]
from cv_xgb import (BASE_PARAMETERS, fit_predict_booster, nested_exp5_inputs,
                    normalized_training_weights, validate_probabilities)


class CrossFitTests(unittest.TestCase):
    def test_every_correction_training_row_excludes_outer_and_own_fold(self):
        folds = np.repeat(np.arange(5), 3)
        calls = []

        class FakeHarness:
            def fit_predict(self, mask):
                calls.append(mask.copy())
                # Encode training-fold membership so each row's assignment is testable.
                code = sum(1 << int(f) for f in set(folds[mask]))
                return {"exp5": (np.full(len(folds), code, dtype=float),
                                  np.tile([code, 2 * code, 3 * code, 4 * code], (len(folds), 1)))}

        for outer in range(5):
            calls.clear()
            win, size = nested_exp5_inputs(FakeHarness(), folds, outer)
            self.assertEqual(len(calls), 4)
            self.assertTrue(all(not mask[folds == outer].any() for mask in calls))
            for inner in range(5):
                if inner == outer:
                    continue
                expected = sum(1 << f for f in range(5) if f not in (inner, outer))
                np.testing.assert_array_equal(win[folds == inner], expected)
                np.testing.assert_array_equal(size[folds == inner], np.tile(
                    [expected, 2 * expected, 3 * expected, 4 * expected], (3, 1)))

    def test_weights_balance_episodes_and_conditions_with_mean_one(self):
        episodes = np.array(["a"] * 1 + ["b"] * 3 + ["c"] * 2 + ["held"] * 4)
        conditions = np.array(["stable"] * 4 + ["drift"] * 2 + ["short_history"] * 4)
        mask = episodes != "held"
        snaps = SimpleNamespace(episode=episodes, condition=conditions)
        weights = normalized_training_weights(snaps, mask)
        self.assertAlmostEqual(float(weights.mean()), 1.0)
        self.assertAlmostEqual(float(weights[:1].sum()), float(weights[1:4].sum()))
        self.assertAlmostEqual(float(weights[:4].sum()), float(weights[4:].sum()))

    def test_zero_tree_correction_returns_base_probabilities(self):
        x = np.array([[0.0], [1.0], [np.nan], [3.0], [4.0]], dtype=np.float32)
        base = np.array([[.55, .20, .12, .08, .05], [.1, .2, .3, .1, .3],
                         [.2, .2, .2, .2, .2], [.8, .05, .05, .05, .05],
                         [.01, .09, .2, .3, .4]])
        actual = fit_predict_booster(x, np.arange(5), np.ones(5), x, ["feature"],
                                     parameters={**BASE_PARAMETERS, "max_depth": 2}, rounds=0,
                                     train_base=base, predict_base=base)
        np.testing.assert_allclose(actual, base, atol=6e-8, rtol=0)
        with self.assertRaises(ValueError):
            fit_predict_booster(x, np.arange(5), np.ones(5), x, ["feature"],
                                parameters=BASE_PARAMETERS, rounds=0, train_base=base)

    def test_invalid_probability_vectors_are_rejected(self):
        for invalid in (np.ones((2, 5)), np.full((2, 5), np.nan),
                        np.full((2, 4), .25), np.array([[-.1, .3, .3, .3, .2]])):
            with self.assertRaises(ValueError):
                validate_probabilities(invalid, len(invalid))


if __name__ == "__main__":
    unittest.main()
