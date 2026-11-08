"""Numerical and probability invariants for the fixed 13-parameter calibration."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "research")]
from followup_calibration import (fit_calibrator, identity_theta,
                                  objective_gradient, predict_probabilities)


def fixture():
    rng = np.random.default_rng(1729)
    w = rng.uniform(.04, .96, 36)
    size = rng.dirichlet([3, 2, 1, .5], 36)
    kinds = np.tile(np.arange(3), 12)
    labels = np.tile(np.arange(5), 8)[:36]
    weights = rng.uniform(.5, 1.5, 36)
    weights /= weights.sum()
    return w, size, kinds, labels, weights


class CalibrationTests(unittest.TestCase):
    def test_identity_preserves_input_probabilities(self):
        w, size, kinds, _, _ = fixture()
        expected = np.column_stack([1-w, w[:, None]*size])
        actual = predict_probabilities(w, size, kinds, identity_theta())
        np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-15)

    def test_analytic_gradient_matches_finite_differences(self):
        data = fixture()
        theta = identity_theta() + np.linspace(-.13, .19, 13)
        loss, gradient = objective_gradient(theta, *data)
        self.assertTrue(np.isfinite(loss))
        numerical = np.empty(13)
        for j in range(13):
            step = np.zeros(13)
            step[j] = 1e-6
            numerical[j] = (objective_gradient(theta+step, *data)[0]
                            - objective_gradient(theta-step, *data)[0]) / 2e-6
        np.testing.assert_allclose(gradient, numerical, rtol=2e-5, atol=2e-8)

    def test_conditional_size_offsets_affect_only_their_type(self):
        w, size, kinds, _, _ = fixture()
        theta = identity_theta()
        theta[4] = .7  # first positive-payout offset, machine type 0
        base = predict_probabilities(w, size, kinds, identity_theta())
        actual = predict_probabilities(w, size, kinds, theta)
        np.testing.assert_allclose(actual[kinds != 0], base[kinds != 0], rtol=0, atol=3e-15)
        np.testing.assert_allclose(actual[:, 0], base[:, 0], rtol=0, atol=3e-15)
        self.assertTrue((actual[kinds == 0, 1] > base[kinds == 0, 1]).all())
        np.testing.assert_allclose(actual.sum(1), 1, rtol=0, atol=3e-15)

    def test_repeated_rows_with_split_weights_preserve_objective(self):
        data = fixture()
        theta = identity_theta() + np.linspace(-.05, .1, 13)
        first = objective_gradient(theta, *data)
        repeated = [np.repeat(a, 2, axis=0) for a in data]
        repeated[-1] /= 2
        second = objective_gradient(theta, *repeated)
        self.assertAlmostEqual(first[0], second[0], places=14)
        np.testing.assert_allclose(first[1], second[1], rtol=0, atol=3e-14)

    def test_fit_improves_penalized_training_objective_with_valid_predictions(self):
        data = fixture()
        initial = objective_gradient(identity_theta(), *data)[0]
        theta, diagnostics = fit_calibrator(*data)
        final = objective_gradient(theta, *data)[0]
        self.assertLessEqual(final, initial + 1e-10)
        self.assertTrue(np.isfinite(theta).all())
        self.assertGreaterEqual(theta[0], .25)
        self.assertLessEqual(theta[0], 4)
        prediction = predict_probabilities(*data[:3], theta)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue((prediction >= 0).all())
        np.testing.assert_allclose(prediction.sum(1), 1, rtol=0, atol=3e-15)
        self.assertIsInstance(diagnostics, dict)


if __name__ == "__main__":
    unittest.main()
