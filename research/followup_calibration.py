"""Thirteen-parameter win/type-size calibration, with an analytic Brier gradient.

theta = [win_slope, three type win intercepts, nine type/size offsets].
The last size offset in each type is zero. Only the centered four-offset vector
is penalized, making the regularization independent of the reference class.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, softmax

PENALTY = 0.001
BOUNDS = [(0.25, 4.0)] + [(-4.0, 4.0)] * 12


def identity_theta():
    return np.r_[1.0, np.zeros(12)]


def size_offsets(theta):
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (13,) or not np.isfinite(theta).all():
        raise ValueError("Expected thirteen finite calibration parameters.")
    return np.column_stack([theta[4:].reshape(3, 3), np.zeros(3)])


def _inputs(w, size, machine_type):
    w, size, machine_type = np.asarray(w, float), np.asarray(size, float), np.asarray(machine_type)
    if (w.ndim != 1 or size.shape != (len(w), 4) or machine_type.shape != w.shape
            or machine_type.dtype.kind not in "iu" or (machine_type < 0).any() or (machine_type > 2).any()
            or not np.isfinite(w).all() or (w < 0).any() or (w > 1).any()
            or not np.isfinite(size).all() or (size < 0).any()
            or not np.allclose(size.sum(axis=1), 1.0, rtol=0, atol=1e-10)):
        raise ValueError("Invalid win probabilities, conditional sizes, or type indices.")
    w = np.clip(w, 1e-6, 1 - 1e-6)
    return np.log(w / (1 - w)), np.log(np.clip(size, 1e-12, 1)), machine_type


def _predict(theta, logit_w, log_size, machine_type):
    offsets = size_offsets(theta)
    theta = np.asarray(theta)
    win = expit(theta[0] * logit_w + theta[1 + machine_type])
    sizes = softmax(log_size + offsets[machine_type], axis=1)
    return np.column_stack([1 - win, win[:, None] * sizes]), win, sizes, offsets


def predict_probabilities(w, size, machine_type, theta):
    """Apply the fitted map, returning one five-class distribution per row."""
    inputs = _inputs(w, size, machine_type)
    return _predict(theta, *inputs)[0]


def _objective(theta, logit_w, log_size, machine_type, y, weights, penalty):
    probabilities, win, sizes, offsets = _predict(theta, logit_w, log_size, machine_type)
    residual = probabilities.copy()
    residual[np.arange(len(y)), y] -= 1
    centered = offsets - offsets.mean(axis=1, keepdims=True)
    value = float(weights @ (residual ** 2).sum(axis=1) + penalty * (centered ** 2).sum())

    size_residual_mean = (residual[:, 1:] * sizes).sum(axis=1)
    d_win_logit = 2 * (-residual[:, 0] + size_residual_mean) * win * (1 - win)
    weighted_win = weights * d_win_logit
    gradient = np.zeros(13)
    gradient[0] = weighted_win @ logit_w
    gradient[1:4] = np.bincount(machine_type, weights=weighted_win, minlength=3)
    d_size_logit = 2 * win[:, None] * sizes * (residual[:, 1:] - size_residual_mean[:, None])
    offset_gradient = np.zeros((3, 3))
    np.add.at(offset_gradient, machine_type, weights[:, None] * d_size_logit[:, :3])
    offset_gradient += 2 * penalty * centered[:, :3]
    gradient[4:] = offset_gradient.ravel()
    return value, gradient


def _training_inputs(w, size, machine_type, y, weights):
    inputs = _inputs(w, size, machine_type)
    y, weights = np.asarray(y), np.asarray(weights, float)
    if (y.shape != inputs[0].shape or y.dtype.kind not in "iu" or (y < 0).any() or (y > 4).any()
            or weights.shape != y.shape or not np.isfinite(weights).all()
            or (weights < 0).any() or weights.sum() <= 0):
        raise ValueError("Invalid outcome indices or fitting weights.")
    return (*inputs, y, weights / weights.sum())


def objective_gradient(theta, w, size, machine_type, y, weights, *, penalty=PENALTY):
    """Weighted-mean Brier plus penalty and its analytic thirteen-vector gradient."""
    return _objective(theta, *_training_inputs(w, size, machine_type, y, weights), penalty)


def fit_calibrator(w, size, machine_type, y, weights, *, penalty=PENALTY):
    """Fit from identity with the preregistered bounds; return theta and diagnostics."""
    inputs = _training_inputs(w, size, machine_type, y, weights)
    result = minimize(lambda theta: _objective(theta, *inputs, penalty), identity_theta(), jac=True,
                      method="L-BFGS-B", bounds=BOUNDS,
                      options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 40})
    if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise RuntimeError("Calibration optimization returned non-finite values.")
    active = [{"index": i, "side": side, "value": float(value)}
              for i, (value, bounds) in enumerate(zip(result.x, BOUNDS))
              for side, bound in zip(("lower", "upper"), bounds) if abs(value - bound) <= 1e-6]
    diagnostics = {"success": bool(result.success), "message": str(result.message),
                   "iterations": int(result.nit), "evaluations": int(result.nfev),
                   "objective": float(result.fun), "gradient_inf_norm": float(np.abs(result.jac).max()),
                   "active_bounds": active, "penalty": penalty, "theta": result.x.tolist()}
    return result.x, diagnostics
