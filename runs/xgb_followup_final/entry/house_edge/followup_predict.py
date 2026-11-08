"""Experimental frozen followup inference; standard library, with no fitting.

The base forecaster must use identity win calibration. Tree feature columns
contain its original five probabilities rounded to float32; base margins use
the unrounded probabilities. This module does not change the active model or
entry registry. Generate any deployment CSV with this same portable runtime.
"""

from collections.abc import Mapping
import math

from .history import machine_payouts
from .xgb_features_portable import build_row, feature_names
from .xgb_portable import float32

TYPES = ("balanced", "jackpot", "steady")
PROBABILITY_NAMES = ("exp5_p_0", "exp5_p_1", "exp5_p_2", "exp5_p_5", "exp5_p_20")
KINDS = ("type_size_cal", "compact_xgb", "full_xgb", "half_full")


def _identity_base(base_model):
    """Check calibration attributes when exposed, without mutating the model."""
    if hasattr(base_model, "calibration_slope") and base_model.calibration_slope != 1.0:
        raise ValueError("Followup inference requires identity base calibration (slope=1).")
    if hasattr(base_model, "calibration_intercepts"):
        intercepts = base_model.calibration_intercepts
        if (not isinstance(intercepts, Mapping) or set(intercepts) != set(TYPES)
                or any(intercepts[kind] != 0.0 for kind in TYPES)):
            raise ValueError("Followup inference requires zero base calibration intercepts for all types.")


def _parameters(config, key, count):
    try:
        original = list(config[key])
        values = [float(value) for value in original]
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{key} must contain {count} finite parameters.") from error
    if len(values) != count or any(isinstance(v, bool) for v in original) or not all(map(math.isfinite, values)):
        raise ValueError(f"{key} must contain {count} finite parameters.")
    return values


def _probabilities(values, *, normalize):
    try:
        values = [float(value) for value in values]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Expected five valid probabilities.") from error
    if len(values) != 5 or any(not math.isfinite(p) or p < 0 or p > 1 for p in values):
        raise ValueError("Expected five finite probabilities in [0,1].")
    total = math.fsum(values)
    if abs(total - 1.0) > 2e-6:
        raise ValueError("Probabilities must sum to one within 2e-6.")
    return [p / total for p in values] if normalize else values


def _sigmoid(value):
    # Stable at either tail, including finite parameters whose product overflows.
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _calibrated(raw, machine_type, theta, *, size_offsets):
    raw = _probabilities(raw, normalize=True)
    win = math.fsum(raw[1:])
    if win <= 0:
        raise ValueError("Calibration requires positive base winning probability to recover conditional sizes.")
    sizes = [p / win for p in raw[1:]]
    win = min(1.0 - 1e-6, max(1e-6, win))
    kind = TYPES.index(machine_type)
    calibrated_win = _sigmoid(theta[0] * math.log(win / (1.0 - win)) + theta[1 + kind])
    if size_offsets:
        offsets = theta[4 + 3 * kind:7 + 3 * kind] + [0.0]
        log_sizes = [math.log(max(1e-12, min(1.0, p))) + offset for p, offset in zip(sizes, offsets)]
        maximum = max(log_sizes)
        scores = [math.exp(p - maximum) for p in log_sizes]
        total = math.fsum(scores)
        sizes = [p / total for p in scores]
    else:
        # EXP-5's five() applies this second clamp after win calibration.
        calibrated_win = min(1.0 - 1e-6, max(1e-6, calibrated_win))
    return _probabilities([1.0 - calibrated_win] + [calibrated_win * p for p in sizes], normalize=True)


def _check_booster(booster, names):
    if booster is None or not callable(getattr(booster, "predict_row", None)):
        raise ValueError("This followup kind requires a booster with predict_row(features, base_prob).")
    if hasattr(booster, "num_features") and booster.num_features != len(names):
        raise ValueError(f"Booster must expect {len(names)} features for this followup kind.")
    saved_names = getattr(booster, "feature_names", ())
    if saved_names and tuple(saved_names) != tuple(names):
        raise ValueError("Booster feature names/order do not match the followup feature schema.")


def predict_followup(base_model, postprocess_dict, queries, history, booster=None):
    """Apply one frozen followup to one identity-calibrated base prediction batch.

    Config kinds are ``type_size_cal`` (13-value ``theta``), ``compact_xgb``
    (type indicators + raw probabilities), ``full_xgb`` (history features +
    raw probabilities), or ``half_full`` (equal full-tree/EXP-5-calibrated mix,
    with 4-value ``baseline_theta``). Both theta formats begin with win slope
    and intercepts in balanced/jackpot/steady order. The size map then has
    three offsets per type, with payout 20 as the zero-offset reference.

    Query/history iterables are materialized once. Each episode must have one
    shared query cutoff, matching the kit's generated episodes. The kit rejects
    any exposed record at or after that cutoff; history features independently
    truncate every peer at the current query's cutoff.
    """
    if not isinstance(postprocess_dict, Mapping) or postprocess_dict.get("kind") not in KINDS:
        raise ValueError(f"Followup kind must be one of {KINDS}.")
    kind = postprocess_dict["kind"]
    _identity_base(base_model)
    theta = _parameters(postprocess_dict, "theta", 13) if kind == "type_size_cal" else None
    baseline = _parameters(postprocess_dict, "baseline_theta", 4) if kind == "half_full" else None
    if kind != "type_size_cal":
        names = ([f"type_{t}" for t in TYPES] if kind == "compact_xgb" else feature_names())
        _check_booster(booster, names + list(PROBABILITY_NAMES))

    queries, history = tuple(queries), tuple(history)
    machines = machine_payouts(queries, history)
    episodes = {}
    for query, payouts in machines.values():
        if query.machine_type not in TYPES:
            raise ValueError(f"Unknown machine type: {query.machine_type}.")
        episodes.setdefault(query.episode_id, []).append((query, payouts))
    if any(len({query.cutoff for query, _ in episode}) != 1 for episode in episodes.values()):
        raise ValueError("Followup inference requires a shared cutoff within each episode.")
    raw_predictions = base_model.predict(queries, history)
    if not isinstance(raw_predictions, Mapping) or set(raw_predictions) != set(machines):
        raise ValueError("Base predictions must contain exactly the input query IDs.")
    raw_predictions = {qid: _probabilities(values, normalize=False) for qid, values in raw_predictions.items()}

    result = {}
    for qid, (query, own) in machines.items():
        raw = raw_predictions[qid]
        if kind == "type_size_cal":
            probabilities = _calibrated(raw, query.machine_type, theta, size_offsets=True)
        else:
            if kind == "compact_xgb":
                row = [float(query.machine_type == t) for t in TYPES]
            else:
                peers = [payouts for other, payouts in episodes[query.episode_id]
                         if other.machine_id != query.machine_id]
                row = build_row(query, own, peers)
            row.extend(float32(p) for p in raw)
            probabilities = _probabilities(booster.predict_row(row, raw), normalize=True)
            if kind == "half_full":
                calibrated = _calibrated(raw, query.machine_type, baseline, size_offsets=False)
                probabilities = [0.5 * a + 0.5 * b for a, b in zip(probabilities, calibrated)]
        result[qid] = _probabilities(probabilities, normalize=True)
    return result
