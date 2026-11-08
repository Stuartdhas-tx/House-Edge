"""Standard-library feature builder matching research/xgb_features.py.

The caller supplies only same-episode peers, excluding the query machine.
Every history is truncated strictly before the requested cutoff. Final values
are rounded to IEEE float32, as in the NumPy training feature matrix.
"""

import math
import struct

from .rules import MAX_LAG, PAIRS

PAYOUTS = (0, 1, 2, 5, 20)
TYPES = ("balanced", "jackpot", "steady")
WINDOWS = (16, 32, 64, None)
RULE_WINDOWS = (32, 64, None)
SIZE_WINDOWS = (32, None)
RULE_NAMES = ["null"] + [f"lag{k}" for k in range(1, MAX_LAG + 1)] + [f"xor{a}_{b}" for a, b in PAIRS]
SUMMARY_FIELDS = ("n", "coverage", "win_n", "win_rate", "big_rate", "payout_mean", "payout_sd",
                  "p0", "p1", "p2", "p5", "p20", "size1", "size2", "size5", "size20")
RULE_FIELDS = ("current_n", "current_wins", "current_rate", "other_n", "other_wins", "other_rate",
               "contrast_abs", "contrast_current")
SIZE_FIELDS = ("current_n", "current_p1", "current_p2", "current_p5", "current_p20",
               "other_n", "other_p1", "other_p2", "other_p5", "other_p20", "cell0_n", "cell1_n")
PEER_FIELDS = ("machine_n", "n", "win_rate", "big_rate", "payout_mean", "win_rate_mean",
               "win_rate_sd", "coverage_mean")
PAYOUT_INDEX = {value: i for i, value in enumerate(PAYOUTS)}
NAN = float("nan")


def _window_name(window):
    return "all" if window is None else str(window)


def feature_names():
    """The exact ordered training feature schema; callers receive a fresh list."""
    names = [f"type_{t}" for t in TYPES]
    names += ["cutoff", "signal", "observed_n", "coverage", "missing_n", "last_observed_age", "trailing_observed_n"]
    names += [f"lag{k}_{field}" for k in range(1, MAX_LAG + 1) for field in ("payout", "win", "big", "missing")]
    names += [f"history_{_window_name(w)}_{field}" for w in WINDOWS for field in SUMMARY_FIELDS]
    names += ["win_rate_16_minus_all", "win_rate_32_minus_all", "big_rate_16_minus_all", "payout_mean_16_minus_all"]
    names += [f"rule_{name}_cell" for name in RULE_NAMES]
    names += [f"rule_{name}_{_window_name(w)}_{field}" for w in RULE_WINDOWS for name in RULE_NAMES for field in RULE_FIELDS]
    names += [f"size_lag{k}_{_window_name(w)}_{field}" for w in SIZE_WINDOWS
              for k in range(1, MAX_LAG + 1) for field in SIZE_FIELDS]
    names += [f"peer_{_window_name(w)}_{field}" for w in WINDOWS for field in PEER_FIELDS]
    return names


_ROW_FLOAT32 = struct.Struct(f"<{len(feature_names())}f")


def _divide(numerator, denominator):
    return numerator / denominator if denominator > 0 else NAN


def _start(tick, window):
    return 0 if window is None else max(0, tick - window)


def _counts(past, start):
    counts = [0] * len(PAYOUTS)
    for tick, value in past.items():
        if tick >= start:
            counts[PAYOUT_INDEX[value]] += 1
    return counts


def _summary(counts, span):
    total = sum(counts)
    wins = sum(counts[1:])
    mean = _divide(sum(c * value for c, value in zip(counts, PAYOUTS)), total)
    second = _divide(sum(c * value * value for c, value in zip(counts, PAYOUTS)), total)
    sd = math.sqrt(max(0.0, second - mean * mean)) if total else NAN
    return ([total, _divide(total, span), wins, _divide(wins, total), _divide(sum(counts[2:]), total), mean, sd]
            + [_divide(c, total) for c in counts] + [_divide(c, wins) for c in counts[1:]])


def _cells(wins, tick):
    lags = [wins.get(tick - k) for k in range(1, MAX_LAG + 1)]
    return ([0] + lags + [None if lags[a - 1] is None or lags[b - 1] is None
                          else lags[a - 1] ^ lags[b - 1] for a, b in PAIRS])


def _rule_statistics(past, tick):
    wins = {u: int(v > 0) for u, v in past.items()}
    # Count each exact-available rule independently; missing unrelated lags do
    # not discard an otherwise usable response, matching the tree features.
    statistics = [([[0, 0] for _ in RULE_NAMES], [[0, 0] for _ in RULE_NAMES]) for _ in RULE_WINDOWS]
    starts = [_start(tick, window) for window in RULE_WINDOWS]
    for u, win in wins.items():
        cells = _cells(wins, u)
        for start, (counts, successes) in zip(starts, statistics):
            if u >= start:
                for r, cell in enumerate(cells):
                    if cell is not None:
                        counts[r][cell] += 1
                        successes[r][cell] += win
    return _cells(wins, tick), statistics


def _size_statistics(past, tick):
    statistics = [[[[0] * 4 for _ in range(2)] for _ in range(MAX_LAG)] for _ in SIZE_WINDOWS]
    starts = [_start(tick, window) for window in SIZE_WINDOWS]
    for u, value in past.items():
        if value <= 0:
            continue
        category = PAYOUT_INDEX[value] - 1
        for lag in range(1, MAX_LAG + 1):
            previous = past.get(u - lag)
            if previous is not None:
                cell = int(previous >= 2)
                for start, counts in zip(starts, statistics):
                    if u >= start:
                        counts[lag - 1][cell][category] += 1
    return statistics


def build_row(query, own_payouts, peers, cutoff=None):
    """Build one 1,259-feature row without NumPy or any fitted state.

    ``peers`` is a list of other machines' payout dictionaries in this episode.
    ``cutoff`` defaults to the query cutoff; an earlier pseudo-query masks the
    query-time signal. Missing exact ticks are unknown, never zero payouts.
    """
    tick = query.cutoff if cutoff is None else cutoff
    if not isinstance(tick, int) or isinstance(tick, bool) or tick < 0:
        raise ValueError("cutoff must be a nonnegative integer.")
    past = {u: v for u, v in own_payouts.items() if 0 <= u < tick}
    peer_pasts = [{u: v for u, v in history.items() if 0 <= u < tick} for history in peers]
    total = len(past)
    last = max(past, default=-1)
    trailing = 0
    while tick - 1 - trailing in past:
        trailing += 1
    row = [query.machine_type == kind for kind in TYPES]
    row += [tick, query.signal if tick == query.cutoff else NAN, total, _divide(total, tick), tick - total,
            tick - 1 - last if last >= 0 else NAN, trailing]
    current_big = []
    for lag in range(1, MAX_LAG + 1):
        value = past.get(tick - lag)
        if value is None:
            row += [NAN, NAN, NAN, 1.0]
            current_big.append(None)
        else:
            row += [value, int(value > 0), int(value >= 2), 0.0]
            current_big.append(int(value >= 2))
    summaries = [_summary(_counts(past, _start(tick, window)), tick if window is None else min(tick, window))
                 for window in WINDOWS]
    for summary in summaries:
        row.extend(summary)
    row += [summaries[0][3] - summaries[-1][3], summaries[1][3] - summaries[-1][3],
            summaries[0][4] - summaries[-1][4], summaries[0][5] - summaries[-1][5]]
    current, rule_statistics = _rule_statistics(past, tick)
    row.extend(NAN if cell is None else cell for cell in current)
    for counts, successes in rule_statistics:
        for r, cell in enumerate(current):
            rates = [(successes[r][c] + 0.5) / (counts[r][c] + 1.0) for c in (0, 1)]
            if cell is None:
                row.extend([NAN] * 6)
            else:
                row += [counts[r][cell], successes[r][cell], rates[cell],
                        counts[r][1 - cell], successes[r][1 - cell], rates[1 - cell]]
            row += [abs(rates[1] - rates[0]), NAN if cell is None else rates[cell] - rates[1 - cell]]
    for counts in _size_statistics(past, tick):
        for lag, cell in enumerate(current_big):
            totals = [sum(counts[lag][c]) for c in (0, 1)]
            if cell is None:
                row.extend([NAN] * 10)
            else:
                for c in (cell, 1 - cell):
                    row.append(totals[c])
                    row.extend((count + 0.5) / (totals[c] + 2.0) for count in counts[lag][c])
            row.extend(totals)
    for window in WINDOWS:
        start = _start(tick, window)
        span = tick if window is None else min(tick, window)
        counts = [_counts(history, start) for history in peer_pasts]
        pooled = [sum(c[j] for c in counts) for j in range(len(PAYOUTS))]
        pn = sum(pooled)
        rates = [sum(c[1:]) / sum(c) for c in counts if sum(c) > 0]
        mean = sum(rates) / len(rates) if rates else NAN
        sd = math.sqrt(sum((r - mean) ** 2 for r in rates) / len(rates)) if rates else NAN
        coverage = [sum(c) / span for c in counts] if span else []
        row += [len(peer_pasts), pn, _divide(sum(pooled[1:]), pn), _divide(sum(pooled[2:]), pn),
                _divide(sum(c * value for c, value in zip(pooled, PAYOUTS)), pn), mean, sd,
                sum(coverage) / len(coverage) if coverage else NAN]
    # A single round-to-float32 operation reproduces the training matrix dtype.
    return list(_ROW_FLOAT32.unpack(_ROW_FLOAT32.pack(*row)))
