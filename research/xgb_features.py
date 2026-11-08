"""Deterministic, unfitted features for the episode-grouped tree benchmark.

``build_features(snaps)`` reads only snapshot metadata and observed histories.
No label, condition, identifier value, fitted global prior, or future payout is
included. IDs only identify a machine and its same-episode peers. Missing exact
lags and unavailable rates are NaN; query-time signal is unknown on pseudo rows.
"""

from collections import defaultdict

import numpy as np

from house_edge.rules import MAX_LAG, PAIRS, RULES

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


def _window_name(window):
    return "all" if window is None else str(window)


def _feature_names():
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


def _divide(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    result = np.full(numerator.shape, np.nan)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def _prefix(values):
    return np.concatenate([np.zeros((1,) + values.shape[1:], dtype=np.int32),
                           np.cumsum(values, axis=0, dtype=np.int32)])


class _History:
    """Calendar-tick prefixes: prefix[t] always excludes tick t itself."""

    def __init__(self, payouts, end, with_rules):
        self.values = np.full(end, np.nan)
        for tick, payout in payouts.items():
            if 0 <= tick < end:
                self.values[tick] = payout
        observed = np.isfinite(self.values)
        self.payout_counts = _prefix(np.stack([self.values == value for value in PAYOUTS], axis=1))
        indexes = np.arange(end)
        last_seen = np.maximum.accumulate(np.where(observed, indexes, -1)) if end else indexes
        last_missing = np.maximum.accumulate(np.where(~observed, indexes, -1)) if end else indexes
        self.last_seen = np.concatenate([[-1], last_seen])
        self.trailing = np.concatenate([[0], indexes - last_missing])
        self.win = np.where(observed, self.values > 0, np.nan)
        self.big = np.where(observed, self.values >= 2, np.nan)
        if not with_rules:
            return
        lags = np.full((end, MAX_LAG), np.nan)
        for lag in range(1, MAX_LAG + 1):
            if lag < end:
                lags[lag:, lag - 1] = self.win[:-lag]
        cells = np.zeros((end, len(RULES)))
        cells[:, 1:1 + MAX_LAG] = lags
        for r, (a, b) in enumerate(PAIRS, 1 + MAX_LAG):
            valid = np.isfinite(lags[:, a - 1]) & np.isfinite(lags[:, b - 1])
            cells[:, r] = np.where(valid, lags[:, a - 1] != lags[:, b - 1], np.nan)
        counts = np.stack([observed[:, None] & (cells == c) for c in (0, 1)], axis=2)
        wins = counts & (self.win == 1)[:, None, None]
        self.rule_n, self.rule_wins = _prefix(counts), _prefix(wins)
        # Size responses use exact observed lag categories; no complete-window deletion.
        size = np.zeros((end, MAX_LAG, 2, 4), dtype=np.int32)
        positive = observed & (self.values > 0)
        for lag in range(1, MAX_LAG + 1):
            earlier = np.full(end, np.nan)
            if lag < end:
                earlier[lag:] = self.big[:-lag]
            for cell in (0, 1):
                selected = positive & (earlier == cell)
                for j, value in enumerate(PAYOUTS[1:]):
                    size[:, lag - 1, cell, j] = selected & (self.values == value)
        self.size_counts = _prefix(size)

    def window_counts(self, ticks, window):
        start = np.zeros_like(ticks) if window is None else np.maximum(0, ticks - window)
        return self.payout_counts[ticks] - self.payout_counts[start]

    def summary(self, ticks, window):
        counts = self.window_counts(ticks, window).astype(float)
        total = counts.sum(axis=1)
        wins = counts[:, 1:].sum(axis=1)
        big = counts[:, 2:].sum(axis=1)
        span = ticks if window is None else np.minimum(ticks, window)
        mean = _divide(counts @ np.asarray(PAYOUTS), total)
        variance = _divide(counts @ np.square(PAYOUTS), total) - mean ** 2
        return np.column_stack([total, _divide(total, span), wins, _divide(wins, total), _divide(big, total),
                                mean, np.sqrt(np.maximum(0, variance)), _divide(counts, total[:, None]),
                                _divide(counts[:, 1:], wins[:, None])])

    def exact_lags(self, ticks, values):
        locations = ticks[:, None] - np.arange(1, MAX_LAG + 1)[None, :]
        valid = (locations >= 0) & (locations < len(values))
        result = np.full(locations.shape, np.nan)
        result[valid] = values[locations[valid]]
        return result


def _own_features(query, history, ticks, real):
    lag_payout = history.exact_lags(ticks, history.values)
    lag_win = np.where(np.isfinite(lag_payout), lag_payout > 0, np.nan)
    lag_big = np.where(np.isfinite(lag_payout), lag_payout >= 2, np.nan)
    current = np.zeros((len(ticks), len(RULES)))
    current[:, 1:1 + MAX_LAG] = lag_win
    for r, (a, b) in enumerate(PAIRS, 1 + MAX_LAG):
        known = np.isfinite(lag_win[:, a - 1]) & np.isfinite(lag_win[:, b - 1])
        current[:, r] = np.where(known, lag_win[:, a - 1] != lag_win[:, b - 1], np.nan)
    summaries = [history.summary(ticks, window) for window in WINDOWS]
    all_summary = summaries[-1]
    total = all_summary[:, 0]
    last = history.last_seen[ticks]
    signal = np.where(real & (ticks == query.cutoff), query.signal, np.nan)
    blocks = [np.tile([query.machine_type == kind for kind in TYPES], (len(ticks), 1)),
              np.column_stack([ticks, signal, total, _divide(total, ticks), ticks - total,
                               np.where(last >= 0, ticks - 1 - last, np.nan), history.trailing[ticks]]),
              np.stack([lag_payout, lag_win, lag_big, ~np.isfinite(lag_payout)], axis=2).reshape(len(ticks), -1)]
    blocks.extend(summaries)
    blocks.append(np.column_stack([summaries[0][:, 3] - all_summary[:, 3], summaries[1][:, 3] - all_summary[:, 3],
                                   summaries[0][:, 4] - all_summary[:, 4], summaries[0][:, 5] - all_summary[:, 5]]))
    blocks.append(current)
    known = np.isfinite(current)
    cell = np.where(known, current, 0).astype(np.intp)
    row = np.arange(len(ticks))[:, None]
    rule = np.arange(len(RULES))[None, :]
    for window in RULE_WINDOWS:
        start = np.zeros_like(ticks) if window is None else np.maximum(0, ticks - window)
        n = history.rule_n[ticks] - history.rule_n[start]
        k = history.rule_wins[ticks] - history.rule_wins[start]
        # Fixed Jeffreys smoothing is local to this machine, never a global fit.
        rate = (k + 0.5) / (n + 1.0)
        n_cur, n_other = n[row, rule, cell], n[row, rule, 1 - cell]
        k_cur, k_other = k[row, rule, cell], k[row, rule, 1 - cell]
        r_cur, r_other = rate[row, rule, cell], rate[row, rule, 1 - cell]
        fields = [np.where(known, x, np.nan) for x in (n_cur, k_cur, r_cur, n_other, k_other, r_other)]
        fields += [np.abs(rate[:, :, 1] - rate[:, :, 0]), np.where(known, r_cur - r_other, np.nan)]
        blocks.append(np.stack(fields, axis=2).reshape(len(ticks), -1))
    known = np.isfinite(lag_big)
    cell = np.where(known, lag_big, 0).astype(np.intp)
    lag = np.arange(MAX_LAG)[None, :]
    for window in SIZE_WINDOWS:
        start = np.zeros_like(ticks) if window is None else np.maximum(0, ticks - window)
        counts = history.size_counts[ticks] - history.size_counts[start]
        n = counts.sum(axis=3)
        rate = (counts + 0.5) / (n[:, :, :, None] + 2.0)
        cur_n, other_n = n[row, lag, cell], n[row, lag, 1 - cell]
        cur_rate, other_rate = rate[row, lag, cell], rate[row, lag, 1 - cell]
        cur = np.concatenate([cur_n[:, :, None], cur_rate], axis=2)
        other = np.concatenate([other_n[:, :, None], other_rate], axis=2)
        fields = np.concatenate([np.where(known[:, :, None], cur, np.nan),
                                 np.where(known[:, :, None], other, np.nan), n], axis=2)
        blocks.append(fields.reshape(len(ticks), -1))
    return np.concatenate(blocks, axis=1)


def build_features(snaps):
    """Return float32 features and names, preserving ``snaps.rows`` order.

    Only ``rows`` and ``machines`` are required, so synthetic snapshot fixtures
    can exercise missing-data behavior even where common.Snapshots omits rows.
    Empty empirical rates are NaN. Rule rates use fixed Beta(.5,.5) smoothing;
    size-cell rates use fixed Dirichlet(.5,.5,.5,.5) smoothing, with support
    counts alongside. Peers are all other available machines in that episode,
    each truncated independently at this row's tick.
    """
    names = _feature_names()
    result = np.full((len(snaps.rows), len(names)), np.nan, dtype=np.float32)
    if not snaps.rows:
        return result, names
    own_rows, episode_rows = defaultdict(list), defaultdict(list)
    episode_machines = defaultdict(list)
    for qid, (query, _) in snaps.machines.items():
        episode_machines[query.episode_id].append(qid)
    for i, (qid, episode, _, _, tick, _) in enumerate(snaps.rows):
        if tick < 0:
            raise ValueError("Snapshot cutoffs must be nonnegative.")
        own_rows[qid].append(i)
        episode_rows[episode].append(i)
    max_tick = {episode: max(snaps.rows[i][4] for i in rows) for episode, rows in episode_rows.items()}
    histories = {}
    for episode in episode_rows:
        for qid in episode_machines[episode]:
            histories[qid] = _History(snaps.machines[qid][1], max_tick[episode], qid in own_rows)
    peer_width = len(WINDOWS) * len(PEER_FIELDS)
    for qid, indexes in own_rows.items():
        query, _ = snaps.machines[qid]
        ticks = np.array([snaps.rows[i][4] for i in indexes], dtype=np.intp)
        real = np.array([snaps.rows[i][5] for i in indexes], dtype=bool)
        result[indexes, :-peer_width] = _own_features(query, histories[qid], ticks, real)
    # Prefixes include no model fit; peer aggregation is conditional on each tick.
    for episode, indexes in episode_rows.items():
        qids = episode_machines[episode]
        positions = {qid: j for j, qid in enumerate(qids)}
        ticks, inverse = np.unique([snaps.rows[i][4] for i in indexes], return_inverse=True)
        for wi, window in enumerate(WINDOWS):
            counts = np.stack([histories[qid].window_counts(ticks, window) for qid in qids])
            total = counts.sum(axis=2)
            wins = counts[:, :, 1:].sum(axis=2)
            rates = _divide(wins, total)
            span = ticks if window is None else np.minimum(ticks, window)
            coverage = _divide(total, span[None, :])
            pooled = counts.sum(axis=0)
            for local, i in enumerate(indexes):
                own = positions[snaps.rows[i][0]]
                ti = inverse[local]
                pc = pooled[ti] - counts[own, ti]
                pn = pc.sum()
                others = np.arange(len(qids)) != own
                rr = rates[others, ti]
                rr = rr[np.isfinite(rr)]
                cov = coverage[others, ti]
                cov = cov[np.isfinite(cov)]
                row = [int(others.sum()), pn,
                       pc[1:].sum() / pn if pn else np.nan,
                       pc[2:].sum() / pn if pn else np.nan,
                       pc @ np.asarray(PAYOUTS) / pn if pn else np.nan,
                       rr.mean() if len(rr) else np.nan, rr.std() if len(rr) else np.nan,
                       cov.mean() if len(cov) else np.nan]
                start = len(names) - peer_width + wi * len(PEER_FIELDS)
                result[i, start:start + len(PEER_FIELDS)] = row
    if np.isinf(result).any():
        raise AssertionError("Features must be finite or NaN, never infinite.")
    return result, names
