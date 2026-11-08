"""Candidate lag rules on win indicators and their (recency-weighted) cell counts.

A rule maps the win indicators at exact lags t-1..t-8 to a binary cell:
  rule 0      null rule, a single cell
  rules 1-8   single lag k: cell = win(t-k)
  rules 9-36  pair (a, b): cell = win(t-a) XOR win(t-b)
A missing tick is unknown: a lag is read only from the tick exactly k before, never from the kth row.
"""

from .history import win_map

MAX_LAG = 8
PAIRS = tuple((a, b) for a in range(1, MAX_LAG + 1) for b in range(a + 1, MAX_LAG + 1))
RULES = (("null",),) + tuple(("lag", k) for k in range(1, MAX_LAG + 1)) + tuple(("xor", a, b) for a, b in PAIRS)
N_RULES = len(RULES)


def rule_cells(wins, tick):
    """Cell of every rule for predicting `tick`; None where a needed lag is unobserved."""
    lag = [None] + [wins.get(tick - k) for k in range(1, MAX_LAG + 1)]
    cells = [0]
    cells.extend(lag[1:])
    cells.extend(None if lag[a] is None or lag[b] is None else lag[a] ^ lag[b] for a, b in PAIRS)
    return cells


def training_events(payouts):
    """(tick, win, flat count indices) for every tick whose own value and lags 1..8 are all observed.

    Using one common tick set keeps marginal likelihoods comparable across rules.
    """
    wins = win_map(payouts)
    events = []
    for tick in sorted(payouts):
        cells = rule_cells(wins, tick)
        if None not in cells:
            events.append((tick, wins[tick], tuple(2 * r + c for r, c in enumerate(cells))))
    return events


def weight(age, half_life):
    return 1.0 if half_life is None else 2.0 ** (-age / half_life)


def weighted_counts(events, tick, half_life):
    """Weighted (n, k) per rule cell over events strictly before `tick`; weight halves every half_life ticks."""
    n = [0.0] * (2 * N_RULES)
    k = [0.0] * (2 * N_RULES)
    for event_tick, win, indices in events:
        if event_tick >= tick:
            break
        w = weight(tick - 1 - event_tick, half_life)
        for i in indices:
            n[i] += w
        if win:
            for i in indices:
                k[i] += w
    return n, k
