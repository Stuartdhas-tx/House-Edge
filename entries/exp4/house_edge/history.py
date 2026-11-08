"""Per-machine histories built with the kit's leak guard (rejects any tick at/after the cutoff)."""

from slot_forecasting.schema import index_history, unique_queries

PAYOUTS = (0, 1, 2, 5, 20)
SIZES = (1, 2, 5, 20)


def machine_payouts(queries, history):
    """{query_id: (query, {tick: payout})} using only ticks before each query's cutoff."""
    indexed = index_history(queries, history)
    result = {}
    for query in sorted(unique_queries(queries).values(), key=lambda q: q.query_id):
        plays = indexed.get((query.episode_id, query.machine_id), [])
        result[query.query_id] = (query, {p.tick: p.payout for p in plays if p.tick < query.cutoff})
    return result


def win_map(payouts):
    return {tick: int(value > 0) for tick, value in payouts.items()}


def size_counts(payouts, tick):
    """Counts of winning payout sizes (1, 2, 5, 20) observed strictly before `tick`."""
    counts = [0, 0, 0, 0]
    for t, value in payouts.items():
        if t < tick and value > 0:
            counts[SIZES.index(value)] += 1
    return counts
