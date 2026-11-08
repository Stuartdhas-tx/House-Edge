"""Leakage, lag-exactness, determinism and research/shipped equivalence checks. Run: python tests/test_leakage.py [model.json]"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from slot_forecasting.io import load_inputs  # noqa: E402
from slot_forecasting.schema import Play, ValidationError  # noqa: E402
from house_edge.history import machine_payouts  # noqa: E402
from house_edge.registry import load_model  # noqa: E402
from house_edge.rules import N_RULES, RULES, rule_cells, training_events  # noqa: E402


def machine_level(model, payouts, tick, mtype):
    """Every model exposes win_probability/size_distribution for a single machine."""
    if hasattr(model, "predict_machine"):
        return model.predict_machine(payouts, tick, mtype)
    w, _ = model.win_probability(payouts, tick, mtype)
    return (w, *model.size_distribution(payouts, tick, mtype))


def main(model_path):
    model = load_model(model_path)
    queries, history = load_inputs(ROOT / "public/validation")
    machines = machine_payouts(queries, history)
    rng = random.Random(7)
    sample = sorted(machines)[:: 37]
    checked = 0
    for qid in sample:
        q, payouts = machines[qid]
        t = q.cutoff - rng.randint(1, 12)
        truncated = {u: v for u, v in payouts.items() if u < t}
        base = machine_level(model, truncated, t, q.machine_type)
        # 1. future ticks present in the dict must not matter
        assert machine_level(model, payouts, t, q.machine_type) == base, f"future leak {qid}"
        # 2. randomized future payouts must not matter
        shuffled = dict(truncated)
        for u in range(t, q.cutoff + 30):
            shuffled[u] = rng.choice((0, 1, 2, 5, 20))
        assert machine_level(model, shuffled, t, q.machine_type) == base, f"shuffle-future leak {qid}"
        checked += 1
    # 3. lag exactness: a missing tick makes rules using that lag unknown, never "the previous row"
    q, payouts = machines[sample[0]]
    t = q.cutoff
    holed = {u: v for u, v in payouts.items() if u != t - 3}
    cells = rule_cells({u: int(v > 0) for u, v in holed.items()}, t)
    for r, rule in enumerate(RULES):
        uses3 = 3 in rule[1:]
        assert (cells[r] is None) == uses3, f"lag-exactness failed for rule {rule}"
    events = {e[0] for e in training_events(holed)}
    assert all(not (u - 8 <= t - 3 <= u) for u in events), "training event uses a window containing a missing tick"
    # 4. missing is not zero
    zeroed = dict(holed)
    zeroed[t - 3] = 0
    assert machine_level(model, holed, t, q.machine_type) != machine_level(model, zeroed, t, q.machine_type), "missing treated as zero"
    # 5. the kit loader rejects any tick at/after the cutoff
    bad = list(history) + [Play(q.episode_id, q.machine_id, q.cutoff, 0)]
    try:
        model.predict(queries, bad)
        raise AssertionError("future tick was accepted")
    except ValidationError:
        pass
    # 6. determinism
    a, b = model.predict(queries, history), model.predict(queries, history)
    assert a == b, "non-deterministic predictions"
    # 7. cross-machine pooling: other machines' ticks >= t' (real or randomized) must not change any prediction
    if hasattr(model, "episode_win_probabilities"):
        episodes = {}
        for qid, (q, _) in machines.items():
            episodes.setdefault(q.episode_id, []).append(qid)
        for ep in sorted(episodes)[::12]:
            qids = sorted(episodes[ep])
            t = machines[qids[0]][0].cutoff - rng.randint(1, 12)
            def tables(transform):
                return [model.machine_tables(transform(machines[x][1]), t, machines[x][0].machine_type) for x in qids]
            trunc = model.episode_win_probabilities(tables(lambda p: {u: v for u, v in p.items() if u < t}))
            full = model.episode_win_probabilities(tables(lambda p: p))
            noisy = model.episode_win_probabilities(tables(lambda p: {**{u: v for u, v in p.items() if u < t},
                                                                       **{u: rng.choice((0, 1, 2, 5, 20)) for u in range(t, t + 40)}}))
            assert trunc == full == noisy, f"cross-machine future leak in {ep}"
        print("cross-machine episode pooling leak test passed")
    assert len(RULES) == N_RULES == 37
    print(f"LEAK TESTS PASSED: {checked} truncation/shuffle checks, lag-exactness, missing!=zero, kit future guard, determinism")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "runs/exp1/model.json")
