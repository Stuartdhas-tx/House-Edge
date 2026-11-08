"""Portable feature equivalence and causal invariants; TRAIN inputs only.

Run: python tests/test_xgb_features_portable.py
The NumPy reference is used only in tests, never by the portable module.
"""

import csv
from dataclasses import replace
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from house_edge.history import machine_payouts
from house_edge.xgb_features_portable import build_row, feature_names
from research.xgb_features import build_features
from slot_forecasting.io import load_inputs
from slot_forecasting.schema import Query


def fixture():
    rng = random.Random(20261107)
    machines, rows = {}, []
    for episode in ("first", "second"):
        for m in range(8):
            q = Query(f"{episode}_{m}", episode, f"m{m}", ("balanced", "jackpot", "steady")[m % 3],
                      m % 3 - 1, 64)
            payouts = {u: rng.choice((0, 0, 1, 2, 5, 20)) for u in range(64) if rng.random() > 0.2}
            machines[q.query_id] = (q, payouts)
            rows.extend((q.query_id, episode, q.machine_type, "unused", t, t == 64) for t in (0, 7, 32, 64))
    return SimpleNamespace(machines=machines, rows=rows)


def portable_features(snaps):
    output = []
    for qid, episode, _, _, tick, _ in snaps.rows:
        query, own = snaps.machines[qid]
        peers = [history for key, (other, history) in snaps.machines.items()
                 if key != qid and other.episode_id == episode]
        output.append(build_row(query, own, peers, cutoff=tick))
    return np.asarray(output, dtype=np.float32)


class PortableFeatureTests(unittest.TestCase):
    def assert_reference_equal(self, snaps):
        expected, names = build_features(snaps)
        actual = portable_features(snaps)
        self.assertEqual(feature_names(), names)
        self.assertEqual(actual.shape, expected.shape)
        # Exact float32 agreement is stronger than the requested 1e-6 tolerance
        # and protects tree split decisions for these exercised observations.
        np.testing.assert_array_equal(actual, expected)
        self.assertFalse(np.isinf(actual).any())
        return actual, names

    def test_schema_and_synthetic_numpy_equivalence(self):
        self.assertEqual(len(feature_names()), 1259)
        self.assertEqual(len(set(feature_names())), 1259)
        self.assert_reference_equal(fixture())

    def test_train_real_and_pseudo_cutoffs_include_short_histories(self):
        queries, history = load_inputs(ROOT / "public/train")
        with (ROOT / "public/train/conditions.csv").open(newline="") as handle:
            conditions = {r["query_id"]: r["condition"] for r in csv.DictReader(handle)}
        chosen = {}
        for q in sorted(queries, key=lambda q: q.episode_id):
            episodes = chosen.setdefault(conditions[q.query_id], set())
            if len(episodes) < 2:
                episodes.add(q.episode_id)
        episodes = set().union(*chosen.values())
        machines = {qid: item for qid, item in machine_payouts(queries, history).items()
                    if item[0].episode_id in episodes}
        rows = [(qid, q.episode_id, q.machine_type, "unused", tick, tick == q.cutoff)
                for qid, (q, _) in machines.items() for tick in (q.cutoff - 11, q.cutoff)]
        self.assertEqual(len(machines), 64)
        self.assertTrue(any(len(p) == 64 for _, p in machines.values()))
        self.assertTrue(any(len(p) > 100 for _, p in machines.values()))
        self.assert_reference_equal(SimpleNamespace(machines=machines, rows=rows))

    def test_future_invariance_on_self_and_each_peer(self):
        snaps = fixture()
        qid = "first_0"
        q, own = snaps.machines[qid]
        peers = [p for key, (other, p) in snaps.machines.items() if key != qid and other.episode_id == q.episode_id]
        cutoff = 32
        short = [{u: v for u, v in p.items() if u < cutoff} for p in [own, *peers]]
        expected = np.asarray(build_row(q, short[0], short[1:], cutoff=cutoff))
        np.testing.assert_array_equal(expected, build_row(q, own, peers, cutoff=cutoff))
        rng = random.Random(314159)
        for changed_machine in range(len(short)):
            histories = [dict(p) for p in short]
            histories[changed_machine].update({u: rng.choice((0, 1, 2, 5, 20)) for u in range(cutoff, 100)})
            with self.subTest(changed_machine=changed_machine):
                np.testing.assert_array_equal(expected, build_row(q, histories[0], histories[1:], cutoff=cutoff))

    def test_exact_lag_missingness_signal_and_identifiers(self):
        q = Query("q", "e", "m", "balanced", 1, 64)
        own = {u: (0, 1, 2, 5, 20)[u % 5] for u in range(64)}
        del own[29]
        names = feature_names()
        row = build_row(q, own, [], cutoff=32)
        self.assertTrue(np.isnan(row[names.index("lag3_payout")]))
        self.assertEqual(row[names.index("lag3_missing")], 1.0)
        self.assertEqual(row[names.index("lag4_payout")], own[28])
        self.assertTrue(np.isnan(row[names.index("rule_xor1_3_cell")]))
        self.assertTrue(np.isnan(row[names.index("signal")]))
        renamed = replace(q, query_id="different", episode_id="other", machine_id="anonymous", signal=-1)
        np.testing.assert_array_equal(row, build_row(renamed, own, [], cutoff=32))
        self.assertEqual(build_row(q, own, [])[names.index("signal")], 1.0)
        own[29] = 0
        self.assertEqual(build_row(q, own, [], cutoff=32)[names.index("lag3_payout")], 0.0)

    def test_empty_histories_and_no_peers_match_reference(self):
        q = Query("q", "e", "m", "steady", 0, 20)
        for tick in (0, 3, 20):
            snaps = SimpleNamespace(machines={q.query_id: (q, {})},
                                    rows=[(q.query_id, q.episode_id, q.machine_type, "", tick, tick == 20)])
            with self.subTest(tick=tick):
                self.assert_reference_equal(snaps)

    def test_stdlib_only_in_fresh_process_without_site_packages(self):
        script = (
            "from house_edge.xgb_features_portable import build_row\n"
            "from slot_forecasting.schema import Query\n"
            "q=Query('q','e','m','steady',0,3)\n"
            "assert len(build_row(q,{0:0,1:2,2:1},[]))==1259\n"
        )
        result = subprocess.run([sys.executable, "-S", "-c", script], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
