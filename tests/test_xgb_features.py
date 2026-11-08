"""Synthetic invariants for tree features; never accesses validation data."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.xgb_features import build_features  # noqa: E402
from slot_forecasting.schema import Query  # noqa: E402


def fixture():
    machines, rows = {}, []
    for episode in ("ep_a", "ep_b"):
        for j in range(3):
            qid = f"{episode}_{j}"
            q = Query(qid, episode, f"machine{j}", ("balanced", "jackpot", "steady")[j], j - 1, 64)
            payouts = {t: (0, 1, 2, 5, 20)[(3 * t + j) % 5] for t in range(64)}
            machines[qid] = (q, payouts)
            rows.extend((qid, episode, q.machine_type, "unused_condition", t, t == 64) for t in (32, 64))
    return SimpleNamespace(machines=machines, rows=rows)


class FeatureTests(unittest.TestCase):
    def test_schema_determinism_and_no_label_access(self):
        s = fixture()
        a, names = build_features(s)
        b, other_names = build_features(s)
        self.assertEqual(a.dtype, np.float32)
        self.assertEqual(names, other_names)
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(a.shape, (len(s.rows), len(names)))
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.isinf(a).any())
        self.assertFalse(any("condition" in name or "query_id" in name or "episode_id" in name for name in names))

    def test_future_invariance_on_every_machine_and_peer(self):
        s = fixture()
        a, names = build_features(s)
        changed = fixture()
        changed.machines = {qid: (q, {t: (20 if value != 20 else 0) if t >= 32 else value
                                     for t, value in p.items()}) for qid, (q, p) in s.machines.items()}
        b, _ = build_features(changed)
        pseudo = np.array([row[4] == 32 for row in s.rows])
        np.testing.assert_array_equal(a[pseudo], b[pseudo])
        self.assertTrue(np.any(a[~pseudo] != b[~pseudo]))
        # Adding observations beyond every prediction cutoff is also immaterial.
        for q, payouts in changed.machines.values():
            payouts.update({t: 20 for t in range(64, 80)})
        c, _ = build_features(changed)
        np.testing.assert_array_equal(b, c)

    def test_missing_is_not_zero_and_lags_are_exact(self):
        s = fixture()
        qid = s.rows[0][0]
        del s.machines[qid][1][29]  # exactly lag 3 for cutoff 32
        a, names = build_features(s)
        self.assertTrue(np.isnan(a[0, names.index("lag3_payout")]))
        self.assertTrue(np.isnan(a[0, names.index("lag3_win")]))
        self.assertEqual(a[0, names.index("lag3_missing")], 1)
        self.assertEqual(a[0, names.index("lag4_payout")], s.machines[qid][1][28])
        self.assertTrue(np.isnan(a[0, names.index("rule_xor1_3_cell")]))
        unaffected = a[0, names.index("rule_xor1_4_cell")]
        s.machines[qid][1][29] = 0
        b, _ = build_features(s)
        self.assertEqual(b[0, names.index("lag3_win")], 0)
        self.assertEqual(b[0, names.index("lag3_missing")], 0)
        self.assertEqual(b[0, names.index("rule_xor1_4_cell")], unaffected)

    def test_signal_is_unknown_on_pseudo_rows(self):
        s = fixture()
        a, names = build_features(s)
        signal_column = names.index("signal")
        for i, row in enumerate(s.rows):
            if row[5]:
                self.assertEqual(a[i, signal_column], s.machines[row[0]][0].signal)
            else:
                self.assertTrue(np.isnan(a[i, signal_column]))
        changed = fixture()
        changed.machines = {qid: (replace(q, signal=-q.signal), p) for qid, (q, p) in s.machines.items()}
        b, _ = build_features(changed)
        pseudo = np.array([not row[5] for row in s.rows])
        np.testing.assert_array_equal(a[pseudo], b[pseudo])

    def test_conditions_and_identifier_values_do_not_enter_features(self):
        s = fixture()
        a, names = build_features(s)
        renamed = SimpleNamespace(machines={}, rows=[])
        for index, (qid, (q, p)) in enumerate(reversed(list(s.machines.items()))):
            new_id = f"anonymous{index}"
            renamed.machines[new_id] = (replace(q, query_id=new_id, episode_id="renamed_" + q.episode_id,
                                                machine_id=f"different{index}"), p)
        mapping = {q.episode_id.removeprefix("renamed_") + "|" + str(q.signal): key
                   for key, (q, _) in renamed.machines.items()}
        for qid, ep, mt, condition, tick, real in s.rows:
            new_id = mapping[ep + "|" + str(s.machines[qid][0].signal)]
            renamed.rows.append((new_id, "renamed_" + ep, mt, "changed_condition", tick, real))
        b, other_names = build_features(renamed)
        self.assertEqual(names, other_names)
        np.testing.assert_array_equal(a, b)

    def test_rule_counts_use_only_exact_available_lags(self):
        s = fixture()
        qid = s.rows[0][0]
        payouts = s.machines[qid][1]
        del payouts[10]
        a, names = build_features(s)
        current = int(payouts[31] > 0) ^ int(payouts[29] > 0)
        selected = [u for u in range(32) if u in payouts and u - 1 in payouts and u - 3 in payouts
                    and (int(payouts[u - 1] > 0) ^ int(payouts[u - 3] > 0)) == current]
        n = len(selected)
        wins = sum(payouts[u] > 0 for u in selected)
        self.assertEqual(a[0, names.index("rule_xor1_3_all_current_n")], n)
        self.assertEqual(a[0, names.index("rule_xor1_3_all_current_wins")], wins)
        self.assertAlmostEqual(a[0, names.index("rule_xor1_3_all_current_rate")], (wins + .5) / (n + 1))

    def test_peers_exclude_self_and_other_episodes(self):
        s = fixture()
        a, names = build_features(s)
        peer_names = [i for i, name in enumerate(names) if name.startswith("peer_")]
        qid = s.rows[0][0]
        q, _ = s.machines[qid]
        s.machines[qid] = (q, {t: 0 for t in range(64)})
        for key, (query, payouts) in list(s.machines.items()):
            if query.episode_id != q.episode_id:
                s.machines[key] = (query, {t: 20 for t in payouts})
        b, _ = build_features(s)
        np.testing.assert_array_equal(a[0, peer_names], b[0, peer_names])

    def test_empty_history_and_zero_cutoff(self):
        q = Query("q", "e", "m", "steady", 1, 0)
        s = SimpleNamespace(machines={"q": (q, {})}, rows=[("q", "e", "steady", "", 0, True)])
        a, names = build_features(s)
        self.assertEqual(a[0, names.index("observed_n")], 0)
        self.assertEqual(a[0, names.index("peer_all_machine_n")], 0)
        self.assertTrue(np.isnan(a[0, names.index("lag1_payout")]))
        self.assertFalse(np.isinf(a).any())


if __name__ == "__main__":
    unittest.main()
