"""Research-only helpers (numpy allowed here; never shipped in an entry).

Pseudo-queries: for each training machine with cutoff c we predict ticks t' in [c-K+1, c].
Every machine's inputs are truncated to ticks < t' (snapshots only read events before t'),
the label is the payout at t' (labels.csv for t' = c), and the query-time signal is never attached.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slot_forecasting.baseline import FrequencyForecaster  # noqa: E402
from slot_forecasting.experiments import episode_folds, subset_score  # noqa: E402
from slot_forecasting.io import load_inputs, read_labels, read_rows, read_submission, read_targets  # noqa: E402
from slot_forecasting.rounds import VERSION, round_metadata, score_round  # noqa: E402
from slot_forecasting.schema import CONDITIONS  # noqa: E402

from house_edge.history import SIZES, machine_payouts  # noqa: E402
from house_edge.rules import N_RULES, rule_cells, training_events  # noqa: E402
from house_edge.history import win_map  # noqa: E402

PUBLIC = ROOT / "public"
PAYOUTS = (0, 1, 2, 5, 20)
TYPES = ("balanced", "jackpot", "steady")


def load_split(split):
    queries, history = load_inputs(PUBLIC / split)
    labels = read_labels(PUBLIC / split / "labels.csv", queries) if (PUBLIC / split / "labels.csv").exists() else None
    cond_path = PUBLIC / split / "conditions.csv"
    conditions = {r["query_id"]: r["condition"] for r in read_rows(cond_path, ("query_id", "condition"))} if cond_path.exists() else None
    return queries, history, labels, conditions


def episode_cv_folds(queries, conditions, k=5):
    """Episode-grouped folds, stratified by condition (labels used for fold balance only)."""
    episodes = {}
    for q in queries:
        episodes[q.episode_id] = conditions[q.query_id]
    fold = {}
    for c in CONDITIONS:
        ordered = sorted((e for e, cc in episodes.items() if cc == c), key=lambda e: hashlib.sha256(e.encode()).hexdigest())
        for i, e in enumerate(ordered):
            fold[e] = i % k
    return fold


class Snapshots:
    """Pseudo-query table for one split: one row per (machine, pseudo-cutoff)."""

    def __init__(self, queries, history, labels, conditions, k_back=16):
        self.machines = machine_payouts(queries, history)
        rows, cells, sizes, ys = [], [], [], []
        self.events = {}
        for qid, (q, payouts) in self.machines.items():
            self.events[qid] = training_events(payouts)
            wins = win_map(payouts)
            for t in range(q.cutoff - k_back + 1, q.cutoff + 1):
                y = labels[qid] if t == q.cutoff else payouts.get(t)
                if y is None:
                    continue
                c = rule_cells(wins, t)
                if None in c:
                    continue
                rows.append((qid, q.episode_id, q.machine_type, conditions[qid] if conditions else "", t, t == q.cutoff))
                cells.append(c)
                sc = [0, 0, 0, 0]
                for tt, v in payouts.items():
                    if tt < t and v > 0:
                        sc[SIZES.index(v)] += 1
                sizes.append(sc)
                ys.append(PAYOUTS.index(y))
        self.rows = rows
        self.qid = np.array([r[0] for r in rows])
        self.episode = np.array([r[1] for r in rows])
        self.mtype = np.array([TYPES.index(r[2]) for r in rows])
        self.condition = np.array([r[3] for r in rows])
        self.tick = np.array([r[4] for r in rows])
        self.real = np.array([r[5] for r in rows])
        self.cells = np.array(cells, dtype=np.int64)
        self.size_counts = np.array(sizes, dtype=float)
        self.y = np.array(ys)
        self.Y = np.eye(5)[self.y]
        self.cell_index = 2 * np.arange(N_RULES)[None, :] + self.cells
        self._counts = {}

    def counts(self, half_life):
        """(N, K) arrays [rows, 2*N_RULES] of recency-weighted counts over events before each row's tick."""
        if half_life in self._counts:
            return self._counts[half_life]
        N = np.zeros((len(self.rows), 2 * N_RULES))
        K = np.zeros_like(N)
        by_machine = {}
        for i, r in enumerate(self.rows):
            by_machine.setdefault(r[0], []).append(i)
        for qid, idx in by_machine.items():
            idx = sorted(idx, key=lambda i: self.rows[i][4])
            events = self.events[qid]
            ref = self.rows[idx[-1]][4]
            n = np.zeros(2 * N_RULES)
            kk = np.zeros(2 * N_RULES)
            j = 0
            for i in idx:
                t = self.rows[i][4]
                while j < len(events) and events[j][0] < t:
                    u, win, ind = events[j]
                    w = 1.0 if half_life is None else 2.0 ** ((u - ref) / half_life)
                    ind = list(ind)
                    n[ind] += w
                    if win:
                        kk[ind] += w
                    j += 1
                s = 1.0 if half_life is None else 2.0 ** ((ref - t + 1) / half_life)
                N[i] = n * s
                K[i] = kk * s
        self._counts[half_life] = (N, K)
        return N, K


def condition_loss(per_row, snaps, mask):
    """Official-style aggregation: mean within episode, mean over episodes per condition, mean of conditions."""
    out = {}
    for c in CONDITIONS:
        m = mask & (snaps.condition == c)
        if not m.any():
            continue
        eps, inv = np.unique(snaps.episode[m], return_inverse=True)
        ep_mean = np.bincount(inv, weights=per_row[m]) / np.bincount(inv)
        out[c] = float(ep_mean.mean())
    out["mean"] = float(np.mean([out[c] for c in CONDITIONS if c in out]))
    return out


def brier(P, snaps):
    return ((P - snaps.Y) ** 2).sum(1)


def pooled_recent_snapshots(train_queries, train_history, train_labels, snaps):
    """Kit pooled_recent (strength 20, half-life 24) at every pseudo-cutoff; context prior fit on the train split."""
    model = FrequencyForecaster("pooled_recent").fit(train_queries, train_history, train_labels)
    out = np.zeros((len(snaps.rows), 5))
    for i, (qid, _, mtype, _, t, _) in enumerate(snaps.rows):
        q, payouts = snaps.machines[qid]
        prior = model.context_priors.get(f"{q.machine_type}|{q.signal}", model.global_prior)
        counts = [20.0 * p for p in prior]
        for tick, v in payouts.items():
            if tick < t:
                counts[PAYOUTS.index(v)] += 2 ** (-(t - 1 - tick) / 24.0)
        total = sum(counts)
        out[i] = [c / total for c in counts]
    return out


def score_validation_csv(csv_path):
    """Official technical points on all validation episodes plus the kit's tuning/audit halves."""
    full = score_round(csv_path, PUBLIC)
    queries, _, _, conditions = load_split("validation")
    targets = read_targets(PUBLIC / "validation/targets.csv")
    reference = read_submission(PUBLIC / "validation/reference.csv")
    preds = read_submission(csv_path, {q.query_id for q in queries})
    tuning, audit = episode_folds(targets, conditions)
    mp = round_metadata(PUBLIC)["max_points"]
    halves = {name: subset_score(preds, targets, conditions, reference, eps, mp)
              for name, eps in (("tuning", tuning), ("audit", audit))}

    def brief(r):
        return {"points": round(r["technical_points"], 4), "mean_ratio": round(r["mean_loss_ratio"], 4),
                **{c: round(r["by_condition"][c]["loss_ratio"], 4) for c in CONDITIONS}}
    return {"full": brief(full), "tuning": brief(halves["tuning"]), "audit": brief(halves["audit"])}


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
