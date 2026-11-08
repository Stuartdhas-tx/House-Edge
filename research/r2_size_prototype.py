"""R1b + R2 (TRAIN only).
R1b: which payout values at the size lag a split the regime? (mean size-LLR by payout value, oriented per machine)
R2:  prototype size model tied to the win pair's first lag, scored by 5-fold episode CV with EXP-4's OOF win
     probabilities held fixed. Components are re-fit by EM with the cell structure on training folds only.
"""
import json
import numpy as np
from common import Snapshots, brier, episode_cv_folds, load_split, TYPES
from cv_pair_rule import agg_weights, five, machine_stats
from house_edge.rules import PAIRS

q, h, labels, cond = load_split("train")
snaps = Snapshots(q, h, labels, cond, k_back=16)
fold = np.array([episode_cv_folds(q, cond)[e] for e in snaps.episode])
comp = json.load(open("../runs/exp4/model.json"))["size_components"]
SIZE_IDX = {1: 0, 2: 1, 5: 2, 20: 3}


def first_lag(S, t):
    """Win pair chosen from ticks < t, sign-agnostic across two halves (rules may flip); returns its first lag."""
    W = {u: int(v > 0) for u, v in S.items() if u < t}
    T = [u for u in W if all(u - k in W for k in range(1, 9))]
    if len(T) < 16:
        return None
    mid = T[len(T) // 2]
    best = None
    for a, b in PAIRS:
        score = 0.0
        for part in ([u for u in T if u < mid], [u for u in T if u >= mid]):
            x = np.array([W[u - a] ^ W[u - b] for u in part]); y = np.array([W[u] for u in part])
            if min(x.sum(), (1 - x).sum()) >= 3:
                score += abs(y[x == 1].mean() - y[x == 0].mean())
        if best is None or score > best[0]:
            best = (score, a)
    return best[1]


# ---- R1b: payout value at lag a vs size LLR (machines with clear rules, oriented by their own sign) ----
by_val = {v: [] for v in (0, 1, 2, 5, 20)}
for qid, (qq, S) in snaps.machines.items():
    if cond[qid] == "short_history":
        continue
    a = first_lag(S, qq.cutoff)
    A, B, _ = comp[qq.machine_type]
    U = [u for u in S if S[u] > 0 and (u - a) in S]
    L = np.array([np.log(A[SIZE_IDX[S[u]]] / B[SIZE_IDX[S[u]]]) for u in U])
    lagv = np.array([S[u - a] for u in U])
    big = lagv >= 2
    if big.sum() < 6 or (~big).sum() < 6:
        continue
    sgn = np.sign(L[big].mean() - L[~big].mean())
    for v in by_val:
        if (lagv == v).sum():
            by_val[v].append(sgn * L[lagv == v].mean())
print("R1b mean oriented size-LLR by payout value at the size lag (positive = 'big' side):")
print("   ", {v: round(float(np.mean(x)), 3) for v, x in by_val.items()}, "| n machines", len(by_val[0]))

# ---- R2 prototype ----
stats = machine_stats(snaps)
row_a = {}
for i, (qid, _, _, _, t, _) in enumerate(snaps.rows):
    row_a[i] = first_lag(snaps.machines[qid][1], t)


def groups_for(S, t, a):
    """Winning-size counts split by whether the payout a ticks earlier was >= 2 (both ticks observed, < t)."""
    g = np.zeros((2, 4))
    for u, v in S.items():
        if u < t and v > 0 and (u - a) in S:
            g[int(S[u - a] >= 2), SIZE_IDX[v]] += 1
    return g


def em_cells(G, A, B, iters=100):
    """G: (M, 2, 4) per-machine cell counts. Latent orientation: cell1->A,cell0->B (z=1) or swapped."""
    A, B, pi = np.array(A), np.array(B), 0.5
    for _ in range(iters):
        l1 = np.log(pi) + G[:, 1] @ np.log(A) + G[:, 0] @ np.log(B)
        l0 = np.log(1 - pi) + G[:, 0] @ np.log(A) + G[:, 1] @ np.log(B)
        r = 1 / (1 + np.exp(np.clip(l0 - l1, -700, 700)))
        A = (r[:, None] * G[:, 1] + (1 - r)[:, None] * G[:, 0]).sum(0) + 0.5; A /= A.sum()
        B = (r[:, None] * G[:, 0] + (1 - r)[:, None] * G[:, 1]).sum(0) + 0.5; B /= B.sum()
        pi = float(np.clip(r.mean(), 0.05, 0.95))
    if A[0] < B[0]:
        A, B = B, A
    return A, B


# w from EXP-4 OOF (reuse cv_exp4 machinery)
import cv_exp4 as E4  # builds its own snapshots (identical rows) at import
assert len(E4.snaps.rows) == len(snaps.rows)
w_oof = np.zeros(len(fold))
for f in range(5):
    w_oof[fold == f] = E4.fit_w(fold != f)[0][fold == f]
size_exp4 = {f: E4.fit_w(fold != f)[1] for f in range(5)}
W_all = {f: agg_weights(snaps, fold == f) for f in range(5)}
W_real = {f: agg_weights(snaps, (fold == f) & snaps.real) for f in range(5)}
res = {"exp4_size": [], "rule_size": [], "rule_size_t.5": []}
comps = {}
for f in range(5):
    train_m = sorted({snaps.rows[i][0] for i in np.flatnonzero(fold != f)})
    comps[f] = {}
    for ti, t in enumerate(TYPES):
        G = []
        for qid in train_m:
            qq, S = snaps.machines[qid]
            if qq.machine_type == t:
                a = first_lag(S, qq.cutoff)
                if a:
                    G.append(groups_for(S, qq.cutoff, a))
        A0, B0, _ = comp[t]
        comps[f][ti] = em_cells(np.array(G), A0, B0)
    for name, temper in (("rule_size", 1.0), ("rule_size_t.5", 0.5)):
        size = np.zeros((len(fold), 4))
        for i, (qid, _, mt, _, t, _) in enumerate(snaps.rows):
            A, B = comps[f][TYPES.index(mt)]
            S = snaps.machines[qid][1]
            a = row_a[i]
            if a is None or (t - a) not in S:
                size[i] = 0.5 * (A + B); continue
            g = groups_for(S, t, a)
            # log-odds that cell1 -> A (orientation), from the machine's own past wins
            lo = (g[1] @ np.log(A) + g[0] @ np.log(B)) - (g[0] @ np.log(A) + g[1] @ np.log(B))
            p1 = 1 / (1 + np.exp(-np.clip(temper * lo, -700, 700)))
            cur = int(S[t - a] >= 2)
            pa = p1 if cur == 1 else 1 - p1
            size[i] = pa * A + (1 - pa) * B
        P = five(w_oof, size)
        res[name].append((W_all[f] @ brier(P, snaps), W_real[f] @ brier(P, snaps)))
    P = five(w_oof, size_exp4[f])
    res["exp4_size"].append((W_all[f] @ brier(P, snaps), W_real[f] @ brier(P, snaps)))
print("\nR2 prototype (EXP-4 OOF win probabilities fixed; only the size model changes)")
base = np.array(res["exp4_size"])
for name, v in res.items():
    v = np.array(v); d = v - base
    print(f"   {name:14s} all-rows {v[:,0].mean():.5f} real-rows {v[:,1].mean():.5f} | delta vs EXP-4 all {d[:,0].mean():+.5f} "
          f"(folds better {int((d[:,0]<0).sum())}/5) real {d[:,1].mean():+.5f} (folds better {int((d[:,1]<0).sum())}/5)")
print("   EM components with cell structure (all-fold-0 train):", {TYPES[t]: [np.round(c, 3).tolist() for c in comps[0][t]] for t in range(3)})
