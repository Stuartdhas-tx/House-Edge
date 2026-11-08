"""EXP-6 CV (TRAIN only): EXP-5 with oriented hi/lo cell levels per (type, signal) instead of per type.

The query signal is observed only at the real cutoff, so only real-cutoff rows use signal levels; earlier pseudo-query
rows keep type levels (signal is never attached to them). Decision metric: real-cutoff rows, paired per fold vs EXP-5.
Per fold f everything is refit without fold f, exactly as cv_exp5.py.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from common import CONDITIONS, TYPES, Snapshots, brier, condition_loss, episode_cv_folds, load_split, write_json
from cv_change import ChangeTables, log_offset_prior
from cv_exp3 import em_sizes
from cv_oriented import expand_prior, hi_lo_levels, oriented_tables
from cv_pair_rule import agg_weights, five, machine_stats, type_priors
from house_edge.rules import N_RULES, PAIRS
from house_edge.size_rule import first_lag

ROOT = Path(__file__).resolve().parents[1]
C4 = json.loads((ROOT / "configs/exp4.json").read_text())
SIZE_IDX = {1: 0, 2: 1, 5: 2, 20: 3}
t0 = time.perf_counter()
q, h, labels, cond = load_split("train")
snaps = Snapshots(q, h, labels, cond, k_back=16)
tables = ChangeTables(snaps)
fold = np.array([episode_cv_folds(q, cond)[e] for e in snaps.episode])
_, group = np.unique([f"{e}|{t}" for e, t in zip(snaps.episode, snaps.tick)], return_inverse=True)
stats = machine_stats(snaps)
lpd = log_offset_prior(C4["no_change_prior"], len(tables.D))
rule_of = np.concatenate([[0], np.arange(1, N_RULES), np.arange(1, N_RULES)])
first_of_rule = np.array([0] * 9 + [a for a, _ in PAIRS])          # rule -> first lag (0 = null/single)
first_of_h = first_of_rule[rule_of]
sc = snaps.size_counts
every = np.ones(len(fold), bool)
sig = np.array([snaps.machines[r[0]][0].signal + 1 for r in snaps.rows])  # 0, 1, 2 for signal -1, 0, 1


def hi_lo_by_signal(qids):
    """(3 types, 3 signals, 2): hi/lo levels from training machines with that query signal."""
    out = np.zeros((3, 3, 2))
    for s in range(3):
        out[:, s] = hi_lo_levels(snaps, [x for x in qids if snaps.machines[x][0].signal + 1 == s])
    return out


def em_cells(G, A, B, iters=100):
    A, B, pi = np.array(A, float), np.array(B, float), 0.5
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


def groups(S, t, a):
    g = np.zeros((2, 4))
    for u, v in S.items():
        if u < t and v > 0 and (u - a) in S:
            g[int(S[u - a] >= 2), SIZE_IDX[v]] += 1
    return g


# per-row cell evidence for every candidate first lag (independent of the fold)
Dg = np.zeros((len(fold), 8, 4)); cur = np.full((len(fold), 8), -1)
for i, (qid, _, _, _, t, _) in enumerate(snaps.rows):
    S = snaps.machines[qid][1]
    for a in range(1, 8):
        g = groups(S, t, a)
        Dg[i, a] = g[1] - g[0]
        if (t - a) in S:
            cur[i, a] = int(S[t - a] >= 2)
grp_at_cutoff = {}
for qid, (qq, S) in snaps.machines.items():
    a = first_lag(S, qq.cutoff)
    grp_at_cutoff[qid] = (TYPES.index(qq.machine_type), groups(S, qq.cutoff, a) if a is not None else None)
print(f"precompute {time.perf_counter()-t0:.0f}s", flush=True)


def eb_oriented(LMo_real, beta):
    per_h = logsumexp(lpd[None, :, None] + beta * LMo_real, axis=1)
    prior = np.full(N_RULES, 1.0 / N_RULES)
    for _ in range(C4["eb_iterations"]):
        s = expand_prior(np.log(prior))[None, :] + per_h
        post = np.exp(s - logsumexp(s, axis=1, keepdims=True))
        mass = C4["prior_smoothing"] + np.array([post[:, rule_of == r].sum() for r in range(N_RULES)])
        prior = mass / mass.sum()
    return np.log(prior)


def win_and_masses(LM, PR, log_prior, beta, sigma):
    scores = log_prior[None, None, :] + beta * LM
    Lm = logsumexp(scores, axis=2)
    post = np.exp(scores - Lm[:, :, None])
    condp = (post * PR).sum(2)
    indep = lpd[None, :] + Lm
    G = group.max() + 1
    sums = np.zeros((G, Lm.shape[1])); np.add.at(sums, group, Lm)
    shared = lpd[None, :] + sums
    log_zs = np.log(sigma) + logsumexp(shared, axis=1)
    log_zi = np.log(1 - sigma) + np.bincount(group, weights=logsumexp(indep, axis=1), minlength=G)
    p_sh = 1 / (1 + np.exp(np.minimum(700, log_zi - log_zs)))
    p_sh[np.bincount(group, minlength=G) <= 1] = 0
    pd = p_sh[group][:, None] * np.exp(shared - logsumexp(shared, axis=1, keepdims=True))[group] \
        + (1 - p_sh[group])[:, None] * np.exp(indep - logsumexp(indep, axis=1, keepdims=True))
    w = (pd * condp).sum(1)
    marg = np.einsum("sd,sdh->sh", pd, post)
    masses = np.zeros((len(w), 8))
    for a in range(8):
        masses[:, a] = marg[:, first_of_h == a].sum(1)
    return w, masses


cache = {}


def fit(train_mask, use_signal):
    key = (hashlib.sha256(np.packbits(train_mask).tobytes()).hexdigest(), use_signal)
    if key in cache:
        return cache[key]
    qids = sorted({snaps.rows[i][0] for i in np.flatnonzero(train_mask)})
    mu, _ = type_priors(stats, qids)
    hl_row = hi_lo_levels(snaps, qids)[snaps.mtype]
    if use_signal:
        hl_row[snaps.real] = hi_lo_by_signal(qids)[snaps.mtype, sig][snaps.real]
    LMo, PRo = oriented_tables(tables, snaps, mu[snaps.mtype], C4["alpha"], hl_row, C4["orient_strength"])
    lp = eb_oriented(LMo[train_mask & snaps.real], C4["beta"])
    w, masses = win_and_masses(LMo, PRo, expand_prior(lp), C4["beta"], C4["shared_prior"])
    del LMo, PRo
    static, rule = {}, {}
    for ti in range(3):
        static[ti] = em_sizes(np.array([stats[x][3] for x in qids if stats[x][0] == ti], float))
        G = np.array([grp_at_cutoff[x][1] for x in qids if grp_at_cutoff[x][0] == ti and grp_at_cutoff[x][1] is not None])
        rule[ti] = em_cells(G, static[ti][0], static[ti][1])
    size4 = np.zeros((len(fold), 4)); size5 = np.zeros((len(fold), 4))
    for ti in range(3):
        m = snaps.mtype == ti
        A, B, pi = static[ti]
        pa = 1 / (1 + np.exp(-(C4["regime_temper"] * (sc[m] @ (np.log(A) - np.log(B))) + np.log(pi / (1 - pi)))))
        size4[m] = pa[:, None] * A + (1 - pa)[:, None] * B
        A, B = rule[ti]
        lo = Dg[m] @ (np.log(A) - np.log(B))                          # (n, 8)
        p1 = 1 / (1 + np.exp(-np.clip(lo, -700, 700)))
        c = cur[m]
        pa_lag = np.where(c < 0, 0.5, np.where(c == 1, p1, 1 - p1))
        PA = 0.5 * masses[m, 0] + (masses[m, 1:] * pa_lag[:, 1:]).sum(1)
        size5[m] = PA[:, None] * A + (1 - PA)[:, None] * B
    cache[key] = (w, size4, size5, rule)
    return cache[key]


def calibrate(w, th):
    w = np.clip(w, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(th[0] * np.log(w / (1 - w)) + th[1 + snaps.mtype])))


def fit_theta(w_oof, size, mask):
    Wm = agg_weights(snaps, mask)
    return minimize(lambda th: float(Wm @ brier(five(calibrate(w_oof, th), size), snaps)), np.array([1.0, 0, 0, 0]),
                    method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 2000}).x


oof, theta_all, in_sample = {}, {}, {}
for name, use_signal in (("exp5", False), ("exp6", True)):
    folds = {f: fit(fold != f, use_signal) for f in range(5)}
    w_oof = np.zeros(len(fold))
    for f in range(5):
        w_oof[fold == f] = folds[f][0][fold == f]
    oof[name] = np.zeros(len(fold))
    for f in range(5):
        w, _, s5, _ = folds[f]
        held = fold == f
        oof[name][held] = brier(five(calibrate(w, fit_theta(w_oof, s5, fold != f)), s5), snaps)[held]
    w_all, _, s5_all, _ = fit(every, use_signal)
    theta_all[name] = fit_theta(w_oof, s5_all, every)
    in_sample[name] = condition_loss(brier(five(calibrate(w_all, theta_all[name]), s5_all), snaps), snaps, snaps.real)
    print(f"{name} fitted {time.perf_counter()-t0:.0f}s", flush=True)
rep = {}
for name, v in oof.items():
    rep[name] = {"all": condition_loss(v, snaps, every), "real": condition_loss(v, snaps, snaps.real),
                 "fold_all": [condition_loss(v, snaps, fold == f)["mean"] for f in range(5)],
                 "fold_real": [condition_loss(v, snaps, (fold == f) & snaps.real)["mean"] for f in range(5)],
                 "in_sample_real": in_sample[name], "theta_all_train": theta_all[name].tolist()}
rep["theta_all_train"] = theta_all["exp6"].tolist()
rep["signal_levels_all_train"] = {TYPES[t]: {str(s - 1): hl.tolist() for s, hl in enumerate(lv)}
                                  for t, lv in enumerate(hi_lo_by_signal(sorted(snaps.machines)))}
rep["seconds"] = time.perf_counter() - t0
write_json(ROOT / "runs/exp6_cv/cv.json", rep)
for name in oof:
    r = rep[name]
    print(f"{name} OOF all {r['all']['mean']:.5f} | real {r['real']['mean']:.5f} | real by condition "
          + " / ".join(f"{c[:6]} {r['real'][c]:.4f}" for c in CONDITIONS))
for kind in ("real", "all"):
    d = np.array(rep["exp6"][f"fold_{kind}"]) - np.array(rep["exp5"][f"fold_{kind}"])
    print(f"EXP-6 minus EXP-5 [{kind}] per fold {np.round(d, 5).tolist()} mean {d.mean():+.5f} better {int((d < 0).sum())}/5")
print("in-sample real EXP-6 vs OOF real:", round(rep["exp6"]["in_sample_real"]["mean"], 5), round(rep["exp6"]["real"]["mean"], 5))
print("theta EXP-6:", np.round(theta_all["exp6"], 4).tolist(), "| EXP-5:", np.round(theta_all["exp5"], 4).tolist())
print("signal levels (hi, lo):", {k: {s: np.round(x, 3).tolist() for s, x in v.items()} for k, v in rep["signal_levels_all_train"].items()})
print(f"runtime {rep['seconds']:.0f}s")
