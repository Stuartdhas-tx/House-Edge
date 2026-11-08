"""EXP-2 CV (TRAIN pseudo-queries only): change-point rules + episode shared-change pooling.

Numpy mirror of house_edge.change_point.ChangePointForecaster (checked by tests/test_equivalence_change.py).
"""
import argparse
import itertools
import time
from pathlib import Path

import numpy as np
from scipy.special import gammaln, logsumexp

from common import CONDITIONS, Snapshots, brier, condition_loss, episode_cv_folds, load_split, pooled_recent_snapshots, write_json
from cv_pair_rule import agg_weights, five, machine_stats, type_priors
from house_edge.rules import N_RULES

A_GRID = [0.5, 1.0, 2.0]
B_GRID = [0.5, 1.0, 1.5]
RHO_GRID = [0.5, 0.8, 0.95]
SIGMA_GRID = [0.0, 0.5]
KAPPA = 50.0


class ChangeTables:
    def __init__(self, snaps, d_min=8, d_max=200, d_step=4, min_pre=8):
        self.snaps = snaps
        self.D = np.arange(d_min, d_max + 1, d_step)
        self.min_pre = min_pre
        base, cumN, cumK, cumE, offset = {}, [], [], [], 0
        for qid, (q, _) in snaps.machines.items():
            c = q.cutoff
            n = np.zeros((c + 1, 2 * N_RULES)); k = np.zeros_like(n); e = np.zeros(c + 1)
            for u, win, idx in snaps.events[qid]:
                n[u + 1, list(idx)] += 1
                if win:
                    k[u + 1, list(idx)] += 1
                e[u + 1] += 1
            cumN.append(np.cumsum(n, 0)); cumK.append(np.cumsum(k, 0)); cumE.append(np.cumsum(e))
            base[qid] = offset
            offset += c + 1
        self.cumN, self.cumK, self.cumE = np.vstack(cumN), np.vstack(cumK), np.concatenate(cumE)
        self.row_base = np.array([base[q] for q in snaps.qid])

    def build(self, mu_row, alpha, chunk=512):
        s = self.snaps
        S, nD = len(s.rows), len(self.D)
        LM = np.empty((S, nD + 1, N_RULES)); PR = np.empty_like(LM)
        for r0 in range(0, S, chunk):
            r1 = min(S, r0 + chunk)
            t = s.tick[r0:r1]; b = self.row_base[r0:r1]
            a0 = alpha * mu_row[r0:r1]; b0 = alpha * (1 - mu_row[r0:r1])
            Nall, Kall = self.cumN[b + t], self.cumK[b + t]
            X = t[:, None] - self.D[None, :]
            idx = b[:, None] + np.clip(X, 0, None)
            valid = (X >= 0) & (self.cumE[idx] >= self.min_pre)
            Npre, Kpre = self.cumN[idx], self.cumK[idx]
            Npost, Kpost = Nall[:, None, :] - Npre, Kall[:, None, :] - Kpre
            cell = s.cell_index[r0:r1]

            def lml(N, K, a, bb):
                const = gammaln(alpha) - gammaln(a) - gammaln(bb)
                term = np.where(N > 0, const + gammaln(a + K) + gammaln(bb + N - K) - gammaln(alpha + N), 0.0)
                return term.reshape(*N.shape[:-1], N_RULES, 2).sum(-1)

            A2, B2 = a0[:, None], b0[:, None]
            none_ml = lml(Nall, Kall, A2, B2)
            none_pr = (A2 + np.take_along_axis(Kall, cell, 1)) / (alpha + np.take_along_axis(Nall, cell, 1))
            A3, B3 = a0[:, None, None], b0[:, None, None]
            d_ml = lml(Npre, Kpre, A3, B3) + lml(Npost, Kpost, A3, B3)
            c3 = np.broadcast_to(cell[:, None, :], (r1 - r0, nD, N_RULES))
            d_pr = (A3 + np.take_along_axis(Kpost, c3, 2)) / (alpha + np.take_along_axis(Npost, c3, 2))
            d_ml = np.where(valid[:, :, None], d_ml, none_ml[:, None, :])
            d_pr = np.where(valid[:, :, None], d_pr, none_pr[:, None, :])
            LM[r0:r1, 0], PR[r0:r1, 0] = none_ml, none_pr
            LM[r0:r1, 1:], PR[r0:r1, 1:] = d_ml, d_pr
        return LM, PR


def log_offset_prior(rho, n):
    return np.array([np.log(rho)] + [np.log((1 - rho) / n)] * n)


def eb_prior_change(LM_real, beta, log_p_d, smoothing=1.0, iterations=20):
    per_rule = logsumexp(log_p_d[None, :, None] + beta * LM_real, axis=1)  # (n, R)
    prior = np.full(N_RULES, 1.0 / N_RULES)
    for _ in range(iterations):
        sc = np.log(prior)[None, :] + per_rule
        post = np.exp(sc - logsumexp(sc, axis=1, keepdims=True))
        mass = smoothing + post.sum(0)
        prior = mass / mass.sum()
    return np.log(prior)


def win_probs(LM, PR, log_prior, beta, log_p_d, sigma, group):
    scores = log_prior[None, None, :] + beta * LM
    Lm = logsumexp(scores, axis=2)                                   # (S, nD+1)
    cond = (np.exp(scores - Lm[:, :, None]) * PR).sum(2)             # (S, nD+1)
    indep = log_p_d[None, :] + Lm
    G = group.max() + 1
    sums = np.zeros((G, Lm.shape[1])); np.add.at(sums, group, Lm)
    sizes = np.bincount(group, minlength=G)
    shared = log_p_d[None, :] + sums                                 # (G, nD+1)
    if sigma > 0:
        log_zs = np.log(sigma) + logsumexp(shared, axis=1)
        zi_rows = logsumexp(indep, axis=1)
        log_zi = np.log(1 - sigma) + np.bincount(group, weights=zi_rows, minlength=G)
        p_sh = 1.0 / (1.0 + np.exp(np.minimum(700.0, log_zi - log_zs)))
        p_sh[sizes <= 1] = 0.0
    else:
        p_sh = np.zeros(G)
    sp = np.exp(shared - logsumexp(shared, axis=1, keepdims=True))[group]
    ip = np.exp(indep - logsumexp(indep, axis=1, keepdims=True))
    ps = p_sh[group][:, None]
    return ((ps * sp + (1 - ps) * ip) * cond).sum(1), p_sh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/exp2_cv")
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit("Choose a new output folder.")
    t0 = time.perf_counter()
    q, h, labels, cond = load_split("train")
    snaps = Snapshots(q, h, labels, cond, k_back=16)
    tables = ChangeTables(snaps)
    fold_of_ep = episode_cv_folds(q, cond)
    fold = np.array([fold_of_ep[e] for e in snaps.episode])
    keys = [f"{e}|{t}" for e, t in zip(snaps.episode, snaps.tick)]
    _, group = np.unique(keys, return_inverse=True)
    stats = machine_stats(snaps)
    base = brier(pooled_recent_snapshots(q, h, labels, snaps), snaps)
    settings = list(range(5)) + ["all"]
    fold_train = {f: (fold != f) if f != "all" else np.ones(len(fold), bool) for f in settings}
    priors = {f: type_priors(stats, sorted({snaps.rows[i][0] for i in np.flatnonzero(fold_train[f])})) for f in settings}
    W_in = {f: agg_weights(snaps, fold_train[f]) for f in settings}
    W_out = {f: agg_weights(snaps, fold == f) for f in range(5)}
    nD = len(tables.D)
    results, rows_cache = {}, {}
    for alpha in A_GRID:
        for f in settings:
            mu, size_p = priors[f]
            LM, PR = tables.build(mu[snaps.mtype], alpha)
            size = (KAPPA * size_p[snaps.mtype] + snaps.size_counts) / (KAPPA + snaps.size_counts.sum(1))[:, None]
            real = fold_train[f] & snaps.real
            for beta, rho, sigma in itertools.product(B_GRID, RHO_GRID, SIGMA_GRID):
                lpd = log_offset_prior(rho, nD)
                lp = eb_prior_change(LM[real], beta, lpd)
                w, p_sh = win_probs(LM, PR, lp, beta, lpd, sigma, group)
                loss = brier(five(w, size), snaps)
                key = (alpha, beta, rho, sigma)
                results.setdefault(key, {})[f] = (float(W_in[f] @ loss), float(W_out[f] @ loss) if f != "all" else None)
                rows_cache[(key, f)] = loss
            del LM, PR
        print(f"alpha={alpha} done {time.perf_counter()-t0:.0f}s", flush=True)
    combos = list(results)
    cv = {c: (float(np.mean([results[c][f][1] for f in range(5)])), [results[c][f][1] for f in range(5)]) for c in combos}
    chosen = min(combos, key=lambda c: (cv[c][0], str(c)))
    nested = []
    for f in range(5):
        best = min(combos, key=lambda c: (results[c][f][0], str(c)))
        nested.append({"fold": f, "chosen": list(best), "in_fold": results[best][f][0], "held_out": results[best][f][1]})
    oof = np.zeros(len(fold))
    for f in range(5):
        oof[fold == f] = rows_cache[(chosen, f)][fold == f]
    every = np.ones(len(fold), bool)
    report = {"chosen": dict(zip(("alpha", "beta", "no_change_prior", "shared_prior"), chosen)), "size_strength": KAPPA,
              "cv_mean_heldout": cv[chosen][0], "cv_per_fold": cv[chosen][1],
              "oof_by_condition": condition_loss(oof, snaps, every),
              "in_sample_by_condition": condition_loss(rows_cache[(chosen, "all")], snaps, every),
              "baseline_by_condition": condition_loss(base, snaps, every), "nested": nested,
              "top": [{"combo": list(c), "cv": cv[c][0], "sd": float(np.std(cv[c][1]))} for c in sorted(combos, key=lambda c: cv[c][0])[:8]],
              "best_no_pooling": min(([list(c), cv[c][0]] for c in combos if c[3] == 0.0), key=lambda x: x[1]),
              "seconds": time.perf_counter() - t0}
    write_json(out / "cv.json", report)
    for k in ("chosen", "cv_mean_heldout", "cv_per_fold", "oof_by_condition", "in_sample_by_condition", "best_no_pooling", "seconds"):
        print(k, report[k])
    for n in nested:
        print("  nested", n)
    for t in report["top"]:
        print("  top", t)


if __name__ == "__main__":
    main()
