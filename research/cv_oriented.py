"""EXP-4 CV (TRAIN only): oriented hierarchical cell priors for the change-point model.

Each non-null rule gets an orientation o (which cell is the 'high' cell, prior 1/2). The high cell has prior
Beta(s*m_hi[T], s*(1-m_hi[T])), the low cell Beta(s*m_lo[T], ...). Pre-change orientation is marginalised
(a change may flip it); the post-change orientation predicts. m_hi/m_lo per type are moment estimates from the
training machines' clearest full-history XOR rule (observable contrast > 0.5; no condition labels).
Compared against EXP-2 (+EXP-3 size/calibration fixed) with identical folds.
"""
import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import gammaln, logsumexp

from common import CONDITIONS, TYPES, Snapshots, brier, condition_loss, episode_cv_folds, load_split
from cv_change import ChangeTables, eb_prior_change, log_offset_prior, win_probs
from cv_pair_rule import agg_weights, five, machine_stats, type_priors
from house_edge.rules import N_RULES, PAIRS

ROOT = Path(__file__).resolve().parents[1]
C2 = json.loads((ROOT / "configs/exp2.json").read_text())
import sys
S_GRID, B_GRID = [2.0, 5.0, 10.0, 20.0], [1.0]


def hi_lo_levels(snaps, qids):
    """Per type: mean high/low cell win rate of training machines whose best full-history XOR contrast > 0.5."""
    acc = {t: [] for t in range(3)}
    for qid in qids:
        q, S = snaps.machines[qid]
        W = {t: int(v > 0) for t, v in S.items()}
        ticks = [t for t in S if all(t - k in S for k in range(1, 9))]
        best = None
        for a, b in PAIRS:
            x = np.array([W[t - a] ^ W[t - b] for t in ticks]); y = np.array([W[t] for t in ticks])
            if x.sum() < 5 or (1 - x).sum() < 5:
                continue
            r1, r0 = y[x == 1].mean(), y[x == 0].mean()
            if best is None or abs(r1 - r0) > best[0]:
                best = (abs(r1 - r0), max(r1, r0), min(r1, r0))
        if best and best[0] > 0.5:
            acc[TYPES.index(q.machine_type)].append(best[1:])
    return np.array([np.mean(acc[t], 0) for t in range(3)])  # (3, 2): hi, lo


def oriented_tables(tables, snaps, mu_row, alpha, hl_row, s, chunk=512):
    """Like ChangeTables.build but hypotheses = null + (rule, post orientation): 1 + 2*(N_RULES-1)."""
    S, nD = len(snaps.rows), len(tables.D)
    H = 1 + 2 * (N_RULES - 1)
    LM = np.empty((S, nD + 1, H)); PR = np.empty_like(LM)

    def cell_ml(N, K, a, b):
        return np.where(N > 0, gammaln(a + b) - gammaln(a) - gammaln(b) + gammaln(a + K) + gammaln(b + N - K) - gammaln(a + b + N), 0.0)

    for r0 in range(0, S, chunk):
        r1 = min(S, r0 + chunk)
        t = snaps.tick[r0:r1]; bse = tables.row_base[r0:r1]
        Nall, Kall = tables.cumN[bse + t], tables.cumK[bse + t]
        X = t[:, None] - tables.D[None, :]
        idx = bse[:, None] + np.clip(X, 0, None)
        valid = (X >= 0) & (tables.cumE[idx] >= tables.min_pre)
        Npre, Kpre = tables.cumN[idx], tables.cumK[idx]
        # stack "no change" as offset 0: pre empty, post = all
        Npost = np.concatenate([Nall[:, None], Nall[:, None] - Npre], 1)
        Kpost = np.concatenate([Kall[:, None], Kall[:, None] - Kpre], 1)
        Npr = np.concatenate([np.zeros_like(Nall)[:, None], Npre], 1)
        Kpr = np.concatenate([np.zeros_like(Kall)[:, None], Kpre], 1)
        n = r1 - r0
        hi = hl_row[r0:r1, 0][:, None, None]; lo = hl_row[r0:r1, 1][:, None, None]
        mu = mu_row[r0:r1][:, None, None]
        # reshape counts to (n, D+1, R, 2)
        sh = (n, nD + 1, N_RULES, 2)
        Np, Kp, Nq, Kq = Npost.reshape(sh), Kpost.reshape(sh), Npr.reshape(sh), Kpr.reshape(sh)
        cell = snaps.cell_index[r0:r1] - 2 * np.arange(N_RULES)[None, :]  # current cell 0/1 per rule
        out_ml = np.empty((n, nD + 1, H)); out_pr = np.empty_like(out_ml)
        # null rule (rule 0): symmetric prior alpha*mu on its single cell
        a0, b0 = alpha * mu[..., 0], alpha * (1 - mu[..., 0])
        out_ml[:, :, 0] = cell_ml(Np[:, :, 0, 0], Kp[:, :, 0, 0], a0, b0) + cell_ml(Nq[:, :, 0, 0], Kq[:, :, 0, 0], a0, b0)
        out_pr[:, :, 0] = (a0 + Kp[:, :, 0, 0]) / (alpha + Np[:, :, 0, 0])
        for o in (0, 1):
            # orientation o: cell o is high
            ah = s * hi[..., 0][..., None]; bh = s * (1 - hi[..., 0])[..., None]
            al = s * lo[..., 0][..., None]; bl = s * (1 - lo[..., 0])[..., None]
            A = [ah, al] if o == 0 else [al, ah]
            B = [bh, bl] if o == 0 else [bl, bh]
            post_ml = sum(cell_ml(Np[:, :, 1:, c], Kp[:, :, 1:, c], A[c], B[c]) for c in (0, 1))
            if o == 0:
                pre_ml0 = sum(cell_ml(Nq[:, :, 1:, c], Kq[:, :, 1:, c], A[c], B[c]) for c in (0, 1))
            else:
                pre_ml1 = sum(cell_ml(Nq[:, :, 1:, c], Kq[:, :, 1:, c], A[c], B[c]) for c in (0, 1))
            cur = cell[:, None, 1:]
            Ncur = np.take_along_axis(Np[:, :, 1:, :], np.broadcast_to(cur[..., None], (n, nD + 1, N_RULES - 1, 1)), 3)[..., 0]
            Kcur = np.take_along_axis(Kp[:, :, 1:, :], np.broadcast_to(cur[..., None], (n, nD + 1, N_RULES - 1, 1)), 3)[..., 0]
            a_cur = np.where(cur == o, ah, al); b_cur = np.where(cur == o, bh, bl)
            out_pr[:, :, 1 + o * (N_RULES - 1):1 + (o + 1) * (N_RULES - 1)] = (a_cur + Kcur) / (a_cur + b_cur + Ncur)
            out_ml[:, :, 1 + o * (N_RULES - 1):1 + (o + 1) * (N_RULES - 1)] = post_ml + np.log(0.5)
        pre_marg = np.logaddexp(pre_ml0, pre_ml1) + np.log(0.5)
        pre_marg[:, 0] = 0.0  # no change: empty pre segment
        out_ml[:, :, 1:] += np.concatenate([pre_marg, pre_marg], 2)
        # invalid offsets reuse "no change"
        v = np.concatenate([np.ones((n, 1), bool), valid], 1)[:, :, None]
        out_ml = np.where(v, out_ml, out_ml[:, :1]); out_pr = np.where(v, out_pr, out_pr[:, :1])
        LM[r0:r1], PR[r0:r1] = out_ml, out_pr
    return LM, PR


def expand_prior(lp_rules):
    """Rule-level EB prior -> hypothesis prior (orientation mass is already inside the likelihood)."""
    return np.concatenate([lp_rules[:1], lp_rules[1:], lp_rules[1:]])


def main():
    global S_GRID, B_GRID
    if len(sys.argv) > 1:
        S_GRID = [float(x) for x in sys.argv[1].split(",")]
    if len(sys.argv) > 2:
        B_GRID = [float(x) for x in sys.argv[2].split(",")]
    t0 = time.perf_counter()
    q, h, labels, cond = load_split("train")
    snaps = Snapshots(q, h, labels, cond, k_back=16)
    tables = ChangeTables(snaps)
    fold = np.array([episode_cv_folds(q, cond)[e] for e in snaps.episode])
    _, group = np.unique([f"{e}|{t}" for e, t in zip(snaps.episode, snaps.tick)], return_inverse=True)
    stats = machine_stats(snaps)
    lpd = log_offset_prior(C2["no_change_prior"], len(tables.D))
    W = {f: agg_weights(snaps, fold == f) for f in range(5)}
    k = C2["size_strength"]
    base_fold, res = {}, {}
    for f in range(5):
        qids = sorted({snaps.rows[i][0] for i in np.flatnonzero(fold != f)})
        mu, size_p = type_priors(stats, qids)
        size = (k * size_p[snaps.mtype] + snaps.size_counts) / (k + snaps.size_counts.sum(1))[:, None]
        real = (fold != f) & snaps.real
        LM, PR = tables.build(mu[snaps.mtype], C2["alpha"])
        lp = eb_prior_change(LM[real], C2["beta"], lpd)
        w, _ = win_probs(LM, PR, lp, C2["beta"], lpd, C2["shared_prior"], group)
        base_fold[f] = brier(five(w, size), snaps)
        hl = hi_lo_levels(snaps, qids)
        if f == 0:
            print("hi/lo levels (fold 0 train):", {TYPES[i]: hl[i].round(3).tolist() for i in range(3)}, flush=True)
        for s, alpha_null in itertools.product(S_GRID, (0.5,)):
            LMo, PRo = oriented_tables(tables, snaps, mu[snaps.mtype], alpha_null, hl[snaps.mtype], s)
            # EB over rules: collapse orientation hypotheses into their rule for the prior update
            for beta in B_GRID:
                Hn = LMo.shape[2]
                rule_of = np.concatenate([[0], np.arange(1, N_RULES), np.arange(1, N_RULES)])
                prior = np.full(N_RULES, 1.0 / N_RULES)
                per_h = logsumexp(lpd[None, :, None] + beta * LMo[real], axis=1)  # (n, H)
                for _ in range(20):
                    sc = expand_prior(np.log(prior))[None, :] + per_h
                    post = np.exp(sc - logsumexp(sc, axis=1, keepdims=True))
                    mass = 1.0 + np.array([post[:, rule_of == r].sum() for r in range(N_RULES)])
                    prior = mass / mass.sum()
                wo, _ = win_probs(LMo, PRo, expand_prior(np.log(prior)), beta, lpd, C2["shared_prior"], group)
                res[(s, alpha_null, beta, f)] = brier(five(wo, size), snaps)
            del LMo, PRo
        print(f"fold {f} done {time.perf_counter()-t0:.0f}s", flush=True)
    every = np.ones(len(fold), bool)
    base_oof = np.zeros(len(fold))
    for f in range(5):
        base_oof[fold == f] = base_fold[f][fold == f]
    print("EXP-2 OOF:", {c: round(v, 5) for c, v in condition_loss(base_oof, snaps, every).items()})
    for key in sorted({kk[:3] for kk in res}):
        oof = np.zeros(len(fold))
        for f in range(5):
            oof[fold == f] = res[(*key, f)][fold == f]
        d = np.array([W[f] @ (res[(*key, f)] - base_fold[f]) for f in range(5)])
        cl = condition_loss(oof, snaps, every)
        print(f"oriented s={key[0]:5.1f} beta={key[2]:.2f}: OOF {cl['mean']:.5f} | delta vs EXP-2 per fold {np.round(d, 5).tolist()} (better {int((d < 0).sum())}/5) | "
              + " / ".join(f"{c[:6]} {cl[c]:.4f}" for c in CONDITIONS))


if __name__ == "__main__":
    main()
