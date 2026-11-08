"""EXP-1 hyperparameter search with episode-grouped 5-fold CV on TRAIN pseudo-queries only.

Loss = realized 5-class Brier, aggregated like the official metric (episode -> condition -> mean).
E[Brier] = ||p-q||^2 + (1-||q||^2), so Brier differences estimate official-error differences.
Validation targets are never read here.
"""
import argparse
import itertools
import time
from pathlib import Path

import numpy as np
from scipy.special import gammaln

from common import (CONDITIONS, TYPES, Snapshots, brier, condition_loss, episode_cv_folds,
                    load_split, pooled_recent_snapshots, write_json)
from house_edge.history import SIZES
from house_edge.rules import N_RULES

H_GRID = [6, 8, 10, 12, 16, 24, 32, None]
A_GRID = [0.25, 0.5, 1, 2, 4]
B_GRID = [0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
K_GRID = [25, 50, 100, 200]


def machine_stats(snaps):
    """Per machine: (type index, wins, observed ticks, size counts) over its whole pre-cutoff history."""
    stats = {}
    for qid, (q, payouts) in snaps.machines.items():
        sc = [0, 0, 0, 0]
        for v in payouts.values():
            if v > 0:
                sc[SIZES.index(v)] += 1
        stats[qid] = (TYPES.index(q.machine_type), sum(v > 0 for v in payouts.values()), len(payouts), sc)
    return stats


def type_priors(stats, qids):
    wins = np.zeros(3); ticks = np.zeros(3); sizes = np.zeros((3, 4))
    for qid in qids:
        t, w, n, sc = stats[qid]
        wins[t] += w; ticks[t] += n; sizes[t] += sc
    mu = (wins + 1.0) / (ticks + 2.0)
    size = (sizes + 0.5) / (sizes.sum(1, keepdims=True) + 2.0)
    return mu, size


def rule_arrays(N, K, cell_index, mu_row, alpha):
    a0 = alpha * mu_row[:, None]
    b0 = alpha * (1 - mu_row)[:, None]
    const = gammaln(alpha) - gammaln(a0) - gammaln(b0)
    term = np.where(N > 0, const + gammaln(a0 + K) + gammaln(b0 + N - K) - gammaln(alpha + N), 0.0)
    logml = term.reshape(len(N), N_RULES, 2).sum(2)
    nc = np.take_along_axis(N, cell_index, 1)
    kc = np.take_along_axis(K, cell_index, 1)
    return logml, (a0 + kc) / (alpha + nc)


def softmax_rows(s):
    s = s - s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def eb_prior(logml_real, beta, smoothing=1.0, iterations=20):
    prior = np.full(N_RULES, 1.0 / N_RULES)
    for _ in range(iterations):
        post = softmax_rows(np.log(prior)[None, :] + beta * logml_real)
        mass = smoothing + post.sum(0)
        prior = mass / mass.sum()
    return np.log(prior)


def agg_weights(snaps, mask):
    """Row weights reproducing condition_loss as a dot product (mean in episode -> condition -> mean)."""
    w = np.zeros(len(mask))
    conds = [c for c in CONDITIONS if (mask & (snaps.condition == c)).any()]
    for c in conds:
        m = mask & (snaps.condition == c)
        eps = np.unique(snaps.episode[m])
        for e in eps:
            me = m & (snaps.episode == e)
            w[me] = 1.0 / (len(conds) * len(eps) * me.sum())
    return w


def five(w, size):
    w = np.clip(w, 1e-6, 1 - 1e-6)
    P = np.column_stack([1 - w, w[:, None] * size])
    return P / P.sum(1, keepdims=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/exp1_cv")
    p.add_argument("--k-back", type=int, default=16)
    out = Path(p.parse_args().out)
    if out.exists():
        raise SystemExit("Choose a new output folder.")
    t0 = time.perf_counter()
    q, h, labels, cond = load_split("train")
    snaps = Snapshots(q, h, labels, cond, k_back=p.parse_args().k_back)
    fold_of_ep = episode_cv_folds(q, cond)
    fold = np.array([fold_of_ep[e] for e in snaps.episode])
    stats = machine_stats(snaps)
    base = brier(pooled_recent_snapshots(q, h, labels, snaps), snaps)
    print(f"snapshots={len(snaps.rows)} machines={len(snaps.machines)} build={time.perf_counter()-t0:.1f}s")

    settings = list(range(5)) + ["all"]
    fold_train = {f: (fold != f) if f != "all" else np.ones(len(fold), bool) for f in settings}
    # per-setting type priors from the training part only
    priors = {}
    for f in settings:
        qids = sorted({snaps.rows[i][0] for i in np.flatnonzero(fold_train[f])})
        priors[f] = type_priors(stats, qids)

    W_in = {f: agg_weights(snaps, fold_train[f]) for f in settings}
    W_out = {f: agg_weights(snaps, fold == f) for f in range(5)}

    def rows_for(hl, alpha, f):
        N, K = snaps.counts(hl)
        mu, size_p = priors[f]
        logml, pred = rule_arrays(N, K, snaps.cell_index, mu[snaps.mtype], alpha)
        return logml, pred, size_p

    def losses(logml, pred, size_p, f, beta, kappa):
        log_prior = eb_prior(logml[fold_train[f] & snaps.real], beta)
        w = (softmax_rows(log_prior[None, :] + beta * logml) * pred).sum(1)
        size = (kappa * size_p[snaps.mtype] + snaps.size_counts) / (kappa + snaps.size_counts.sum(1))[:, None]
        return brier(five(w, size), snaps)

    results = {}  # (h, alpha, beta, kappa) -> {setting: (in_loss, held_out_loss)}
    for hl in H_GRID:
        for alpha in A_GRID:
            for f in settings:
                logml, pred, size_p = rows_for(hl, alpha, f)
                train_real = fold_train[f] & snaps.real
                for beta in B_GRID:
                    log_prior = eb_prior(logml[train_real], beta)
                    w = (softmax_rows(log_prior[None, :] + beta * logml) * pred).sum(1)
                    for kappa in K_GRID:
                        size = (kappa * size_p[snaps.mtype] + snaps.size_counts) / (kappa + snaps.size_counts.sum(1))[:, None]
                        loss = brier(five(w, size), snaps)
                        results.setdefault((hl, alpha, beta, kappa), {})[f] = (
                            float(W_in[f] @ loss), float(W_out[f] @ loss) if f != "all" else None)
        print(f"h={hl} done {time.perf_counter()-t0:.0f}s", flush=True)

    combos = list(results)
    cv = {c: (float(np.mean([results[c][f][1] for f in range(5)])), [results[c][f][1] for f in range(5)]) for c in combos}
    base_fold = [float(W_out[f] @ base) for f in range(5)]
    nested = []
    for f in range(5):
        best = min(combos, key=lambda c: (results[c][f][0], str(c)))
        nested.append({"fold": f, "chosen": best, "in_fold": results[best][f][0], "held_out": results[best][f][1],
                       "baseline_held_out": base_fold[f], "baseline_in_fold": float(W_in[f] @ base)})
    chosen = min(combos, key=lambda c: (cv[c][0], str(c)))
    hl, alpha, beta, kappa = chosen
    oof = np.zeros(len(fold))
    for f in range(5):
        logml, pred, size_p = rows_for(hl, alpha, f)
        oof[fold == f] = losses(logml, pred, size_p, f, beta, kappa)[fold == f]
    logml, pred, size_p = rows_for(hl, alpha, "all")
    in_rows = losses(logml, pred, size_p, "all", beta, kappa)
    everything = np.ones(len(fold), bool)
    oof_cond = condition_loss(oof, snaps, everything)
    base_cond = condition_loss(base, snaps, everything)
    in_all = condition_loss(in_rows, snaps, everything)
    rng = np.random.default_rng(314159)
    eps_by_c = {c: np.unique(snaps.episode[snaps.condition == c]) for c in CONDITIONS}
    delta = oof - base
    ep_delta = {e: delta[snaps.episode == e].mean() for e in np.unique(snaps.episode)}
    boots = []
    for _ in range(2000):
        boots.append(np.mean([np.mean([ep_delta[e] for e in rng.choice(eps_by_c[c], len(eps_by_c[c]))]) for c in CONDITIONS]))
    report = {
        "n_snapshots": len(snaps.rows), "k_back": p.parse_args().k_back,
        "chosen": {"half_life": chosen[0], "alpha": chosen[1], "beta": chosen[2], "size_strength": chosen[3]},
        "cv_mean_heldout": cv[chosen][0], "cv_per_fold": cv[chosen][1],
        "baseline_cv_per_fold": base_fold, "baseline_mean": float(np.mean(base_fold)),
        "oof_by_condition": oof_cond, "baseline_by_condition": base_cond,
        "in_sample_all_train_by_condition": in_all,
        "oof_delta_vs_pooled_recent": condition_loss(delta, snaps, everything)["mean"],
        "oof_delta_by_condition": condition_loss(delta, snaps, everything),
        "oof_delta_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "nested": [{**n, "chosen": list(n["chosen"])} for n in nested],
        "top10": [{"combo": list(c), "cv": cv[c][0], "fold_sd": float(np.std(cv[c][1]))}
                  for c in sorted(combos, key=lambda c: cv[c][0])[:10]],
        "seconds": time.perf_counter() - t0,
    }
    write_json(out / "cv.json", report)
    print(json_brief(report))


def json_brief(r):
    lines = [f"chosen={r['chosen']}  CV held-out Brier={r['cv_mean_heldout']:.5f} (folds {np.round(r['cv_per_fold'],4).tolist()})",
             f"baseline pooled_recent CV Brier={r['baseline_mean']:.5f} (folds {np.round(r['baseline_cv_per_fold'],4).tolist()})",
             f"OOF delta vs pooled_recent={r['oof_delta_vs_pooled_recent']:.5f} CI95={np.round(r['oof_delta_ci95'],5).tolist()}",
             "OOF by condition: " + str({k: round(v, 4) for k, v in r['oof_by_condition'].items()}),
             "baseline by cond: " + str({k: round(v, 4) for k, v in r['baseline_by_condition'].items()}),
             "in-sample (all train) by cond: " + str({k: round(v, 4) for k, v in r['in_sample_all_train_by_condition'].items()}),
             "nested (select on 4 folds, score 5th):"]
    for n in r["nested"]:
        lines.append(f"  fold {n['fold']}: chosen={n['chosen']} in={n['in_fold']:.5f} held-out={n['held_out']:.5f} "
                     f"baseline held-out={n['baseline_held_out']:.5f} gap={n['held_out']-n['in_fold']:+.5f}")
    lines.append("top combos (h, alpha, beta, kappa) by CV: " + "; ".join(f"{t['combo']}={t['cv']:.5f}±{t['fold_sd']:.4f}" for t in r["top10"][:6]))
    lines.append(f"runtime {r['seconds']:.0f}s")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
