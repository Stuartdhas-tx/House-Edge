"""Train-only research pilot: tie P(2,5,20 | payout>=2) across size regimes.

Reuses the EXP-5 research harness up to (but not including) its execution/report
section. Both candidates use uncalibrated, episode-held-out win predictions;
this avoids the known nested-calibration defect. No validation inputs are read.
This is a fixed-architecture pilot, not an unbiased estimate of model selection.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=root / "runs/review_size_tying")
    output = parser.parse_args().out
    if output.exists():
        raise SystemExit("Choose a fresh output directory; refusing to overwrite results.")
    source_path = root / "research/cv_exp5.py"
    source = source_path.read_text()
    marker = "folds = {f: fit(fold != f) for f in range(5)}"
    if source.count(marker) != 1:
        raise RuntimeError("EXP-5 harness changed; inspect it before reusing this pilot.")
    env = {"__file__": str(source_path), "__name__": "review_exp5_harness"}
    start = time.perf_counter()
    exec(compile(source.split(marker)[0], str(source_path), "exec"), env)
    original = env["win_and_masses"]
    captured = {}

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured["masses"] = result[1]
        return result

    env["win_and_masses"] = capture
    fold, snaps = env["fold"], env["snaps"]
    losses = {name: np.zeros(len(fold)) for name in ("original_uncalibrated", "tied_uncalibrated")}

    def tied_em(groups, a, b, iterations=100):
        # Two copies of the original 0.5 pseudocount for each shared big class.
        big = groups.sum(axis=(0, 1))[1:] + 1.0
        big /= big.sum()
        a = np.r_[a[0], (1 - a[0]) * big]
        b = np.r_[b[0], (1 - b[0]) * big]
        pi = 0.5
        for _ in range(iterations):
            l1 = np.log(pi) + groups[:, 1] @ np.log(a) + groups[:, 0] @ np.log(b)
            l0 = np.log(1 - pi) + groups[:, 0] @ np.log(a) + groups[:, 1] @ np.log(b)
            r = 1 / (1 + np.exp(np.clip(l0 - l1, -700, 700)))
            ac = (r[:, None] * groups[:, 1] + (1 - r)[:, None] * groups[:, 0]).sum(0)
            bc = (r[:, None] * groups[:, 0] + (1 - r)[:, None] * groups[:, 1]).sum(0)
            pa = (ac[0] + 0.5) / (ac.sum() + 2.0)
            pb = (bc[0] + 0.5) / (bc.sum() + 2.0)
            a, b = np.r_[pa, (1 - pa) * big], np.r_[pb, (1 - pb) * big]
            pi = float(np.clip(r.mean(), 0.05, 0.95))
        return (a, b) if a[0] >= b[0] else (b, a)

    for f in range(5):
        train, held = fold != f, fold == f
        w, _, baseline_size, _ = env["fit"](train)
        masses = captured["masses"]
        qids = sorted(set(snaps.qid[train]))
        size = np.zeros_like(baseline_size)
        for ti in range(3):
            stats = env["stats"]
            static = env["em_sizes"](np.array([stats[x][3] for x in qids if stats[x][0] == ti], float))
            grouped = env["grp_at_cutoff"]
            groups = np.array([grouped[x][1] for x in qids if grouped[x][0] == ti and grouped[x][1] is not None])
            a, b = tied_em(groups, static[0], static[1])
            m = held & (snaps.mtype == ti)
            lo = env["Dg"][m] @ (np.log(a) - np.log(b))
            orientation = 1 / (1 + np.exp(-np.clip(lo, -700, 700)))
            current = env["cur"][m]
            pa_lag = np.where(current < 0, 0.5, np.where(current == 1, orientation, 1 - orientation))
            pa = 0.5 * masses[m, 0] + (masses[m, 1:] * pa_lag[:, 1:]).sum(1)
            size[m] = pa[:, None] * a + (1 - pa)[:, None] * b
        losses["original_uncalibrated"][held] = env["brier"](env["five"](w, baseline_size), snaps)[held]
        losses["tied_uncalibrated"][held] = env["brier"](env["five"](w, size), snaps)[held]
        print(f"fold {f} completed in {time.perf_counter() - start:.1f}s", flush=True)

    every = np.ones(len(fold), bool)
    report = {"source_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "design": "Fixed candidates; five episode folds; no calibration; train data only; raw Brier aggregation",
              "results": {}}
    for name, loss in losses.items():
        report["results"][name] = {
            "all": env["condition_loss"](loss, snaps, every),
            "real": env["condition_loss"](loss, snaps, snaps.real),
            "fold_real": [env["condition_loss"](loss, snaps, (fold == f) & snaps.real)["mean"] for f in range(5)],
            "fold_all": [env["condition_loss"](loss, snaps, fold == f)["mean"] for f in range(5)]}
    delta = losses["tied_uncalibrated"] - losses["original_uncalibrated"]
    rng = np.random.default_rng(314159)
    report["delta_tied_minus_original"] = {}
    for name, mask in (("all", every), ("real", snaps.real)):
        condition_samples = []
        for c in env["CONDITIONS"]:
            m = mask & (snaps.condition == c)
            episodes = np.unique(snaps.episode[m])
            ep_loss = np.array([delta[m & (snaps.episode == e)].mean() for e in episodes])
            draws = rng.integers(len(ep_loss), size=(5000, len(ep_loss)))
            condition_samples.append(ep_loss[draws].mean(axis=1))
        samples = np.mean(condition_samples, axis=0)
        report["delta_tied_minus_original"][name] = {
            "by_condition": env["condition_loss"](delta, snaps, mask),
            "paired_episode_bootstrap_95": np.quantile(samples, [0.025, 0.975]).tolist()}
    report["seconds"] = time.perf_counter() - start
    output.mkdir(parents=True)
    (output / "cv.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
