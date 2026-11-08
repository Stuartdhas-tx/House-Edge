"""Round two in one command. Run from the NEW kit folder after tools/adapt_setup.py has copied our code into it.

Builds, tests and verifies upload-ready entries in order of priority, printing READY lines as soon as each exists:
  a_baseline (organizer pooled_recent) -> a_exp5_maincfg (main settings) -> a_exp5_recal -> a_exp6 [-> a_exp4_recal]
then prints the pick. Logs go to round2_logs/, the summary to round2_report.json.

Pick rule, fixed before seeing round-two data (choose on TRAIN CV, validation only confirms):
  1. a_exp5_recal (EXP-5 with calibration refit on round-two train CV) by default.
  2. a_exp4_recal instead if train CV has EXP-5 beating EXP-4 on real-cutoff rows in fewer than 4/5 folds
     (the size rule did not carry over).
  3. a_exp6 instead only if train CV has EXP-6 beating EXP-5 in 5/5 folds on real-cutoff rows AND its paired
     validation gain over EXP-5 has a 95% episode-bootstrap interval above zero.
  4. The best earlier ready entry if the pick failed a check; a_baseline if every model scores below it.
Usage: python tools/round2_run.py [--allow-round main] [--python313 PATH]
"""
import argparse
import csv
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "round2_logs"
PY313 = Path.home() / ".local/share/uv/python/cpython-3.13.9-macos-aarch64-none/bin/python3.13"
PAYOUT_COLS = ("p_0", "p_1", "p_2", "p_5", "p_20")
T0 = time.time()


def say(msg):
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


def run(name, args, cwd=ROOT):
    """Run a python command; full output to round2_logs/<name>.log; raise with its tail on failure."""
    proc = subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True)
    (LOGS / f"{name}.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed:\n" + "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-8:]))
    return proc.stdout


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_text(base, config, round_name):
    """Entry README text: base description with this round's training split and calibration numbers."""
    text = (ROOT / "configs" / base).read_text()
    text = text.replace("main-round TRAIN split", f"{round_name}-round TRAIN split")
    c = json.loads((ROOT / "configs" / config).read_text())
    i = c["calibration_intercepts"]
    return re.sub(r"logit\(w'\) = [0-9.]+\*logit\(w\) \+ intercept\[type\] \([^)]*\)",
                  f"logit(w') = {c['calibration_slope']:.4f}*logit(w) + intercept[type] (balanced {i['balanced']:+.3f}, "
                  f"jackpot {i['jackpot']:+.3f}, steady {i['steady']:+.3f})", text)


def build_entry(name, config, text_base, tests, verify_python, round_name):
    """Fit + score on validation, run tests, package, verify a fresh-process reproduction. Returns a report row."""
    run(f"{name}_fit", ["research/run_model.py", "--config", f"configs/{config}", "--out", f"runs/{name}"])
    run(f"{name}_check", ["check_submission.py", f"runs/{name}/submission.csv"])
    for test in tests:
        run(f"{name}_{Path(test).stem}", [test, f"runs/{name}/model.json"])
    (ROOT / "configs" / f"{name}_text.md").write_text(model_text(text_base, config, round_name), encoding="utf-8")
    run(f"{name}_entry", ["tools/make_entry.py", "--run", f"runs/{name}", "--name", name,
                          "--model-text", (ROOT / "configs" / f"{name}_text.md").read_text()])
    out = run(f"{name}_verify", ["tools/verify_entry.py", f"zips/{name}.zip", f"runs/{name}/submission.csv",
                                 "--python", verify_python])
    if "REPRODUCTION OK" not in out:
        raise RuntimeError(f"{name}: reproduction not confirmed")
    val = json.loads((ROOT / f"runs/{name}/run.json").read_text())["validation"]
    return {"csv": f"runs/{name}/submission.csv", "zip": f"zips/{name}.zip", "csv_sha256": sha(ROOT / f"runs/{name}/submission.csv"),
            "zip_sha256": sha(ROOT / f"zips/{name}.zip"), "points": {h: val[h]["points"] for h in ("full", "tuning", "audit")},
            "ratios": {c: val["full"][c] for c in ("stable", "short_history", "drift", "shared_change")},
            "byte_identical": '"byte_identical": true' in out}


def episode_losses(csv_path):
    """Per-episode mean squared error vs targets and vs the frozen reference, with the condition."""
    pub = ROOT / "public/validation"
    read = lambda p: {r["query_id"]: r for r in csv.DictReader(open(p, newline=""))}
    pred, tgt, ref = read(csv_path), read(pub / "targets.csv"), read(pub / "reference.csv")
    cond = {r["query_id"]: r["condition"] for r in csv.DictReader(open(pub / "conditions.csv", newline=""))}
    ep = defaultdict(lambda: [None, [], []])
    for qid, t in tgt.items():
        e = ep[t["episode_id"]]
        e[0] = cond[qid]
        e[1].append(sum((float(pred[qid][c]) - float(t[c])) ** 2 for c in PAYOUT_COLS))
        e[2].append(sum((float(ref[qid][c]) - float(t[c])) ** 2 for c in PAYOUT_COLS))
    return {k: (v[0], sum(v[1]) / len(v[1]), sum(v[2]) / len(v[2])) for k, v in ep.items()}


def paired_bootstrap(csv_new, csv_old, max_points, reps=2000, seed=314159):
    """95% interval of the points gain new - old, resampling episodes within each condition."""
    new, old = episode_losses(csv_new), episode_losses(csv_old)
    by_cond = defaultdict(list)
    for e, (c, _, _) in new.items():
        by_cond[c].append(e)

    def points(losses, sample):
        acc = defaultdict(lambda: [0.0, 0.0])
        for e in sample:
            acc[new[e][0]][0] += losses[e][1]
            acc[new[e][0]][1] += losses[e][2]
        return max_points / (1 + sum(a / max(b, 1e-9) for a, b in acc.values()) / len(acc))

    rng = random.Random(seed)
    gains = []
    for _ in range(reps):
        sample = [rng.choice(v) for v in by_cond.values() for _ in v]
        gains.append(points(new, sample) - points(old, sample))
    gains.sort()
    return gains[int(0.025 * reps)], gains[int(0.975 * reps) - 1]


def cv_folds(cv_path, new, old):
    """Real-cutoff OOF per fold (new vs old) from a cv_exp5/cv_exp6 report: (#folds new better, mean delta)."""
    rep = json.loads(Path(cv_path).read_text())
    d = [a - b for a, b in zip(rep[new]["fold_real"], rep[old]["fold_real"])]
    return sum(x < 0 for x in d), sum(d) / len(d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--allow-round", default="adaptation", help="expected metadata round (dry runs only: main)")
    p.add_argument("--python313", default=str(PY313) if PY313.exists() else sys.executable)
    a = p.parse_args()
    meta = json.loads((ROOT / "public/metadata.json").read_text())
    if meta["round"] != a.allow_round:
        raise SystemExit(f"metadata round is {meta['round']!r}, expected {a.allow_round!r}")
    LOGS.mkdir(exist_ok=False)
    rnd, maxp = meta["round"], float(meta["max_points"])
    report = {"round": rnd, "max_points": maxp, "season": meta["season_id"], "entries": {}, "notes": []}
    ready = report["entries"]

    def entry(name, *args):
        try:
            ready[name] = build_entry(name, *args, a.python313, rnd)
            r = ready[name]
            say(f"READY {name}: val {r['points']['full']:.3f} (tuning {r['points']['tuning']:.3f} / audit "
                f"{r['points']['audit']:.3f}) | {r['csv']} + {r['zip']}")
        except Exception as exc:  # keep going: earlier entries stay usable
            report["notes"].append(str(exc))
            say(f"FAILED {name}: {exc}")

    say(f"round {rnd}, max points {maxp}")
    run("baseline", ["run_starter.py", "--method", "pooled_recent", "--out", "a_baseline"])
    base_pts = json.loads((ROOT / "a_baseline/run.json").read_text())["validation"]["contest"]["technical_points"]
    try:  # organizer starter ZIP, re-zipped with the team/round README header, then verified
        run("baseline_entry", ["build_entry.py", "--starter", "a_baseline", "--out", "zips/a_baseline_starter.zip"])
        run("baseline_readme", ["tools/make_entry.py", "--from-zip", "zips/a_baseline_starter.zip", "--name", "a_baseline"])
        if "REPRODUCTION OK" not in run("baseline_verify", ["tools/verify_entry.py", "zips/a_baseline.zip",
                                                            "a_baseline/submission.csv", "--python", a.python313]):
            raise RuntimeError("a_baseline: reproduction not confirmed")
        ready["a_baseline"] = {"csv": "a_baseline/submission.csv", "zip": "zips/a_baseline.zip", "points": {"full": base_pts},
                               "csv_sha256": sha(ROOT / "a_baseline/submission.csv"), "zip_sha256": sha(ROOT / "zips/a_baseline.zip")}
        say(f"READY a_baseline: val {base_pts:.3f} | a_baseline/submission.csv + zips/a_baseline.zip")
    except Exception as exc:
        report["notes"].append(str(exc))
        say(f"FAILED a_baseline: {exc}")

    entry("a_exp5_maincfg", "exp5.json", "exp5_model_text.md", ["tests/test_leakage.py"])

    run("cv_exp5", ["cv_exp5.py"], cwd=ROOT / "research")
    k5, d5 = cv_folds(ROOT / "runs/exp5_cv/cv.json", "exp5", "exp4")
    report["cv_exp5_vs_exp4"] = {"folds_better": k5, "mean_delta": d5}
    say(f"train CV EXP-5 vs EXP-4 (real rows): better {k5}/5, mean {d5:+.5f}")
    run("cfg_exp5", ["tools/config_from_cv.py", "--base", "configs/exp5.json", "--cv", "runs/exp5_cv/cv.json",
                     "--out", "configs/a_exp5_recal.json"])
    entry("a_exp5_recal", "a_exp5_recal.json", "exp5_model_text.md", ["tests/test_leakage.py", "tests/test_equivalence_exp5.py"])

    run("cv_exp6", ["cv_exp6.py"], cwd=ROOT / "research")
    k6, d6 = cv_folds(ROOT / "runs/exp6_cv/cv.json", "exp6", "exp5")
    report["cv_exp6_vs_exp5"] = {"folds_better": k6, "mean_delta": d6}
    say(f"train CV EXP-6 vs EXP-5 (real rows): better {k6}/5, mean {d6:+.5f}")
    run("cfg_exp6", ["tools/config_from_cv.py", "--base", "configs/exp6.json", "--cv", "runs/exp6_cv/cv.json",
                     "--out", "configs/a_exp6.json"])
    entry("a_exp6", "a_exp6.json", "exp6_model_text.md", ["tests/test_leakage.py", "tests/test_equivalence_exp6.py"])

    pick = "a_exp5_recal"
    if k5 < 4:
        say("size rule did not carry over on train CV -> building EXP-4 recalibrated")
        run("final_cv_report", ["research/final_cv_report.py"])
        run("cv_exp4", ["research/cv_exp4.py"])
        run("cfg_exp4", ["tools/config_from_cv.py", "--base", "configs/exp4.json", "--cv", "runs/exp4_cv/cv.json",
                         "--out", "configs/a_exp4_recal.json"])
        entry("a_exp4_recal", "a_exp4_recal.json", "exp4_model_text.md", ["tests/test_leakage.py", "tests/test_equivalence_exp4.py"])
        pick = "a_exp4_recal"
    elif k6 == 5 and "a_exp6" in ready and "a_exp5_recal" in ready:
        lo, hi = paired_bootstrap(ROOT / ready["a_exp6"]["csv"], ROOT / ready["a_exp5_recal"]["csv"], maxp)
        report["val_exp6_minus_exp5_ci95"] = [lo, hi]
        say(f"validation EXP-6 minus EXP-5: 95% [{lo:+.3f}, {hi:+.3f}]")
        if lo > 0:
            pick = "a_exp6"
    for fallback in (pick, "a_exp5_recal", "a_exp5_maincfg", "a_baseline"):
        if fallback in ready:
            pick = fallback
            break
    if pick != "a_baseline" and ready[pick]["points"]["full"] < base_pts:
        report["notes"].append(f"{pick} scored below the baseline on validation; picking a_baseline")
        pick = "a_baseline"
    report["pick"] = pick

    # structure diagnostics for the log (do not change the pick)
    try:
        run("exp4_ref", ["research/run_model.py", "--config", "configs/exp4.json", "--out", "runs/exp4", "--no-score"])
        diag = run("r1_size_structure", ["r1_size_structure.py"], cwd=ROOT / "research")
        diag += run("r3_rule_priors", ["r3_rule_priors.py"], cwd=ROOT / "research")
        report["diagnostics"] = [ln for ln in diag.splitlines() if "size lag vs win pair" in ln or "pair share overall" in ln]
    except Exception as exc:
        report["notes"].append(f"diagnostics: {exc}")
    report["seconds"] = time.time() - T0
    (ROOT / "round2_report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    print("\nSUMMARY (validation points full / tuning / audit)")
    for name, r in ready.items():
        pts = r["points"]
        print(f"  {name:15s} {pts['full']:.3f}" + (f" / {pts['tuning']:.3f} / {pts['audit']:.3f}" if "tuning" in pts else ""))
    for line in report.get("diagnostics", []):
        print("  diag:", line.strip())
    for note in report["notes"]:
        print("  note:", note.splitlines()[0])
    r = ready[pick]
    print(f"\nPICK {pick}\n  upload CSV {r['csv']}  sha256 {r['csv_sha256']}\n  upload ZIP {r['zip']}  sha256 {r['zip_sha256']}")
    say("done")


if __name__ == "__main__":
    main()
