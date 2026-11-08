"""Copy our code into a freshly extracted round-two kit, then profile its data for anything the model can't handle.

Never touches the new kit's organizer files or public/ data, and refuses to overwrite any existing file.
Usage: python tools/adapt_setup.py "/path/to/adaptation-kit" [--allow-round main]
"""
import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_DIRS = ("house_edge", "research", "tools", "configs", "tests")
NOTES = ("Claude.md", "DESIGN.md", "EXPERIMENT_LOG.md", "PROJECT_SUMMARY.md", "ROUND2_PLAYBOOK.md")
TYPES = {"balanced", "jackpot", "steady"}  # house_edge/pair_rule.py TYPES
PAYOUTS = {0, 1, 2, 5, 20}


def copy_code(kit):
    plan = []
    for d in CODE_DIRS:
        for f in sorted((ROOT / d).rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:
                plan.append((f, kit / f.relative_to(ROOT)))
    for name in NOTES:
        if (ROOT / name).exists():
            # main-round notes go in a subfolder so the new kit's own docs stay authoritative
            dest = kit / name if name in ("Claude.md", "ROUND2_PLAYBOOK.md") else kit / "main-round-notes" / name
            plan.append((ROOT / name, dest))
    clashes = [str(d) for _, d in plan if d.exists()]
    if clashes:
        raise SystemExit("Refusing to overwrite existing files:\n  " + "\n  ".join(clashes[:20]))
    for src, dest in plan:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    print(f"copied {len(plan)} files into {kit}")


def profile(kit):
    """Print what the model assumes vs what the new data contains; return a list of blocking problems."""
    problems = []
    for split in ("train", "validation", "test"):
        d = kit / "public" / split
        queries = list(csv.DictReader(open(d / "queries.csv", newline="")))
        history = list(csv.DictReader(open(d / "history.csv", newline="")))
        types = Counter(q["machine_type"] for q in queries)
        signals = Counter(q["signal"] for q in queries)
        cutoffs = [int(q["cutoff"]) for q in queries]
        payouts = Counter(int(r["payout"]) for r in history)
        per_episode = Counter(q["episode_id"] for q in queries)
        rows = Counter((r["episode_id"], r["machine_id"]) for r in history)
        line = (f"{split:10s} queries {len(queries):5d} episodes {len(per_episode):4d} machines/episode {sorted(set(per_episode.values()))} "
                f"cutoffs {min(cutoffs)}-{max(cutoffs)} rows/machine {min(rows.values())}-{max(rows.values())}\n"
                f"           types {dict(types)} signals {dict(signals)} payouts {dict(sorted(payouts.items()))}")
        print(line)
        if set(types) - TYPES:
            problems.append(f"{split}: unknown machine types {sorted(set(types) - TYPES)} (TYPES is hard-coded)")
        if set(payouts) - PAYOUTS:
            problems.append(f"{split}: unknown payout values {sorted(set(payouts) - PAYOUTS)}")
        if (d / "conditions.csv").exists():
            print("           conditions", dict(Counter(r["condition"] for r in csv.DictReader(open(d / "conditions.csv", newline="")))))
    return problems


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kit")
    p.add_argument("--allow-round", default="adaptation", help="expected metadata round (dry runs only: main)")
    a = p.parse_args()
    kit = Path(a.kit).resolve()
    if kit == ROOT:
        raise SystemExit("Point this at the NEW kit folder, not the main-round kit.")
    meta = json.loads((kit / "public/metadata.json").read_text())
    print(f"new kit round={meta['round']} max_points={meta['max_points']} suite={meta.get('suite_version')} season={meta['season_id']}")
    if meta["round"] != a.allow_round:
        raise SystemExit(f"metadata round is {meta['round']!r}, expected {a.allow_round!r}")
    old = json.loads((ROOT / "public/metadata.json").read_text())
    for k in sorted(set(meta) | set(old)):
        if k not in ("round", "season_id", "max_points") and meta.get(k) != old.get(k):
            print(f"  metadata changed: {k}: {old.get(k)!r} -> {meta.get(k)!r}")
    copy_code(kit)
    problems = profile(kit)
    if problems:
        print("BLOCKING:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("profile OK: types and payouts match what the model handles")


if __name__ == "__main__":
    main()
