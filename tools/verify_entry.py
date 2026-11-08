"""Fresh-process reproduction check of a code ZIP against the CSV you intend to upload.

Extracts the ZIP with the kit's safe extractor into a new temp folder, runs its reproduce.json command
(with /data and /output mapped to local paths) in a clean subprocess, then compares probabilities.
Usage: python tools/verify_entry.py zips/exp1.zip runs/exp1/submission.csv [--python /path/to/python3.13]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slot_forecasting.io import read_queries, read_submission  # noqa: E402
from slot_forecasting.submission_contract import extract_archive  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("zip")
    p.add_argument("expected_csv")
    p.add_argument("--python", default=sys.executable)
    a = p.parse_args()
    content = Path(a.zip).read_bytes()
    with tempfile.TemporaryDirectory(prefix="verify-") as tmp:
        sub = Path(tmp) / "submission"
        config = extract_archive(content, sub)
        out_dir = Path(tmp) / "output"
        out_dir.mkdir()
        mapping = {"/data": str(ROOT / "public"), "/output/submission.csv": str(out_dir / "submission.csv")}
        command = [mapping.get(x, x) for x in config["command"]]
        assert command[0] == "python", command
        command[0] = a.python
        for f in sub.rglob("*"):  # mimic the read-only /submission mount
            f.chmod(0o555 if f.is_dir() else 0o444)
        sub.chmod(0o555)
        env = {"PATH": os.environ.get("PATH", ""), "HOME": tmp, "PYTHONNOUSERSITE": "1"}
        run = subprocess.run(command, cwd=sub, env=env, capture_output=True, text=True, timeout=1800)
        for f in [sub, *sub.rglob("*")]:
            f.chmod(0o755 if f.is_dir() else 0o644)
        if run.returncode != 0:
            print(run.stdout, run.stderr)
            raise SystemExit("REPRODUCTION FAILED: command exited non-zero")
        produced = out_dir / "submission.csv"
        ids = {q.query_id for q in read_queries(ROOT / "public/test/queries.csv")}
        new, old = read_submission(produced, ids), read_submission(a.expected_csv, ids)
        diff = max(abs(x - y) for k in ids for x, y in zip(new[k], old[k]))
        same_bytes = produced.read_bytes() == Path(a.expected_csv).read_bytes()
        version = subprocess.run([a.python, "--version"], capture_output=True, text=True).stdout.strip()
        print(json.dumps({"python": version, "max_abs_probability_diff": diff, "byte_identical": same_bytes,
                          "zip_sha256": hashlib.sha256(content).hexdigest(),
                          "csv_sha256": hashlib.sha256(Path(a.expected_csv).read_bytes()).hexdigest()}, indent=1))
        if diff > 1e-8:
            raise SystemExit("REPRODUCTION FAILED: difference above 1e-8")
        print("REPRODUCTION OK")


if __name__ == "__main__":
    main()
