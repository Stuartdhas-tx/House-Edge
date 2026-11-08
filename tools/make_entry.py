"""Assemble entries/<name>/ from a run folder, then zip it with the kit's build_entry.py.

Usage: python tools/make_entry.py --run runs/exp1 --name exp1
       python tools/make_entry.py --from-zip zips/a_baseline_starter.zip --name a_baseline  (add team/round to a built ZIP)
The README names the team and round, as organizers require (must match the Form and Devpost).
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TEAM = "Razor"
ROUND_NUMBER = {"main": 1, "adaptation": 2}

PREDICT = '''"""Recreate the forecast CSV offline: python predict.py --data /data --out /output/submission.csv"""
import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # /submission is read-only
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from house_edge.registry import load_model  # noqa: E402
from slot_forecasting.io import load_inputs, write_submission  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="current round public folder (contains test/)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    queries, history = load_inputs(Path(args.data) / "test")
    model = load_model(HERE / "model.json")
    write_submission(args.out, model.predict(queries, history))


if __name__ == "__main__":
    main()
'''

README = """# House Edge entry: {name} (Team {team}, Round {round_no})

- Team name: {team}
- Round: {round_no} ({round}) · Season ID: {season}
- Forecast file: submitted separately as `submission.csv` (SHA-256 `{csv_sha}`)
- Environment: `python` image, **Python standard library only** (no requirements.txt). Fresh-process reproduction verified byte-identical on Python {pyver} and 3.13.9 (macOS arm64).
- Hardware: any CPU, single core; measured test prediction time {pred_s:.0f} s, well under 1 GiB RAM.

## Recreate the forecast

Run from this folder:

```sh
python predict.py --data /data --out /output/submission.csv
```

`predict.py` loads the saved parameters in `model.json` and predicts `/data/test` deterministically (no randomness,
no network, no training at prediction time). It writes only to `--out`.

## Model

{model_text}

Settings: `model.json` (hyperparameters, type base rates, size priors, learned rule prior). Hyperparameters were
chosen by episode-grouped 5-fold cross-validation on TRAIN pseudo-queries; all parameters were fit on the {round}-round
training split only. Validation targets were used for scoring, never for fitting.

Training command (from the team repository): `python research/run_model.py --config configs/{name}.json --out runs/{name}`

Seeds: none needed (deterministic). `slot_forecasting/{{__init__,io,schema}}.py` are unmodified copies of the organizer kit.
"""


def build_zip(dest, zip_path):
    zip_path.parent.mkdir(exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "build_entry.py"), "--source", str(dest), "--out", str(zip_path)],
                   cwd=ROOT, check=True)


def retitle(source_zip, dest, zip_path, meta):
    """Copy a built entry (e.g. the organizer starter) into dest with the team/round README header, then re-zip."""
    from slot_forecasting.submission_contract import extract_archive
    extract_archive(Path(source_zip).read_bytes(), dest)
    readme = dest / "README.md"
    title, _, body = readme.read_text(encoding="utf-8").partition("\n")
    rnd = meta["round"]
    readme.write_text(f"{title} (Team {TEAM}, Round {ROUND_NUMBER.get(rnd, '?')})\n\n- Team name: {TEAM}\n"
                      f"- Round: {ROUND_NUMBER.get(rnd, '?')} ({rnd}) · Season ID: {meta['season_id']}\n{body}",
                      encoding="utf-8")
    build_zip(dest, zip_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run")
    p.add_argument("--from-zip")
    p.add_argument("--name", required=True)
    p.add_argument("--model-text", default="")
    a = p.parse_args()
    if bool(a.run) == bool(a.from_zip):
        raise SystemExit("Give exactly one of --run or --from-zip.")
    dest = ROOT / "entries" / a.name
    zip_path = ROOT / "zips" / f"{a.name}.zip"
    if dest.exists() or zip_path.exists():
        raise SystemExit("Entry or zip already exists; choose a new name.")
    meta = json.loads((ROOT / "public/metadata.json").read_text())
    if a.from_zip:
        retitle(ROOT / a.from_zip, dest, zip_path, meta)
        return
    run = ROOT / a.run
    (dest / "house_edge").mkdir(parents=True)
    (dest / "slot_forecasting").mkdir()
    for f in sorted((ROOT / "house_edge").glob("*.py")):
        shutil.copyfile(f, dest / "house_edge" / f.name)
    for name in ("__init__.py", "io.py", "schema.py"):
        shutil.copyfile(ROOT / "slot_forecasting" / name, dest / "slot_forecasting" / name)
    shutil.copyfile(run / "model.json", dest / "model.json")
    (dest / "predict.py").write_text(PREDICT, encoding="utf-8")
    (dest / "reproduce.json").write_text(json.dumps(
        {"image": "python", "command": ["python", "predict.py", "--data", "/data", "--out", "/output/submission.csv"]},
        indent=2) + "\n", encoding="utf-8")
    run_info = json.loads((run / "run.json").read_text())
    (dest / "README.md").write_text(README.format(
        name=a.name, team=TEAM, round_no=ROUND_NUMBER.get(meta["round"], "?"), round=meta["round"],
        season=meta["season_id"], pyver=run_info.get("python", "?"), pred_s=run_info["seconds"]["predict_test"],
        csv_sha=hashlib.sha256((run / "submission.csv").read_bytes()).hexdigest(),
        model_text=a.model_text or "See model.json."), encoding="utf-8")
    build_zip(dest, zip_path)


if __name__ == "__main__":
    main()
