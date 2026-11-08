"""Write a config whose calibration comes from a cv_exp5.py or cv_exp4.py report, keeping every other setting.

theta_all_train is [slope, balanced, jackpot, steady] (research/common.py TYPES order).
Usage: python tools/config_from_cv.py --base configs/exp5.json --cv runs/exp5_cv/cv.json --out configs/a_exp5.json
"""
import argparse
import json
from pathlib import Path

TYPES = ("balanced", "jackpot", "steady")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--cv", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit("Choose a new config name.")
    config = json.loads(Path(a.base).read_text())
    theta = json.loads(Path(a.cv).read_text())["theta_all_train"]
    config["calibration_slope"] = theta[0]
    config["calibration_intercepts"] = dict(zip(TYPES, theta[1:]))
    out.write_text(json.dumps(config, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: config[k] for k in ("calibration_slope", "calibration_intercepts")}, indent=1))


if __name__ == "__main__":
    main()
