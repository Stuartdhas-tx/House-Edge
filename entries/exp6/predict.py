"""Recreate the forecast CSV offline: python predict.py --data /data --out /output/submission.csv"""
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
