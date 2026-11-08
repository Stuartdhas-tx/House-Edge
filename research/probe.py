"""Reproduce the DESIGN.md diagnostic `probe_hl30` (reference only; NOT a submission candidate).

It hard-codes three pairs and size vectors that were read off validation targets, so its validation
score is not a generalization estimate.
"""
import argparse
from pathlib import Path

from common import PUBLIC, load_split, score_validation_csv, write_json
from slot_forecasting.io import write_submission

PAIRS = [(2, 5), (4, 7), (1, 3)]
SV = {"balanced": ([0.4999, 0.3479, 0.1305, 0.0217], [0.2141, 0.5467, 0.2050, 0.0342]),
      "jackpot": ([0.3351, 0.2449, 0.3149, 0.1050], [0.1208, 0.3239, 0.4165, 0.1388]),
      "steady": ([0.7540, 0.1968, 0.0482, 0.0010], [0.4551, 0.4359, 0.1068, 0.0022])}


def probe(queries, history):
    from house_edge.history import machine_payouts
    out = {}
    for qid, (q, S) in machine_payouts(queries, history).items():
        cut, best = q.cutoff, None
        for a, b in PAIRS:
            num = {True: [0.5, 1.0], False: [0.5, 1.0]}
            for t in S:
                if t - a in S and t - b in S:
                    w = 2 ** (-(cut - 1 - t) / 30)
                    c = (S[t - a] > 0) != (S[t - b] > 0)
                    num[c][0] += w * (S[t] > 0)
                    num[c][1] += w
            pd_, pe = num[True][0] / num[True][1], num[False][0] / num[False][1]
            if best is None or abs(pd_ - pe) > best[0]:
                best = (abs(pd_ - pe), a, b, pd_, pe)
        _, a, b, pd_, pe = best
        w = pd_ if (S[cut - a] > 0) != (S[cut - b] > 0) else pe
        A, B = SV[q.machine_type]
        row = [1 - w] + [w * (0.54 * x + 0.46 * y) for x, y in zip(A, B)]
        out[qid] = tuple(v / sum(row) for v in row)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/probe_hl30")
    out = Path(p.parse_args().out)
    if out.exists():
        raise SystemExit("Choose a new output folder.")
    q, h, _, _ = load_split("validation")
    write_submission(out / "validation_predictions.csv", probe(q, h))
    result = score_validation_csv(out / "validation_predictions.csv")
    write_json(out / "scores.json", result)
    print(result)
