# House Edge(Team name Razor)

Round 1

Round 2 (Adaptation): files in `round2-adaptation/`, clearly labeled and separate. Round 1 files are unchanged.

| File | SHA-256 |
|---|---|
| `round2-adaptation/submission.csv` (1,536 Adaptation test queries) | `762712ef11cb883915b2699e3815d82b9f69aaec66136bb79ca314432af2495d` |
| `round2-adaptation/entry.zip` (README names Team Razor, Round 2 adaptation) | `d4e8992e3869b06740f551750a47ad53128a655fc716ab33611aa8fcc28e8878` |

Model: EXP-5 refitted on Adaptation TRAIN (identity calibration) + XGBoost correction, standard-library inference (`python predict.py --data /data --out /output/submission.csv`). Local validation 38.709/40. Reproduction verified byte-identical on Python 3.12.12 and 3.13.9.
