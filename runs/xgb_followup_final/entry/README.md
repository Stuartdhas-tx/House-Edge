# House Edge Main — full_xgb

Team: use the same registered team selected on the submission Form. Round: Main.
Inference: Python 3.11+ standard library only; tested interpreter versions are recorded alongside the submitted artifacts after verification. No downloads, installation, GPU, or training are needed for reproduction.

Run `python predict.py --data /data --out /output/submission.csv` from this folder. The script reads only test queries/history and the supplied checkpoints. The forecast CSV is submitted separately.

EXP-5 global parameters were fitted on organizer Main TRAIN. The fixed postprocessor was trained on original-partition episode-OOF EXP-5 predictions and historical features. Three fixed TRAIN partitions were used as sensitivity checks to select this candidate. No validation targets were used to train coefficients. See training-run.json for exact settings, seeds, hashes, and training dependency versions; the experiment plans describe selection and its limitations. Training sources and requirements-training.txt are included for local research with the organizer public/ directory, not needed by the offline reproduction command.

XGBoost trees, if present, are evaluated with the included portable Python runtime. The authoritative CSV uses that runtime; small differences from native XGBoost's softmax are expected. Validation and test prediction use the same saved files and preprocessing.
