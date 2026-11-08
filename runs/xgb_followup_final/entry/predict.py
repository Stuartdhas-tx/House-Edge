import argparse
import json
from pathlib import Path
from house_edge.registry import load_model
from house_edge.followup_predict import predict_followup
from house_edge.xgb_portable import PortableXGBoost
from slot_forecasting.io import load_inputs, write_submission

p = argparse.ArgumentParser()
p.add_argument('--data', required=True)
p.add_argument('--out', required=True)
p.add_argument('--split', choices=('test', 'validation'), default='test')
a = p.parse_args()
root = Path(__file__).resolve().parent
base = load_model(root / 'base_model.json')
post = json.loads((root / 'postprocess.json').read_text())
booster = None if post['kind'] == 'type_size_cal' else PortableXGBoost.load(root / 'booster.json')
queries, history = load_inputs(Path(a.data) / a.split)
write_submission(a.out, predict_followup(base, post, queries, history, booster=booster))
