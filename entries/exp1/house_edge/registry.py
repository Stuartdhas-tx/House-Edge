"""Name -> forecaster class, so configs and predict.py stay model-agnostic."""
import json
from pathlib import Path

from .pair_rule import PairRuleForecaster

MODELS = {"pair_rule": PairRuleForecaster}


def build_model(config):
    config = dict(config)
    return MODELS[config.pop("model")](**config)


def load_model(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return MODELS[data["model"]].from_dict(data)
