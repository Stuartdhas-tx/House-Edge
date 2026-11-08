"""Name -> forecaster class, so configs and predict.py stay model-agnostic."""
import json
from pathlib import Path

from .calibrated import CalibratedChangePointForecaster
from .change_point import ChangePointForecaster
from .oriented import OrientedForecaster
from .pair_rule import PairRuleForecaster
from .size_rule import SizeRuleForecaster

MODELS = {"pair_rule": PairRuleForecaster, "change_point": ChangePointForecaster,
          "calibrated_change_point": CalibratedChangePointForecaster,
          "oriented": OrientedForecaster, "size_rule": SizeRuleForecaster}


def build_model(config):
    config = dict(config)
    return MODELS[config.pop("model")](**config)


def load_model(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return MODELS[data["model"]].from_dict(data)
