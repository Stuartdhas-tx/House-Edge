"""Standard-library inference for numeric multiclass XGBoost correction trees.

This deliberately supports only scalar-leaf gbtree models, five classes, and
explicit base probabilities. It does not fit models or implement categorical
splits, vector leaves, DART, or model-default base scores. Float32 arithmetic is
emulated with struct, including each margin accumulation. Python's exponential
may differ slightly from the native expf implementation; exact native equality
is not promised. Generate a deployment CSV with this same portable predictor.
"""

import json
import math
from pathlib import Path
import struct


def float32(value):
    """Round a scalar to IEEE binary32, returning an ordinary Python float."""
    try:
        return struct.unpack("=f", struct.pack("=f", float(value)))[0]
    except (OverflowError, TypeError, ValueError, struct.error) as error:
        raise ValueError("Value cannot be represented as float32.") from error


def _integer(value, name, minimum=0):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be an integer.") from error
    if str(result) != str(value) or result < minimum:
        raise ValueError(f"Invalid {name}: {value!r}.")
    return result


def _finite32(value, name):
    result = float32(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite float32.")
    return result


class PortableXGBoost:
    """Validated immutable-in-use tree arrays, loaded from XGBoost model JSON."""

    def __init__(self, model):
        try:
            self._parse(model)
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError("Malformed XGBoost model JSON.") from error

    @classmethod
    def load(cls, path):
        def reject_constant(value):
            raise ValueError(f"Nonfinite JSON constant: {value}.")
        with Path(path).open(encoding="utf-8") as handle:
            return cls(json.load(handle, parse_constant=reject_constant))

    def _parse(self, model):
        learner = model["learner"]
        objective = learner["objective"]
        parameters = learner["learner_model_param"]
        if objective["name"] != "multi:softprob":
            raise ValueError("Only multi:softprob is supported.")
        if (_integer(parameters["num_class"], "num_class") != 5
                or _integer(objective["softmax_multiclass_param"]["num_class"], "objective num_class") != 5
                or _integer(parameters["num_target"], "num_target") != 1):
            raise ValueError("Only five-class, single-target models are supported.")
        self.num_features = _integer(parameters["num_feature"], "num_feature", 1)
        self.feature_names = tuple(learner.get("feature_names", []))
        if self.feature_names and (len(self.feature_names) != self.num_features
                                   or any(not isinstance(name, str) for name in self.feature_names)
                                   or len(set(self.feature_names)) != len(self.feature_names)):
            raise ValueError("Feature names must be unique and match num_feature.")
        feature_types = learner.get("feature_types", [])
        if feature_types and (len(feature_types) != self.num_features
                              or any(kind not in ("float", "int", "q", "i") for kind in feature_types)):
            raise ValueError("Only numeric feature types are supported.")
        booster = learner["gradient_booster"]
        if booster["name"] != "gbtree":
            raise ValueError("Only gbtree boosters are supported.")
        forest = booster["model"]
        cats = forest.get("cats")
        empty_cats = (cats in (None, {}, []) or
                      (isinstance(cats, dict) and set(cats) <= {"enc", "feature_segments", "sorted_idx"}
                       and all(value == [] for value in cats.values())))
        if not empty_cats:
            raise ValueError("Categorical encodings are unsupported.")
        tree_parameters = forest["gbtree_model_param"]
        if _integer(tree_parameters["num_parallel_tree"], "num_parallel_tree") != 1:
            raise ValueError("Only one tree per class per boosting iteration is supported.")
        trees, tree_info = forest["trees"], forest["tree_info"]
        count = _integer(tree_parameters["num_trees"], "num_trees")
        if not isinstance(trees, list) or not isinstance(tree_info, list) or len(trees) != count or len(tree_info) != count:
            raise ValueError("Tree count and class mapping lengths differ.")
        classes = tuple(_integer(value, "tree class") for value in tree_info)
        if any(value >= 5 for value in classes):
            raise ValueError("Every tree must map to one of the five output classes.")
        pointers = forest.get("iteration_indptr")
        if pointers is not None:
            pointers = [_integer(value, "iteration pointer") for value in pointers]
            if (not pointers or pointers[0] != 0 or pointers[-1] != count
                    or any(b - a != 5 for a, b in zip(pointers, pointers[1:]))):
                raise ValueError("Iteration boundaries must contain one scalar tree per class.")
            for a, b in zip(pointers, pointers[1:]):
                if sorted(classes[a:b]) != list(range(5)):
                    raise ValueError("Each iteration must contain all five classes exactly once.")
        self.tree_info = classes
        parsed = []
        for tree_id, tree in enumerate(trees):
            if _integer(tree["id"], "tree id") != tree_id:
                raise ValueError("Tree IDs must match forest order.")
            parameter = tree["tree_param"]
            n = _integer(parameter["num_nodes"], "num_nodes", 1)
            if (_integer(parameter["num_deleted"], "num_deleted") != 0
                    or _integer(parameter["size_leaf_vector"], "size_leaf_vector") != 1
                    or _integer(parameter["num_feature"], "tree num_feature") != self.num_features):
                raise ValueError("Only undeleted scalar-leaf trees with matching features are supported.")
            for key in ("categories", "categories_nodes", "categories_segments", "categories_sizes"):
                if tree.get(key, []):
                    raise ValueError("Categorical tree splits are unsupported.")
            required = ("left_children", "right_children", "split_indices", "split_conditions", "default_left",
                        "split_type", "base_weights", "parents")
            for key in required:
                if not isinstance(tree[key], list) or len(tree[key]) != n:
                    raise ValueError(f"Tree array {key} has the wrong length.")
            for key in ("loss_changes", "sum_hessian"):
                if key in tree and len(tree[key]) != n:
                    raise ValueError(f"Tree array {key} has the wrong length.")
            left = tuple(_integer(v, "left child", -1) for v in tree["left_children"])
            right = tuple(_integer(v, "right child", -1) for v in tree["right_children"])
            feature = tuple(_integer(v, "split feature") for v in tree["split_indices"])
            default = tuple(_integer(v, "default_left") for v in tree["default_left"])
            threshold = tuple(_finite32(v, "split threshold/leaf value") for v in tree["split_conditions"])
            if any(_integer(v, "split_type") != 0 for v in tree["split_type"]):
                raise ValueError("Only numerical split_type=0 is supported.")
            for value in tree["base_weights"]:
                _finite32(value, "base weight")
            for node in range(n):
                if default[node] not in (0, 1):
                    raise ValueError("default_left must be zero or one.")
                if (left[node] == -1) != (right[node] == -1):
                    raise ValueError("A leaf must have both children set to -1.")
                if left[node] != -1 and (left[node] >= n or right[node] >= n or feature[node] >= self.num_features):
                    raise ValueError("Split child or feature index is out of bounds.")
            visited, pending = set(), [0]
            while pending:
                node = pending.pop()
                if node in visited:
                    raise ValueError("Tree contains a cycle or a shared child.")
                visited.add(node)
                if left[node] != -1:
                    pending.extend((left[node], right[node]))
            if len(visited) != n:
                raise ValueError("Tree contains unreachable nodes.")
            # Leaf predictions are split_conditions, not internal base_weights.
            parsed.append((left, right, feature, threshold, default))
        self.trees = tuple(parsed)

    def _features(self, features):
        if len(features) != self.num_features:
            raise ValueError(f"Expected {self.num_features} features.")
        values = [math.nan if value is None else float32(value) for value in features]
        if any(math.isinf(value) for value in values):
            raise ValueError("Features may be finite or missing, but not infinite.")
        return values

    @staticmethod
    def _leaf(tree, values):
        left, right, feature, threshold, default = tree
        node = 0
        while left[node] != -1:
            value = values[feature[node]]
            go_left = bool(default[node]) if math.isnan(value) else value < threshold[node]
            node = left[node] if go_left else right[node]
        return node

    @staticmethod
    def _base_margin(base_prob):
        if len(base_prob) != 5:
            raise ValueError("Exactly five base probabilities are required.")
        values = [float(value) for value in base_prob]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("Base probabilities must be finite and in [0,1].")
        total = math.fsum(values)
        if abs(total - 1) > 2e-6:
            raise ValueError("Base probabilities must sum to one within 2e-6.")
        return [float32(math.log(max(1e-12, min(1.0, value / total)))) for value in values]

    def leaf_indices_row(self, features):
        """Return leaf node IDs in forest order for branch-path verification."""
        values = self._features(features)
        return [self._leaf(tree, values) for tree in self.trees]

    def predict_margin_row(self, features, base_prob):
        """Return five accumulated float32 raw margins before softmax."""
        values = self._features(features)
        margin = self._base_margin(base_prob)
        for tree, output in zip(self.trees, self.tree_info):
            leaf = self._leaf(tree, values)
            margin[output] = float32(margin[output] + tree[3][leaf])
        return margin

    def predict_row(self, features, base_prob):
        """Predict from explicit raw EXP-5 probabilities and matching features."""
        margin = self.predict_margin_row(features, base_prob)
        maximum = max(margin)
        exponentials = [float32(math.exp(float32(value - maximum))) for value in margin]
        total = math.fsum(exponentials)
        native_like = [float32(value / total) for value in exponentials]
        normalized = math.fsum(native_like)
        return [value / normalized for value in native_like]

    def predict(self, features, base_probs):
        """Batch convenience API; returns a list of five-probability lists."""
        if len(features) != len(base_probs):
            raise ValueError("Feature and base-probability row counts differ.")
        return [self.predict_row(row, base) for row, base in zip(features, base_probs)]
