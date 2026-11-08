"""EXP-6: EXP-5 with oriented hi/lo cell levels per (machine type, query signal) instead of per type.

The signal is observed at the query cutoff, so it only selects the prior levels for that query's machine. Levels are
learned from training machines grouped by their own query signal (hi_lo_levels on each signal's machines).
"""

from .history import machine_payouts
from .oriented import hi_lo_levels
from .pair_rule import TYPES
from .size_rule import SizeRuleForecaster

SIGNALS = ("-1", "0", "1")


class SignalLevelForecaster(SizeRuleForecaster):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.signal_levels = None

    def fit(self, queries, history, labels=None):
        machines = list(machine_payouts(queries, history).values())
        self.signal_levels = {s: hi_lo_levels(((q.machine_type, p) for q, p in machines if str(q.signal) == s),
                                              self.contrast_threshold) for s in SIGNALS}
        return super().fit(queries, history, labels)

    def query_tables(self, query, payouts):
        type_levels = self.levels
        self.levels = self.signal_levels[str(query.signal)]
        try:
            return super().query_tables(query, payouts)
        finally:
            self.levels = type_levels

    def to_dict(self):
        d = super().to_dict()
        d.update(model="signal_levels", signal_levels=self.signal_levels)
        return d

    @classmethod
    def from_dict(cls, d):
        if d.get("model") != "signal_levels":
            raise ValueError("Unsupported model file.")
        model = super().from_dict(dict(d, model="size_rule"))
        model.signal_levels = d["signal_levels"]
        if set(model.signal_levels) != set(SIGNALS) or any(set(v) != set(TYPES) for v in model.signal_levels.values()):
            raise ValueError("Malformed model file.")
        return model
