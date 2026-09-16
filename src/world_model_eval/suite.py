"""Run a model over many scenarios and aggregate into one report.

A scenario is an initial state plus an action sequence. For each, the harness
rolls the LEARNED model and the TRUTH model forward under the same actions and
compares them. The report aggregates trust horizons (with spread), the
consistency classification, the plausibility pass rate and the calibration.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import metrics as M
from .rollout import rollout_from_model


@dataclass
class Scenario:
    name: str
    initial_state: list
    actions: list
    dt: float = 0.02


@dataclass
class WorldModelReport:
    tol: float
    per_scenario: list = field(default_factory=list)
    channels: list = field(default_factory=list)

    def trust_steps(self):
        return [s["trust_horizon"]["steps"] for s in self.per_scenario]

    def summary(self):
        ts = self.trust_steps()
        kinds = [s["consistency"]["kind"] for s in self.per_scenario]
        plaus_fail = [s["plausibility_failed"] for s in self.per_scenario]
        cals = [s["calibration"] for s in self.per_scenario
                if s["calibration"] == s["calibration"]]  # drop NaN
        return {
            "n_scenarios": len(self.per_scenario),
            "trust_steps_median": float(np.median(ts)) if ts else 0.0,
            "trust_steps_min": int(np.min(ts)) if ts else 0,
            "trust_steps_max": int(np.max(ts)) if ts else 0,
            "consistency": {k: kinds.count(k) for k in set(kinds)},
            "plausibility_pass_rate": float(np.mean([f == 0 for f in plaus_fail]))
                if plaus_fail else 1.0,
            "calibration_mean": float(np.mean(cals)) if cals else None,
        }


def evaluate(learned, truth, scenarios, tol=0.05, vel_limit=None,
             accel_limit=None, passive_energy=None) -> WorldModelReport:
    channels = getattr(truth, "channels", [])
    rep = WorldModelReport(tol=tol, channels=list(channels))
    for sc in scenarios:
        pred = rollout_from_model(learned, sc.initial_state, sc.actions, sc.dt, channels)
        ref = rollout_from_model(truth, sc.initial_state, sc.actions, sc.dt, channels)
        se = M.step_error(pred, ref)
        th = M.trust_horizon(pred, ref, tol)
        con = M.long_horizon_consistency(pred, ref)
        pl = M.physics_plausibility(pred, vel_limit=vel_limit, accel_limit=accel_limit,
                                    passive_energy=passive_energy)
        cal = M.calibration(pred, ref)
        do = M.divergence_onset(pred, ref)
        rep.per_scenario.append({
            "scenario": sc.name,
            "final_error": se.value,
            "error_curve": se.detail["curve"],
            "trust_horizon": {"steps": int(th.detail["steps"]),
                              "seconds": th.detail["seconds"],
                              "explain": th.explain},
            "consistency": {"kind": con.detail.get("kind", "flat"),
                            "value": con.value, "explain": con.explain},
            "plausibility_failed": int(pl.value),
            "plausibility_explain": pl.explain,
            "calibration": cal.value,
            "calibration_explain": cal.explain,
            "divergence_onset": int(do.value),
        })
    return rep
