#!/usr/bin/env python3
"""Compare a well-calibrated, a biased, and an unstable learned model.

    python3 examples/three_models.py

Shows the trust horizon shrinking from the full horizon, to a few steps, to
almost nothing, and the unstable model failing the physics-plausibility check.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from world_model_eval import report as R
from world_model_eval.cli import _scenarios_from_dir
from world_model_eval.models import MockLearnedModel, MockTrueModel
from world_model_eval.suite import evaluate

SC = _scenarios_from_dir(os.path.join(os.path.dirname(__file__), "scenarios"))
DT = 0.02
MODELS = {
    "well-calibrated": MockLearnedModel(DT, step_noise=0.0008, sigma=0.02),
    "biased-drift": MockLearnedModel(DT, bias=0.006, divergence=0.001, sigma=0.02),
    "unstable": MockLearnedModel(DT, divergence=0.004, unstable=True, energy_leak=0.04,
                                 sigma=0.02),
}
for name, m in MODELS.items():
    m.reset([1.0, 0.0])  # each is fresh per scenario inside evaluate()
    rep = evaluate(m, MockTrueModel(DT), SC, tol=0.05, vel_limit=60.0,
                   passive_energy=MockTrueModel.energy)
    s = rep.summary()
    print("\n=== %s ===" % name)
    print("  median trust horizon : %.0f steps" % s["trust_steps_median"])
    print("  growth               : %s" % ", ".join("%s x%d" % kv for kv in s["consistency"].items()))
    print("  plausibility pass    : %.0f%%" % (100 * s["plausibility_pass_rate"]))
    print("  calibration gap      : %s"
          % ("%.3f" % s["calibration_mean"] if s["calibration_mean"] is not None else "n/a"))
