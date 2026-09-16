"""trusthorizon-worldmodel: metrics and a harness to decide whether a learned
simulator's rollout can be trusted, and for how many steps.

This package does not build a world model. It judges one you already have.
"""
from . import metrics, report, sim2real, suite
from .models import MockLearnedModel, MockTrueModel
from .rollout import Rollout, RolloutError, check_aligned, rollout_from_model
from .sim2real import sim_to_real_trust
from .suite import Scenario, WorldModelReport, evaluate

__version__ = "0.1.0"
__all__ = [
    "Rollout", "RolloutError", "check_aligned", "rollout_from_model",
    "MockTrueModel", "MockLearnedModel",
    "metrics", "suite", "report", "sim2real",
    "Scenario", "WorldModelReport", "evaluate", "sim_to_real_trust",
]
