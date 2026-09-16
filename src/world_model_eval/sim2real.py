"""The concrete sim-to-real question: is this model good enough to plan through?

A learned simulator is usable as a bridge for a task only over the horizon
where its error stays inside the task's tolerance. This turns "the world model
is pretty good" into "you may plan 42 steps ahead for a 30 mm task, and your
task needs 60, so no."
"""
from __future__ import annotations

from dataclasses import dataclass

from .metrics import trust_horizon
from .rollout import Rollout


@dataclass
class TrustVerdict:
    task_tol: float
    trust_steps: int
    trust_seconds: float
    needed_steps: int | None
    usable: bool | None
    explain: str


def sim_to_real_trust(pred: Rollout, ref: Rollout, task_tol: float,
                      needed_steps: int | None = None) -> TrustVerdict:
    """Horizon over which the model is usable as a bridge for a task.

    task_tol is the task's own error budget (e.g. a 30 mm grasp gate = 0.03).
    If you pass how many steps your plan needs, the verdict says yes/no.
    """
    th = trust_horizon(pred, ref, task_tol)
    steps = int(th.detail["steps"])
    secs = float(th.detail["seconds"])
    if needed_steps is None:
        usable = None
        expl = ("usable as a bridge for %d steps (%.2f s) at task tolerance %.3g; "
                "state how many steps your plan needs for a yes/no"
                % (steps, secs, task_tol))
    else:
        usable = steps >= needed_steps
        expl = ("plan needs %d steps, model is trustworthy for %d at tolerance %.3g -> %s"
                % (needed_steps, steps, task_tol,
                   "USABLE" if usable else "NOT usable, the plan runs past the trust horizon"))
    return TrustVerdict(task_tol, steps, secs, needed_steps, usable, expl)
