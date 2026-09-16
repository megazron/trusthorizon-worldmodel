"""The metrics. Each returns a small result object with a number and a plain
explanation, so a report reads as sentences rather than a wall of floats.

Every metric compares a PREDICTED rollout against an aligned REFERENCE rollout,
except the plausibility check, which judges a single rollout against physics it
should not violate even when it is wrong.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .rollout import Rollout, check_aligned


@dataclass
class Result:
    name: str
    value: float
    explain: str
    detail: dict = field(default_factory=dict)


def _per_step_error(pred: Rollout, ref: Rollout) -> np.ndarray:
    """RMSE across state dims at each time index, length T+1 (index 0 is ~0)."""
    check_aligned(pred, ref)
    diff = pred.states - ref.states
    return np.sqrt(np.mean(diff * diff, axis=1))


def step_error(pred: Rollout, ref: Rollout) -> Result:
    """Per-step state error and its growth curve.

    The headline number is the FINAL error; the curve in `detail` is what every
    other metric reads.
    """
    curve = _per_step_error(pred, ref)
    return Result("step_error", float(curve[-1]),
                  "state RMSE grows from %.3g at step 1 to %.3g at the horizon (%d steps)"
                  % (curve[1] if len(curve) > 1 else 0.0, curve[-1], len(curve) - 1),
                  {"curve": curve.tolist(), "mean": float(np.mean(curve[1:]))})


def trust_horizon(pred: Rollout, ref: Rollout, tol: float) -> Result:
    """Largest H such that error stays under `tol` for EVERY step up to H.

    This is the headline question: how many steps can I trust this rollout. It
    is a first-crossing, not a mean, because a model you can trust for 40 steps
    and then not is trustworthy for 40 steps, not on average.
    """
    curve = _per_step_error(pred, ref)
    H = 0
    for t in range(1, len(curve)):
        if curve[t] <= tol:
            H = t
        else:
            break
    secs = H * pred.dt
    if H == len(curve) - 1:
        expl = "error stays under %.3g for the whole %d-step horizon (%.2f s)" % (tol, H, secs)
    else:
        expl = "error first exceeds %.3g at step %d; trust the rollout for %d steps (%.2f s)" % (
            tol, H + 1, H, secs)
    return Result("trust_horizon", float(H), expl,
                  {"steps": H, "seconds": secs, "tol": tol,
                   "full_horizon": len(curve) - 1})


def long_horizon_consistency(pred: Rollout, ref: Rollout) -> Result:
    """Does error grow LINEARLY (drift) or EXPONENTIALLY (instability)?

    Fits err ~ a + b t and log(err) ~ c + r t and reports which fits better by
    R^2. Exponential growth is the dangerous one: it means small errors compound
    and the model is unusable past a short horizon however good it looks early.
    """
    curve = np.asarray(_per_step_error(pred, ref))
    t = np.arange(len(curve))
    mask = t >= 1
    tt = t[mask].astype(float)
    ee = curve[mask]
    if len(ee) < 3 or np.allclose(ee, ee[0]):
        return Result("long_horizon_consistency", 0.0,
                      "too short or too flat to classify growth", {"kind": "flat"})

    def r2(y, yh):
        ss = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - np.sum((y - yh) ** 2) / ss if ss > 0 else 0.0

    bl, al = np.polyfit(tt, ee, 1)
    lin_r2 = r2(ee, al + bl * tt)
    pos = ee > 1e-12
    exp_r2, rate = -np.inf, 0.0
    if np.sum(pos) >= 3:
        rr, cc = np.polyfit(tt[pos], np.log(ee[pos]), 1)
        exp_r2 = r2(np.log(ee[pos]), cc + rr * tt[pos])
        rate = float(rr)
    if exp_r2 > lin_r2 + 0.02 and rate > 0:
        kind = "exponential"
        expl = ("error grows EXPONENTIALLY at rate %.3g per step (R2 %.3f) -- "
                "unstable; small errors compound" % (rate, exp_r2))
        val = rate
    else:
        kind = "linear"
        expl = ("error grows roughly LINEARLY at %.3g per step (R2 %.3f) -- "
                "steady drift, not a blow-up" % (bl, lin_r2))
        val = float(bl)
    return Result("long_horizon_consistency", val, expl,
                  {"kind": kind, "linear_slope": float(bl), "linear_r2": float(lin_r2),
                   "exp_rate": rate, "exp_r2": float(exp_r2)})


def physics_plausibility(roll: Rollout, vel_channel="v", vel_limit=None,
                         accel_limit=None, penetration_channel=None,
                         penetration_tol=1e-3, passive_energy=None) -> Result:
    """Cheap invariants a model must not violate even when it is inaccurate.

    - no NaN/inf anywhere in the states
    - |velocity| within vel_limit (if given)
    - |accel| = |dv/dt| within accel_limit (if given)
    - contact penetration under penetration_tol (if a channel is given)
    - passive energy non-increasing beyond input work, using a supplied
      `passive_energy(states) -> array` (if given and the run is unforced)

    Returns the number of FAILED checks as the value; detail lists each.
    """
    checks = []

    def add(name, ok, worst, note):
        checks.append({"check": name, "ok": bool(ok), "worst": float(worst), "note": note})

    finite = np.isfinite(roll.states).all()
    add("finite", finite, 0.0 if finite else 1.0,
        "no NaN/inf in states" if finite else "state contains NaN or inf")

    v = roll.channel(vel_channel)
    if v is not None and vel_limit is not None:
        worst = float(np.max(np.abs(v)))
        add("velocity_limit", worst <= vel_limit, worst,
            "max |v| %.3g vs limit %.3g" % (worst, vel_limit))
    if v is not None and accel_limit is not None and roll.horizon >= 1:
        acc = np.diff(v) / roll.dt
        worst = float(np.max(np.abs(acc))) if acc.size else 0.0
        add("accel_limit", worst <= accel_limit, worst,
            "max |a| %.3g vs limit %.3g" % (worst, accel_limit))
    if penetration_channel is not None:
        p = roll.channel(penetration_channel)
        if p is not None:
            worst = float(np.max(np.maximum(p, 0.0)))
            add("penetration", worst <= penetration_tol, worst,
                "max penetration %.3g vs tol %.3g" % (worst, penetration_tol))
    if passive_energy is not None and np.allclose(roll.actions, 0.0):
        e = np.asarray(passive_energy(roll.states), float)
        rise = float(np.max(np.diff(e))) if e.size > 1 else 0.0
        add("passive_energy", rise <= 1e-6, rise,
            "largest passive energy increase %.3g (should be <= 0)" % rise)

    failed = [c for c in checks if not c["ok"]]
    expl = ("all %d plausibility checks pass" % len(checks) if not failed
            else "%d of %d plausibility checks FAIL: %s"
                 % (len(failed), len(checks), ", ".join(c["check"] for c in failed)))
    return Result("physics_plausibility", float(len(failed)), expl,
                  {"checks": checks, "n_checks": len(checks)})


def calibration(pred: Rollout, ref: Rollout, ks=(1.0, 2.0, 3.0)) -> Result:
    """Is the model's own uncertainty honest?

    Needs pred.meta['sigma'] (a per-step std, from a model that reports it). For
    each k, computes the fraction of state dims/steps whose true error falls
    within +-k*sigma, and compares to the Gaussian target (68/95/99.7%). The
    value is an ECE-style mean absolute gap. A confidently-wrong model (tiny
    sigma, big error) scores badly here, which is the point: it is worse than an
    honestly-uncertain one.
    """
    check_aligned(pred, ref)
    sigma = pred.meta.get("sigma")
    if sigma is None:
        return Result("calibration", float("nan"),
                      "model reports no uncertainty; calibration not applicable",
                      {"available": False})
    sigma = np.asarray(sigma, float)                       # (T, d)
    err = np.abs(pred.states[1:] - ref.states[1:])         # (T, d)
    target = {1.0: 0.6827, 2.0: 0.9545, 3.0: 0.9973}
    rows, gaps = [], []
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sigma > 0, err / sigma, np.inf)
    for k in ks:
        cov = float(np.mean(z <= k))
        tgt = target.get(k, 0.0)
        rows.append({"k": k, "coverage": cov, "target": tgt})
        gaps.append(abs(cov - tgt))
    ece = float(np.mean(gaps))
    expl = ("calibration gap %.3f (0 is perfect); 1-sigma coverage %.2f vs 0.68"
            % (ece, rows[0]["coverage"]))
    return Result("calibration", ece, expl, {"available": True, "bands": rows})


def divergence_onset(pred: Rollout, ref: Rollout, noise_floor=None,
                     factor=3.0) -> Result:
    """The step where the two trajectories separate beyond noise.

    Uses the median of the early per-step error as the noise floor (or a value
    you pass), and reports the first step whose error exceeds `factor` times it.
    """
    curve = np.asarray(_per_step_error(pred, ref))
    if len(curve) < 3:
        return Result("divergence_onset", float(len(curve) - 1),
                      "too short to locate an onset", {})
    if noise_floor is None:
        early = curve[1:max(2, len(curve) // 5)]
        noise_floor = float(np.median(early)) if early.size else 0.0
    thresh = max(noise_floor * factor, 1e-12)
    onset = len(curve) - 1
    for t in range(1, len(curve)):
        if curve[t] > thresh:
            onset = t
            break
    return Result("divergence_onset", float(onset),
                  "trajectories separate beyond %.2fx the noise floor at step %d"
                  % (factor, onset),
                  {"noise_floor": noise_floor, "threshold": thresh})
