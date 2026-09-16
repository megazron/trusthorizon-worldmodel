#!/usr/bin/env python3
"""Regenerate docs/img/ from the kit's own mock models, so the figures cannot
drift from the code.

    python3 docs/make_figures.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IMG = os.path.join(HERE, "img")
os.makedirs(IMG, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "src"))

import matplotlib                                                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

from world_model_eval import metrics as M                         # noqa: E402
from world_model_eval.models import MockLearnedModel, MockTrueModel  # noqa: E402
from world_model_eval.rollout import rollout_from_model           # noqa: E402

DT = 0.02
GOOD, WARN, BAD, ACCENT = "#2b8a3e", "#b25a00", "#b02a37", "#0b7285"
plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 11})


def _rolls(n=90):
    x0, acts = [1.0, 0.0], (0.2 * np.sin(np.linspace(0, 6, n))).reshape(-1, 1)
    ch = MockTrueModel.channels
    ref = rollout_from_model(MockTrueModel(DT), x0, acts, DT, ch)
    good = rollout_from_model(MockLearnedModel(DT, step_noise=0.0008, sigma=0.02), x0, acts, DT, ch)
    biased = rollout_from_model(MockLearnedModel(DT, bias=0.006, divergence=0.001, sigma=0.02), x0, acts, DT, ch)
    unstable = rollout_from_model(MockLearnedModel(DT, divergence=0.004, unstable=True, sigma=0.02), x0, acts, DT, ch)
    return ref, good, biased, unstable


def error_growth():
    ref, good, biased, unstable = _rolls()
    tol = 0.05
    fig, ax = plt.subplots(figsize=(9, 5))
    offs = {"well-calibrated": (8, -4), "biased drift": (10, 10), "unstable": (10, 22)}
    for roll, name, col in [(good, "well-calibrated", GOOD),
                            (biased, "biased drift", WARN),
                            (unstable, "unstable", BAD)]:
        curve = np.array(M.step_error(roll, ref).detail["curve"])
        t = np.arange(len(curve))
        ax.plot(t, curve, color=col, lw=2, label=name)
        H = int(M.trust_horizon(roll, ref, tol).value)
        ax.plot([H], [curve[H]], "o", color=col, ms=8)
        ax.annotate("trust %d steps" % H, (H, curve[H]), textcoords="offset points",
                    xytext=offs[name], color=col, fontsize=10)
    ax.axhline(tol, ls="--", color="#6b7580", lw=1)
    ax.text(len(np.array(M.step_error(good, ref).detail["curve"])) - 1, tol * 1.3,
            "task tolerance %.2g" % tol, color="#6b7580", fontsize=10, ha="right")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("state error (RMSE, log)")
    ax.set_title("How many steps can I trust this rollout")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "error_growth.png"))
    plt.close(fig)
    print("wrote error_growth.png")


def consistency():
    ref, good, biased, unstable = _rolls()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for roll, name, col in [(biased, "linear drift", WARN), (unstable, "exponential", BAD)]:
        curve = np.array(M.step_error(roll, ref).detail["curve"])
        res = M.long_horizon_consistency(roll, ref)
        ax.plot(np.arange(len(curve)), curve, color=col, lw=2,
                label="%s (%s)" % (name, res.detail["kind"]))
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("error (log)")
    ax.set_title("Linear drift is survivable; exponential blow-up is not")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "consistency.png"))
    plt.close(fig)
    print("wrote consistency.png")


def calibration_fig():
    ref, good, _, _ = _rolls()
    # overconfident: same trajectory, tiny sigma
    over = rollout_from_model(MockLearnedModel(DT, step_noise=0.02, sigma=0.002),
                              [1.0, 0.0], (0.2 * np.sin(np.linspace(0, 6, 90))).reshape(-1, 1),
                              DT, MockTrueModel.channels)
    ks = np.linspace(0.5, 3.0, 6)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for roll, name, col in [(good, "honest uncertainty", GOOD),
                            (over, "overconfident", BAD)]:
        res = M.calibration(roll, ref, ks=tuple(ks))
        cov = [b["coverage"] for b in res.detail["bands"]]
        ax.plot([b["target"] for b in res.detail["bands"]], cov, "o-", color=col,
                label="%s (gap %.2f)" % (name, res.value))
    ax.plot([0, 1], [0, 1], "--", color="#6b7580", lw=1, label="perfect")
    ax.set_xlabel("claimed coverage (k-sigma target)")
    ax.set_ylabel("actual coverage")
    ax.set_title("A confidently-wrong model is worse than an uncertain one")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "calibration.png"))
    plt.close(fig)
    print("wrote calibration.png")


def sim2real_fig():
    ref, good, biased, _ = _rolls()
    tols = np.geomspace(0.01, 0.3, 24)
    fig, ax = plt.subplots(figsize=(8, 5))
    for roll, name, col in [(good, "well-calibrated", GOOD), (biased, "biased drift", WARN)]:
        H = [int(M.trust_horizon(roll, ref, float(t)).value) for t in tols]
        ax.plot(tols, H, "o-", color=col, label=name, ms=4)
    need = 40
    ax.axhline(need, ls="--", color=ACCENT, lw=1.2)
    ax.text(0.011, need + 1.5, "a plan needing %d steps" % need, color=ACCENT, fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel("task tolerance (error budget)")
    ax.set_ylabel("trust horizon (steps)")
    ax.set_title("Is the model good enough to plan through, for THIS task")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "sim2real_trust.png"))
    plt.close(fig)
    print("wrote sim2real_trust.png")


if __name__ == "__main__":
    error_growth()
    consistency()
    calibration_fig()
    sim2real_fig()
