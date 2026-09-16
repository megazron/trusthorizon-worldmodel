"""wmeval CLI."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

import numpy as np

from . import metrics as M
from . import report as R
from .models import MockLearnedModel, MockTrueModel
from .rollout import Rollout, rollout_from_model
from .sim2real import sim_to_real_trust
from .suite import Scenario, evaluate


def _load_model(spec):
    if spec in ("mock", "mock-good"):
        return MockLearnedModel(bias=0.0, step_noise=0.002, sigma=0.02)
    if spec == "mock-biased":
        return MockLearnedModel(bias=0.01, divergence=0.002, sigma=0.02)
    if spec == "mock-unstable":
        return MockLearnedModel(divergence=0.004, unstable=True, energy_leak=0.02)
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)()


def _scenarios_from_dir(path):
    scs = []
    for fn in sorted(os.listdir(path)):
        if fn.endswith(".json"):
            d = json.load(open(os.path.join(path, fn)))
            scs.append(Scenario(d.get("name", fn), d["initial_state"], d["actions"],
                                d.get("dt", 0.02)))
    return scs


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="wmeval",
        description="Decide whether a world model's rollout can be trusted, "
                    "and for how many steps.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval", help="compare two aligned rollouts")
    e.add_argument("--pred", required=True)
    e.add_argument("--ref", required=True)
    e.add_argument("--tol", type=float, default=0.05)
    e.add_argument("-o", "--out")

    s = sub.add_parser("suite", help="run a mock/plugged model over a scenario dir")
    s.add_argument("--model", default="mock")
    s.add_argument("--scenarios", required=True)
    s.add_argument("--tol", type=float, default=0.05)
    s.add_argument("--format", choices=["text", "json", "html"], default="text")
    s.add_argument("-o", "--out")

    t = sub.add_parser("trust", help="is the model good enough for a task?")
    t.add_argument("--pred", required=True)
    t.add_argument("--ref", required=True)
    t.add_argument("--task-tol", type=float, required=True)
    t.add_argument("--needed-steps", type=int)

    sub.add_parser("selftest", help="run the built-in known-answer checks")

    a = ap.parse_args(argv)

    if a.cmd == "eval":
        pred = Rollout.load_json(a.pred)
        ref = Rollout.load_json(a.ref)
        for res in (M.step_error(pred, ref), M.trust_horizon(pred, ref, a.tol),
                    M.long_horizon_consistency(pred, ref),
                    M.divergence_onset(pred, ref), M.calibration(pred, ref)):
            print("%-24s %s" % (res.name, res.explain))
        if a.out:
            json.dump({"trust_horizon": M.trust_horizon(pred, ref, a.tol).detail},
                      open(a.out, "w"), indent=2)
            print("wrote", a.out)
        return 0

    if a.cmd == "suite":
        rep = evaluate(_load_model(a.model), MockTrueModel(),
                       _scenarios_from_dir(a.scenarios), tol=a.tol,
                       vel_limit=50.0, passive_energy=MockTrueModel.energy)
        text = {"text": R.to_text, "json": R.to_json, "html": R.to_html}[a.format](rep)
        if a.out:
            open(a.out, "w").write(text)
            print("wrote", a.out)
        else:
            print(text)
        return 0

    if a.cmd == "trust":
        pred = Rollout.load_json(a.pred)
        ref = Rollout.load_json(a.ref)
        v = sim_to_real_trust(pred, ref, a.task_tol, a.needed_steps)
        print(v.explain)
        return 0 if (v.usable is None or v.usable) else 1

    if a.cmd == "selftest":
        return _selftest()
    return 0


def _selftest():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print("  %-52s %s%s" % (name, "OK" if cond else "FAIL",
                                (" -- " + detail) if detail else ""))

    dt = 0.02
    acts = np.zeros((80, 1))
    x0 = [1.0, 0.0]
    truth = MockTrueModel(dt)
    ref = rollout_from_model(truth, x0, acts, dt, MockTrueModel.channels)

    good = rollout_from_model(MockLearnedModel(dt, bias=0.0, step_noise=0.0005, sigma=0.02),
                              x0, acts, dt, MockTrueModel.channels)
    unstable = rollout_from_model(
        MockLearnedModel(dt, divergence=0.004, unstable=True, energy_leak=0.05),
        x0, acts, dt, MockTrueModel.channels)

    th_g = M.trust_horizon(good, ref, 0.05).value
    th_u = M.trust_horizon(unstable, ref, 0.05).value
    check("a good model is trusted longer than an unstable one", th_g > th_u,
          "good %d vs unstable %d steps" % (th_g, th_u))
    check("the unstable model is classified exponential",
          M.long_horizon_consistency(unstable, ref).detail["kind"] == "exponential")
    check("the energy-leaking model fails plausibility",
          M.physics_plausibility(unstable, passive_energy=MockTrueModel.energy).value > 0)
    check("the passive true model passes plausibility",
          M.physics_plausibility(ref, passive_energy=MockTrueModel.energy).value == 0)
    print("trusthorizon-worldmodel selftest %s" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
