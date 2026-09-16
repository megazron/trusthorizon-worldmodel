import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from world_model_eval import metrics as M                        # noqa: E402
from world_model_eval import report as R                         # noqa: E402
from world_model_eval.models import MockLearnedModel, MockTrueModel  # noqa: E402
from world_model_eval.rollout import (Rollout, RolloutError,     # noqa: E402
                                      check_aligned, rollout_from_model)
from world_model_eval.sim2real import sim_to_real_trust          # noqa: E402
from world_model_eval.suite import Scenario, evaluate            # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DT = 0.02


def _roll(states, actions=None, channels=("x", "v")):
    states = np.asarray(states, float)
    T = len(states) - 1
    actions = np.zeros((T, 1)) if actions is None else np.asarray(actions, float)
    times = np.arange(T + 1) * DT
    return Rollout(states, actions, times, list(channels))


# ---- rollout ---------------------------------------------------------------
def test_rollout_shape_validation():
    with pytest.raises(RolloutError):
        Rollout(np.zeros((5, 2)), np.zeros((3, 1)), np.arange(5) * DT)  # T mismatch


def test_times_must_increase():
    with pytest.raises(RolloutError):
        Rollout(np.zeros((3, 2)), np.zeros((2, 1)), np.array([0.0, 0.0, 0.1]))


def test_align_refuses_different_initial_state():
    a = _roll([[0, 0], [1, 0], [2, 0]])
    b = _roll([[9, 0], [1, 0], [2, 0]])
    with pytest.raises(RolloutError):
        check_aligned(a, b)


def test_align_refuses_different_actions():
    a = _roll([[0, 0], [1, 0]], actions=[[0.0]])
    b = _roll([[0, 0], [1, 0]], actions=[[1.0]])
    with pytest.raises(RolloutError):
        check_aligned(a, b)


def test_json_round_trip(tmp_path):
    r = _roll([[0, 0], [1, 1], [2, 2]])
    p = tmp_path / "r.json"
    r.save_json(str(p))
    r2 = Rollout.load_json(str(p))
    assert np.allclose(r.states, r2.states) and r2.channels == r.channels


# ---- step_error / trust_horizon -------------------------------------------
def test_step_error_known():
    ref = _roll([[0, 0], [0, 0], [0, 0]])
    pred = _roll([[0, 0], [3, 4], [0, 0]])   # RMSE at step1 = sqrt((9+16)/2)
    curve = M.step_error(pred, ref).detail["curve"]
    assert abs(curve[1] - np.sqrt(25 / 2)) < 1e-9


def test_trust_horizon_first_crossing():
    # errors: 0, .01, .02, .2, .01  -> with tol .05 trust = 2 (stops at first cross)
    ref = _roll([[0, 0]] * 5)
    errs = [0.0, 0.01, 0.02, 0.2, 0.01]
    states = [[e * np.sqrt(2), 0.0] for e in errs]  # RMSE over 2 dims of [a,0] = a/sqrt2
    pred = _roll(states)
    th = M.trust_horizon(pred, ref, 0.05)
    assert th.value == 2
    assert abs(th.detail["seconds"] - 2 * DT) < 1e-9


def test_trust_horizon_full():
    ref = _roll([[0, 0]] * 5)
    pred = _roll([[0.0, 0]] + [[0.001, 0]] * 4)
    assert M.trust_horizon(pred, ref, 0.05).value == 4


# ---- consistency -----------------------------------------------------------
def test_linear_growth_classified_linear():
    ref = _roll([[0, 0]] * 30)
    pred = _roll([[0.01 * t * np.sqrt(2), 0] for t in range(30)])
    assert M.long_horizon_consistency(pred, ref).detail["kind"] == "linear"


def test_exponential_growth_classified_exponential():
    ref = _roll([[0, 0]] * 30)
    pred = _roll([[0.0, 0]] + [[1e-4 * (1.3 ** t) * np.sqrt(2), 0] for t in range(1, 30)])
    assert M.long_horizon_consistency(pred, ref).detail["kind"] == "exponential"


# ---- plausibility ----------------------------------------------------------
def test_energy_injection_caught():
    x0 = [1.0, 0.0]
    acts = np.zeros((60, 1))
    leaky = rollout_from_model(MockLearnedModel(DT, energy_leak=0.05), x0, acts, DT,
                               MockTrueModel.channels)
    res = M.physics_plausibility(leaky, passive_energy=MockTrueModel.energy)
    assert res.value > 0
    assert any(c["check"] == "passive_energy" and not c["ok"] for c in res.detail["checks"])


def test_passive_true_model_conserves_energy():
    x0 = [1.0, 0.0]
    acts = np.zeros((60, 1))
    ref = rollout_from_model(MockTrueModel(DT), x0, acts, DT, MockTrueModel.channels)
    assert M.physics_plausibility(ref, passive_energy=MockTrueModel.energy).value == 0


def test_nan_caught():
    bad = _roll([[0, 0], [np.nan, 0], [0, 0]])
    res = M.physics_plausibility(bad)
    assert any(c["check"] == "finite" and not c["ok"] for c in res.detail["checks"])


def test_velocity_limit_caught():
    fast = _roll([[0, 0], [0, 100.0], [0, 0]])
    res = M.physics_plausibility(fast, vel_limit=10.0)
    assert any(c["check"] == "velocity_limit" and not c["ok"] for c in res.detail["checks"])


# ---- calibration -----------------------------------------------------------
def test_calibration_good_when_sigma_matches_error():
    ref = _roll([[0, 0]] * 40)
    rng = np.random.default_rng(0)
    sig = 0.1
    states = [[0.0, 0.0]]
    errs = rng.normal(0, sig, size=(39, 2))
    for e in errs:
        states.append(e.tolist())
    pred = _roll(states)
    pred.meta["sigma"] = np.full((39, 2), sig)
    ece = M.calibration(pred, ref).value
    assert ece < 0.1


def test_calibration_bad_when_overconfident():
    ref = _roll([[0, 0]] * 40)
    states = [[0.0, 0.0]] + [[0.5, 0.5]] * 39   # big error
    pred = _roll(states)
    pred.meta["sigma"] = np.full((39, 2), 0.001)  # tiny claimed uncertainty
    assert M.calibration(pred, ref).value > 0.3


def test_calibration_absent_without_sigma():
    ref = _roll([[0, 0]] * 5)
    pred = _roll([[0.0, 0]] + [[0.1, 0]] * 4)
    assert M.calibration(pred, ref).detail["available"] is False


# ---- divergence ------------------------------------------------------------
def test_divergence_onset_at_known_step():
    ref = _roll([[0, 0]] * 20)
    errs = [0.001] * 20
    errs[0] = 0.0
    errs[10] = 0.5
    for t in range(11, 20):
        errs[t] = 0.5
    pred = _roll([[e * np.sqrt(2), 0] for e in errs])
    assert M.divergence_onset(pred, ref).value == 10


# ---- sim2real --------------------------------------------------------------
def test_sim2real_yes_no():
    ref = _roll([[0, 0]] * 20)
    pred = _roll([[0.001 * t * np.sqrt(2), 0] for t in range(20)])  # linear
    yes = sim_to_real_trust(pred, ref, task_tol=0.05, needed_steps=5)
    no = sim_to_real_trust(pred, ref, task_tol=0.05, needed_steps=100)
    assert yes.usable is True and no.usable is False


# ---- suite / report --------------------------------------------------------
def test_suite_and_report_round_trip():
    scs = [Scenario("s%d" % i, [1.0, 0.0], np.zeros((40, 1)).tolist(), DT)
           for i in range(3)]
    rep = evaluate(MockLearnedModel(DT, step_noise=0.001, sigma=0.02),
                   MockTrueModel(DT), scs, tol=0.05,
                   passive_energy=MockTrueModel.energy)
    import json
    d = json.loads(R.to_json(rep))
    assert d["summary"]["n_scenarios"] == 3
    assert R.to_html(rep).startswith("<!doctype html>")
    assert "world-model-eval report" in R.to_text(rep)


def test_good_model_trusted_longer_than_unstable():
    x0, acts = [1.0, 0.0], np.zeros((80, 1))
    ref = rollout_from_model(MockTrueModel(DT), x0, acts, DT, MockTrueModel.channels)
    good = rollout_from_model(MockLearnedModel(DT, step_noise=0.0005), x0, acts, DT,
                              MockTrueModel.channels)
    bad = rollout_from_model(MockLearnedModel(DT, divergence=0.004, unstable=True),
                             x0, acts, DT, MockTrueModel.channels)
    assert M.trust_horizon(good, ref, 0.05).value > M.trust_horizon(bad, ref, 0.05).value


# ---- CLI -------------------------------------------------------------------
def test_cli_selftest():
    r = subprocess.run([sys.executable, "-m", "world_model_eval.cli", "selftest"],
                       cwd=ROOT, env=dict(os.environ, PYTHONPATH="src"),
                       capture_output=True, text=True)
    assert r.returncode == 0 and "PASSED" in r.stdout


def test_cli_eval(tmp_path):
    x0, acts = [1.0, 0.0], np.zeros((40, 1))
    ref = rollout_from_model(MockTrueModel(DT), x0, acts, DT, MockTrueModel.channels)
    pred = rollout_from_model(MockLearnedModel(DT, step_noise=0.001), x0, acts, DT,
                              MockTrueModel.channels)
    pj, rj = tmp_path / "p.json", tmp_path / "r.json"
    pred.save_json(str(pj))
    ref.save_json(str(rj))
    r = subprocess.run([sys.executable, "-m", "world_model_eval.cli", "eval",
                        "--pred", str(pj), "--ref", str(rj), "--tol", "0.05"],
                       cwd=ROOT, env=dict(os.environ, PYTHONPATH="src"),
                       capture_output=True, text=True)
    assert r.returncode == 0 and "trust_horizon" in r.stdout
