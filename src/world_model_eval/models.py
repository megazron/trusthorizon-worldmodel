"""Mock dynamics so every metric can be exercised against a known answer.

Nothing here is a real world model. `MockTrueModel` is the ground truth; the
`MockLearnedModel` wraps it and injects a chosen error (bias, per-step noise, a
slow divergence, an energy leak, and optional uncertainty), so a test can ask
"does the trust horizon come out where I built the divergence" and get a
definite yes.
"""
from __future__ import annotations

import numpy as np


class MockTrueModel:
    """A damped point mass in 1-D with a soft floor at x=0 (a toy 'contact').

    State is [x, v]. The action is a scalar force. Passive energy (no action)
    never increases -- that is the invariant the plausibility checks lean on.
    Channels: ['x', 'v']; a derived contact-penetration channel is exposed by
    `penetration()` for the plausibility test.
    """

    channels = ["x", "v"]

    def __init__(self, dt: float = 0.02, damping: float = 0.5, k_floor: float = 40.0):
        self.dt = float(dt)
        self.damping = float(damping)
        self.k_floor = float(k_floor)
        self.s = np.zeros(2)

    def reset(self, state):
        self.s = np.asarray(state, float).copy()
        return self.s.copy()

    def _accel(self, x, v, f):
        contact = -self.k_floor * min(x, 0.0)      # push back up when x < 0
        return f - self.damping * v + contact

    def step(self, action):
        f = float(np.ravel(action)[0])
        x, v = self.s
        a = self._accel(x, v, f)
        v = v + a * self.dt
        x = x + v * self.dt
        self.s = np.array([x, v])
        return self.s.copy()

    @staticmethod
    def energy(states):
        """0.5 v^2 (unit mass); potential is 0 in the free region."""
        v = states[:, 1]
        return 0.5 * v * v


class MockLearnedModel:
    """Wraps the true model and injects a chosen, known error.

    bias           constant added to the next state each step (systematic).
    step_noise     std of zero-mean noise added each step (seeded).
    divergence     per-step multiplicative growth of a drift term; >0 makes
                   error grow linearly, the `unstable` flag makes it exponential.
    unstable       feed a fraction of the error back into the state so it
                   compounds -- an exponential blow-up.
    energy_leak    inject energy each step (breaks the passive-energy invariant,
                   so the plausibility check must catch it).
    sigma          if not None, report this per-step uncertainty (a std). Set it
                   to match the real error for a calibrated model, or far below
                   it for an overconfident one.
    """

    channels = ["x", "v"]

    def __init__(self, dt=0.02, damping=0.5, k_floor=40.0, bias=0.0,
                 step_noise=0.0, divergence=0.0, unstable=False, energy_leak=0.0,
                 sigma=None, seed=0):
        self.true = MockTrueModel(dt, damping, k_floor)
        self.dt = float(dt)
        self.bias = np.asarray(bias, float) if np.ndim(bias) else float(bias)
        self.step_noise = float(step_noise)
        self.divergence = float(divergence)
        self.unstable = bool(unstable)
        self.energy_leak = float(energy_leak)
        self.sigma = None if sigma is None else float(sigma)
        self.rng = np.random.default_rng(seed)
        self._k = 0
        self._acc = None            # accumulated drift, for the unstable case

    def reset(self, state):
        self.true.reset(state)
        self._k = 0
        self._acc = np.zeros(2)
        return self.true.s.copy()

    def step(self, action):
        # Advance the model's OWN state (self.true.s carries the learned
        # trajectory, so any error already in it propagates forward).
        nxt = self.true.step(action)
        self._k += 1
        err = np.zeros_like(nxt)
        err = err + self.bias
        if self.step_noise:
            err = err + self.rng.normal(0.0, self.step_noise, size=nxt.shape)
        if self.divergence and not self.unstable:
            err = err + self.divergence * self._k          # linear-in-time drift
        if self.energy_leak:
            nxt = nxt + np.array([0.0, self.energy_leak])  # inject velocity = energy
        if self.unstable:
            # compound the accumulated drift each step -> exponential growth.
            rate = self.divergence if self.divergence else 0.05
            seed = self.divergence if self.divergence else 0.001
            self._acc = self._acc * (1.0 + max(rate, 0.02) * 10.0) + seed
            err = err + self._acc
        nxt = nxt + err
        self.true.s = nxt.copy()      # the error is now part of the state it carries
        if self.sigma is not None:
            return nxt.copy(), np.full_like(nxt, self.sigma)
        return nxt.copy()
