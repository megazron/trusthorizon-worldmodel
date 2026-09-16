"""A rollout: a sequence of states with the actions that produced them.

A rollout is the unit this whole package reasons over. You have a PREDICTED
rollout from a world model and a REFERENCE rollout from ground truth (real
logs, or a simulator you trust), aligned by the same initial state and the
same action sequence. Every metric compares two aligned rollouts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


class RolloutError(ValueError):
    """A rollout is malformed or two rollouts do not align; the message says how."""


@dataclass
class Rollout:
    """states: (T+1, d)  actions: (T, a)  times: (T+1,) monotonic seconds.

    `channels` optionally names the state dimensions so plausibility checks can
    find, e.g., the velocity or contact-penetration columns by name.
    """

    states: np.ndarray
    actions: np.ndarray
    times: np.ndarray
    channels: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.states = np.asarray(self.states, float)
        self.actions = np.asarray(self.actions, float)
        self.times = np.asarray(self.times, float)
        if self.states.ndim != 2:
            raise RolloutError("states must be 2-D (T+1, d), got shape %r"
                               % (self.states.shape,))
        T1, d = self.states.shape
        if self.actions.ndim != 2 or self.actions.shape[0] != T1 - 1:
            raise RolloutError(
                "actions must be (T, a) with T = len(states)-1 = %d, got shape %r"
                % (T1 - 1, self.actions.shape))
        if self.times.shape != (T1,):
            raise RolloutError("times must be (T+1,) = (%d,), got %r"
                               % (T1, self.times.shape))
        if np.any(np.diff(self.times) <= 0):
            raise RolloutError("times must be strictly increasing")
        if self.channels and len(self.channels) != d:
            raise RolloutError("channels names %d dims but states have %d"
                               % (len(self.channels), d))

    @property
    def horizon(self) -> int:
        """Number of steps, T (states has T+1 entries)."""
        return self.states.shape[0] - 1

    @property
    def dt(self) -> float:
        return float(np.median(np.diff(self.times)))

    def channel(self, name):
        """The column for a named channel, or None if it is not present."""
        if name in self.channels:
            return self.states[:, self.channels.index(name)]
        return None

    def to_dict(self):
        return {"states": self.states.tolist(), "actions": self.actions.tolist(),
                "times": self.times.tolist(), "channels": list(self.channels),
                "meta": dict(self.meta)}

    @classmethod
    def from_dict(cls, d):
        return cls(np.asarray(d["states"], float), np.asarray(d["actions"], float),
                   np.asarray(d["times"], float), list(d.get("channels", [])),
                   dict(d.get("meta", {})))

    def save_json(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load_json(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save_npz(self, path):
        np.savez(path, states=self.states, actions=self.actions,
                 times=self.times, channels=np.array(self.channels, dtype=object))

    @classmethod
    def load_npz(cls, path):
        z = np.load(path, allow_pickle=True)
        ch = list(z["channels"]) if "channels" in z else []
        return cls(z["states"], z["actions"], z["times"], ch)


def check_aligned(pred: Rollout, ref: Rollout, state_tol: float = 1e-6,
                  action_tol: float = 1e-6) -> None:
    """Refuse to compare two rollouts that are not the same experiment.

    Same horizon, same initial state, same action sequence. A metric computed
    over misaligned rollouts is a number nobody should trust, so this raises
    rather than silently truncating.
    """
    if pred.horizon != ref.horizon:
        raise RolloutError("horizons differ: pred %d vs ref %d"
                           % (pred.horizon, ref.horizon))
    if pred.states.shape[1] != ref.states.shape[1]:
        raise RolloutError("state dimensions differ: pred %d vs ref %d"
                           % (pred.states.shape[1], ref.states.shape[1]))
    d0 = float(np.max(np.abs(pred.states[0] - ref.states[0]))) if pred.horizon >= 0 else 0.0
    if d0 > state_tol:
        raise RolloutError(
            "initial states differ by %.3g > %.3g -- these are not the same "
            "rollout; a world model must be reset to the reference's start"
            % (d0, state_tol))
    if pred.actions.size and ref.actions.size:
        da = float(np.max(np.abs(pred.actions - ref.actions)))
        if da > action_tol:
            raise RolloutError(
                "action sequences differ by %.3g > %.3g -- compare rollouts "
                "driven by the SAME actions, or the error is not the model's"
                % (da, action_tol))


def rollout_from_model(model, initial_state, actions, dt: float = 0.02,
                       channels=None) -> Rollout:
    """Roll a WorldModel forward under a fixed action sequence.

    Collects any per-step uncertainty the model returns into meta['sigma'].
    """
    actions = np.asarray(actions, float)
    if actions.ndim == 1:
        actions = actions[:, None]
    model.reset(np.asarray(initial_state, float))
    states = [np.asarray(initial_state, float)]
    sigmas = []
    for a in actions:
        out = model.step(a)
        if isinstance(out, tuple):
            nxt, sig = out
        else:
            nxt, sig = out, None
        states.append(np.asarray(nxt, float))
        if sig is not None:
            sigmas.append(np.asarray(sig, float))
    T = len(actions)
    times = np.arange(T + 1) * dt
    meta = {}
    if sigmas:
        meta["sigma"] = np.array(sigmas)          # (T, d), one per step's next state
    return Rollout(np.array(states), actions, times, channels or [], meta)
