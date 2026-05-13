"""Unit tests for :class:`riftgym.sb3.thread_vec_env.ThreadVecEnv`.

Skipped wholesale if ``stable-baselines3`` isn't importable (the parent
:class:`VecEnv` lives in sb3). Tests run against tiny in-process fake
envs so no docker/bridge needed.
"""

from __future__ import annotations

import os

import pytest

# See test_snapshot_pool.py for why we gate on an env var instead of
# pytest.importorskip — some envs crash at C level on sb3 import.
if not os.environ.get("RIFTGYM_SB3_TESTS"):
    pytest.skip("set RIFTGYM_SB3_TESTS=1 to run sb3 tests", allow_module_level=True)

from typing import Any, ClassVar

import numpy as np
from gymnasium import spaces

from riftgym.sb3.thread_vec_env import ThreadVecEnv


class _FakeEnv:
    """Minimal gymnasium-shaped env. step returns deterministic obs/reward
    so tests can assert exactly what came out of each worker."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(self, idx: int, terminate_at: int = 10) -> None:
        self.idx = idx
        self.terminate_at = terminate_at
        self.step_count = 0
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def reset(self, *, seed: int | None = None, options: Any = None) -> tuple[Any, dict]:
        del seed, options
        self.step_count = 0
        return np.array([self.idx, 0.0], dtype=np.float32), {}

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict]:
        self.step_count += 1
        obs = np.array([self.idx, float(self.step_count)], dtype=np.float32)
        terminated = self.step_count >= self.terminate_at
        truncated = False
        reward = float(action)
        return obs, reward, terminated, truncated, {"step": self.step_count}

    def close(self) -> None:
        pass

    def get_wrapper_attr(self, name: str) -> Any:
        return getattr(self, name)


def _fns(n: int) -> list:
    return [(lambda i=i: _FakeEnv(i)) for i in range(n)]


def test_basic_reset_returns_one_obs_per_env() -> None:
    vec = ThreadVecEnv(_fns(3))
    try:
        obs = vec.reset()
        assert obs.shape == (3, 2)
        # First column carries the env idx; reset puts step_count = 0
        assert obs[:, 0].tolist() == [0, 1, 2]
        assert obs[:, 1].tolist() == [0.0, 0.0, 0.0]
    finally:
        vec.close()


def test_step_runs_workers_in_parallel() -> None:
    """Each worker advances its own step counter independently. After
    1 step we should see step_count=1 in every env's obs."""
    vec = ThreadVecEnv(_fns(3))
    try:
        vec.reset()
        actions = np.array([1, 2, 0])
        vec.step_async(actions)
        obs, rewards, dones, infos = vec.step_wait()
        assert obs.shape == (3, 2)
        assert obs[:, 1].tolist() == [1.0, 1.0, 1.0]
        assert rewards.tolist() == [1.0, 2.0, 0.0]
        assert dones.tolist() == [False, False, False]
        assert all(i["step"] == 1 for i in infos)
    finally:
        vec.close()


def test_auto_reset_on_done() -> None:
    """When an env signals terminated, ThreadVecEnv must auto-reset it
    (SB3 contract) and stash the pre-reset obs in
    ``info['terminal_observation']``."""
    vec = ThreadVecEnv([lambda: _FakeEnv(0, terminate_at=2)])
    try:
        vec.reset()
        vec.step_async(np.array([0]))
        _obs, _r, dones1, _ = vec.step_wait()
        assert not dones1[0]
        vec.step_async(np.array([0]))
        obs, _r, dones2, infos = vec.step_wait()
        assert dones2[0]
        assert "terminal_observation" in infos[0]
        # After auto-reset, the returned obs should match a fresh reset
        # (step_count=0 in the second column).
        assert obs[0, 1] == 0.0
    finally:
        vec.close()


def test_worker_exception_surfaces_through_step_wait() -> None:
    """If a worker raises (e.g. ServerDiedError in production), the
    main thread must see a RuntimeError from step_wait instead of
    hanging on res_q.get(). This was the failure mode the __exc__ tag
    was added to prevent."""

    class _ExplodingEnv(_FakeEnv):
        def step(self, action: int):  # type: ignore[override]
            raise RuntimeError("kaboom")

    vec = ThreadVecEnv([lambda: _ExplodingEnv(0)])
    try:
        vec.reset()
        vec.step_async(np.array([0]))
        with pytest.raises(RuntimeError, match="env 0 worker raised"):
            vec.step_wait()
    finally:
        vec.close()


def test_close_is_idempotent() -> None:
    vec = ThreadVecEnv(_fns(2))
    vec.close()
    vec.close()  # no exception


def test_rejects_empty_env_fns() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ThreadVecEnv([])


def test_get_attr_returns_inner_value() -> None:
    """get_attr must reach the inner env (via get_wrapper_attr), not
    just the outer Monitor/wrapper. We don't use Monitor here, but
    the call path is the same."""
    vec = ThreadVecEnv(_fns(3))
    try:
        idxs = vec.get_attr("idx")
        assert idxs == [0, 1, 2]
    finally:
        vec.close()
