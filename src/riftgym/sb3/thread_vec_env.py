"""Same-process VecEnv that parallelizes envs via threads, not subprocs.

Why threads over subprocs:

- :class:`stable_baselines3.common.vec_env.SubprocVecEnv` parallelizes
  across OS processes. Closures (notably a mirror-self-play opp_policy
  that captures the live PPO model) can't be shared with envs without
  IPC marshaling, which is impractical for the live-policy use case.
- :class:`stable_baselines3.common.vec_env.DummyVecEnv` shares state
  but runs envs sequentially — no parallelism.

:class:`riftgym.env.lol_gym.LoLGymEnv` is socket-bound: most of
``step()`` blocks on the bridge TCP read. Python releases the GIL
during socket I/O, so threads actually parallelize wall-clock here.

Caveats:

- GIL still serializes encode/decode/policy.predict() Python work.
  At ``frame_skip=8`` / ``RL_HZ=30``, each step waits ~270 ms on the
  socket and spends a few microseconds on Python — non-issue.
- ``step``/``reset`` on each env must be thread-safe relative to the
  others. Each LoLGymEnv either owns its own socket (n=1 path) or
  shares a :class:`riftgym.env.session.ServerSession` whose ``send_lock``
  serializes outbound writes, so they are.
- ``get_attr`` / ``set_attr`` / ``env_method`` assume workers are
  quiescent (between ``step_wait`` and the next ``step_async``). SB3
  only calls these from the trainer between rollouts, so fine.
"""

from __future__ import annotations

import contextlib
import queue
import threading
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

EnvFn = Callable[[], Any]
_SENTINEL: object = object()
_EXC_TAG: str = "__exc__"


class ThreadVecEnv(VecEnv):
    """One worker thread per env. See module docstring."""

    def __init__(self, env_fns: Sequence[EnvFn]) -> None:
        if not env_fns:
            raise ValueError("ThreadVecEnv needs at least one env_fn")
        # Build envs in the main thread first so any post-construction
        # setup (e.g. ``env.unwrapped.opp_policy = ...``) can hit them
        # via ``self.envs[i]`` before workers start. SB3's MaskablePPO
        # pokes at the inner env via ``get_wrapper_attr`` between
        # rollouts, so workers stay strictly inside step/reset.
        self.envs: list[Any] = [fn() for fn in env_fns]
        env0 = self.envs[0]
        super().__init__(len(self.envs), env0.observation_space, env0.action_space)
        self.metadata = env0.metadata

        self._cmd_qs: list[queue.Queue[Any]] = [queue.Queue() for _ in self.envs]
        self._res_qs: list[queue.Queue[Any]] = [queue.Queue() for _ in self.envs]
        self._workers: list[threading.Thread] = []
        for i, env in enumerate(self.envs):
            t = threading.Thread(
                target=self._worker_loop,
                args=(i, env, self._cmd_qs[i], self._res_qs[i]),
                name=f"ThreadVecEnv-{i}",
                daemon=True,
            )
            t.start()
            self._workers.append(t)

        self._actions: Any = None
        self._closed = False

    @staticmethod
    def _worker_loop(
        idx: int,
        env: Any,
        cmd_q: queue.Queue[Any],
        res_q: queue.Queue[Any],
    ) -> None:
        """Per-env worker. Blocks on ``cmd_q``. Commands:

        * ``("step", action)`` → res_q.put((obs, reward, terminated, truncated, info))
        * ``("reset", (seed, options))`` → res_q.put((obs, info))
        * ``(_SENTINEL, None)`` → close env, exit

        Exception handling: if ``env.step`` / ``env.reset`` raises, push
        the exception via ``(_EXC_TAG, exc)`` and keep looping. Without
        this, an exception (e.g. ``ServerDiedError`` from a dead bridge
        socket) kills the worker thread silently and ``step_wait`` hangs
        forever on ``res_q.get()``.
        """
        del idx  # used for thread name only
        while True:
            cmd, payload = cmd_q.get()
            if cmd is _SENTINEL:
                with contextlib.suppress(BaseException):
                    env.close()
                return
            try:
                if cmd == "step":
                    obs, reward, terminated, truncated, info = env.step(payload)
                    res_q.put((obs, reward, terminated, truncated, info))
                elif cmd == "reset":
                    seed, options = payload
                    obs, info = env.reset(seed=seed, options=options)
                    res_q.put((obs, info))
                else:
                    res_q.put((_EXC_TAG, RuntimeError(f"unknown cmd: {cmd!r}")))
            except BaseException as exc:
                res_q.put((_EXC_TAG, exc))

    @staticmethod
    def _check_exc(result: Any, idx: int, op: str) -> None:
        if (
            isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[0], str)
            and result[0] == _EXC_TAG
        ):
            raise RuntimeError(f"env {idx} worker raised during {op}") from result[1]

    def step_async(self, actions: NDArray[Any]) -> None:
        self._actions = actions
        for q, a in zip(self._cmd_qs, actions, strict=True):
            q.put(("step", a))

    def step_wait(
        self,
    ) -> tuple[NDArray[Any], NDArray[np.float32], NDArray[np.bool_], list[dict[str, Any]]]:
        n = self.num_envs
        obs_list: list[Any] = [None] * n
        rewards = np.zeros(n, dtype=np.float32)
        dones = np.zeros(n, dtype=bool)
        infos: list[dict[str, Any]] = [{} for _ in range(n)]

        for i, q in enumerate(self._res_qs):
            result = q.get()
            self._check_exc(result, i, "step")
            obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
            info = dict(info) if info else {}
            if truncated and not terminated:
                info["TimeLimit.truncated"] = True
            if done:
                # SB3 contract: terminal_observation in info, obs slot
                # gets the reset obs. Auto-reset happens in the main
                # thread (worker just finished its step); workers stay
                # blocked on cmd_q for the next round.
                info["terminal_observation"] = obs
                reset_obs, _reset_info = self.envs[i].reset(
                    seed=self._seeds[i] if self._seeds else None,
                    options=self._options[i] if self._options else None,
                )
                obs = reset_obs
            obs_list[i] = obs
            rewards[i] = reward
            dones[i] = done
            infos[i] = info

        stacked = np.stack(obs_list, axis=0)
        return stacked, rewards, dones, infos

    def reset(self) -> NDArray[Any]:
        for i, q in enumerate(self._cmd_qs):
            seed = self._seeds[i] if self._seeds else None
            options = self._options[i] if self._options else None
            q.put(("reset", (seed, options)))
        obs_list: list[Any] = [None] * self.num_envs
        for i, q in enumerate(self._res_qs):
            result = q.get()
            self._check_exc(result, i, "reset")
            obs, info = result
            obs_list[i] = obs
            self.reset_infos[i] = dict(info) if info else {}
        self._reset_seeds()
        self._reset_options()
        return np.stack(obs_list, axis=0)

    def close(self) -> None:
        if self._closed:
            return
        for q in self._cmd_qs:
            q.put((_SENTINEL, None))
        for t in self._workers:
            t.join(timeout=5.0)
        self._closed = True

    # ---- synchronous helpers (assume workers quiescent) -------------------

    def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:
        # ``get_wrapper_attr`` traverses gymnasium Wrapper layers (e.g.
        # Monitor). Plain ``getattr`` only sees the outermost wrapper
        # and breaks sb3-contrib's MaskablePPO probing for
        # ``action_masks`` on the inner LoLGymEnv through Monitor.
        idxs = self._get_indices(indices)
        return [self.envs[i].get_wrapper_attr(attr_name) for i in idxs]

    def set_attr(self, attr_name: str, value: Any, indices: Any = None) -> None:
        idxs = self._get_indices(indices)
        for i in idxs:
            setattr(self.envs[i], attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: Any = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        idxs = self._get_indices(indices)
        return [
            self.envs[i].get_wrapper_attr(method_name)(*method_args, **method_kwargs)
            for i in idxs
        ]

    def env_is_wrapped(self, wrapper_class: type, indices: Any = None) -> list[bool]:
        from stable_baselines3.common.vec_env.util import is_wrapped

        idxs = self._get_indices(indices)
        return [is_wrapped(self.envs[i], wrapper_class) for i in idxs]

    def get_images(self) -> list[Any]:  # type: ignore[override]
        return [None] * self.num_envs
