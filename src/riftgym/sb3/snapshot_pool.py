"""Snapshot pool for self-play opp diversification (OpenAI Five Appendix N).

Mirror self-play with a 100% live opp_policy converges to a narrow
defensive local best-response — both sides defend each other into a
degenerate equilibrium that doesn't generalise. OpenAI Five (and
earlier Bansal et al. 2017) mitigate this by mixing past parameters
into the opp distribution: with some probability ``p`` the opp_policy
uses a randomly-sampled past checkpoint instead of the live policy.
This anchors the policy against forgetting how to handle aggressive /
off-equilibrium opponents.

Usage from the trainer::

    pool = SnapshotPool(capacity=8)
    for env in vec_env.envs:
        env.unwrapped.snapshot_pool = pool
        env.unwrapped.snapshot_prob = 0.2
    ckpt_cb = SnapshotCheckpointCallback(
        pool=pool,
        save_freq=...,
        save_path=...,
        name_prefix="ppo_lol",
    )
"""

from __future__ import annotations

import collections
import pathlib
import random
from typing import Any

from sb3_contrib import MaskablePPO  # pyright: ignore[reportMissingImports]
from stable_baselines3.common.callbacks import CheckpointCallback


class SnapshotPool:
    """Bounded pool of past model checkpoints for opp_policy sampling.

    Stores up to ``capacity`` checkpoint paths (most recent kept; old
    ones drop off the deque). Loaded models are cached in memory so the
    per-episode :meth:`sample` call is a dict lookup. Capacity is
    measured in checkpoints, not bytes — at typical policy size (~few
    MB each), ``capacity=8`` fits in <50 MB.
    """

    def __init__(self, capacity: int = 8) -> None:
        self.capacity = capacity
        # Deque of paths; oldest drops off when at capacity.
        self._paths: collections.deque[str] = collections.deque(maxlen=capacity)
        # path -> loaded MaskablePPO. Trimmed to match _paths after each add.
        self._cache: dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self._paths)

    @property
    def paths(self) -> list[str]:
        return list(self._paths)

    def add_path(self, path: str | pathlib.Path) -> None:
        """Register a checkpoint. Loads + caches eagerly so the first
        :meth:`sample` after add doesn't pay disk-read latency on a hot
        training step.
        """
        path_str = str(path)
        if path_str in self._paths:
            # Already registered — no-op (handles double-call from the
            # CheckpointCallback wrapper if save_freq logic is jittery).
            return
        self._paths.append(path_str)
        if path_str not in self._cache:
            self._cache[path_str] = MaskablePPO.load(path_str, device="cpu")
        # Drop cached models whose path fell off the deque.
        kept = set(self._paths)
        for cached in list(self._cache.keys()):
            if cached not in kept:
                del self._cache[cached]

    def sample(self, rng: random.Random | None = None) -> Any | None:
        """Random model from the pool. Returns ``None`` if empty (caller
        should fall back to the live policy).

        Pass ``rng`` for deterministic sampling in tests; defaults to
        the module-level :mod:`random` so callers can leave it alone.
        """
        if not self._paths:
            return None
        chooser = rng if rng is not None else random
        path = chooser.choice(list(self._paths))
        return self._cache.get(path)


class SnapshotCheckpointCallback(CheckpointCallback):
    """:class:`CheckpointCallback` that also registers each saved
    checkpoint into a :class:`SnapshotPool`. Drop-in replacement: same
    constructor args plus ``pool``.
    """

    def __init__(self, pool: SnapshotPool, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pool = pool

    def _on_step(self) -> bool:
        result = super()._on_step()
        # CheckpointCallback writes when ``n_calls % save_freq == 0``
        # (and ``n_calls > 0``). The path it just wrote can be
        # reconstructed via the same naming convention parent uses.
        if self.n_calls > 0 and self.n_calls % self.save_freq == 0:
            saved = pathlib.Path(self.save_path) / (
                f"{self.name_prefix}_{self.num_timesteps}_steps.zip"
            )
            if saved.exists():
                self.pool.add_path(str(saved))
        return result
