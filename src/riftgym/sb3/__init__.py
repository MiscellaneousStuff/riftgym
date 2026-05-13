"""SB3-coupled helpers (vec env, snapshot pool, trainer).

Importing this subpackage requires ``stable-baselines3`` and ``sb3-contrib``.
Install with ``pip install riftgym[sb3]``.
"""

from __future__ import annotations

from importlib.util import find_spec

if find_spec("stable_baselines3") is None or find_spec("sb3_contrib") is None:
    raise ImportError(
        "riftgym.sb3 requires stable-baselines3 and sb3-contrib. "
        "Install with: pip install 'riftgym[sb3]'"
    )

from riftgym.sb3.policy import make_mirror_opp
from riftgym.sb3.snapshot_pool import SnapshotCheckpointCallback, SnapshotPool
from riftgym.sb3.thread_vec_env import ThreadVecEnv

__all__ = [
    "SnapshotCheckpointCallback",
    "SnapshotPool",
    "ThreadVecEnv",
    "make_mirror_opp",
]
