"""Unit tests for :class:`riftgym.sb3.trainer.TrainConfig`.

Gated on sb3 like the other sb3 tests because trainer.py imports
sb3 transitively (TrainConfig itself is sb3-free but lives in the
sb3 subpackage, which has an import-time gate).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

if not os.environ.get("RIFTGYM_SB3_TESTS"):
    pytest.skip("set RIFTGYM_SB3_TESTS=1 to run sb3 tests", allow_module_level=True)

from unittest.mock import MagicMock

from riftgym.sb3.trainer import (
    DEFAULT_FRAME_SKIP,
    DEFAULT_MAX_EP_STEPS,
    DEFAULT_TOTAL_STEPS,
    TrainConfig,
    _n_envs,
    _resolve_batch_size,
)


def _rc():
    """Run config stub — TrainConfig requires one but doesn't introspect it
    until train_ppo() runs."""
    return MagicMock()


def test_defaults_match_baseline() -> None:
    cfg = TrainConfig(run_config=_rc())
    assert cfg.n_servers == 4
    assert cfg.lanes == 5
    assert cfg.mirror_both_sides is True
    assert cfg.total_timesteps == DEFAULT_TOTAL_STEPS
    assert cfg.frame_skip == DEFAULT_FRAME_SKIP
    assert cfg.max_episode_steps == DEFAULT_MAX_EP_STEPS
    assert cfg.opp_snapshot_prob == 0.0
    assert cfg.reset_jitter_hp == 0.0
    assert cfg.batch_size is None  # signals auto-scale
    assert cfg.n_steps_per_env is None


def test_log_dir_normalized_to_path() -> None:
    """Accept str or Path; always store Path so downstream code is
    type-stable."""
    cfg = TrainConfig(run_config=_rc(), log_dir="runs/exp01")  # type: ignore[arg-type]
    assert isinstance(cfg.log_dir, Path)
    assert cfg.log_dir == Path("runs/exp01")


def test_n_envs_default() -> None:
    """4 servers x 5 lanes x 2 sides = 40 envs (the baseline recipe)."""
    cfg = TrainConfig(run_config=_rc())
    assert _n_envs(cfg) == 40


def test_n_envs_without_mirror() -> None:
    cfg = TrainConfig(run_config=_rc(), mirror_both_sides=False)
    assert _n_envs(cfg) == 20


def test_n_envs_single() -> None:
    cfg = TrainConfig(
        run_config=_rc(), n_servers=1, lanes=1, mirror_both_sides=False
    )
    assert _n_envs(cfg) == 1


def test_batch_size_autoscale_at_baseline() -> None:
    """40 envs x 128 n_steps = 5120 rollout. 5120 // 80 = 64 → max(64, 64)
    = 64. Validates the "preserve ~80 batches per rollout" rule."""
    cfg = TrainConfig(run_config=_rc())
    bs = _resolve_batch_size(cfg, n_envs=40, n_steps_per_env=128)
    assert bs == 64


def test_batch_size_autoscale_above_baseline() -> None:
    """160 envs x 128 = 20480 rollout, 20480 // 80 = 256. Same rule."""
    cfg = TrainConfig(run_config=_rc())
    bs = _resolve_batch_size(cfg, n_envs=160, n_steps_per_env=128)
    assert bs == 256


def test_batch_size_autoscale_single_env() -> None:
    """n_envs == 1 is the legacy single-lane case; SB3's default 64 is
    too coarse for a single env, so we use 8."""
    cfg = TrainConfig(run_config=_rc())
    bs = _resolve_batch_size(cfg, n_envs=1, n_steps_per_env=64)
    assert bs == 8


def test_batch_size_override_wins() -> None:
    cfg = TrainConfig(run_config=_rc(), batch_size=512)
    bs = _resolve_batch_size(cfg, n_envs=40, n_steps_per_env=128)
    assert bs == 512
