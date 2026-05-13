"""Unit tests for :class:`riftgym.sb3.snapshot_pool.SnapshotPool`.

Skipped wholesale if ``stable-baselines3`` isn't importable
(``riftgym[sb3]`` not installed, or the local env has a broken
transitive dep like pyarrow). The pool itself is pure-Python deque +
dict bookkeeping; we mock ``MaskablePPO.load`` so tests don't need a
real checkpoint on disk.
"""

from __future__ import annotations

import os

import pytest

# Gated on env var, not pytest.importorskip, because some environments
# crash hard (C-level abort) when sb3 is imported — typically a stale
# pyarrow/keras transitive dep. importorskip catches ImportError but
# not SIGABRT, so we just don't touch sb3 unless the user opts in.
if not os.environ.get("RIFTGYM_SB3_TESTS"):
    pytest.skip("set RIFTGYM_SB3_TESTS=1 to run sb3 tests", allow_module_level=True)

import random
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from riftgym.sb3.snapshot_pool import SnapshotPool


def _fake_load(path: str, **_kwargs: Any) -> MagicMock:
    """Stand-in for ``MaskablePPO.load`` that returns a tagged mock.
    Tagging by path lets tests assert which snapshot got sampled."""
    m = MagicMock()
    m._tag = path
    return m


@pytest.fixture
def mocked_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "riftgym.sb3.snapshot_pool.MaskablePPO.load", _fake_load, raising=True
    )


def test_empty_pool_returns_none(mocked_load: None) -> None:
    pool = SnapshotPool(capacity=4)
    assert pool.sample() is None
    assert len(pool) == 0


def test_add_path_loads_eagerly(mocked_load: None, tmp_path: Path) -> None:
    """Eager load means the first sample() after add doesn't pay disk
    read latency on a hot training step. Mock asserts MaskablePPO.load
    was called exactly once during the add."""
    pool = SnapshotPool(capacity=4)
    with patch(
        "riftgym.sb3.snapshot_pool.MaskablePPO.load", wraps=_fake_load
    ) as load_mock:
        pool.add_path(tmp_path / "ckpt1.zip")
        assert load_mock.call_count == 1
    assert len(pool) == 1


def test_lru_eviction_drops_oldest(mocked_load: None) -> None:
    """Past the configured capacity, the oldest path drops off and its
    cached model is freed too."""
    pool = SnapshotPool(capacity=2)
    pool.add_path("/a")
    pool.add_path("/b")
    pool.add_path("/c")  # evicts /a
    assert pool.paths == ["/b", "/c"]
    assert "/a" not in pool._cache
    assert {"/b", "/c"} == set(pool._cache.keys())


def test_add_path_idempotent(mocked_load: None) -> None:
    """Same path added twice (e.g. CheckpointCallback double-fires on
    a jittery save_freq) is a no-op; we don't reload the model."""
    pool = SnapshotPool(capacity=4)
    with patch(
        "riftgym.sb3.snapshot_pool.MaskablePPO.load", wraps=_fake_load
    ) as load_mock:
        pool.add_path("/a")
        pool.add_path("/a")
        assert load_mock.call_count == 1
    assert pool.paths == ["/a"]


def test_sample_returns_a_known_model(mocked_load: None) -> None:
    pool = SnapshotPool(capacity=4)
    pool.add_path("/a")
    pool.add_path("/b")
    rng = random.Random(0)
    sampled = pool.sample(rng=rng)
    assert sampled is not None
    assert sampled._tag in ("/a", "/b")


def test_sample_rng_determinism(mocked_load: None) -> None:
    """Passing rng makes sampling reproducible for tests / debugging."""
    pool = SnapshotPool(capacity=4)
    for p in ("/a", "/b", "/c", "/d"):
        pool.add_path(p)
    seq1 = [pool.sample(rng=random.Random(42))._tag for _ in range(5)]
    seq2 = [pool.sample(rng=random.Random(42))._tag for _ in range(5)]
    assert seq1 == seq2
