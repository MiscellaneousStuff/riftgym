"""Unit tests for :mod:`riftgym.launcher.affinity`.

These tests stay platform-agnostic by not actually pinning anything.
The real `psutil.Process.cpu_affinity` call is guarded inside
`pin_current_process_to`, which returns False on platforms that don't
support it (macOS) — tests can run there without skipping.
"""

from __future__ import annotations

import pytest

from riftgym.launcher import affinity


def test_detect_threads_per_core_is_positive() -> None:
    """Whether psutil is installed or not, the helper must return >= 1.
    Returning 0 would zero-out the per-server core math and crash
    affinity_for_server."""
    tpc = affinity.detect_threads_per_core()
    assert tpc >= 1


def test_affinity_for_server_layout_smt2() -> None:
    """SMT-2 (consumer x86): each server gets two adjacent logical
    cores so .NET background threads share a physical core with the
    GameLoop instead of contending across cores."""
    assert affinity.affinity_for_server(0, threads_per_core=2) == [0, 1]
    assert affinity.affinity_for_server(1, threads_per_core=2) == [2, 3]
    assert affinity.affinity_for_server(3, threads_per_core=2) == [6, 7]


def test_affinity_for_server_layout_non_smt() -> None:
    """Non-SMT (Graviton, Apple Silicon): one logical core per server.
    Without this, using the SMT-2 layout treats two distinct physical
    cores as one server's slot and exhausts the box at N=cores/2."""
    assert affinity.affinity_for_server(0, threads_per_core=1) == [0]
    assert affinity.affinity_for_server(4, threads_per_core=1) == [4]


def test_pin_current_process_to_returns_bool() -> None:
    """Pinning is best-effort: returns True on Linux+psutil, False
    everywhere else. The contract is just that it doesn't crash and
    returns a bool the caller can log."""
    result = affinity.pin_current_process_to([0])
    assert isinstance(result, bool)


def test_pin_without_psutil_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(affinity, "psutil", None)
    assert affinity.pin_current_process_to([0]) is False


def test_plan_trainer_cores_without_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without psutil we can't introspect topology, so return None and
    let the caller skip pinning."""
    monkeypatch.setattr(affinity, "psutil", None)
    assert affinity.plan_trainer_cores(4) is None


def test_plan_trainer_cores_with_fake_topology(monkeypatch: pytest.MonkeyPatch) -> None:
    """8 logical cores, SMT-2, 2 servers → servers consume cores 0..3,
    trainer gets 4..7. Validates the reservation math without needing
    a real machine of a specific size."""

    class _FakePsutil:
        @staticmethod
        def cpu_count(logical: bool = True) -> int:
            return 8 if logical else 4

    monkeypatch.setattr(affinity, "psutil", _FakePsutil)
    cores = affinity.plan_trainer_cores(n_servers=2)
    assert cores == [4, 5, 6, 7]


def test_plan_trainer_cores_when_fleet_takes_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the server fleet would consume every core, there's nothing
    left for the trainer — skip pinning instead of starving it."""

    class _FakePsutil:
        @staticmethod
        def cpu_count(logical: bool = True) -> int:
            # Same count for logical/physical → threads_per_core = 1
            return 4

    monkeypatch.setattr(affinity, "psutil", _FakePsutil)
    cores = affinity.plan_trainer_cores(n_servers=4)
    assert cores is None
