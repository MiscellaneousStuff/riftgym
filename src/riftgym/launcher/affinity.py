"""CPU-pin helpers for trainer + spawned containers.

Pinning the trainer to a dedicated set of cores keeps it off the
physical cores hosting the brokenwings server containers, so PPO
gradient updates don't preempt the game loop on a busy box.

Everything degrades to a no-op when ``psutil`` isn't installed —
the ``riftgym[perf]`` extra is optional. The trainer logs a one-shot
warning so users know pinning silently didn't happen.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

log = logging.getLogger(__name__)

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]


def psutil_available() -> bool:
    return psutil is not None


def detect_threads_per_core() -> int:
    """Return logical-cores-per-physical-core.

    1 means no SMT (Apple Silicon, AWS Graviton, most ARM). 2 means
    standard SMT (consumer x86, EPYC). Falls back to 1 if topology
    can't be determined — safer than over-claiming sibling cores and
    oversubscribing on a box that has no SMT (the c7g.4xlarge bug
    brokenwings ran into).
    """
    if psutil is None:
        return 1
    logical = psutil.cpu_count(logical=True)
    physical = psutil.cpu_count(logical=False)
    if not logical or not physical or physical <= 0:
        return 1
    return max(1, logical // physical)


def affinity_for_server(server_idx: int, threads_per_core: int) -> list[int]:
    """Logical-core list for server ``server_idx`` under the given SMT layout.

    On SMT-2 (consumer Zen/Intel) returns both siblings ``{2i, 2i+1}``
    so .NET background threads (GC, JIT, log4net, thread-pool) share
    one physical core with the GameLoop instead of getting scheduled
    onto a different physical core. On non-SMT (Graviton, Apple) returns
    one logical core per server — using the SMT layout there would
    treat two distinct physical cores as one server's slot and
    exhaust the box at N = cores/2.
    """
    base = threads_per_core * server_idx
    return [base + k for k in range(threads_per_core)]


def pin_current_process_to(cores: Sequence[int]) -> bool:
    """Bind the current process to the given logical cores.

    Returns True on success, False if psutil is missing or the OS
    rejected the affinity request (e.g. macOS, which doesn't support
    `cpu_affinity`). Caller logs / continues either way — pinning is
    a perf hint, not correctness.
    """
    if psutil is None:
        log.warning(
            "pin_current_process_to(%s): psutil not installed; "
            "install riftgym[perf] to enable CPU pinning",
            list(cores),
        )
        return False
    try:
        psutil.Process(os.getpid()).cpu_affinity(list(cores))
        return True
    except (AttributeError, OSError, ValueError) as exc:
        # AttributeError: cpu_affinity isn't supported on this platform
        # (macOS). OSError/ValueError: invalid core list / permission.
        log.warning(
            "pin_current_process_to(%s) failed: %s; continuing without pinning",
            list(cores),
            exc,
        )
        return False


def plan_trainer_cores(n_servers: int) -> list[int] | None:
    """Reserve the trailing cores for the trainer.

    Each server gets ``threads_per_core`` logical cores starting at 0;
    everything left over goes to the trainer. Returns ``None`` if
    psutil isn't available, no cores are free, or the math doesn't
    fit — caller should skip pinning in those cases.
    """
    if psutil is None:
        return None
    logical = psutil.cpu_count(logical=True)
    if not logical:
        return None
    tpc = detect_threads_per_core()
    server_cores_end = n_servers * tpc
    if server_cores_end >= logical:
        return None
    return list(range(server_cores_end, logical))
