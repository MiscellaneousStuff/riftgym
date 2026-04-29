"""Abstract base classes for ``RunConfig`` and ``ServerHandle``.

A ``RunConfig`` describes how to launch *one* brokenwings game server
process — whether that's via ``docker run`` (the public default,
:class:`riftgym.run_configs.container.ContainerRunConfig`) or directly
from a local .NET build (the developer path, added later). Each call to
:meth:`RunConfig.start` returns a uniform :class:`ServerHandle` so the
orchestrator (``ServerLauncher``) doesn't care how the server actually
came up.

Patterned on pysc2's / pylol's ``run_configs.lib.RunConfig`` plus
auto-discovered platform subclasses; we just have fewer subclasses for
now.
"""

from __future__ import annotations

import socket
import time
from abc import ABC, abstractmethod
from typing import Protocol


class ServerHandle(Protocol):
    """One running game server. Returned from :meth:`RunConfig.start`."""

    @property
    def game_port(self) -> int: ...

    @property
    def rl_port(self) -> int: ...

    def is_alive(self) -> bool: ...

    def terminate(self) -> None:
        """Graceful stop. Should be idempotent."""
        ...

    def wait(self, timeout: float | None = None) -> int | None:
        """Block until the server exits. Returns exit code or ``None`` on timeout."""
        ...

    def logs(self) -> str:
        """Best-effort capture of the server's stdout/stderr."""
        ...


class RunConfig(ABC):
    """Describes how to launch one brokenwings game server.

    Subclasses register themselves automatically (recursive
    :meth:`__subclasses__`); :func:`riftgym.run_configs.get_run_config`
    picks the highest-:meth:`priority` applicable subclass.
    """

    @abstractmethod
    def start(self, *, game_port: int, rl_port: int) -> ServerHandle:
        """Launch a server bound to the given host-visible ports.

        ``game_port`` is the LoL ENet UDP port (clients connect here).
        ``rl_port`` is the TCP port the RL bridge will listen on. The
        returned :class:`ServerHandle` is responsible for tearing the
        server back down on :meth:`ServerHandle.terminate`.
        """

    @classmethod
    def priority(cls) -> int | None:
        """Higher = preferred. ``None`` means "not applicable on this host"."""
        return None

    @classmethod
    def all_subclasses(cls) -> list[type[RunConfig]]:
        out: list[type[RunConfig]] = []
        for s in cls.__subclasses__():
            out.append(s)
            out.extend(s.all_subclasses())
        return out


def wait_for_port(
    host: str,
    port: int,
    *,
    timeout_s: float = 60.0,
    poll_interval_s: float = 0.5,
) -> None:
    """Block until a real bridge is listening on ``host:port``.

    A bare TCP-accept probe isn't enough: Docker's userland port proxy
    accepts host-side connections before the in-container server has
    actually bound the port, then drops them when the forward fails.
    So we connect AND read at least one byte — the brokenwings RL
    bridge emits an obs frame within ~33 ms of accept (one tick at
    ``RL_HZ=30``), so a successful read confirms a real listener.

    Raises :class:`TimeoutError` on miss. Used by ``ServerLauncher``
    after :meth:`RunConfig.start` returns.
    """
    deadline = time.monotonic() + timeout_s
    last_err: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=poll_interval_s) as s:
                s.settimeout(2.0)
                data = s.recv(1)
                if not data:
                    raise OSError("bridge closed without emitting a frame")
                return
        except OSError as exc:
            last_err = exc
            time.sleep(poll_interval_s)
    raise TimeoutError(
        f"server on {host}:{port} did not produce a bridge frame within "
        f"{timeout_s:.1f}s (last error: {last_err})"
    )
