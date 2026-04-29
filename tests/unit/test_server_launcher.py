"""Unit tests for ``ServerLauncher`` against a fake RunConfig.

The fake spawns a tiny in-process TCP listener on the requested rl_port
to satisfy ``wait_for_port`` without needing real Docker.
"""

from __future__ import annotations

import socket
import threading
from dataclasses import dataclass, field

import pytest

from riftgym.launcher import ServerLauncher
from riftgym.run_configs.lib import RunConfig, ServerHandle


class _FakeHandle:
    def __init__(self, listener: socket.socket, *, game_port: int, rl_port: int) -> None:
        self._listener = listener
        self.game_port = game_port
        self.rl_port = rl_port
        self._terminated = False

    def is_alive(self) -> bool:
        return not self._terminated

    def terminate(self) -> None:
        if self._terminated:
            return
        self._terminated = True
        self._listener.close()

    def wait(self, timeout: float | None = None) -> int | None:
        return 0

    def logs(self) -> str:
        return ""


@dataclass(slots=True)
class _FakeRunConfig(RunConfig):
    started_ports: list[tuple[int, int]] = field(default_factory=list)

    def start(self, *, game_port: int, rl_port: int) -> ServerHandle:
        self.started_ports.append((game_port, rl_port))
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", rl_port))
        listener.listen(1)
        # Accept-and-close in a background thread so wait_for_port's
        # connect resolves immediately.
        threading.Thread(target=_accept_then_close, args=(listener,), daemon=True).start()
        return _FakeHandle(listener, game_port=game_port, rl_port=rl_port)


def _accept_then_close(listener: socket.socket) -> None:
    """Accept any number of connections, send one byte (so wait_for_port's
    recv() succeeds), then close. wait_for_port now requires a real
    frame from the bridge to consider the server up — a bare accept
    isn't enough."""
    try:
        while True:
            client, _ = listener.accept()
            try:
                client.sendall(b"x")
            finally:
                client.close()
    except OSError:
        return


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_launches_n_servers_and_assigns_ports() -> None:
    rc = _FakeRunConfig()
    base_game = _free_port()
    base_rl = _free_port()
    with ServerLauncher(
        n=3,
        run_config=rc,
        base_game_port=base_game,
        base_rl_port=base_rl,
        port_stride=2,
        port_ready_timeout_s=5.0,
    ) as L:
        assert L.rl_ports == [base_rl, base_rl + 2, base_rl + 4]
        assert L.game_ports == [base_game, base_game + 2, base_game + 4]
        assert len(L.handles) == 3
    # After exit, all handles are torn down
    assert all(not h.is_alive() for h in L.handles)


def test_n_must_be_at_least_one() -> None:
    rc = _FakeRunConfig()
    with pytest.raises(ValueError, match="n must be >= 1"):
        ServerLauncher(n=0, run_config=rc)
