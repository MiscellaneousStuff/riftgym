"""Unit tests for ``riftgym.env.protocol``.

Uses an in-process TCP loopback server to drive the ``BridgeConnection``
without needing a real game server.
"""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator

import pytest

from riftgym.env.protocol import BridgeConnection, ServerDiedError


class _FakeBridge:
    """Trivial echo server that speaks newline-delimited JSON."""

    def __init__(self) -> None:
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(1)
        self.host, self.port = self._listener.getsockname()
        self._client_sock: socket.socket | None = None
        self._thread = threading.Thread(target=self._accept, daemon=True)
        self._thread.start()
        self._connected = threading.Event()

    def _accept(self) -> None:
        sock, _ = self._listener.accept()
        self._client_sock = sock
        self._connected.set()

    def wait_connected(self, timeout: float = 2.0) -> None:
        assert self._connected.wait(timeout), "client never connected"

    def send_frame(self, obj: dict[str, object]) -> None:
        assert self._client_sock is not None
        self._client_sock.sendall((json.dumps(obj) + "\n").encode())

    def recv_frame(self, timeout: float = 2.0) -> dict[str, object]:
        assert self._client_sock is not None
        self._client_sock.settimeout(timeout)
        f = self._client_sock.makefile("rb")
        line = f.readline()
        return json.loads(line)

    def close_client(self) -> None:
        if self._client_sock is not None:
            self._client_sock.close()
            self._client_sock = None

    def close(self) -> None:
        self.close_client()
        self._listener.close()


@pytest.fixture
def bridge() -> Iterator[_FakeBridge]:
    b = _FakeBridge()
    try:
        yield b
    finally:
        b.close()


def test_connect_send_recv_roundtrip(bridge: _FakeBridge) -> None:
    conn = BridgeConnection(bridge.host, bridge.port)
    conn.connect()
    bridge.wait_connected()

    conn.send({"type": "noop"})
    assert bridge.recv_frame() == {"type": "noop"}

    bridge.send_frame({"tick": 100, "state": "GAMELOOP"})
    assert conn.recv() == {"tick": 100, "state": "GAMELOOP"}

    conn.close()


def test_recv_raises_on_eof(bridge: _FakeBridge) -> None:
    conn = BridgeConnection(bridge.host, bridge.port)
    conn.connect()
    bridge.wait_connected()
    bridge.close_client()

    with pytest.raises(ServerDiedError):
        conn.recv()
    conn.close()


def test_send_before_connect_raises() -> None:
    conn = BridgeConnection("127.0.0.1", 1)
    with pytest.raises(ServerDiedError):
        conn.send({"type": "noop"})


def test_context_manager_closes(bridge: _FakeBridge) -> None:
    with BridgeConnection(bridge.host, bridge.port) as conn:
        bridge.wait_connected()
        assert conn.connected
    assert not conn.connected
