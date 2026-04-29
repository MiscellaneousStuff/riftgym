"""TCP + newline-delimited JSON framing for the brokenwings RL bridge.

Wire format: each frame is one ``json.dumps(obj) + "\\n"`` UTF-8 line, in
either direction. Actions go client→server, observations server→client.

The bridge is currently unauth'd and intended for loopback (or a Docker
``127.0.0.1:<port>:5120`` host-side mapping). Cross-host use needs the
auth work tracked for v0.2.
"""

from __future__ import annotations

import contextlib
import json
import socket
from types import TracebackType
from typing import Any


class ServerDiedError(ConnectionError):
    """The bridge socket closed while a caller was waiting for a frame.

    Most likely the underlying server process crashed or was killed.
    Surfaced as a typed error so callers can fail fast instead of hanging
    on a ``readline()`` that will never return.
    """


# 65 KiB read buffer. ``buffering=0`` falls back to RawIOBase.readline,
# which reads ONE BYTE per syscall (~5000 syscalls per 5 KB obs frame).
# Each syscall releases the GIL; in multi-threaded VecEnv setups the GIL
# re-acquisitions serialize across env threads and crater throughput by
# ~6.5x.
_READ_BUFFER_SIZE = 65536


class BridgeConnection:
    """Low-level wrapper for one bridge TCP connection.

    Stateless about game semantics — caller decides what JSON to send and
    how to interpret what comes back. The session/env classes built on top
    of this handle reset/step/claim/etc.
    """

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._sock: socket.socket | None = None
        self._f: Any = None  # BufferedRWPair from socket.makefile

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def connect(self) -> None:
        if self._sock is not None:
            return
        self._sock = socket.create_connection((self.host, self.port))
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._f = self._sock.makefile("rwb", buffering=_READ_BUFFER_SIZE)

    def send(self, obj: dict[str, Any]) -> None:
        if self._f is None:
            raise ServerDiedError(f"bridge {self.host}:{self.port} not connected")
        line = (json.dumps(obj) + "\n").encode()
        try:
            self._f.write(line)
            self._f.flush()
        except (OSError, ValueError) as exc:
            raise ServerDiedError(
                f"bridge {self.host}:{self.port} disconnected during send"
            ) from exc

    def recv(self) -> dict[str, Any]:
        if self._f is None:
            raise ServerDiedError(f"bridge {self.host}:{self.port} not connected")
        try:
            line = self._f.readline()
        except (OSError, ValueError) as exc:
            raise ServerDiedError(
                f"bridge {self.host}:{self.port} disconnected during recv"
            ) from exc
        if not line:
            raise ServerDiedError(f"bridge {self.host}:{self.port} closed")
        result: dict[str, Any] = json.loads(line)
        return result

    def close(self) -> None:
        if self._f is not None:
            with contextlib.suppress(OSError):
                self._f.close()
        if self._sock is not None:
            with contextlib.suppress(OSError):
                self._sock.close()
        self._f = self._sock = None

    def __enter__(self) -> BridgeConnection:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
