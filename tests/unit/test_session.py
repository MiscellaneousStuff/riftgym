"""Unit tests for :class:`riftgym.env.session.ServerSession`.

The bridge socket is replaced with one half of a :func:`socket.socketpair`
(via a ``socket.create_connection`` monkeypatch) so the reader thread
runs against real socket semantics. The test owns the other half and
feeds JSON frames or closes the socket to simulate server crashes.
"""

from __future__ import annotations

import json
import queue
import socket
import threading
import time
from typing import Any

import pytest

from riftgym.env.protocol import ServerDiedError
from riftgym.env.session import LaneSpec, ServerSession


@pytest.fixture
def fake_pair(monkeypatch: pytest.MonkeyPatch) -> tuple[socket.socket, socket.socket]:
    """Return ``(client_sock, server_sock)``.

    Monkeypatches ``socket.create_connection`` (the one imported into
    ``riftgym.env.session``) to return ``client_sock`` — that's what
    ServerSession's reader thread will read from. The test writes
    JSON frames into ``server_sock`` to simulate the brokenwings RL
    bridge sending observations.

    AF_UNIX sockets on Darwin don't support ``IPPROTO_TCP/TCP_NODELAY``;
    the session's ``_connect`` wraps that call in suppress(OSError) so
    the fake just works without further hoops here.
    """
    a, b = socket.socketpair()

    def fake_create_connection(_addr: object, *_a: object, **_kw: object) -> socket.socket:
        return a

    monkeypatch.setattr("riftgym.env.session.socket.create_connection", fake_create_connection)
    yield a, b
    for s in (a, b):
        try:
            s.close()
        except OSError:
            pass


def _lane(me: int = 0, opp: int = 1) -> LaneSpec:
    return LaneSpec(me_cid=me, opp_cid=opp, me_spot=(0.0, 0.0), opp_spot=(0.0, 0.0))


def _send_frame(sock: socket.socket, obj: dict[str, Any]) -> None:
    sock.sendall((json.dumps(obj) + "\n").encode())


def _wait_for(q: queue.Queue[Any], timeout: float = 2.0) -> Any:
    return q.get(timeout=timeout)


def test_rejects_empty_lanes() -> None:
    with pytest.raises(ValueError, match="at least one lane"):
        ServerSession(lanes=[])


def test_reader_broadcasts_to_all_lanes(fake_pair: tuple[socket.socket, socket.socket]) -> None:
    """One frame in → that same frame appears on every lane queue."""
    _, server = fake_pair
    session = ServerSession(lanes=[_lane(0, 1), _lane(2, 3), _lane(4, 5)])
    # Kick connect by hitting the private _connect — _ensure_initialized
    # would block on GAMELOOP; we don't want to do the full claim/init
    # dance for a fan-out test.
    session._connect()
    try:
        _send_frame(server, {"state": "GAMELOOP", "tick": 1, "champs": []})
        for i in range(session.n_lanes):
            obs = _wait_for(session._queues[i])
            assert obs == {"state": "GAMELOOP", "tick": 1, "champs": []}
    finally:
        session.close()


def test_dead_sentinel_propagates_on_disconnect(
    fake_pair: tuple[socket.socket, socket.socket],
) -> None:
    """Closing the server end → reader exits → every blocked lane wakes
    with ServerDiedError. This is the failure mode the sentinel was added
    to prevent (silent hang in q.get() forever)."""
    _, server = fake_pair
    session = ServerSession(lanes=[_lane(0, 1), _lane(2, 3)])
    session._connect()

    # Block two lanes on q.get() in background threads, then kill the
    # server end and assert both unblock with ServerDiedError.
    results: list[BaseException | dict[str, Any]] = [None, None]  # type: ignore[list-item]

    def waiter(idx: int) -> None:
        try:
            results[idx] = session._get_or_die(session._queues[idx])
        except BaseException as exc:
            results[idx] = exc

    threads = [threading.Thread(target=waiter, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()

    # Give the threads a beat to actually block on get().
    time.sleep(0.05)
    server.close()

    for t in threads:
        t.join(timeout=2.0)
        assert not t.is_alive(), "lane waiter never woke from queue.get()"

    for r in results:
        assert isinstance(r, ServerDiedError), f"expected ServerDiedError, got {r!r}"

    session.close()


def test_get_or_die_rearms_sentinel(fake_pair: tuple[socket.socket, socket.socket]) -> None:
    """After the reader posts _DEAD_SENTINEL, a subsequent caller on the
    same queue must also see ServerDiedError (the sentinel re-arms),
    not block forever. Matters when one lane's step raises and the
    error handler then calls reset, which would otherwise hang."""
    _, server = fake_pair
    session = ServerSession(lanes=[_lane(0, 1)])
    session._connect()

    server.close()
    # First call observes the sentinel.
    with pytest.raises(ServerDiedError):
        session._get_or_die(session._queues[0])
    # Second call must also raise, not block.
    with pytest.raises(ServerDiedError):
        session._get_or_die(session._queues[0])

    session.close()


def test_send_raises_after_close(fake_pair: tuple[socket.socket, socket.socket]) -> None:
    """`_send` on a closed session raises ServerDiedError instead of
    `AttributeError: 'NoneType' object has no attribute 'write'`."""
    _, _ = fake_pair
    session = ServerSession(lanes=[_lane()])
    session._connect()
    session.close()
    with pytest.raises(ServerDiedError):
        session._send({"type": "noop"})


def test_close_is_idempotent(fake_pair: tuple[socket.socket, socket.socket]) -> None:
    _, _ = fake_pair
    session = ServerSession(lanes=[_lane()])
    session._connect()
    session.close()
    session.close()  # no exception, no resurrection
    assert session._sock is None


def test_malformed_frames_dont_kill_reader(
    fake_pair: tuple[socket.socket, socket.socket],
) -> None:
    """Reader skips malformed JSON instead of exiting — otherwise one
    bad frame on a long-running training run would silently kill every
    lane."""
    _, server = fake_pair
    session = ServerSession(lanes=[_lane()])
    session._connect()

    server.sendall(b"this is not json\n")
    _send_frame(server, {"state": "GAMELOOP", "tick": 99, "champs": []})

    obs = _wait_for(session._queues[0])
    assert obs["tick"] == 99
    session.close()
