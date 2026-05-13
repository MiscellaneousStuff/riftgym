"""Multi-lane shared-socket coordinator for one brokenwings server.

A :class:`ServerSession` owns one TCP bridge connection plus a reader
thread that fans every observation frame out to a per-lane queue.
Multiple :class:`riftgym.env.lol_gym.LoLGymEnv` instances share the
session, each indexed by ``lane_idx``; the session multiplexes their
``step``/``reset`` traffic over the single socket.

This is the core "one server, N gym envs" abstraction the training
stack is built on. With ``mirror_both_sides=True`` each physical 1v1
yields two session lanes (Blue=me, Purple=me) sharing one tick, so a
single physics step produces two PPO samples at zero additional
simulation cost.

Why it's not built on :class:`riftgym.env.protocol.BridgeConnection`:
that class is single-consumer (one ``readline`` per caller). Sessions
need multi-consumer fan-out — exactly one reader thread, N lane queues
— so they own their own socket. They do share
:class:`~riftgym.env.protocol.ServerDiedError` for typed crash handling.

Concurrency:
  - Sends are serialized by :attr:`_send_lock`. JSON frames are
    newline-delimited so interleaved writes from different lanes are
    still well-formed at the bridge.
  - The reader thread is the only consumer of the socket. Per-lane
    queues are unbounded; lanes drain ``frame_skip`` obs per step so
    under steady-state ThreadVecEnv load the queues stay shallow.
  - :attr:`_init_lock` serializes the one-shot init so concurrent
    first-time ``reset_lane`` calls don't race on claim/level_spells.

Crash handling:
  When the reader thread sees socket EOF or any read error, it pushes
  :data:`_DEAD_SENTINEL` onto every lane queue exactly once. Lane
  workers blocked on ``q.get()`` unblock immediately and raise
  :class:`~riftgym.env.protocol.ServerDiedError` on their next
  step/reset. Without this, a server crash silently hung every blocked
  lane forever — the original failure mode the sentinel was added to
  prevent.
"""

from __future__ import annotations

import contextlib
import json
import logging
import queue
import socket
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any

from riftgym.env.protocol import ServerDiedError

log = logging.getLogger(__name__)

Action = dict[str, Any]
Obs = dict[str, Any]
Spot = tuple[float, float]


@dataclass(slots=True, frozen=True)
class LaneSpec:
    """One session lane: which champ pair to control and where to spawn them.

    ``me_cid`` / ``opp_cid`` are server-side champion client_ids. In
    ``mirror_both_sides=True`` mode two LaneSpecs share the same physical
    pair with me/opp swapped — see
    :func:`riftgym.env.multilane.make_multilane_envs`.
    """

    me_cid: int
    opp_cid: int
    me_spot: Spot
    opp_spot: Spot


# Read buffer for the socket file. ``buffering=0`` falls back to
# RawIOBase.readline (one byte per syscall, GIL re-acquired each time);
# in multi-threaded VecEnv setups that craters throughput by ~6.5x.
_READ_BUFFER_SIZE = 65536

# Sentinel pushed onto every lane queue when the reader thread exits.
# Identity-checked (``is _DEAD_SENTINEL``), so contents don't matter.
_DEAD_SENTINEL: object = object()


class ServerSession:
    """One server, many lanes. See module docstring."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 5120,
        lanes: Sequence[LaneSpec],
        frame_skip: int = 8,
        reset_settle_s: float = 0.5,
        reset_timeout_s: float = 15.0,
        wipe_progression: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.lanes: list[LaneSpec] = list(lanes)
        if not self.lanes:
            raise ValueError("ServerSession needs at least one lane")
        self.frame_skip = max(1, int(frame_skip))
        self.reset_settle_s = reset_settle_s
        self.reset_timeout_s = reset_timeout_s
        self.wipe_progression = wipe_progression

        self._sock: socket.socket | None = None
        self._f: Any = None  # BufferedRWPair from socket.makefile
        self._send_lock = threading.Lock()
        self._init_lock = threading.Lock()
        self._reader_thread: threading.Thread | None = None
        self._closed = False
        self._initialized = False
        self._queues: list[queue.Queue[Any]] = [queue.Queue() for _ in self.lanes]

    @property
    def n_lanes(self) -> int:
        return len(self.lanes)

    def _connect(self) -> None:
        if self._sock is not None:
            return
        self._sock = socket.create_connection((self.host, self.port))
        # TCP_NODELAY is a perf hint for the loopback TCP bridge; if the
        # underlying socket isn't TCP (e.g. AF_UNIX in tests), just no-op.
        with contextlib.suppress(OSError):
            self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._f = self._sock.makefile("rwb", buffering=_READ_BUFFER_SIZE)
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name=f"ServerSession-{self.port}",
            daemon=True,
        )
        self._reader_thread.start()

    def _reader_loop(self) -> None:
        try:
            while not self._closed:
                line = self._f.readline()
                if not line:
                    return
                try:
                    obs = json.loads(line)
                except (ValueError, TypeError):
                    # Malformed frame — skip rather than killing the
                    # reader. Lane queues stay live for the next valid
                    # obs. (Never observed in practice with the
                    # newline-delimited JSON wire format.)
                    continue
                for q in self._queues:
                    q.put(obs)
        except (OSError, ValueError):
            # OSError: socket died (ECONNRESET, EBADF). ValueError: file
            # closed concurrently by .close(). Both → broadcast death.
            pass
        finally:
            # Reader exiting means the bridge socket is gone (server
            # crash, explicit close, or read error). Wake every blocked
            # lane with the dead sentinel so they raise ServerDiedError
            # on their next get() instead of hanging forever.
            for q in self._queues:
                q.put(_DEAD_SENTINEL)

    def _send(self, obj: dict[str, Any]) -> None:
        line = (json.dumps(obj) + "\n").encode()
        with self._send_lock:
            if self._f is None:
                raise ServerDiedError(f"server on port {self.port} not connected")
            try:
                self._f.write(line)
                self._f.flush()
            except (OSError, ValueError) as exc:
                # OSError covers ConnectionResetError, BrokenPipeError, etc.
                # ValueError fires if the file was closed concurrently. The
                # reader thread will (or already has) pushed _DEAD_SENTINEL
                # into the lane queues; surface the death as a typed error.
                raise ServerDiedError(
                    f"server on port {self.port} disconnected during send"
                ) from exc

    @staticmethod
    def _drain_queue(q: queue.Queue[Any]) -> None:
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                return

    def _get_or_die(self, q: queue.Queue[Any]) -> Obs:
        item = q.get()
        if item is _DEAD_SENTINEL:
            # Re-arm the sentinel so a subsequent caller (e.g. a different
            # lane on the same dead server, or a reset call after step
            # raised) also sees it instead of blocking again.
            q.put(_DEAD_SENTINEL)
            raise ServerDiedError(
                f"server on port {self.port} disconnected; bridge reader exited"
            )
        return item  # type: ignore[no-any-return]

    def _ensure_initialized(self, lane_idx: int) -> None:
        with self._init_lock:
            self._connect()
            if self._initialized:
                return
            q = self._queues[lane_idx]
            forced = False
            while True:
                obs = self._get_or_die(q)
                if obs.get("state") == "GAMELOOP":
                    break
                if not forced:
                    self._send({"type": "force_start"})
                    forced = True
            for lane in self.lanes:
                self._send({"type": "claim", "client_id": lane.me_cid})
                self._send({"type": "claim", "client_id": lane.opp_cid})
            for lane in self.lanes:
                for cid in (lane.me_cid, lane.opp_cid):
                    for slot in range(4):
                        self._send(
                            {"type": "level_spell", "client_id": cid, "slot": slot}
                        )
            self._initialized = True

    def reset_lane(
        self,
        lane_idx: int,
        extra_spot_fields: dict[int, dict[str, Any]] | None = None,
    ) -> Obs:
        """Reset one lane's pair via the bridge's ``cids`` filter.

        Args:
            lane_idx: index into :attr:`lanes`.
            extra_spot_fields: optional ``client_id -> dict`` of fields
                merged into that cid's spot entry. Used by
                :class:`~riftgym.env.lol_gym.LoLGymEnv` for OAI-Five
                initial-state HP/MP randomization.
        """
        self._ensure_initialized(lane_idx)
        lane = self.lanes[lane_idx]
        spots: list[dict[str, Any]] = [
            {"client_id": lane.me_cid, "x": lane.me_spot[0], "y": lane.me_spot[1]},
            {"client_id": lane.opp_cid, "x": lane.opp_spot[0], "y": lane.opp_spot[1]},
        ]
        if extra_spot_fields:
            for s in spots:
                extra = extra_spot_fields.get(int(s["client_id"]))
                if extra:
                    s.update(extra)
        msg: dict[str, Any] = {
            "type": "reset",
            "wipe_progression": self.wipe_progression,
            "cids": [lane.me_cid, lane.opp_cid],
            "spots": spots,
        }
        q = self._queues[lane_idx]
        # Drain any stale obs queued from before we sent the reset so
        # floor_tick anchors on a post-reset frame.
        self._drain_queue(q)
        self._send(msg)
        obs = self._get_or_die(q)
        start_tick = obs["tick"]
        floor_tick = start_tick + self.reset_settle_s * 1000.0
        deadline_tick = start_tick + self.reset_timeout_s * 1000.0
        while True:
            by_cid = {c["client_id"]: c for c in obs["champs"]}
            me = by_cid.get(lane.me_cid, {})
            opp = by_cid.get(lane.opp_cid, {})
            ok = bool(
                me.get("alive")
                and me.get("hp", 0) > 0
                and opp.get("alive")
                and opp.get("hp", 0) > 0
            )
            if ok and obs["tick"] >= floor_tick:
                return obs
            if obs["tick"] >= deadline_tick:
                raise TimeoutError(
                    f"reset_lane({lane_idx}) timed out after "
                    f"{self.reset_timeout_s}s game-time; "
                    f"me_cid={lane.me_cid} opp_cid={lane.opp_cid} obs={by_cid}"
                )
            obs = self._get_or_die(q)

    def step_lane(
        self, lane_idx: int, actions: Sequence[Action] | None = None
    ) -> Obs:
        """Send any ``actions`` then drain ``frame_skip`` obs frames.

        Returns the last drained obs — same shape as
        :meth:`riftgym.env.lol_env.LoLEnv.step` so ``LoLGymEnv`` can
        delegate to either uniformly.
        """
        if actions:
            for a in actions:
                self._send(a)
        q = self._queues[lane_idx]
        obs: Obs | None = None
        for _ in range(self.frame_skip):
            obs = self._get_or_die(q)
        assert obs is not None  # frame_skip >= 1 by construction
        return obs

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Wake the reader thread BEFORE closing the BufferedRWPair.
        # Python's buffered IO refuses to close while another thread
        # holds it open for read (lock around the read buffer), so
        # closing `_f` first deadlocks. Socket shutdown delivers EOF
        # to the reader's readline(), which then unwinds cleanly.
        if self._sock is not None:
            with contextlib.suppress(OSError):
                self._sock.shutdown(socket.SHUT_RDWR)
        if self._f is not None:
            with contextlib.suppress(OSError):
                self._f.close()
        if self._sock is not None:
            with contextlib.suppress(OSError):
                self._sock.close()
        self._f = self._sock = None

    def __enter__(self) -> ServerSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
