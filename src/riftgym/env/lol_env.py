"""Minimal LoL environment over the brokenwings RLBridge socket.

No gym/dm_env deps, no reward, no termination flag — just the
connect/reset/step/close loop. Build a gym wrapper on top of this if you
want gymnasium semantics; see :class:`riftgym.env.lol_gym.LoLGymEnv`.

Actions and observations are async on the bridge: actions queue on
``Game.pendingActions`` (applied next iteration); observations fire
free-running at ``RL_HZ``. The frame returned by ``reset()``/``step()``
may not yet reflect the action just sent. Live with it for now.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType
from typing import Any

from riftgym.env.protocol import BridgeConnection

Action = dict[str, Any]
Spot = dict[str, Any]
Obs = dict[str, Any]


class LoLEnv:
    """One bridge connection driving one game.

    Args:
        host: bridge host (loopback by default).
        port: bridge TCP port (default 5120).
        claim_ids: champion client_ids this env will control. Claimed on
            the first ``reset()``; PauseAI fires for these so the engine BT
            stops driving them.
        spots: optional list of ``{"client_id", "x", "y"}`` dicts. Sent
            with every ``reset()`` to teleport champions into known
            positions before the episode starts.
        wipe_progression: whether ``reset()`` resets level/gold/items
            (1v1 trial semantics) or just respawns (5v5 round-reset).
        reset_settle_s: floor on game-time waited after the reset before
            ``reset()`` returns. Real exit gate is "all claim_ids alive
            with hp > 0", but we still wait at least this long so spell
            cooldowns / teleports / FaceDirection have time to land.
        reset_timeout_s: hard timeout in game-time on ``reset()``. If a
            claimed champ isn't alive after this long, raises
            :class:`TimeoutError`.
        frame_skip: number of obs frames to drain per ``step()``.
            Action rate = ``RL_HZ / frame_skip``. Default 8 → ~3.75 Hz at
            ``RL_HZ=30``. Also keeps the socket buffer drained.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 5120,
        claim_ids: Sequence[int] | None = None,
        spots: Sequence[Spot] | None = None,
        wipe_progression: bool = True,
        reset_settle_s: float = 0.5,
        reset_timeout_s: float = 15.0,
        frame_skip: int = 8,
    ) -> None:
        self.host = host
        self.port = port
        self.claim_ids: list[int] = list(claim_ids) if claim_ids else []
        self.spots: list[Spot] | None = [dict(s) for s in spots] if spots else None
        self.wipe_progression = wipe_progression
        self.reset_settle_s = reset_settle_s
        self.reset_timeout_s = reset_timeout_s
        self.frame_skip = max(1, int(frame_skip))

        self._bridge = BridgeConnection(host, port)
        self._claimed = False

    def connect(self) -> None:
        self._bridge.connect()

    def reset(
        self,
        extra_spot_fields: dict[int, dict[str, Any]] | None = None,
    ) -> Obs:
        """Reset the game and return the post-reset observation.

        Args:
            extra_spot_fields: optional ``client_id -> dict`` of fields
                merged into that cid's spot entry before sending. Used by
                gym wrappers for per-episode HP/MP jitter (OAI Five
                initial-state randomization) without rewiring the spot
                config baked in at construction.
        """
        self._bridge.connect()
        self._wait_gameloop()
        if not self._claimed:
            for cid in self.claim_ids:
                self._bridge.send({"type": "claim", "client_id": cid})
            self._claimed = True

        msg: dict[str, Any] = {"type": "reset", "wipe_progression": self.wipe_progression}
        if self.spots:
            spots = [dict(s) for s in self.spots]
            if extra_spot_fields:
                for s in spots:
                    cid = s.get("client_id")
                    if cid is None:
                        continue
                    extra = extra_spot_fields.get(cid)
                    if extra:
                        s.update(extra)
            msg["spots"] = spots

        self._bridge.send(msg)
        return self._wait_alive()

    def step(self, actions: Sequence[Action] | None = None) -> Obs:
        if actions:
            for a in actions:
                self._bridge.send(a)
        obs: Obs | None = None
        for _ in range(self.frame_skip):
            obs = self._bridge.recv()
        assert obs is not None  # frame_skip >= 1 by construction
        return obs

    def close(self) -> None:
        self._bridge.close()

    def _wait_gameloop(self) -> Obs:
        forced = False
        while True:
            obs = self._bridge.recv()
            if obs.get("state") == "GAMELOOP":
                return obs
            if not forced:
                self._bridge.send({"type": "force_start"})
                forced = True

    def _wait_alive(self) -> Obs:
        obs = self._bridge.recv()
        start_tick = obs["tick"]
        floor_tick = start_tick + self.reset_settle_s * 1000.0
        deadline_tick = start_tick + self.reset_timeout_s * 1000.0
        while True:
            by_cid = {c["client_id"]: c for c in obs["champs"]}
            all_alive = all(
                by_cid.get(cid, {}).get("alive") and by_cid[cid]["hp"] > 0 for cid in self.claim_ids
            )
            if all_alive and obs["tick"] >= floor_tick:
                return obs
            if obs["tick"] >= deadline_tick:
                raise TimeoutError(
                    f"reset() timed out after {self.reset_timeout_s}s game-time; "
                    f"claim_ids not all alive. obs={by_cid}"
                )
            obs = self._bridge.recv()

    def __enter__(self) -> LoLEnv:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
