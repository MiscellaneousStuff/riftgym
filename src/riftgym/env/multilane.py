"""Multilane factory: one server, N gym envs sharing one bridge socket.

This is the topology the trainer uses. Each call to
:func:`make_multilane_envs` builds:

1. A :class:`riftgym.env.session.ServerSession` that owns the bridge TCP
   socket and a reader thread broadcasting obs frames to per-lane queues.
2. A flat list of :class:`riftgym.env.lol_gym.LoLGymEnv` instances, each
   wired to the session at a distinct ``lane_idx``.

Two physical 1v1 modes:

- ``mirror_both_sides=False``: one env per lane, agent on Blue side,
  Purple driven by ``opp_policy`` (or the engine BT if ``opp_policy``
  is None).
- ``mirror_both_sides=True``: TWO envs per physical lane sharing the
  same physics tick. Env A controls Blue (me_cid=Blue, opp_cid=Purple);
  Env B controls Purple (me_cid=Purple, opp_cid=Blue). Both set
  ``omit_opp_action=True`` so each env sends only its own me-cid action;
  per tick the bridge sees exactly two actions per physical lane
  (Blue + Purple), not four. Doubles training data per wall-second at
  zero additional simulation cost. (OAI-Five §3.2-style mirror.)

Caller owns the returned ``session`` and must ``session.close()`` when
done. The launcher / ThreadVecEnv layers wire this up via context
managers / atexit hooks; nothing here ties into those frameworks
directly so the function stays usable from scripts and tests.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from riftgym.env.lol_gym import LoLGymEnv
from riftgym.env.session import LaneSpec, ServerSession
from riftgym.lib.lane_spots import LANE_PAIR_OFFSET, LANE_SPOT_TABLE

Action = dict[str, Any]
Obs = dict[str, Any]
RewardFn = Callable[..., float]


def build_lane_specs(
    n_lanes: int,
    *,
    blue_cids: Sequence[int] | None = None,
    purple_cids: Sequence[int] | None = None,
) -> list[LaneSpec]:
    """Build N :class:`LaneSpec`s for N 1v1 pairs.

    The brokenwings multilane GameInfo configs add Blue players first,
    then Purple, so the runtime cids are ``0..n_lanes-1`` (Blue) and
    ``n_lanes..2*n_lanes-1`` (Purple) by default. Lane ``i`` pairs
    ``(Blue_i, Purple_i)`` at ``LANE_SPOT_TABLE[i]`` with
    ``±LANE_PAIR_OFFSET`` on the X axis to keep them in range of each
    other.

    Override ``blue_cids`` / ``purple_cids`` if a custom settings JSON
    interleaves teams differently.
    """
    if n_lanes < 1 or n_lanes > len(LANE_SPOT_TABLE):
        raise ValueError(f"n_lanes {n_lanes} not in [1, {len(LANE_SPOT_TABLE)}]")
    blue = list(blue_cids) if blue_cids is not None else list(range(n_lanes))
    purple = (
        list(purple_cids)
        if purple_cids is not None
        else list(range(n_lanes, 2 * n_lanes))
    )
    if len(blue) < n_lanes or len(purple) < n_lanes:
        raise ValueError(
            f"need >= {n_lanes} blue and purple cids; got {blue} {purple}"
        )
    specs: list[LaneSpec] = []
    for i in range(n_lanes):
        x, y = LANE_SPOT_TABLE[i][1]
        specs.append(
            LaneSpec(
                me_cid=blue[i],
                opp_cid=purple[i],
                me_spot=(x - LANE_PAIR_OFFSET, y),
                opp_spot=(x + LANE_PAIR_OFFSET, y),
            )
        )
    return specs


def make_multilane_envs(
    *,
    host: str = "127.0.0.1",
    port: int = 5120,
    n_lanes: int = 3,
    lane_specs: Sequence[LaneSpec] | None = None,
    max_episode_steps: int = 80,
    frame_skip: int = 8,
    reward_fn: RewardFn | None = None,
    e_flee: bool = False,
    mirror_both_sides: bool = False,
) -> tuple[ServerSession, list[LoLGymEnv]]:
    """Build a :class:`ServerSession` plus per-lane :class:`LoLGymEnv`s.

    Returns ``(session, envs)``. With ``mirror_both_sides=True`` the
    returned ``envs`` has length ``2 * n_lanes`` — see module docstring
    for the per-physical-lane Blue/Purple pair semantics.

    Caller owns ``session`` and is responsible for ``session.close()``.
    """
    specs = list(lane_specs) if lane_specs is not None else build_lane_specs(n_lanes)
    if not specs:
        raise ValueError("need at least one lane spec")

    if mirror_both_sides:
        # Two session lanes per physical 1v1: same cid pair, swapped
        # me/opp. The session indexes queues by session-lane (not by
        # physical lane), so each LoLGymEnv gets its own queue and
        # observes every obs frame independently.
        session_lanes: list[LaneSpec] = []
        for spec in specs:
            session_lanes.append(spec)
            session_lanes.append(
                LaneSpec(
                    me_cid=spec.opp_cid,
                    opp_cid=spec.me_cid,
                    me_spot=spec.opp_spot,
                    opp_spot=spec.me_spot,
                )
            )
    else:
        session_lanes = list(specs)

    session = ServerSession(
        host=host,
        port=port,
        lanes=session_lanes,
        frame_skip=frame_skip,
    )
    envs: list[LoLGymEnv] = []
    for i, slane in enumerate(session_lanes):
        env = LoLGymEnv(
            session=session,
            lane_idx=i,
            me_cid=slane.me_cid,
            opp_cid=slane.opp_cid,
            me_spot=slane.me_spot,
            opp_spot=slane.opp_spot,
            max_episode_steps=max_episode_steps,
            frame_skip=frame_skip,
            reward_fn=reward_fn,
            e_flee=e_flee,
            # When mirroring, each env sends only its own me-cid action;
            # the paired env on the same physical lane sends the other
            # side. Together they make exactly 2 actions/tick/lane.
            omit_opp_action=mirror_both_sides,
        )
        envs.append(env)
    return session, envs
