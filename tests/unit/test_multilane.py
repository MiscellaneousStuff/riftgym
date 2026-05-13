"""Unit tests for :mod:`riftgym.env.multilane`.

No live server / docker — these are pure data-structure assertions
about the lane-spec factory plus the env-wiring shape of
:func:`make_multilane_envs`. The session it constructs never actually
connects (no ``_ensure_initialized`` called).
"""

from __future__ import annotations

import pytest

from riftgym.env.lol_gym import LoLGymEnv
from riftgym.env.multilane import build_lane_specs, make_multilane_envs
from riftgym.env.session import LaneSpec, ServerSession
from riftgym.lib.lane_spots import LANE_PAIR_OFFSET, LANE_SPOT_TABLE


def test_build_lane_specs_default_cids() -> None:
    """With no cid overrides, Blue cids = 0..N-1 and Purple = N..2N-1.
    This matches the canonical brokenwings multilane GameInfo layout."""
    specs = build_lane_specs(5)
    assert len(specs) == 5
    for i, spec in enumerate(specs):
        assert isinstance(spec, LaneSpec)
        assert spec.me_cid == i
        assert spec.opp_cid == 5 + i


def test_build_lane_specs_uses_lane_spot_table() -> None:
    """Lane i's spots straddle LANE_SPOT_TABLE[i] on the X axis with
    ±LANE_PAIR_OFFSET. Keeps champs in range of each other while not
    blowing the lane bubble."""
    specs = build_lane_specs(3)
    for i, spec in enumerate(specs):
        x, y = LANE_SPOT_TABLE[i][1]
        assert spec.me_spot == (x - LANE_PAIR_OFFSET, y)
        assert spec.opp_spot == (x + LANE_PAIR_OFFSET, y)


def test_build_lane_specs_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="not in"):
        build_lane_specs(0)
    with pytest.raises(ValueError, match="not in"):
        build_lane_specs(len(LANE_SPOT_TABLE) + 1)


def test_build_lane_specs_custom_cids() -> None:
    """Custom cid lists are useful when a settings JSON interleaves
    Blue/Purple instead of contiguous blocks."""
    specs = build_lane_specs(2, blue_cids=[10, 11], purple_cids=[20, 21])
    assert [s.me_cid for s in specs] == [10, 11]
    assert [s.opp_cid for s in specs] == [20, 21]


def test_build_lane_specs_rejects_short_cid_lists() -> None:
    with pytest.raises(ValueError, match="need >="):
        build_lane_specs(3, blue_cids=[0, 1])


def test_make_multilane_envs_one_env_per_lane() -> None:
    """Default (no mirror) mode: N specs in → N envs out, one per lane."""
    session, envs = make_multilane_envs(n_lanes=3)
    try:
        assert isinstance(session, ServerSession)
        assert len(envs) == 3
        assert session.n_lanes == 3
        for i, env in enumerate(envs):
            assert isinstance(env, LoLGymEnv)
            assert env._lane_idx == i
            assert env._session is session
            assert env._env is None  # multilane envs don't own their own bridge
            # me/opp wiring matches the session's lane spec
            assert env.me_cid == session.lanes[i].me_cid
            assert env.opp_cid == session.lanes[i].opp_cid
    finally:
        session.close()


def test_make_multilane_envs_mirror_doubles_envs() -> None:
    """`mirror_both_sides=True`: each physical 1v1 yields TWO session
    lanes with swapped me/opp, and TWO LoLGymEnvs sharing one tick."""
    session, envs = make_multilane_envs(n_lanes=3, mirror_both_sides=True)
    try:
        assert len(envs) == 6
        assert session.n_lanes == 6
        # Pair (0, 1), (2, 3), (4, 5) are mirrors. LoLGymEnv doesn't
        # store me_spot/opp_spot as attrs (they live in the session
        # lane spec for multilane mode), so check the session side.
        for phys in range(3):
            blue, purple = session.lanes[2 * phys], session.lanes[2 * phys + 1]
            assert blue.me_cid == purple.opp_cid
            assert blue.opp_cid == purple.me_cid
            assert blue.me_spot == purple.opp_spot
            assert blue.opp_spot == purple.me_spot
        for phys in range(3):
            blue_env, purple_env = envs[2 * phys], envs[2 * phys + 1]
            assert blue_env.me_cid == purple_env.opp_cid
            assert blue_env.opp_cid == purple_env.me_cid
    finally:
        session.close()


def test_make_multilane_envs_mirror_omits_opp_action() -> None:
    """Mirror mode: each env sends only its own me-cid action; the
    paired env sends the other side. Without `omit_opp_action=True`
    we'd send 4 actions/tick/lane instead of 2."""
    session, envs = make_multilane_envs(n_lanes=2, mirror_both_sides=True)
    try:
        for env in envs:
            assert env.omit_opp_action is True
    finally:
        session.close()


def test_make_multilane_envs_default_does_not_omit_opp_action() -> None:
    """Non-mirror mode: opp_policy drives the opp side, so the env
    sends both actions."""
    session, envs = make_multilane_envs(n_lanes=2, mirror_both_sides=False)
    try:
        for env in envs:
            assert env.omit_opp_action is False
    finally:
        session.close()


def test_make_multilane_envs_rejects_empty_specs() -> None:
    with pytest.raises(ValueError):
        make_multilane_envs(lane_specs=[])
