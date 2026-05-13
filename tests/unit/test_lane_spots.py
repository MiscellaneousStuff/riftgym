"""Unit tests for :mod:`riftgym.lib.lane_spots`."""

from __future__ import annotations

import math

from riftgym.lib.lane_spots import LANE_PAIR_OFFSET, LANE_SPOT_TABLE, MAX_LANES


def test_table_has_canonical_names_first() -> None:
    """First five entries cover the canonical lanes; the rest are
    scratch for >5-lane training experiments."""
    names = [n for n, _ in LANE_SPOT_TABLE[:5]]
    assert names == ["TOP", "TOP_JNG", "MID", "BOT_JNG", "BOT"]


def test_max_lanes_matches_table_len() -> None:
    assert MAX_LANES == len(LANE_SPOT_TABLE)


def test_spots_fit_inside_summoners_rift() -> None:
    """Map is roughly 14000 units square; lane bubbles must fit
    entirely inside even after the ±LANE_PAIR_OFFSET spread."""
    for _name, (x, y) in LANE_SPOT_TABLE:
        assert 0.0 < x - LANE_PAIR_OFFSET
        assert x + LANE_PAIR_OFFSET < 14000.0
        assert 0.0 < y < 14000.0


def test_adjacent_lanes_are_well_separated() -> None:
    """All adjacent spots must be >= 2000u apart — well outside
    Ezreal Q range (~1150u). Without this gap a cross-lane skillshot
    could blur lane identity and corrupt the per-lane reward signal."""
    spots = [s for _, s in LANE_SPOT_TABLE]
    for i, a in enumerate(spots):
        for b in spots[i + 1 :]:
            d = math.hypot(a[0] - b[0], a[1] - b[1])
            assert d >= 2000.0, f"lanes too close: {a} <-> {b} = {d:.0f}u"
