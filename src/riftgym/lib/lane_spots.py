"""Lane spot table for multilane training.

Mirrors brokenwings's per-lane spawn coordinates so configuration sweeps
between the two repos stay aligned. All adjacent spots are >= 2000u
apart, well outside Ezreal's Q range (~1150u), so cross-lane skillshots
and auto-attacks can't blur lane identity. All spots fit inside the
14000-unit-square Summoner's Rift map.

Each entry is a ``(name, (x, y))`` pair; index in :data:`LANE_SPOT_TABLE`
is the lane index used by :class:`riftgym.env.session.ServerSession`.
"""

from __future__ import annotations

LaneSpot = tuple[float, float]

# Order matters — index = lane_idx. The first five mirror the canonical
# top/jungle/mid/jungle/bot positions; the trailing entries are extra
# scratch lanes for >5-lane training experiments.
LANE_SPOT_TABLE: list[tuple[str, LaneSpot]] = [
    ("TOP",     ( 3000.0, 12000.0)),
    ("TOP_JNG", ( 5500.0,  9500.0)),
    ("MID",     ( 7500.0,  7500.0)),
    ("BOT_JNG", ( 9500.0,  5500.0)),
    ("BOT",     (12000.0,  3000.0)),
    ("LANE_5",  ( 3000.0,  9000.0)),
    ("LANE_6",  ( 5500.0,  6000.0)),
    ("LANE_7",  ( 9500.0,  9000.0)),
    ("LANE_8",  (12000.0,  6000.0)),
    ("LANE_9",  ( 7500.0,  3500.0)),
]

# Half-distance between the two champs in a 1v1 lane pair on the X axis.
# Keeps Blue and Purple well inside auto-attack + Q range of each other
# while leaving room for movement before they leave the lane bubble.
LANE_PAIR_OFFSET = 250.0

MAX_LANES = len(LANE_SPOT_TABLE)
