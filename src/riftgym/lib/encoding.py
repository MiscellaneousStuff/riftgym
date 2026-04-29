"""Observation encoding and action mask helpers for the 1v1 LoL gym env.

These were ``_encode`` / ``_mask_for`` / ``_random_action`` in
``brokenwings/tools/lol_gym.py``; promoted to public names here so the
trainer's mirror-opp closure can reuse the same encoding logic.

Action space (13 discrete actions, indices match :data:`MOVE_DIRS` order):

  - 0..7  move 300u in 8 compass dirs (N, NE, E, SE, S, SW, W, NW)
  - 8     attack opponent (target_net_id)
  - 9..12 cast Q / W / E / R aimed at opponent's current position

Observation: flat 110-dim float32 vector. Per champ (me then opp):
7 base feats + 8 nearest-missile feats * 6 fields each (= 55 per champ).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

# ---- constants -------------------------------------------------------------

MAP_SCALE = 14000.0  # Summoner's Rift is ~14800 square; keeps x_norm in ~[0, 1]
MAX_LEVEL = 18
MAX_GOLD = 2000.0
MOVE_STEP = 300.0
E_BLINK_RANGE = 475.0  # Ezreal Arcane Shift max blink

# (dx, dy) for actions 0..7. Engine is (X, worldZ); "y" is really world-Z.
MOVE_DIRS: tuple[tuple[float, float], ...] = (
    (0.0, 1.0),  # N
    (0.7, 0.7),  # NE
    (1.0, 0.0),  # E
    (0.7, -0.7),  # SE
    (0.0, -1.0),  # S
    (-0.7, -0.7),  # SW
    (-1.0, 0.0),  # W
    (-0.7, 0.7),  # NW
)

N_MISSILES_PER_CHAMP = 8
MISSILE_FIELDS = 6  # rel_x, rel_y, vx, vy, is_enemy, is_present
MISSILE_RADIUS = 2500.0
VEL_SCALE = 3000.0  # max sensible missile speed; scales velocity to ~[-1, 1]
PER_CHAMP_FEATS = 7  # hp_frac, mp_frac, level, x, y, gold, alive
PER_CHAMP_MISSILE_FEATS = N_MISSILES_PER_CHAMP * MISSILE_FIELDS  # 48
OBS_DIM = 2 * (PER_CHAMP_FEATS + PER_CHAMP_MISSILE_FEATS)  # 110
N_ACTIONS = 13

Obs = dict[str, Any]
Champ = dict[str, Any]
Action = dict[str, Any]


# ---- helpers ---------------------------------------------------------------


def find_champ(obs: Obs, cid: int) -> Champ:
    for c in obs["champs"]:
        if c["client_id"] == cid:
            return c
    raise KeyError(f"cid={cid} not in obs")


def _missile_feats_for(
    missiles: list[dict[str, Any]],
    champ: Champ,
    opp_team: str,
) -> list[float]:
    """Nearest ``N_MISSILES_PER_CHAMP`` missiles within ``MISSILE_RADIUS`` of
    ``champ``, encoded as ``(rel_x, rel_y, vx, vy, is_enemy, is_present)``,
    normalized. Returns a flat list of length ``N_MISSILES_PER_CHAMP *
    MISSILE_FIELDS``, padded with zeros when fewer missiles are in range.
    """
    cx, cy = champ["x"], champ["y"]
    in_range: list[tuple[float, float, float, dict[str, Any]]] = []
    for m in missiles:
        dx = m["x"] - cx
        dy = m["y"] - cy
        d2 = dx * dx + dy * dy
        if d2 <= MISSILE_RADIUS * MISSILE_RADIUS:
            in_range.append((d2, dx, dy, m))
    in_range.sort(key=lambda t: t[0])
    out: list[float] = []
    for _d2, dx, dy, m in in_range[:N_MISSILES_PER_CHAMP]:
        is_enemy = 1.0 if m["team"] != champ["team"] else 0.0
        out.extend(
            [
                dx / MISSILE_RADIUS,
                dy / MISSILE_RADIUS,
                m["vx"] / VEL_SCALE,
                m["vy"] / VEL_SCALE,
                is_enemy,
                1.0,
            ]
        )
    pad = N_MISSILES_PER_CHAMP - len(in_range[:N_MISSILES_PER_CHAMP])
    out.extend([0.0] * (pad * MISSILE_FIELDS))
    return out


def encode(obs: Obs, me_cid: int, opp_cid: int) -> NDArray[np.float32]:
    """Flatten a raw bridge observation to a 110-dim float32 vector."""
    me = find_champ(obs, me_cid)
    opp = find_champ(obs, opp_cid)
    missiles: list[dict[str, Any]] = obs.get("missiles", [])

    def feats(c: Champ) -> list[float]:
        return [
            c["hp"] / max(1.0, c["max_hp"]),
            c["mp"] / max(1.0, c["max_mp"]),
            c["level"] / MAX_LEVEL,
            c["x"] / MAP_SCALE,
            c["y"] / MAP_SCALE,
            min(1.0, c["gold"] / MAX_GOLD),
            1.0 if c["alive"] else 0.0,
        ]

    me_vec = feats(me) + _missile_feats_for(missiles, me, opp["team"])
    opp_vec = feats(opp) + _missile_feats_for(missiles, opp, me["team"])
    return np.asarray(me_vec + opp_vec, dtype=np.float32)


def action_mask(obs: Obs, my_cid: int, other_cid: int) -> NDArray[np.bool_]:
    """13-bool action mask from ``my_cid``'s perspective.

    - moves (0..7): always allowed
    - attack (8): only if opp is alive
    - spells (9..12): only if me is alive AND off-cooldown AND mana sufficient
    """
    mask = np.ones(N_ACTIONS, dtype=np.bool_)
    me = find_champ(obs, my_cid)
    opp = find_champ(obs, other_cid)
    mask[8] = bool(opp["alive"])
    spells = me.get("spells") or []
    me_alive = bool(me["alive"])
    me_mp = float(me.get("mp", 0.0))
    for slot in range(4):
        idx = 9 + slot
        if not me_alive or slot >= len(spells):
            mask[idx] = False
            continue
        sp = spells[slot]
        on_cd = float(sp.get("cooldown_s", 0.0)) > 0.0
        mana_ok = me_mp >= float(sp.get("mana_cost", 0.0))
        mask[idx] = (not on_cd) and mana_ok
    return mask


def random_action(rng: np.random.Generator, cid: int, opp_cid: int, obs: Obs) -> Action:
    """Uniform random action over move / attack / Q W E R aimed at opp.

    Same shape as the legacy ``random_policy_smoke`` policy, but seeded via
    a caller-provided :class:`numpy.random.Generator` instead of the
    module-level :mod:`random` state.
    """
    me = find_champ(obs, cid)
    op = find_champ(obs, opp_cid)
    kind = rng.choice(["move", "attack", "spell_q", "spell_w", "spell_e", "spell_r"])
    if kind == "move":
        choices = [(0, 0), (300, 0), (-300, 0), (0, 300), (0, -300)]
        idx = int(rng.integers(0, len(choices)))
        dx, dy = choices[idx]
        return {
            "type": "move",
            "client_id": cid,
            "x": me["x"] + dx,
            "y": me["y"] + dy,
        }
    if kind == "attack":
        return {"type": "attack", "client_id": cid, "target_net_id": op["net_id"]}
    slot = {"spell_q": 0, "spell_w": 1, "spell_e": 2, "spell_r": 3}[str(kind)]
    return {
        "type": "spell",
        "client_id": cid,
        "slot": slot,
        "x": op["x"],
        "y": op["y"],
    }
