"""Per-step reward functions for the 1v1 environment."""

from __future__ import annotations

import math
from typing import Any

Obs = dict[str, Any]


def calc_reward(
    prev_obs: Obs,
    curr_obs: Obs,
    me: int,
    opp: int,
    *,
    hp_scale: float = 0.01,
    kill_bonus: float = 10.0,
    cs_scale: float = 0.01,
) -> float:
    """Combat reward from ``me``'s perspective on a 1v1.

    + ``hp_scale`` * damage dealt to opp (also rewards opp HP loss from any source)
    - ``hp_scale`` * damage taken by me (symmetric — rewards self-healing)
    + ``kill_bonus`` if opp transitioned alive→dead this step
    - ``kill_bonus`` if me transitioned alive→dead this step
    + ``cs_scale`` * my gold gained this step (clipped at 0; ignores shop spends)

    Simultaneous deaths net to 0 (intentional).
    """
    p = {c["client_id"]: c for c in prev_obs["champs"]}
    c = {c["client_id"]: c for c in curr_obs["champs"]}
    mp, mc = p[me], c[me]
    op, oc = p[opp], c[opp]

    r = 0.0
    r += hp_scale * (op["hp"] - oc["hp"])
    r -= hp_scale * (mp["hp"] - mc["hp"])
    if op["alive"] and not oc["alive"]:
        r += kill_bonus
    if mp["alive"] and not mc["alive"]:
        r -= kill_bonus
    r += cs_scale * max(0.0, mc["gold"] - mp["gold"])
    return r


def calc_reward_distance(
    prev_obs: Obs,
    curr_obs: Obs,
    me: int,
    opp: int,
    *,
    dist_scale: float = 0.001,
) -> float:
    """Delta-distance reward — reward moving away from opp each step.

    + ``dist_scale`` * (curr_distance - prev_distance)

    E blink ≈ 475u → +0.475/step. Walk ≈ 300u/step at 267 ms frame_skip
    → +0.3/step. Optimal policy is flee-E-spam when off cooldown, walk
    away otherwise. Episodes terminate on either death; getting killed
    by a random opp cuts the reward stream short — natural penalty.
    """
    p = {c["client_id"]: c for c in prev_obs["champs"]}
    c = {c["client_id"]: c for c in curr_obs["champs"]}
    prev_d = math.hypot(p[me]["x"] - p[opp]["x"], p[me]["y"] - p[opp]["y"])
    curr_d = math.hypot(c[me]["x"] - c[opp]["x"], c[me]["y"] - c[opp]["y"])
    return dist_scale * (curr_d - prev_d)
