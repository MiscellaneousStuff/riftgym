"""Unit tests for ``riftgym.lib.encoding``.

Uses canned obs dicts mimicking the bridge's wire format — no game server
needed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from riftgym.lib import encoding


def _champ(
    *,
    cid: int,
    team: str = "TEAM_ORDER",
    x: float = 7000.0,
    y: float = 7000.0,
    hp: float = 600.0,
    max_hp: float = 600.0,
    mp: float = 300.0,
    max_mp: float = 300.0,
    gold: float = 500.0,
    level: int = 1,
    alive: bool = True,
    net_id: int = 1000,
    spells: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "client_id": cid,
        "net_id": net_id,
        "team": team,
        "champion": "Ezreal",
        "alive": alive,
        "x": x,
        "y": y,
        "hp": hp,
        "max_hp": max_hp,
        "mp": mp,
        "max_mp": max_mp,
        "gold": gold,
        "level": level,
        "spells": spells if spells is not None else [],
    }


def _obs(
    champs: list[dict[str, Any]], missiles: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return {
        "tick": 1000,
        "state": "GAMELOOP",
        "champs": champs,
        "missiles": missiles or [],
    }


def test_encode_shape_and_dtype() -> None:
    obs = _obs([_champ(cid=0), _champ(cid=1, team="TEAM_CHAOS")])
    feats = encoding.encode(obs, me_cid=0, opp_cid=1)
    assert feats.shape == (encoding.OBS_DIM,)
    assert feats.dtype == np.float32


def test_encode_normalization() -> None:
    obs = _obs(
        [
            _champ(cid=0, hp=300.0, max_hp=600.0, x=7000.0, y=7000.0, level=9),
            _champ(cid=1, team="TEAM_CHAOS", x=7000.0, y=7000.0),
        ]
    )
    feats = encoding.encode(obs, me_cid=0, opp_cid=1)
    # Per-champ feats: hp_frac, mp_frac, level_frac, x_norm, y_norm, gold_norm, alive
    assert feats[0] == np.float32(0.5)
    assert feats[2] == np.float32(9.0 / encoding.MAX_LEVEL)
    assert feats[3] == np.float32(7000.0 / encoding.MAP_SCALE)


def test_action_mask_dead_opp_blocks_attack() -> None:
    obs = _obs([_champ(cid=0), _champ(cid=1, alive=False)])
    mask = encoding.action_mask(obs, my_cid=0, other_cid=1)
    assert mask.shape == (encoding.N_ACTIONS,)
    assert mask[8] is np.bool_(False) or mask[8] == False  # noqa: E712 — numpy bool comparison
    # All move slots remain allowed
    assert all(mask[:8])


def test_action_mask_spell_on_cooldown_blocked() -> None:
    obs = _obs(
        [
            _champ(
                cid=0,
                spells=[
                    {"level": 1, "cooldown_s": 0.0, "mana_cost": 50.0},
                    {"level": 1, "cooldown_s": 5.0, "mana_cost": 50.0},  # on cooldown
                    {"level": 1, "cooldown_s": 0.0, "mana_cost": 9999.0},  # mana-blocked
                    {"level": 1, "cooldown_s": 0.0, "mana_cost": 50.0},
                ],
                mp=100.0,
            ),
            _champ(cid=1),
        ]
    )
    mask = encoding.action_mask(obs, my_cid=0, other_cid=1)
    assert mask[9] == True  # noqa: E712 — Q ready
    assert mask[10] == False  # noqa: E712 — W on cooldown
    assert mask[11] == False  # noqa: E712 — E mana-blocked
    assert mask[12] == True  # noqa: E712 — R ready


def test_random_action_is_valid_dict() -> None:
    obs = _obs([_champ(cid=0), _champ(cid=1)])
    rng = np.random.default_rng(42)
    a = encoding.random_action(rng, cid=0, opp_cid=1, obs=obs)
    assert "type" in a
    assert a["client_id"] == 0
    assert a["type"] in {"move", "attack", "spell"}


def test_random_action_seeded_reproducible() -> None:
    obs = _obs([_champ(cid=0), _champ(cid=1)])
    a1 = encoding.random_action(np.random.default_rng(42), cid=0, opp_cid=1, obs=obs)
    a2 = encoding.random_action(np.random.default_rng(42), cid=0, opp_cid=1, obs=obs)
    assert a1 == a2
