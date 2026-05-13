"""Unit tests for :func:`riftgym.sb3.policy.make_mirror_opp`.

Skipped wholesale if sb3 isn't importable, but the closure itself
doesn't actually touch sb3 — only the underlying model does. We pass
a MagicMock model with a ``.predict`` matching the sb3 contract.
"""

from __future__ import annotations

import os

import pytest

# See test_snapshot_pool.py for why we gate on an env var instead of
# pytest.importorskip — some envs crash at C level on sb3 import.
if not os.environ.get("RIFTGYM_SB3_TESTS"):
    pytest.skip("set RIFTGYM_SB3_TESTS=1 to run sb3 tests", allow_module_level=True)

from typing import Any
from unittest.mock import MagicMock

import numpy as np

from riftgym.sb3.policy import make_mirror_opp


class _FakeEnv:
    """Just enough of LoLGymEnv's surface for the mirror closure to run.
    ``_decode_for(action, obs, my_cid, other_cid)`` is what mirror calls
    to convert the predicted action into a bridge action dict."""

    def __init__(self, me_cid: int = 0, opp_cid: int = 1) -> None:
        self.me_cid = me_cid
        self.opp_cid = opp_cid

    def _decode_for(
        self, action: int, obs: dict[str, Any], my_cid: int, other_cid: int
    ) -> dict[str, Any]:
        return {
            "type": "fake_action",
            "client_id": my_cid,
            "other": other_cid,
            "action_idx": int(action),
        }


def _obs_with_two_champs() -> dict[str, Any]:
    """Minimal obs the encoder can ingest. Schema mirrors a real bridge
    frame (see Tier 1 connectivity smoke): every champ needs hp/max_hp,
    mp/max_mp, level, x/y, gold, alive, net_id, team, plus a `spells`
    list so the action mask doesn't crash on its cooldown / mana
    lookups."""
    return {
        "tick": 1000,
        "champs": [
            {
                "client_id": 0, "alive": True, "hp": 100, "max_hp": 100,
                "mp": 50, "max_mp": 50, "x": 6000.0, "y": 6000.0,
                "level": 1, "gold": 0, "net_id": 100, "team": "TEAM_ORDER",
                "spells": [{"level": 1, "cooldown_s": 0.0, "mana_cost": 0.0}] * 4,
            },
            {
                "client_id": 1, "alive": True, "hp": 100, "max_hp": 100,
                "mp": 50, "max_mp": 50, "x": 8000.0, "y": 8000.0,
                "level": 1, "gold": 0, "net_id": 200, "team": "TEAM_CHAOS",
                "spells": [{"level": 1, "cooldown_s": 0.0, "mana_cost": 0.0}] * 4,
            },
        ],
        "missiles": [],
    }


def test_mirror_uses_live_model_by_default() -> None:
    model = MagicMock()
    model.predict.return_value = (np.int64(8), None)  # attack
    env = _FakeEnv(me_cid=0, opp_cid=1)
    opp_policy = make_mirror_opp(model)

    action = opp_policy(env, _obs_with_two_champs())

    assert model.predict.called
    # The closure encodes obs from opp's perspective; decode must
    # therefore label the action with opp_cid as 'client_id'.
    assert action == {
        "type": "fake_action",
        "client_id": 1,
        "other": 0,
        "action_idx": 8,
    }


def test_mirror_passes_action_mask_to_predict() -> None:
    """Opp must not sample illegal actions (cooldown / dead target etc).
    The closure must forward an action mask to predict()."""
    model = MagicMock()
    model.predict.return_value = (np.int64(0), None)
    opp_policy = make_mirror_opp(model)
    opp_policy(_FakeEnv(), _obs_with_two_champs())

    _, kwargs = model.predict.call_args
    assert "action_masks" in kwargs
    mask = kwargs["action_masks"]
    assert mask.shape == (13,)
    assert mask.dtype == np.bool_


def test_mirror_falls_back_when_model_rejects_action_masks() -> None:
    """Plain (non-Maskable) sb3 models reject action_masks kwarg. The
    closure must catch the TypeError and retry without it."""

    class _ModelNoMask:
        def predict(self, obs, *, deterministic: bool = False, **kwargs: Any):
            if "action_masks" in kwargs:
                raise TypeError("got an unexpected keyword argument 'action_masks'")
            return np.int64(5), None

    opp_policy = make_mirror_opp(_ModelNoMask())
    action = opp_policy(_FakeEnv(), _obs_with_two_champs())
    assert action["action_idx"] == 5


def test_mirror_uses_episode_snapshot_override() -> None:
    """When env has ``_opp_model_override`` set (per SnapshotPool reset),
    use that frozen snapshot instead of the live model — OAI-Five
    Appendix N strategy diversification."""
    live_model = MagicMock()
    live_model.predict.return_value = (np.int64(0), None)
    snapshot = MagicMock()
    snapshot.predict.return_value = (np.int64(12), None)  # R

    env = _FakeEnv()
    env._opp_model_override = snapshot
    opp_policy = make_mirror_opp(live_model)

    action = opp_policy(env, _obs_with_two_champs())

    assert action["action_idx"] == 12
    snapshot.predict.assert_called_once()
    live_model.predict.assert_not_called()
