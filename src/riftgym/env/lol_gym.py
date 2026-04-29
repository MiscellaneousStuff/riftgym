"""Gymnasium ``Env`` wrapper over :class:`riftgym.env.lol_env.LoLEnv` for 1v1.

Agent controls ``me_cid`` (default 0). Opponent ``opp_cid`` (default 1) plays
a uniform-random policy by default; pass ``opp_policy`` to override (e.g. a
trained model, or set ``opp_policy=None`` to leave the engine BT in control).

Action space: ``Discrete(13)``. See :mod:`riftgym.lib.encoding`.
Observation space: ``Box(float32, (110,))``. See :mod:`riftgym.lib.encoding`.

Episode termination:
  - ``terminated`` when either champ dies this step
  - ``truncated`` when ``step_count >= max_episode_steps``
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from riftgym.env.lol_env import LoLEnv
from riftgym.env.rewards import calc_reward
from riftgym.lib.encoding import (
    E_BLINK_RANGE,
    MOVE_DIRS,
    MOVE_STEP,
    N_ACTIONS,
    OBS_DIM,
    action_mask,
    encode,
    find_champ,
    random_action,
)

Obs = dict[str, Any]
Action = dict[str, Any]
RewardFn = Callable[..., float]
OppPolicy = Callable[["LoLGymEnv", Obs], Action]


def _default_random_opp(env: LoLGymEnv, obs: Obs) -> Action:
    return random_action(env.rng, env.opp_cid, env.me_cid, obs)


class LoLGymEnv(gym.Env[NDArray[np.float32], np.int64]):
    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 5120,
        me_cid: int = 0,
        opp_cid: int = 1,
        me_spot: tuple[float, float] = (6000.0, 6000.0),
        opp_spot: tuple[float, float] = (8000.0, 8000.0),
        max_episode_steps: int = 80,
        frame_skip: int = 8,
        reward_fn: RewardFn | None = None,
        e_flee: bool = False,
        opp_policy: OppPolicy | None = None,
        omit_opp_action: bool = False,
        reset_jitter_hp: float = 0.0,
        reset_jitter_mp: float = 0.0,
    ) -> None:
        super().__init__()
        self.me_cid = me_cid
        self.opp_cid = opp_cid
        self.max_episode_steps = max_episode_steps
        self.e_flee = e_flee
        self.reward_fn: RewardFn = reward_fn if reward_fn is not None else calc_reward
        self.opp_policy: OppPolicy = opp_policy if opp_policy is not None else _default_random_opp
        # When True, ``step()`` does not call ``opp_policy`` and does not
        # send the opp action. Useful when something else is driving the
        # opp client_id (e.g. a human, or the engine BT for vs-engine-bot
        # eval).
        self.omit_opp_action = omit_opp_action
        # Initial-state randomization (OAI Five Appendix O.2). When > 0,
        # each episode samples a per-cid fraction in [1 - jitter, 1.0]
        # and the bridge clamps CurrentHealth/Mana to that fraction of
        # max. Asymmetric jitter breaks the symmetric-Nash equilibrium
        # mirror self-play converges to.
        self.reset_jitter_hp = reset_jitter_hp
        self.reset_jitter_mp = reset_jitter_mp

        # Seeded by reset(seed=...). Default opp + jitter draw from this.
        self.rng: np.random.Generator = np.random.default_rng()

        self._env = LoLEnv(
            host=host,
            port=port,
            claim_ids=[me_cid, opp_cid],
            spots=[
                {"client_id": me_cid, "x": me_spot[0], "y": me_spot[1]},
                {"client_id": opp_cid, "x": opp_spot[0], "y": opp_spot[1]},
            ],
            frame_skip=frame_skip,
        )
        self._prev_obs: Obs | None = None
        self._step_count = 0
        self._leveled = False

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        extra_spot_fields: dict[int, dict[str, Any]] | None = None
        if self.reset_jitter_hp > 0.0 or self.reset_jitter_mp > 0.0:
            extra_spot_fields = {}
            for cid in (self.me_cid, self.opp_cid):
                fields: dict[str, Any] = {}
                if self.reset_jitter_hp > 0.0:
                    fields["hp_frac"] = 1.0 - float(self.rng.uniform(0.0, self.reset_jitter_hp))
                if self.reset_jitter_mp > 0.0:
                    fields["mp_frac"] = 1.0 - float(self.rng.uniform(0.0, self.reset_jitter_mp))
                extra_spot_fields[cid] = fields

        obs = self._env.reset(extra_spot_fields=extra_spot_fields)
        if not self._leveled:
            self._level_spells_once()
            obs = self._env.step([])
        obs = self._level_unleveled_spells(obs)
        self._prev_obs = obs
        self._step_count = 0
        return encode(obs, self.me_cid, self.opp_cid), {}

    def step(
        self, action: np.int64 | int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        assert self._prev_obs is not None, "step() before reset()"
        my_a = self._decode(int(action), self._prev_obs)
        if self.omit_opp_action:
            actions = [my_a]
        else:
            opp_a = self.opp_policy(self, self._prev_obs)
            actions = [my_a, opp_a]
        obs = self._env.step(actions)
        reward = self.reward_fn(self._prev_obs, obs, me=self.me_cid, opp=self.opp_cid)
        me = find_champ(obs, self.me_cid)
        opp = find_champ(obs, self.opp_cid)
        terminated = (not me["alive"]) or (not opp["alive"])
        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps
        self._prev_obs = obs
        info = {"me_hp": me["hp"], "opp_hp": opp["hp"]}
        return (
            encode(obs, self.me_cid, self.opp_cid),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def close(self) -> None:
        self._env.close()

    def action_masks(self) -> NDArray[np.bool_]:
        """Action mask for sb3-contrib's MaskablePPO. 13-bool array, True = allowed."""
        if self._prev_obs is None:
            return np.ones(N_ACTIONS, dtype=np.bool_)
        return action_mask(self._prev_obs, self.me_cid, self.opp_cid)

    # ---- internals --------------------------------------------------------

    def _decode(self, action: int, obs: Obs) -> Action:
        return self._decode_for(action, obs, self.me_cid, self.opp_cid)

    def _decode_for(self, action: int, obs: Obs, my_cid: int, other_cid: int) -> Action:
        me = find_champ(obs, my_cid)
        opp = find_champ(obs, other_cid)
        a = int(action)
        if 0 <= a < 8:
            dx, dy = MOVE_DIRS[a]
            return {
                "type": "move",
                "client_id": my_cid,
                "x": me["x"] + dx * MOVE_STEP,
                "y": me["y"] + dy * MOVE_STEP,
            }
        if a == 8:
            return {
                "type": "attack",
                "client_id": my_cid,
                "target_net_id": opp["net_id"],
            }
        slot = a - 9  # 9..12 -> Q/W/E/R
        if slot == 2 and self.e_flee:
            dx = me["x"] - opp["x"]
            dy = me["y"] - opp["y"]
            mag = math.hypot(dx, dy) or 1.0
            tx = me["x"] + (dx / mag) * E_BLINK_RANGE
            ty = me["y"] + (dy / mag) * E_BLINK_RANGE
            return {
                "type": "spell",
                "client_id": my_cid,
                "slot": slot,
                "x": tx,
                "y": ty,
            }
        return {
            "type": "spell",
            "client_id": my_cid,
            "slot": slot,
            "x": opp["x"],
            "y": opp["y"],
        }

    def _level_spells_once(self) -> None:
        acts: list[Action] = []
        for cid in (self.me_cid, self.opp_cid):
            for slot in range(4):
                acts.append({"type": "level_spell", "client_id": cid, "slot": slot})
        self._env.step(acts)
        self._leveled = True

    def _level_unleveled_spells(self, obs: Obs) -> Obs:
        """Idempotent re-level: fires ``level_spell`` only for spells whose
        current level is 0 in ``obs``. Handles the human-vs-bot edge case
        where a real LoL client connection resets the human cid's spell
        state after the initial ``_level_spells_once`` ran. Safe to call
        every reset because it no-ops once everything's at level 1.
        """
        by_cid = {c["client_id"]: c for c in obs["champs"]}
        acts: list[Action] = []
        for cid in (self.me_cid, self.opp_cid):
            c = by_cid.get(cid)
            if c is None:
                continue
            spells = c.get("spells") or []
            for slot in range(min(4, len(spells))):
                if (spells[slot] or {}).get("level", 0) == 0:
                    acts.append({"type": "level_spell", "client_id": cid, "slot": slot})
        if not acts:
            return obs
        return self._env.step(acts)
