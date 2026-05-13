"""Mirror self-play opp policy.

:func:`make_mirror_opp` builds the closure
:class:`riftgym.env.lol_gym.LoLGymEnv` calls as ``opp_policy``. The
closure encodes the obs from opp's perspective (me/opp swapped), asks
the model for an action, then decodes it back into the bridge action
dict for the opp client_id.

Using the **live model** means opp updates in lockstep with the trainer
— true mirror self-play. Nonstationary, but no snapshot bookkeeping.

**Snapshot-pool variant**: if the env has ``_opp_model_override`` set
(per :meth:`LoLGymEnv.reset` when a :class:`SnapshotPool` is attached),
that frozen snapshot acts as opp for the duration of the episode.
Otherwise falls back to the live ``model``. Per-episode resampling is
the OpenAI-Five Appendix N pattern that prevents strategy collapse.

For :class:`MaskablePPO` models, also passes opp's action mask via
``predict()`` so opp doesn't sample illegal actions (cast on cooldown,
attack dead opp, etc.) and waste a step.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from riftgym.lib.encoding import action_mask, encode

Obs = dict[str, Any]
Action = dict[str, Any]
OppPolicy = Callable[[Any, Obs], Action]


def make_mirror_opp(model: Any, deterministic: bool = False) -> OppPolicy:
    """Return an ``opp_policy`` callable wired to ``model``.

    Args:
        model: an SB3-style policy with a ``.predict(obs, deterministic,
            action_masks)`` API. Typically a :class:`MaskablePPO`.
        deterministic: argmax sampling instead of stochastic. Useful
            for eval; leave False for training so the opp explores.
    """

    def _mirror(env: Any, obs: Obs) -> Action:
        # Per-episode snapshot pool override (set by env.reset() when a
        # SnapshotPool with prob > 0 is attached). Falls back to the
        # live model whenever the override isn't set.
        active = getattr(env, "_opp_model_override", None) or model
        opp_feat = encode(obs, env.opp_cid, env.me_cid)
        opp_mask = action_mask(obs, env.opp_cid, env.me_cid)
        try:
            action, _ = active.predict(
                opp_feat,
                deterministic=deterministic,
                action_masks=opp_mask,
            )
        except TypeError:
            # Fallback for non-Maskable models that reject the kwarg.
            action, _ = active.predict(opp_feat, deterministic=deterministic)
        # ``_decode_for`` lives on LoLGymEnv; tight intra-package
        # coupling between env and sb3 layers is intentional here —
        # both ship together as riftgym.
        return env._decode_for(int(action), obs, env.opp_cid, env.me_cid)

    return _mirror
