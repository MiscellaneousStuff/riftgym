"""``riftgym-eval`` — run a trained checkpoint against the engine BT.

End-to-end milestone for issue #1: from a fresh install, one command:

    riftgym-eval ./checkpoints/ppo_lol_40000_steps.zip \\
        --vs-engine-bot --episodes 50

  - pulls the brokenwings image (or uses a local one with ``--pull never``)
  - starts the server-only container
  - loads the checkpoint, plays N episodes (agent cid=0 vs engine BT cid=1)
  - prints W/L/D + win rate, exits cleanly

Currently the only mode is ``--vs-engine-bot``. Mirror self-play and
head-to-head modes will land in follow-up issues.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
from pathlib import Path
from typing import Any

from riftgym.launcher import (
    DEFAULT_BASE_GAME_PORT,
    DEFAULT_BASE_RL_PORT,
    ServerLauncher,
)
from riftgym.run_configs import ContainerRunConfig

DEFAULT_IMAGE = "ghcr.io/miscellaneousstuff/brokenwings"
DEFAULT_TAG = "latest"

# Tight 1v1 spawn — ~849u apart on the diagonal, well inside Ezreal Q
# range and inside the engine BT's KillChampionAttackSequence threat
# radius. Forces immediate engagement on respawn instead of the BT
# walking off to its assigned 1v1 lane.
BLUE_SPOT_VS_BT = (6700.0, 6700.0)
PURPLE_SPOT_VS_BT = (7300.0, 7300.0)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="riftgym-eval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("model", type=Path, help="Path to a saved MaskablePPO model.zip")
    p.add_argument(
        "--vs-engine-bot",
        action="store_true",
        required=True,
        help="Required for now. Agent plays cid=0; engine BT plays cid=1 "
        "unclaimed. Other modes coming in follow-up issues.",
    )
    p.add_argument("--episodes", type=int, default=50, help="Number of episodes to run.")
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Argmax action selection instead of stochastic.",
    )
    p.add_argument("--image", default=DEFAULT_IMAGE, help="Container image (no tag).")
    p.add_argument("--tag", default=DEFAULT_TAG, help="Container image tag.")
    p.add_argument(
        "--pull",
        choices=("missing", "always", "never"),
        default="missing",
        help="Image pull policy. Use 'never' for local dev images.",
    )
    p.add_argument("--base-game-port", type=int, default=DEFAULT_BASE_GAME_PORT)
    p.add_argument("--base-rl-port", type=int, default=DEFAULT_BASE_RL_PORT)
    p.add_argument(
        "--port-ready-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for the bridge port to come up. Cold-starts "
        "of the brokenwings image (Roslyn script compile) can take 30-60s.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    )

    if not args.model.exists():
        print(f"checkpoint not found: {args.model}", file=sys.stderr)
        return 2

    # Imported here so ``riftgym-eval --help`` works even when sb3 isn't
    # installed (it'd be a silly UX regression to require sb3 for --help).
    try:
        from sb3_contrib import MaskablePPO  # pyright: ignore[reportMissingImports]
    except ImportError:
        print(
            "riftgym-eval requires the 'sb3' extra. Install with:\n    pip install 'riftgym[sb3]'",
            file=sys.stderr,
        )
        return 1

    from riftgym.env.lol_gym import LoLGymEnv

    rc = ContainerRunConfig(
        image=args.image,
        tag=args.tag,
        pull_policy=args.pull,
        # Brokenwings's image defaults to play.sh (human-vs-bot demo);
        # riftgym wants the server-only entrypoint. tini reaps the .NET
        # process on container stop.
        entrypoint=("/usr/bin/tini", "--", "/app/entrypoints/server.sh"),
    )

    print(f"Loading checkpoint: {args.model}", flush=True)
    model = MaskablePPO.load(args.model.as_posix(), device="cpu")

    launcher = ServerLauncher(
        n=1,
        run_config=rc,
        base_game_port=args.base_game_port,
        base_rl_port=args.base_rl_port,
        port_ready_timeout_s=args.port_ready_timeout,
    )
    launcher.start()
    try:
        rl_port = launcher.rl_ports[0]
        print(f"Bridge ready on 127.0.0.1:{rl_port}. Connecting env.", flush=True)

        env = LoLGymEnv(
            port=rl_port,
            me_cid=0,
            opp_cid=1,
            me_spot=BLUE_SPOT_VS_BT,
            opp_spot=PURPLE_SPOT_VS_BT,
            max_episode_steps=80,
            frame_skip=8,
            # Don't claim opp; let the engine BT drive cid=1.
            claim_opp=False,
            # Don't let the env send opp actions (no opp_policy).
            omit_opp_action=True,
        )
        try:
            return _run_eval(env, model, episodes=args.episodes, deterministic=args.deterministic)
        finally:
            env.close()
    except BaseException:
        # Dump container logs on failure so the user can see why the
        # server died. Without this, --rm tears down the container
        # before there's any chance to ``docker logs`` it.
        for i, h in enumerate(launcher.handles):
            logs = ""
            with contextlib.suppress(Exception):
                logs = h.logs()
            if logs:
                sys.stderr.write(f"\n--- container {i} logs ---\n{logs}\n")
        raise
    finally:
        launcher.close()


def _run_eval(
    env: Any,
    model: Any,
    *,
    episodes: int,
    deterministic: bool,
) -> int:
    import numpy as np

    wins = losses = draws = 0
    for ep in range(1, episodes + 1):
        obs, _info = env.reset()
        terminated = truncated = False
        ep_len = 0
        info: dict[str, Any] = {}
        while not (terminated or truncated):
            mask = env.action_masks().copy()
            # Apples-to-apples vs the engine BT: it can't reach its
            # UseUltimate branch in our 1v1 sandbox (KillChampionScore
            # gate never satisfies without minions/teammates), so it
            # effectively has Q/W/E only. Mask R off the agent too.
            mask[12] = False
            action_arr, _ = model.predict(
                obs,
                deterministic=deterministic,
                action_masks=mask,
            )
            obs, _reward, terminated, truncated, info = env.step(int(np.asarray(action_arr).item()))
            ep_len += 1

        me_hp = info.get("me_hp", 0)
        opp_hp = info.get("opp_hp", 0)
        me_dead = me_hp <= 0
        opp_dead = opp_hp <= 0
        if me_dead and opp_dead:
            outcome = "DKO"
            draws += 1
        elif opp_dead:
            outcome = "WIN"
            wins += 1
        elif me_dead:
            outcome = "LOSS"
            losses += 1
        else:
            outcome = "DRAW"
            draws += 1
        wr = wins / ep
        print(
            f"ep {ep:3d}  {outcome:4s}  len={ep_len:3d}  "
            f"me_hp={me_hp:.0f}  opp_hp={opp_hp:.0f}  "
            f"| W/L/D={wins}/{losses}/{draws}  win_rate={wr:.2%}",
            flush=True,
        )

    print("\n=== vs engine BT — final ===", flush=True)
    print(f"episodes:  {episodes}", flush=True)
    if episodes > 0:
        print(f"wins:      {wins} ({wins / episodes:.2%})", flush=True)
        print(f"losses:    {losses} ({losses / episodes:.2%})", flush=True)
        print(f"draws:     {draws} ({draws / episodes:.2%})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
