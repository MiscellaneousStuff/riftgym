"""``riftgym-train`` — PPO over LoLGymEnv with mirror self-play.

Thin argparse → :class:`riftgym.sb3.trainer.TrainConfig` wrapper.
All training logic lives in :func:`riftgym.sb3.trainer.train_ppo`.

Defaults match the brokenwings 40k baseline recipe::

    riftgym-train --auto-launch 4 --lanes 5 --mirror-both-sides \\
        --total-steps 200000 --compose-file ./compose.yaml

After ``--total-steps`` env steps the final model lands at
``<log_dir>/model.zip``. Resume with ``--resume <ckpt.zip>``;
``--total-steps`` then counts ADDITIONAL steps, not a global cap.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from riftgym.run_configs import ComposeRunConfig, ContainerRunConfig

DEFAULT_IMAGE = "ghcr.io/miscellaneousstuff/brokenwings"
DEFAULT_TAG = "release"


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="riftgym-train",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--auto-launch",
        type=int,
        default=4,
        help="Number of servers to spawn. Default 4 = the brokenwings baseline recipe.",
    )
    p.add_argument(
        "--lanes",
        type=int,
        default=5,
        help="1v1 lanes per server. Multilane requires a matching settings JSON.",
    )
    p.add_argument(
        "--mirror-both-sides",
        action="store_true",
        default=True,
        help="Each physical 1v1 yields TWO gym envs (Blue + Purple). Doubles "
        "training data per wall-second.",
    )
    p.add_argument(
        "--no-mirror-both-sides",
        action="store_false",
        dest="mirror_both_sides",
        help="Disable mirror (single env per lane, opp_policy drives the other side).",
    )
    p.add_argument("--total-steps", type=int, default=200_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="PPO minibatch size. Default auto-scales with n_envs to "
        "preserve ~80 batches per rollout. Override at your own risk.",
    )
    p.add_argument("--frame-skip", type=int, default=8)
    p.add_argument("--max-episode-steps", type=int, default=80)
    p.add_argument(
        "--checkpoint-count",
        type=int,
        default=10,
        help="Number of checkpoints to save across the run.",
    )
    p.add_argument(
        "--opp-snapshot-prob",
        type=float,
        default=0.0,
        help="Per-episode probability of sampling a past checkpoint as opp_policy "
        "instead of the live policy (OAI-Five §N). 0.0 = legacy 100%% live mirror. "
        "0.2 matches OAI-Five.",
    )
    p.add_argument("--opp-snapshot-keep", type=int, default=8)
    p.add_argument(
        "--reset-jitter-hp",
        type=float,
        default=0.0,
        help="Per-cid HP jitter (OAI-Five §O.2). Breaks symmetric-Nash equilibrium.",
    )
    p.add_argument("--reset-jitter-mp", type=float, default=0.0)
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a saved checkpoint zip. --total-steps then counts "
        "ADDITIONAL steps to run, not a global cap.",
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Where to write model.zip, TB events, monitor CSVs. Default: "
        "runs/ppo_lol_<tag>.",
    )
    p.add_argument("--base-game-port", type=int, default=5119)
    p.add_argument("--base-rl-port", type=int, default=5120)
    p.add_argument("--port-ready-timeout", type=float, default=120.0)
    p.add_argument(
        "--pin-trainer",
        action="store_true",
        help="Pin the trainer process to the trailing cores (those not "
        "reserved for spawned servers). Requires riftgym[perf] (psutil).",
    )
    # Deployment shape (compose by default; bare docker as fallback).
    p.add_argument(
        "--compose-file",
        type=Path,
        default=None,
        help="Compose YAML to bring servers up via `docker compose`. "
        "Default behavior bares `docker run` via ContainerRunConfig.",
    )
    p.add_argument(
        "--compose-service",
        default="server",
        help="Compose service to bring up. Default `server`.",
    )
    p.add_argument(
        "--settings-json",
        type=Path,
        default=None,
        help="Settings JSON to mount into the server container (compose path). "
        "For multilane training, use riftgym.run_configs.default_multilane5_settings().",
    )
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--tag", default=DEFAULT_TAG)
    p.add_argument(
        "--pull",
        choices=("missing", "always", "never"),
        default="missing",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p


def _build_run_config(args: argparse.Namespace) -> ContainerRunConfig | ComposeRunConfig:
    if args.compose_file is not None:
        return ComposeRunConfig(
            compose_file=args.compose_file,
            service=args.compose_service,
            pull_policy=args.pull,
            image_override=(args.image if args.image != DEFAULT_IMAGE else None),
            tag_override=(args.tag if args.tag != DEFAULT_TAG else None),
            settings_json=args.settings_json,
        )
    if args.settings_json is not None:
        print(
            "warning: --settings-json is only supported with --compose-file; "
            "ignoring under the bare `docker run` path.",
            file=sys.stderr,
        )
    return ContainerRunConfig(
        image=args.image,
        tag=args.tag,
        pull_policy=args.pull,
        entrypoint=("/usr/bin/tini", "--", "/app/entrypoints/server.sh"),
    )


def _default_log_dir(args: argparse.Namespace) -> Path:
    n_envs = args.auto_launch * args.lanes * (2 if args.mirror_both_sides else 1)
    if args.lanes == 1:
        tag = f"par{args.auto_launch}"
    else:
        tag = f"par{args.auto_launch}x{args.lanes}lane"
        if args.mirror_both_sides:
            tag += "_2sides"
    if n_envs == 1:
        return Path("runs/ppo_lol_mirror_v0")
    return Path(f"runs/ppo_lol_{tag}_v0")


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    )

    # Imported here so ``riftgym-train --help`` works without sb3
    # installed. ``riftgym.sb3.trainer`` doesn't touch sb3 at import
    # time — only inside train_ppo() — so import is cheap.
    try:
        from riftgym.sb3.trainer import TrainConfig, train_ppo
    except ImportError:
        print(
            "riftgym-train requires the 'sb3' extra. Install with:\n"
            "    pip install 'riftgym[sb3]'",
            file=sys.stderr,
        )
        return 1

    rc = _build_run_config(args)
    log_dir = args.log_dir or _default_log_dir(args)

    cfg = TrainConfig(
        run_config=rc,
        n_servers=args.auto_launch,
        lanes=args.lanes,
        mirror_both_sides=args.mirror_both_sides,
        base_game_port=args.base_game_port,
        base_rl_port=args.base_rl_port,
        total_timesteps=args.total_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        frame_skip=args.frame_skip,
        max_episode_steps=args.max_episode_steps,
        opp_snapshot_prob=args.opp_snapshot_prob,
        opp_snapshot_keep=args.opp_snapshot_keep,
        reset_jitter_hp=args.reset_jitter_hp,
        reset_jitter_mp=args.reset_jitter_mp,
        log_dir=log_dir,
        checkpoint_count=args.checkpoint_count,
        resume_from=args.resume,
        port_ready_timeout_s=args.port_ready_timeout,
        pin_trainer=args.pin_trainer,
    )

    def _on_ready(idx: int, game_port: int, rl_port: int) -> None:
        print(f"server {idx} ready (game UDP={game_port} rl TCP={rl_port})", flush=True)

    model_path = train_ppo(cfg, on_server_ready=_on_ready)
    print(f"\nSaved: {model_path}", flush=True)
    print(f"View logs: tensorboard --logdir {cfg.log_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
