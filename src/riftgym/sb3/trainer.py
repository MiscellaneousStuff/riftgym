"""PPO trainer over LoLGymEnv with mirror self-play.

Library entry point: :func:`train_ppo` takes a :class:`TrainConfig` and
runs to completion, saving the final model and intermediate checkpoints
to ``log_dir``. The CLI in :mod:`riftgym.bin.ppo_train` is a thin
argparse wrapper over this — keep the CLI minimal and the library
testable.

Topology built per :func:`train_ppo`:

1. :class:`riftgym.launcher.ServerLauncher` spawns ``cfg.n_servers``
   brokenwings servers via the configured :class:`RunConfig` (compose
   by default, bare ``docker run`` as fallback).
2. :func:`riftgym.env.multilane.make_multilane_envs` per server →
   ``n_servers * lanes * (2 if mirror_both_sides else 1)`` gym envs.
3. :class:`riftgym.sb3.thread_vec_env.ThreadVecEnv` wraps them.
4. :class:`MaskablePPO` trains over the vec env with action masks.
5. Mirror self-play via :func:`riftgym.sb3.policy.make_mirror_opp` — the
   live model is the opp policy for every env. Optionally diversified
   with a :class:`SnapshotPool` of past checkpoints (OAI-Five §N).
6. Per-episode HP/MP jitter (OAI-Five §O.2) via env attributes — breaks
   the symmetric-Nash equilibrium mirror self-play converges to.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from riftgym.env.lol_gym import LoLGymEnv
from riftgym.env.multilane import make_multilane_envs
from riftgym.env.session import ServerSession
from riftgym.launcher import ServerLauncher
from riftgym.run_configs.lib import RunConfig
from riftgym.sb3.policy import make_mirror_opp
from riftgym.sb3.snapshot_pool import SnapshotCheckpointCallback, SnapshotPool
from riftgym.sb3.thread_vec_env import ThreadVecEnv

log = logging.getLogger(__name__)

DEFAULT_TOTAL_STEPS = 200_000
DEFAULT_FRAME_SKIP = 8
DEFAULT_MAX_EP_STEPS = 80


@dataclass(slots=True)
class TrainConfig:
    """All knobs for one PPO training run.

    Defaults match the brokenwings 40k baseline recipe — the same
    hyperparams that produced the ~77% WR vs engine BT checkpoint.
    """

    # Server fleet shape
    run_config: RunConfig
    n_servers: int = 4
    lanes: int = 5
    mirror_both_sides: bool = True
    base_game_port: int = 5119
    base_rl_port: int = 5120

    # Training schedule
    total_timesteps: int = DEFAULT_TOTAL_STEPS
    learning_rate: float = 3e-4
    batch_size: int | None = None  # None = auto-scale with n_envs
    n_steps_per_env: int | None = None  # None = 128 (or 64 if n_envs==1)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    frame_skip: int = DEFAULT_FRAME_SKIP
    max_episode_steps: int = DEFAULT_MAX_EP_STEPS

    # Opponent diversity (OAI-Five Appendix N)
    opp_snapshot_prob: float = 0.0
    opp_snapshot_keep: int = 8

    # Initial-state randomization (OAI-Five Appendix O.2)
    reset_jitter_hp: float = 0.0
    reset_jitter_mp: float = 0.0

    # Output
    log_dir: Path = field(default_factory=lambda: Path("runs/exp"))
    checkpoint_count: int = 10
    resume_from: Path | None = None
    port_ready_timeout_s: float = 120.0

    # Affinity (best-effort)
    pin_trainer: bool = False

    def __post_init__(self) -> None:
        self.log_dir = Path(self.log_dir)


def _n_envs(cfg: TrainConfig) -> int:
    return cfg.n_servers * cfg.lanes * (2 if cfg.mirror_both_sides else 1)


def _resolve_batch_size(cfg: TrainConfig, n_envs: int, n_steps_per_env: int) -> int:
    """Auto-scale batch size to preserve ~80 batches per rollout — the
    40-env recipe brokenwings's 152k local baseline came from. Without
    this, n_envs >> 40 produces hundreds of correlated batches per
    rollout at the same per-sample LR, hurting sample efficiency.
    CLI override wins.
    """
    if cfg.batch_size is not None:
        return cfg.batch_size
    if n_envs == 1:
        return 8
    rollout = n_steps_per_env * n_envs
    return max(64, rollout // 80)


def train_ppo(
    cfg: TrainConfig,
    *,
    on_server_ready: Callable[[int, int, int], None] | None = None,
) -> Path:
    """Run one PPO training to completion. Returns the path to the
    final saved model (``log_dir/model.zip``).

    Heavy imports (MaskablePPO, Monitor, CheckpointCallback) happen
    inside this function so importing :mod:`riftgym.sb3.trainer`
    doesn't drag sb3 in until the user actually trains.
    """
    from sb3_contrib import MaskablePPO  # pyright: ignore[reportMissingImports]
    from stable_baselines3.common.callbacks import (
        CallbackList,
        CheckpointCallback,
    )
    from stable_baselines3.common.monitor import Monitor

    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    if cfg.pin_trainer:
        from riftgym.launcher.affinity import (
            pin_current_process_to,
            plan_trainer_cores,
        )

        cores = plan_trainer_cores(cfg.n_servers)
        if cores is not None:
            pin_current_process_to(cores)

    launcher = ServerLauncher(
        n=cfg.n_servers,
        run_config=cfg.run_config,
        base_game_port=cfg.base_game_port,
        base_rl_port=cfg.base_rl_port,
        port_ready_timeout_s=cfg.port_ready_timeout_s,
    )

    sessions: list[ServerSession] = []
    try:
        launcher.start()
        if on_server_ready is not None:
            for i, (gp, rp) in enumerate(
                zip(launcher.game_ports, launcher.rl_ports, strict=True)
            ):
                on_server_ready(i, gp, rp)

        # Build envs: one ServerSession per spawned server, multiple
        # LoLGymEnvs per session (one per lane x side).
        wrapped_envs: list[Any] = []
        for srv_idx, rl_port in enumerate(launcher.rl_ports):
            session, lane_envs = make_multilane_envs(
                port=rl_port,
                n_lanes=cfg.lanes,
                max_episode_steps=cfg.max_episode_steps,
                frame_skip=cfg.frame_skip,
                mirror_both_sides=cfg.mirror_both_sides,
            )
            sessions.append(session)
            for lane_idx, env in enumerate(lane_envs):
                wrapped_envs.append((env, srv_idx, lane_idx))

        env_fns = [
            (
                lambda e=env, s=srv_idx, le=lane_idx: Monitor(
                    e, filename=str(cfg.log_dir / f"monitor_s{s}_l{le}")
                )
            )
            for env, srv_idx, lane_idx in wrapped_envs
        ]
        n_envs = len(env_fns)
        assert n_envs == _n_envs(cfg)

        n_steps_per_env = (
            cfg.n_steps_per_env
            if cfg.n_steps_per_env is not None
            else (64 if n_envs == 1 else 128)
        )
        batch_size = _resolve_batch_size(cfg, n_envs, n_steps_per_env)

        vec_env = ThreadVecEnv(env_fns)
        log.info(
            "hparams n_envs=%d n_steps=%d rollout=%d batch_size=%d lr=%g",
            n_envs,
            n_steps_per_env,
            n_steps_per_env * n_envs,
            batch_size,
            cfg.learning_rate,
        )

        if cfg.resume_from is not None:
            log.info("resuming from %s", cfg.resume_from)
            model = MaskablePPO.load(
                str(cfg.resume_from),
                env=vec_env,
                device="cpu",
                tensorboard_log=str(cfg.log_dir),
            )
            log.info("resumed at num_timesteps=%d", model.num_timesteps)
        else:
            model = MaskablePPO(
                "MlpPolicy",
                vec_env,
                device="cpu",
                n_steps=n_steps_per_env,
                batch_size=batch_size,
                learning_rate=cfg.learning_rate,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
                clip_range=cfg.clip_range,
                ent_coef=cfg.ent_coef,
                verbose=1,
                tensorboard_log=str(cfg.log_dir),
            )

        # Mirror self-play: every env's opp_policy is the live model.
        # ThreadVecEnv keeps envs in-process so the closure capturing
        # ``model`` reaches all of them.
        mirror = make_mirror_opp(model, deterministic=False)
        opp_pool: SnapshotPool | None = None
        if cfg.opp_snapshot_prob > 0.0:
            opp_pool = SnapshotPool(capacity=cfg.opp_snapshot_keep)

        for env_wrapped in vec_env.envs:
            base: LoLGymEnv = env_wrapped.unwrapped
            base.opp_policy = mirror
            # snapshot_pool/snapshot_prob aren't formal attrs on
            # LoLGymEnv yet — they're consumed in reset() when the env
            # has them set (Phase 4.5 wiring TBD). For now we set them
            # unconditionally and the env tolerates extras.
            base.snapshot_pool = opp_pool  # type: ignore[attr-defined]
            base.snapshot_prob = cfg.opp_snapshot_prob  # type: ignore[attr-defined]
            base.reset_jitter_hp = cfg.reset_jitter_hp
            base.reset_jitter_mp = cfg.reset_jitter_mp

        # Checkpoint cadence: ``checkpoint_count`` snapshots across the
        # run. save_freq is per-env steps, so divide total by n_envs too.
        save_freq_per_env = max(
            50, cfg.total_timesteps // (cfg.checkpoint_count * n_envs)
        )
        ckpt_kwargs: dict[str, Any] = {
            "save_freq": save_freq_per_env,
            "save_path": str(cfg.log_dir),
            "name_prefix": "ppo_lol",
            "save_replay_buffer": False,
            "save_vecnormalize": False,
        }
        ckpt_cb: Any
        if opp_pool is not None:
            ckpt_cb = SnapshotCheckpointCallback(pool=opp_pool, **ckpt_kwargs)
        else:
            ckpt_cb = CheckpointCallback(**ckpt_kwargs)

        callbacks: list[Any] = [ckpt_cb]
        cb = callbacks[0] if len(callbacks) == 1 else CallbackList(callbacks)

        model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=cb,
            progress_bar=True,
            reset_num_timesteps=(cfg.resume_from is None),
        )

        model_path = cfg.log_dir / "model.zip"
        model.save(str(model_path.with_suffix("")))
        vec_env.close()
        log.info("saved %s", model_path)
        return model_path
    finally:
        for s in sessions:
            try:
                s.close()
            except Exception:
                log.exception("error closing session during teardown")
        launcher.close()


__all__ = [
    "DEFAULT_FRAME_SKIP",
    "DEFAULT_MAX_EP_STEPS",
    "DEFAULT_TOTAL_STEPS",
    "TrainConfig",
    "train_ppo",
]
