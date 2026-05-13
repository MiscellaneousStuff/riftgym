"""Unit tests for the ``riftgym-train`` argparse wrapper.

The CLI imports sb3 lazily inside main(), so the parser itself can be
constructed (and tested) without sb3 installed. The full train_ppo
loop tests live behind RIFTGYM_SB3_TESTS=1.
"""

from __future__ import annotations

from pathlib import Path

from riftgym.bin.ppo_train import (
    DEFAULT_IMAGE,
    DEFAULT_TAG,
    _build_arg_parser,
    _build_run_config,
    _default_log_dir,
)
from riftgym.run_configs import ComposeRunConfig, ContainerRunConfig


def test_parser_has_essential_training_flags() -> None:
    """Hand-rolled smoke check that the CLI exposes the flag set the
    trainer doc describes. Trips if someone renames a flag without
    updating docs / muscle memory."""
    p = _build_arg_parser()
    help_text = p.format_help()
    for flag in (
        "--auto-launch",
        "--lanes",
        "--mirror-both-sides",
        "--total-steps",
        "--learning-rate",
        "--compose-file",
        "--settings-json",
        "--opp-snapshot-prob",
        "--reset-jitter-hp",
        "--resume",
        "--pin-trainer",
    ):
        assert flag in help_text, f"missing flag in CLI help: {flag}"


def test_defaults_match_baseline_recipe() -> None:
    """Defaults match brokenwings's 40k baseline recipe."""
    args = _build_arg_parser().parse_args([])
    assert args.auto_launch == 4
    assert args.lanes == 5
    assert args.mirror_both_sides is True
    assert args.total_steps == 200_000
    assert args.learning_rate == 3e-4
    assert args.frame_skip == 8
    assert args.max_episode_steps == 80
    assert args.opp_snapshot_prob == 0.0
    assert args.reset_jitter_hp == 0.0


def test_no_mirror_both_sides_disables_flag() -> None:
    args = _build_arg_parser().parse_args(["--no-mirror-both-sides"])
    assert args.mirror_both_sides is False


def test_build_run_config_picks_compose_when_file_given(tmp_path: Path) -> None:
    """Compose path: --compose-file FILE → ComposeRunConfig. Image and
    tag overrides flow through env vars consumed by the YAML."""
    f = tmp_path / "compose.yaml"
    f.write_text("services: {server: {image: brokenwings}}")
    args = _build_arg_parser().parse_args(["--compose-file", str(f)])
    rc = _build_run_config(args)
    assert isinstance(rc, ComposeRunConfig)
    assert rc.compose_file == f


def test_build_run_config_falls_back_to_container_run_config() -> None:
    """No --compose-file → bare `docker run` via ContainerRunConfig.
    Default image+tag preserved; entrypoint overridden to server.sh
    (image default is the human-vs-bot demo)."""
    args = _build_arg_parser().parse_args([])
    rc = _build_run_config(args)
    assert isinstance(rc, ContainerRunConfig)
    assert rc.image == DEFAULT_IMAGE
    assert rc.tag == DEFAULT_TAG


def test_default_log_dir_reflects_topology() -> None:
    """Default log dir name encodes (n_servers x lanes), with `_2sides`
    suffix when mirror_both_sides is on. Keeps multiple training runs
    distinguishable on disk without forcing --log-dir."""
    args = _build_arg_parser().parse_args(
        ["--auto-launch", "4", "--lanes", "5"]
    )
    assert _default_log_dir(args) == Path("runs/ppo_lol_par4x5lane_2sides_v0")

    args2 = _build_arg_parser().parse_args(
        ["--auto-launch", "2", "--lanes", "1", "--no-mirror-both-sides"]
    )
    assert _default_log_dir(args2) == Path("runs/ppo_lol_par2_v0")
