"""Unit tests for ``riftgym.run_configs.compose.ComposeRunConfig``.

Mocks ``subprocess.run`` so the assertions are about argv shape + env-var
injection rather than against a real docker daemon. Integration tests
that exercise a live compose stack live in
``tests/integration/test_compose_eval_smoke.py`` (gated on
``RIFTGYM_INTEGRATION=1``).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from riftgym.run_configs import ComposeRunConfig, RunConfig
from riftgym.run_configs.compose import ComposeHandle


@pytest.fixture
def compose_file(tmp_path: Path) -> Path:
    p = tmp_path / "compose.yaml"
    p.write_text("services:\n  server: {image: scratch}\n")
    return p


def test_subclass_of_run_config() -> None:
    assert issubclass(ComposeRunConfig, RunConfig)


def test_priority_higher_than_container_run_config() -> None:
    from riftgym.run_configs import ContainerRunConfig

    assert ComposeRunConfig.priority() > (ContainerRunConfig.priority() or 0)


def test_all_subclasses_includes_compose() -> None:
    assert ComposeRunConfig in RunConfig.all_subclasses()


def test_constructable_with_defaults(compose_file: Path) -> None:
    rc = ComposeRunConfig(compose_file=compose_file)
    assert rc.service == "server"
    assert rc.project_name is None
    assert rc.pull_policy == "missing"
    assert rc.image_override is None
    assert rc.settings_json is None


def test_start_invokes_compose_up_with_port_env_vars(compose_file: Path) -> None:
    """`docker compose up -d server` is invoked with the right argv +
    RIFTGYM_*_PORT env vars set so the YAML's ``${...}`` substitutions
    produce disjoint host ports for multi-server topologies."""
    rc = ComposeRunConfig(compose_file=compose_file, project_name="riftgym-test")

    captured: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured.setdefault("cmds", []).append((cmd, kwargs.get("env", {})))
        if cmd[1:3] == ["compose", "-f"] and "up" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if cmd[1:3] == ["compose", "-f"] and "ps" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="abc123\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("riftgym.run_configs.compose.subprocess.run", side_effect=fake_run):
        with patch("riftgym.run_configs.compose.atexit.register"):
            handle = rc.start(game_port=5121, rl_port=5122)

    assert isinstance(handle, ComposeHandle)
    assert handle.container_id == "abc123"
    assert handle.project == "riftgym-test"
    assert handle.game_port == 5121
    assert handle.rl_port == 5122

    up_cmd, up_env = captured["cmds"][0]
    assert up_cmd[1:3] == ["compose", "-f"]
    assert "-p" in up_cmd and up_cmd[up_cmd.index("-p") + 1] == "riftgym-test"
    assert up_cmd[-3:] == ["up", "-d", "server"]
    assert up_env["RIFTGYM_GAME_PORT"] == "5121"
    assert up_env["RIFTGYM_RL_PORT"] == "5122"


def test_start_auto_generates_unique_project_name(compose_file: Path) -> None:
    """``project_name=None`` must produce a unique name per call so
    multi-server (n>1) doesn't collide on the default project name."""
    rc = ComposeRunConfig(compose_file=compose_file)

    seen_projects: list[str] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "-p" in cmd:
            seen_projects.append(cmd[cmd.index("-p") + 1])
        if "ps" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="cid\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("riftgym.run_configs.compose.subprocess.run", side_effect=fake_run):
        with patch("riftgym.run_configs.compose.atexit.register"):
            h1 = rc.start(game_port=5119, rl_port=5120)
            h2 = rc.start(game_port=5121, rl_port=5122)

    assert h1.project != h2.project
    assert h1.project.startswith("riftgym-srv-")
    assert h2.project.startswith("riftgym-srv-")
    # Same project name appears twice per start (`up` + `ps`); two starts → 4 entries.
    assert len(set(seen_projects)) == 2


def test_settings_json_sets_env_vars(compose_file: Path, tmp_path: Path) -> None:
    settings = tmp_path / "GameInfo-multilane5.json"
    settings.write_text("{}")
    rc = ComposeRunConfig(compose_file=compose_file, settings_json=settings)

    captured_env: dict[str, str] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "up" in cmd:
            captured_env.update(kwargs.get("env", {}))
        if "ps" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="cid\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("riftgym.run_configs.compose.subprocess.run", side_effect=fake_run):
        with patch("riftgym.run_configs.compose.atexit.register"):
            rc.start(game_port=5119, rl_port=5120)

    assert captured_env["RIFTGYM_SETTINGS_JSON"] == str(settings.resolve())
    assert captured_env["BROKENWINGS_GAME_INFO"] == "Settings/GameInfo-override.json"


def test_image_and_tag_overrides_set_env_vars(compose_file: Path) -> None:
    rc = ComposeRunConfig(
        compose_file=compose_file,
        image_override="brokenwings",
        tag_override="dev",
    )

    captured_env: dict[str, str] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "up" in cmd:
            captured_env.update(kwargs.get("env", {}))
        if "ps" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="cid\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("riftgym.run_configs.compose.subprocess.run", side_effect=fake_run):
        with patch("riftgym.run_configs.compose.atexit.register"):
            rc.start(game_port=5119, rl_port=5120)

    assert captured_env["BROKENWINGS_IMAGE"] == "brokenwings"
    assert captured_env["BROKENWINGS_TAG"] == "dev"


def test_pull_policy_passed_when_not_missing(compose_file: Path) -> None:
    rc = ComposeRunConfig(compose_file=compose_file, pull_policy="always")

    captured_cmds: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured_cmds.append(cmd)
        if "ps" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="cid\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("riftgym.run_configs.compose.subprocess.run", side_effect=fake_run):
        with patch("riftgym.run_configs.compose.atexit.register"):
            rc.start(game_port=5119, rl_port=5120)

    up_cmd = captured_cmds[0]
    assert "--pull" in up_cmd
    assert up_cmd[up_cmd.index("--pull") + 1] == "always"


def test_missing_compose_file_raises(tmp_path: Path) -> None:
    rc = ComposeRunConfig(compose_file=tmp_path / "nope.yaml")
    with pytest.raises(FileNotFoundError, match="compose file not found"):
        rc.start(game_port=5119, rl_port=5120)


def test_missing_settings_json_raises(compose_file: Path, tmp_path: Path) -> None:
    rc = ComposeRunConfig(compose_file=compose_file, settings_json=tmp_path / "nope.json")
    with pytest.raises(FileNotFoundError, match="settings JSON not found"):
        rc.start(game_port=5119, rl_port=5120)


def test_terminate_invokes_compose_down(compose_file: Path) -> None:
    handle = ComposeHandle(
        project="riftgym-test",
        compose_file=compose_file,
        service="server",
        container_id="abc123",
        game_port=5119,
        rl_port=5120,
        stop_timeout_s=3.0,
    )

    captured: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("riftgym.run_configs.compose.subprocess.run", side_effect=fake_run):
        handle.terminate()

    assert len(captured) == 1
    cmd = captured[0]
    assert cmd[1:3] == ["compose", "-f"]
    assert "-p" in cmd and cmd[cmd.index("-p") + 1] == "riftgym-test"
    assert "down" in cmd
    assert cmd[cmd.index("-t") + 1] == "3"


def test_terminate_is_idempotent(compose_file: Path) -> None:
    handle = ComposeHandle(
        project="riftgym-test",
        compose_file=compose_file,
        service="server",
        container_id="abc123",
        game_port=5119,
        rl_port=5120,
    )

    call_count = 0

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal call_count
        call_count += 1
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("riftgym.run_configs.compose.subprocess.run", side_effect=fake_run):
        handle.terminate()
        handle.terminate()
        handle.terminate()

    assert call_count == 1


def test_start_cleans_up_when_ps_returns_no_container(compose_file: Path) -> None:
    """If `compose up` succeeds but `compose ps -q` returns nothing
    (e.g., service crashed during boot), tear down the project so we
    don't leak it."""
    rc = ComposeRunConfig(compose_file=compose_file)

    captured: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured.append(cmd)
        if "ps" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="boom")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("riftgym.run_configs.compose.subprocess.run", side_effect=fake_run):
        with patch("riftgym.run_configs.compose.atexit.register"):
            with pytest.raises(RuntimeError, match="no running container"):
                rc.start(game_port=5119, rl_port=5120)

    # up, ps, down (cleanup)
    assert any("down" in cmd for cmd in captured)
