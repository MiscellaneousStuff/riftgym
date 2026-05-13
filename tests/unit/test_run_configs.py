"""Unit tests for ``riftgym.run_configs``."""

from __future__ import annotations

import json

from riftgym.run_configs import (
    ContainerRunConfig,
    RunConfig,
    default_multilane5_settings,
)


def test_container_run_config_is_subclass_of_run_config() -> None:
    assert issubclass(ContainerRunConfig, RunConfig)


def test_container_run_config_priority_is_set() -> None:
    assert ContainerRunConfig.priority() is not None


def test_all_subclasses_includes_container() -> None:
    subclasses = RunConfig.all_subclasses()
    assert ContainerRunConfig in subclasses


def test_container_run_config_constructable() -> None:
    rc = ContainerRunConfig(image="ghcr.io/miscellaneousstuff/brokenwings", tag="latest")
    assert rc.image == "ghcr.io/miscellaneousstuff/brokenwings"
    assert rc.tag == "latest"
    assert rc.host_bind == "127.0.0.1"
    assert rc.pull_policy == "missing"


def test_default_multilane5_settings_path_exists() -> None:
    """The bundled multilane5 JSON must be present on disk after install
    (regular or editable). If this fails after a fresh `pip install`,
    the `tool.hatch.build.targets.wheel.force-include` block in
    pyproject.toml has regressed."""
    path = default_multilane5_settings()
    assert path.exists(), f"bundled settings JSON missing: {path}"
    assert path.is_file()
    assert path.suffix == ".json"
    assert path.name == "GameInfo-multilane5.json"


def test_default_multilane5_settings_has_training_friendly_overrides() -> None:
    """Sanity-check the training overrides that protect long runs from
    crashes / mask logic regressions. Each of these has a specific
    reason — see the helper's docstring for the brokenwings issue refs."""
    data = json.loads(default_multilane5_settings().read_text())
    gi = data["gameInfo"]
    assert gi["MINION_SPAWNS_ENABLED"] is False  # brokenwings #5 workaround
    assert gi["COOLDOWNS_ENABLED"] is True  # action-mask logic needs this
    assert gi["MANACOSTS_ENABLED"] is True
    assert gi["RL_ENABLED"] is True
    assert gi["RL_HZ"] == 30
    assert gi["HEADLESS"] is True
    assert gi["KEEP_ALIVE_WHEN_EMPTY"] is True
    assert gi["RL_BIND_ADDRESS"] == "Any"  # container host-side port mapping re-clamps to loopback


def test_default_multilane5_settings_has_5_lane_layout() -> None:
    """5 lanes x 2 sides = 10 champion slots (Blue0..4, Purple0..4)."""
    players = json.loads(default_multilane5_settings().read_text())["players"]
    assert len(players) == 10
    blue = [p for p in players if p["team"] == "BLUE"]
    purple = [p for p in players if p["team"] == "PURPLE"]
    assert len(blue) == 5
    assert len(purple) == 5
