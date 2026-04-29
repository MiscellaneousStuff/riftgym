"""Unit tests for ``riftgym.run_configs``."""

from __future__ import annotations

from riftgym.run_configs import ContainerRunConfig, RunConfig


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
