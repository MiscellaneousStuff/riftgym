"""Layer-2 integration smoke for ComposeRunConfig.

Validates the docker-compose plumbing — `up -d`, `ps -q`, `down`,
container-id lookup, project teardown — using a tiny ``alpine`` image
with a ``sleep`` command. No brokenwings image required.

If this passes but the brokenwings smoke fails, the bug is brokenwings-
or bridge-specific (image boot, RL bridge port, settings JSON). If this
fails, the bug is in our compose plumbing.

Gated on ``RIFTGYM_INTEGRATION=1`` (and Docker being reachable) since it
shells out to a real ``docker compose`` daemon.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from riftgym.run_configs import ComposeRunConfig

pytestmark = pytest.mark.integration

STUB_COMPOSE = Path(__file__).parent / "_stub_compose.yaml"


def _docker_available() -> bool:
    return shutil.which("docker") is not None


@pytest.fixture(autouse=True)
def _gate() -> None:
    if os.environ.get("RIFTGYM_INTEGRATION") != "1":
        pytest.skip("set RIFTGYM_INTEGRATION=1 to run integration tests")
    if not _docker_available():
        pytest.skip("docker CLI not on PATH")


def _list_compose_projects() -> set[str]:
    result = subprocess.run(
        ["docker", "compose", "ls", "--all", "--format", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return set()
    try:
        rows = json.loads(result.stdout)
    except json.JSONDecodeError:
        return set()
    return {r["Name"] for r in rows if "Name" in r}


def test_compose_up_and_down_lifecycle() -> None:
    """ComposeRunConfig brings the project up, returns a live handle,
    and tears the project back down on terminate()."""
    rc = ComposeRunConfig(
        compose_file=STUB_COMPOSE,
        project_name=f"riftgym-test-{int(time.time())}",
    )

    projects_before = _list_compose_projects()
    handle = rc.start(game_port=15119, rl_port=15120)
    try:
        assert handle.container_id, "compose ps -q returned no container"
        assert handle.is_alive(), "container should be running after up"
        assert handle.project not in projects_before
        assert handle.project in _list_compose_projects()
    finally:
        handle.terminate()

    assert not handle.is_alive(), "container still alive after terminate"
    assert handle.project not in _list_compose_projects(), \
        "compose project leaked after terminate"


def test_terminate_via_context_manager() -> None:
    """The handle's __exit__ tears the project down — same path the
    ServerLauncher's `with` block exercises in production."""
    rc = ComposeRunConfig(
        compose_file=STUB_COMPOSE,
        project_name=f"riftgym-test-ctx-{int(time.time())}",
    )

    with rc.start(game_port=15121, rl_port=15122) as handle:
        project = handle.project
        assert handle.is_alive()

    assert project not in _list_compose_projects()


def test_two_concurrent_projects_get_disjoint_names() -> None:
    """n>1 topology check: two simultaneous starts must produce two
    distinct compose projects on disjoint ports."""
    rc = ComposeRunConfig(compose_file=STUB_COMPOSE)

    h1 = rc.start(game_port=15123, rl_port=15124)
    try:
        h2 = rc.start(game_port=15125, rl_port=15126)
        try:
            assert h1.project != h2.project
            assert h1.container_id != h2.container_id
            assert h1.is_alive() and h2.is_alive()
        finally:
            h2.terminate()
    finally:
        h1.terminate()

    projects = _list_compose_projects()
    assert h1.project not in projects
    assert h2.project not in projects
