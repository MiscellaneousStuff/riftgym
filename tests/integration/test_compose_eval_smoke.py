"""Integration smoke for the compose deployment path.

Mirrors ``test_eval_smoke.py`` but routes through ``ComposeRunConfig``
and the repo-root ``compose.yaml``. Skipped unless
``RIFTGYM_INTEGRATION=1`` and a checkpoint is available.

Verifies, end-to-end, that:
  1. ``riftgym-eval --compose-file ./compose.yaml`` runs to completion
  2. The compose project is torn down after exit (no leaked containers
     or compose projects).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def checkpoint() -> Path:
    if os.environ.get("RIFTGYM_INTEGRATION") != "1":
        pytest.skip("set RIFTGYM_INTEGRATION=1 to run integration tests")
    explicit = os.environ.get("RIFTGYM_TEST_CHECKPOINT")
    cand = Path(explicit) if explicit else REPO_ROOT / "checkpoints" / "agent.zip"
    if not cand.exists():
        pytest.skip(f"checkpoint not found at {cand}; set RIFTGYM_TEST_CHECKPOINT")
    return cand


@pytest.fixture(scope="module")
def compose_file() -> Path:
    cf = REPO_ROOT / "compose.yaml"
    if not cf.exists():
        pytest.skip(f"compose file not found at {cf}")
    return cf


def _list_riftgym_compose_projects() -> list[str]:
    """Return any compose projects whose name starts with 'riftgym-srv-'.

    Used to verify nothing leaks after the eval exits. Unlike `docker
    ps`, `docker compose ls` is the canonical way to inspect compose
    project state and survives stopped (but not removed) containers.
    """
    result = subprocess.run(
        ["docker", "compose", "ls", "--all", "--format", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    import json as _json

    try:
        rows = _json.loads(result.stdout) if result.stdout.strip() else []
    except _json.JSONDecodeError:
        return []
    return [r["Name"] for r in rows if r.get("Name", "").startswith("riftgym-srv-")]


def test_riftgym_eval_compose_completes_1_episode(
    checkpoint: Path, compose_file: Path
) -> None:
    from riftgym.bin.eval import main

    image = os.environ.get("RIFTGYM_TEST_IMAGE", "brokenwings")
    tag = os.environ.get("RIFTGYM_TEST_TAG", "latest")
    pull = os.environ.get("RIFTGYM_TEST_PULL", "never")

    leaked_before = _list_riftgym_compose_projects()
    rc = main(
        [
            str(checkpoint),
            "--vs-engine-bot",
            "--episodes",
            "1",
            "--compose-file",
            str(compose_file),
            "--image",
            image,
            "--tag",
            tag,
            "--pull",
            pull,
            "--port-ready-timeout",
            "180",
        ]
    )
    assert rc == 0, f"riftgym-eval --compose-file exited with {rc}"

    leaked_after = _list_riftgym_compose_projects()
    new_leaks = set(leaked_after) - set(leaked_before)
    assert not new_leaks, f"compose projects leaked: {new_leaks}"
