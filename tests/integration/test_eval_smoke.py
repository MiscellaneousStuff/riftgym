"""End-to-end integration smoke for issue #1.

Runs ``riftgym-eval`` against a real containerized brokenwings server.
Skipped unless ``RIFTGYM_INTEGRATION=1`` is set and a checkpoint is
available at ``checkpoints/agent.zip`` (or ``RIFTGYM_TEST_CHECKPOINT``).

The full milestone target is 50 episodes against the published image;
this test runs 5 against a locally-built ``brokenwings:latest`` so CI on
a self-hosted runner can validate the bridge contract without burning
cycles.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def checkpoint() -> Path:
    if os.environ.get("RIFTGYM_INTEGRATION") != "1":
        pytest.skip("set RIFTGYM_INTEGRATION=1 to run integration tests")
    explicit = os.environ.get("RIFTGYM_TEST_CHECKPOINT")
    cand = Path(explicit) if explicit else Path("checkpoints/agent.zip")
    if not cand.exists():
        pytest.skip(f"checkpoint not found at {cand}; set RIFTGYM_TEST_CHECKPOINT")
    return cand


def test_riftgym_eval_completes_5_episodes(checkpoint: Path) -> None:
    from riftgym.bin.eval import main

    image = os.environ.get("RIFTGYM_TEST_IMAGE", "brokenwings")
    tag = os.environ.get("RIFTGYM_TEST_TAG", "latest")
    pull = os.environ.get("RIFTGYM_TEST_PULL", "never")

    rc = main(
        [
            str(checkpoint),
            "--vs-engine-bot",
            "--episodes",
            "5",
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
    assert rc == 0, f"riftgym-eval exited with {rc}"
