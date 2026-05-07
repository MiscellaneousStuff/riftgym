"""Shared docker-CLI helpers used by ContainerRunConfig and ComposeRunConfig.

Kept module-private (``_docker``) so it doesn't clutter the public
``riftgym.run_configs`` surface.
"""

from __future__ import annotations

import shutil


def docker_bin() -> str:
    """Return the path to the ``docker`` CLI, or raise if missing.

    Both ``docker run`` (ContainerRunConfig) and ``docker compose ...``
    (ComposeRunConfig) reach the daemon via this single CLI binary.
    """
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError(
            "`docker` CLI not found on PATH. RunConfigs that talk to a "
            "container runtime require Docker Desktop or a docker-compatible "
            "runtime. Install Docker, or use a different RunConfig (e.g. a "
            "future LocalBuildRunConfig for running from a local .NET build)."
        )
    return docker
