"""``ContainerRunConfig`` — public-default RunConfig that runs the brokenwings
server inside a Docker container.

Wraps the ``docker`` CLI (no docker SDK dependency — riftgym should be
installable without pulling in the docker-py wheel). Pulls the image on
first use, runs detached, returns a :class:`ContainerHandle` that
``ServerLauncher`` can clean up on exit.

The brokenwings image is published at ``ghcr.io/miscellaneousstuff/brokenwings``.
For local development point at a locally-built image with
``ContainerRunConfig(image="brokenwings", tag="latest", pull_policy="never")``.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from riftgym.run_configs.lib import RunConfig, ServerHandle

log = logging.getLogger(__name__)

PullPolicy = Literal["missing", "always", "never"]


def _docker_bin() -> str:
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError(
            "`docker` CLI not found on PATH. ContainerRunConfig requires Docker "
            "Desktop or a docker-compatible runtime. Install Docker, or use a "
            "different RunConfig (e.g. BrokenwingsLocalBuildRunConfig for "
            "running from a local .NET build)."
        )
    return docker


class ContainerHandle:
    """Live ``docker run -d`` container. Tearing down stops + removes it."""

    def __init__(
        self,
        container_id: str,
        *,
        game_port: int,
        rl_port: int,
        stop_timeout_s: float = 5.0,
    ) -> None:
        self._id = container_id
        self._game_port = game_port
        self._rl_port = rl_port
        self._stop_timeout_s = stop_timeout_s
        self._terminated = False

    @property
    def container_id(self) -> str:
        return self._id

    @property
    def game_port(self) -> int:
        return self._game_port

    @property
    def rl_port(self) -> int:
        return self._rl_port

    def is_alive(self) -> bool:
        if self._terminated:
            return False
        result = subprocess.run(
            [_docker_bin(), "inspect", "-f", "{{.State.Running}}", self._id],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"

    def terminate(self) -> None:
        if self._terminated:
            return
        self._terminated = True
        # `--rm` on `docker run` removes the container after stop, so
        # `docker stop` is the only call we need. Suppress errors so a
        # double-terminate or already-dead container doesn't raise.
        with contextlib.suppress(subprocess.CalledProcessError, FileNotFoundError):
            subprocess.run(
                [
                    _docker_bin(),
                    "stop",
                    "-t",
                    str(int(self._stop_timeout_s)),
                    self._id,
                ],
                capture_output=True,
                check=False,
            )

    def wait(self, timeout: float | None = None) -> int | None:
        cmd = [_docker_bin(), "wait", self._id]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired:
            return None
        if result.returncode != 0:
            return None
        try:
            return int(result.stdout.strip())
        except ValueError:
            return None

    def logs(self) -> str:
        result = subprocess.run(
            [_docker_bin(), "logs", self._id],
            capture_output=True,
            text=True,
            check=False,
        )
        return (result.stdout or "") + (result.stderr or "")

    def __enter__(self) -> ContainerHandle:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.terminate()


@dataclass(slots=True)
class ContainerRunConfig(RunConfig):
    """Run the brokenwings server inside a Docker container.

    Args:
        image: image name without tag (e.g. ``ghcr.io/miscellaneousstuff/brokenwings``).
        tag: image tag. Default ``latest``; pin a digest for reproducibility.
        pull_policy: ``missing`` (pull only if absent), ``always`` (pull every
            time), ``never`` (fail if missing — useful for local dev images).
        host_bind: host-side bind address for the published ports. Defaults
            to ``127.0.0.1`` because the bridge is unauth'd; do not change
            until v0.2 of the bridge protocol ships shared-secret auth.
        container_game_port: port the server binds inside the container
            (UDP, LoL ENet). Default 5119.
        container_rl_port: port the RL bridge binds inside the container
            (TCP). Default 5120.
        env: extra environment variables to set inside the container.
        extra_run_args: pass-through args appended to ``docker run`` before
            the image name (e.g. ``("--cpus", "4.0")``).
        stop_timeout_s: grace period passed to ``docker stop`` on
            :meth:`ContainerHandle.terminate`.
        name_prefix: prefix for the auto-generated ``--name``.
    """

    image: str
    tag: str = "latest"
    pull_policy: PullPolicy = "missing"
    host_bind: str = "127.0.0.1"
    container_game_port: int = 5119
    container_rl_port: int = 5120
    env: dict[str, str] = field(default_factory=dict)
    extra_run_args: tuple[str, ...] = ()
    stop_timeout_s: float = 5.0
    name_prefix: str = "riftgym"

    @classmethod
    def priority(cls) -> int:
        # Public-default path. Lower-priority than future explicit local
        # builds so that a user with a real brokenwings checkout is opted
        # into the dev path automatically.
        return 1

    def start(self, *, game_port: int, rl_port: int) -> ServerHandle:
        self._ensure_image_present()
        name = f"{self.name_prefix}-{uuid.uuid4().hex[:8]}"
        cmd: list[str] = [
            _docker_bin(),
            "run",
            "-d",
            "--rm",
            "--name",
            name,
            "-p",
            f"{self.host_bind}:{game_port}:{self.container_game_port}/udp",
            "-p",
            f"{self.host_bind}:{rl_port}:{self.container_rl_port}/tcp",
        ]
        for k, v in self.env.items():
            cmd.extend(["-e", f"{k}={v}"])
        cmd.extend(self.extra_run_args)
        cmd.append(f"{self.image}:{self.tag}")

        log.info(
            "starting brokenwings container %s (game=%d, rl=%d, image=%s:%s)",
            name,
            game_port,
            rl_port,
            self.image,
            self.tag,
        )
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"docker run failed (rc={result.returncode}): {result.stderr.strip()}"
            )
        container_id = result.stdout.strip()
        return ContainerHandle(
            container_id,
            game_port=game_port,
            rl_port=rl_port,
            stop_timeout_s=self.stop_timeout_s,
        )

    def _ensure_image_present(self) -> None:
        ref = f"{self.image}:{self.tag}"
        if self.pull_policy == "always":
            self._pull(ref)
            return
        if self.pull_policy == "never":
            return
        # missing — only pull if image is absent locally
        result = subprocess.run(
            [_docker_bin(), "image", "inspect", ref],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return
        self._pull(ref)

    @staticmethod
    def _pull(ref: str) -> None:
        log.info("pulling %s", ref)
        result = subprocess.run(
            [_docker_bin(), "pull", ref], capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"docker pull {ref} failed (rc={result.returncode}): {result.stderr.strip()}"
            )


# Optional bind-mount + extra args helpers — small wrapper, kept here so
# the dataclass surface stays narrow.
def with_settings_override(base: ContainerRunConfig, settings_json: Path) -> ContainerRunConfig:
    """Return a new ``ContainerRunConfig`` that mounts ``settings_json``
    over the container's default ``Settings/GameInfo.json``.

    The mount target matches the entrypoint's expected layout in
    ``brokenwings/docker/Dockerfile``.
    """
    target = "/app/ChildrenOfTheGraveServerConsole/bin/Release/net10.0/Settings/GameInfo.json"
    extra = (
        *base.extra_run_args,
        "-v",
        f"{settings_json.resolve()}:{target}:ro",
    )
    return ContainerRunConfig(
        image=base.image,
        tag=base.tag,
        pull_policy=base.pull_policy,
        host_bind=base.host_bind,
        container_game_port=base.container_game_port,
        container_rl_port=base.container_rl_port,
        env=dict(base.env),
        extra_run_args=extra,
        stop_timeout_s=base.stop_timeout_s,
        name_prefix=base.name_prefix,
    )
