"""``ComposeRunConfig`` — RunConfig that brings the brokenwings server up
via ``docker compose`` instead of bare ``docker run``.

Pairs with ``riftgym/compose.yaml`` at the repo root, which declares a
single ``server`` service that pulls the public brokenwings image from
GHCR (or a locally-built image, via ``BROKENWINGS_IMAGE`` override).
For training (n>1 servers), each call to :meth:`start` creates an
isolated compose project — different ``--project-name`` plus different
host port pairs injected via ``RIFTGYM_GAME_PORT`` / ``RIFTGYM_RL_PORT``
env vars that the YAML's ``${...}`` substitutions read.

Why this peer of :class:`ContainerRunConfig` exists: compose is the
declarative deployment shape we want for multi-server training, and it
keeps the host-side mount + env-var wiring (settings JSON, future RL
auth tokens) in one YAML file users can read instead of buried in
``docker run`` argv. Single-host, multi-container topologies are the
sweet spot for compose; we use exactly that shape here.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from riftgym.run_configs._docker import docker_bin
from riftgym.run_configs.lib import RunConfig, ServerHandle

log = logging.getLogger(__name__)

PullPolicy = Literal["missing", "always", "never"]

# Path inside the brokenwings container where compose mounts the
# settings-JSON override (see riftgym/compose.yaml `volumes:` block).
# Set as `BROKENWINGS_GAME_INFO` so server.sh reads it instead of the
# default `Settings/GameInfo.json`.
_CONTAINER_OVERRIDE_PATH = "Settings/GameInfo-override.json"


class ComposeHandle:
    """Live ``docker compose up -d`` server. Tearing down runs ``compose down``.

    Each handle owns one compose project (``-p <project_name>``) so
    multi-server topologies don't collide on the default project name
    (which is the YAML file's parent directory).
    """

    def __init__(
        self,
        *,
        project: str,
        compose_file: Path,
        service: str,
        container_id: str,
        game_port: int,
        rl_port: int,
        stop_timeout_s: float = 5.0,
    ) -> None:
        self._project = project
        self._compose_file = compose_file
        self._service = service
        self._container_id = container_id
        self._game_port = game_port
        self._rl_port = rl_port
        self._stop_timeout_s = stop_timeout_s
        self._terminated = False
        # Belt-and-braces cleanup mirrors ContainerHandle: a Ctrl+C
        # before the surrounding context manager runs would otherwise
        # leave the compose project running until reboot, holding host
        # ports.
        atexit.register(self.terminate)

    @property
    def project(self) -> str:
        return self._project

    @property
    def container_id(self) -> str:
        return self._container_id

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
            [docker_bin(), "inspect", "-f", "{{.State.Running}}", self._container_id],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"

    def terminate(self) -> None:
        if self._terminated:
            return
        self._terminated = True
        atexit.unregister(self.terminate)
        with contextlib.suppress(subprocess.CalledProcessError, FileNotFoundError):
            subprocess.run(
                [
                    docker_bin(),
                    "compose",
                    "-f",
                    str(self._compose_file),
                    "-p",
                    self._project,
                    "down",
                    "-t",
                    str(int(self._stop_timeout_s)),
                ],
                capture_output=True,
                check=False,
            )

    def wait(self, timeout: float | None = None) -> int | None:
        cmd = [docker_bin(), "wait", self._container_id]
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
            [
                docker_bin(),
                "compose",
                "-f",
                str(self._compose_file),
                "-p",
                self._project,
                "logs",
                self._service,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return (result.stdout or "") + (result.stderr or "")

    def __enter__(self) -> ComposeHandle:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.terminate()


@dataclass(slots=True)
class ComposeRunConfig(RunConfig):
    """Run the brokenwings server via ``docker compose up -d <service>``.

    Args:
        compose_file: path to a compose YAML declaring the server
            service. Typically ``./compose.yaml`` at the riftgym repo
            root; the YAML reads ``RIFTGYM_GAME_PORT``,
            ``RIFTGYM_RL_PORT``, ``RIFTGYM_SETTINGS_JSON`` env vars
            for per-instance overrides.
        service: service name in the YAML to bring up. Default
            ``server``.
        project_name: ``--project-name`` for the compose project. ``None``
            (default) auto-generates a unique name per :meth:`start`
            call so multi-server topologies don't collide.
        image_override: forwarded as ``BROKENWINGS_IMAGE`` env var,
            consumed by the YAML's ``${BROKENWINGS_IMAGE:-...}``.
        tag_override: forwarded as ``BROKENWINGS_TAG``.
        pull_policy: ``missing`` / ``always`` / ``never``. Forwarded as
            ``--pull`` to ``docker compose up`` when not ``missing``
            (compose's default is ``missing``).
        settings_json: optional path to a host-side settings JSON to
            mount into the container. The YAML mounts it at
            ``Settings/GameInfo-override.json`` (under the server's
            working directory) and ``BROKENWINGS_GAME_INFO`` is set to
            that relative path so ``server.sh`` reads it.
        extra_env: additional env vars forwarded to ``docker compose
            up``. Useful for ``RL_HZ``, ``HEADLESS``, etc.
        stop_timeout_s: ``-t`` for ``docker compose down``.
    """

    compose_file: Path
    service: str = "server"
    project_name: str | None = None
    image_override: str | None = None
    tag_override: str | None = None
    pull_policy: PullPolicy = "missing"
    settings_json: Path | None = None
    extra_env: dict[str, str] = field(default_factory=dict)
    stop_timeout_s: float = 5.0

    @classmethod
    def priority(cls) -> int:
        # Higher than ContainerRunConfig (1): when both could apply, the
        # declarative compose path is preferred. ``get_run_config()``
        # consumers stay opt-in by passing the right keyword args.
        return 2

    def start(self, *, game_port: int, rl_port: int) -> ServerHandle:
        compose_file = Path(self.compose_file).resolve()
        if not compose_file.exists():
            raise FileNotFoundError(f"compose file not found: {compose_file}")

        project = self.project_name or f"riftgym-srv-{uuid.uuid4().hex[:8]}"

        env = os.environ.copy()
        env["RIFTGYM_GAME_PORT"] = str(game_port)
        env["RIFTGYM_RL_PORT"] = str(rl_port)
        if self.image_override is not None:
            env["BROKENWINGS_IMAGE"] = self.image_override
        if self.tag_override is not None:
            env["BROKENWINGS_TAG"] = self.tag_override
        if self.settings_json is not None:
            settings_path = Path(self.settings_json).resolve()
            if not settings_path.exists():
                raise FileNotFoundError(f"settings JSON not found: {settings_path}")
            env["RIFTGYM_SETTINGS_JSON"] = str(settings_path)
            env["BROKENWINGS_GAME_INFO"] = _CONTAINER_OVERRIDE_PATH
        env.update(self.extra_env)

        up_cmd: list[str] = [
            docker_bin(),
            "compose",
            "-f",
            str(compose_file),
            "-p",
            project,
            "up",
            "-d",
        ]
        if self.pull_policy != "missing":
            up_cmd.extend(["--pull", self.pull_policy])
        up_cmd.append(self.service)

        log.info(
            "compose up -d project=%s service=%s game=%d rl=%d image=%s tag=%s",
            project,
            self.service,
            game_port,
            rl_port,
            self.image_override or "<default>",
            self.tag_override or "<default>",
        )
        result = subprocess.run(up_cmd, env=env, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"docker compose up failed (rc={result.returncode}): "
                f"stdout={result.stdout.strip()} stderr={result.stderr.strip()}"
            )

        ps = subprocess.run(
            [
                docker_bin(),
                "compose",
                "-f",
                str(compose_file),
                "-p",
                project,
                "ps",
                "-q",
                self.service,
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        container_id = ps.stdout.strip().splitlines()[0] if ps.stdout.strip() else ""
        if not container_id:
            # Best-effort cleanup so we don't leak a half-up project on
            # the unlikely path that `up -d` succeeded but `ps -q`
            # returned nothing.
            with contextlib.suppress(subprocess.CalledProcessError, FileNotFoundError):
                subprocess.run(
                    [docker_bin(), "compose", "-f", str(compose_file), "-p", project, "down"],
                    env=env,
                    capture_output=True,
                    check=False,
                )
            raise RuntimeError(
                f"compose project '{project}' came up but service '{self.service}' "
                f"has no running container (ps stderr: {ps.stderr.strip()})"
            )

        return ComposeHandle(
            project=project,
            compose_file=compose_file,
            service=self.service,
            container_id=container_id,
            game_port=game_port,
            rl_port=rl_port,
            stop_timeout_s=self.stop_timeout_s,
        )
