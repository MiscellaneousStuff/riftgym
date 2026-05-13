"""RunConfig discovery + factory."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from riftgym.run_configs.compose import ComposeHandle, ComposeRunConfig
from riftgym.run_configs.container import ContainerHandle, ContainerRunConfig
from riftgym.run_configs.lib import RunConfig, ServerHandle, wait_for_port

__all__ = [
    "ComposeHandle",
    "ComposeRunConfig",
    "ContainerHandle",
    "ContainerRunConfig",
    "RunConfig",
    "ServerHandle",
    "default_multilane5_settings",
    "get_run_config",
    "wait_for_port",
]


def default_multilane5_settings() -> Path:
    """Filesystem path to the bundled multilane5 training settings JSON.

    Ships with riftgym (under ``src/riftgym/_data/``) so callers don't
    need a brokenwings checkout to launch training. The JSON mirrors
    brokenwings's ``docker/Settings/GameInfo-multilane5.json``:
    5-lane Ezreal mirror, ``MINION_SPAWNS_ENABLED=false`` (workaround
    for brokenwings #5 — minion pushes cause natural game-overs that
    kill long training runs), ``RL_HZ=30``, ``HEADLESS=true``,
    ``KEEP_ALIVE_WHEN_EMPTY=true``, ``RL_BIND_ADDRESS=Any`` (containers
    rebind via host-side port mapping clamp).

    Pass to :class:`ComposeRunConfig` to mount it into the server::

        rc = ComposeRunConfig(
            compose_file=Path("./compose.yaml"),
            settings_json=default_multilane5_settings(),
        )

    Only works for filesystem-backed installations (regular and editable
    wheels) — riftgym doesn't ship as zipapp.
    """
    resource = files("riftgym").joinpath("_data", "GameInfo-multilane5.json")
    return Path(str(resource))


def get_run_config(**overrides: object) -> RunConfig:
    """Pick the highest-:meth:`RunConfig.priority` applicable subclass
    and instantiate it with ``overrides``.

    For now there's only :class:`ContainerRunConfig`, so this is mostly a
    forward-looking hook — once a local-build subclass lands, the user
    will get the dev path automatically when their environment opts in.
    """
    candidates = [c for c in RunConfig.all_subclasses() if c.priority() is not None]
    if not candidates:
        raise RuntimeError("no RunConfig subclass applies on this host")
    candidates.sort(key=lambda c: c.priority() or 0, reverse=True)
    return candidates[0](**overrides)  # type: ignore[arg-type]
