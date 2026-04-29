"""RunConfig discovery + factory."""

from __future__ import annotations

from riftgym.run_configs.container import ContainerHandle, ContainerRunConfig
from riftgym.run_configs.lib import RunConfig, ServerHandle, wait_for_port

__all__ = [
    "ContainerHandle",
    "ContainerRunConfig",
    "RunConfig",
    "ServerHandle",
    "get_run_config",
    "wait_for_port",
]


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
