"""Spawn N brokenwings servers with disjoint host ports and clean them up.

Thin orchestrator around :class:`riftgym.run_configs.lib.RunConfig`. The
RunConfig knows how to start one server (subprocess, container, ...);
this class assigns ports, calls :meth:`RunConfig.start` ``n`` times,
waits for each bridge port to accept TCP, and tears everything down on
:meth:`__exit__`.

Default port layout matches brokenwings/tools/launch_servers.py: base
``5119`` (game UDP) + ``5120`` (RL TCP), stride ``2``, so server ``i``
binds host ports ``(5119 + 2i, 5120 + 2i)``.
"""

from __future__ import annotations

import logging
from types import TracebackType

from riftgym.run_configs.lib import RunConfig, ServerHandle, wait_for_port

log = logging.getLogger(__name__)

DEFAULT_BASE_GAME_PORT = 5119
DEFAULT_BASE_RL_PORT = 5120
DEFAULT_PORT_STRIDE = 2


class ServerLauncher:
    """Context-managed launcher for ``n`` brokenwings servers.

    Args:
        n: number of servers to spawn.
        run_config: how to launch one server. Same RunConfig is used
            for all ``n``; each call gets unique host-side ports.
        base_game_port: host port for server 0's game socket (UDP).
        base_rl_port: host port for server 0's RL bridge socket (TCP).
        port_stride: distance between consecutive servers' port bases.
            Default ``2`` matches the brokenwings convention.
        rl_host: address used to probe the RL bridge after start. Defaults
            to ``127.0.0.1``; should match the host-side bind chosen by
            the RunConfig (e.g. ``ContainerRunConfig.host_bind``).
        port_ready_timeout_s: per-server timeout waiting for the bridge
            port to accept TCP. Container cold-starts can take 10-30s
            (Roslyn script compile on first run).
    """

    def __init__(
        self,
        *,
        n: int,
        run_config: RunConfig,
        base_game_port: int = DEFAULT_BASE_GAME_PORT,
        base_rl_port: int = DEFAULT_BASE_RL_PORT,
        port_stride: int = DEFAULT_PORT_STRIDE,
        rl_host: str = "127.0.0.1",
        port_ready_timeout_s: float = 60.0,
    ) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.n = n
        self.run_config = run_config
        self.base_game_port = base_game_port
        self.base_rl_port = base_rl_port
        self.port_stride = port_stride
        self.rl_host = rl_host
        self.port_ready_timeout_s = port_ready_timeout_s

        self._handles: list[ServerHandle] = []
        self._game_ports: list[int] = []
        self._rl_ports: list[int] = []

    @property
    def handles(self) -> list[ServerHandle]:
        return list(self._handles)

    @property
    def game_ports(self) -> list[int]:
        return list(self._game_ports)

    @property
    def rl_ports(self) -> list[int]:
        return list(self._rl_ports)

    def start(self) -> None:
        try:
            for i in range(self.n):
                game_port = self.base_game_port + i * self.port_stride
                rl_port = self.base_rl_port + i * self.port_stride
                handle = self.run_config.start(game_port=game_port, rl_port=rl_port)
                self._handles.append(handle)
                self._game_ports.append(game_port)
                self._rl_ports.append(rl_port)

            # Wait for all bridges to accept connections. Done after all
            # spawns have been kicked off so cold-start is parallelized.
            for i, rl_port in enumerate(self._rl_ports):
                log.info("waiting for server %d bridge on %s:%d", i, self.rl_host, rl_port)
                wait_for_port(self.rl_host, rl_port, timeout_s=self.port_ready_timeout_s)
        except BaseException:
            # Don't leak containers if mid-launch failure throws.
            self.close()
            raise

    def close(self) -> None:
        for h in self._handles:
            try:
                h.terminate()
            except Exception:
                log.exception("error terminating server handle")
        self._handles.clear()
        self._game_ports.clear()
        self._rl_ports.clear()

    def __enter__(self) -> ServerLauncher:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
