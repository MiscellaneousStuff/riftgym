"""``riftgym-launch`` — start N brokenwings servers and block until killed.

Useful for debugging riftgym connectivity from a shell (then connect a
script or a real LoL client to the host-mapped ports). The container is
torn down when this command exits.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from types import FrameType

from riftgym.launcher import (
    DEFAULT_BASE_GAME_PORT,
    DEFAULT_BASE_RL_PORT,
    ServerLauncher,
)
from riftgym.run_configs import ContainerRunConfig

DEFAULT_IMAGE = "ghcr.io/miscellaneousstuff/brokenwings"
DEFAULT_TAG = "latest"


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="riftgym-launch", description=__doc__)
    p.add_argument("-n", type=int, default=1, help="Number of servers to spawn.")
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--tag", default=DEFAULT_TAG)
    p.add_argument("--pull", choices=("missing", "always", "never"), default="missing")
    p.add_argument("--base-game-port", type=int, default=DEFAULT_BASE_GAME_PORT)
    p.add_argument("--base-rl-port", type=int, default=DEFAULT_BASE_RL_PORT)
    p.add_argument("--port-ready-timeout", type=float, default=120.0)
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    )

    rc = ContainerRunConfig(image=args.image, tag=args.tag, pull_policy=args.pull)
    launcher = ServerLauncher(
        n=args.n,
        run_config=rc,
        base_game_port=args.base_game_port,
        base_rl_port=args.base_rl_port,
        port_ready_timeout_s=args.port_ready_timeout,
    )

    def _handle_signal(signum: int, _frame: FrameType | None) -> None:
        print(f"\nreceived signal {signum}; tearing down servers", flush=True)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        launcher.start()
        for i, (gp, rp) in enumerate(zip(launcher.game_ports, launcher.rl_ports, strict=True)):
            print(f"server {i}: game UDP={gp}  rl TCP={rp}", flush=True)
        print("press Ctrl+C to stop", flush=True)
        signal.pause()
    except TimeoutError as exc:
        print(f"server failed to come up: {exc}", file=sys.stderr)
        return 1
    finally:
        launcher.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
