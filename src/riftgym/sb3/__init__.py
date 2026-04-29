"""SB3-coupled helpers (vec env, snapshot pool, trainer).

Importing this subpackage requires ``stable-baselines3`` and ``sb3-contrib``.
Install with ``pip install riftgym[sb3]``.
"""

from __future__ import annotations

from importlib.util import find_spec

if find_spec("stable_baselines3") is None or find_spec("sb3_contrib") is None:
    raise ImportError(
        "riftgym.sb3 requires stable-baselines3 and sb3-contrib. "
        "Install with: pip install 'riftgym[sb3]'"
    )
