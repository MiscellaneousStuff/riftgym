"""Trivial sanity test so CI has something to run before real code lands."""

from __future__ import annotations

import riftgym


def test_version() -> None:
    assert isinstance(riftgym.__version__, str)
    assert riftgym.__version__
