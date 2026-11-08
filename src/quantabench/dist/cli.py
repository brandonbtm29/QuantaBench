"""Command-line entry points of the dist package. Each module may define ``register_cli(subparsers)``;
``cli.main`` calls ``register(subparsers)`` once, so builders never edit the same parser block."""

from __future__ import annotations

import importlib


def register(subparsers) -> None:
    for name in ("profile", "lanes", "worker", "client", "timings"):
        mod = importlib.import_module(f"{__package__}.{name}")
        fn = getattr(mod, "register_cli", None)
        if fn is not None:
            fn(subparsers)
