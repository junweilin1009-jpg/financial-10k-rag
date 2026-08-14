"""Shared, deliberately minimal logging configuration for application entrypoints."""

from __future__ import annotations

import logging


def configure_logging(*, verbose: bool = False) -> None:
    """Configure human-readable stderr logs without overriding host applications."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
