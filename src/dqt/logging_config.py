"""Shared loguru setup — default INFO so DEBUG SQL dumps stay quiet.

Call :func:`configure_logging` once per process (done automatically from
``dqt.__init__``).

Override with ``DQT_LOG_LEVEL=DEBUG`` or
``configure_logging("DEBUG", force=True)``.
"""

from __future__ import annotations

import os
import sys

from loguru import logger

__all__ = ["DEFAULT_LEVEL", "configure_logging"]

DEFAULT_LEVEL = "INFO"
_CONFIGURED = False
_APPLIED_LEVEL: str | None = None


def configure_logging(
    level: str | None = None,
    *,
    force: bool = False,
) -> str:
    """Install a single stderr sink at the resolved level.

    Resolution: explicit ``level`` → ``DQT_LOG_LEVEL`` env → ``INFO``.
    Idempotent unless ``force=True``. Returns the level currently applied.
    """
    global _CONFIGURED, _APPLIED_LEVEL
    resolved = (level or os.environ.get("DQT_LOG_LEVEL") or DEFAULT_LEVEL).upper()
    if _CONFIGURED and not force:
        return _APPLIED_LEVEL or resolved
    logger.remove()
    logger.add(sys.stderr, level=resolved)
    _CONFIGURED = True
    _APPLIED_LEVEL = resolved
    return resolved
