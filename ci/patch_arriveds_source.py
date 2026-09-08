"""Swap arriveds git source for the in-repo CI stub (GitHub Actions)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"

_GIT_SOURCE = (
    'arriveds = { git = "ssh://git@github.com/rarko-arrive/arrive-ds.git", rev = "v0.3.2" }'
)
_STUB_SOURCE = 'arriveds = { path = "ci/arriveds-stub", editable = true }'


def main() -> int:
    text = PYPROJECT.read_text(encoding="utf-8")
    if _STUB_SOURCE in text:
        print("pyproject.toml already uses arriveds CI stub")
        return 0
    if _GIT_SOURCE not in text:
        print(
            "expected arriveds git source line not found in pyproject.toml",
            file=sys.stderr,
        )
        return 1
    PYPROJECT.write_text(text.replace(_GIT_SOURCE, _STUB_SOURCE), encoding="utf-8")
    print("patched arriveds source → ci/arriveds-stub")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
