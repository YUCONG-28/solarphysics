# SPDX-License-Identifier: GPL-3.0-only
"""Import-safe command entry point for Solar Physics App 1.0."""

from __future__ import annotations

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Load PyQt6 only after confirming no legacy Qt binding is in-process."""

    conflicts = sorted(
        name
        for name in sys.modules
        if name == "PyQt5"
        or name.startswith("PyQt5.")
        or name == "PySide6"
        or name.startswith("PySide6.")
    )
    if conflicts:
        print(
            "App 1.0 must run in a dedicated PyQt6 process; "
            f"legacy Qt modules are already loaded: {', '.join(conflicts[:3])}",
            file=sys.stderr,
        )
        return 2

    from .application import main as application_main

    return application_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
