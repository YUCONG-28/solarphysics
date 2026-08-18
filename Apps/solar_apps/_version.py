"""Single source of truth for the ``solarphysics`` monorepository version.

This is the Apps-side mirror of ``Python/solar_toolkit/_version.py``. The two
must stay identical; the ``tools release`` command keeps them in sync and is
the only supported way to change this number.
"""

from __future__ import annotations

__version__ = "0.3.0"
