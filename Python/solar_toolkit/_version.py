"""Single source of truth for the ``solarphysics`` monorepository version.

Both the reusable ``solar-physics-toolkit`` distribution and the
``solarphysics-apps`` distribution read their version from this value through
the ``attr`` directive in their ``pyproject.toml`` files. The release command
keeps the two copies in ``Python/solar_toolkit/_version.py`` and
``Apps/solar_apps/_version.py`` identical; release is the only supported way
to change this number.
"""

from __future__ import annotations

__version__ = "0.3.0"
