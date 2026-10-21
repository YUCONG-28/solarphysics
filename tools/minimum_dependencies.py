"""Check the library's declared runtime floors and its minimum CI environment.

The constraint file tests direct runtime floors, not minimum optional, test or
transitive dependencies. It is not an exact environment or artifact lock.
"""

from __future__ import annotations

import argparse
import tomllib
from importlib.metadata import version
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def check(*, verify_installed: bool = False) -> None:
    """Reject stale floors or a runtime that did not install those floors."""
    project = tomllib.loads((ROOT / "Python/pyproject.toml").read_text())["project"]
    constraints = {}
    for line in (
        (ROOT / "environment/library-minimum-py312.txt").read_text().splitlines()
    ):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        requirement = Requirement(line)
        specifiers = list(requirement.specifier)
        if len(specifiers) != 1 or specifiers[0].operator != "==":
            raise ValueError(f"Expected one exact constraint: {line}")
        name = canonicalize_name(requirement.name)
        if name in constraints:
            raise ValueError(f"Duplicate minimum constraint: {name}")
        constraints[name] = Version(specifiers[0].version)
    expected = {}
    for text in project["dependencies"]:
        requirement = Requirement(text)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        floors = [
            Version(item.version)
            for item in requirement.specifier
            if item.operator in (">=", "==")
        ]
        if not floors:
            raise ValueError(f"No inclusive dependency floor: {text}")
        floor = max(floors)
        if floor not in requirement.specifier:
            raise ValueError(f"Invalid dependency floor: {text}")
        expected[canonicalize_name(requirement.name)] = floor
    if constraints != expected:
        raise ValueError(
            f"Minimum constraints differ from declarations: {constraints} != {expected}"
        )
    if verify_installed:
        for name, expected_version in expected.items():
            actual = Version(version(name))
            if actual != expected_version:
                raise ValueError(
                    f"{name}: expected minimum {expected_version}, installed {actual}"
                )
    print(
        f"Verified {len(expected)} direct runtime dependency floors"
        + (" and installed versions" if verify_installed else "")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-installed", action="store_true")
    check(verify_installed=parser.parse_args().verify_installed)
