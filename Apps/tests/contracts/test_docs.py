"""Documentation completeness, portability, and privacy contracts."""

from __future__ import annotations

import re
from pathlib import Path

APPS_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = APPS_ROOT.parent
ROOT_README = REPO_ROOT / "README.md"
APPS_README = APPS_ROOT / "README.md"

MACHINE_PATHS = (
    re.compile(r"\bD:[\\/]solarphysics\b", re.I),
    re.compile(r"\bD:[\\/]miniforge3\b", re.I),
    re.compile(r"\b[A-Z]:[\\/]Users[\\/](?!<(?:user|username)>)", re.I),
)
PRIVATE_EMAIL = re.compile(r"\b[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@gmail\.com\b", re.I)


def test_root_readme_remains_library_facing_and_links_apps_manual() -> None:
    text = ROOT_README.read_text(encoding="utf-8")
    assert "# Solar Physics Toolkit" in text
    assert "[Apps manual](Apps/README.md)" in text
    assert "solar_toolkit" in text
    assert "Local repository" not in text
    assert "PUBLIC_BASE" not in text and "SHA256SUMS" not in text


def test_apps_manual_documents_complete_public_contract() -> None:
    text = APPS_README.read_text(encoding="utf-8")
    required = (
        "solarphysics_env_latest",
        "solarphysics_env",
        "Solar Physics App 1.0 is the stable native application",
        "frontend app-v1",
        "app-v1-preview",
        "deprecated compatibility",
        "frontend aia-radio-composite",
        "frontend workbench",
        "frontend image-viewer",
        "frontend image-composer",
        "frontend bad-frame-review",
        "frontend source-map",
        "frontend dart-spectrogram",
        "frontend roi-lightcurve",
        "frontend radio-composite",
        "frontend source-trajectory",
        "Compatibility aliases",
        "Auto",
        "Light",
        "Dark",
        "Reset UI State",
        "StateStore",
        "RecentPathMemory",
        "SpatialRadioDisplay",
        "Allowed roots",
        "Privacy and publication policy",
        "Troubleshooting",
        "MPL-2.0",
    )
    missing = [value for value in required if value not in text]
    assert missing == []


def test_public_markdown_has_no_machine_identity_or_operation_record() -> None:
    markdown = sorted(
        [ROOT_README, REPO_ROOT / "ARCHITECTURE.md"] + list(APPS_ROOT.rglob("*.md"))
    )
    offenders: list[str] = []
    for path in markdown:
        text = path.read_text(encoding="utf-8", errors="replace")
        reasons = []
        if any(pattern.search(text) for pattern in MACHINE_PATHS):
            reasons.append("machine-specific path")
        if PRIVATE_EMAIL.search(text):
            reasons.append("private email")
        if "PUBLIC_BASE" in text or "SHA256SUMS" in text:
            reasons.append("historical publication manifest")
        if re.search(r"\b\d+\s+passed\b", text, re.I):
            reasons.append("test execution count")
        if reasons:
            offenders.append(
                f"{path.relative_to(REPO_ROOT).as_posix()}: {', '.join(reasons)}"
            )
    assert offenders == []


def test_documented_python_commands_use_miniforge() -> None:
    markdown = [ROOT_README, APPS_README, *sorted((APPS_ROOT / "docs").rglob("*.md"))]
    offenders: list[str] = []
    for path in markdown:
        in_powershell = False
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if stripped == "```powershell":
                in_powershell = True
                continue
            if stripped == "```":
                in_powershell = False
                continue
            if stripped.startswith("#"):
                continue
            if not in_powershell or "python" not in stripped.casefold():
                continue
            if "$Conda run" not in stripped and "run.ps1" not in stripped:
                offenders.append(f"{path.name}:{line_number}: {stripped}")
    assert offenders == []
