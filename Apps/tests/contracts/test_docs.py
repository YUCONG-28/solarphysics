"""Documentation completeness, portability, and privacy contracts."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

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


def _public_markdown() -> list[Path]:
    """Include new documentation, excluding ignored runtime/build artifacts."""
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "*.md"],
        cwd=REPO_ROOT,
    )
    return sorted(
        {
            path
            for name in names.decode("utf-8").split("\0")
            if name and (path := REPO_ROOT / name).is_file()
        }
    )


def _prose(text: str) -> str:
    return re.sub(r"^(`{3,}|~{3,}).*?^\1[^\n]*$", "", text, flags=re.M | re.S)


def _links(text: str) -> list[str]:
    prose = _prose(text)
    inline = re.findall(r"\[[^\]\n]*\]\(([^)\n]+)\)", prose)
    references = re.findall(r"^\s*\[[^\]\n]+\]:\s*(\S+)", prose, re.M)
    return [value.split(' "', 1)[0].strip("<>") for value in inline + references]


def _anchors(text: str) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", _prose(text), re.M):
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        count = counts.get(slug, 0)
        counts[slug] = count + 1
        anchors.add(f"{slug}-{count}" if count else slug)
    anchors.update(re.findall(r"<a\s+(?:id|name)=[\"\']([^\"\']+)", text))
    return anchors


@pytest.mark.parametrize("name", ["README.md", "README.en.md"])
def test_root_readme_is_library_first_and_links_application_adapters(name: str) -> None:
    text = (REPO_ROOT / name).read_text(encoding="utf-8")
    assert text.startswith("# Solar Physics Toolkit")
    assert "(Python/README.md)" in text
    assert "(Apps/README.md)" in text
    assert text.index("(Python/README.md)") < text.index("(Apps/README.md)")
    assert "solar_toolkit" in text
    assert "Local repository" not in text
    assert "PUBLIC_BASE" not in text and "SHA256SUMS" not in text


@pytest.mark.parametrize("stem", ["README", "CONTRIBUTING"])
def test_bilingual_entry_points_expose_the_same_guides(stem: str) -> None:
    def targets(name: str) -> set[str]:
        return {
            link.replace(".en.md", ".md")
            for link in _links((REPO_ROOT / name).read_text(encoding="utf-8"))
        }

    assert targets(f"{stem}.md") == targets(f"{stem}.en.md")
    for name in (f"{stem}.md", f"{stem}.en.md"):
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        assert f"({stem}.md)" in text
        assert f"({stem}.en.md)" in text


def test_public_documentation_links_and_anchors_resolve() -> None:
    offenders = []
    for source in _public_markdown():
        for link in _links(source.read_text(encoding="utf-8")):
            target = urlsplit(link)
            if target.scheme or target.netloc:
                continue
            path = (
                (source.parent / unquote(target.path)).resolve()
                if target.path
                else source
            )
            reason = None
            if not path.is_relative_to(REPO_ROOT) or not path.exists():
                reason = "missing or external local target"
            elif target.fragment and path.suffix == ".md":
                if unquote(target.fragment) not in _anchors(
                    path.read_text(encoding="utf-8")
                ):
                    reason = "missing heading anchor"
            if reason:
                offenders.append(f"{source.relative_to(REPO_ROOT)}: {link}: {reason}")
    assert offenders == []


def test_apps_guides_document_current_public_contract() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            APPS_README,
            APPS_ROOT / "docs" / "architecture.md",
            APPS_ROOT / "docs" / "development.md",
        )
    )
    required = (
        "solarphysics_env_latest",
        "solarphysics_env",
        "Python 3.14",
        "run.sh",
        "run.ps1",
        "frontend app-v1",
        "app-v1-preview",
        "--config-file",
        "paths.local.yaml",
        "Local/",
        "Auto",
        "Light",
        "Dark",
        ".spapp.json",
        ".spflow.json",
        "MPL-2.0",
    )
    missing = [value for value in required if value not in text]
    assert missing == []


def test_public_markdown_has_no_machine_identity_or_operation_record() -> None:
    offenders: list[str] = []
    for path in _public_markdown():
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
    markdown = [
        ROOT_README,
        REPO_ROOT / "README.en.md",
        REPO_ROOT / "CONTRIBUTING.md",
        REPO_ROOT / "CONTRIBUTING.en.md",
        APPS_README,
        *sorted((APPS_ROOT / "docs").rglob("*.md")),
    ]
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
