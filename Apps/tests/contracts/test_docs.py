"""Documentation completeness, portability, and privacy contracts."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest
import yaml

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


@pytest.mark.parametrize("name", ["README.md", "README.zh-CN.md"])
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
            urlsplit(link).path.replace(".zh-CN.md", ".md")
            for link in _links((REPO_ROOT / name).read_text(encoding="utf-8"))
        }

    assert targets(f"{stem}.md") == targets(f"{stem}.zh-CN.md")
    for name in (f"{stem}.md", f"{stem}.zh-CN.md"):
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        assert f"({stem}.md)" in text
        assert f"({stem}.zh-CN.md)" in text


@pytest.mark.parametrize(
    ("stem", "previous_anchors", "chinese_anchors"),
    [
        (
            "README",
            {
                "start-with-the-library",
                "application-demonstration",
                "repository-layout",
                "contributing-and-maintenance",
                "licensing-and-citation",
            },
            {"从基础模块开始", "应用展示", "仓库结构", "贡献与维护", "许可与引用"},
        ),
        (
            "CONTRIBUTING",
            {
                "prepare-the-environment",
                "submit-a-reproducible-change",
                "public-content-boundary",
            },
            {"贡献指南", "准备环境", "提交可复现的修改", "公开内容边界"},
        ),
    ],
)
def test_previous_language_paths_and_anchors_remain_usable(
    stem: str, previous_anchors: set[str], chinese_anchors: set[str]
) -> None:
    compatibility = (REPO_ROOT / f"{stem}.en.md").read_text(encoding="utf-8")
    canonical = (REPO_ROOT / f"{stem}.md").read_text(encoding="utf-8")
    assert previous_anchors <= _anchors(compatibility)
    assert chinese_anchors <= _anchors(canonical)
    targets = set(_links(compatibility))
    for anchor in previous_anchors:
        assert f"{stem}.md#{anchor}" in targets
    # Installation instructions have one maintained English home.
    assert "```bash" not in compatibility and "```powershell" not in compatibility


def test_documentation_index_reaches_current_guides_and_metadata() -> None:
    home_links = set(_links(ROOT_README.read_text(encoding="utf-8")))
    assert "docs/README.md" in home_links
    index = REPO_ROOT / "docs" / "README.md"
    paths = {urlsplit(link).path for link in _links(index.read_text(encoding="utf-8"))}
    assert {
        "../Python/README.md",
        "../Python/docs/FUNCTION_MAP.md",
        "../Apps/README.md",
        "../CONTRIBUTING.md",
        "../CONTRIBUTING.zh-CN.md",
        "../environment/README.md",
        "../Python/CHANGELOG.md",
        "../LICENSING.md",
        "../CITATION.cff",
        "radio_comparison.md",
        "time_matching.md",
        "aia_radio_composite.md",
        "migration.md",
    } <= paths


def test_package_and_citation_metadata_identify_the_same_current_author() -> None:
    for partition in ("Python", "Apps"):
        metadata = tomllib.loads(
            (REPO_ROOT / partition / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]
        assert metadata["authors"] == [{"name": "Y. Li"}]
    for path in (REPO_ROOT / "CITATION.cff", REPO_ROOT / "Python" / "CITATION.cff"):
        citation = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert citation["cff-version"] == "1.2.0"
        assert citation["type"] == "software"
        assert citation["authors"] == [
            {
                "family-names": "Li",
                "given-names": "Y.",
                "affiliation": "Shandong University",
            }
        ]
        assert (
            citation["repository-code"] == "https://github.com/YUCONG-28/solarphysics"
        )
    # A repository-wide MIT label would hide the Apps component exceptions.
    root_citation = yaml.safe_load((REPO_ROOT / "CITATION.cff").read_text())
    assert root_citation.get("license") != "MIT"
    scope = (REPO_ROOT / "LICENSING.md").read_text(encoding="utf-8")
    assert {
        "LICENSE",
        "Python/LICENSE",
        "Apps/LICENSE",
        "Apps/solar_apps/frontends/app_v1/LICENSE.md",
        "Apps/LICENSES/GPL-3.0-only.txt",
        "Apps/solar_apps/ui/media/NOTICE.txt",
    } <= set(_links(scope))


def test_homepage_installation_examples_have_the_documented_output(
    tmp_path: Path,
) -> None:
    environment = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "Python"))
    for name in ("README.md", "README.zh-CN.md"):
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        assert (
            "create -n solarphysics_env_latest -c conda-forge python=3.14 pip" in text
        )
        examples = set(re.findall(r"python (Python/examples/[^\s`]+\.py)", text))
        assert examples
        for example in examples:
            result = subprocess.run(
                [sys.executable, str(REPO_ROOT / example)],
                cwd=tmp_path,
                env=environment,
                text=True,
                capture_output=True,
                check=True,
                timeout=60,
            )
            assert result.stdout.strip() in re.findall(
                r"```text\n(.*?)\n```", text, re.S
            )
        snippets = re.findall(r"```python\n(.*?)\n```", text, re.S)
        assert snippets
        for snippet in snippets:
            result = subprocess.run(
                [sys.executable, "-c", snippet],
                cwd=tmp_path,
                env=environment,
                text=True,
                capture_output=True,
                check=True,
                timeout=60,
            )
            assert result.stdout.strip() == "0"


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
        REPO_ROOT / "README.zh-CN.md",
        REPO_ROOT / "README.en.md",
        REPO_ROOT / "CONTRIBUTING.md",
        REPO_ROOT / "CONTRIBUTING.zh-CN.md",
        REPO_ROOT / "CONTRIBUTING.en.md",
        REPO_ROOT / "Python" / "README.md",
        REPO_ROOT / "Python" / "CONTRIBUTING.md",
        APPS_README,
        *sorted((APPS_ROOT / "docs").rglob("*.md")),
    ]
    offenders: list[str] = []
    for path in markdown:
        language = ""
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if stripped in ("```powershell", "```bash"):
                language = stripped.removeprefix("```")
                continue
            if stripped == "```":
                language = ""
                continue
            if stripped.startswith("#"):
                continue
            if not language or not re.search(r"\bpython\s", stripped, re.I):
                continue
            if "run -n solarphysics_env" not in stripped:
                offenders.append(f"{path.name}:{line_number}: {stripped}")
    assert offenders == []
