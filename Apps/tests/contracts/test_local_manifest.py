"""Publishable Apps/ and ignored Local/ boundary contracts."""

from __future__ import annotations

from pathlib import Path

APPS_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = APPS_ROOT.parent


def test_private_runtime_and_migration_evidence_are_fully_ignored() -> None:
    ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8-sig").splitlines()
    assert "/Local/" in ignore
    assert "/Local-migration-backup/" in ignore


def test_apps_contains_no_private_snapshot_or_machine_data() -> None:
    forbidden_names = {
        "PUBLIC_BASE.md",
        "SHA256SUMS",
        "paths.local.yaml",
    }
    offenders = [
        str(path.relative_to(APPS_ROOT))
        for path in APPS_ROOT.rglob("*")
        if path.is_file()
        and (
            path.name in forbidden_names
            or path.suffix.casefold() in {".xlsx", ".xls"}
            or "legacy" in {part.casefold() for part in path.parts}
            or "history" in {part.casefold() for part in path.parts}
        )
    ]
    assert not offenders, f"Private or historical files under Apps/: {offenders}"


def test_only_one_public_machine_config_template_exists() -> None:
    templates = sorted((APPS_ROOT / "configs" / "examples").glob("*.yaml"))
    assert [path.name for path in templates] == ["paths.example.yaml"]
    text = templates[0].read_text(encoding="utf-8-sig")
    assert "allowed_roots: []" in text
    assert "paths.local.yaml" not in str(templates[0])


def test_package_metadata_includes_frontend_and_notice_assets() -> None:
    metadata = (APPS_ROOT / "pyproject.toml").read_text(encoding="utf-8-sig")
    assert "**/*" not in metadata
    assert '"ui/media/mediabunny-1.50.8.cjs"' in metadata
    assert '"frontends/image_viewer/templates/index.html"' in metadata
    assert "[project.scripts]" not in metadata
    assert list((APPS_ROOT / "solar_apps").rglob("NOTICE.txt"))
