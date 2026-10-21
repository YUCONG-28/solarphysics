from __future__ import annotations

import errno
from pathlib import Path

import pytest
from astropy.io import fits

from solar_toolkit.data import stereo_manifest


def _require_symlink_support(target: Path, link: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError as exc:
        if (
            exc.errno in {errno.EPERM, errno.EACCES}
            or getattr(exc, "winerror", None) == 1314
        ):
            pytest.skip("Creating symlinks requires platform permission")
        raise
    link.unlink()


def test_manifest_links_resolve_to_files_outside_the_view_tree(tmp_path):
    raw = tmp_path / "separate-source" / "nested" / "example.fts"
    raw.parent.mkdir(parents=True)
    fits.PrimaryHDU(header=fits.Header({"WAVELNTH": 171})).writeto(raw)
    view = tmp_path / "output" / "views"
    view.mkdir(parents=True)
    _require_symlink_support(raw, view / "probe")

    rows = stereo_manifest.build_manifest(
        [raw], manifest_path=tmp_path / "manifest.csv", link_root=view
    )
    link = view / "171" / raw.name
    assert rows[0]["path"] == str(raw)
    assert link.resolve(strict=True) == raw.resolve(strict=True)
    assert link.read_bytes() == raw.read_bytes()
    stereo_manifest.make_relative_symlink(raw, link)
    assert link.resolve(strict=True) == raw.resolve(strict=True)


def test_manifest_link_does_not_replace_an_ordinary_file(tmp_path):
    source = tmp_path / "source.fts"
    source.write_bytes(b"synthetic source")
    link = tmp_path / "ordinary.fts"
    link.write_bytes(b"preserve original")
    with pytest.raises(FileExistsError, match="non-symlink"):
        stereo_manifest.make_relative_symlink(source, link)
    assert link.read_bytes() == b"preserve original"


def test_manifest_link_uses_absolute_target_when_drives_have_no_relative_path(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.fts"
    source.write_bytes(b"synthetic source")
    captured = []

    def no_relative_path(*_args, **_kwargs):
        raise ValueError("different drives")

    monkeypatch.setattr(stereo_manifest.os.path, "relpath", no_relative_path)
    monkeypatch.setattr(
        Path, "symlink_to", lambda _link, target: captured.append(target)
    )
    stereo_manifest.make_relative_symlink(source, tmp_path / "view.fts")
    assert captured == [source.resolve()]
