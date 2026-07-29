from __future__ import annotations

import sys
from pathlib import Path


def test_configure_chinese_fonts_returns_selected_font(monkeypatch):
    from solar_toolkit.visualization import configure_chinese_fonts

    class Font:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(
        "matplotlib.font_manager.fontManager.ttflist",
        [Font("Microsoft YaHei"), Font("DejaVu Sans")],
    )

    selected = configure_chinese_fonts(["SimHei", "Microsoft YaHei"])

    assert selected == "Microsoft YaHei"


def test_plotting_font_setup_uses_installed_font_selector(monkeypatch):
    from solar_toolkit.visualization import plotting

    calls: list[object] = []
    monkeypatch.setattr(
        "solar_toolkit.visualization.configure_chinese_fonts",
        lambda: calls.append(True),
    )

    plotting.setup_chinese_font()

    assert calls == [True]


def test_media_fallback_uses_the_running_python_environment():
    from solar_toolkit.visualization import media

    expected_bin = Path(sys.executable).resolve().parent / "Library" / "bin"

    assert media.CONDA_BIN == expected_bin
    assert media.CONDA_FFMPEG == expected_bin / "ffmpeg.exe"
    assert media.CONDA_FFPROBE == expected_bin / "ffprobe.exe"


def test_media_fallback_tracks_the_selected_python_environment(tmp_path):
    from solar_toolkit.visualization import media

    primary_python = tmp_path / "primary" / "python.exe"
    backup_python = tmp_path / "backup" / "python.exe"

    primary_bin = media._conda_bin_for_python(primary_python)
    backup_bin = media._conda_bin_for_python(backup_python)

    assert primary_bin == primary_python.resolve().parent / "Library" / "bin"
    assert backup_bin == backup_python.resolve().parent / "Library" / "bin"
    assert primary_bin != backup_bin
