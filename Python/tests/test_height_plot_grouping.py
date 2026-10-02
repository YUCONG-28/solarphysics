from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from solar_toolkit.radio.height_comparison import model_label
from solar_toolkit.radio.height_plots import plot_gaussian_vs_newkirk_height_time


def test_height_time_plot_does_not_connect_raw_multifrequency_rows():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "height_time.png"
        result = plot_gaussian_vs_newkirk_height_time(
            pd.DataFrame(
                [
                    _height_row("2000-01-01T04:48:30", 238.0, 0.10, 0.20),
                    _height_row("2000-01-01T04:48:30", 149.0, 0.12, 0.35),
                    _height_row("2000-01-01T04:48:31", 205.0, 0.13, 0.25),
                ]
            ),
            path,
            {"connect_raw_points": False},
        )

        assert result["status"] == "saved"
        assert result["connected_line_count"] == 0
        assert path.exists()


def test_empty_height_time_input_does_not_save_blank_plot():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "empty_height_time.png"
        result = plot_gaussian_vs_newkirk_height_time(pd.DataFrame(), path)

        assert result["status"] == "skipped"
        assert not path.exists()


def test_model_label_uses_multiplication_sign():
    assert model_label(2.0, 2) == "2× Newkirk, s=2"


def _height_row(time, freq, gaussian_height, newkirk_height):
    return {
        "time": time,
        "frequency_mhz": freq,
        "source_type": "unknown",
        "gaussian_x_arcsec": 1000.0,
        "gaussian_y_arcsec": 0.0,
        "gaussian_height_rsun": gaussian_height,
        "newkirk_multiplier": 2.0,
        "harmonic": 2,
        "newkirk_height_rsun": newkirk_height,
        "height_residual_rsun": gaussian_height - newkirk_height,
        "height_valid": True,
    }


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
