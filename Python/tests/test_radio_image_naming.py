"""Radio workflow coverage for the shared image naming contract."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from solar_toolkit.radio._image_naming import build_radio_image_filename
from solar_toolkit.radio.drift_products import save_drift_selection_artifacts
from solar_toolkit.radio.roi_lightcurve import build_radio_roi_product_filenames


def _radio_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "obs_time": [
                "2025-01-24T04:48:31.900Z",
                "2025-01-24T04:48:30.100Z",
            ],
            "freq_mhz": [223.5, 223.5],
            "polarization": ["LL", "RR"],
        }
    )


def test_radio_filename_derives_sorted_range_frequency_and_polarization() -> None:
    name = build_radio_image_filename(
        _radio_rows(),
        sequence=2,
        product="source_map",
        generated_at=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
    )
    assert name == (
        "0002_20250124T044830Z-20250124T044831Z_"
        "radio_223p5mhz_lcp_plus_rcp_source_map.png"
    )


def test_roi_product_filenames_sequence_only_selected_images() -> None:
    names = build_radio_roi_product_filenames(
        _radio_rows(),
        selected_products=("csv", "lightcurve_png", "lightcurve_detail_png"),
        generated_at=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
    )
    assert names["csv"] == "radio_roi_statistics.csv"
    assert names["lightcurve_png"].startswith("0001_20250124T044830Z-")
    assert names["lightcurve_detail_png"].startswith("0002_20250124T044830Z-")


def test_drift_products_reference_actual_dynamic_names(tmp_path: Path) -> None:
    times = [
        dt.datetime(2025, 1, 24, 4, 48, 30),
        dt.datetime(2025, 1, 24, 4, 48, 31),
    ]
    result = save_drift_selection_artifacts(
        np.arange(8, dtype=float).reshape(4, 2),
        times,
        np.array([100.0, 110.0, 120.0, 130.0]),
        [],
        tmp_path,
        config={
            "save_selection_csv": False,
            "save_per_drift_cutouts": False,
            "preserve_existing": False,
            "dpi": 50,
        },
    )
    raw = Path(result["raw_preview_png"])
    annotated = Path(result["annotated_preview_png"])
    assert raw.name.startswith("0001_20250124T044830Z-20250124T044831Z_radio_")
    assert raw.name.endswith("_dynamic_spectrum_raw.png")
    assert annotated.name.startswith("0002_20250124T044830Z-")
    metadata = json.loads(Path(result["metadata_json"]).read_text(encoding="utf-8"))
    assert metadata["raw_preview_png"] == raw.name
    assert metadata["annotated_preview_png"] == annotated.name
    assert not (tmp_path / "spectrogram_drift_rate_selection_preview_raw.png").exists()
