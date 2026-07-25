from __future__ import annotations

from pathlib import Path

from solar_toolkit.aia._euv_processor_impl import (
    _difference_save_dir,
    _mosaic_save_dir,
)
from solar_toolkit.aia.config import AIAConfig


def test_explicit_output_root_routes_mosaic_without_changing_input() -> None:
    config = AIAConfig(
        data_path="observations/aia",
        output_dir="private/outputs/aia",
    )

    save_dir = _mosaic_save_dir(config)
    assert save_dir.is_relative_to(Path("private/outputs/aia"))
    assert not save_dir.is_relative_to(Path("observations/aia"))


def test_explicit_output_root_routes_per_band_differences() -> None:
    config = AIAConfig(
        data_path="observations/aia",
        output_dir="private/outputs/aia",
        use_band_subdirs=True,
    )

    assert _difference_save_dir(Path(config.data_path), 171, config) == (
        Path("private/outputs/aia")
        / "171"
        / config.difference_output_subdir
        / f"{config.difference_method}_difference"
    )


def test_default_output_root_preserves_historical_data_location() -> None:
    config = AIAConfig(data_path="observations/aia")

    assert Path(config.output_dir) == Path(config.data_path)
    assert _mosaic_save_dir(config).is_relative_to(Path(config.data_path))
