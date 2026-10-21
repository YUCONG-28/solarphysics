"""Compatibility contracts for the deprecated shared-utility module."""

from __future__ import annotations

import ast
import datetime as dt
import importlib
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from solar_toolkit.exceptions import SolarToolkitDeprecationWarning

CANONICAL_EXPORTS = {
    "SolarDataConfig": "solar_toolkit._utils.validation",
    "SolarLogger": "solar_toolkit._utils.logging",
    "add_frequency_highlight_lines": "solar_toolkit.visualization.plotting",
    "align_maps_to_reference": "solar_toolkit.map.operations",
    "create_aia_submap": "solar_toolkit.map.operations",
    "create_figure_with_white_background": "solar_toolkit.visualization.plotting",
    "create_magnetic_contour_levels": "solar_toolkit.hmi.processing",
    "extract_time_from_filename": "solar_toolkit.time.parsing",
    "filter_files_by_time_range": "solar_toolkit.io.discovery",
    "find_closest_file_by_time": "solar_toolkit.io.discovery",
    "format_time_for_display": "solar_toolkit.time.formatting",
    "format_time_for_filename": "solar_toolkit.time.formatting",
    "get_aia_wavelength_config": "solar_toolkit.visualization.plotting",
    "get_sorted_fits_files": "solar_toolkit.io.discovery",
    "monitor_memory_usage": "solar_toolkit._utils.memory",
    "normalize_aia_exposure": "solar_toolkit.map.operations",
    "optimized_gc_collect": "solar_toolkit._utils.memory",
    "parse_isot_time": "solar_toolkit.time.parsing",
    "process_hmi_magnetic_field": "solar_toolkit.hmi.processing",
    "safe_delete": "solar_toolkit._utils.memory",
    "setup_chinese_font": "solar_toolkit.visualization.plotting",
    "timing_decorator": "solar_toolkit._utils.logging",
    "validate_frequency_range": "solar_toolkit._utils.validation",
    "validate_time_range": "solar_toolkit._utils.validation",
}


@pytest.fixture(scope="module")
def compatibility_module():
    import solar_toolkit

    sys.modules.pop("solar_toolkit.solar_analysis_utils", None)
    vars(solar_toolkit).pop("solar_analysis_utils", None)
    with pytest.warns(
        SolarToolkitDeprecationWarning,
        match=r"deprecated since .*0\.2\.0.*1\.0\.0",
    ):
        return importlib.import_module("solar_toolkit.solar_analysis_utils")


def test_compatibility_module_is_only_a_forwarding_layer(compatibility_module):
    source_path = Path(compatibility_module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in tree.body
    )
    assert set(compatibility_module.__all__) == set(CANONICAL_EXPORTS)


@pytest.mark.parametrize(("name", "module_name"), CANONICAL_EXPORTS.items())
def test_legacy_names_are_canonical_objects(
    compatibility_module,
    name,
    module_name,
):
    canonical_module = importlib.import_module(module_name)

    assert getattr(compatibility_module, name) is getattr(canonical_module, name)


def test_compatibility_import_remains_lightweight():
    code = """
import sys
import warnings
from solar_toolkit.exceptions import SolarToolkitDeprecationWarning

warnings.simplefilter("ignore", SolarToolkitDeprecationWarning)
import solar_toolkit.solar_analysis_utils

assert "sunpy.map" not in sys.modules
assert "matplotlib.pyplot" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_legacy_time_and_file_helpers_keep_defaults(compatibility_module, tmp_path):
    first = tmp_path / "aia_2000-01-01T044800Z.fits"
    second = tmp_path / "aia_2000-01-01T044900Z.fits"
    invalid = tmp_path / "invalid.fits"
    first.write_bytes(b"x" * 2048)
    second.write_bytes(b"x" * 2048)
    invalid.write_bytes(b"x" * 2048)

    with pytest.warns(UserWarning, match="invalid timestamp"):
        files = compatibility_module.get_sorted_fits_files(str(tmp_path))

    assert [path for path, _ in files] == [first, second]
    assert (
        compatibility_module.find_closest_file_by_time(
            dt.datetime(2000, 1, 1, 4, 48, 40),
            files,
        )
        == files[1]
    )
    assert (
        compatibility_module.find_closest_file_by_time(
            dt.datetime(2000, 1, 1, 5, 48, 40),
            files,
            max_diff_seconds=10,
        )
        is None
    )
    assert (
        compatibility_module.filter_files_by_time_range(
            files,
            dt.datetime(2000, 1, 1, 4, 48, 30),
            dt.datetime(2000, 1, 1, 4, 50),
        )
        == files[1:]
    )
    assert compatibility_module.extract_time_from_filename(
        "context_2000-01-01.png"
    ) == dt.datetime(2000, 1, 1)
    parsed = compatibility_module.parse_isot_time("2000-01-01T04:48:00")
    assert compatibility_module.format_time_for_display(parsed) == "2000-01-01 04:48:00"
    assert compatibility_module.format_time_for_filename(parsed) == "20000101_044800"


def test_legacy_science_and_config_defaults_are_preserved(
    compatibility_module, tmp_path
):
    config = compatibility_module.SolarDataConfig()
    assert config.to_dict() == {
        "data_dir": None,
        "output_dir": None,
        "roi_bounds": None,
        "dpi": 300,
        "fig_width": 10.0,
        "use_parallel": True,
        "max_workers": None,
        "chunk_mem_mb": 50,
        "save_images": True,
        "show_images": False,
    }
    assert compatibility_module.get_aia_wavelength_config(171) == {
        "cmap": "sdoaia171",
        "vmin": 16,
        "vmax": 6666,
    }
    assert compatibility_module.get_aia_wavelength_config(999) == {
        "cmap": "sdoaia94",
        "vmin": 1.0,
        "vmax": 1e4,
    }
    assert (
        inspect.signature(compatibility_module.process_hmi_magnetic_field)
        .parameters["sigma"]
        .default
        == 3
    )
    assert (
        inspect.signature(compatibility_module.create_magnetic_contour_levels)
        .parameters["base_level"]
        .default
        is None
    )
    assert compatibility_module.validate_time_range(
        dt.datetime(2000, 1, 1),
        dt.datetime(2000, 1, 2),
    )
    assert compatibility_module.validate_frequency_range(100.0, 200.0)

    config_path = tmp_path / "config.json"
    config.save_to_file(str(config_path))
    reloaded = compatibility_module.SolarDataConfig.load_from_file(
        str(config_path)
    ).to_dict()
    expected = config.to_dict()
    assert reloaded == expected


@pytest.mark.parametrize("suffix", ["yaml", "yml"])
def test_default_configuration_yaml_round_trips_safely(
    compatibility_module, tmp_path, suffix
):
    import yaml

    config = compatibility_module.SolarDataConfig()
    path = tmp_path / f"config.{suffix}"
    config.save_to_file(str(path))
    # Portable YAML must be readable by SafeLoader without Python-specific tags.
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = config.to_dict()
    assert parsed == expected
    assert (
        compatibility_module.SolarDataConfig.load_from_file(str(path)).to_dict()
        == expected
    )


@pytest.mark.parametrize("suffix", ["json", "yaml", "yml"])
def test_configuration_preserves_explicit_paths_and_portable_roi(
    compatibility_module, tmp_path, suffix
):
    supplied = {
        "data_dir": "synthetic/input",
        "output_dir": "synthetic/output",
        "roi_bounds": (-1, 1, -2, 2),
    }
    config = compatibility_module.SolarDataConfig(supplied)
    assert all(config.to_dict()[key] == value for key, value in supplied.items())
    path = tmp_path / f"explicit.{suffix}"
    config.save_to_file(str(path))
    restored = compatibility_module.SolarDataConfig.load_from_file(str(path)).to_dict()
    expected = config.to_dict()
    expected["roi_bounds"] = list(supplied["roi_bounds"])
    assert restored == expected
    assert supplied["roi_bounds"] == (-1, 1, -2, 2)


def test_time_validation_compares_mixed_utc_representations(compatibility_module):
    start = dt.datetime(2000, 1, 1)
    end = dt.datetime(2000, 1, 1, 8, 0, 1, tzinfo=dt.timezone(dt.timedelta(hours=8)))
    assert compatibility_module.validate_time_range(start, end)
    with pytest.raises(ValueError, match="must precede"):
        compatibility_module.validate_time_range(start, end - dt.timedelta(seconds=1))
    with pytest.raises(ValueError, match="must precede"):
        compatibility_module.validate_time_range(end, start)


def test_time_validation_preserves_tai_scale_and_nanosecond_order(compatibility_module):
    from astropy.time import Time

    start = Time("2000-01-01T00:00:00.000000001", scale="utc")
    end = Time("2000-01-01T00:00:00.000000002", scale="utc").tai
    assert compatibility_module.validate_time_range(start, end)
    with pytest.raises(ValueError, match="must precede"):
        compatibility_module.validate_time_range(start, start.tai)


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (float("nan"), 200),
        (100, float("nan")),
        (100, float("inf")),
        (-float("inf"), 200),
        (0, 200),
        (-1, 200),
        (True, 200),
        (None, 200),
    ],
)
def test_frequency_validation_rejects_non_finite_or_non_positive_values(
    compatibility_module, start, end
):
    with pytest.raises(ValueError, match="finite positive"):
        compatibility_module.validate_frequency_range(start, end)
