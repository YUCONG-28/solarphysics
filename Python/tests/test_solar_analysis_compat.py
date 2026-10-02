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
        "data_dir": "D:/solar_data",
        "output_dir": "D:/solar_data/output",
        "roi_bounds": (-700, -100, -100, 400),
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
    expected["roi_bounds"] = list(expected["roi_bounds"])
    assert reloaded == expected
