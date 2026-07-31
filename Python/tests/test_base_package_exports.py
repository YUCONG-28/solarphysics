"""Public API contracts for the small foundation packages."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    ("package_name", "exports"),
    [
        (
            "solar_toolkit.time",
            {
                "extract_time_from_filename": "solar_toolkit.time.parsing",
                "filter_by_time_range": "solar_toolkit.time.selection",
                "nearest_by_time": "solar_toolkit.time.selection",
                "parse_time": "solar_toolkit.time.parsing",
            },
        ),
        (
            "solar_toolkit.io",
            {
                "natural_key": "solar_toolkit.io.sorting",
                "read_fits_data_header": "solar_toolkit.io.fits",
                "read_manifest": "solar_toolkit.io.manifest",
                "scan_files": "solar_toolkit.io.discovery",
                "scan_fits": "solar_toolkit.io.discovery",
                "write_manifest": "solar_toolkit.io.manifest",
            },
        ),
        (
            "solar_toolkit.data",
            {
                "ObservationFile": "solar_toolkit.data.inventory",
                "build_inventory": "solar_toolkit.data.inventory",
                "inventory": "solar_toolkit.data.inventory",
                "stereo_manifest": "solar_toolkit.data.stereo_manifest",
            },
        ),
        (
            "solar_toolkit.map",
            {
                "crop_roi": "solar_toolkit.map.image",
                "get_display_extent": "solar_toolkit.map.metadata",
                "get_map_obs_time": "solar_toolkit.map.metadata",
                "normalize_image": "solar_toolkit.map.image",
            },
        ),
        (
            "solar_toolkit.timeseries",
            {
                "crop_time_range": "solar_toolkit.timeseries.tables",
                "derivative_series": "solar_toolkit.timeseries.processing",
                "normalize_time_column": "solar_toolkit.timeseries.tables",
                "smooth_series": "solar_toolkit.timeseries.processing",
            },
        ),
        (
            "solar_toolkit.net",
            {
                "DownloadResult": "solar_toolkit.net.downloads",
                "collect_links": "solar_toolkit.net.links",
                "download_url": "solar_toolkit.net.downloads",
                "filter_links": "solar_toolkit.net.links",
                "downloads": "solar_toolkit.net.downloads",
                "jsoc": "solar_toolkit.net.jsoc",
                "links": "solar_toolkit.net.links",
                "observations": "solar_toolkit.net.observations",
                "soar": "solar_toolkit.net.soar",
                "stereo": "solar_toolkit.net.stereo",
                "suvi": "solar_toolkit.net.suvi",
            },
        ),
        (
            "solar_toolkit.cme",
            {
                "extract_lasco_timestamp": "solar_toolkit.cme.files",
                "running_difference": "solar_toolkit.cme.processing",
                "scan_lasco_files": "solar_toolkit.cme.files",
            },
        ),
        (
            "solar_toolkit.xray_dem",
            {
                "calculate_derivative": "solar_toolkit.xray_dem.processing",
                "load_sxr_data": "solar_toolkit.xray_dem.sxr",
                "smooth_flux_data": "solar_toolkit.xray_dem.processing",
                "hxi": "solar_toolkit.xray_dem.hxi",
                "processing": "solar_toolkit.xray_dem.processing",
                "sxr": "solar_toolkit.xray_dem.sxr",
            },
        ),
    ],
)
def test_base_namespace_reexports_canonical_symbols(package_name, exports):
    package = importlib.import_module(package_name)

    assert package.__all__ == list(exports)
    for name, module_name in exports.items():
        implementation = importlib.import_module(module_name)
        if module_name == f"{package_name}.{name}":
            assert getattr(package, name) is implementation
        else:
            assert getattr(package, name) is getattr(implementation, name)


@pytest.mark.parametrize(
    "module_name",
    [
        "solar_toolkit.time.parsing",
        "solar_toolkit.time.selection",
        "solar_toolkit.io.sorting",
        "solar_toolkit.io.discovery",
        "solar_toolkit.io.fits",
        "solar_toolkit.io.manifest",
        "solar_toolkit.data.inventory",
        "solar_toolkit.data.stereo_manifest",
        "solar_toolkit.map.metadata",
        "solar_toolkit.map.image",
        "solar_toolkit.timeseries.tables",
        "solar_toolkit.timeseries.processing",
        "solar_toolkit.net.links",
        "solar_toolkit.net.downloads",
        "solar_toolkit.net.jsoc",
        "solar_toolkit.cme.files",
        "solar_toolkit.cme.processing",
        "solar_toolkit.xray_dem.sxr",
        "solar_toolkit.xray_dem.processing",
    ],
)
def test_base_implementation_modules_have_explicit_exports(module_name):
    module = importlib.import_module(module_name)

    assert isinstance(module.__all__, list)
    assert module.__all__
    assert all(hasattr(module, name) for name in module.__all__)
