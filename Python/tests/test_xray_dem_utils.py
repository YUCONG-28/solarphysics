from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from solar_toolkit.xray_dem.sxr import load_goes_sxr_dataset, load_sxr_data


def test_load_sxr_data_smooth_and_derivative(tmp_path):
    from solar_toolkit.xray_dem import (
        calculate_derivative,
        load_sxr_data,
        smooth_flux_data,
    )

    csv_path = tmp_path / "goes.csv"
    pd.DataFrame(
        {
            "time": [
                "2000-01-01T04:48:00Z",
                "2000-01-01T04:49:00Z",
                "2000-01-01T04:50:00Z",
            ],
            "xrsa": [1.0, 3.0, 5.0],
        }
    ).to_csv(csv_path, index=False)

    loaded = load_sxr_data(csv_path, "2000-01-01T04:48:30Z", "2000-01-01T04:50:30Z")
    smoothed = smooth_flux_data(pd.Series([1.0, 3.0, 5.0]), window_length=3)
    derivative = calculate_derivative(pd.Series([1.0, 3.0, 5.0]), spacing_seconds=60)

    assert loaded["xrsa"].tolist() == [3.0, 5.0]
    assert np.allclose(smoothed, [4 / 3, 3.0, 8 / 3])
    assert np.allclose(derivative, [1 / 30, 1 / 30, 1 / 30])


@pytest.fixture(params=["csv", "txt", "nc"])
def synthetic_sxr_input(request, tmp_path):
    """Use the same unsorted, duplicate-containing UTC samples for each format."""
    times = pd.to_datetime(
        [
            "2000-01-01T00:00:02Z",
            "2000-01-01T00:00:00Z",
            "2000-01-01T00:00:01Z",
            "2000-01-01T00:00:01Z",
        ],
        utc=True,
    )
    values = [3.0, 1.0, 2.0, 4.0]
    path = tmp_path / f"synthetic_sxr.{request.param}"
    if request.param == "nc":
        xr = pytest.importorskip("xarray")
        xr.Dataset(
            {"flux": ("time", values)},
            coords={"time": times.tz_convert(None).to_numpy()},
        ).to_netcdf(path, engine="scipy")
    else:
        pd.DataFrame({"time": times, "flux": values}).to_csv(path, index=False)
    return path


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (None, None, [3.0, 1.0, 2.0, 4.0]),
        ("2000-01-01T00:00:01Z", None, [3.0, 2.0, 4.0]),
        (None, "2000-01-01T00:00:01", [1.0, 2.0, 4.0]),
        ("2000-01-01T00:00:01Z", "2000-01-01T00:00:01Z", [2.0, 4.0]),
        ("2000-01-01T01:00:01+01:00", "2000-01-01T00:00:02Z", [3.0, 2.0, 4.0]),
        ("2000-01-01T00:00:03Z", None, []),
    ],
)
def test_sxr_formats_share_inclusive_optional_utc_bounds(
    synthetic_sxr_input, start, end, expected
):
    result = load_sxr_data(synthetic_sxr_input, start, end)

    if synthetic_sxr_input.suffix == ".nc":
        import xarray as xr

        assert isinstance(result, xr.Dataset)
        values = result["flux"].values.tolist()
    else:
        assert isinstance(result, pd.DataFrame)
        assert result["obs_time"].dt.tz is None
        values = result["flux"].tolist()
    assert values == expected


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2000-01-01T00:00:02Z", "2000-01-01T00:00:01Z"),
        ("NaT", None),
        (None, pd.NaT),
    ],
)
def test_sxr_formats_reject_invalid_bounds(synthetic_sxr_input, start, end):
    with pytest.raises(ValueError):
        load_sxr_data(synthetic_sxr_input, start, end)


def test_netcdf_result_is_detached_and_empty_window_is_explicit(tmp_path):
    xr = pytest.importorskip("xarray")
    path = tmp_path / "synthetic_sxr.nc"
    xr.Dataset(
        {"flux": ("time", [1.0, 2.0])},
        coords={"time": pd.date_range("2000-01-01", periods=2, freq="s")},
    ).to_netcdf(path, engine="scipy")

    result = load_goes_sxr_dataset(path, end_time="2000-01-01T00:00:00Z")
    empty = load_goes_sxr_dataset(path, start_time="2000-01-01T00:00:02Z")
    assert isinstance(empty, xr.Dataset)
    assert empty.sizes["time"] == 0
    with pytest.raises(ValueError, match="No SXR samples"):
        load_goes_sxr_dataset(
            path, start_time="2000-01-01T00:00:02Z", require_data=True
        )
    path.unlink()
    np.testing.assert_array_equal(result["flux"].values, [1.0])


def test_sxr_table_custom_time_column_uses_utc_bound(tmp_path):
    path = tmp_path / "synthetic_sxr.csv"
    pd.DataFrame(
        {
            "stamp": ["2000-01-01T00:00:00Z", "2000-01-01T00:00:01Z"],
            "flux": [1.0, 2.0],
        }
    ).to_csv(path, index=False)

    result = load_sxr_data(path, "2000-01-01T01:00:01+01:00", time_column="stamp")
    assert result["flux"].tolist() == [2.0]
    assert result["obs_time"].iloc[0] == pd.Timestamp("2000-01-01T00:00:01")


def test_netcdf_reader_reports_missing_optional_dependency(tmp_path, monkeypatch):
    import sys

    path = tmp_path / "synthetic_sxr.nc"
    path.touch()
    monkeypatch.setitem(sys.modules, "xarray", None)
    with pytest.raises(ImportError, match="optional 'xarray' dependency"):
        load_goes_sxr_dataset(path)


@pytest.mark.parametrize(
    ("option", "first", "last"),
    [("--start-time", "00:00:40", "00:01:20"), ("--end-time", "00:00:00", "00:00:40")],
)
def test_sxr_example_main_accepts_one_sided_window(tmp_path, option, first, last):
    script = Path(__file__).resolve().parents[1] / "examples/sxr/sxr_example.py"
    namespace = runpy.run_path(str(script))
    main = namespace["main"]
    loaded = []

    def capture_loaded_window(*args, **kwargs):
        frame = load_sxr_data(*args, **kwargs)
        loaded.append(frame)
        return frame

    main.__globals__["load_sxr_data"] = capture_loaded_window
    output = tmp_path / "example_output"
    result = main(["--output-dir", str(output), option, "2000-01-01T00:00:40Z"])

    assert result == 0
    assert len(loaded) == 1
    assert len(loaded[0]) == 41
    assert loaded[0]["obs_time"].iloc[0] == pd.Timestamp(f"2000-01-01T{first}")
    assert loaded[0]["obs_time"].iloc[-1] == pd.Timestamp(f"2000-01-01T{last}")
    assert (output / "synthetic_sxr.csv").is_file()
    assert (output / "sxr_example.png").stat().st_size > 0
