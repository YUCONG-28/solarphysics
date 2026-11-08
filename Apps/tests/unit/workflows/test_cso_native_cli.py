"""Synthetic native-CSO CLI, gap rendering and transactional output contracts."""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from solar_apps.workflows.radio import cso_native, cso_workflow, dispatcher
from solar_toolkit.time import normalize_time


def _write_cso(tmp_path, *, compressed=False, offsets=None):
    offsets = np.array([0, 1, 2, 7, 8] if offsets is None else offsets, dtype=float)
    frequency = np.array([100, 102, 104, 106], dtype=float)
    right = np.tile([9, 1, 6, 6], (len(offsets), 1)).T.astype(float)
    left = np.tile([1, 9, 2, 2], (len(offsets), 1)).T.astype(float)
    left[0, 1] = np.nan  # One unpaired frequency must not enter either channel.
    primary = fits.PrimaryHDU(np.stack((right, left)))
    primary.header.update(
        {
            "DATE-OBS": "2000-01-01T00:00:00",
            "DATE-END": normalize_time("2000-01-01T00:00:00")
            .shift(float(offsets[-1]))
            .utc,
            "TIMESYS": "UTC",
            "BUNIT": "SFU",
            "POLARIZA": "RCP and LCP",
        }
    )
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name="time", format=f"{len(offsets)}D", unit="s", array=[offsets]
            ),
            fits.Column(name="frequency", format="4D", unit="MHz", array=[frequency]),
        ]
    )
    path = tmp_path / "synthetic.fits"
    fits.HDUList([primary, table]).writeto(path)
    if compressed:
        packed = Path(str(path) + ".gz")
        with gzip.open(packed, "wb") as stream:
            stream.write(path.read_bytes())
        path.unlink()
        return packed
    return path


def _arguments(path, output):
    return [
        "--native-polarization",
        "--file-path",
        str(path),
        "--time-start",
        "1999-12-31T23:59:59",
        "--time-end",
        "2000-01-01T00:00:10",
        "--frequency-start",
        "100",
        "--frequency-end",
        "108",
        "--frequency-bin-mhz",
        "4",
        "--output-dir",
        str(output),
        "--chunk-columns",
        "2",
    ]


def _fake_science(monkeypatch):
    from solar_toolkit.radio import cso_extract

    base = 946684800.0
    arrays = {
        "intensity": np.array([[10.0, 12.0, 14.0]]),
        "polarization_percent": np.array([[20.0, 30.0, 40.0]]),
        "time_tai_ns": np.array([1, 2, 3], dtype=np.int64),
        "time_unix": base + np.array([0, 1, 8]),
        "frequency_edges_mhz": np.array([100, 104]),
        "sample_count": np.ones((1, 3), dtype=np.int64),
    }
    metadata = {
        "requested_time_bounds_unix": [base - 1, base + 10],
        "segments": [
            {
                "start_index": 0,
                "stop_index": 2,
                "time_edges_unix": [base - 0.5, base + 0.5, base + 1.5],
            },
            {
                "start_index": 2,
                "stop_index": 3,
                "time_edges_unix": [base + 7.5, base + 8.5],
            },
        ],
        "missing_support_unix": [
            [base - 1, base - 0.5],
            [base + 1.5, base + 7.5],
            [base + 8.5, base + 10],
        ],
    }
    base_ns = normalize_time("2000-01-01T00:00:00").tai_ns
    metadata["requested_time_bounds_tai_ns"] = [base_ns - 10**9, base_ns + 10 * 10**9]
    metadata["requested_time_bounds_utc"] = [
        "1999-12-31T23:59:59.000000000",
        "2000-01-01T00:00:10.000000000",
    ]
    for segment in metadata["segments"]:
        segment["time_edges_tai_ns"] = [
            base_ns + round((time - base) * 1e9) for time in segment["time_edges_unix"]
        ]
    metadata["missing_support_tai_ns"] = [
        [base_ns + round((time - base) * 1e9) for time in pair]
        for pair in metadata["missing_support_unix"]
    ]
    calls = []

    def extract(*args, **kwargs):
        calls.append((args, kwargs))
        return arrays, metadata

    monkeypatch.setattr(cso_extract, "extract_native_polarization", extract)
    return arrays, metadata, calls


def test_registered_workflow_and_help_do_not_initialize_configuration(
    monkeypatch, capsys
):
    assert dispatcher._COMMANDS["cso"] == (
        "solar_apps.workflows.radio.cso_workflow",
        "main",
    )
    monkeypatch.setattr(
        cso_workflow, "PlotConfig", lambda: pytest.fail("loaded config")
    )
    with pytest.raises(SystemExit) as stopped:
        dispatcher.main(["cso", "--help"])
    assert stopped.value.code == 0
    assert "--native-polarization" in capsys.readouterr().out


@pytest.mark.parametrize("extra", [[], ["--unknown"], ["--rebin-time", "2"]])
def test_native_mode_rejects_incomplete_or_inapplicable_options_without_config(
    extra, monkeypatch
):
    monkeypatch.setattr(
        cso_workflow, "PlotConfig", lambda: pytest.fail("loaded config")
    )
    with pytest.raises(SystemExit) as stopped:
        cso_workflow.main(["--native-polarization", *extra])
    assert stopped.value.code == 2


@pytest.mark.parametrize("storage", ["plain", "gzip", "fallback"])
def test_actual_native_cli_preserves_times_pairs_and_gaps(tmp_path, storage):
    path = _write_cso(tmp_path, compressed=storage != "plain")
    supplied_path = Path(str(path)[:-3]) if storage == "fallback" else path
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    output = tmp_path / "delivered"
    assert dispatcher.main(["cso", *_arguments(supplied_path, output)]) == 0
    assert {p.name for p in output.iterdir()} == {
        "native_polarization.npz",
        "native_polarization.json",
        "native_polarization.png",
    }
    with np.load(output / "native_polarization.npz", allow_pickle=False) as values:
        expected = (
            normalize_time("2000-01-01T00:00:00").tai_ns
            + np.array([0, 1, 2, 7, 8], dtype=np.int64) * 10**9
        )
        np.testing.assert_array_equal(values["time_tai_ns"], expected)
        np.testing.assert_array_equal(values["sample_count"][0], [2, 1, 2, 2, 2])
        np.testing.assert_allclose(values["polarization_percent"][0], [0, -80, 0, 0, 0])
        np.testing.assert_allclose(values["polarization_percent"][1], 50)
        np.testing.assert_allclose(values["intensity"][1], 8)
    report = json.loads((output / "native_polarization.json").read_text())
    assert len(report["segments"]) == report["rendering"]["segments_drawn"] == 2
    assert report["missing_support_unix"]
    assert (output / "native_polarization.png").read_bytes().startswith(b"\x89PNG")
    for name, recorded in report["outputs"].items():
        assert (
            hashlib.sha256((output / name).read_bytes()).hexdigest()
            == recorded["sha256"]
        )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original
    assert not list(tmp_path.glob(".delivered.stage-*"))


def test_export_delegates_all_scientific_inputs_and_does_not_modify_arrays(
    tmp_path, monkeypatch
):
    arrays, metadata, calls = _fake_science(monkeypatch)
    before = {key: value.copy() for key, value in arrays.items()}
    output = tmp_path / "empty"
    output.mkdir()
    cso_native.export_native_polarization(
        ["one.fits", "two.fits.gz"],
        ("start", "end"),
        (100, 104),
        frequency_bin_mhz=4,
        output_dir=output,
        cache_root="private-cache",
        chunk_columns=7,
        gap_factor=2,
    )
    assert calls == [
        (
            (["one.fits", "two.fits.gz"], ("start", "end"), (100, 104)),
            {
                "frequency_bin_mhz": 4,
                "cache_root": "private-cache",
                "chunk_columns": 7,
                "gap_factor": 2,
            },
        )
    ]
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], before[key])
    assert "outputs" not in metadata and "rendering" not in metadata


def test_render_uses_a_separate_mesh_for_each_support_segment(tmp_path, monkeypatch):
    from matplotlib.axes import Axes

    arrays, metadata, _ = _fake_science(monkeypatch)
    calls = []
    original = Axes.pcolormesh

    def record(self, *args, **kwargs):
        if kwargs.get("rasterized") is True:
            calls.append((np.asarray(args[0]), np.asarray(args[2])))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "pcolormesh", record)
    cso_native._render(arrays, metadata, tmp_path / "image.png")
    assert len(calls) == 4
    assert [data.shape for _, data in calls] == [(1, 2), (1, 1), (1, 2), (1, 1)]
    assert calls[0][0][-1] < calls[1][0][0]
    assert calls[2][0][-1] < calls[3][0][0]


def test_empty_window_exports_missing_support_without_fabricated_samples(tmp_path):
    path = _write_cso(tmp_path)
    output = tmp_path / "empty-window"
    args = _arguments(path, output)
    args[args.index("--time-start") + 1] = "2000-01-01T00:00:20"
    args[args.index("--time-end") + 1] = "2000-01-01T00:00:30"
    assert cso_workflow.main(args) == 0
    with np.load(output / "native_polarization.npz", allow_pickle=False) as values:
        assert values["intensity"].shape == (2, 0)
        assert values["time_tai_ns"].size == 0
    report = json.loads((output / "native_polarization.json").read_text())
    assert report["segments"] == []
    assert report["rendering"]["segments_drawn"] == 0
    assert report["missing_support_unix"] == [report["requested_time_bounds_unix"]]


def test_all_missing_intervals_are_drawn_without_cross_gap_interpolation(
    tmp_path, monkeypatch
):
    from matplotlib.axes import Axes

    arrays, metadata, _ = _fake_science(monkeypatch)
    spans = []
    original = Axes.axvspan

    def record(self, left, right, **kwargs):
        spans.append((left, right, kwargs.get("hatch")))
        return original(self, left, right, **kwargs)

    monkeypatch.setattr(Axes, "axvspan", record)
    cso_native._render(arrays, metadata, tmp_path / "missing.png")
    expected = [
        (
            *cso_native._plot_offsets(
                pair, metadata["requested_time_bounds_tai_ns"][0]
            ),
            "///",
        )
        for pair in metadata["missing_support_tai_ns"]
    ]
    assert spans == expected * 2


@pytest.mark.parametrize("cadence", [0.01, 1e-8])
def test_actual_short_window_png_has_distinct_fractional_ticks(
    tmp_path, monkeypatch, cadence
):
    from matplotlib.figure import Figure

    offsets = np.array([0, 1, 2, 7, 8]) * cadence
    path = _write_cso(tmp_path, offsets=offsets)
    output = tmp_path / "short-window"
    args = _arguments(path, output)
    start = normalize_time("2000-01-01T00:00:00").shift(-cadence)
    end = normalize_time("2000-01-01T00:00:00").shift(10 * cadence)
    args[args.index("--time-start") + 1] = start.utc
    args[args.index("--time-end") + 1] = end.utc
    captured = []
    original = Figure.savefig

    def save(figure, *args, **kwargs):
        result = original(figure, *args, **kwargs)
        axis = figure.axes[1]
        captured.append(
            ([label.get_text() for label in axis.get_xticklabels()], axis.get_xlabel())
        )
        return result

    monkeypatch.setattr(Figure, "savefig", save)
    assert cso_workflow.main(args) == 0
    labels, axis_label = captured[0]
    assert len(labels) > 2 and len(labels) == len(set(labels))
    assert any("." in label for label in labels)
    assert start.utc in axis_label and "TAI seconds" in axis_label
    report = json.loads((output / "native_polarization.json").read_text())
    assert len(report["segments"]) == 2
    assert "integer-ns subtraction" in report["rendering"]["time_axis"]


def test_plot_offsets_remain_continuous_across_a_leap_second():
    start = normalize_time("2016-12-31T23:59:60.500000000")
    end = normalize_time("2017-01-01T00:00:00.500000000")
    np.testing.assert_array_equal(
        cso_native._plot_offsets([start.tai_ns, end.tai_ns], start.tai_ns), [0, 1]
    )


@pytest.mark.parametrize(
    "option,value",
    [("--frequency-bin-mhz", "4"), ("--chunk-columns", "8"), ("--gap-factor", "2")],
)
def test_native_only_options_are_not_silently_ignored_by_legacy_mode(
    monkeypatch, option, value
):
    monkeypatch.setattr(
        cso_workflow, "PlotConfig", lambda: pytest.fail("loaded config")
    )
    with pytest.raises(SystemExit) as stopped:
        cso_workflow.main([option, value])
    assert stopped.value.code == 2


def test_legacy_fallback_is_validated_before_shared_gzip_reader(tmp_path, monkeypatch):
    path = _write_cso(tmp_path, compressed=True)
    requested = Path(str(path)[:-3])
    before = path.read_bytes()
    cache = tmp_path / "cache"
    monkeypatch.setattr(cso_workflow, "apply_config_to_object", lambda *_a: None)
    observed = []

    def process(cfg, spectra):
        observed.append((cso_workflow.get_config_file_paths(cfg), len(spectra)))
        assert len(spectra) == 2
        np.testing.assert_array_equal(spectra[0]._raw[0], [9] * 5)

    monkeypatch.setattr(cso_workflow, "process_and_plot", process)
    assert (
        cso_workflow.main(
            [
                "--file-path",
                str(requested),
                "--time-start",
                "2000-01-01T00:00:00",
                "--time-end",
                "2000-01-01T00:00:08",
                "--frequency-start",
                "100",
                "--frequency-end",
                "108",
                "--cache-root",
                str(cache),
                "--disable-drift",
            ]
        )
        == 0
    )
    assert observed == [([str(path)], 2)]
    assert path.read_bytes() == before and not requested.exists()
    assert not list(cache.iterdir())


def test_explicit_paths_replace_legacy_configured_multifile_input(
    tmp_path, monkeypatch
):
    path = _write_cso(tmp_path, compressed=True)
    monkeypatch.setattr(cso_workflow, "apply_config_to_object", lambda *_a: None)
    cfg = cso_workflow.PlotConfig()
    cfg.file_paths = ["old-configured-path.fits"]
    args = cso_workflow.build_parser().parse_args(
        [
            "--file-path",
            str(path)[:-3],
            "--time-start",
            "2000-01-01T00:00:00",
            "--time-end",
            "2000-01-01T00:00:08",
        ]
    )
    cso_workflow._apply_cli_overrides(cfg, args)
    assert cso_workflow.ConfigManager.validate_all(cfg) == (True, [])
    assert cso_workflow.get_config_file_paths(cfg) == [str(path)]
    args = cso_workflow.build_parser().parse_args(["--file-paths", str(path)])
    cso_workflow._apply_cli_overrides(cfg, args)
    assert cfg.file_paths == [str(path)]
    assert cso_workflow.get_config_file_paths(cfg) == [str(path)]


@pytest.mark.parametrize("existing", ["directory", "file", "symlink"])
def test_existing_outputs_are_never_replaced_even_on_retry(
    tmp_path, monkeypatch, existing
):
    _, _, calls = _fake_science(monkeypatch)
    output = tmp_path / "output"
    if existing == "directory":
        output.mkdir()
        protected = output / "previous.bin"
        protected.write_bytes(b"previous result")
    elif existing == "symlink":
        protected = tmp_path / "previous.bin"
        protected.write_bytes(b"previous result")
        output.symlink_to(protected)
    else:
        protected = output
        output.write_bytes(b"previous result")
    with pytest.raises(FileExistsError):
        cso_native.export_native_polarization(
            [], (), (), frequency_bin_mhz=4, output_dir=output
        )
    assert protected.read_bytes() == b"previous result"
    assert not calls
    assert not list(tmp_path.glob(".output.stage-*"))


@pytest.mark.parametrize("empty_exists", [False, True])
@pytest.mark.parametrize("failure", ["plot", "npz", "rename", "json"])
def test_failed_generation_leaves_no_partial_delivery(
    tmp_path, monkeypatch, empty_exists, failure
):
    _, metadata, _ = _fake_science(monkeypatch)
    output = tmp_path / "output"
    if empty_exists:
        output.mkdir()
    if failure == "plot":

        def fail_plot(_a, _m, path):
            path.write_bytes(b"partial PNG")
            raise OSError("plot failure")

        monkeypatch.setattr(cso_native, "_render", fail_plot)
    elif failure == "npz":

        def fail_npz(path, **_kwargs):
            path.write_bytes(b"partial NPZ")
            raise OSError("NPZ failure")

        monkeypatch.setattr(np, "savez_compressed", fail_npz)
    elif failure == "rename":
        monkeypatch.setattr(
            Path, "rename", lambda *_a: (_ for _ in ()).throw(OSError("rename failure"))
        )
    else:
        metadata["bad"] = float("nan")
    with pytest.raises((OSError, ValueError)):
        cso_native.export_native_polarization(
            [], (), (), frequency_bin_mhz=4, output_dir=output
        )
    assert output.exists() is empty_exists
    if empty_exists:
        assert not list(output.iterdir())
    assert not list(tmp_path.glob(".output.stage-*"))


def test_legacy_main_keeps_processing_and_closes_shared_reader(tmp_path, monkeypatch):
    from types import SimpleNamespace

    calls = []
    cfg = SimpleNamespace(file_path="default", file_paths=None)
    handle = SimpleNamespace(close=lambda: calls.append("closed"))
    monkeypatch.setattr(cso_workflow, "PlotConfig", lambda: cfg)
    monkeypatch.setattr(
        cso_workflow.ConfigManager, "validate_all", lambda _cfg: (True, [])
    )
    monkeypatch.setattr(
        cso_workflow, "get_config_file_paths", lambda value: [value.file_path]
    )
    monkeypatch.setattr(cso_workflow, "_print_config_summary", lambda *_a: None)
    monkeypatch.setattr(
        cso_workflow,
        "read_cso_fits",
        lambda path, **kw: (calls.append((path, kw)) or ["spec"], handle),
    )
    monkeypatch.setattr(
        cso_workflow,
        "process_and_plot",
        lambda value, spectra: calls.append((value, spectra)),
    )
    assert (
        cso_workflow.main(
            ["--file-path", "input.fits.gz", "--cache-root", str(tmp_path)]
        )
        == 0
    )
    assert calls == [
        ("input.fits.gz", {"cache_root": str(tmp_path)}),
        (cfg, ["spec"]),
        "closed",
    ]
