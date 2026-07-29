"""Focused numerical and proxy-model tests."""

from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pytest

import spike_typeIII_visual.main as main_module
from spike_typeIII_visual.config import MHDConfig, RadioConfig, profile_config
from spike_typeIII_visual.main import (
    _resolve_animation_formats,
    _write_manifest,
)
from spike_typeIII_visual.physics.jet import (
    diagnose_jet,
    find_sustained_onset,
    map_active_interval_to_radio_time,
    reconnection_flux_rate,
)
from spike_typeIII_visual.physics.radio import (
    electron_beam_kinematics,
    synthesize_radio_proxy,
    typeiii_ridge_frequency_mhz,
)
from spike_typeIII_visual.physics.rmhd import (
    SpectralGrid,
    double_harris_flux,
    ideal_energy_exchange_residual,
    solve_rmhd,
)
from spike_typeIII_visual.validate_outputs import (
    EXPECTED_MANIFEST_PATHS,
    _animation_formats_from_metadata,
    _expected_manifest_paths,
    _validate_checksum_manifest,
)
from spike_typeIII_visual.visualization import animations as animation_module
from spike_typeIII_visual.visualization.scientific_animations import (
    iter_reconnection_topology,
)


def small_config() -> MHDConfig:
    return MHDConfig(nx=32, ny=32, steps=16, snapshot_stride=4)


def test_event_and_control_profiles_share_aligned_long_times() -> None:
    medium = profile_config("cuda-medium-event", 7).mhd
    event = profile_config("cuda-fine-event", 7).mhd
    control = profile_config("cuda-fine-control", 7).mhd
    assert medium.dt * medium.steps == pytest.approx(8.0)
    assert event.dt * event.steps == pytest.approx(8.0)
    assert control.dt * control.steps == pytest.approx(8.0)
    assert event.steps // event.snapshot_stride + 1 == 401
    assert control.steps // control.snapshot_stride + 1 == 401
    assert event.perturbation_amplitude == pytest.approx(0.04)
    assert control.perturbation_amplitude == 0.0


def test_periodic_double_harris_flux_is_finite() -> None:
    config = small_config()
    grid = SpectralGrid.from_config(config)
    flux = double_harris_flux(config, grid)
    assert flux.shape == (config.ny, config.nx)
    assert np.isfinite(flux).all()
    magnetic_x = -grid.derivative_y(flux)
    assert float(np.max(magnetic_x)) > 0.5
    assert float(np.min(magnetic_x)) < -0.5


def test_strict_two_thirds_mask_excludes_exact_cutoff_modes() -> None:
    config = MHDConfig(nx=96, ny=96, steps=1)
    grid = SpectralGrid.from_config(config)
    modes = np.fft.fftfreq(config.nx) * config.nx
    cutoff_index = int(np.argmin(np.abs(modes - 32.0)))
    retained_index = int(np.argmin(np.abs(modes - 31.0)))
    zero_index = int(np.argmin(np.abs(modes)))
    assert not grid.dealias_mask[zero_index, cutoff_index]
    assert grid.dealias_mask[zero_index, retained_index]


def test_spectral_operators_and_poisson_inversion() -> None:
    config = MHDConfig(nx=48, ny=48, steps=1)
    grid = SpectralGrid.from_config(config)
    field = np.sin(2.0 * grid.x_mesh) * np.cos(3.0 * grid.y_mesh)
    assert (
        np.max(
            np.abs(
                grid.derivative_x(field)
                - 2.0 * np.cos(2.0 * grid.x_mesh) * np.cos(3.0 * grid.y_mesh)
            )
        )
        < 1.0e-11
    )
    expected_laplacian = -13.0 * field
    assert np.max(np.abs(grid.laplacian(field) - expected_laplacian)) < 1.0e-10
    assert np.max(np.abs(grid.poisson_solve(expected_laplacian) - field)) < 1.0e-10


def test_physical_lorentz_sign_closes_ideal_energy_exchange() -> None:
    physical = MHDConfig(
        nx=48,
        ny=48,
        steps=1,
        resistivity=0.0,
        viscosity=0.0,
        lorentz_convention="physical",
    )
    grid = SpectralGrid.from_config(physical)
    rng = np.random.default_rng(4)
    psi = grid.filter(rng.standard_normal((physical.ny, physical.nx)))
    phi = grid.filter(rng.standard_normal((physical.ny, physical.nx)))
    omega = grid.laplacian(phi)
    physical_residual = ideal_energy_exchange_residual(
        psi,
        omega,
        physical,
        grid,
    )
    legacy_residual = ideal_energy_exchange_residual(
        psi,
        omega,
        replace(physical, lorentz_convention="legacy"),
        grid,
    )
    assert abs(physical_residual) < 1.0e-10
    assert abs(legacy_residual) > 1.0e-8


def test_mhd_run_is_finite_and_divergence_free() -> None:
    result = solve_rmhd(small_config())
    assert np.isfinite(result.psi).all()
    assert np.isfinite(result.omega).all()
    assert result.divergence_rms < 1e-10
    assert result.max_speed[-1] > 0.0


def test_short_run_energy_drift_is_bounded() -> None:
    result = solve_rmhd(replace(small_config(), steps=24))
    total = result.magnetic_energy + result.kinetic_energy
    drift = abs(float((total[-1] - total[0]) / total[0]))
    assert drift < 0.05


def test_radio_proxy_is_monotonic_and_deterministic() -> None:
    mhd = solve_rmhd(small_config())
    radio_config = RadioConfig(time_samples=40, frequency_samples=64, spike_count=5)
    first = synthesize_radio_proxy(mhd, radio_config, seed=11)
    second = synthesize_radio_proxy(mhd, radio_config, seed=11)
    assert np.all(np.diff(first.ridge_frequency_mhz) < 0.0)
    assert np.array_equal(first.intensity, second.intensity)
    assert np.array_equal(first.spike_catalog, second.spike_catalog)
    assert first.intensity.shape == (
        radio_config.frequency_samples,
        radio_config.time_samples,
    )
    onset_end = min(
        radio_config.spike_onset_cap_s,
        radio_config.duration_s * radio_config.spike_onset_fraction,
    )
    spike_times = first.spike_catalog[:, 0]
    spike_frequencies = first.spike_catalog[:, 1]
    ridge_at_spikes = np.interp(
        spike_times,
        first.times_s,
        first.ridge_frequency_mhz,
    )
    assert np.all(spike_times >= radio_config.spike_onset_start_s)
    assert np.all(spike_times <= onset_end)
    assert np.all(spike_frequencies > ridge_at_spikes)
    assert np.all(spike_frequencies <= radio_config.max_frequency_mhz)


def test_jet_diagnostic_and_compressed_mapping_are_deterministic() -> None:
    mhd_config = replace(small_config(), steps=40, snapshot_stride=2)
    mhd = solve_rmhd(mhd_config)
    jet_config = main_module.profile_config("quick", 11).jet
    first = diagnose_jet(mhd, mhd_config, jet_config)
    second = diagnose_jet(mhd, mhd_config, jet_config)
    assert np.array_equal(first.jet_activity, second.jet_activity)
    assert np.array_equal(first.reconnection_activity, second.reconnection_activity)
    assert first.positive_speed.shape == mhd.times.shape
    assert first.negative_speed.shape == mhd.times.shape
    assert find_sustained_onset(np.array([0.0, 0.7, 0.8, 0.2]), 0.6, 2) == 1
    mapped = map_active_interval_to_radio_time(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 0.5, 1.0]),
        1.0,
        np.array([0.0, 0.08, 0.415, 0.75, 1.0]),
        0.08,
        0.75,
    )
    assert np.allclose(mapped, [0.0, 0.5, 0.75, 1.0, 0.0])


def test_reconnection_activity_uses_flux_difference_derivative() -> None:
    mhd_config = replace(small_config(), steps=40, snapshot_stride=2)
    mhd = solve_rmhd(mhd_config)
    rate = reconnection_flux_rate(mhd)
    expected = np.abs(
        np.gradient(mhd.flux_difference, mhd.times, edge_order=2)
    )
    assert np.allclose(rate, expected)
    jet = diagnose_jet(
        mhd,
        mhd_config,
        profile_config("quick", 11).jet,
    )
    span = float(np.ptp(expected))
    normalized = (
        np.zeros_like(expected)
        if span <= 1.0e-15
        else np.clip((expected - expected[0]) / span, 0.0, 1.0)
    )
    assert np.allclose(jet.reconnection_activity, normalized)


def test_scientific_topology_preview_has_expected_canvas() -> None:
    event_config = replace(small_config(), steps=8, snapshot_stride=4)
    control_config = replace(event_config, perturbation_amplitude=0.0)
    event = solve_rmhd(event_config)
    control = solve_rmhd(control_config)
    frame = next(
        iter(
            iter_reconnection_topology(
                event,
                control,
                event_config,
                "scientific-preview",
            )
        )
    )
    assert frame.shape == (540, 960, 3)
    assert frame.dtype == np.uint8


def test_jet_coupling_returns_valid_empty_catalog_without_relaxation() -> None:
    mhd_config = small_config()
    mhd = solve_rmhd(mhd_config)
    jet_config = main_module.profile_config("quick", 11).jet
    jet = diagnose_jet(mhd, mhd_config, jet_config)
    inactive = replace(
        jet,
        jet_activity=np.zeros_like(jet.jet_activity),
        reconnection_activity=np.zeros_like(jet.reconnection_activity),
        onset_index=None,
        onset_time_normalized=None,
    )
    radio = synthesize_radio_proxy(
        mhd,
        RadioConfig(time_samples=40, frequency_samples=64, spike_count=5),
        seed=11,
        jet_result=inactive,
        jet_config=jet_config,
        spike_coupling="jet",
    )
    assert radio.spike_catalog.shape == (0, 5)
    assert radio.event_status == "no_event"
    assert radio.jet_coincidence_fraction is None


def test_typeiii_ridge_matches_exponential_closed_form() -> None:
    config = RadioConfig()
    times = np.linspace(0.0, config.duration_s, config.time_samples)
    height_mm, _ = electron_beam_kinematics(
        times,
        config.beam_speed_fraction_c,
    )
    base_density = (config.start_frequency_mhz * 1.0e6 / 8_980.0) ** 2
    ridge = typeiii_ridge_frequency_mhz(
        height_mm,
        base_density,
        config.density_scale_height_mm,
    )
    expected = config.start_frequency_mhz * np.exp(
        -height_mm / (2.0 * config.density_scale_height_mm)
    )
    assert np.allclose(ridge, expected)
    assert np.all(np.diff(ridge) < 0.0)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("spike_onset_fraction", 0.0),
        ("spike_onset_start_s", -0.01),
        ("spike_frequency_offset_min_mhz", 0.0),
        ("spike_frequency_offset_max_mhz", 5.0),
    ],
)
def test_invalid_spike_topping_config_is_rejected(
    field_name: str,
    value: float,
) -> None:
    kwargs = {field.name: field.default for field in fields(RadioConfig)}
    kwargs[field_name] = value
    with pytest.raises(ValueError):
        RadioConfig(**kwargs)


def test_animation_format_selection_and_compatibility_alias() -> None:
    assert _resolve_animation_formats(None, False) == ("gif",)
    assert _resolve_animation_formats("none", False) == ()
    assert _resolve_animation_formats("mp4", False) == ("mp4",)
    assert _resolve_animation_formats("both", False) == ("gif", "mp4")
    assert _resolve_animation_formats(None, True) == ()
    with pytest.raises(ValueError, match="cannot be combined"):
        _resolve_animation_formats("gif", True)


def test_both_animation_formats_reuse_each_render(
    tmp_path,
    monkeypatch,
) -> None:
    frames = [np.zeros((2, 2, 3), dtype=np.uint8)]
    render_calls: list[str] = []
    written_paths = []

    def renderer(name: str):
        def _render(_source):
            render_calls.append(name)
            return frames

        return _render

    monkeypatch.setattr(
        animation_module,
        "_render_tearing_frames",
        renderer("tearing"),
    )
    monkeypatch.setattr(
        animation_module,
        "_render_jet_frames",
        renderer("jet"),
    )
    monkeypatch.setattr(
        animation_module,
        "_render_electron_frames",
        renderer("electron_beam"),
    )
    monkeypatch.setattr(
        animation_module,
        "_render_typeiii_frames",
        renderer("typeIII"),
    )
    monkeypatch.setattr(
        animation_module,
        "require_mp4_backend",
        lambda: "/mock/ffmpeg",
    )
    monkeypatch.setattr(
        animation_module,
        "_write",
        lambda path, _frames: written_paths.append(path),
    )

    paths = animation_module.save_animations(
        object(),
        object(),
        tmp_path,
        formats=("gif", "mp4"),
    )

    assert sorted(render_calls) == [
        "electron_beam",
        "jet",
        "tearing",
        "typeIII",
    ]
    assert len(written_paths) == 8
    assert paths == written_paths
    assert {path.suffix for path in paths} == {".gif", ".mp4"}


def test_mp4_backend_failure_has_installation_hint(monkeypatch) -> None:
    def missing_backend() -> str:
        raise ImportError("imageio_ffmpeg is absent")

    monkeypatch.setattr(
        animation_module,
        "_load_imageio_ffmpeg_executable",
        missing_backend,
    )
    with pytest.raises(RuntimeError, match="conda install"):
        animation_module.require_mp4_backend()


def test_mp4_writer_uses_compatible_h264_settings(
    tmp_path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def capture_write(path, frames, **kwargs) -> None:
        captured["path"] = path
        captured["frames"] = frames
        captured["kwargs"] = kwargs

    monkeypatch.setattr(
        animation_module,
        "require_mp4_backend",
        lambda: "/mock/ffmpeg",
    )
    writer = type("Writer", (), {"mimsave": staticmethod(capture_write)})
    monkeypatch.setattr(animation_module, "_imageio_module", lambda: writer)
    frames = [np.zeros((540, 960, 3), dtype=np.uint8)]

    animation_module._write(tmp_path / "typeIII.mp4", frames)

    assert captured["frames"] is frames
    assert captured["kwargs"] == {
        "fps": 10,
        "codec": "libx264",
        "pixelformat": "yuv420p",
        "macro_block_size": 2,
    }


def test_run_checks_mp4_backend_before_solver(
    tmp_path,
    monkeypatch,
) -> None:
    solver_called = False

    def unavailable_backend() -> str:
        raise RuntimeError("missing backend")

    def unexpected_solver(*_args, **_kwargs):
        nonlocal solver_called
        solver_called = True
        raise AssertionError("solver must not run")

    monkeypatch.setattr(main_module, "require_mp4_backend", unavailable_backend)
    monkeypatch.setattr(main_module, "solve_rmhd", unexpected_solver)
    output_dir = tmp_path / "output"

    with pytest.raises(RuntimeError, match="missing backend"):
        main_module.run(
            main_module.profile_config("quick", 11),
            output_dir,
            ("mp4",),
        )
    assert not solver_called
    assert not output_dir.exists()


def test_animation_metadata_controls_expected_manifest_paths() -> None:
    legacy_formats, legacy_errors = _animation_formats_from_metadata({})
    assert legacy_formats == ("gif",)
    assert not legacy_errors

    none_formats, none_errors = _animation_formats_from_metadata(
        {"exports": {"animation_formats": []}}
    )
    assert none_formats == ()
    assert not none_errors

    both_paths = _expected_manifest_paths(("gif", "mp4"))
    assert "animations/tearing.gif" in both_paths
    assert "animations/tearing.mp4" in both_paths
    assert len(both_paths - _expected_manifest_paths(())) == 8


def test_checksum_manifest_uses_lf_and_posix_paths(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    data_path = data_dir / "diagnostics.csv"
    data_path.write_bytes(b"time,value\n0,1\n")

    manifest_path = _write_manifest(tmp_path, [data_path])
    raw = manifest_path.read_bytes()

    assert raw.endswith(b"\n")
    assert b"\r" not in raw
    assert raw.decode("utf-8").endswith("  data/diagnostics.csv\n")


def test_checksum_manifest_detects_digest_mismatch(tmp_path) -> None:
    expected_paths = set(EXPECTED_MANIFEST_PATHS)
    paths = []
    for relative_name in sorted(expected_paths):
        path = tmp_path / relative_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative_name.encode("utf-8"))
        paths.append(path)
    _write_manifest(tmp_path, paths)

    changed_path = tmp_path / "data" / "diagnostics.csv"
    changed_path.write_bytes(b"changed\n")
    record, errors = _validate_checksum_manifest(tmp_path)

    assert record["verified_files"] == len(expected_paths) - 1
    assert errors == ["Checksum mismatch: data/diagnostics.csv"]
