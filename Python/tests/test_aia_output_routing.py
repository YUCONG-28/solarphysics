from __future__ import annotations

from pathlib import Path

from solar_toolkit.aia._euv_processor_impl import (
    _difference_save_dir,
    _mosaic_save_dir,
    _run_difference_batch,
    _run_single_batch,
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


def test_single_worker_uses_serial_execution(monkeypatch) -> None:
    calls: list[tuple[Path, int]] = []
    config = AIAConfig(max_workers=1)
    files = [Path("one.fits"), Path("two.fits")]
    monkeypatch.setattr(
        "solar_toolkit.aia._euv_processor_impl._resolve_single_files",
        lambda _config: files,
    )

    def process(path, _config, sequence, _generated):
        calls.append((path, sequence))
        return True, "ok"

    monkeypatch.setattr(
        "solar_toolkit.aia._euv_processor_impl._process_single_worker",
        process,
    )

    class ForbiddenPool:
        def __init__(self, *args, **kwargs):
            raise AssertionError("ProcessPoolExecutor must not be used for one worker")

    monkeypatch.setattr(
        "solar_toolkit.aia._euv_processor_impl.ProcessPoolExecutor",
        ForbiddenPool,
    )

    _run_single_batch(config)

    assert calls == [(files[0], 1), (files[1], 2)]


def test_difference_worker_uses_serial_execution(monkeypatch) -> None:
    calls: list[int] = []
    config = AIAConfig(
        max_workers=1,
        draw_difference=True,
        difference_wavelengths=(171, 193),
    )

    def process(wave, _config, _generated):
        calls.append(wave)
        return True, "ok"

    monkeypatch.setattr(
        "solar_toolkit.aia._euv_processor_impl._process_difference_band_worker",
        process,
    )

    class ForbiddenPool:
        def __init__(self, *args, **kwargs):
            raise AssertionError("ProcessPoolExecutor must not be used for one worker")

    monkeypatch.setattr(
        "solar_toolkit.aia._euv_processor_impl.ProcessPoolExecutor",
        ForbiddenPool,
    )

    _run_difference_batch(config)

    assert calls == [171, 193]
