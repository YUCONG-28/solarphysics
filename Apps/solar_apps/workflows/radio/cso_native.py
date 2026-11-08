"""Transactional output and segmented display for native CSO polarization.

Scientific selection, paired aggregation and gap detection belong to
``solar_toolkit.radio.cso_extract``. This adapter only writes and displays
its returned arrays and provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

__all__ = ["export_native_polarization"]

_STEM = "native_polarization"


def _check_destination(destination: Path) -> None:
    if destination.is_symlink():
        raise FileExistsError(f"Output directory must not be a symlink: {destination}")
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise FileExistsError(
            f"Output already exists and is not empty: {destination}; choose a new or empty directory"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plot_offsets(tai_ns, origin):
    # Subtract integers before converting to floating-point display seconds.
    # This preserves native short-window precision and continuous leap seconds.
    return np.array([int(value) - int(origin) for value in tai_ns], dtype=float) / 1e9


def _render(arrays, metadata, output: Path) -> dict:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    intensity = np.asarray(arrays["intensity"], dtype=float)
    logged = np.full(intensity.shape, np.nan)
    positive = np.isfinite(intensity) & (intensity > 0)
    np.log10(intensity, out=logged, where=positive)
    finite = logged[np.isfinite(logged)]
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    intensity_norm = Normalize(lo, hi if hi > lo else lo + 1.0)
    polarization_norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
    frequency = np.asarray(arrays["frequency_edges_mhz"], dtype=float)
    polarization = np.asarray(arrays["polarization_percent"], dtype=float)
    time_bounds = metadata["requested_time_bounds_tai_ns"]
    origin = time_bounds[0]
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True, sharey=True, layout="constrained"
    )
    try:
        for axis, data, norm, cmap, title, label in zip(
            axes,
            (logged, polarization),
            (intensity_norm, polarization_norm),
            ("viridis", "RdBu_r"),
            ("CSO total intensity", "CSO circular polarization"),
            (r"log$_{10}$ intensity (SFU)", "100 × (ΣR − ΣL) / (ΣR + ΣL) (%)"),
            strict=True,
        ):
            for segment in metadata["segments"]:
                start, stop = segment["start_index"], segment["stop_index"]
                edges = _plot_offsets(segment["time_edges_tai_ns"], origin)
                if len(edges) != stop - start + 1 or np.any(np.diff(edges) <= 0):
                    raise ValueError(
                        "Segment edges must increase and match its native columns"
                    )
                # A mesh per support segment prevents a large cell from joining
                # observations across a real gap. No time binning is performed.
                axis.pcolormesh(
                    edges,
                    frequency,
                    data[:, start:stop],
                    shading="flat",
                    norm=norm,
                    cmap=cmap,
                    rasterized=True,
                )
            for missing in metadata["missing_support_tai_ns"]:
                left, right = _plot_offsets(missing, origin)
                if right > left:
                    axis.axvspan(
                        left,
                        right,
                        facecolor="0.88",
                        hatch="///",
                        edgecolor="0.6",
                        linewidth=0,
                    )
            if intensity.shape[1] == 0:
                axis.text(
                    0.5,
                    0.5,
                    "No samples in requested window",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            axis.set_ylabel("Frequency (MHz)")
            axis.set_title(title)
            axis.set_ylim(frequency[0], frequency[-1])
            fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axis, label=label)
        if time_bounds[1] > time_bounds[0]:
            axes[-1].set_xlim(*_plot_offsets(time_bounds, origin))
        axes[-1].xaxis.set_major_locator(MaxNLocator(nbins=8, min_n_ticks=3))
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        axes[-1].xaxis.set_major_formatter(formatter)
        axes[-1].set_xlabel(
            f"TAI seconds since {metadata['requested_time_bounds_utc'][0]} UTC"
        )
        fig.savefig(output, dpi=160)
    finally:
        plt.close(fig)
    return {
        "intensity_display": "log10 positive SFU values; other samples remain missing",
        "polarization_display": "scientific polarization_percent without intensity threshold",
        "segments_drawn": len(metadata["segments"]),
        "missing_support": "hatched; no time interpolation or cross-gap mesh",
        "time_axis": "TAI second offsets from the explicit UTC window start; integer-ns subtraction before conversion",
    }


def export_native_polarization(
    paths,
    time_bounds,
    frequency_bounds,
    *,
    frequency_bin_mhz,
    output_dir,
    cache_root=None,
    chunk_columns=128,
    gap_factor=1.5,
) -> dict:
    """Write one new NPZ/JSON/PNG bundle, committing the directory last.

    Existing nonempty directories, files and symlinks are refused, including a
    retry after success. Choose a new or empty output directory for each run.
    Generation failures leave no partial bundle; source files are read only.
    All scientific calculations and metadata come from the shared library.
    """
    from solar_toolkit.radio.cso_extract import extract_native_polarization

    destination = Path(output_dir).expanduser().absolute()
    _check_destination(destination)
    arrays, metadata = extract_native_polarization(
        paths,
        time_bounds,
        frequency_bounds,
        frequency_bin_mhz=frequency_bin_mhz,
        cache_root=cache_root,
        chunk_columns=chunk_columns,
        gap_factor=gap_factor,
    )
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise ValueError("Native CSO arrays must not require pickle")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent)
    )
    removed_empty = False
    try:
        archive = staging / f"{_STEM}.npz"
        image = staging / f"{_STEM}.png"
        np.savez_compressed(archive, **arrays)
        rendering = _render(arrays, metadata, image)
        report = dict(metadata)
        report["rendering"] = rendering
        report["outputs"] = {
            archive.name: {
                "sha256": _sha256(archive),
                "size_bytes": archive.stat().st_size,
            },
            image.name: {"sha256": _sha256(image), "size_bytes": image.stat().st_size},
        }
        (staging / f"{_STEM}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _check_destination(destination)
        if os.name == "nt" and destination.exists():
            # Windows cannot rename a directory over an existing empty one.
            destination.rmdir()
            removed_empty = True
        staging.rename(destination)
        return report
    except BaseException:
        if removed_empty and not destination.exists():
            destination.mkdir()
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
