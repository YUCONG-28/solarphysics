"""Compose SXR loading, smoothing, derivatives and plotting with explicit inputs."""

from __future__ import annotations

import argparse
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd

from solar_toolkit.xray_dem.processing import calculate_derivative, smooth_flux_data
from solar_toolkit.xray_dem.sxr import load_sxr_data


def _private_output_path(path: Path, repo_root: Path) -> Path:
    """Resolve each write target while keeping the allowed Local root lexical."""
    resolved = path.expanduser().resolve()
    if resolved.is_relative_to(repo_root) and not resolved.is_relative_to(
        repo_root / "Local"
    ):
        raise ValueError("Outputs must be outside public source or under Local.")
    return resolved


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="User-owned SXR CSV or NetCDF file.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-time")
    parser.add_argument("--end-time")
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[3]
    source = args.input
    synthetic = source is None
    try:
        output = _private_output_path(args.output_dir, repo_root)
        image_path = _private_output_path(output / "sxr_example.png", repo_root)
        if synthetic:
            source = _private_output_path(output / "synthetic_sxr.csv", repo_root)
    except ValueError as exc:
        parser.error(str(exc))
    if synthetic:
        if source.exists():
            parser.error(
                "Choose a new output directory; synthetic input already exists."
            )
    output.mkdir(parents=True, exist_ok=True)
    if synthetic:
        seconds = np.arange(81, dtype=float)
        times = pd.date_range("2000-01-01", periods=len(seconds), freq="s", tz="UTC")
        flux = 1e-7 + 1e-6 * np.exp(-0.5 * ((seconds - 40) / 10) ** 2)
        pd.DataFrame({"time": times, "xrsa_flux": flux / 10, "xrsb_flux": flux}).to_csv(
            source, index=False
        )
    data = load_sxr_data(source, args.start_time, args.end_time)
    if not isinstance(data, pd.DataFrame):
        data = data[["xrsa_flux", "xrsb_flux"]].to_dataframe().reset_index()
    if not {"time", "xrsa_flux", "xrsb_flux"} <= set(data.columns):
        raise ValueError("Input must contain time, xrsa_flux and xrsb_flux.")
    if len(data) < 5:
        raise ValueError("At least five SXR samples are required.")
    times = pd.to_datetime(data["time"], utc=True)
    long_flux = data["xrsb_flux"].to_numpy(dtype=float)
    smoothed = smooth_flux_data(long_flux, window_length=5)
    derivative = calculate_derivative(times, smoothed, method="gradient")
    if not np.all(np.isfinite(smoothed)) or not np.all(np.isfinite(derivative)):
        raise ValueError("This example requires finite flux samples.")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    axes[0].plot(times, data["xrsa_flux"], label="Short channel")
    axes[0].plot(times, long_flux, label="Long channel")
    axes[0].plot(times, smoothed, label="Moving average")
    axes[0].set_ylabel("Flux (W m$^{-2}$)")
    axes[0].legend()
    axes[0].set_title("Synthetic SXR data" if synthetic else "SXR data")
    axes[1].plot(times, derivative)
    axes[1].set_ylabel("dF/dt (W m$^{-2}$ s$^{-1}$)")
    axes[1].set_xlabel("UTC")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))
    figure.tight_layout()
    figure.savefig(image_path, dpi=120)
    plt.close(figure)
    print("SXR composition example completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
