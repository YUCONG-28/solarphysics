# 模块用途: 绘制 RHESSI/HESSI 风格硬 X 射线光变曲线。
# 主要输入: 硬 X 射线事件或光变数据。
# 主要输出/运行说明: 输出 HXR 时间序列图。
""" """

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from solar_apps.workflows.common.image_naming import build_scientific_image_filename

from solar_toolkit.xray_dem.hxi import HXI_ENERGY_CHANNELS, load_hxi_lightcurve

DEFAULT_INPUT_DIRECTORY = "data/hxi"
DEFAULT_OUTPUT_DIRECTORY = "outputs/hxi"


def process_hxi_fits(input_dir, output_dir):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = [
        "PingFang SC",
        "Hiragino Sans GB",
        "SimHei",
        "Microsoft YaHei",
        "sans-serif",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    # Check if input directory exists
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist!")
        return

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all FITS files in the directory
    fits_files = sorted([f for f in input_path.iterdir() if f.suffix == ".fits"])

    if not fits_files:
        print(f"Warning: No FITS files found in '{input_dir}'!")
        return

    # Process each FITS file
    batch_generated_at = datetime.now(timezone.utc)
    for sequence, fits_file in enumerate(fits_files, start=1):
        try:
            print(f"Processing: {fits_file}")

            lightcurve = load_hxi_lightcurve(fits_file)
            utc_times = lightcurve["times"]

            # Create figure
            plt.figure(figsize=(25, 16))
            ax1 = plt.gca()

            # Plot light curves
            for channel in HXI_ENERGY_CHANNELS:
                plt.semilogy(
                    utc_times,
                    lightcurve["data"][channel],
                    label=f"HXI {channel}",
                )

            # Set axis labels and title
            plt.ylabel(
                "Counts s\u207b\u00b9 detector\u207b\u00b9",
                fontsize=22,
                labelpad=12,
            )
            plt.legend(loc="upper left", ncol=1, fontsize=18)

            # Add minute-level grid lines
            ax1.xaxis.set_minor_locator(mdates.MinuteLocator())
            ax1.xaxis.grid(True, which="minor", linestyle="--", color="gray", alpha=0.5)

            # Set time format
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            plt.gcf().autofmt_xdate()  # Auto-rotate date labels
            plt.xlabel("Time (UTC)", fontsize=22, labelpad=12)

            # Use filename as part of title
            file_name = fits_file.stem
            plt.title(f"{file_name}", fontsize=22, fontweight="bold")

            # Save image (use FITS filename as image name)
            valid_times = list(utc_times)
            img_name = build_scientific_image_filename(
                sequence=sequence,
                start_time=valid_times[0] if valid_times else None,
                end_time=valid_times[-1] if len(valid_times) > 1 else None,
                instrument="hxi",
                product="lightcurve",
                generated_at=batch_generated_at,
            )
            img_path = output_path / img_name
            plt.savefig(img_path, dpi=300, bbox_inches="tight")
            plt.show()
            print(f"Image saved to: {img_path}")

            plt.close()  # Close figure to free memory

        except Exception as e:
            print(f"Error processing file {fits_file.name}: {str(e)}")


def build_parser() -> argparse.ArgumentParser:
    """Build the HXI light-curve batch command-line parser."""

    parser = argparse.ArgumentParser(
        description="Plot ASO-S/HXI light curves for every FITS file in a directory."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIRECTORY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIRECTORY)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the historical HXI batch light-curve workflow."""

    args = build_parser().parse_args(argv)
    process_hxi_fits(args.input_dir, args.output_dir)
    print("Processing complete!")
    return 0


__all__ = [
    "DEFAULT_INPUT_DIRECTORY",
    "DEFAULT_OUTPUT_DIRECTORY",
    "build_parser",
    "main",
    "process_hxi_fits",
]


if __name__ == "__main__":
    raise SystemExit(main())
