"""AIA/HMI magnetic-contour overlay workflow.

English: Match AIA and HMI observations by time, reproject both instruments to
a shared AIA region of interest, and render HMI contours over AIA imagery.
Importing this module performs no file discovery or plotting.

中文：按时间匹配 AIA 与 HMI 观测，将两类数据重投影到共同的 AIA 感兴趣区域，并在
AIA 图像上绘制 HMI 磁场等值线。导入本模块不会发现文件或执行绘图。
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

from solar_toolkit.visualization.image_naming import (
    ImageFilenameSpec,
    build_image_filename,
    format_utc_filename_time,
)

DEFAULT_AIA_DIR = Path("data/aia/171")
DEFAULT_HMI_DIR = Path("data/hmi")
DEFAULT_OUTPUT_DIR = Path("outputs/aia_hmi")
DEFAULT_ROI_BOUNDS = (-700.0, -100.0, -100.0, 400.0)

# Several AIA frames can match the same nearest HMI file; cache the aligned
# (reprojected, copied) map so the expensive reprojection runs once per HMI.
_HMI_ALIGN_CACHE_MAX_ITEMS = 8
_HMI_ALIGN_CACHE: dict[tuple[str, str], object] = {}


def run_overlay_workflow(
    input_dir_aia: str | Path = DEFAULT_AIA_DIR,
    input_dir_hmi: str | Path = DEFAULT_HMI_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    roi_bounds: tuple[float, float, float, float] = DEFAULT_ROI_BOUNDS,
    threshold_gauss: float = 0.0,
    gaussian_sigma: float = 3.0,
    vmin: float = 16.0,
    vmax: float = 6666.0,
    contour_level_gauss: float = 50.0,
    max_time_diff_seconds: float = 24.0,
    contour_colors: tuple[str, str] = ("b", "r"),
    contour_linewidths: tuple[float, float] = (1.0, 1.0),
    figure_size: tuple[float, float] = (10.0, 8.0),
    dpi: int = 300,
    show_plot: bool = False,
    show_progress: bool = True,
) -> list[Path]:
    """Render AIA/HMI overlays and return successfully generated image paths.

    The defaults preserve the historical 171 Å workflow, including its ROI,
    24-second matching tolerance, ±50 G contours, logarithmic AIA scaling, and
    ``YYYYMMDD_HHMMSS.png`` output names.
    """

    import astropy.units as u
    import matplotlib.colors as mpl_colors
    import matplotlib.pyplot as plt
    import sunpy.map
    from astropy.coordinates import SkyCoord
    from matplotlib.lines import Line2D
    from tqdm import tqdm

    from solar_toolkit._utils.memory import (
        monitor_memory_usage,
        optimized_gc_collect,
    )
    from solar_toolkit.hmi.processing import (
        create_magnetic_contour_levels,
        process_hmi_magnetic_field,
    )
    from solar_toolkit.io.discovery import get_sorted_fits_files
    from solar_toolkit.map.operations import (
        align_maps_to_reference,
        normalize_aia_exposure,
    )
    from solar_toolkit.time.formatting import (
        format_time_for_display,
    )
    from solar_toolkit.visualization.plotting import (
        create_figure_with_white_background,
        setup_chinese_font,
    )

    setup_chinese_font()

    aia_dir = Path(input_dir_aia)
    hmi_dir = Path(input_dir_hmi)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    threshold = threshold_gauss * u.Gauss
    norm = mpl_colors.LogNorm(vmin=vmin, vmax=vmax)

    print("Loading AIA files...")
    direct_aia_files = any(aia_dir.glob("*.fits"))
    default_wave_dir = aia_dir / "171"
    aia_search_dir = (
        default_wave_dir
        if not direct_aia_files and default_wave_dir.is_dir()
        else aia_dir
    )
    aia_files = get_sorted_fits_files(
        str(aia_search_dir),
        recursive=not direct_aia_files and aia_search_dir == aia_dir,
    )
    print("Loading HMI files...")
    hmi_files = get_sorted_fits_files(str(hmi_dir), recursive=True)
    if not aia_files:
        raise ValueError("No valid FITS files were found in the AIA directory")
    if not hmi_files:
        raise ValueError("No valid FITS files were found in the HMI directory")
    if len(aia_files) < 2:
        raise ValueError("At least two AIA files are required to define the target WCS")

    print(f"Found {len(aia_files)} AIA files and {len(hmi_files)} HMI files")

    reference_map = sunpy.map.Map(aia_files[1][0])
    x_min, y_min, x_max, y_max = roi_bounds
    roi_bottom_left = SkyCoord(
        Tx=x_min * u.arcsec,
        Ty=y_min * u.arcsec,
        frame=reference_map.coordinate_frame,
    )
    roi_top_right = SkyCoord(
        Tx=x_max * u.arcsec,
        Ty=y_max * u.arcsec,
        frame=reference_map.coordinate_frame,
    )
    cutout_map = reference_map.submap(
        roi_bottom_left,
        top_right=roi_top_right,
    )
    target_wcs = cutout_map.wcs
    del reference_map, cutout_map, roi_bottom_left, roi_top_right
    optimized_gc_collect()

    levels = create_magnetic_contour_levels(contour_level_gauss * u.Gauss)
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=contour_colors[0],
            lw=contour_linewidths[0],
            alpha=0.8,
            label=f"-{contour_level_gauss:g} Gauss",
        ),
        Line2D(
            [0],
            [0],
            color=contour_colors[1],
            lw=contour_linewidths[1],
            alpha=0.8,
            label=f"{contour_level_gauss:g} Gauss",
        ),
    ]

    monitor_memory_usage("Memory before processing")
    hmi_index = 0
    processed_files = 0
    output_paths: list[Path] = []
    start_time = time.time()
    batch_generated_at = dt.datetime.now(dt.UTC)
    iterator = (
        tqdm(aia_files, desc="Processing", unit="file") if show_progress else aia_files
    )

    for sequence, (aia_path, aia_time) in enumerate(iterator, start=1):
        if hmi_index >= len(hmi_files):
            print("All HMI files have been processed; stopping")
            break

        while hmi_index < len(hmi_files) - 1:
            current_hmi_time = hmi_files[hmi_index][1]
            next_hmi_time = hmi_files[hmi_index + 1][1]
            current_diff = abs((aia_time - current_hmi_time).total_seconds())
            next_diff = abs((aia_time - next_hmi_time).total_seconds())
            if next_diff < current_diff:
                hmi_index += 1
            else:
                break

        hmi_path, hmi_time = hmi_files[hmi_index]
        time_diff = abs((aia_time - hmi_time).total_seconds())
        if time_diff > max_time_diff_seconds:
            print(
                f"Skipping {Path(aia_path).name}: no HMI match within "
                f"{max_time_diff_seconds:g} s (nearest difference {time_diff:.1f} s)"
            )
            processed_files += 1
            continue

        figure = None
        try:
            try:
                aia_map = sunpy.map.Map(aia_path)
                normalized_aia_map = normalize_aia_exposure(aia_map)
                aligned_aia_map = align_maps_to_reference(
                    normalized_aia_map,
                    target_wcs,
                )
            except Exception as exc:
                print(f"Could not read AIA file {aia_path}: {exc}")
                processed_files += 1
                continue

            cache_key = (
                str(hmi_path),
                target_wcs.to_header().tostring(),
            )
            aligned_hmi_map = _HMI_ALIGN_CACHE.get(cache_key)
            if aligned_hmi_map is None:
                try:
                    hmi_map = sunpy.map.Map(hmi_path)
                    aligned = align_maps_to_reference(hmi_map, target_wcs)
                    # Keep a private copy so later consumers never mutate the
                    # cached reprojected map.
                    aligned_hmi_map = sunpy.map.Map(
                        aligned.data.copy(),
                        aligned.meta.copy(),
                    )
                except Exception as exc:
                    print(f"Could not read HMI file {hmi_path}: {exc}")
                    processed_files += 1
                    continue
                if len(_HMI_ALIGN_CACHE) >= _HMI_ALIGN_CACHE_MAX_ITEMS:
                    _HMI_ALIGN_CACHE.pop(next(iter(_HMI_ALIGN_CACHE)))
                _HMI_ALIGN_CACHE[cache_key] = aligned_hmi_map

            hmi_smoothed = process_hmi_magnetic_field(
                aligned_hmi_map,
                threshold,
                gaussian_sigma,
            )

            figure, axes = create_figure_with_white_background(figsize=figure_size)
            axes = figure.add_subplot(projection=aligned_aia_map)
            aligned_aia_map.plot(axes=axes, norm=norm)
            aligned_aia_map.draw_grid(axes=axes)
            hmi_smoothed.draw_contours(
                levels,
                axes=axes,
                colors=list(contour_colors),
                alpha=0.8,
                filled=False,
                linewidths=list(contour_linewidths),
            )
            axes.axis(axes.axis())
            axes.legend(
                handles=legend_elements,
                loc="upper right",
                bbox_to_anchor=(1, 1),
                frameon=True,
            )

            title_time = format_time_for_display(aia_time)
            axes.set_title(title_time, fontsize=16, pad=34)
            filename_time = aia_time
            try:
                format_utc_filename_time(filename_time)
                time_source = "observation"
            except (TypeError, ValueError):
                filename_time = batch_generated_at
                time_source = "generated"
            try:
                channel = f"{int(aligned_aia_map.wavelength.value)}a"
            except Exception:
                channel = "171a"
            output_path = destination / build_image_filename(
                ImageFilenameSpec(
                    sequence=sequence,
                    start_time=filename_time,
                    instrument="aia_hmi",
                    channel=channel,
                    product="magnetic_contour_overlay",
                    time_source=time_source,
                )
            )
            plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
            output_paths.append(output_path)
            if show_plot:
                plt.show()
        except Exception as exc:
            print(f"Could not process {aia_path}: {exc}")
            plt.close("all")
            optimized_gc_collect()
        finally:
            if figure is not None:
                plt.close(figure)

        processed_files += 1
        if processed_files % 10 == 0:
            elapsed = time.time() - start_time
            files_per_second = processed_files / elapsed if elapsed > 0 else 0.0
            monitor_memory_usage(f"Processed {processed_files}/{len(aia_files)} files")
            print(f"Processing speed: {files_per_second:.2f} files/s")

    total_time = time.time() - start_time
    monitor_memory_usage("Memory after processing")
    average_speed = processed_files / total_time if total_time > 0 else 0.0
    print(f"Processed {processed_files} files in {total_time:.2f} s")
    print(f"Average processing speed: {average_speed:.2f} files/s")
    return output_paths


__all__ = [
    "DEFAULT_AIA_DIR",
    "DEFAULT_HMI_DIR",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_ROI_BOUNDS",
    "run_overlay_workflow",
]
