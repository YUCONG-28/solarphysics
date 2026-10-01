"""HMI magnetogram plotting workflow.

English: Plot local HMI FITS files after reprojecting them to a fixed
helioprojective region of interest. Importing this module does not inspect the
filesystem or create figures.

中文：将本地 HMI FITS 文件重投影到固定的日面投影感兴趣区域后绘图。导入本模块不会扫描
文件系统或创建图形。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from solar_toolkit.visualization.image_naming import (
    ImageFilenameSpec,
    build_image_filename,
    format_utc_filename_time,
)

DEFAULT_DATA_DIR = Path("data/hmi")
DEFAULT_OUTPUT_DIR = Path("outputs/hmi")
DEFAULT_ROI_BOUNDS = (180.0, -340.0, 520.0, 20.0)


def run_magnetogram_workflow(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    roi_bounds: tuple[float, float, float, float] = DEFAULT_ROI_BOUNDS,
    frame_count: int = 1,
    show_plot: bool = False,
    dpi: int = 200,
) -> list[Path]:
    """Render aligned HMI magnetograms and return the generated image paths.

    Parameters
    ----------
    data_dir
        Directory containing the input ``.fits`` files.
    output_dir
        Directory receiving images under the shared scientific naming contract.
    roi_bounds
        ``(x_min, y_min, x_max, y_max)`` in arcseconds. The default preserves
        the historical active-region crop.
    frame_count
        Number of aligned frames to render. The historical workflow rendered
        only the first frame.
    show_plot
        Display each figure interactively after saving it.
    dpi
        Output resolution passed to Matplotlib.
    """

    import astropy.units as u
    import matplotlib.pyplot as plt
    import sunpy.map
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import propagate_with_solar_surface

    input_dir = Path(data_dir)
    file_paths = sorted(
        path for path in input_dir.iterdir() if path.suffix.lower() == ".fits"
    )
    if len(file_paths) < 2:
        raise ValueError(
            "At least two HMI FITS files are required to define the target WCS"
        )
    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")

    hmi_sequence = sunpy.map.Map(file_paths, sequence=True)
    reference_map = sunpy.map.Map(hmi_sequence[1])
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

    with propagate_with_solar_surface():
        aligned_sequence = sunpy.map.Map(
            [solar_map.reproject_to(cutout_map.wcs) for solar_map in hmi_sequence],
            sequence=True,
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    limit = min(frame_count, len(aligned_sequence), len(file_paths))
    batch_generated_at = dt.datetime.now(dt.timezone.utc)
    for index in range(limit):
        figure = plt.figure()
        axes = figure.add_subplot(projection=aligned_sequence[index])
        aligned_sequence[index].plot(axes=axes)

        base_name = file_paths[index].name
        observation_time = getattr(aligned_sequence[index], "date", None)
        try:
            format_utc_filename_time(observation_time)
            time_source = "observation"
        except (TypeError, ValueError):
            observation_time = batch_generated_at
            time_source = "generated"
        output_path = destination / build_image_filename(
            ImageFilenameSpec(
                sequence=index + 1,
                start_time=observation_time,
                instrument="hmi",
                product="magnetogram",
                time_source=time_source,
            )
        )
        time_str = base_name.split(".")[2]
        plt.title(time_str)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        output_paths.append(output_path)

        if show_plot:
            plt.show()
        plt.close(figure)

    return output_paths


__all__ = [
    "DEFAULT_DATA_DIR",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_ROI_BOUNDS",
    "run_magnetogram_workflow",
]
