"""Application-layer helpers for shared scientific image filenames."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from solar_toolkit.radio._image_naming import build_radio_image_filename
from solar_toolkit.visualization.image_naming import (
    ImageFilenameSpec,
    build_image_filename,
    format_utc_filename_time,
)

IMAGE_SUFFIXES = frozenset(
    {".bmp", ".gif", ".jpeg", ".jpg", ".jp2", ".png", ".svg", ".tif", ".tiff", ".webp"}
)
# Fixed names are reserved for internal browser protocols and are never used
# as suggestions for user-visible scientific-image downloads.
INTERNAL_FIXED_IMAGE_NAMES = frozenset({"preview.png", "thumbnail.png"})


def build_scientific_image_filename(
    *,
    sequence: int,
    start_time: Any,
    instrument: str,
    product: str | Sequence[str],
    generated_at: dt.datetime,
    end_time: Any | None = None,
    channel: Any | None = None,
    polarization: str | Sequence[str] | None = None,
    qualifiers: str | Sequence[str] = (),
    extension: str = ".png",
) -> str:
    """Build one name, using the caller's single batch fallback if needed."""

    try:
        start_text = format_utc_filename_time(start_time)
        if end_time is not None:
            end_text = format_utc_filename_time(end_time)
            if end_text < start_text:
                start_time, end_time = end_time, start_time
        time_source = "observation"
    except (TypeError, ValueError):
        start_time = generated_at
        end_time = None
        time_source = "generated"
    return build_image_filename(
        ImageFilenameSpec(
            sequence=sequence,
            start_time=start_time,
            end_time=end_time,
            instrument=instrument,
            channel=channel,
            polarization=polarization,
            product=product,
            qualifiers=qualifiers,
            extension=extension,
            time_source=time_source,
        )
    )


def configured_radio_image_path(
    output_dir: str | Path,
    configured_name: str,
    data,
    *,
    sequence: int,
    product: str,
    generated_at: dt.datetime,
    frequency_mhz: float | None = None,
    polarization: str | Sequence[str] | None = None,
) -> Path:
    """Honor explicit image filenames; otherwise treat config as a product id."""

    configured = str(configured_name or product).strip()
    configured_path = Path(configured)
    if configured_path.suffix.casefold() in IMAGE_SUFFIXES:
        return Path(output_dir) / configured_path
    return Path(output_dir) / build_radio_image_filename(
        data,
        sequence=sequence,
        product=configured or product,
        generated_at=generated_at,
        frequency_mhz=frequency_mhz,
        polarization=polarization,
    )


def configured_scientific_image_path(
    configured_path: str | Path | None,
    *,
    output_dir: str | Path = ".",
    sequence: int,
    start_time: Any,
    instrument: str,
    product: str,
    generated_at: dt.datetime,
    end_time: Any | None = None,
    channel: Any | None = None,
    polarization: str | Sequence[str] | None = None,
    qualifiers: str | Sequence[str] = (),
) -> Path:
    """Preserve an explicit image path or generate a contract filename."""

    if configured_path not in (None, ""):
        explicit = Path(configured_path)
        if explicit.suffix.casefold() in IMAGE_SUFFIXES:
            return explicit
    return Path(output_dir) / build_scientific_image_filename(
        sequence=sequence,
        start_time=start_time,
        end_time=end_time,
        instrument=instrument,
        channel=channel,
        polarization=polarization,
        product=product,
        qualifiers=qualifiers,
        generated_at=generated_at,
    )
