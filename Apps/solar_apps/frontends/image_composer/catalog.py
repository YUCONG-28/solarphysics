"""Image discovery and timestamp extraction for the image composer."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image

from .models import ImageRecord

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
)

_EXIF_IFD_TAG = 34665
_EXIF_TIME_FIELDS = (
    (36867, 37521, "exif:DateTimeOriginal"),
    (36868, 37522, "exif:DateTimeDigitized"),
    (306, 37520, "exif:DateTime"),
)
_FILENAME_PATTERNS = (
    re.compile(
        r"(?<!\d)(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})"
        r"[T_ -]?(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})"
        r"(?:[._-](?P<fraction>\d{1,6}))?(?!\d)"
    ),
    re.compile(
        r"(?<!\d)(?P<year>\d{4})[-_](?P<month>\d{2})[-_]"
        r"(?P<day>\d{2})[T_ -](?P<hour>\d{2})[-_:]"
        r"(?P<minute>\d{2})[-_:](?P<second>\d{2})"
        r"(?:[._-](?P<fraction>\d{1,6}))?(?!\d)"
    ),
)


def natural_key(path: str | Path) -> list[int | str]:
    """Return a case-insensitive key where digit runs compare numerically."""

    name = Path(path).name
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", name)
    ]


def discover_images(folder: str | Path) -> list[Path]:
    """Return supported files from the first folder level in natural order."""

    root = Path(folder).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Folder does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a folder: {root}")
    files = [
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    ]
    return sorted(files, key=natural_key)


def scan_folder(folder: str | Path) -> list[ImageRecord]:
    """Discover images and extract one auditable timestamp for each image."""

    records: list[ImageRecord] = []
    for ordinal, path in enumerate(discover_images(folder), start=1):
        timestamp, source = extract_timestamp(path)
        records.append(
            ImageRecord(
                ordinal=ordinal,
                path=path,
                timestamp=timestamp,
                time_source=source,
            )
        )
    return records


def extract_timestamp(path: str | Path) -> tuple[datetime, str]:
    """Extract EXIF, filename, or local modification time in fixed priority."""

    candidate = Path(path)
    exif_time = _timestamp_from_exif(candidate)
    if exif_time is not None:
        return exif_time
    filename_time = timestamp_from_filename(candidate.name)
    if filename_time is not None:
        return filename_time, "filename"
    return datetime.fromtimestamp(candidate.stat().st_mtime), "mtime"


def timestamp_from_filename(filename: str) -> datetime | None:
    for pattern in _FILENAME_PATTERNS:
        match = pattern.search(filename)
        if match is None:
            continue
        fields = match.groupdict()
        fraction = (fields.get("fraction") or "").ljust(6, "0")[:6]
        try:
            return datetime(
                int(fields["year"]),
                int(fields["month"]),
                int(fields["day"]),
                int(fields["hour"]),
                int(fields["minute"]),
                int(fields["second"]),
                int(fraction or 0),
            )
        except ValueError:
            continue
    return None


def _timestamp_from_exif(path: Path) -> tuple[datetime, str] | None:
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            values: dict[int, Any] = dict(exif)
            try:
                exif_ifd = ExifTags.IFD.Exif
            except AttributeError:
                exif_ifd = _EXIF_IFD_TAG
            try:
                values.update(exif.get_ifd(exif_ifd))
            except KeyError, TypeError, ValueError:
                pass
    except OSError, ValueError:
        return None

    for time_tag, subsecond_tag, source in _EXIF_TIME_FIELDS:
        parsed = _parse_exif_datetime(values.get(time_tag), values.get(subsecond_tag))
        if parsed is not None:
            return parsed, source
    return None


def _parse_exif_datetime(value: Any, subsecond: Any) -> datetime | None:
    text = _text_value(value)
    if not text:
        return None
    try:
        parsed = datetime.strptime(text.strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None
    fraction = re.sub(r"\D", "", _text_value(subsecond) or "")[:6]
    if fraction:
        parsed = parsed.replace(microsecond=int(fraction.ljust(6, "0")))
    return parsed


def _text_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip("\x00")
    return str(value).strip("\x00")
