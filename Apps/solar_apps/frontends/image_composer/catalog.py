# SPDX-License-Identifier: MIT
"""Image discovery and timestamp extraction for the image composer."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image

from .models import ImageRecord

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
)

_EXIF_IFD_TAG = 34665
_EXIF_TIME_FIELDS = (
    (36867, 37521, 36881, "exif:DateTimeOriginal"),
    (36868, 37522, 36882, "exif:DateTimeDigitized"),
    (306, 37520, 36880, "exif:DateTime"),
)
_FILENAME_PATTERNS = (
    re.compile(
        r"(?<!\d)(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})"
        r"[T_ -]?(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})"
        r"(?:[._-](?P<fraction>\d{1,9}))?(?!\d)"
    ),
    re.compile(
        r"(?<!\d)(?P<year>\d{4})[-_](?P<month>\d{2})[-_]"
        r"(?P<day>\d{2})[T_ -](?P<hour>\d{2})[-_:]"
        r"(?P<minute>\d{2})[-_:](?P<second>\d{2})"
        r"(?:[._-](?P<fraction>\d{1,9}))?(?!\d)"
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
        timestamp_text, source = extract_timestamp_text(path)
        timestamp = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
        records.append(
            ImageRecord(
                ordinal=ordinal,
                path=path,
                timestamp=timestamp,
                time_source=source,
                timestamp_text=timestamp_text,
                readable=_image_readable(path),
            )
        )
    return records


def _image_readable(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except OSError, ValueError:
        return False


def extract_timestamp(path: str | Path) -> tuple[datetime, str]:
    """Return an aware UTC time; the scanner retains the full precision text."""

    text, source = extract_timestamp_text(path)
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC), source


def extract_timestamp_text(path: str | Path) -> tuple[str, str]:
    """Read EXIF/filename as UTC by default; mtime is always explicit UTC."""

    candidate = Path(path)
    exif_time = _timestamp_from_exif(candidate)
    if exif_time is not None:
        return exif_time
    filename_time = _filename_timestamp_text(candidate.name)
    if filename_time is not None:
        return filename_time, "filename"
    seconds, nanoseconds = divmod(candidate.stat().st_mtime_ns, 1_000_000_000)
    base = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds)
    return f"{base:%Y-%m-%dT%H:%M:%S}.{nanoseconds:09d}Z", "mtime"


def timestamp_from_filename(filename: str) -> datetime | None:
    text = _filename_timestamp_text(filename)
    return (
        datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)
        if text
        else None
    )


def _filename_timestamp_text(filename: str) -> str | None:
    for pattern in _FILENAME_PATTERNS:
        match = pattern.search(filename)
        if match is None:
            continue
        fields = match.groupdict()
        fraction = fields.get("fraction") or ""
        # A negative ISO offset can resemble the legacy dash-fraction form.
        negative_zone = (
            "-" + fraction + filename[match.end() : match.end() + 3]
            if len(fraction) == 2
            and match.group().endswith("-" + fraction)
            and re.match(r":\d{2}", filename[match.end() :])
            else None
        )
        if negative_zone:
            fraction = ""
        zone_match = re.match(
            r"(Z|[+]\d{2}:?\d{2}|-\d{2}:\d{2})", filename[match.end() :], re.I
        )
        zone_text = negative_zone or (zone_match.group(1) if zone_match else "+00:00")
        if zone_text.upper() == "Z":
            zone_text = "+00:00"
        elif len(zone_text) == 5:
            zone_text = zone_text[:3] + ":" + zone_text[3:]
        text = (
            f"{fields['year']}-{fields['month']}-{fields['day']}T"
            f"{fields['hour']}:{fields['minute']}:{fields['second']}"
            + (f".{fraction}" if fraction else "")
            + zone_text
        )
        try:
            datetime.fromisoformat(text)
        except ValueError:
            continue
        return text
    return None


def _timestamp_from_exif(path: Path) -> tuple[str, str] | None:
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

    for time_tag, subsecond_tag, offset_tag, source in _EXIF_TIME_FIELDS:
        parsed = _parse_exif_datetime(values.get(time_tag), values.get(subsecond_tag))
        if parsed is not None:
            fraction = re.sub(r"\D", "", _text_value(values.get(subsecond_tag)) or "")[
                :9
            ]
            zone = _text_value(values.get(offset_tag)) or "+00:00"
            try:
                datetime.fromisoformat("2000-01-01T00:00:00" + zone)
            except ValueError:
                continue
            text = (
                parsed.strftime("%Y-%m-%dT%H:%M:%S")
                + (f".{fraction}" if fraction else "")
                + zone
            )
            return text, source
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
    return parsed.replace(tzinfo=UTC)


def _text_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip("\x00")
    return str(value).strip("\x00")
