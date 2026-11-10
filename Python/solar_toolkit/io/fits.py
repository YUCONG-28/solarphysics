"""FITS input helpers.

English: Read a data array and copied header from a selected FITS HDU.

中文：从指定 FITS HDU 读取数据数组及其独立的 header 副本。
"""

from __future__ import annotations

import gzip
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path


def resolve_fits_path(path: str | Path) -> Path:
    """Return an existing explicit path, otherwise try only ``path + '.gz'``.

    No instrument, extension or basename is inferred. An existing explicit
    file always wins, including when a compressed counterpart also exists.
    """
    source = Path(path).expanduser()
    if source.is_file():
        return source
    if source.exists():
        raise IsADirectoryError(f"Not a FITS file: {source}")
    compressed = Path(str(source) + ".gz")
    if source.suffix.casefold() != ".gz" and compressed.is_file():
        return compressed
    raise FileNotFoundError(f"FITS file does not exist: {source}")


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _check_identity(path: Path, expected: tuple[int, int, int, int]) -> None:
    try:
        unchanged = _file_identity(path) == expected
    except OSError:
        unchanged = False
    if not unchanged:
        raise RuntimeError(f"FITS source changed during access: {path}")


@contextmanager
def materialize_fits(path: str | Path, *, cache_root: str | Path | None = None):
    """Yield a seekable path for one call, cleaning any gzip expansion on exit.

    Plain files are yielded directly. Stream-gzip input is fully expanded into
    a unique temporary directory, beneath ``cache_root`` when supplied or the
    system temporary directory otherwise. No expansion is reused or retained.
    The caller must close readers before leaving the context. Source device,
    inode, size and nanosecond mtime are checked after expansion and a successful
    call; consumer errors propagate unchanged. Source bytes are never modified.
    Tile-compressed FITS are left to Astropy at an explicit existing path.
    """
    source = resolve_fits_path(path)
    identity = _file_identity(source)
    if source.suffix.casefold() != ".gz":
        yield source
        _check_identity(source, identity)
        return
    directory = None if cache_root is None else Path(cache_root).expanduser()
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="solar-fits-", dir=directory) as temporary:
        expanded = Path(temporary) / source.name[:-3]
        with gzip.open(source, "rb") as reader, expanded.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
        _check_identity(source, identity)
        yield expanded
        _check_identity(source, identity)


def read_fits_data_header(path: str | Path, *, hdu_index: int | None = 0):
    """Return detached data and a copied header, closing all file resources.

    FITS scaling and integer BLANK handling follow Astropy. An empty HDU keeps
    its ``None`` data. Large-image streaming should use ``fits.open`` and its
    section interface in an explicit :func:`materialize_fits` context instead.
    The legacy default remains HDU 0, including its empty ``None`` data. Explicit
    ``hdu_index=None`` selects the first nonempty image HDU and merges its header
    over primary metadata. Integers retain explicit Astropy HDU selection.
    """

    import numpy as np
    from astropy.io import fits

    with materialize_fits(path) as readable, fits.open(readable, memmap=False) as hdul:
        if hdu_index is None:
            for hdu in hdul:
                if (
                    isinstance(hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU))
                    and hdu.data is not None
                ):
                    header = hdul[0].header.copy()
                    header.update(hdu.header)
                    return np.array(hdu.data, copy=True), header
            raise ValueError(f"No image HDU found in FITS file: {path}")
        hdu = hdul[hdu_index]
        data = None if hdu.data is None else np.array(hdu.data, copy=True)
        return data, hdu.header.copy()


def _read_solar_map(path: str | Path):
    """Construct one detached SunPy map from the first nonempty image HDU."""
    import sunpy.map

    data, header = read_fits_data_header(path, hdu_index=None)
    return sunpy.map.Map(data, header)


__all__ = ["materialize_fits", "read_fits_data_header", "resolve_fits_path"]
