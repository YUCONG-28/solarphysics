"""FITS input helpers.

English: Read a data array and copied header from a selected FITS HDU.

中文：从指定 FITS HDU 读取数据数组及其独立的 header 副本。
"""

from __future__ import annotations

from pathlib import Path


def read_fits_data_header(path: str | Path, *, hdu_index: int = 0):
    """Return detached data and a copied header, closing all file resources.

    FITS scaling and integer BLANK handling follow Astropy. An empty HDU keeps
    its ``None`` data. Large-image streaming should use ``fits.open`` and its
    section interface in an explicit caller-owned context instead.
    """

    import numpy as np
    from astropy.io import fits

    with fits.open(path, memmap=False) as hdul:
        hdu = hdul[hdu_index]
        data = None if hdu.data is None else np.array(hdu.data, copy=True)
        return data, hdu.header.copy()


__all__ = ["read_fits_data_header"]
