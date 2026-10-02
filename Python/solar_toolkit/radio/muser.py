"""MUSER-L combined image and PWR_I spectrum readers.

Frequency extensions, rather than a primary-header frequency, are authoritative.
"""

from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

__all__ = [
    "MuserSpectrum",
    "read_muser_spectrum",
    "read_muser_images",
    "peak_level",
    "hpc_grid",
]


@dataclass(frozen=True)
class MuserSpectrum:
    data: np.ndarray
    frequency_mhz: np.ndarray
    time: Time
    unit: str


def read_muser_spectrum(path):
    """Read total power I without assuming XX or YY is circular polarization."""
    with fits.open(path, memmap=False) as hdus:
        if hdus[0].header.get("POLARIZA", "").strip() != "PWR_I":
            raise ValueError("A PWR_I spectrum is required")
        data = np.array(hdus[0].data, dtype=float)
        freq = np.array(hdus["FMHZ"].data["FMHZ"], dtype=float).ravel()
        time = Time(
            np.array(hdus["TIME"].data["TIME"]).ravel(), format="jd", scale="utc"
        )
        unit = hdus[0].header.get("BUNIT", "Arbitrary")
    if data.shape != (len(freq), len(time)):
        raise ValueError("Spectrum axes do not match its data")
    order = np.argsort(freq, kind="stable")
    freq, data = freq[order], data[order]
    if (
        not np.all(np.isfinite(freq))
        or not np.all(np.diff(time.jd) > 0)
        or not np.all(np.diff(freq) > 0)
    ):
        raise ValueError("Spectrum axes must increase strictly")
    return MuserSpectrum(data, freq, time, unit)


def read_muser_images(path):
    """Return frequency, image, 2D celestial header and beam for every SPW."""
    with fits.open(path, memmap=False) as hdus:
        header = hdus[0].header
        if header.get("CTYPE4") != "STOKES" or header.get("CRVAL4") != 1:
            raise ValueError("Only explicit Stokes I images are supported")
        data = hdus[0].data
        if data.ndim != 5 or data.shape[1:3] != (1, 1):
            raise ValueError("Expected (frequency, 1, 1, y, x) combined image")
        freq = np.asarray(hdus["FMHZ"].data).ravel()
        if len(freq) != data.shape[0]:
            raise ValueError("Image frequency extension does not match its planes")
        base = WCS(header, naxis=2).to_header()
        for key in [
            "DATE-OBS",
            "HGLN_OBS",
            "HGLT_OBS",
            "DSUN_OBS",
            "RSUN_OBS",
            "RSUN_REF",
            "BUNIT",
            "TELESCOP",
        ]:
            if key in header:
                base[key] = header[key]
        result = []
        for i, frequency in enumerate(freq):
            plane_header = base.copy()
            beam = {key: float(hdus[key].data[i]) for key in ["BMAJ", "BMIN", "BPA"]}
            for key, value in beam.items():
                plane_header[key] = value
            result.append(
                (
                    float(frequency),
                    np.array(data[i, 0, 0], dtype=float),
                    plane_header,
                    beam,
                )
            )
    return result


def peak_level(image, fraction=0.9):
    """A contour level relative to the finite positive image peak."""
    positive = np.asarray(image)[np.isfinite(image) & (np.asarray(image) > 0)]
    if positive.size == 0 or not 0 < fraction < 1:
        raise ValueError("A positive finite peak and 0 < fraction < 1 are required")
    return float(positive.max() * fraction)


def hpc_grid(header, shape):
    """Project pixel centers to the observed HPLN/HPLT angular grid, in arcsec.

    No solar-surface assumption, radius rescaling, or empirical shift is applied.
    """
    if not str(header.get("CTYPE1", "")).startswith("HPLN") or not str(
        header.get("CTYPE2", "")
    ).startswith("HPLT"):
        raise ValueError("Explicit helioprojective WCS is required")
    y, x = np.indices(shape)
    lon, lat = WCS(header, naxis=2).all_pix2world(x, y, 0)
    return ((lon + 180) % 360 - 180) * 3600, lat * 3600
