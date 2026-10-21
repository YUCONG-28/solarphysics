"""Fit a deterministic synthetic source; this is not observation evidence."""

from __future__ import annotations

import argparse
import json

import numpy as np
from astropy.io import fits

from solar_toolkit.modeling.gaussian import elliptical_gaussian_2d
from solar_toolkit.radio.muser_localization import (
    evaluate_radio_model,
    fit_radio_plane,
)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    y, x = np.indices((64, 64), dtype=float)
    # Pixel coordinates, Gaussian sigma in pixels, rotation in radians,
    # and arbitrary linear intensity units are all explicit synthetic inputs.
    data = elliptical_gaussian_2d((x, y), 10.0, 24.3, 33.7, 3.0, 5.0, 0.2) + 1.0
    header = fits.Header(
        {
            "CTYPE1": "HPLN-TAN",
            "CTYPE2": "HPLT-TAN",
            "CUNIT1": "arcsec",
            "CUNIT2": "arcsec",
            "CRPIX1": 32.5,
            "CRPIX2": 32.5,
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CDELT1": 5.0,
            "CDELT2": 5.0,
        }
    )
    result = fit_radio_plane(data, header, (4, 4, 60, 60))
    assert result["fit_success"], result["quality_reasons"]
    np.testing.assert_allclose(
        [result["center_x_pixel"], result["center_y_pixel"]], [24.3, 33.7], atol=1e-3
    )
    np.testing.assert_allclose(
        evaluate_radio_model(data.shape, result), data, atol=1e-5
    )
    print(
        json.dumps(
            {
                "synthetic": True,
                "center_pixel": [result["center_x_pixel"], result["center_y_pixel"]],
                "wcs_roundtrip_error_pixel": result["wcs_roundtrip_error_pixel"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
