"""Within-observer angular resampling and audited residual translation."""

import astropy.units as u
import numpy as np
import sunpy.map
from astropy.coordinates import SkyCoord
from astropy.io import fits
from scipy import ndimage as ndi
from scipy.optimize import least_squares

from .jet_annotations import file_sha256, intensity_per_second, observation_info

__all__ = ["registered_difference"]


def registered_difference(current, previous_path, original_hash, target):
    """WCS map followed by a bounded within-view translation audit.

    Two disjoint peripheral regions assess registration consistency. They are
    engineering background assumptions, not a verified stationary baseline.
    """
    previous = sunpy.map.Map(previous_path)
    yy, xx = np.indices(current.data.shape, dtype=float)
    world = current.pixel_to_world(xx * u.pix, yy * u.pix)
    # Resample angular coordinates within one observer; do not force off-limb
    # pixels onto the photosphere during a cross-time SkyCoord transformation.
    angular = SkyCoord(world.Tx, world.Ty, frame=previous.coordinate_frame)
    pixels = previous.world_to_pixel(angular)
    prior = ndi.map_coordinates(
        intensity_per_second(previous),
        [pixels.y.value, pixels.x.value],
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )
    data = np.asarray(current.data, float)
    height, width = data.shape
    edge = (
        (xx < 0.2 * width)
        | (xx > 0.8 * width)
        | (yy < 0.2 * height)
        | (yy > 0.8 * height)
    )
    finite = (
        ndi.binary_erosion(np.isfinite(data) & np.isfinite(prior), iterations=6) & edge
    )
    scale = max(1, float(np.nanmedian(abs(data))))
    ref = np.arcsinh(data / scale)
    moving = np.arcsinh(prior / scale)
    # Split into checkerboard blocks; each split covers all four image edges.
    split = ((xx // 16 + yy // 16).astype(int) % 2) == 0
    shifts = []
    for region in (finite & split, finite & ~split):
        points = np.argwhere(region)[:: max(1, int(region.sum() / 3000))]
        if len(points) < 50:
            raise ValueError("insufficient_peripheral_registration_pixels")
        values = ref[points[:, 0], points[:, 1]]

        def residual(params, points=points, values=values):
            sampled = ndi.map_coordinates(
                moving,
                [points[:, 0] + params[0], points[:, 1] + params[1]],
                order=1,
                mode="constant",
                cval=np.nan,
                prefilter=False,
            )
            difference = sampled - values
            # Unknown residuals are excluded, never intensity zeros.
            return np.where(np.isfinite(difference), difference, 0)

        fit = least_squares(
            residual,
            [0, 0],
            bounds=(-4.9, 4.9),
            loss="soft_l1",
            f_scale=0.1,
            diff_step=0.1,
        )
        shifts.append(fit.x)
    disagreement = float(np.linalg.norm(shifts[0] - shifts[1]))
    audit = {
        "train_dy_dx_pixel": shifts[0].tolist(),
        "heldout_dy_dx_pixel": shifts[1].tolist(),
        "disagreement_pixel": disagreement,
        "stable_region_assumption": "outer_20_percent_unverified",
        "previous_image": str(previous_path),
        "previous_sha256": file_sha256(previous_path),
        "baseline": "previous_frame_not_static_background",
    }
    if disagreement > 2 or max(abs(shifts[0])) >= 4.85:
        audit["status"] = "registration_not_accepted"
        return None, audit
    aligned = ndi.map_coordinates(
        prior,
        [yy + shifts[0][0], xx + shifts[0][1]],
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )
    delta = data - aligned
    header = current.fits_header.copy()
    header["JETREG"] = True
    header["JORIGHSH"] = original_hash
    header["JREGERR"] = disagreement
    header["JREGDY"] = shifts[0][0]
    header["JREGDX"] = shifts[0][1]
    header["JREFDATE"] = observation_info(previous)["midpoint_utc"]
    fits.writeto(target, delta, header)
    audit["status"] = "accepted_engineering_registration_not_identity_validation"
    return target, audit
