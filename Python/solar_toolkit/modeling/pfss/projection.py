"""Project genuinely 3D field lines, explicitly freezing their Carrington pattern."""

import numpy as np


def visible_from_observer(xyz, observer_xyz):
    """Visibility of finite 3D points, testing the observer-to-source segment."""
    xyz = np.asarray(xyz, float)
    obs = np.asarray(observer_xyz, float)
    delta = xyz - obs
    denom = np.sum(delta**2, axis=1)
    t = -np.sum(obs * delta, axis=1) / np.where(denom > 0, denom, np.nan)
    closest = obs + t[:, None] * delta
    hidden = (t > 0) & (t < 1) & (np.linalg.norm(closest, axis=1) < 1)
    return ~hidden & np.isfinite(xyz).all(axis=1) & (denom > 0)


def project_fieldlines(records, *, observer, obstime):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import frames

    frame = frames.HeliographicCarrington(obstime=obstime, observer="earth")
    obs = observer.transform_to(frame).cartesian.xyz.to_value(u.R_sun)
    projected = []
    for record in records:
        xyz = np.asarray(record["xyz_carrington_rsun"])
        if len(xyz) == 0:
            projected.append(
                dict(
                    fieldline_id=record["fieldline_id"],
                    hpc_arcsec=np.empty((0, 2)),
                    visible=np.zeros(0, bool),
                )
            )
            continue
        coords = SkyCoord(
            x=xyz[:, 0] * u.R_sun,
            y=xyz[:, 1] * u.R_sun,
            z=xyz[:, 2] * u.R_sun,
            representation_type="cartesian",
            frame=frame,
        )
        hpc = coords.transform_to(
            frames.Helioprojective(observer=observer, obstime=obstime)
        )
        xy = np.column_stack([hpc.Tx.to_value(u.arcsec), hpc.Ty.to_value(u.arcsec)])
        visible = visible_from_observer(xyz, obs) & np.isfinite(xy).all(axis=1)
        xy[~visible] = np.nan
        # Do not connect discontinuous longitudes/branches through the image.
        jump = np.r_[False, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1800]
        xy[jump] = np.nan
        projected.append(
            dict(
                fieldline_id=record["fieldline_id"],
                hpc_arcsec=xy,
                visible=visible & ~jump,
                time_assumption="frozen Carrington pattern",
                obstime=str(obstime),
            )
        )
    return projected


__all__ = ["visible_from_observer", "project_fieldlines"]
