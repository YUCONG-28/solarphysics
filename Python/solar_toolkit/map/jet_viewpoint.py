"""Display-only solar-shell mappings with traceable native coordinates.

Only the jointly near shell hemisphere is rendered. This makes the selected
branch explicit; transparent regions are not reconstructed coronal emission.
"""

from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.ndimage import map_coordinates
from sunpy.coordinates import frames


@dataclass
class ViewMapping:
    source_x: np.ndarray
    source_y: np.ndarray
    valid: np.ndarray
    target_shape: tuple
    height_rsun: float
    reason: str = "near_shell_branch_only; not measured depth"


def observer(map_, frame):
    from .euvi_preprocessing import geometry_issues

    issues = geometry_issues(map_)
    if issues:
        raise ValueError("跨视角几何不可用：" + ", ".join(issues))
    return (
        map_.observer_coordinate.transform_to(frame).cartesian.xyz.to_value(u.m)
        / 695700000
    )


def _basis(o):
    z = o / np.linalg.norm(o)
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    return x, np.cross(z, x), z


def _rays(tx, ty, o):
    x, y, z = _basis(o)
    tx, ty = np.deg2rad(np.asarray(tx) / 3600), np.deg2rad(np.asarray(ty) / 3600)
    return (
        np.cos(ty)[..., None] * np.sin(tx)[..., None] * x
        + np.sin(ty)[..., None] * y
        - np.cos(ty)[..., None] * np.cos(tx)[..., None] * z
    )


def _project(points, o):
    x, y, z = _basis(o)
    v = points - o
    v /= np.linalg.norm(v, axis=-1)[..., None]
    return (
        np.rad2deg(np.arctan2(v @ x, -v @ z)) * 3600,
        np.rad2deg(np.arcsin(np.clip(v @ y, -1, 1))) * 3600,
    )


def map_pixels(source_map, target_map, x, y, height_rsun=0.0):
    """Map target pixels to source pixels on one explicitly assumed shell."""
    if not np.isfinite(height_rsun) or height_rsun < 0:
        raise ValueError("Display height must be finite and non-negative")
    frame = frames.HeliographicStonyhurst(obstime=target_map.date)
    ot, os = observer(target_map, frame), observer(source_map, frame)
    radius = 1 + height_rsun
    if radius >= min(np.linalg.norm(ot), np.linalg.norm(os)):
        raise ValueError("Display shell must be inside both observer distances")
    world = target_map.pixel_to_world(np.asarray(x) * u.pix, np.asarray(y) * u.pix)
    ray = _rays(world.Tx.to_value(u.arcsec), world.Ty.to_value(u.arcsec), ot)
    along = -(ray @ ot)
    disc = along**2 - (ot @ ot - radius**2)
    # Near tangencies are ill-conditioned and explicitly excluded.
    valid = (disc > radius**2 * 1e-8) & (along > 0)
    distance = along - np.sqrt(np.maximum(disc, 0))
    points = ot + distance[..., None] * ray
    valid &= np.sum(points * (os - points), axis=-1) > radius**2 * 1e-4
    tx, ty = _project(points, os)
    px = source_map.world_to_pixel(
        SkyCoord(tx * u.arcsec, ty * u.arcsec, frame=source_map.coordinate_frame)
    )
    sx, sy = np.asarray(px.x.value), np.asarray(px.y.value)
    h, w = source_map.data.shape
    valid &= (
        np.isfinite(sx)
        & np.isfinite(sy)
        & (sx >= -1e-6)
        & (sy >= -1e-6)
        & (sx <= w - 1 + 1e-6)
        & (sy <= h - 1 + 1e-6)
    )
    return np.clip(sx, 0, w - 1), np.clip(sy, 0, h - 1), valid


def build_mapping(source_map, target_map, height_rsun=0.0, stride=1):
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("Display stride must be a positive integer")
    h, w = target_map.data.shape
    yy, xx = np.mgrid[:h:stride, :w:stride]
    sx, sy, valid = map_pixels(source_map, target_map, xx, yy, height_rsun)
    return ViewMapping(sx, sy, valid, (h, w), height_rsun)


def sample_mapping(data, mapping):
    finite = np.isfinite(data)
    coordinates = [mapping.source_y, mapping.source_x]
    values = map_coordinates(
        np.where(finite, data, 0.0),
        coordinates,
        order=1,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    support = map_coordinates(
        finite.astype(float),
        coordinates,
        order=1,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    return np.where(mapping.valid & (support > 0.9999), values, np.nan)


def reproject_photosphere(source_map, target_map, data, *, stride=1, mapping=None):
    """Official SunPy preview, with native mapping masks and unchanged input WCS.

    Both temporary maps use the project's fixed physical radius. Display centres
    remain at native reference pixels 0, stride, ... (not bin midpoints).
    """
    import sunpy.map

    if np.shape(data) != source_map.data.shape:
        raise ValueError("Preview data must match the native source map")
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("Display stride must be a positive integer")
    meta = source_map.meta.copy()
    meta["rsun_ref"] = 695700000.0
    source = sunpy.map.Map(data, meta)
    header = target_map.wcs.to_header()
    header["RSUN_REF"] = 695700000.0
    for axis in [1, 2]:
        header[f"CRPIX{axis}"] = (header[f"CRPIX{axis}"] - 1) / stride + 1
        header[f"CDELT{axis}"] *= stride
    shape = tuple((n + stride - 1) // stride for n in target_map.data.shape)
    reference = sunpy.map.Map(np.zeros(shape, dtype=np.uint8), header)
    result, footprint = source.reproject_to(reference.wcs, return_footprint=True)
    if mapping is None:
        mapping = build_mapping(source_map, target_map, 0, stride)
    if mapping.height_rsun != 0 or mapping.valid.shape != shape:
        raise ValueError(
            "Official photospheric preview requires a matching height-zero mapping"
        )
    # The stricter native mask excludes tangencies, back branches and NaN support.
    supported = np.isfinite(sample_mapping(data, mapping))
    return np.where(supported & (footprint > 0), result.data, np.nan)


def native_roi_mask(source_map, target_map, vertices, height_rsun=0.0):
    """Rasterize a display ROI back at native source centres without hole filling."""
    from matplotlib.path import Path

    pts = np.asarray(vertices, float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3 or not np.isfinite(pts).all():
        raise ValueError("Select a finite polygon with at least three vertices")
    polygon = Path(np.vstack([pts, pts[0]]))
    result = np.zeros(source_map.data.shape, dtype=bool)
    for start in range(0, result.shape[0], 128):
        stop = min(start + 128, result.shape[0])
        yy, xx = np.mgrid[start:stop, : result.shape[1]]
        tx, ty, valid = map_pixels(target_map, source_map, xx, yy, height_rsun)
        inside = polygon.contains_points(
            np.c_[tx.ravel(), ty.ravel()], radius=1e-8
        ).reshape(xx.shape)
        result[start:stop] = inside & valid & np.isfinite(source_map.data[start:stop])
    return result


def epipolar_pixels(source_map, target_map, xy):
    """Exact epipolar locus in target native pixels; no depth or snapping."""
    frame = frames.HeliographicStonyhurst(obstime=target_map.date)
    a, b = observer(source_map, frame), observer(target_map, frame)
    world = source_map.pixel_to_world(xy[0] * u.pix, xy[1] * u.pix)
    ray = _rays(world.Tx.to_value(u.arcsec), world.Ty.to_value(u.arcsec), a)
    normal = np.cross(ray, b - a)
    if np.linalg.norm(normal) < 1e-7:
        return np.empty((0, 2))
    normal /= np.linalg.norm(normal)
    ex, ey, ez = _basis(b)
    if abs(normal @ ey) < 1e-8:
        return np.empty((0, 2))
    h, w = target_map.data.shape
    corners = target_map.pixel_to_world(
        np.array([0, w - 1]) * u.pix, np.array([h / 2, h / 2]) * u.pix
    )
    tx = np.linspace(
        corners.Tx.to_value(u.rad).min(), corners.Tx.to_value(u.rad).max(), 400
    )
    ty = np.arctan(
        (normal @ ez * np.cos(tx) - normal @ ex * np.sin(tx)) / (normal @ ey)
    )
    px = target_map.world_to_pixel(
        SkyCoord(tx * u.rad, ty * u.rad, frame=target_map.coordinate_frame)
    )
    return np.c_[px.x.value, px.y.value]


__all__ = [
    "ViewMapping",
    "observer",
    "map_pixels",
    "build_mapping",
    "sample_mapping",
    "reproject_photosphere",
    "native_roi_mask",
    "epipolar_pixels",
]
