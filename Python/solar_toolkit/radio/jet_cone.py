"""Conditional jet-frustum geometry, independent of density and magnetic models.

Lengths are heliocentric solar radii. The origin of the cone is an observed
outer cross-section, not an assumed photospheric footpoint. Intervals describe
allowed positions on a ray, never a probability distribution or a best point.
"""

from dataclasses import dataclass

import numpy as np


def _vector(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("Expected a finite three-dimensional vector")
    return result


@dataclass(frozen=True)
class JetCone:
    origin_rsun: tuple
    axis: tuple
    width_rsun: float
    half_angle_deg: float
    evidence_id: str = "unverified"

    def __post_init__(self):
        _vector(self.origin_rsun)
        axis = _vector(self.axis)
        if np.linalg.norm(axis) == 0:
            raise ValueError("Cone axis must be nonzero")
        if not np.isfinite(self.width_rsun) or self.width_rsun < 0:
            raise ValueError("Initial radius must be finite and nonnegative")
        if not np.isfinite(self.half_angle_deg) or not 0 <= self.half_angle_deg < 90:
            raise ValueError("Half angle must be in [0, 90) degrees")

    def coordinates(self):
        a = _vector(self.axis)
        return _vector(self.origin_rsun), a / np.linalg.norm(a)


def _roots(a, b, c):
    # Coefficients use a unit direction and a shifted heliocentric origin.
    # Do not suppress small nonzero quadratic terms: they can bound long rays.
    if a == 0:
        return [] if b == 0 else [-c / b]
    disc = b * b - 4 * a * c
    tolerance = 32 * np.finfo(float).eps * (b * b + abs(4 * a * c))
    if disc < -tolerance:
        return []
    disc = max(0.0, disc)
    q = -0.5 * (b + np.copysign(np.sqrt(disc), b))
    if q == 0:
        return [-b / (2 * a)]
    return sorted(set([q / a, c / q]))


def intersect_jet_cone(observer_rsun, ray, cone):
    """Return all visible forward-ray intervals inside a semi-infinite frustum.

    Solar occultation clips at the first photospheric intersection; no arbitrary
    maximum radius is imposed. Tangencies are retained as singleton intervals.
    Every result remains conditional, irrespective of its geometric overlap.
    """
    observer = _vector(observer_rsun)
    direction = _vector(ray)
    if np.linalg.norm(direction) == 0 or np.linalg.norm(observer) <= 1:
        raise ValueError("Require a nonzero ray and an observer outside the Sun")
    direction = direction / np.linalg.norm(direction)
    anchor, axis = cone.coordinates()
    shift = -float(observer @ direction)
    center = observer + shift * direction
    delta = center - anchor
    k = np.tan(np.deg2rad(cone.half_angle_deg))
    s0, sd = float(delta @ axis), float(direction @ axis)
    perp, velocity = delta - s0 * axis, direction - sd * axis
    a = float(velocity @ velocity - (k * sd) ** 2)
    b = float(2 * (perp @ velocity - (cone.width_rsun + k * s0) * k * sd))
    c = float(perp @ perp - (cone.width_rsun + k * s0) ** 2)
    lower, upper = -shift, np.inf
    impact2 = float(center @ center)
    if impact2 < 1:
        surface = -np.sqrt(max(0.0, 1 - impact2))
        if surface >= lower:
            upper = surface
    cuts = [lower, upper]
    cuts.extend(x for x in _roots(a, b, c) if lower <= x <= upper)
    if sd != 0 and lower <= -s0 / sd <= upper:
        cuts.append(-s0 / sd)
    cuts = sorted(set(cuts))

    def inside(t):
        s = s0 + sd * t
        q = perp + t * velocity
        radius = cone.width_rsun + k * s
        eps = 2e-11 * max(1.0, np.linalg.norm(q), abs(radius))
        return s >= -eps and np.linalg.norm(q) <= radius + eps

    spans = []
    for lo, hi in zip(cuts[:-1], cuts[1:], strict=True):
        probe = (lo + hi) / 2 if np.isfinite(hi) else lo + max(1.0, abs(lo))
        if inside(probe):
            spans.append([lo, hi])
    for t in cuts:
        if np.isfinite(t) and inside(t) and not any(lo <= t <= hi for lo, hi in spans):
            spans.append([t, t])
    merged = []
    for lo, hi in sorted(spans):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    records = []
    for n, (lo, hi) in enumerate(merged):
        near = center + lo * direction
        far = center + hi * direction if np.isfinite(hi) else np.full(3, np.nan)
        tmin = np.clip(-float(center @ direction), lo, hi)
        lowest = center + tmin * direction
        height_max = (
            max(np.linalg.norm(near), np.linalg.norm(far)) - 1
            if np.isfinite(hi)
            else np.inf
        )
        records.append(
            dict(
                interval_id=n,
                distance_min_rsun=lo + shift,
                distance_max_rsun=hi + shift,
                near_xyz_rsun=near.tolist(),
                far_xyz_rsun=far.tolist(),
                minimum_height_xyz_rsun=lowest.tolist(),
                height_min_rsun=max(0.0, float(np.linalg.norm(lowest) - 1)),
                height_max_rsun=float(height_max),
                extension_min_rsun=(
                    float(min(s0 + sd * lo, s0 + sd * hi))
                    if np.isfinite(hi)
                    else float(s0 + sd * lo)
                ),
                extension_max_rsun=(
                    float(max(s0 + sd * lo, s0 + sd * hi))
                    if np.isfinite(hi)
                    else (np.inf if sd > 0 else float(s0 + sd * lo))
                ),
                method="conditional_stereo_jet_cone_extension",
                evidence_id=cone.evidence_id,
                independent_geometry_valid=False,
                interval_semantics="allowed geometry, not confidence interval",
            )
        )
    return records


def height_interval_distance(height, lower, upper):
    """Distance to a specified valid interval; zero is conditional inclusion."""
    if (
        not np.isfinite(height)
        or not np.isfinite(lower)
        or np.isnan(upper)
        or lower > upper
    ):
        return np.nan
    return float(max(lower - height, height - upper, 0.0))


__all__ = ["JetCone", "intersect_jet_cone", "height_interval_distance"]
