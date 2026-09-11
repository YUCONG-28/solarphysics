"""Trace through the public backend and validate each returned field line."""

import warnings

import numpy as np


def uniform_seeds(frame, *, nlon=16, nlat=16, radius=1.001):
    """Equal-area seeds, away from poles and exactly-on-boundary starts."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    if min(nlon, nlat) < 1 or not np.isfinite(radius) or radius <= 1:
        raise ValueError("Use positive seed counts and radius > 1")
    lon, sinlat = np.meshgrid(
        (np.arange(nlon) + 0.5) * 360 / nlon, -1 + (np.arange(nlat) + 0.5) * 2 / nlat
    )
    return SkyCoord(
        lon.ravel() * u.deg,
        np.arcsin(sinlat.ravel()) * u.rad,
        radius * u.R_sun,
        frame=frame,
    )


def validate_line(points, *, rss, polarity, warnings_seen=(), endpoint_tolerance=0.01):
    points = np.asarray(points, dtype=float)
    reason = "ok"
    classification = "failed"
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or len(points) < 3
        or not np.isfinite(points).all()
    ):
        reason = "invalid_or_short_coordinates"
    else:
        radii = np.linalg.norm(points, axis=1)
        ends = radii[[0, -1]]
        photo = np.abs(ends - 1) <= endpoint_tolerance
        outer = np.abs(ends - rss) <= endpoint_tolerance
        if any("step" in str(w).lower() for w in warnings_seen):
            reason = "tracer_step_warning"
        elif np.any(radii < 1 - endpoint_tolerance) or np.any(
            radii > rss + endpoint_tolerance
        ):
            reason = "outside_pfss_domain"
        elif np.sum(photo) == 2:
            classification = "closed"
        elif np.sum(photo) == 1 and np.sum(outer) == 1 and polarity in (-1, 1):
            classification = "open_positive" if polarity == 1 else "open_negative"
        else:
            reason = "incomplete_boundary_termination"
    return {
        "classification": classification,
        "trace_valid": reason == "ok",
        "reason": reason,
    }


def trace_fieldlines(
    output,
    seeds,
    *,
    rss,
    max_seeds=512,
    tracer="performance",
    max_steps="auto",
    step_size=1,
):
    """Per-seed warnings are retained, and failures never count as closed lines."""
    import astropy.units as u
    from sunkit_magex.pfss.tracing import PerformanceTracer, PythonTracer

    if len(seeds) > max_seeds:
        raise ValueError("Seed count exceeds the explicit resource budget")
    if tracer not in {"performance", "python"}:
        raise ValueError("Unknown tracer")
    engine = (
        PerformanceTracer(max_steps=max_steps, step_size=step_size)
        if tracer == "performance"
        else PythonTracer(atol=1e-7, rtol=1e-7)
    )
    records = []
    for index in range(len(seeds)):
        messages = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                line = engine.trace(seeds[index : index + 1], output)[0]
            messages = [str(w.message) for w in caught]
            points = line.coords.cartesian.xyz.to_value(u.R_sun).T
            polarity = int(line.polarity)
            state = validate_line(
                points, rss=rss, polarity=polarity, warnings_seen=messages
            )
        except Exception as exc:
            points = np.empty((0, 3))
            polarity = 0
            state = dict(
                classification="failed",
                trace_valid=False,
                reason=f"{type(exc).__name__}: {exc}",
            )
        records.append(
            dict(
                fieldline_id=f"line-{index:05d}",
                seed_index=index,
                xyz_carrington_rsun=points,
                polarity=polarity,
                warnings=messages,
                tracer=tracer,
                geometry_method="conditional_pfss_field_line",
                independent_geometry_valid=False,
                **state,
            )
        )
    return records


__all__ = ["uniform_seeds", "validate_line", "trace_fieldlines"]
