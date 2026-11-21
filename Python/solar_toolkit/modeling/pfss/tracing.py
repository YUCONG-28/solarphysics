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
    sample_field_strength=False,
):
    """Trace geometry, optionally sampling aligned field magnitudes in Gauss.

    Sampling is explicit and uses public ``Output.get_bvec`` on copied line
    coordinates; geometry and seeds are unchanged. ``b_gauss`` has one value
    per returned XYZ point, with NaN for unavailable samples. Sampling failures
    have their own status/reason and never turn a valid trace into a failed one.
    The default records retain their previous fields and require no sampling.
    """
    import astropy.units as u

    if not isinstance(sample_field_strength, bool):
        raise ValueError("sample_field_strength must be a boolean")
    if len(seeds) > max_seeds:
        raise ValueError("Seed count exceeds the explicit resource budget")
    if tracer not in {"performance", "python"}:
        raise ValueError("Unknown tracer")
    from sunkit_magex.pfss.tracing import PerformanceTracer, PythonTracer

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
            coordinates = line.coords
            points = coordinates.cartesian.xyz.to_value(u.R_sun).T
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
        record = dict(
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
        if sample_field_strength:
            record.update(
                _sample_field(output, coordinates, points)
                if state["trace_valid"]
                else dict(
                    b_gauss=np.full(len(points), np.nan),
                    field_strength_status="trace_invalid",
                    field_strength_reason="Trace failed validation",
                    field_strength_warnings=[],
                )
            )
        records.append(record)
    return records


def _sample_field(output, coordinates, points):
    """Sample without mutating the coordinate representation used by tracing."""
    import astropy.units as u

    from solar_toolkit._utils.numeric import as_value

    magnitudes = np.full(len(points), np.nan)
    caught = []
    reason = ""
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sampled = output.get_bvec(coordinates.copy(), out_type="cartesian")
            if not isinstance(sampled, u.Quantity):
                raise ValueError("Backend field vectors must declare magnetic units")
            vectors = as_value(sampled, u.G, "backend field vectors")
        if vectors.shape != (len(points), 3):
            raise ValueError("Backend field vectors must have shape (N, 3)")
        magnitudes = np.hypot.reduce(vectors, axis=1)
        magnitudes[~np.isfinite(magnitudes)] = np.nan
        finite = np.isfinite(magnitudes)
        status = (
            "ok" if finite.all() else "partial" if finite.any() else "sampling_failed"
        )
        if not finite.all():
            reason = "Backend returned unavailable field samples; no interpolation or extrapolation applied"
    except Exception as exc:
        status = "sampling_failed"
        reason = f"{type(exc).__name__}: {exc}"
    return dict(
        b_gauss=magnitudes,
        field_strength_status=status,
        field_strength_reason=reason,
        field_strength_warnings=[str(w.message) for w in caught],
        field_strength_method="sunkit_magex.pfss.Output.get_bvec_cartesian_norm",
    )


__all__ = ["uniform_seeds", "validate_line", "trace_fieldlines"]
