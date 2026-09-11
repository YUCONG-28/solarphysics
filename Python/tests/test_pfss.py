"""Offline PFSS acceptance tests; no observation download or GUI is required."""

import json
import subprocess
import sys

import numpy as np
import pytest

from solar_toolkit.modeling.pfss import PFSSConfig, backend_status, prepare_boundary
from solar_toolkit.modeling.pfss.boundary import _overlap_weights
from solar_toolkit.modeling.pfss.bundle import load_bundle, save_bundle
from solar_toolkit.modeling.pfss.tracing import validate_line


def synthetic_map(nphi=72, ns=36, degree=1, projection="CEA"):
    import sunpy.map
    from scipy.special import eval_legendre

    sinlat = -1 + (np.arange(ns) + 0.5) * 2 / ns
    values = eval_legendre(degree, sinlat)[:, None] * np.ones((1, nphi))
    header = dict(
        ctype1=f"CRLN-{projection}",
        ctype2=f"CRLT-{projection}",
        cunit1="deg",
        cunit2="deg",
        crpix1=(nphi + 1) / 2,
        crpix2=(ns + 1) / 2,
        crval1=180,
        crval2=0,
        cdelt1=360 / nphi,
        cdelt2=(360 / np.pi if projection == "CEA" else 180) / ns,
        bunit="G",
        date_obs="2025-01-24T04:48:30",
        hgln_obs=0,
        hglt_obs=0,
        dsun_obs=149597870700,
        rsun_ref=695700000,
    )
    return sunpy.map.Map(values, header)


def test_optional_import_does_not_import_backend():
    code = "import sys; import solar_toolkit.modeling.pfss; assert 'sunkit_magex' not in sys.modules"
    subprocess.run([sys.executable, "-B", "-c", code], check=True)


@pytest.mark.parametrize(
    "kwargs",
    [{"rss": 1}, {"rss": float("nan")}, {"nr": 3}, {"ns": 2.5}, {"max_seeds": 0}],
)
def test_config_invalid(kwargs):
    with pytest.raises(ValueError):
        PFSSConfig(**kwargs)


def test_equal_area_rebin_preserves_signed_flux():
    rng = np.random.default_rng(51)
    data = rng.normal(size=(43, 91))
    result = _overlap_weights(43, 18) @ data @ _overlap_weights(91, 36).T
    assert np.isclose(result.mean(), data.mean(), atol=1e-14)


def test_boundary_and_flux():
    m, report = prepare_boundary(synthetic_map(), nphi=36, ns=18, component="Br")
    assert m.data.shape == (18, 36)
    assert report["signed_flux_change_over_unsigned"] < 1e-12
    assert report["net_flux_correction"] == "none"


def test_hmi_sine_latitude_normalization_is_idempotent():
    from sunpy.map.sources.sdo import HMISynopticMap

    m = synthetic_map()
    header = m.meta.copy()
    header.update(
        dict(
            cunit2="Sine Latitude",
            cdelt2=2 / 36,
            telescop="SDO/HMI",
            content="Carrington Synoptic Chart Of Br Field",
        )
    )
    raw = HMISynopticMap(m.data, header)
    once, _ = prepare_boundary(raw, nphi=36, ns=18, component="Br")
    twice, _ = prepare_boundary(once, nphi=36, ns=18, component="Br")
    assert np.isclose(once.wcs.wcs.cdelt[1] * 18, 360 / np.pi)
    np.testing.assert_allclose(once.data, twice.data)
    np.testing.assert_allclose(once.wcs.wcs.cdelt, twice.wcs.wcs.cdelt)


@pytest.mark.parametrize(
    "change", ["nan", "partial", "los", "unit", "offcenter", "projection"]
)
def test_boundary_rejects_invalid_input(change):
    import sunpy.map

    m = synthetic_map()
    data = m.data.copy()
    h = m.meta.copy()
    if change == "nan":
        data[0, 0] = np.nan
    if change == "partial":
        h["cdelt1"] /= 2
    if change == "los":
        h["series"] = "hmi.M_45s"
    if change == "unit":
        h["bunit"] = "DN"
    if change == "offcenter":
        h["crpix2"] += 1
    if change == "projection":
        h["ctype1"] = "HPLN-TAN"
        h["ctype2"] = "HPLT-TAN"
    with pytest.raises(ValueError):
        prepare_boundary(sunpy.map.Map(data, h), nphi=36, ns=18, component="Br")


def test_axis_flip_preserves_world_field():
    import sunpy.map

    m = synthetic_map()
    h = m.meta.copy()
    h["cdelt1"] *= -1
    h["cdelt2"] *= -1
    a, _ = prepare_boundary(m, nphi=36, ns=18, component="Br")
    b, report = prepare_boundary(
        sunpy.map.Map(m.data[::-1, ::-1], h), nphi=36, ns=18, component="Br"
    )
    np.testing.assert_allclose(a.data, b.data)
    assert report["flipped_axes"] == [0, 1]


def test_failure_not_closed():
    p = np.array([[1, 0, 0], [1.2, 0, 0], [2.5, 0, 0]])
    assert validate_line(p, rss=2.5, polarity=1)["classification"] == "open_positive"
    assert not validate_line(
        p, rss=2.5, polarity=1, warnings_seen=["maximum steps reached"]
    )["trace_valid"]
    p[-1] = [1.5, 0, 0]
    assert validate_line(p, rss=2.5, polarity=0)["classification"] == "failed"


def test_bundle_roundtrip_and_tamper(tmp_path):
    records = [
        dict(
            fieldline_id="a",
            xyz_carrington_rsun=np.array([[1, 0, 0], [2, 0, 0]]),
            trace_valid=True,
            classification="open_positive",
        )
    ]
    projected = [dict(hpc_arcsec=np.zeros((2, 2)))]
    save_bundle(tmp_path, records, projected, {"obstime": "2025-01-24T04:48:30"})
    _, loaded = load_bundle(tmp_path)
    np.testing.assert_array_equal(
        loaded[0]["xyz_carrington_rsun"], records[0]["xyz_carrington_rsun"]
    )
    (tmp_path / "metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(tmp_path)


def test_manifest_path_escape(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "files": [{"path": "../x", "bytes": 0, "sha256": ""}],
            }
        )
    )
    with pytest.raises(ValueError, match="Unsafe"):
        load_bundle(tmp_path)


def test_cache_separates_geometry_density_and_display(tmp_path, monkeypatch):
    from solar_toolkit.modeling.pfss import cache

    monkeypatch.setattr(cache, "version", lambda _: "test-fixed")
    for filename in ["boundary", "aia", "sources"]:
        (tmp_path / filename).write_bytes(b"fixture")
    config = {
        "boundary_fits": str(tmp_path / "boundary"),
        "aia_fits": str(tmp_path / "aia"),
        "sources_csv": str(tmp_path / "sources"),
        "event_utc": "2025-01-24T04:48:30",
    }
    a = cache.cache_keys(config)
    b = cache.cache_keys(
        {
            **config,
            "output_dir": "unused",
            "newkirk_multiplier": 4,
            "display_color": "red",
        }
    )
    assert a == b
    c = cache.cache_keys({**config, "pfss": {"rss": 3}})
    assert a["boundary"] == c["boundary"] and a["trace"] != c["trace"]
    (tmp_path / "sources").write_bytes(b"different")
    d = cache.cache_keys(config)
    assert a["trace"] == d["trace"] and a["association"] != d["association"]


def test_height_interface_keeps_pfss_conditional_even_with_bad_evidence_flag():
    import pandas as pd

    from solar_toolkit.radio.height_comparison import (
        build_gaussian_newkirk_height_table,
    )
    from solar_toolkit.radio.source_geometry import normalize_source_observations

    source = normalize_source_observations(
        pd.DataFrame(
            [
                dict(
                    time="2025-01-24T04:48:30Z",
                    freq=150,
                    center_x_arcsec=1200,
                    center_y_arcsec=-300,
                    quality_flag="ok",
                    trajectory_valid=True,
                )
            ]
        )
    )
    geometry = pd.DataFrame(
        [
            dict(
                source_id=source.iloc[0].source_id,
                height_rsun=0.5,
                geometry_valid=True,
                method="conditional_pfss_field_line",
                evidence_id="test",
                independent_geometry_valid=True,
            )
        ]
    )
    result = build_gaussian_newkirk_height_table(
        source, {"solar_radius_arcsec": 960}, geometry
    )
    assert result.height_3d_valid.all() and not result.height_3d_independent_valid.any()


def test_loaded_traces_can_be_resaved_without_python_objects(tmp_path):
    records = [
        dict(
            fieldline_id="a",
            xyz_carrington_rsun=np.array([[1, 0, 0], [2, 0, 0]]),
            trace_valid=True,
            classification="open_positive",
        )
    ]
    projections = [dict(hpc_arcsec=np.zeros((2, 2)))]
    save_bundle(tmp_path / "first", records, projections, {})
    _, loaded = load_bundle(tmp_path / "first")
    save_bundle(tmp_path / "second", loaded, projections, {})
    _, again = load_bundle(tmp_path / "second")
    np.testing.assert_array_equal(
        again[0]["xyz_carrington_rsun"], records[0]["xyz_carrington_rsun"]
    )


def test_occultation_uses_finite_segment():
    from solar_toolkit.modeling.pfss.projection import visible_from_observer

    points = [[1.2, 0, 0], [-1.2, 0, 0], [-1.2, 1.2, 0], [10, 0, 0]]
    assert visible_from_observer(points, [215, 0, 0]).tolist() == [
        True,
        False,
        True,
        True,
    ]


def test_header_times_and_exposure():
    from types import SimpleNamespace

    from solar_toolkit.modeling.pfss.observations import (
        header_time,
        observation_midpoint,
    )

    assert header_time("2025.01.24_04:49:07_TAI").isot == "2025-01-24T04:48:30.000"
    frame = SimpleNamespace(meta={"date-obs": "2025-01-24T04:48:30.000", "exptime": 2})
    assert observation_midpoint(frame).isot == "2025-01-24T04:48:31.000"
    frame.meta["exptime"] = float("nan")
    with pytest.raises(ValueError):
        observation_midpoint(frame)


def test_multiple_shell_hits_and_degeneracy():
    from solar_toolkit.radio.fieldline_association import (
        model_shell_candidates,
        sphere_intersections,
    )

    points = np.array([[1, 0, 0], [2, 0, 0], [1, 1, 0]])
    assert len(sphere_intersections(points, 1.5)) == 2
    lines = [dict(fieldline_id="fixed", trace_valid=True, xyz_carrington_rsun=points)]
    original = points.copy()
    rows = model_shell_candidates(lines, [150, 228], rss=2.5)
    a = [r for r in rows if r["multiplier"] == 1 and r["harmonic"] == 2]
    b = [r for r in rows if r["multiplier"] == 4 and r["harmonic"] == 1]
    assert len(a) == len(b)
    for one, two in zip(a, b, strict=True):
        np.testing.assert_allclose(
            one["xyz_carrington_rsun"], two["xyz_carrington_rsun"]
        )
        assert one["newkirk_height_rsun"] == two["newkirk_height_rsun"]
    np.testing.assert_array_equal(points, original)
    outside = model_shell_candidates(lines, [1e6, 1], rss=2.5)
    assert not any(r["model_valid"] for r in outside)


def test_association_never_promotes_missing_errors():
    import pandas as pd

    from solar_toolkit.radio.fieldline_association import associate_sources
    from solar_toolkit.radio.source_geometry import project_hpc

    obs = np.array([215, 0, 0])
    points = np.array([[1, 1.1, -0.2], [1, 1.1, 0], [1, 1.1, 0.2]])
    xy = np.array([project_hpc(p, obs) for p in points])
    records = [dict(fieldline_id="fixed", trace_valid=True, xyz_carrington_rsun=points)]
    sources = pd.DataFrame(
        [
            dict(
                source_id="s",
                time_utc="2025-01-24T04:48:30Z",
                frequency_mhz=150,
                center_x_arcsec=xy[1, 0],
                center_y_arcsec=xy[1, 1],
            )
        ]
    )
    a = associate_sources(
        sources, records, [dict(hpc_arcsec=xy)], observer_carrington_rsun=obs
    )
    assert (
        len(a) > 0
        and not a.geometry_valid.any()
        and not a.independent_geometry_valid.any()
    )
    assert a.height_rsun.isna().all()
    b = associate_sources(
        sources,
        records,
        [dict(hpc_arcsec=xy)],
        observer_carrington_rsun=obs,
        max_residual_arcsec=1,
    )
    assert b.geometry_valid.any() and not b.independent_geometry_valid.any()
    assert (b.loc[b.geometry_valid, "separation_rsun"] < 1e-10).all()


def test_projection_matches_exact_ray():
    import astropy.units as u
    from astropy.time import Time
    from sunpy.coordinates import frames, get_earth

    from solar_toolkit.modeling.pfss.projection import project_fieldlines
    from solar_toolkit.radio.source_geometry import project_hpc

    time = Time("2025-01-24T04:48:30")
    obs = get_earth(time)
    cart = obs.transform_to(
        frames.HeliographicCarrington(observer="earth", obstime=time)
    ).cartesian.xyz.to_value(u.R_sun)
    point = cart / np.linalg.norm(cart) * 1.2 + np.array([0, 0, 0.2])
    result = project_fieldlines(
        [dict(fieldline_id="known", xyz_carrington_rsun=np.array([point]))],
        observer=obs,
        obstime=time,
    )
    np.testing.assert_allclose(
        result[0]["hpc_arcsec"][0], project_hpc(point, cart), atol=1e-7
    )


def test_car_reprojects_values_not_just_header():
    import sunpy.map

    m = synthetic_map(36, 18, projection="CAR")
    lat = -90 + (np.arange(18) + 0.5) * 10
    m = sunpy.map.Map(
        np.sin(np.deg2rad(lat))[:, None] * np.ones((1, 36)), m.meta.copy()
    )
    result, report = prepare_boundary(m, nphi=18, ns=9, component="Br")
    assert report["car_reprojected"]
    expected = -1 + (np.arange(9) + 0.5) * 2 / 9
    assert np.max(abs(result.data[:, 0] - expected)) < 0.035


@pytest.mark.skipif(
    not backend_status()["available"], reason="optional PFSS backend not installed"
)
def test_interior_analytic_vector_and_convergence():
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    from solar_toolkit.modeling.pfss import solve_pfss

    errors = []
    radius = np.array([1.2, 1.5, 2.0])
    theta = np.deg2rad([35, 70, 120])
    rss = 2.5
    # Phi=A*(r-rss^3/r^2)*cos(theta), normalize Br(1)=cos(theta).
    coefficient = -1 / (1 + 2 * rss**3)
    expected = np.column_stack(
        [
            -coefficient * (1 + 2 * rss**3 / radius**3) * np.cos(theta),
            coefficient * (1 - rss**3 / radius**3) * np.sin(theta),
            np.zeros(3),
        ]
    )
    for n, nr in [(36, 18), (72, 35)]:
        m, _ = prepare_boundary(synthetic_map(n, n), nphi=n, ns=n, component="Br")
        out, _ = solve_pfss(m, PFSSConfig(nphi=n, ns=n, nr=nr))
        coords = SkyCoord(
            [15, 130, 250] * u.deg,
            (90 - np.rad2deg(theta)) * u.deg,
            radius * u.R_sun,
            frame=out.coordinate_frame,
        )
        actual = out.get_bvec(coords, out_type="spherical").to_value(u.G)
        error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        errors.append(error)
    assert errors[-1] < 0.05
    assert errors[-1] < errors[0]


@pytest.mark.skipif(
    not backend_status()["available"], reason="optional PFSS backend not installed"
)
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_analytic_source_surface_field_and_open_flux(degree):
    from scipy.special import eval_legendre

    from solar_toolkit.modeling.pfss import solve_pfss

    nphi, ns, nr, rss = 72, 72, 35, 2.5
    m, _ = prepare_boundary(
        synthetic_map(nphi, ns, degree), nphi=nphi, ns=ns, component="Br"
    )
    output, _ = solve_pfss(m, PFSSConfig(nphi=nphi, ns=ns, nr=nr, rss=rss))
    expected = (
        (2 * degree + 1)
        * rss ** (degree - 1)
        / (degree + (degree + 1) * rss ** (2 * degree + 1))
        * eval_legendre(degree, -1 + (np.arange(ns) + 0.5) * 2 / ns)
    )
    actual = output.source_surface_br.data
    error = np.sqrt(np.mean((actual - expected[:, None]) ** 2)) / np.sqrt(
        np.mean(expected**2)
    )
    assert error <= 0.05
    assert abs(np.mean(abs(actual)) / np.mean(abs(expected)) - 1) <= 0.05


@pytest.mark.skipif(
    not backend_status()["available"], reason="optional PFSS backend not installed"
)
def test_tracer_crosscheck_dipole_footpoints():
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    from solar_toolkit.modeling.pfss import solve_pfss, trace_fieldlines

    m, _ = prepare_boundary(synthetic_map(72, 72), nphi=72, ns=72, component="Br")
    out, _ = solve_pfss(m, PFSSConfig(nphi=72, ns=72))
    seeds = SkyCoord(
        [40, 90] * u.deg, [65, -65] * u.deg, 1.001 * u.R_sun, frame=out.coordinate_frame
    )
    a = trace_fieldlines(out, seeds, rss=2.5)
    b = trace_fieldlines(out, seeds, rss=2.5, tracer="python")
    for line, other, latitude in zip(a, b, [65, -65], strict=True):
        assert line["trace_valid"] and other["trace_valid"]
        assert line["classification"] == other["classification"]
        for v in [line, other]:
            xyz = v["xyz_carrington_rsun"]
            foot = xyz[np.argmin(np.linalg.norm(xyz, axis=1))]
            lat = np.degrees(np.arcsin(foot[2] / np.linalg.norm(foot)))
            assert abs(lat - latitude) < 1
