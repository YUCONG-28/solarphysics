"""Optional sampled magnitudes do not change PFSS geometry or old bundles."""

import json
import sys
from types import ModuleType, SimpleNamespace

import astropy.units as u
import numpy as np
import pytest

from solar_toolkit.modeling.pfss import backend_status
from solar_toolkit.modeling.pfss.bundle import (
    complete_manifest,
    load_bundle,
    save_bundle,
)
from solar_toolkit.modeling.pfss.tracing import trace_fieldlines


def mocked_tracer(monkeypatch):
    class Coordinates:
        representation_type = "cartesian"
        cartesian = SimpleNamespace(
            xyz=np.array([[1.0, 1.5, 2.5], [0, 0, 0], [0, 0, 0]]) * u.R_sun
        )

        def copy(self):
            return Coordinates()

    coordinates = Coordinates()

    class Tracer:
        def __init__(self, **kwargs):
            pass

        def trace(self, seeds, output):
            return [SimpleNamespace(coords=coordinates, polarity=1)]

    module = ModuleType("sunkit_magex.pfss.tracing")
    module.PerformanceTracer = module.PythonTracer = Tracer
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return coordinates


def test_opt_out_never_samples_and_opt_in_converts_backend_units_without_mutation(
    monkeypatch,
):
    coordinates = mocked_tracer(monkeypatch)

    class Output:
        calls = 0

        def get_bvec(self, coords, out_type):
            self.calls += 1
            assert coords is not coordinates and out_type == "cartesian"
            coords.representation_type = "spherical"
            return np.tile([0.3, 0.4, 0], (3, 1)) * u.mT

    output = Output()
    plain = trace_fieldlines(output, [0], rss=2.5)[0]
    assert output.calls == 0 and "b_gauss" not in plain
    sampled = trace_fieldlines(output, [0], rss=2.5, sample_field_strength=True)[0]
    np.testing.assert_allclose(sampled["b_gauss"], 5)
    np.testing.assert_array_equal(
        sampled["xyz_carrington_rsun"], plain["xyz_carrington_rsun"]
    )
    assert sampled["trace_valid"] and sampled["field_strength_status"] == "ok"
    assert coordinates.representation_type == "cartesian"


@pytest.mark.parametrize("case", ["failure", "unitless", "shape", "partial", "missing"])
def test_sampling_failures_are_separate_from_trace_validity(monkeypatch, case):
    mocked_tracer(monkeypatch)

    def sample(*args, **kwargs):
        if case == "failure":
            raise RuntimeError("synthetic sampling failure")
        if case == "unitless":
            return np.ones((3, 3))
        if case == "shape":
            return np.ones((3, 2)) * u.G
        values = np.full((3, 3), np.nan)
        if case == "partial":
            values[1] = [3, 4, 0]
        return values * u.G

    record = trace_fieldlines(
        SimpleNamespace(get_bvec=sample), [0], rss=2.5, sample_field_strength=True
    )[0]
    assert record["trace_valid"] and record["classification"] == "open_positive"
    assert record["field_strength_status"] == (
        "partial" if case == "partial" else "sampling_failed"
    )
    assert record["b_gauss"].shape == (3,)
    assert np.isfinite(record["b_gauss"]).sum() == (1 if case == "partial" else 0)


def bundle_input():
    return [
        dict(
            fieldline_id="a",
            trace_valid=True,
            xyz_carrington_rsun=np.array([[1.0, 0, 0], [1.5, 0, 0], [2.0, 0, 0]]),
        )
    ], [dict(hpc_arcsec=np.zeros((3, 2)))]


def test_old_and_optional_magnetic_bundles_roundtrip_without_backend(tmp_path):
    records, projections = bundle_input()
    save_bundle(tmp_path / "old", records, projections, {})
    metadata, old = load_bundle(tmp_path / "old")
    assert "b_gauss" not in old[0] and "field_strength_unit" not in metadata
    records[0].update(
        b_gauss=np.array([0.003, 0.004, np.nan]) * u.T, field_strength_status="partial"
    )
    save_bundle(tmp_path / "new", records, projections, {})
    metadata, new = load_bundle(tmp_path / "new")
    np.testing.assert_array_equal(new[0]["b_gauss"], [30, 40, np.nan])
    assert metadata["field_strength_unit"] == "G" and metadata["schema_version"] == 1
    save_bundle(tmp_path / "resaved", new, projections, {})
    np.testing.assert_array_equal(
        load_bundle(tmp_path / "resaved")[1][0]["b_gauss"], [30, 40, np.nan]
    )


@pytest.mark.parametrize(
    "value", [[1, 2], [-1, 2, 3], [1 + 2j, 2, 3], [True, False, True], ["1", "2", "3"]]
)
def test_malformed_saved_magnitudes_are_rejected(tmp_path, value):
    records, projections = bundle_input()
    records[0]["b_gauss"] = value
    with pytest.raises(ValueError):
        save_bundle(tmp_path, records, projections, {})
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize(
    "value",
    [
        np.ones(3, dtype=complex),
        np.ones(3, dtype=bool),
        np.array(["1", "2", "3"]),
        np.array([1, 2]),
        np.array([1.0, np.inf, 3.0]),
    ],
)
def test_reader_validates_magnetic_array_structure_after_checksum(tmp_path, value):
    records, projections = bundle_input()
    records[0]["b_gauss"] = np.ones(3)
    save_bundle(tmp_path, records, projections, {})
    path = tmp_path / "fieldlines.npz"
    with np.load(path) as archive:
        arrays = {name: archive[name] for name in archive}
    arrays["b_gauss_0"] = value
    np.savez_compressed(path, **arrays)
    complete_manifest(tmp_path)
    with pytest.raises(ValueError, match="[Ff]ield-strength"):
        load_bundle(tmp_path)


def test_reader_refuses_unknown_magnetic_unit(tmp_path):
    records, projections = bundle_input()
    records[0]["b_gauss"] = np.ones(3)
    save_bundle(tmp_path, records, projections, {})
    path = tmp_path / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["field_strength_unit"] = "T"
    path.write_text(json.dumps(metadata))
    complete_manifest(tmp_path)
    with pytest.raises(ValueError, match="unit"):
        load_bundle(tmp_path)


def test_cache_opt_out_equivalence_and_opt_in_only_invalidates_trace_onward(
    tmp_path, monkeypatch
):
    from solar_toolkit.modeling.pfss import cache

    monkeypatch.setattr(cache, "version", lambda name: "synthetic")
    source = tmp_path / "input"
    source.write_bytes(b"synthetic")
    config = dict(
        boundary_fits=str(source), aia_fits=str(source), event_utc="2000-01-01T00:00:00"
    )
    a = cache.cache_keys(config)
    assert a == cache.cache_keys({**config, "sample_field_strength": False})
    b = cache.cache_keys({**config, "sample_field_strength": True})
    assert all(a[k] == b[k] for k in ("boundary", "solve"))
    assert all(a[k] != b[k] for k in ("trace", "projection", "association"))
    with pytest.raises(ValueError, match="boolean"):
        cache.cache_keys({"sample_field_strength": "false"})


@pytest.mark.skipif(
    not backend_status()["available"],
    reason="optional public PFSS backend not installed",
)
def test_public_backend_dipole_sampling_matches_vectors_and_keeps_trace_geometry():
    import sunpy.map
    from astropy.coordinates import SkyCoord

    from solar_toolkit.modeling.pfss import PFSSConfig, prepare_boundary, solve_pfss

    n = 24
    data = (-1 + (np.arange(n) + 0.5) * 2 / n)[:, None] * np.ones((1, n))
    header = dict(
        ctype1="CRLN-CEA",
        ctype2="CRLT-CEA",
        cunit1="deg",
        cunit2="deg",
        crpix1=(n + 1) / 2,
        crpix2=(n + 1) / 2,
        crval1=180,
        crval2=0,
        cdelt1=360 / n,
        cdelt2=(360 / np.pi) / n,
        bunit="G",
        date_obs="2000-01-01T00:00:00",
        hgln_obs=0,
        hglt_obs=0,
        dsun_obs=149597870700,
        rsun_ref=695700000,
    )
    boundary, _ = prepare_boundary(
        sunpy.map.Map(data, header), nphi=n, ns=n, component="Br"
    )
    output, _ = solve_pfss(boundary, PFSSConfig(nphi=n, ns=n, nr=16))
    seeds = SkyCoord(
        [40] * u.deg, [65] * u.deg, 1.001 * u.R_sun, frame=output.coordinate_frame
    )
    (plain,) = trace_fieldlines(output, seeds, rss=2.5)
    (sampled,) = trace_fieldlines(output, seeds, rss=2.5, sample_field_strength=True)
    assert sampled["trace_valid"] and sampled["field_strength_status"] in {
        "ok",
        "partial",
    }
    np.testing.assert_array_equal(
        sampled["xyz_carrington_rsun"], plain["xyz_carrington_rsun"]
    )
    xyz = sampled["xyz_carrington_rsun"]
    coords = SkyCoord(
        x=xyz[:, 0] * u.R_sun,
        y=xyz[:, 1] * u.R_sun,
        z=xyz[:, 2] * u.R_sun,
        representation_type="cartesian",
        frame=output.coordinate_frame,
    )
    expected = np.hypot.reduce(
        output.get_bvec(coords, out_type="cartesian").to_value(u.G), axis=1
    )
    np.testing.assert_allclose(sampled["b_gauss"], expected, equal_nan=True)
    assert np.isfinite(sampled["b_gauss"]).sum() > 3
