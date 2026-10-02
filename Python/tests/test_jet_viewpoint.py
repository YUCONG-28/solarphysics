import numpy as np
import pytest
from astropy.io import fits

from solar_toolkit.map.jet_annotations import JetDocument, load_session, save_session
from solar_toolkit.map.jet_timeline import geometry_precheck, pair_frames
from solar_toolkit.map.jet_viewpoint import (
    build_mapping,
    map_pixels,
    native_roi_mask,
    reproject_photosphere,
    sample_mapping,
)
from solar_toolkit.radio.source_geometry import project_hpc


def maps(tmp_path):
    point = np.array([0.5, 0.85, -0.2])
    point = point / np.linalg.norm(point) * 1.08
    result = []
    for i, lon in enumerate([0.0, 30.0]):
        b = np.deg2rad(-5)
        longitude = np.deg2rad(lon)
        o = 212 * np.array(
            [np.cos(b) * np.cos(longitude), np.cos(b) * np.sin(longitude), np.sin(b)]
        )
        tx, ty = project_hpc(point, o)
        header = fits.Header(
            dict(
                CTYPE1="HPLN-TAN",
                CTYPE2="HPLT-TAN",
                CUNIT1="arcsec",
                CUNIT2="arcsec",
                CRPIX1=64,
                CRPIX2=48,
                CRVAL1=tx,
                CRVAL2=ty,
                CDELT1=0.6,
                CDELT2=0.6,
                EXPTIME=2,
                DATE_OBS="2000-01-01T12:00:30",
                DSUN_OBS=212 * 695700000,
                HGLN_OBS=lon,
                HGLT_OBS=-5,
                RSUN_REF=695700000,
                TELESCOP="SDO/AIA",
                INSTRUME="AIA",
                WAVELNTH=304,
                WAVEUNIT="angstrom",
                BUNIT="DN",
                SYNTHET=True,
            )
        )
        yy, xx = np.mgrid[:96, :128]
        data = 5 + 80 * np.exp(-0.5 * ((yy - 48) / 3) ** 2)
        path = tmp_path / f"{i}.fits"
        fits.writeto(path, data, header)
        result.append(JetDocument(path))
    return result


def test_identity_missing_and_wrong_height(tmp_path):
    a, b = maps(tmp_path)
    m = build_mapping(a.map, a.map, 0.08)
    sampled = sample_mapping(a.raw, m)
    assert np.nanmax(abs(sampled - a.raw)) < 1e-5
    px = np.array([40.0, 60.0, 80.0])
    py = np.array([40.0, 48.0, 55.0])
    sx, sy, ok = map_pixels(b.map, a.map, px, py, 0.08)
    tx, ty, back = map_pixels(a.map, b.map, sx, sy, 0.08)
    assert ok.all() and back.all()
    assert np.max(np.hypot(tx - px, ty - py)) < 0.1
    wx, wy, _ = map_pixels(b.map, a.map, px, py, 0.0)
    assert np.max(np.hypot(wx - sx, wy - sy)) > 1
    raw = a.raw.copy()
    raw[45:51, 60:65] = np.nan
    assert np.isnan(sample_mapping(raw, m)[47, 62])
    with pytest.raises(ValueError):
        build_mapping(a.map, b.map, -1)


def test_native_draft_roundtrip_and_geometry_uses_native(tmp_path):
    a, b = maps(tmp_path)
    mask = native_roi_mask(
        a.map, a.map, [[10, 40], [115, 40], [115, 57], [10, 57]], 0.08
    )
    a.set_native_roi(mask, {"height_rsun": 0.08})
    a.parameters(method="manual", value=15, min_area=1)
    a.select([20, 48])
    a.trace([20, 48])
    a.confirm()
    p = save_session(tmp_path / "saved", [a, b])
    docs, bundle = load_session(p)
    assert bundle["version"] == 2
    assert docs[0].state == a.state
    assert np.array_equal(docs[0].selected, a.selected)
    r = geometry_precheck(docs)
    assert not r["geometry_valid"] and not r["display_geometry_used"]
    before = a.state["axis"].copy()
    a.set_source("intensity")
    assert a.state["axis"] == before


def record(inst, t, sha):
    return {
        "instrument": inst,
        "band": 304,
        "midpoint_utc": f"2000-01-01T12:00:{t:02d}",
        "dsun_m": 147e9,
        "image_sha256": sha,
    }


def test_pairing_real_frames_repeats_ties_missing():
    records = [record("AIA", t, str(t)) for t in [0, 12, 24, 36, 48]] + [
        record("EUVI", t, "e" + str(t)) for t in [0, 40]
    ]
    pairs = pair_frames(records, "AIA", 304)
    assert len(pairs) == 5 and pairs[1]["repeated_other"]
    tied = pair_frames(
        [record("AIA", 20, "a"), record("EUVI", 0, "e0"), record("EUVI", 40, "e40")],
        "AIA",
    )
    assert tied[0]["status"] == "ambiguous_nearest"
    missing = pair_frames(
        [record("AIA", 59, "a"), record("EUVI", 0, "e0"), record("EUVI", 12, "e12")],
        "AIA",
    )
    assert missing[0]["status"] == "outside_tolerance"


def test_scaled_rotated_crop_and_backside(tmp_path):
    from sunpy.map import Map

    a, b = maps(tmp_path)
    meta = b.map.meta.copy()
    meta.update(dict(cdelt1=0.9, cdelt2=0.8, crota2=17, crpix1=58, crpix2=43))
    transformed = Map(b.map.data, meta)
    x, y = np.array([40.0, 60.0, 80.0]), np.array([40.0, 48.0, 55.0])
    sx, sy, valid = map_pixels(transformed, a.map, x, y, 0.08)
    tx, ty, back = map_pixels(a.map, transformed, sx, sy, 0.08)
    assert valid.all() and back.all()
    assert np.max(np.hypot(tx - x, ty - y)) < 0.1
    meta["hgln_obs"] = 180
    backside = Map(b.map.data, meta)
    assert not build_mapping(backside, a.map, 0.08).valid.any()


def test_off_disk_shell_and_tangent_exclusion(tmp_path):
    from sunpy.map import Map

    a, _ = maps(tmp_path)
    meta = a.map.meta.copy()
    meta.update(dict(crval1=1050.0, crval2=0.0))
    off = Map(a.map.data, meta)
    assert not map_pixels(off, off, [63.0], [47.0], 0)[2].any()
    assert map_pixels(off, off, [63.0], [47.0], 0.2)[2].all()
    meta["crval1"] = np.rad2deg(np.arcsin(1 / 212)) * 3600
    tangent = Map(a.map.data, meta)
    assert not map_pixels(tangent, tangent, [63.0], [47.0], 0)[2].any()


def test_mapping_agrees_with_independent_sunpy_transform(tmp_path):
    """Use SunPy's 3D conversion, not our own forward/inverse vector basis."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import frames

    a, b = maps(tmp_path)
    x, y = np.array([40.0, 60.0, 80.0]), np.array([40.0, 48.0, 55.0])
    world = a.map.pixel_to_world(x * u.pix, y * u.pix)
    physical = SkyCoord(
        world.Tx,
        world.Ty,
        frame=frames.Helioprojective(
            observer=a.map.observer_coordinate,
            obstime=a.map.date,
            rsun=1.08 * 695700 * u.km,
        ),
    ).make_3d()
    transformed = SkyCoord(physical.transform_to(b.map.coordinate_frame))
    expected = b.map.world_to_pixel(transformed)
    sx, sy, valid = map_pixels(b.map, a.map, x, y, 0.08)
    assert valid.all()
    assert np.max(np.hypot(sx - expected.x.value, sy - expected.y.value)) < 1e-3


@pytest.mark.parametrize("stride", [1, 3])
def test_official_preview_preserves_grid_masks_and_metadata(tmp_path, stride):
    from sunpy.map import Map

    a, _ = maps(tmp_path)
    meta = a.map.meta.copy()
    meta["crval1"] = 850.0
    meta["rsun_ref"] = 696000000.0
    reference = Map(a.map.data, meta)
    data = a.raw.copy()
    data[45:51, 60:66] = np.nan
    original_meta = dict(reference.meta)
    mapping = build_mapping(reference, reference, 0, stride)
    result = reproject_photosphere(
        reference, reference, data, stride=stride, mapping=mapping
    )
    expected = data[::stride, ::stride]
    assert result.shape == expected.shape
    assert np.nanmax(abs(result - expected)) < 1e-5
    assert np.isnan(result[48 // stride, 63 // stride])
    assert dict(reference.meta) == original_meta
    assert reference.meta["rsun_ref"] == 696000000.0
