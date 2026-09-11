"""Jet Lab acceptance: ambiguity, native WCS, missing data and honest widths."""

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from scipy import ndimage as ndi

from solar_toolkit.map.jet_annotations import (
    JetDocument,
    file_sha256,
    load_session,
    pixel_hpc,
    save_session,
)
from solar_toolkit.map.jet_extraction import (
    SegmentationParameters,
    axis_metrics,
    gaussian_width,
    retain_component,
    safe_smooth,
    segment,
    skeleton_paths,
    validate_axis,
)


def make_fits(path, *, flip=False, exposure=2):
    yy, xx = np.mgrid[:96, :128]
    data = (5 + 80 * np.exp(-0.5 * ((yy - 48) / 3) ** 2)) * exposure
    header = fits.Header(
        dict(
            CTYPE1="HPLN-TAN",
            CTYPE2="HPLT-TAN",
            CUNIT1="arcsec",
            CUNIT2="arcsec",
            CRPIX1=64,
            CRPIX2=48,
            CRVAL1=850,
            CRVAL2=-170,
            CDELT1=-0.6 if flip else 0.6,
            CDELT2=0.6,
            EXPTIME=exposure,
            DATE_OBS="2025-01-24T04:48:30",
            DSUN_OBS=147298498065.7,
            HGLN_OBS=0,
            HGLT_OBS=-5.7,
            RSUN_REF=695700000,
            TELESCOP="SDO/AIA",
            INSTRUME="AIA",
            WAVELNTH=171,
            WAVEUNIT="angstrom",
            BUNIT="DN",
        )
    )
    fits.writeto(path, data, header)
    return path


def test_missing_and_polarity():
    data = np.zeros((30, 30))
    data[5:20, 4:10] = 4
    data[5:20, 20:25] = -6
    data[:, 7] = np.nan
    p = SegmentationParameters("manual", 1, closing=2, min_area=1)
    labels, _ = segment(data, p)
    assert not labels[:, 7].any() and not labels[:, 20:].any()
    negative, _ = segment(
        data, SegmentationParameters("manual", 1, polarity="negative", min_area=1)
    )
    assert negative[10, 22] and not negative[10, 5]


@pytest.mark.parametrize("method", ["manual", "percentile", "otsu"])
def test_threshold_methods(method):
    image = np.zeros((40, 40))
    image[10:25, 20:26] = 10
    labels, _ = segment(
        image, SegmentationParameters(method, 5 if method == "manual" else 85)
    )
    assert labels[15, 23] > 0


@pytest.mark.parametrize("case", ["disappear", "split", "merge", "seed_lost"])
def test_no_silent_component_switch(case):
    old = np.zeros((30, 40), int)
    old[5:20, 5:15] = 1
    old[5:20, 25:35] = 2
    new = old.copy()
    seed = [8, 8]
    if case == "disappear":
        new[old == 1] = 0
    if case == "split":
        new[12:20, 5:15] = 3
        new[11, 5:15] = 0
    if case == "merge":
        new[5:20, 5:35] = 1
    if case == "seed_lost":
        new[8, 8] = 0
    selected, reason = retain_component(old, old == 1, new, seed)
    assert selected is None and reason != "retained_reconfirm_axis"


@pytest.mark.parametrize(
    "shape", ["straight", "curved", "branch", "crossing", "missing"]
)
def test_skeleton_connected_candidates(shape):
    mask = np.zeros((100, 100), bool)
    for x in range(10, 90):
        y = 50 if shape != "curved" else int(50 + 15 * np.sin(x / 35))
        mask[y - 3 : y + 4, x] = True
    if shape in {"branch", "crossing"}:
        mask[15:51, 47:54] = True
        if shape == "crossing":
            mask[50:85, 47:54] = True
    if shape == "missing":
        mask[:, 48:52] = False
    inner = [12, 50 if shape != "curved" else 55]
    skel, paths, info = skeleton_paths(mask, inner)
    assert skel.any() and paths
    for path in paths:
        validate_axis(path, mask)
    if shape in {"branch", "crossing"}:
        assert len(paths) >= 2 and info["branches"]
    if shape == "missing":
        assert all(p[:, 0].max() < 48 for p in paths)


def test_smoothing_and_edit_cannot_bridge_gap():
    mask = np.ones((30, 30), bool)
    mask[10:20, 10:20] = False
    with pytest.raises(ValueError):
        validate_axis([[3, 3], [26, 26]], mask)
    path = [[3, 3], [24, 3], [24, 24]]
    smooth, status = safe_smooth(path, mask, sigma=30)
    validate_axis(smooth, mask)


def test_gaussian_valid_and_invalid():
    x = np.linspace(-20, 20, 161)
    good = gaussian_width(x, 8 + 40 * np.exp(-0.5 * (x / 3) ** 2))
    assert good["status"] == "valid"
    assert good["fwhm_pixel"] == pytest.approx(7.06446, rel=1e-5)
    multi = (
        8
        + 40 * np.exp(-0.5 * ((x - 6) / 2) ** 2)
        + 40 * np.exp(-0.5 * ((x + 6) / 2) ** 2)
    )
    assert gaussian_width(x, multi)["status"] == "multiple_peaks"
    assert gaussian_width(x, np.ones(len(x)))["fwhm_pixel"] is None
    bad = multi.copy()
    bad[3] = np.nan
    assert gaussian_width(x, bad)["status"] == "missing_or_insufficient_profile"
    assert (
        gaussian_width(x, 8 + 40 * np.exp(-0.5 * ((x - 18) / 3) ** 2))["fwhm_pixel"]
        is None
    )


def test_low_signal_not_forced():
    rng = np.random.default_rng(42)
    x = np.linspace(-20, 20, 161)
    y = rng.normal(0, 1, len(x)) + 0.1 * np.exp(-x * x / 18)
    assert gaussian_width(x, y)["status"] != "valid"


@pytest.mark.parametrize("flip", [False, True])
def test_wcs_export_roundtrip_and_crop(tmp_path, flip):
    doc = JetDocument(make_fits(tmp_path / "image.fits", flip=flip))
    doc.parameters(method="manual", value=25, min_area=1)
    doc.select([20, 48])
    doc.trace([20, 48])
    doc.confirm()
    doc.add_tiepoint([30.125, 48.375], 7, "manual_confirmed", "bend")
    before = np.asarray(doc.state["axis"])
    doc.reverse()
    assert doc.undo()
    assert np.array_equal(before, doc.state["axis"])
    annotation = save_session(tmp_path / "saved", [doc], sample_id="synthetic")
    docs, bundle = load_session(annotation)
    restored = docs[0]
    np.testing.assert_array_equal(before, restored.state["axis"])
    world = restored.map.pixel_to_world(before[:, 0] * u.pix, before[:, 1] * u.pix)
    px, py = restored.map.world_to_pixel(world)
    assert np.max(abs(np.c_[px.value, py.value] - before)) < 0.1
    crop = doc.map.submap([20, 30] * u.pix, top_right=[90, 70] * u.pix)
    np.testing.assert_allclose(
        pixel_hpc(crop, [[10, 18]]), pixel_hpc(doc.map, [[30, 48]]), atol=1e-6
    )
    mask = fits.getdata(tmp_path / "saved/view_0_mask.fits")
    assert np.array_equal(mask == 1, restored.selected)
    assert bundle["views"][0]["state"]["tiepoints"][0]["verified_3d"] is False
    assert restored.measurements()["projected_only"]
    doc.parameters(value=1000)
    assert doc.selected is None and not doc.state["axis"]
    doc.undo()
    assert np.array_equal(before, doc.state["axis"])


def test_direction_short_segment():
    result = axis_metrics([[0, 0], [10, 10]])
    for item in result["terminal_directions"]:
        assert item["short_segment"] and item["direction_deg"] == 45


def test_exposure_and_display_immutable(tmp_path):
    a = JetDocument(make_fits(tmp_path / "a.fits", exposure=2))
    b = JetDocument(make_fits(tmp_path / "b.fits", exposure=4))
    np.testing.assert_array_equal(a.raw, b.raw)
    with pytest.raises(ValueError):
        a.raw[0, 0] = 5
    assert a.info["midpoint_utc"] == "2025-01-24T04:48:31.000"


def test_partial_bundle_and_hash_changes_rejected(tmp_path):
    doc = JetDocument(make_fits(tmp_path / "a.fits"))
    path = save_session(tmp_path / "saved", [doc])
    (path.parent / "view_0_axis.csv").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        load_session(path)
    (path.parent / "COMPLETE.json").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        load_session(path)


def test_unregistered_difference_rejected(tmp_path):
    doc = JetDocument(make_fits(tmp_path / "a.fits"))
    with pytest.raises(ValueError, match="JETREG"):
        doc.load_difference(doc.path)
    header = doc.map.fits_header.copy()
    header["JETREG"] = True
    header["JORIGHSH"] = doc.sha256
    fits.writeto(tmp_path / "diff.fits", doc.raw, header)
    doc.load_difference(tmp_path / "diff.fits")
    doc.set_source("difference")
    assert np.array_equal(doc.data, doc.raw)


def test_registration_recovers_shift_without_exposure_false_signal(tmp_path):
    import sunpy.map

    from solar_toolkit.map.jet_registration import registered_difference

    rng = np.random.default_rng(7)
    image = 100 + 15 * ndi.gaussian_filter(rng.normal(size=(96, 128)), 1)
    base = make_fits(tmp_path / "template.fits")
    header = fits.getheader(base)
    header["JETNORM"] = True
    header["BUNIT"] = "DN / s"
    fits.writeto(tmp_path / "current.fits", image, header)
    prior = ndi.shift(image, [1.2, -0.7], order=1, mode="constant", cval=np.nan)
    header["JETNORM"] = False
    header["BUNIT"] = "DN"
    header["EXPTIME"] = 4
    fits.writeto(tmp_path / "previous.fits", prior * 4, header)
    smap = sunpy.map.Map(tmp_path / "current.fits")
    path, audit = registered_difference(
        smap,
        tmp_path / "previous.fits",
        file_sha256(tmp_path / "current.fits"),
        tmp_path / "delta.fits",
    )
    assert path is not None
    np.testing.assert_allclose(audit["train_dy_dx_pixel"], [1.2, -0.7], atol=0.25)
    assert audit["disagreement_pixel"] < 0.25
    delta = fits.getdata(path)
    assert abs(np.nanmedian(delta)) < 0.1
    assert not np.isfinite(delta[[0, -1]]).all()
