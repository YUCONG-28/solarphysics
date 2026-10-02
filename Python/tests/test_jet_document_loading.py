"""Native tie-only loading preserves preparation and defers auxiliary work."""

from unittest.mock import patch

import numpy as np
from astropy.io import fits

from solar_toolkit.map import euvi_preprocessing
from solar_toolkit.map import jet_annotations as annotations


def image(path, *, euvi=True):
    yy, xx = np.mgrid[:64, :80]
    data = 2 + 40 * np.exp(-(((yy - 32) / 3) ** 2))
    data[0, 0] = np.nan
    header = fits.Header(
        dict(
            CTYPE1="HPLN-TAN",
            CTYPE2="HPLT-TAN",
            CUNIT1="arcsec",
            CUNIT2="arcsec",
            CRPIX1=40,
            CRPIX2=32,
            CRVAL1=800,
            CRVAL2=-160,
            CDELT1=1,
            CDELT2=1,
            EXPTIME=2,
            DATE_OBS="2000-01-01T12:00:00",
            DSUN_OBS=149600000000.0,
            HGLN_OBS=29 if euvi else 0,
            HGLT_OBS=-5.0,
            RSUN_REF=695700000,
            TELESCOP="STEREO A" if euvi else "SDO/AIA",
            INSTRUME="SECCHI" if euvi else "AIA",
            DETECTOR="EUVI" if euvi else "AIA",
            WAVELNTH=304,
            WAVEUNIT="angstrom",
            BUNIT="DN",
            SYNTHET=True,
        )
    )
    fits.writeto(path, data, header)
    return path, data


def test_euvi_preparation_is_once_and_shared_with_display(tmp_path):
    path, pixels = image(tmp_path / "euvi.fits")
    with patch.object(
        euvi_preprocessing,
        "prepare_euvi_map",
        wraps=euvi_preprocessing.prepare_euvi_map,
    ) as prepare, patch.object(
        euvi_preprocessing, "intensity_array", wraps=euvi_preprocessing.intensity_array
    ) as intensity:
        doc = annotations.JetDocument(path)
    assert prepare.call_count == 1
    assert intensity.call_count == 1
    assert doc.raw is doc.preparation.data
    assert doc.info["preprocessing"] is doc.preparation.audit
    np.testing.assert_allclose(doc.raw, pixels / 2, equal_nan=True)
    assert not doc.raw.flags.writeable and doc.segmentation_ready


def test_non_euvi_intensity_array_is_processed_once(tmp_path):
    path, pixels = image(tmp_path / "aia.fits", euvi=False)
    with patch.object(
        euvi_preprocessing, "intensity_array", wraps=euvi_preprocessing.intensity_array
    ) as prepare:
        doc = annotations.JetDocument(path)
    assert prepare.call_count == 1
    np.testing.assert_allclose(doc.raw, pixels / 2, equal_nan=True)
    assert doc.preparation is None


def test_lazy_ties_undo_and_session_roundtrip_do_not_segment(tmp_path):
    path, _ = image(tmp_path / "euvi.fits")
    with patch.object(
        annotations, "segment", wraps=annotations.segment
    ) as segmentation:
        doc = annotations.JetDocument(path, lazy_segmentation=True)
        doc.add_tiepoint(
            [30.125, 32.75], 1, "possible", "candidate", role="jet", order=1
        )
        assert doc.undo() and not doc.state["tiepoints"]
        doc.add_tiepoint(
            [30.125, 32.75], 1, "possible", "candidate", role="jet", order=1
        )
        saved = annotations.save_session(tmp_path / "saved", [doc])
        restored, _ = annotations.load_session(saved, lazy_segmentation=True)
        assert segmentation.call_count == 0
        assert restored[0].state["tiepoints"] == doc.state["tiepoints"]
        assert not restored[0].segmentation_ready
        restored[0].ensure_segmented()
        assert segmentation.call_count == 1 and restored[0].segmentation_ready
        restored[0].ensure_segmented()
        assert segmentation.call_count == 1


def test_lazy_selection_materializes_equivalent_segmentation(tmp_path):
    path, _ = image(tmp_path / "euvi.fits")
    eager = annotations.JetDocument(path)
    lazy = annotations.JetDocument(path, lazy_segmentation=True)
    lazy.select([40, 32])
    eager.select([40, 32])
    np.testing.assert_array_equal(lazy.labels, eager.labels)
    np.testing.assert_array_equal(lazy.selected, eager.selected)


def test_reused_unit_decision_matches_full_audit(tmp_path):
    path, _ = image(tmp_path / "euvi.fits")
    smap = annotations.sunpy.map.Map(path)
    for unit in ("DN", "DN / s", "photon/s", "unknown"):
        smap.meta["bunit"] = unit
        for missing in (0, 3, "invalid"):
            smap.meta["nmissing"] = missing
            expected = euvi_preprocessing.quantitative_issues(smap)
            _, _, action, warnings = euvi_preprocessing.intensity_array(smap)
            with patch.object(
                euvi_preprocessing,
                "intensity_array",
                side_effect=AssertionError("repeated array work"),
            ):
                actual = euvi_preprocessing.quantitative_issues(
                    smap, intensity_decision=(action, warnings)
                )
            assert actual == expected
