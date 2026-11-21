"""Synthetic contracts for observed SHARP boundaries and imported local grids."""

import gzip
import hashlib
import json

import numpy as np
import pytest
from astropy.io import fits

from solar_toolkit.modeling.nlfff import read_nlfff_grid, read_sharp_cea_boundary
from solar_toolkit.time import normalize_time


def sharp_header(component):
    return fits.Header(
        {
            "HARPNUM": 42,
            "T_REC": "2000.01.01_00:00:32_TAI",
            "QUALITY": 0,
            "SERIES": "hmi.sharp_cea_720s",
            "SEGMENT": component,
            "BUNIT": "G",
            "CTYPE1": "CRLN-CEA",
            "CTYPE2": "CRLT-CEA",
            "CUNIT1": "deg",
            "CUNIT2": "deg",
            "CRPIX1": 2,
            "CRPIX2": 2,
            "CRVAL1": 20,
            "CRVAL2": 5,
            "CDELT1": 0.03,
            "CDELT2": 0.03,
            "PV2_1": 1,
        }
    )


def sharp_files(tmp_path, *, arrays=None):
    paths = []
    if arrays is None:
        arrays = [np.full((3, 4), value, dtype=float) for value in (3, 2, 1)]
    for component, data in zip(("Br", "Bt", "Bp"), arrays, strict=True):
        path = tmp_path / f"input.{component}.fits"
        fits.PrimaryHDU(data, sharp_header(component)).writeto(path)
        paths.append(path)
    return paths


def change_card(path, name, value):
    with fits.open(path, mode="update", memmap=False) as hdus:
        if value is None:
            del hdus[0].header[name]
        else:
            hdus[0].header[name] = value


def test_sharp_component_order_units_provenance_and_input_unchanged(tmp_path):
    paths = sharp_files(tmp_path)
    with fits.open(paths[1], mode="update", memmap=False) as hdus:
        hdus[0].data *= 1e-4
        hdus[0].header["BUNIT"] = "T"
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    result = read_sharp_cea_boundary(*paths)
    assert result.field_G.shape == (3, 3, 4)
    np.testing.assert_allclose(result.field_G[:, 0, 0], [1, -2, 3])
    np.testing.assert_equal(result.bx_G, result.field_G[0])
    np.testing.assert_equal(result.by_G, result.field_G[1])
    np.testing.assert_equal(result.bz_G, result.field_G[2])
    assert result.valid_mask.all()
    assert result.metadata["record_utc"] == "2000-01-01T00:00:00.000000000"
    assert (
        result.metadata["record_tai_ns"]
        == normalize_time("2000-01-01T00:00:00Z").tai_ns
    )
    assert result.metadata["series_verified"]
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths] == hashes
    assert [
        result.metadata["sources"][c]["sha256"] for c in ("Br", "Bt", "Bp")
    ] == hashes
    json.dumps(result.metadata, allow_nan=False)
    result.header["HARPNUM"] = 99
    assert fits.getheader(paths[0])["HARPNUM"] == 42


def test_sharp_missing_samples_are_not_zero_filled(tmp_path):
    arrays = [np.ones((3, 4)) for _ in range(3)]
    arrays[0][0, 1] = np.nan
    arrays[1][1, 1] = np.inf
    result = read_sharp_cea_boundary(*sharp_files(tmp_path, arrays=arrays))
    assert np.isnan(result.bz_G[0, 1]) and np.isnan(result.by_G[1, 1])
    assert result.bx_G[0, 1] == 1 and result.bx_G[1, 1] == 1
    assert not result.valid_mask[0, 1] and not result.valid_mask[1, 1]
    assert result.valid_mask.sum() == 10
    assert result.metadata["missing_pixels"] == 2


def test_sharp_fits_integer_blank_is_preserved_as_missing(tmp_path):
    paths = sharp_files(tmp_path)
    header = sharp_header("Br")
    data = np.ones((3, 4), dtype=np.int16)
    data[0, 0] = -32768
    header["BLANK"] = -32768
    fits.PrimaryHDU(data, header).writeto(paths[0], overwrite=True)
    result = read_sharp_cea_boundary(*paths)
    assert np.isnan(result.bz_G[0, 0]) and not result.valid_mask[0, 0]
    assert result.valid_mask.sum() == 11


def test_sharp_gzip_fallback_and_empty_primary(tmp_path):
    paths = sharp_files(tmp_path)
    header = fits.getheader(paths[0])
    data = fits.getdata(paths[0])
    fits.HDUList([fits.PrimaryHDU(header=header), fits.ImageHDU(data)]).writeto(
        paths[0], overwrite=True
    )
    for path in paths:
        with gzip.open(str(path) + ".gz", "wb") as stream:
            stream.write(path.read_bytes())
        path.unlink()
    result = read_sharp_cea_boundary(*paths)
    np.testing.assert_equal(result.field_G[:, 0, 0], [1, -2, 3])
    assert all(
        record["path"].endswith(".fits.gz")
        for record in result.metadata["sources"].values()
    )


def test_sharp_equal_tai_utc_and_absent_series_are_explicit(tmp_path):
    paths = sharp_files(tmp_path)
    change_card(paths[1], "T_REC", "2000-01-01T00:00:00Z")
    change_card(paths[2], "T_REC", "2000.01.01_00:00:32")
    change_card(paths[1], "SERIES", None)
    result = read_sharp_cea_boundary(*paths)
    assert not result.metadata["series_verified"]
    assert result.metadata["series"] == [
        "hmi.sharp_cea_720s",
        None,
        "hmi.sharp_cea_720s",
    ]


def test_sharp_equivalent_cd_and_pc_wcs(tmp_path):
    paths = sharp_files(tmp_path)
    with fits.open(paths[2], mode="update", memmap=False) as hdus:
        header = hdus[0].header
        del header["CDELT1"], header["CDELT2"]
        header.update({"CD1_1": 0.03, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": 0.03})
    assert read_sharp_cea_boundary(*paths).valid_mask.all()


def test_sharp_filename_identity_and_default_pv_are_supported(tmp_path):
    paths = sharp_files(tmp_path)
    for path in paths:
        change_card(path, "SEGMENT", None)
    change_card(paths[1], "PV2_1", None)
    result = read_sharp_cea_boundary(*paths)
    assert result.metadata["component_identity_source"] == ["filename"] * 3


@pytest.mark.parametrize("mode", ["anonymous", "swapped", "contradiction"])
def test_sharp_component_identity_cannot_be_guessed(tmp_path, mode):
    paths = sharp_files(tmp_path)
    if mode == "anonymous":
        change_card(paths[0], "SEGMENT", None)
        target = tmp_path / "anonymous.fits"
        paths[0].rename(target)
        paths[0] = target
    elif mode == "swapped":
        for path in paths:
            change_card(path, "SEGMENT", None)
        paths[0], paths[2] = paths[2], paths[0]
    else:
        # A valid-looking SEGMENT cannot override a contradictory known name.
        target = tmp_path / "wrong.Bp.fits"
        paths[0].rename(target)
        paths[0] = target
    with pytest.raises(ValueError, match="identity|Filename"):
        read_sharp_cea_boundary(*paths)


def test_sharp_present_observation_time_is_separate_from_record_time(tmp_path):
    paths = sharp_files(tmp_path)
    change_card(paths[0], "T_OBS", "2000.01.01_00:00:35_TAI")
    change_card(paths[1], "T_OBS", "2000-01-01T00:00:03Z")
    assert (
        read_sharp_cea_boundary(*paths).metadata["observation_utc"]
        == "2000-01-01T00:00:03.000000000"
    )
    change_card(paths[1], "T_OBS", "2000-01-01T00:00:04Z")
    with pytest.raises(ValueError, match="T_OBS"):
        read_sharp_cea_boundary(*paths)


@pytest.mark.parametrize(
    "name,value,match",
    [
        ("QUALITY", 1, "QUALITY"),
        ("QUALITY", -1, "QUALITY"),
        ("QUALITY", True, "QUALITY"),
        ("QUALITY", None, "QUALITY"),
        ("HARPNUM", 43, "HARPNUM"),
        ("HARPNUM", None, "HARPNUM"),
        ("HARPNUM", 0, "HARPNUM"),
        ("T_REC", "2000.01.01_00:00:32.000000001_TAI", "T_REC"),
        ("T_REC", None, "T_REC"),
        ("SERIES", "hmi.sharp_cea_720s_nrt", "definitive"),
        ("BUNIT", None, "magnetic"),
        ("BUNIT", "K", "magnetic"),
        ("SEGMENT", "Bp", "SEGMENT"),
        ("CTYPE1", "HPLN-TAN", "CEA"),
        ("CRPIX1", 2.1, "WCS"),
        ("CRVAL2", 5.01, "WCS"),
        ("CDELT1", -0.03, "WCS"),
        ("PV2_1", 0.9, "WCS"),
        ("LONPOLE", 180, "WCS"),
        ("LATPOLE", -85, "WCS"),
    ],
)
def test_sharp_rejects_mismatched_or_unknown_input(tmp_path, name, value, match):
    paths = sharp_files(tmp_path)
    change_card(paths[0], name, value)
    with pytest.raises(ValueError, match=match):
        read_sharp_cea_boundary(*paths)


@pytest.mark.parametrize("mode", ["shape", "ndim", "all_missing", "duplicate"])
def test_sharp_rejects_unusable_component_arrays(tmp_path, mode):
    arrays = [np.ones((3, 4)) for _ in range(3)]
    if mode == "shape":
        arrays[0] = np.ones((3, 5))
    if mode == "ndim":
        arrays[0] = np.ones((1, 3, 4))
    if mode == "all_missing":
        arrays[0][:] = np.nan
    paths = sharp_files(tmp_path, arrays=arrays)
    if mode == "duplicate":
        paths[1] = paths[0]
    with pytest.raises(ValueError):
        read_sharp_cea_boundary(*paths)


def metadata(**changes):
    return {
        "schema_version": 1,
        "frame": "local_west_north_radial",
        "array_axis_order": "component,z,y,x",
        "field_unit": "G",
        "length_unit": "Mm",
        **changes,
    }


def grid_file(tmp_path, *, meta=None, **arrays):
    values = {
        "volume": np.ones((3, 2, 3, 4)),
        "spacing": np.array([1, 2, 3]),
        "metadata": np.asarray(json.dumps(metadata() if meta is None else meta)),
    }
    values.update(arrays)
    path = tmp_path / "grid.npz"
    np.savez(path, **values)
    return path


def test_imported_grid_units_axes_optional_arrays_and_detachment(tmp_path):
    shape = (3, 2, 3, 4)
    path = grid_file(
        tmp_path,
        meta=metadata(field_unit="T", length_unit="km"),
        reference=np.full(shape, 2.0),
        observed=np.full((3, 3, 4), 3.0),
    )
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    result = read_nlfff_grid(path)
    np.testing.assert_equal(result.volume_G, np.full(shape, 1e4))
    np.testing.assert_equal(result.reference_G, np.full(shape, 2e4))
    np.testing.assert_equal(result.observed_G, np.full((3, 3, 4), 3e4))
    np.testing.assert_allclose(result.spacing_Mm, [0.001, 0.002, 0.003])
    assert not result.metadata["model_validated"]
    assert result.metadata["source_metadata"]["field_unit"] == "T"
    assert result.metadata["source"]["sha256"] == before
    result.volume_G[:] = 0
    with np.load(path, allow_pickle=False) as source:
        np.testing.assert_equal(source["volume"], np.ones(shape))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    json.dumps(result.metadata, allow_nan=False)


def test_imported_grid_optional_inputs_absent(tmp_path):
    result = read_nlfff_grid(grid_file(tmp_path))
    assert result.reference_G is None and result.observed_G is None


@pytest.mark.parametrize(
    "key,value",
    [
        ("volume", np.ones((2, 3, 4))),
        ("volume", np.ones((2, 2, 3, 4))),
        ("volume", np.ones((3, 0, 3, 4))),
        ("volume", np.full((3, 2, 3, 4), np.nan)),
        ("volume", np.full((3, 2, 3, 4), np.inf)),
        ("volume", np.ones((3, 2, 3, 4), dtype=complex)),
        ("volume", np.ones((3, 2, 3, 4), dtype=bool)),
        ("volume", np.array([{"unsafe": True}], dtype=object)),
        ("spacing", [1, 0, 3]),
        ("spacing", [1, -2, 3]),
        ("spacing", [1, np.inf, 3]),
        ("spacing", [1, 2]),
        ("spacing", [[1, 2, 3]]),
        ("reference", np.ones((3, 1, 3, 4))),
        ("reference", np.full((3, 2, 3, 4), np.nan)),
        ("observed", np.ones((3, 4, 3))),
        ("observed", np.full((3, 3, 4), np.nan)),
        ("metadata", np.array({}, dtype=object)),
        ("metadata", np.array([json.dumps(metadata())])),
        ("metadata", np.array(json.dumps(metadata()).encode())),
        ("metadata", np.array("not json")),
        ("unknown", np.ones(1)),
    ],
)
def test_imported_grid_rejects_invalid_arrays(tmp_path, key, value):
    with pytest.raises(ValueError):
        read_nlfff_grid(grid_file(tmp_path, **{key: value}))


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"schema_version": None},
        {"frame": "heliographic_carrington"},
        {"array_axis_order": "z,y,x,component"},
        {"field_unit": "K"},
        {"field_unit": None},
        {"length_unit": "s"},
        {"length_unit": None},
        {"unrelated": float("nan")},
        {"unrelated": float("inf")},
    ],
)
def test_imported_grid_rejects_unknown_metadata_conventions(tmp_path, change):
    with pytest.raises(ValueError):
        read_nlfff_grid(grid_file(tmp_path, meta=metadata(**change)))


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        "null",
        '{"schema_version":1,"schema_version":1}',
        json.dumps(metadata())[:-1] + ',"huge":1e400}',
    ],
)
def test_imported_grid_rejects_nonobject_duplicate_or_overflow_json(tmp_path, text):
    with pytest.raises(ValueError):
        read_nlfff_grid(grid_file(tmp_path, metadata=np.array(text)))


def test_imported_grid_requires_npz_and_required_arrays(tmp_path):
    path = tmp_path / "raw.npy"
    np.save(path, np.ones(3))
    with pytest.raises(ValueError, match="NPZ"):
        read_nlfff_grid(path)
    path = tmp_path / "missing.npz"
    np.savez(path, volume=np.ones((3, 2, 3, 4)))
    with pytest.raises(ValueError, match="required"):
        read_nlfff_grid(path)


def test_imported_grid_reports_damaged_npz_as_valueerror(tmp_path):
    path = tmp_path / "damaged.npz"
    path.write_bytes(b"PK\x03\x04not a valid zip archive")
    with pytest.raises(ValueError, match="Damaged NLFFF NPZ"):
        read_nlfff_grid(path)


def test_modeling_lazy_namespace_exports_nlfff():
    from solar_toolkit import modeling

    assert modeling.nlfff.read_nlfff_grid is read_nlfff_grid
