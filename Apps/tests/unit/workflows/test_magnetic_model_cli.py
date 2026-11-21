"""Public magnetic-model workflows on generated, backend-independent fixtures."""

from __future__ import annotations

import gzip
import importlib.abc
import json
import sys
from io import BytesIO

import numpy as np
import pytest
from astropy.io import fits

from solar_apps.cli import router
from solar_toolkit.modeling.pfss.bundle import save_bundle


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    disconnected = tmp_path / "unused-runtime"
    monkeypatch.setenv("SOLAR_APPS_LOCAL_ROOT", str(disconnected))
    monkeypatch.setattr(router, "inspect_miniforge_runtime", lambda: None)
    return disconnected


def _bundle(path, *, stored_rss=True):
    xyz = np.array([[1.1, 0, 0], [2.1, 0, 0]])
    records = [
        {
            "fieldline_id": "measured",
            "trace_valid": True,
            "xyz_carrington_rsun": xyz,
            "field_strength_status": "ok",
            "b_gauss": np.array([2.0, 12.0]),
        },
        {
            "fieldline_id": "old",
            "trace_valid": True,
            "xyz_carrington_rsun": xyz,
        },
        {
            "fieldline_id": "bad",
            "trace_valid": False,
            "reason": "trace_incomplete",
            "xyz_carrington_rsun": np.empty((0, 3)),
        },
    ]
    save_bundle(
        path,
        records,
        [
            {"hpc_arcsec": np.zeros((len(record["xyz_carrington_rsun"]), 2))}
            for record in records
        ],
        {"config": {"rss": 2.5}} if stored_rss else {},
    )
    return records


def _bands_arguments(bundle, allowed_root, *extra):
    return [
        "workflow",
        "pfss",
        "bands",
        "--bundle",
        str(bundle),
        "--frequency-bounds-mhz",
        "11.2",
        "16.8",
        "--mechanism",
        "cyclotron",
        "--harmonic",
        "1",
        "--allowed-roots",
        str(allowed_root),
        *extra,
    ]


class _NoPFSSBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"sunkit_magex", "streamtracer"}:
            pytest.fail(f"Read-only PFSS analysis attempted to import {fullname}")


def test_pfss_bands_use_verified_arrays_without_the_solver(
    runtime, tmp_path, monkeypatch, capsys
):
    from solar_toolkit.radio.fieldline_association import fieldline_band_segments

    bundle = tmp_path / "saved-bundle"
    records = _bundle(bundle)
    before = {part.name: part.read_bytes() for part in bundle.iterdir()}
    monkeypatch.setattr(sys, "meta_path", [_NoPFSSBackend(), *sys.meta_path])
    assert router.main(_bands_arguments(bundle, tmp_path)) == 0

    outcome = json.loads(capsys.readouterr().out)
    assert outcome["geometry_method"] == "conditional_model_candidate"
    first, old, bad = outcome["fieldlines"]
    assert first["status"] == "ok"
    expected = fieldline_band_segments(
        records[0]["xyz_carrington_rsun"],
        (11.2, 16.8),
        mechanism="cyclotron",
        harmonic=1,
        rss=2.5,
        field_strength_gauss=records[0]["b_gauss"],
    )
    assert first["segments"] == [
        {key: value.tolist() for key, value in segment.items()}
        for segment in expected["segments"]
    ]
    np.testing.assert_allclose(
        first["segments"][0]["source_edge_fractions"], [[0.2, 0.4]]
    )
    assert old["status"] == "field_strength_unavailable" and old["segments"] == []
    assert bad["status"] == "trace_invalid" and bad["reason"] == "trace_incomplete"
    assert {part.name: part.read_bytes() for part in bundle.iterdir()} == before
    assert not runtime.exists()


def test_pfss_bands_file_output_respects_bundle_and_existing_files(
    runtime, tmp_path, capsys
):
    bundle = tmp_path / "saved-bundle"
    _bundle(bundle)
    target = tmp_path / "report.json"
    args = _bands_arguments(bundle, tmp_path, "--output", str(target))
    assert router.main(args) == 0
    assert capsys.readouterr().out == ""
    original = target.read_bytes()
    assert json.loads(original)["rss_rsun"] == 2.5
    with pytest.raises(SystemExit) as error:
        router.main(args)
    assert error.value.code == 2
    assert "already exists" in capsys.readouterr().err
    assert target.read_bytes() == original

    with pytest.raises(SystemExit) as error:
        router.main(
            _bands_arguments(bundle, tmp_path, "--output", str(bundle / "new.json"))
        )
    assert error.value.code == 2
    assert "outside the completed PFSS bundle" in capsys.readouterr().err
    assert not (bundle / "new.json").exists()
    assert not runtime.exists()


def test_pfss_bands_requires_rss_for_old_metadata_and_rejects_tampering(
    runtime, tmp_path, capsys
):
    bundle = tmp_path / "saved-bundle"
    _bundle(bundle, stored_rss=False)
    args = _bands_arguments(bundle, tmp_path)
    with pytest.raises(SystemExit) as error:
        router.main(args)
    assert error.value.code == 2
    assert "Supply --rss" in capsys.readouterr().err
    args += ["--rss", "2.5"]
    assert router.main(args) == 0
    capsys.readouterr()
    with (bundle / "fieldlines.npz").open("ab") as stream:
        stream.write(b"synthetic alteration")
    with pytest.raises(SystemExit) as error:
        router.main(args)
    assert error.value.code == 2
    assert "checksum" in capsys.readouterr().err


def test_nlfff_result_inspection_reads_physical_grid_and_checks_root(
    runtime, tmp_path, capsys
):
    folder = tmp_path / "inputs"
    folder.mkdir()
    inferred = tmp_path / "regular-grid.npz"
    volume = np.zeros((3, 2, 3, 4))
    np.savez(
        inferred,
        metadata=json.dumps(
            {
                "schema_version": 1,
                "frame": "local_west_north_radial",
                "array_axis_order": "component,z,y,x",
                "field_unit": "T",
                "length_unit": "km",
            }
        ),
        volume=volume,
        spacing=np.array([200.0, 300.0, 700.0]),
        observed=volume[:, 0],
    )
    arguments = [
        "workflow",
        "nlfff",
        "inspect-result",
        "--result",
        str(inferred),
        "--allowed-roots",
    ]
    with pytest.raises(SystemExit) as error:
        router.main([*arguments, str(folder)])
    assert error.value.code == 2
    assert "outside allowed roots" in capsys.readouterr().err
    source = inferred.read_bytes()
    assert router.main([*arguments, str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["shape"] == [3, 2, 3, 4]
    assert (payload["field_unit"], payload["length_unit"]) == ("G", "Mm")
    np.testing.assert_allclose(payload["spacing_xyz_Mm"], [0.2, 0.3, 0.7])
    assert payload["has_observed"] and not payload["has_reference"]
    assert inferred.read_bytes() == source
    assert not runtime.exists()


def test_nlfff_boundary_inspection_resolves_gzip_component(runtime, tmp_path, capsys):
    header = fits.Header(
        {
            "CTYPE1": "CRLN-CEA",
            "CTYPE2": "CRLT-CEA",
            "CUNIT1": "deg",
            "CUNIT2": "deg",
            "CRPIX1": 2.0,
            "CRPIX2": 2.0,
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CDELT1": 0.03,
            "CDELT2": 0.03,
            "BUNIT": "G",
            "QUALITY": 0,
            "HARPNUM": 1,
            "T_REC": "2000.01.01_12:00:00_TAI",
        }
    )
    paths = [tmp_path / f"synthetic.{name}.fits" for name in ("Br", "Bt", "Bp")]
    array = np.ones((3, 4))
    array[-1, 0] = np.nan
    with BytesIO() as encoded, gzip.open(str(paths[0]) + ".gz", "wb") as compressed:
        fits.PrimaryHDU(array, header).writeto(encoded)
        compressed.write(encoded.getvalue())
    for path in paths[1:]:
        fits.writeto(path, np.ones((3, 4)), header)
    assert (
        router.main(
            [
                "workflow",
                "nlfff",
                "inspect-boundary",
                "--br",
                str(paths[0]),
                "--bt",
                str(paths[1]),
                "--bp",
                str(paths[2]),
                "--allowed-roots",
                str(tmp_path),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["shape"] == [3, 3, 4]
    assert payload["component_order"] == ["west", "north", "radial"]
    assert payload["valid_pixel_count"] == 11 and payload["invalid_pixel_count"] == 1
    assert not paths[0].exists()
    assert not runtime.exists()


@pytest.mark.parametrize(
    "domain_cmd,token",
    [
        (["pfss"], "--config"),
        (["pfss", "bands"], "--frequency-bounds-mhz"),
        (["nlfff", "inspect-boundary"], "--bp"),
        (["nlfff", "inspect-result"], "--result"),
    ],
)
def test_magnetic_workflow_help_needs_no_runtime(runtime, domain_cmd, token, capsys):
    with pytest.raises(SystemExit) as error:
        router.main(["workflow", *domain_cmd, "--help"])
    assert error.value.code == 0
    assert token in capsys.readouterr().out
    assert not runtime.exists()
