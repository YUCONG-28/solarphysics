from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from solar_apps.workflows.radio import source_map_workflow
from solar_apps.workflows.radio.source_map_workflow import _sorted_fits_for_band
from tools.freeze_radio_collection import build


def _record(
    path: Path,
    root: Path,
    *,
    observed_utc: str = "2025-01-24T04:48:30.000Z",
) -> dict[str, object]:
    relative_path = path.relative_to(root).as_posix()
    identity = f"{observed_utc}\0{relative_path}"
    return {
        "record_id": "radio-" + hashlib.sha256(identity.encode()).hexdigest()[:24],
        "observed_utc": observed_utc,
        "relative_path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _manifest(root: Path, paths: list[Path]) -> dict[str, object]:
    records = [_record(path, root) for path in paths]
    return {
        "schema": "solar-radio-frozen-collection-v1",
        "selection": {
            "start_utc": "2025-01-24T04:48:00.000Z",
            "end_utc": "2025-01-24T04:49:00.000Z",
        },
        "record_count": len(records),
        "records": records,
    }


def _write_manifest(root: Path, payload: dict[str, object]) -> None:
    (root / ".frozen-collection-v1.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_unrelated_early_file_does_not_change_frozen_selection(tmp_path: Path) -> None:
    band = tmp_path / "149MHz" / "RR"
    band.mkdir(parents=True)
    selected = band / "b.fits"
    selected.write_bytes(b"selected")
    _write_manifest(tmp_path, _manifest(tmp_path, [selected]))

    assert _sorted_fits_for_band(
        str(band), 0, 1, study_mode="confirmatory"
    ) == [str(selected)]
    (band / "a.fits").write_bytes(b"unrelated")
    assert _sorted_fits_for_band(
        str(band), 0, 1, study_mode="confirmatory"
    ) == [str(selected)]


def test_confirmatory_without_manifest_fails_closed(tmp_path: Path) -> None:
    band = tmp_path / "149MHz" / "RR"
    band.mkdir(parents=True)
    only = band / "only.fits"
    only.write_bytes(b"fits")

    with pytest.raises(FileNotFoundError, match="requires .frozen-collection"):
        _sorted_fits_for_band(
            str(band), 0, 1, study_mode="confirmatory"
        )

    assert _sorted_fits_for_band(
        str(band), 0, 1, study_mode="exploratory"
    ) == [str(only)]
    with pytest.raises(ValueError, match="study_mode must be explicit"):
        _sorted_fits_for_band(str(band), 0, 1)


def test_naive_utc_is_rejected_by_freezer_and_manifest_reader(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        build(
            tmp_path,
            datetime(2025, 1, 24, 4, 48),
            datetime(2025, 1, 24, 4, 49),
        )

    band = tmp_path / "149MHz" / "RR"
    band.mkdir(parents=True)
    selected = band / "b.fits"
    selected.write_bytes(b"selected")
    payload = _manifest(tmp_path, [selected])
    selection = payload["selection"]
    assert isinstance(selection, dict)
    selection["start_utc"] = "2025-01-24T04:48:00"
    _write_manifest(tmp_path, payload)

    with pytest.raises(ValueError, match="Z suffix"):
        _sorted_fits_for_band(
            str(band), 0, 1, study_mode="confirmatory"
        )


def test_end_not_after_start_is_rejected(tmp_path: Path) -> None:
    aware = datetime(2025, 1, 24, 4, 48, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="later than start_utc"):
        build(tmp_path, aware, aware)

    band = tmp_path / "149MHz" / "RR"
    band.mkdir(parents=True)
    selected = band / "b.fits"
    selected.write_bytes(b"selected")
    payload = _manifest(tmp_path, [selected])
    selection = payload["selection"]
    assert isinstance(selection, dict)
    selection["end_utc"] = selection["start_utc"]
    _write_manifest(tmp_path, payload)

    with pytest.raises(ValueError, match="end_utc > start_utc"):
        _sorted_fits_for_band(
            str(band), 0, 1, study_mode="confirmatory"
        )


def test_duplicate_record_id_and_wrong_sha_are_rejected(tmp_path: Path) -> None:
    band = tmp_path / "149MHz" / "RR"
    band.mkdir(parents=True)
    first = band / "first.fits"
    second = band / "second.fits"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    payload = _manifest(tmp_path, [first, second])
    records = payload["records"]
    assert isinstance(records, list)
    assert isinstance(records[0], dict) and isinstance(records[1], dict)
    records[1]["record_id"] = records[0]["record_id"]
    _write_manifest(tmp_path, payload)

    with pytest.raises(ValueError, match="duplicate record_id"):
        _sorted_fits_for_band(
            str(band), 0, 2, study_mode="confirmatory"
        )

    payload = _manifest(tmp_path, [first])
    records = payload["records"]
    assert isinstance(records, list) and isinstance(records[0], dict)
    records[0]["sha256"] = "0" * 64
    _write_manifest(tmp_path, payload)
    with pytest.raises(ValueError, match="SHA mismatch"):
        _sorted_fits_for_band(
            str(band), 0, 1, study_mode="confirmatory"
        )


def test_record_id_is_bound_to_utc_and_path(tmp_path: Path) -> None:
    band = tmp_path / "149MHz" / "RR"
    band.mkdir(parents=True)
    selected = band / "selected.fits"
    selected.write_bytes(b"selected")
    payload = _manifest(tmp_path, [selected])
    records = payload["records"]
    assert isinstance(records, list) and isinstance(records[0], dict)
    records[0]["record_id"] = "radio-" + "0" * 24
    _write_manifest(tmp_path, payload)

    with pytest.raises(ValueError, match="not bound to UTC/path"):
        _sorted_fits_for_band(
            str(band), 0, 1, study_mode="confirmatory"
        )


def test_freezer_emits_canonical_utc_z_and_bound_record_ids(tmp_path: Path) -> None:
    band = tmp_path / "149MHz" / "RR"
    band.mkdir(parents=True)
    selected = band / "149MHz_2025124_044830_000.fits"
    selected.write_bytes(b"fits")

    payload = build(
        tmp_path,
        datetime(2025, 1, 24, 4, 48, tzinfo=timezone.utc),
        datetime(2025, 1, 24, 4, 49, tzinfo=timezone.utc),
    )

    assert payload["selection"] == {
        "start_utc": "2025-01-24T04:48:00.000Z",
        "end_utc": "2025-01-24T04:49:00.000Z",
    }
    record = payload["records"][0]
    assert record["observed_utc"] == "2025-01-24T04:48:30.000Z"
    identity = f"{record['observed_utc']}\0{record['relative_path']}"
    assert record["record_id"] == (
        "radio-" + hashlib.sha256(identity.encode()).hexdigest()[:24]
    )


def test_confirmatory_multi_band_forbids_positional_time_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        source_map_workflow,
        "_sorted_fits_for_band",
        lambda *_args, **_kwargs: ["unparseable-radio-time.fits"],
    )
    cfg = {
        "multi_band_root": "/unused",
        "multi_band_freqs": [149],
        "band_dir_pattern": "{freq}MHz/{polar}",
        "polarization": "RR",
        "start_idx": 0,
        "end_idx": 1,
        "study_mode": "confirmatory",
        "combine_polarizations": False,
        "enable_raw_quality_filter": False,
    }

    with pytest.raises(ValueError, match="positional fallback is forbidden"):
        source_map_workflow._build_multi_band_slots(cfg)
