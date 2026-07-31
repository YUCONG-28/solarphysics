from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from solar_apps.frontends.app_v1 import data_download_worker
from solar_toolkit.net.observations import (
    ObservationCollectionV1,
    ObservationDownloadItemV1,
)


def test_search_worker_persists_empty_result_and_manifest(
    tmp_path: Path, monkeypatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        data_download_worker,
        "search_observations",
        lambda _query: [],
    )
    output = tmp_path / "search"

    status = data_download_worker.main(
        [
            "search",
            "--product-id",
            "stereo-euvi",
            "--start-utc",
            "2025-01-01T00:00:00Z",
            "--end-utc",
            "2025-01-01T00:01:00Z",
            "--spacecraft",
            "stereo-a,stereo-b",
            "--wavelengths",
            "171",
            "--output-dir",
            str(output),
        ]
    )

    result = json.loads((output / "search-results.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert status == 0
    assert result["record_count"] == 0
    assert manifest["products"][0]["kind"] == "remote-observation-set"
    assert '"record_count":0' in capsys.readouterr().out


def test_download_worker_reports_partial_failure(
    tmp_path: Path, monkeypatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    selection = tmp_path / "selection.json"
    selection.write_text('{"records":[{"record_id":"placeholder"}]}', encoding="utf-8")
    monkeypatch.setattr(
        data_download_worker, "read_remote_records", lambda _path: [object()]
    )
    collection = ObservationCollectionV1(
        "collection-test",
        (
            ObservationDownloadItemV1(
                "record-one",
                str(tmp_path / "observations" / "one.fts"),
                "failed",
                error="network unavailable",
            ),
        ),
        created_at_utc=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
    )
    monkeypatch.setattr(
        data_download_worker,
        "download_observations",
        lambda *_args, **_kwargs: collection,
    )
    output = tmp_path / "download"

    status = data_download_worker.main(
        [
            "download",
            "--selection",
            str(selection),
            "--observation-root",
            str(tmp_path / "observations"),
            "--output-dir",
            str(output),
        ]
    )

    receipt = json.loads((output / "download-receipt.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert status == 1
    assert receipt["items"][0]["status"] == "failed"
    assert manifest["status"] == "failed"
    assert '"failed":1' in capsys.readouterr().out
