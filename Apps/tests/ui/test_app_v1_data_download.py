from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from solar_apps.frontends.app_v1.data_download_page import DataDownloadPanel
from solar_apps.frontends.app_v1.runtime import AppV1RuntimePaths
from solar_toolkit.net.observations import (
    ObservationQueryV1,
    RemoteObservationV1,
    write_search_result,
)

_APP = QApplication.instance() or QApplication(["data-download-test"])


def _application() -> QApplication:
    return _APP


def _runtime(tmp_path: Path) -> AppV1RuntimePaths:
    return AppV1RuntimePaths(
        tmp_path / "state",
        tmp_path / "workspaces",
        tmp_path / "outputs",
        tmp_path / "logs",
        tmp_path / "tmp",
        tmp_path / "observations",
    ).ensure()


def _record() -> RemoteObservationV1:
    observed = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
    locator = "soho/lasco/level_05/20250101/20250101_000000_lasco_c2.fts"
    return RemoteObservationV1(
        "record-" + hashlib.sha256(locator.encode()).hexdigest()[:24],
        "soho-lasco",
        "sdac",
        "soho",
        "soho",
        "lasco",
        "c2",
        observed,
        observed,
        locator,
        "20250101_000000_lasco_c2.fts",
        "soho/lasco/20250101/c2/20250101_000000_lasco_c2.fts",
        level="0.5",
        format="fts",
        size_bytes=1024,
        size_is_estimate=True,
        metadata={"fileid": "example.fts"},
    )


def test_product_controls_include_stereo_ab_and_historical_lasco_c1(
    tmp_path: Path,
) -> None:
    application = _application()
    panel = DataDownloadPanel(_runtime(tmp_path))

    assert application is QApplication.instance()
    assert panel.product.count() == 7
    lasco_index = panel.product.findData("soho-lasco")
    panel.product.setCurrentIndex(lasco_index)
    assert panel.detectors.text() == "c2,c3"
    assert panel.historical_c1.isVisibleTo(panel)
    panel.historical_c1.setChecked(True)
    assert panel.detectors.text() == "c2,c3,c1"

    stereo_index = panel.product.findData("stereo-euvi")
    panel.product.setCurrentIndex(stereo_index)
    assert panel.spacecraft.itemData(0) == "stereo-a,stereo-b"
    assert panel.spacecraft.itemText(0) == "All available"
    assert not panel.historical_c1.isVisibleTo(panel)
    panel.close()


def test_search_results_are_unchecked_and_selection_updates_summary(
    tmp_path: Path,
) -> None:
    _application()
    panel = DataDownloadPanel(_runtime(tmp_path))
    query = ObservationQueryV1(
        "query-ui",
        "soho-lasco",
        dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
        dt.datetime(2025, 1, 1, 0, 1, tzinfo=dt.UTC),
        detectors=("c2",),
    )
    artifact = write_search_result(
        tmp_path / "search-results.json",
        query,
        [_record()],
    )

    panel.handle_artifact(artifact)

    assert panel.results.rowCount() == 1
    assert panel.results.columnCount() == 10
    assert panel.results.item(0, 0).checkState() == Qt.CheckState.Unchecked
    assert panel.results.item(0, 1).text() == "sdac"
    assert panel.results.item(0, 4).text() == "SOHO/LASCO"
    assert not panel.download_button.isEnabled()
    assert "Nothing is selected" in panel.summary.text()

    panel._set_all_checked(True)
    assert panel.download_button.isEnabled()
    assert panel.selected_records() == (_record(),)
    assert "1 selected" in panel.summary.text()

    panel._set_all_checked(False)
    assert not panel.download_button.isEnabled()
    panel.close()


def test_download_receipt_reports_partial_failure(tmp_path: Path) -> None:
    _application()
    panel = DataDownloadPanel(_runtime(tmp_path))
    receipt = tmp_path / "download-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "items": [
                    {"record_id": "one", "status": "downloaded"},
                    {"record_id": "two", "status": "failed"},
                ]
            }
        ),
        encoding="utf-8",
    )

    panel.handle_artifact(receipt)

    assert panel.summary.text() == "Download receipt: 2 item(s), 1 failed."
    panel.close()
