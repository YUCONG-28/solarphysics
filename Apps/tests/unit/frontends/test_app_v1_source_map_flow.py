from __future__ import annotations

import json
from pathlib import Path

from solar_apps.frontends.app_v1 import source_map_flow_worker


def test_flow_worker_uses_native_source_map_request_schema(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "radio"
    output = tmp_path / "output"
    source.mkdir()
    calls: list[list[str]] = []

    def run_worker(arguments: list[str]) -> int:
        calls.append(arguments)
        return 0

    monkeypatch.setattr(source_map_flow_worker, "source_map_worker_main", run_worker)

    result = source_map_flow_worker.main(
        [
            "--config",
            "solar_apps.workflows.radio.configs.synthetic_radio_config",
            "--source-path",
            str(source),
            "--start-idx",
            "2",
            "--end-idx",
            "5",
            "--image-time-start",
            "2025-01-24T04:48:30Z",
            "--image-time-end",
            "2025-01-24T04:49:00Z",
            "--output-dir",
            str(output),
            "--allowed-roots",
            str(tmp_path),
        ]
    )

    assert result == 0
    assert len(calls) == 2
    request = json.loads((output / "flow-source-map-request.json").read_text())
    config = request["config"]
    assert config["config"].endswith("synthetic_radio_config")
    assert config["start_idx"] == 2
    assert config["end_idx"] == 5
    assert config["image_time_start"] == "2025-01-24T04:48:30Z"
    assert config["image_time_end"] == "2025-01-24T04:49:00Z"
    assert config["image_window_use_indices"] is True
    assert config["cmap"] == "hot"
    assert "display" not in config
    assert "features" not in config
