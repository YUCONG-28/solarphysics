from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from solar_apps.frontends.radio.source_map import worker
from solar_apps.frontends.radio.source_map.server import (
    _inclusive_frame_range,
    create_app,
)


def _write_map(path: Path, observed: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = fits.Header()
    header["DATE-OBS"] = observed
    header["FREQ"] = 149.0
    header["POLAR"] = "RR"
    header["BUNIT"] = "K"
    header["CRVAL1"] = 0.0
    header["CRPIX1"] = 2.5
    header["CDELT1"] = 10.0
    header["CRVAL2"] = 0.0
    header["CRPIX2"] = 2.5
    header["CDELT2"] = 10.0
    fits.PrimaryHDU(np.ones((4, 4), dtype=np.float32), header=header).writeto(path)


def test_inclusive_frame_range_is_one_based_and_bounded() -> None:
    assert _inclusive_frame_range(None, None, 4) == (1, 4)
    assert _inclusive_frame_range("2", "4", 4) == (2, 4)
    for start, end in ((0, 1), (3, 2), (1, 5), ("x", 2)):
        with pytest.raises(ValueError):
            _inclusive_frame_range(start, end, 4)


def test_sequence_worker_preserves_discovery_ordinals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rendered: list[tuple[str, int]] = []

    def fake_plot_single_band(
        run_path,
        output_dir,
        cfg,
        vmin,
        vmax,
        *,
        sequence,
        write_sidecar,
    ):
        del output_dir, cfg, vmin, vmax, write_sidecar
        rendered.append((str(run_path), int(sequence)))
        return str(tmp_path / f"frame-{sequence}.png")

    from solar_apps.workflows.radio import source_map_workflow

    monkeypatch.setattr(source_map_workflow, "plot_single_band", fake_plot_single_band)
    monkeypatch.setattr(
        worker, "sidecar_path_for", lambda path: Path(path).with_suffix(".json")
    )
    monkeypatch.setattr(
        worker,
        "validate_source_map_artifact",
        lambda image, sidecar: {"image": {"sha256": f"hash-{Path(image).stem}"}},
    )
    progress: list[dict] = []
    result = worker.run_job(
        {
            "config": {"output_dir": str(tmp_path)},
            "candidates": [
                {
                    "id": "file-0001",
                    "mode": "single_band",
                    "run_path": "one.fits",
                    "sequence": 2,
                },
                {
                    "id": "file-0002",
                    "mode": "single_band",
                    "run_path": "two.fits",
                    "sequence": 3,
                },
            ],
        },
        progress=progress.append,
    )

    assert rendered == [("one.fits", 2), ("two.fits", 3)]
    assert [item["sequence"] for item in result["artifacts"]] == [2, 3]
    assert progress[-1]["completed"] == 2
    assert progress[-1]["total"] == 2


def test_in_process_worker_serializes_matplotlib_rendering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = threading.Barrier(2)
    state_lock = threading.Lock()
    active = 0
    maximum_active = 0

    def fake_render(cfg, candidate, *, sequence):
        del cfg, sequence
        nonlocal active, maximum_active
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.04)
        with state_lock:
            active -= 1
        return {"ok": True, "candidate_id": candidate["id"]}

    monkeypatch.setattr(worker, "_render_candidate", fake_render)

    def invoke(candidate_id: str):
        start.wait(timeout=2.0)
        return worker.run_job(
            {
                "config": {"output_dir": "unused"},
                "candidate": {"id": candidate_id},
                "sequence": 1,
            }
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(invoke, ("one", "two")))

    assert {result["candidate_id"] for result in results} == {"one", "two"}
    assert maximum_active == 1


def test_sequence_api_freezes_requested_discovery_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "149MHz" / "RR"
    _write_map(source / "001.fits", "2025-01-24T04:48:30")
    _write_map(source / "002.fits", "2025-01-24T04:48:31")
    _write_map(source / "003.fits", "2025-01-24T04:48:32")
    output = tmp_path / "output"
    app = create_app([tmp_path], stop_on_client_close=False)
    client = app.test_client()
    discovered = client.post(
        "/api/source-maps/discover",
        json={
            "mode": "single_band",
            "source_path": str(source),
            "output_dir": str(output),
            "polarization": "RR",
            "gaussian_overlay": False,
            "spectrogram_panel": False,
            "background_mode": "off",
            "cmap": "hot",
            "color_range_mode": "auto",
            "advanced": {"start_idx": 0, "end_idx": 3},
        },
    )
    assert discovered.status_code == 200, discovered.get_json()
    discovery_id = discovered.get_json()["discovery_id"]
    captured: dict = {}

    def fake_start_sequence(config, candidates):
        captured["config"] = config
        captured["candidates"] = candidates
        return {
            "id": "sequence-job",
            "kind": "sequence",
            "status": "running",
            "total": len(candidates),
            "completed": 0,
            "artifact_ids": [],
        }

    monkeypatch.setattr(
        app.extensions["source_map_jobs"], "start_sequence", fake_start_sequence
    )
    response = client.post(
        "/api/sequence-jobs",
        json={"discovery_id": discovery_id, "start_frame": 2, "end_frame": 3},
    )
    assert response.status_code == 202, response.get_json()
    assert [item["id"] for item in captured["candidates"]] == [
        "file-0001",
        "file-0002",
    ]
    assert [item["sequence"] for item in captured["candidates"]] == [2, 3]
