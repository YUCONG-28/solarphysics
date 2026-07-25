"""Isolated worker process for one source-map render job."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import traceback
from pathlib import Path
from typing import Any, Callable

from .artifacts import sidecar_path_for, validate_source_map_artifact

FIGURE_RENDER_LOCK = threading.RLock()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render one source-map artifact.")
    parser.add_argument("--job-file", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--progress-file")
    return parser


def _render_candidate(
    cfg: dict[str, Any], candidate: dict[str, Any], *, sequence: int
) -> dict[str, Any]:
    from solar_apps.workflows.radio import source_map_workflow as workflow

    output_dir = str(cfg["output_dir"])
    if candidate["mode"] == "multi_band":
        slot = [
            tuple(item) if isinstance(item, list) else item
            for item in candidate["slot"]
        ]
        image_path = workflow.plot_multi_band_slot(
            int(candidate["slot_index"]),
            slot,
            output_dir,
            cfg,
            vmin=None,
            vmax=None,
            sequence=int(sequence),
            write_sidecar=True,
        )
    else:
        image_path = workflow.plot_single_band(
            str(candidate["run_path"]),
            output_dir,
            cfg,
            vmin=None,
            vmax=None,
            sequence=int(sequence),
            write_sidecar=True,
        )
    sidecar_path = sidecar_path_for(image_path)
    metadata = validate_source_map_artifact(image_path, sidecar_path)
    return {
        "ok": True,
        "image_path": str(Path(image_path).resolve()),
        "sidecar_path": str(sidecar_path.resolve()),
        "image_sha256": metadata["image"]["sha256"],
        "candidate_id": str(candidate["id"]),
        "sequence": int(sequence),
    }


def run_job(
    payload: dict,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict:
    cfg = dict(payload["config"])
    raw_candidates = payload.get("candidates")
    if raw_candidates is None:
        candidate = dict(payload["candidate"])
        sequence = int(payload.get("sequence") or 1)
        with FIGURE_RENDER_LOCK:
            return _render_candidate(cfg, candidate, sequence=sequence)

    candidates = [dict(candidate) for candidate in raw_candidates]
    if not candidates:
        raise ValueError("A sequence render requires at least one candidate")
    results: list[dict[str, Any]] = []
    total = len(candidates)
    for offset, candidate in enumerate(candidates):
        sequence = int(candidate.get("sequence") or offset + 1)
        if progress is not None:
            progress(
                {
                    "status": "running",
                    "total": total,
                    "completed": offset,
                    "current_frame": offset + 1,
                    "candidate_id": str(candidate["id"]),
                }
            )
        with FIGURE_RENDER_LOCK:
            results.append(_render_candidate(cfg, candidate, sequence=sequence))
        if progress is not None:
            progress(
                {
                    "status": "running",
                    "total": total,
                    "completed": offset + 1,
                    "current_frame": offset + 1,
                    "candidate_id": str(candidate["id"]),
                }
            )
    return {"ok": True, "artifacts": results}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.stem}-", suffix=".json", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write((json.dumps(payload, indent=2) + "\n").encode("utf-8"))
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result_path = Path(args.result_file)
    progress_path = Path(args.progress_file) if args.progress_file else None
    try:
        payload = json.loads(Path(args.job_file).read_text(encoding="utf-8"))
        result = run_job(
            payload,
            progress=(
                (lambda update: _atomic_write_json(progress_path, update))
                if progress_path is not None
                else None
            ),
        )
        status = 0
    except Exception as exc:
        result = {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}
        status = 1
    _atomic_write_json(result_path, result)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
