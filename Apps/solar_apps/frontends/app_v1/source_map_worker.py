# SPDX-License-Identifier: GPL-3.0-only
"""Process-isolated native Source Map discovery and rendering worker."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any

_PREFIX = "APP_V1_EVENT "


def _emit(kind: str, payload: dict[str, object]) -> None:
    print(
        _PREFIX
        + json.dumps(
            {
                "schema_version": 1,
                "module_id": "source-map",
                "kind": kind,
                "payload": payload,
            },
            separators=(",", ":"),
            default=str,
        ),
        flush=True,
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.stem}-",
        suffix=".json",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(
            (json.dumps(payload, indent=2, default=str) + "\n").encode("utf-8")
        )
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _request(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Source Map request must be a JSON object")
    return payload


def _discover(payload: dict[str, Any], result_path: Path) -> None:
    from solar_apps.frontends.radio.source_map.service import (
        PathPolicy,
        discover_candidates,
        parse_request_config,
        public_candidate,
    )

    policy = PathPolicy(payload["allowed_roots"])
    config = parse_request_config(payload["config"], policy=policy)
    candidates = discover_candidates(config, policy=policy)
    if not candidates:
        raise RuntimeError("No compatible Source Map candidates were found")
    result = {
        "schema_version": 1,
        "config": config,
        "candidates": candidates,
        "public_candidates": [public_candidate(item) for item in candidates],
    }
    _atomic_json(result_path, result)
    _emit("progress", {"percent": 100})
    _emit(
        "artifact",
        {
            "path": str(result_path),
            "role": "source-map-discovery",
            "candidate_count": len(candidates),
        },
    )
    _emit("result", {"status": "succeeded"})


def _render(
    payload: dict[str, Any],
    result_path: Path,
    *,
    sequence: bool,
) -> None:
    from solar_apps.frontends.radio.source_map.worker import run_job

    discovery = _request(payload["discovery_file"])
    candidates = list(discovery["candidates"])
    if not candidates:
        raise RuntimeError("Source Map discovery contains no candidates")
    if sequence:
        start = max(1, int(payload.get("start_frame", 1)))
        end = min(len(candidates), int(payload.get("end_frame", len(candidates))))
        if end < start:
            raise ValueError("End frame cannot precede start frame")
        selected = []
        for ordinal, candidate in enumerate(candidates[start - 1 : end], start=1):
            item = dict(candidate)
            item["sequence"] = ordinal
            selected.append(item)
        job = {"config": discovery["config"], "candidates": selected}
    else:
        index = int(payload.get("candidate_index", 0))
        if index < 0 or index >= len(candidates):
            raise IndexError("Selected Source Map candidate is out of range")
        job = {
            "config": discovery["config"],
            "candidate": candidates[index],
            "sequence": 1,
        }

    def progress(update: dict[str, Any]) -> None:
        total = max(1, int(update.get("total", 1)))
        completed = int(update.get("completed", 0))
        _emit(
            "progress",
            {
                "percent": round(completed / total * 100),
                **update,
            },
        )

    result = run_job(job, progress=progress)
    _atomic_json(result_path, result)
    artifacts = result.get("artifacts") or [result]
    for item in artifacts:
        _emit(
            "artifact",
            {
                "path": str(item["image_path"]),
                "sidecar_path": str(item["sidecar_path"]),
                "role": "source-map-image",
                "candidate_id": str(item.get("candidate_id") or ""),
            },
        )
    _emit("progress", {"percent": 100})
    _emit(
        "result",
        {
            "status": "succeeded",
            "result_path": str(result_path),
            "artifact_count": len(artifacts),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run native App 1.0 Source Map operations."
    )
    parser.add_argument(
        "--operation",
        choices=("discover", "render", "sequence"),
        required=True,
    )
    parser.add_argument("--request-file", required=True)
    parser.add_argument("--result-file", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result_path = Path(args.result_file).expanduser().resolve(strict=False)
    try:
        payload = _request(args.request_file)
        if args.operation == "discover":
            _discover(payload, result_path)
        else:
            _render(payload, result_path, sequence=args.operation == "sequence")
        return 0
    except Exception as exc:
        error = {
            "ok": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _atomic_json(result_path, error)
        _emit("log", {"message": str(exc), "level": "error"})
        _emit("result", {"status": "failed", "result_path": str(result_path)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
