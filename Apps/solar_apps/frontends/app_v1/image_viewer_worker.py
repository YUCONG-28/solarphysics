# SPDX-License-Identifier: GPL-3.0-only
"""Process-isolated multi-folder Image Viewer media export."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

_PREFIX = "APP_V1_EVENT "


def _emit(kind: str, payload: dict[str, object]) -> None:
    print(
        _PREFIX
        + json.dumps(
            {
                "schema_version": 1,
                "module_id": "image-viewer",
                "kind": kind,
                "payload": payload,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--format", choices=("mp4", "gif", "webm"), default="mp4")
    parser.add_argument(
        "--mode", choices=("composite", "separate"), default="composite"
    )
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--recursive", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from solar_apps.frontends.image_viewer.export import (
            ExportConfig,
            export_composite_video,
            export_separate_videos,
        )
        from solar_apps.frontends.image_viewer.server import scan_images

        groups = []
        for folder in args.folder:
            root, images = scan_images(
                folder,
                recursive=args.recursive,
                allowed_roots=[Path(folder).resolve()],
            )
            groups.append(
                {
                    "name": root.name,
                    "folder": str(root),
                    "files": [str(path) for path in images],
                }
            )
        config = ExportConfig(
            output_dir=args.output_dir,
            output_format=args.format,
            fps=args.fps,
            workers=args.workers,
        )
        result = (
            export_composite_video(groups, config)
            if args.mode == "composite"
            else export_separate_videos(groups, config)
        )
        paths = result.get("paths") or [result.get("path")]
        artifacts = [str(path) for path in paths if path]
        if not artifacts:
            raise RuntimeError(str(result.get("reason") or result.get("failures")))
        for path in artifacts:
            _emit("artifact", {"path": path, "role": "image-viewer-media"})
        _emit("progress", {"percent": 100})
        _emit("result", {"status": "succeeded", "artifact_count": len(artifacts)})
        return 0
    except Exception as exc:
        _emit("log", {"level": "error", "message": str(exc)})
        _emit("log", {"level": "debug", "message": traceback.format_exc()})
        _emit("result", {"status": "failed"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
