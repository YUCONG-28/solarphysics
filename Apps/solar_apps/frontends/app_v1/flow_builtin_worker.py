# SPDX-License-Identifier: GPL-3.0-only
"""Small process-isolated data functions used by App 1.0 workflows."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path


def _event(module_id: str, kind: str, **payload: object) -> None:
    print(
        "APP_V1_EVENT "
        + json.dumps(
            {
                "schema_version": 1,
                "run_id": os.environ.get("APP_V1_RUN_ID") or uuid.uuid4().hex,
                "module_id": module_id,
                "kind": kind,
                "payload": payload,
            },
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operation",
        required=True,
        choices=("artifact-input", "image-discover"),
    )
    parser.add_argument("--path")
    parser.add_argument("--artifact-type", default="image")
    parser.add_argument("--input-dir")
    parser.add_argument("--extensions", default=".png,.jpg,.jpeg,.gif,.bmp,.tif,.tiff")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allowed-roots", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    module_id = os.environ.get("APP_V1_MODULE_ID", "workbench")
    roots = tuple(
        Path(item).expanduser().resolve(strict=False)
        for item in str(args.allowed_roots).split(os.pathsep)
        if item.strip()
    )
    if not roots:
        raise PermissionError("No allowed roots were supplied")

    def allowed(path: Path) -> Path:
        resolved = path.expanduser().resolve(strict=True)
        if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
            raise PermissionError(f"Path is outside allowed roots: {resolved}")
        return resolved

    _event(module_id, "progress", percent=5)
    if args.operation == "artifact-input":
        if not args.path:
            raise ValueError("--path is required for artifact-input")
        path = allowed(Path(args.path))
        if not path.is_file():
            raise ValueError("Existing artifact must be a file")
        paths = (path,)
        artifact_type = args.artifact_type
    else:
        if not args.input_dir:
            raise ValueError("--input-dir is required for image-discover")
        root = allowed(Path(args.input_dir))
        extensions = {
            item.strip().casefold()
            for item in str(args.extensions).split(",")
            if item.strip()
        }
        iterator = root.rglob("*") if args.recursive else root.glob("*")
        paths = tuple(
            path
            for path in sorted(iterator)
            if path.is_file() and path.suffix.casefold() in extensions
        )
        artifact_type = "image"
    for index, path in enumerate(paths, start=1):
        _event(
            module_id,
            "artifact",
            path=str(path),
            artifact_type=artifact_type,
            ordinal=index,
        )
    _event(module_id, "progress", percent=100)
    _event(
        module_id,
        "result",
        status="succeeded",
        artifact_count=len(paths),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
