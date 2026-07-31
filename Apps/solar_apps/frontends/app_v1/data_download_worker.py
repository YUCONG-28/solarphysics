# SPDX-License-Identifier: GPL-3.0-only
"""Process-isolated search and download worker for App 1.0 observations."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import threading
from collections.abc import Sequence
from pathlib import Path

from solar_toolkit.net.observations import (
    ObservationQueryV1,
    download_observations,
    read_remote_records,
    search_observations,
    write_search_result,
)

from .contracts import (
    ArtifactManifestV1,
    ArtifactProduct,
    InputReference,
    RunStatus,
)

_EVENT_PREFIX = "APP_V1_EVENT "


def _event(kind: str, payload: dict[str, object]) -> None:
    print(
        _EVENT_PREFIX
        + json.dumps(
            {"schema_version": 1, "kind": kind, "payload": payload},
            separators=(",", ":"),
        ),
        flush=True,
    )


def _values(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _integers(value: str | None) -> tuple[int, ...]:
    return tuple(int(item) for item in _values(value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search and download observations.")
    subparsers = parser.add_subparsers(dest="operation", required=True)

    search = subparsers.add_parser("search")
    search.add_argument("--product-id", required=True)
    search.add_argument("--start-utc", required=True)
    search.add_argument("--end-utc", required=True)
    search.add_argument("--spacecraft", default="")
    search.add_argument("--detectors", default="")
    search.add_argument("--wavelengths", default="")
    search.add_argument("--level")
    search.add_argument("--sample-seconds", type=int)
    search.add_argument("--output-dir", required=True)

    download = subparsers.add_parser("download")
    download.add_argument("--selection", required=True)
    download.add_argument("--observation-root", required=True)
    download.add_argument("--output-dir", required=True)
    download.add_argument("--max-workers", type=int, default=2)
    return parser


def _manifest(
    output: Path,
    *,
    products: tuple[ArtifactProduct, ...],
    parameters: dict[str, object],
    inputs: tuple[InputReference, ...] = (),
    status: RunStatus = RunStatus.SUCCEEDED,
) -> Path:
    run_id = os.getenv("APP_V1_RUN_ID", "data-download-run").lower()
    if not run_id.replace("-", "").isalnum():
        run_id = "data-download-run"
    manifest = ArtifactManifestV1(
        project_id="preview",
        run_id=run_id,
        module_id="data-download",
        status=status,
        inputs=inputs,
        parameters=parameters,
        products=products,
        software={"worker": "app-v1-data-download", "schema_version": 1},
    )
    path = output / "manifest.json"
    path.write_text(manifest.to_json(), encoding="utf-8")
    return path


def _search(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).expanduser().resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    query = ObservationQueryV1(
        query_id=f"query-{os.getenv('APP_V1_RUN_ID', 'preview')[:24].lower()}",
        product_id=args.product_id,
        start_utc=dt.datetime.fromisoformat(args.start_utc.replace("Z", "+00:00")),
        end_utc=dt.datetime.fromisoformat(args.end_utc.replace("Z", "+00:00")),
        spacecraft=_values(args.spacecraft),
        detectors=_values(args.detectors),
        wavelengths=_integers(args.wavelengths),
        level=args.level,
        sample_seconds=args.sample_seconds,
    )
    _event("log", {"message": f"Searching {query.product_id}"})
    _event("progress", {"percent": 5})
    records = search_observations(query)
    target = write_search_result(output / "search-results.json", query, records)
    _event("progress", {"percent": 100})
    _event(
        "artifact",
        {
            "path": str(target),
            "role": "remote-observation-set",
            "record_count": len(records),
        },
    )
    manifest = _manifest(
        output,
        products=(
            ArtifactProduct(
                "remote-observation-set",
                target.name,
                "application/json",
            ),
        ),
        parameters=query.to_dict(),
    )
    _event(
        "result",
        {
            "status": "succeeded",
            "manifest_path": str(manifest),
            "record_count": len(records),
        },
    )
    return 0


def _download(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).expanduser().resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    selection = Path(args.selection).expanduser().resolve(strict=True)
    records = read_remote_records(selection)
    if not records:
        raise ValueError("Select at least one observation before downloading")
    cancelled = threading.Event()

    def stop(_signum=None, _frame=None) -> None:
        cancelled.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def progress(completed, total, item) -> None:  # type: ignore[no-untyped-def]
        _event(
            "progress",
            {
                "percent": round(completed / total * 100),
                "completed": completed,
                "total": total,
                "record_id": item.record_id,
                "status": item.status,
            },
        )

    _event("log", {"message": f"Downloading {len(records)} observation(s)"})
    collection = download_observations(
        records,
        args.observation_root,
        collection_id=f"collection-{os.getenv('APP_V1_RUN_ID', 'preview')[:24].lower()}",
        max_workers=args.max_workers,
        cancelled=cancelled.is_set,
        progress=progress,
    )
    receipt = output / "download-receipt.json"
    receipt.write_text(
        json.dumps(collection.to_dict(), indent=2),
        encoding="utf-8",
    )
    failures = [item for item in collection.items if item.status == "failed"]
    cancelled_items = [item for item in collection.items if item.status == "cancelled"]
    status = (
        RunStatus.FAILED
        if failures
        else RunStatus.CANCELLED if cancelled_items else RunStatus.SUCCEEDED
    )
    manifest = _manifest(
        output,
        products=(
            ArtifactProduct(
                "observation-collection",
                receipt.name,
                "application/json",
            ),
        ),
        parameters={
            "observation_root": str(
                Path(args.observation_root).expanduser().resolve(strict=False)
            ),
            "max_workers": args.max_workers,
            "record_count": len(records),
        },
        inputs=(
            InputReference(
                "search-selection",
                "remote-observation-set",
                str(selection),
            ),
        ),
        status=status,
    )
    _event("artifact", {"path": str(receipt), "role": "observation-collection"})
    _event(
        "result",
        {
            "status": status.value,
            "manifest_path": str(manifest),
            "failed": len(failures),
            "cancelled": len(cancelled_items),
        },
    )
    return 1 if failures else 130 if cancelled_items else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _search(args) if args.operation == "search" else _download(args)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
