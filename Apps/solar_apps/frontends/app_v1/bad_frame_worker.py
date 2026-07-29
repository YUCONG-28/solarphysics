# SPDX-License-Identifier: GPL-3.0-only
"""Supervised native actions for persistent radio bad-frame reviews."""

from __future__ import annotations

import argparse
import json
import shutil
import traceback
from pathlib import Path

_PREFIX = "APP_V1_EVENT "


def _emit(kind: str, payload: dict[str, object]) -> None:
    print(
        _PREFIX
        + json.dumps(
            {
                "schema_version": 1,
                "module_id": "bad-frame-review",
                "kind": kind,
                "payload": payload,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("create", "preview", "label", "finalize", "archive"),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--allowed-root", action="append", default=[])
    parser.add_argument("--input-root")
    parser.add_argument("--review-id")
    parser.add_argument("--target-kind", choices=("candidate", "frame"))
    parser.add_argument("--target-id")
    parser.add_argument("--frequencies", default="")
    parser.add_argument("--polarizations", default="RR,LL")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument(
        "--strategy",
        choices=("rules", "labeling", "shadow"),
        default="rules",
    )
    parser.add_argument(
        "--scope",
        choices=("candidates", "all_scanned"),
        default="candidates",
    )
    parser.add_argument("--sample-count", type=int, default=1200)
    parser.add_argument(
        "--quality",
        choices=("good", "degraded", "bad", "uncertain"),
    )
    parser.add_argument("--event-tags", default="")
    parser.add_argument("--artifact-tags", default="")
    parser.add_argument(
        "--final-status",
        choices=("completed", "skipped"),
        default="completed",
    )
    parser.add_argument("--cmap", default="coolwarm")
    parser.add_argument(
        "--transform",
        choices=("robust_asinh", "linear"),
        default="robust_asinh",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from solar_apps.frontends.radio_bad_frame_review.review import (
            BadFrameReviewStore,
        )

        output_root = Path(args.output_root).expanduser().resolve()
        allowed_roots = [
            Path(item).expanduser().resolve() for item in args.allowed_root
        ]
        if not allowed_roots and args.input_root:
            allowed_roots = [Path(args.input_root).expanduser().resolve()]
        if not allowed_roots:
            raise ValueError("At least one allowed root is required")
        store = BadFrameReviewStore(output_root, allowed_roots)
        if args.action == "create":
            review = store.create_review(
                {
                    "root": args.input_root,
                    "frequencies_mhz": _float_list(args.frequencies),
                    "polarizations": _str_list(args.polarizations),
                    "start_index": args.start_index,
                    "end_index": args.end_index,
                    "candidate_strategy": args.strategy,
                    "review_scope": args.scope,
                    "sample_count": args.sample_count,
                }
            )
            _publish_review(output_root, review)
        elif args.action == "preview":
            _require_review_target(args)
            display = {"cmap": args.cmap, "transform": args.transform}
            if args.target_kind == "candidate":
                content = store.render_candidate_preview(
                    args.review_id,
                    args.target_id,
                    display=display,
                )
            else:
                content = store.render_frame_preview(
                    args.review_id,
                    args.target_id,
                    display=display,
                )
                store.mark_frame_viewed(args.review_id, args.target_id)
            preview = (
                output_root
                / str(args.review_id)
                / f"preview-{args.target_kind}-{args.target_id}.png"
            )
            preview.write_bytes(content)
            _publish_review(output_root, store.load_review(str(args.review_id)))
            _emit("artifact", {"path": str(preview), "role": "review-preview"})
        elif args.action == "label":
            _require_review_target(args)
            if not args.quality:
                raise ValueError("--quality is required for label")
            label = {
                "quality_label": args.quality,
                "event_tags": _str_list(args.event_tags),
                "artifact_tags": _str_list(args.artifact_tags),
            }
            if args.target_kind == "candidate":
                review = store.update_labels(
                    str(args.review_id),
                    {str(args.target_id): label},
                )
            else:
                review = store.update_frame_label(
                    str(args.review_id),
                    str(args.target_id),
                    label,
                )
            _publish_review(output_root, review)
        elif args.action == "finalize":
            if not args.review_id:
                raise ValueError("--review-id is required")
            review = store.finalize(str(args.review_id), args.final_status)
            _publish_review(output_root, review)
        else:
            if not args.review_id:
                raise ValueError("--review-id is required")
            review_dir = output_root / str(args.review_id)
            store.load_review(str(args.review_id))
            archive = Path(
                shutil.make_archive(
                    str(review_dir / "bad-frame-review"),
                    "zip",
                    root_dir=review_dir,
                )
            )
            _emit("artifact", {"path": str(archive), "role": "review-archive"})
        _emit("progress", {"percent": 100})
        _emit("result", {"status": "succeeded"})
        return 0
    except Exception as exc:
        _emit("log", {"level": "error", "message": str(exc)})
        _emit("log", {"level": "debug", "message": traceback.format_exc()})
        _emit("result", {"status": "failed"})
        return 1


def _publish_review(output_root: Path, review: dict[str, object]) -> None:
    review_id = str(review["review_id"])
    review_dir = output_root / review_id
    for path, role in (
        (review_dir / "review.json", "review-manifest"),
        (review_dir / "candidates.csv", "review-table"),
        (review_dir / "viewed_frames.csv", "review-audit"),
    ):
        if path.is_file():
            _emit("artifact", {"path": str(path), "role": role})


def _require_review_target(args: argparse.Namespace) -> None:
    if not args.review_id or not args.target_kind or not args.target_id:
        raise ValueError("--review-id, --target-kind, and --target-id are required")


def _str_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _float_list(raw: str) -> list[float]:
    return [float(item) for item in _str_list(raw)]


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
