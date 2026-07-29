# SPDX-License-Identifier: GPL-3.0-only
"""Render native Source Trajectory playback frames in a supervised worker."""

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
                "module_id": "source-trajectory",
                "kind": kind,
                "payload": payload,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centers", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--frame-mode",
        choices=("current", "tail", "all"),
        default="tail",
    )
    parser.add_argument("--tail-n", type=int, default=5)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--theme", choices=("light", "dark"), default="light")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--aia-dir")
    parser.add_argument("--aia-pattern", default="*.fits")
    parser.add_argument("--max-aia-dt-sec", type=float, default=3600.0)
    parser.add_argument("--aia-max-pixels", type=int, default=384)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from solar_toolkit.aia.background import (
            load_nearest_background,
            scan_aia_folder,
        )
        from solar_toolkit.radio.trajectory import (
            frame_times,
            load_centers_table,
            select_visible_centers,
        )
        from solar_toolkit.visualization.radio_source_overlay import (
            render_radio_source_overlay_png,
        )

        output = Path(args.output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        centers = load_centers_table(args.centers)
        times = frame_times(centers)
        if not times:
            raise ValueError("No valid playback times in center table")
        times = times[: max(1, int(args.max_frames))]
        aia_table = (
            scan_aia_folder(args.aia_dir, pattern=args.aia_pattern)
            if args.aia_dir
            else None
        )
        manifest_frames: list[dict[str, str]] = []
        for index, frame_time in enumerate(times):
            visible = select_visible_centers(
                centers,
                frame_time,
                mode=args.frame_mode,
                tail_n=max(1, int(args.tail_n)),
            )
            background = None
            if aia_table is not None and not aia_table.empty:
                background, _nearest = load_nearest_background(
                    aia_table,
                    frame_time,
                    max_dt_seconds=float(args.max_aia_dt_sec),
                    max_pixels=max(1, int(args.aia_max_pixels)),
                )
            path = output / f"trajectory-frame-{index + 1:06d}.png"
            render_radio_source_overlay_png(
                visible,
                path,
                frame_time=frame_time,
                aia_background=background,
                width=max(320, int(args.width)),
                height=max(240, int(args.height)),
                theme_mode=args.theme,
                title_prefix="Radio source trajectory",
            )
            rendered_time = str(frame_time)
            manifest_frames.append({"path": str(path), "time": rendered_time})
            _emit(
                "artifact",
                {
                    "path": str(path),
                    "role": "trajectory-frame",
                    "frame_index": index,
                    "time": rendered_time,
                },
            )
            _emit(
                "progress",
                {"percent": round(100 * (index + 1) / len(times))},
            )
        manifest = output / "trajectory-playback.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "frame_mode": args.frame_mode,
                    "tail_n": max(1, int(args.tail_n)),
                    "frames": manifest_frames,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        _emit("artifact", {"path": str(manifest), "role": "trajectory-playback"})
        _emit(
            "result",
            {
                "status": "succeeded",
                "manifest_path": str(manifest),
                "frame_count": len(manifest_frames),
            },
        )
        return 0
    except Exception as exc:
        _emit("log", {"level": "error", "message": str(exc)})
        _emit("log", {"level": "debug", "message": traceback.format_exc()})
        _emit("result", {"status": "failed"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
