"""Finalize and probe presentation media without loading RGB frame lists."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .main import _write_manifest
from .visualization.animations import _transcode_delivery, require_mp4_backend

STEMS = ("tearing", "jet", "electron_beam", "typeIII")


def _probe(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required.")
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-count_packets",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_read_packets",
            "-of",
            "json",
            str(path),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(completed.stdout)["streams"][0]


def finalize(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    animations = run_dir / "animations"
    executable = require_mp4_backend()
    records: dict[str, Any] = {}
    delivery_encoders: dict[str, str] = {}
    for stem in STEMS:
        master = animations / f"{stem}_master_ffv1.mkv"
        delivery = animations / f"{stem}.mp4"
        if not master.is_file() or not delivery.is_file():
            raise FileNotFoundError(stem)
        delivery_encoders[stem] = _transcode_delivery(master, delivery)
        preview = animations / f"{stem}.gif"
        subprocess.run(
            [
                executable,
                "-y",
                "-i",
                str(master),
                "-vf",
                "fps=10,scale=960:540:flags=lanczos",
                "-loop",
                "0",
                str(preview),
            ],
            check=True,
        )
        records[stem] = {
            "master": _probe(master),
            "delivery": _probe(delivery),
            "preview": _probe(preview),
            "delivery_encoder": delivery_encoders[stem],
        }
    (animations / "media_encoding.json").write_text(
        json.dumps(
            {
                "master": "ffv1-level3",
                "fps": 30,
                "delivery_encoders": delivery_encoders,
                "nvenc_fallback_policy": "libx264-crf17",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = animations / "media_probe.json"
    report_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_manifest(run_dir, [report_path])
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(finalize(args.run_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
