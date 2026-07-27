# SPDX-License-Identifier: GPL-3.0-only
"""Direct-argument adapter for the native Source Map worker protocol."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from .source_map_worker import main as source_map_worker_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-path", required=True)
    parser.add_argument(
        "--mode", choices=("single_band", "multi_band"), default="single_band"
    )
    parser.add_argument("--frequencies", default="149,164,190")
    parser.add_argument("--polarization", default="RR+LL")
    parser.add_argument("--cmap", default="hot")
    parser.add_argument(
        "--color-range-mode",
        choices=("auto", "fixed", "global"),
        default="auto",
    )
    parser.add_argument("--fixed-vmin", type=float)
    parser.add_argument("--fixed-vmax", type=float)
    parser.add_argument("--gaussian-overlay", action="store_true")
    parser.add_argument("--spectrogram-panel", action="store_true")
    parser.add_argument(
        "--background-mode",
        choices=("off", "noise_map_only", "local_mesh", "local_median"),
        default="off",
    )
    parser.add_argument("--background-display", action="store_true")
    parser.add_argument("--background-fit", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allowed-roots", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output_dir).expanduser().resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.source_path).expanduser().resolve(strict=True)
    allowed_roots = [
        item for item in str(args.allowed_roots).split(os.pathsep) if item.strip()
    ]
    if not allowed_roots:
        raise PermissionError("No allowed roots were supplied")
    request = {
        "allowed_roots": allowed_roots,
        "config": {
            "config_module": args.config,
            "source_path": str(source),
            "output_dir": str(output),
            "mode": args.mode,
            "frequencies": [
                float(value)
                for value in str(args.frequencies).split(",")
                if value.strip()
            ],
            "polarization": args.polarization,
            "display": {
                "cmap": args.cmap,
                "color_range_mode": args.color_range_mode,
                "fixed_vmin": args.fixed_vmin,
                "fixed_vmax": args.fixed_vmax,
            },
            "features": {
                "gaussian_overlay": args.gaussian_overlay,
                "spectrogram_panel": args.spectrogram_panel,
            },
            "background": {
                "mode": args.background_mode,
                "apply_to_display": args.background_display,
                "apply_to_fit": args.background_fit,
            },
        },
    }
    request_file = output / "flow-source-map-request.json"
    discovery_file = output / "flow-source-map-discovery.json"
    request_file.write_text(
        json.dumps(request, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    result = source_map_worker_main(
        [
            "--operation",
            "discover",
            "--request-file",
            str(request_file),
            "--result-file",
            str(discovery_file),
        ]
    )
    if result != 0:
        return result
    render_request = output / "flow-source-map-render-request.json"
    render_request.write_text(
        json.dumps(
            {
                "discovery_file": str(discovery_file),
                "candidate_index": 0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with tempfile.NamedTemporaryFile(
        prefix="source-map-result-",
        suffix=".json",
        dir=output,
        delete=False,
    ) as handle:
        result_path = Path(handle.name)
    return source_map_worker_main(
        [
            "--operation",
            "render",
            "--request-file",
            str(render_request),
            "--result-file",
            str(result_path),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
