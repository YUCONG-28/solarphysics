# SPDX-License-Identifier: GPL-3.0-only
"""Read synchronized artifacts and export parameters; no remote execution."""

import json
from datetime import datetime, timezone
from pathlib import Path

from solar_apps.platform.paths import validate_allowed_path


class PFSSAdapter:
    def __init__(self, allowed_roots):
        self.allowed_roots = tuple(allowed_roots)

    def load(self, directory):
        from solar_toolkit.modeling.pfss.bundle import load_bundle

        path = validate_allowed_path(
            directory, allowed_roots=self.allowed_roots, kind="directory"
        )
        metadata, lines = load_bundle(path)
        manifest = json.loads((path / "manifest.json").read_text())
        members = {item["path"] for item in manifest["files"]}
        required = {
            "resolved_config.json",
            "sources.csv",
            "aia_display.fits",
            "newkirk_forward_ranking.csv",
        }
        if not required.issubset(members):
            raise ValueError(
                "Viewer requires a complete, checksum-listed display bundle"
            )
        return path, metadata, lines

    def export_config(self, destination, original, *, pfss, seed_hpc_box):
        from solar_toolkit.modeling.pfss import PFSSConfig
        from dataclasses import asdict

        validated = PFSSConfig(**pfss)
        if seed_hpc_box is not None:
            import numpy as np

            if (
                len(seed_hpc_box) != 4
                or not np.isfinite(seed_hpc_box).all()
                or seed_hpc_box[0] >= seed_hpc_box[1]
                or seed_hpc_box[2] >= seed_hpc_box[3]
            ):
                raise ValueError("Seed box requires finite xmin < xmax and ymin < ymax")
        target = validate_allowed_path(
            destination,
            allowed_roots=self.allowed_roots,
            kind="save_file",
            default_suffix=".json",
        )
        if target.exists():
            raise FileExistsError("Choose a new configuration filename")
        config = dict(original)
        config.update(pfss=asdict(validated), seed_hpc_box=seed_hpc_box)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        config["output_dir"] = str(Path(config["output_dir"]).parent / f"pfss_{stamp}")
        target.write_text(
            json.dumps(config, indent=2, allow_nan=False), encoding="utf-8"
        )
        return target
