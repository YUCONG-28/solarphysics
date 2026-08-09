from __future__ import annotations

import hashlib
import json
from pathlib import Path

from solar_apps.workflows.radio.source_map_workflow import _sorted_fits_for_band


def test_unrelated_early_file_does_not_change_frozen_selection(tmp_path: Path) -> None:
    band = tmp_path / "149MHz" / "RR"
    band.mkdir(parents=True)
    selected = band / "b.fits"
    selected.write_bytes(b"selected")
    manifest = {
        "schema": "solar-radio-frozen-collection-v1",
        "records": [{
            "record_id": "radio-fixed",
            "observed_utc": "2025-01-24T04:48:30Z",
            "relative_path": "149MHz/RR/b.fits",
            "bytes": selected.stat().st_size,
            "sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
        }],
    }
    (tmp_path / ".frozen-collection-v1.json").write_text(json.dumps(manifest))

    assert _sorted_fits_for_band(str(band), 0, 1) == [str(selected)]
    (band / "a.fits").write_bytes(b"unrelated")
    assert _sorted_fits_for_band(str(band), 0, 1) == [str(selected)]
