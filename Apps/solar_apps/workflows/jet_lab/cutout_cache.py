"""Native crop cache identity and evidence verification."""

import hashlib
import json
from pathlib import Path

from solar_toolkit.map.euvi_preprocessing import VERSION
from solar_toolkit.map.jet_annotations import file_sha256


def identity(smap, source_hash, bounds):
    record = dict(
        version=2,
        source_sha256=source_hash,
        roi_arcsec=list(map(float, bounds)),
        wcs=smap.wcs.to_header().tostring(),
        preprocessing_version=VERSION,
        parameters={"native_sampling": True, "normalization": "declared_units_only"},
    )
    record["key"] = hashlib.sha256(
        json.dumps(record, sort_keys=True).encode()
    ).hexdigest()
    return record


def verify(path, expected=None):
    path = Path(path)
    record = json.loads(Path(str(path) + ".cutout.json").read_text())
    if record["output_sha256"] != file_sha256(path):
        raise ValueError("Changed cached pixels")
    if expected is not None and record["identity"] != expected:
        raise ValueError("Different ROI, WCS or processing")
    for suffix, digest in record["sidecars"].items():
        if file_sha256(Path(str(path) + suffix)) != digest:
            raise ValueError("Missing or changed preprocessing evidence")
    return record


def record(path, expected, sidecars):
    path = Path(path)
    payload = dict(
        identity=expected,
        output_sha256=file_sha256(path),
        sidecars={s: file_sha256(Path(str(path) + s)) for s in sidecars},
    )
    Path(str(path) + ".cutout.json").write_text(json.dumps(payload, indent=2))
