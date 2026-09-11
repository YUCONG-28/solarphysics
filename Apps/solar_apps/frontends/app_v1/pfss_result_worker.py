# SPDX-License-Identifier: GPL-3.0-only
"""Process-isolated bundle validation using the existing App event contract."""

import argparse
import os
import uuid

from .flow_builtin_worker import _event
from .pfss_adapter import PFSSAdapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--allowed-roots", required=True)
    args = parser.parse_args(argv)
    os.environ.setdefault("APP_V1_RUN_ID", uuid.uuid4().hex)
    module_id = os.environ.get("APP_V1_MODULE_ID", "pfss")
    _event(module_id, "progress", percent=5)
    roots = tuple(p for p in args.allowed_roots.split(os.pathsep) if p)
    path, metadata, lines = PFSSAdapter(roots).load(args.input_dir)
    _event(
        module_id,
        "artifact",
        path=str(path / "manifest.json"),
        source_port="manifest",
        artifact_type="manifest",
        ordinal=1,
    )
    _event(module_id, "progress", percent=100)
    _event(
        module_id,
        "result",
        status="succeeded",
        artifact_count=1,
        fieldline_count=len(lines),
        independent_source_height_count=metadata.get(
            "independent_source_height_count", 0
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
