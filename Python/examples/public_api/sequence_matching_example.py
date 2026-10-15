"""Build a complete, monotone hierarchy from synthetic observation frames."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from solar_toolkit.time import (
    SequenceMatchPlan,
    TimeFrame,
    TimeStream,
    build_sequence_match_plan,
    normalize_time,
)

REQUIRES_LOCAL_DATA = False


def build_example_plan() -> SequenceMatchPlan:
    """Keep every input frame, repeating ancestors in the final output order."""

    origin = normalize_time("2000-01-01T00:00:00Z")
    streams = [
        TimeStream(
            source_id,
            [
                TimeFrame(f"{source_id}:{index}", origin.shift(seconds))
                for index, seconds in enumerate(offsets)
            ],
            tolerance_seconds=3,
        )
        for source_id, offsets in (
            ("slow", (0, 10)),
            ("medium", (0, 6, 10)),
            ("fast", (0, 4, 5, 10)),
        )
    ]
    return build_sequence_match_plan(
        streams,
        time_start="2000-01-01T00:00:00Z",
        time_end="2000-01-01T00:01:00Z",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Print exact frame identities and output UTC without reading any files."""

    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    plan = build_example_plan()
    result = {
        "root_source_id": plan.master_source_id,
        "output_source_id": plan.output_source_id,
        "source_order": plan.source_order,
        "rows": [
            {
                "output_utc": row.time.utc,
                "frames": {
                    source: selection.frame.frame_id if selection.frame else None
                    for source, selection in row.selections.items()
                },
            }
            for row in plan.rows
        ],
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
