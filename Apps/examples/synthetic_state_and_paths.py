"""Demonstrate restart-safe state and recent-path memory with synthetic paths."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from solar_apps.platform.layout import RuntimeLayout, validate_private_output_path
from solar_apps.platform.paths.memory import PathMemoryContext, RecentPathMemory
from solar_apps.platform.state import StateStore


def _safe_output_dir(
    requested: str | Path | None,
    *,
    layout: RuntimeLayout,
) -> Path:
    if requested is None:
        candidate = layout.outputs_dir / "examples" / "state-and-path-memory"
    else:
        candidate = Path(requested).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
    return validate_private_output_path(candidate, repo_root=layout.repo_root)


def run_demo(
    *,
    output_dir: str | Path | None = None,
    layout: RuntimeLayout | None = None,
) -> dict[str, Any]:
    """Persist latest synthetic values and prove they survive new readers."""

    selected_layout = layout or RuntimeLayout.discover()
    destination = _safe_output_dir(output_dir, layout=selected_layout)
    synthetic_input, state_path, recent_path, summary_path = (
        validate_private_output_path(
            destination / name, repo_root=selected_layout.repo_root
        )
        for name in (
            "synthetic-input",
            "ui_state.json",
            "recent_paths.json",
            "summary.json",
        )
    )
    selected_layout.ensure()
    synthetic_input.mkdir(parents=True, exist_ok=True)

    state_store = StateStore(
        state_path,
        "synthetic_example",
        allowed_keys=("theme", "input_directory"),
    )
    state_store.save(
        {
            "theme": "auto",
            "input_directory": str(synthetic_input),
        }
    )

    path_store = StateStore(
        recent_path,
        "recent_paths",
        allowed_keys=("field", "operation", "frontend", "global"),
    )
    context = PathMemoryContext(
        frontend="synthetic-example",
        operation="select-input",
        field="input-directory",
    )
    RecentPathMemory(path_store, (destination,)).remember(
        context=context,
        dialog_mode="select_directory",
        paths=(synthetic_input,),
    )

    # Fresh instances represent the next application process after a restart.
    restored_state = StateStore(
        state_path,
        "synthetic_example",
        allowed_keys=("theme", "input_directory"),
    ).load()
    restored_directory = RecentPathMemory(
        StateStore(
            recent_path,
            "recent_paths",
            allowed_keys=("field", "operation", "frontend", "global"),
        ),
        (destination,),
    ).resolve_initial(
        context=context,
        dialog_mode="select_directory",
        current_value="",
    )
    if restored_directory != str(synthetic_input):
        raise RuntimeError("RecentPathMemory did not restore the selected directory")

    summary = {
        "synthetic": True,
        "restored_state": restored_state,
        "restored_directory": restored_directory,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "state": state_path,
        "recent_paths": recent_path,
        "summary": summary_path,
        "restored_state": restored_state,
        "restored_directory": restored_directory,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exercise local UI state and recent-path recovery synthetically."
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Destination directory. Defaults to "
            "Local/outputs/examples/state-and-path-memory/."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the example without performing work at import time."""

    arguments = build_parser().parse_args(argv)
    artifacts = run_demo(output_dir=arguments.output_dir)
    print(f"state: {artifacts['state']}")
    print(f"recent_paths: {artifacts['recent_paths']}")
    print(f"summary: {artifacts['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
