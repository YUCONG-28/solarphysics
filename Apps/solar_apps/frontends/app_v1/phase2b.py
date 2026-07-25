# SPDX-License-Identifier: GPL-3.0-only
"""Import-safe Phase 2B adapters for existing radio frontends and workflows."""

from __future__ import annotations

import json
import os
import socket
import uuid
from pathlib import Path

from solar_apps.platform.layout import RuntimeLayout
from solar_apps.platform.paths.allowed_roots import configured_allowed_roots

from .phase2a import TaskLaunch
from .runtime import AppV1RuntimePaths

_BAD_FRAME = "solar_apps.frontends.radio_bad_frame_review.cli"
_SOURCE_MAP_APP = "solar_apps.frontends.radio.source_map.cli"
_GAUSSIAN = "solar_apps.workflows.radio.source_map_cli"
_ROI = "solar_apps.frontends.radio.roi_lightcurve.roi_lightcurve_launcher"
_COMPOSITE = "solar_apps.frontends.radio.composite_figure.composite_figure_launcher"
_FITS_SUFFIXES = {".fits", ".fit", ".fts"}


class Phase2BAdapter:
    """Prepare confirmed radio work while retaining all scientific owners."""

    def __init__(
        self,
        layout: RuntimeLayout,
        *,
        allowed_roots: tuple[Path, ...] | None = None,
    ) -> None:
        self.layout = layout
        self.runtime = AppV1RuntimePaths.from_layout(layout)
        self.allowed_roots = (
            tuple(
                Path(item).expanduser().resolve(strict=False) for item in allowed_roots
            )
            if allowed_roots is not None
            else configured_allowed_roots(
                environ=os.environ,
                workspace_root=layout.repo_root,
            )
        )

    def build_bad_frame_review(self, input_root: str | Path) -> TaskLaunch:
        selected = self.validate_input_directory(input_root)
        output = self._new_output_dir("bad-frame-review")
        port = self._free_port()
        arguments = (
            "--allowed-roots",
            self._root_list(output),
            "--output-root",
            str(output),
            "--port",
            str(port),
            "--open-browser",
        )
        return self._interactive_launch(
            "Bad Frame Review",
            "bad-frame-review",
            _BAD_FRAME,
            arguments,
            output,
            selected,
            port,
        )

    def build_source_map_app(self, input_root: str | Path) -> TaskLaunch:
        selected = self.validate_input_directory(input_root)
        output = self._new_output_dir("source-map")
        port = self._free_port()
        arguments = (
            "--allowed-roots",
            self._root_list(output),
            "--port",
            str(port),
            "--open-browser",
        )
        return self._interactive_launch(
            "Source Map",
            "source-map",
            _SOURCE_MAP_APP,
            arguments,
            output,
            selected,
            port,
        )

    def build_gaussian_fit(
        self,
        radio_dir: str | Path,
        *,
        source_count: int = 1,
    ) -> TaskLaunch:
        selected = self.validate_input_directory(radio_dir)
        if source_count not in {1, 2, 3}:
            raise ValueError("Gaussian source count must be 1, 2, or 3")
        files = self._fits_files(selected)
        if not files:
            raise FileNotFoundError(f"No radio FITS files found below {selected}")
        output = self._new_output_dir("gaussian-fit")
        workspace = {
            "mode": "single_band",
            "data": {
                "single_file_path": str(files[0]),
                "data_dir": str(files[0].parent),
                "start_idx": 0,
                "end_idx": 1,
                "polarization": "RR",
                "combine_polarizations": False,
            },
            "features": {
                "gaussian_overlay": True,
                "spectrogram_panel": False,
                "save_gaussian_diagnostics": True,
            },
            "gaussian": {
                "gaussian_source_mode": "single" if source_count == 1 else "multi",
                "multi_gaussian_source_count": source_count,
                "multi_gaussian_max_sources": source_count,
            },
            "output": {
                "output_dir": str(output),
                "show_plot": False,
                "save_plot": True,
            },
        }
        arguments = (
            "--config",
            "solar_apps.workflows.radio.configs.radio_20250124_config",
            "--output-dir",
            str(output),
            "--workspace-config-json",
            json.dumps(workspace, separators=(",", ":")),
        )
        summary = "\n".join(
            (
                "Module: Gaussian Source Fit",
                f"Input: {files[0]}",
                f"Parameters: source_count={source_count}; polarization=RR",
                f"Output: {output}",
                "Workload: one real radio FITS frame",
            )
        )
        return TaskLaunch(
            "Gaussian source fit",
            "source-map",
            _GAUSSIAN,
            arguments,
            output,
            summary,
        )

    def build_roi_lightcurve(
        self,
        radio_dir: str | Path,
        *,
        polarization: str = "L+R",
    ) -> TaskLaunch:
        selected = self.validate_input_directory(radio_dir)
        if polarization not in {"L+R", "LCP", "RCP", "all"}:
            raise ValueError("Unsupported ROI polarization")
        output = self._new_output_dir("roi-lightcurve")
        port = self._free_port()
        arguments = (
            "--radio-dir",
            str(selected),
            "--output-dir",
            str(output),
            "--allowed-roots",
            self._root_list(output),
            "--polarization",
            polarization,
            "--port",
            str(port),
            "--browser",
        )
        return self._interactive_launch(
            "ROI Light Curve",
            "roi-lightcurve",
            _ROI,
            arguments,
            output,
            selected,
            port,
        )

    def build_radio_composite(
        self,
        radio_dir: str | Path,
        dart_dir: str | Path,
    ) -> TaskLaunch:
        radio = self.validate_input_directory(radio_dir)
        dart = self.validate_input_directory(dart_dir)
        output = self._new_output_dir("radio-composite")
        port = self._free_port()
        arguments = (
            "--radio-dir",
            str(radio),
            "--dart-dir",
            str(dart),
            "--output-dir",
            str(output),
            "--allowed-roots",
            self._root_list(output),
            "--port",
            str(port),
            "--browser",
        )
        workload = len(self._fits_files(radio)) + len(self._fits_files(dart))
        summary = "\n".join(
            (
                "Module: Radio Composite",
                f"Input: radio={radio}; DART={dart}",
                "Parameters: managed Streamlit workflow",
                f"Output: {output}",
                f"Workload: {workload} FITS candidate(s)",
                f"Endpoint: http://127.0.0.1:{port}",
            )
        )
        return TaskLaunch(
            "Radio Composite",
            "radio-composite",
            _COMPOSITE,
            arguments,
            output,
            summary,
        )

    def validate_input_directory(self, value: str | Path) -> Path:
        if not self.allowed_roots:
            raise PermissionError(
                "No allowed roots are configured for App 1.0 data access."
            )
        path = Path(value).expanduser().resolve(strict=False)
        if not path.is_dir():
            raise NotADirectoryError(f"Input directory does not exist: {path}")
        if not any(path == root or root in path.parents for root in self.allowed_roots):
            raise PermissionError(
                f"Input is outside the configured allowed roots: {path}"
            )
        return path

    def _interactive_launch(
        self,
        title: str,
        module_id: str,
        python_module: str,
        arguments: tuple[str, ...],
        output: Path,
        selected: Path,
        port: int,
    ) -> TaskLaunch:
        workload = len(self._fits_files(selected))
        summary = "\n".join(
            (
                f"Module: {title}",
                f"Input: {selected}",
                "Parameters: existing interactive frontend in a separate process",
                f"Output: {output}",
                f"Workload: {workload} FITS candidate(s)",
                f"Endpoint: http://127.0.0.1:{port}",
            )
        )
        return TaskLaunch(
            title,
            module_id,
            python_module,
            arguments,
            output,
            summary,
        )

    def _new_output_dir(self, module_id: str) -> Path:
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        return self.runtime.run_output_dir("preview", run_id, module_id)

    def _root_list(self, output: Path) -> str:
        roots = [*self.allowed_roots, output]
        return os.pathsep.join(map(str, roots))

    @staticmethod
    def _fits_files(root: Path) -> list[Path]:
        return sorted(
            (
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.casefold() in _FITS_SUFFIXES
            ),
            key=lambda item: item.name.casefold(),
        )

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            return int(probe.getsockname()[1])


__all__ = ["Phase2BAdapter"]
