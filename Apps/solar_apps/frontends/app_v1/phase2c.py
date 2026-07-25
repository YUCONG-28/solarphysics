# SPDX-License-Identifier: GPL-3.0-only
"""Import-safe Phase 2C adapters for radio diagnostics and DEM workflows."""

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

_DART = "solar_apps.frontends.radio.dart_spectrogram.dart_spectrogram_launcher"
_DRIFT = "solar_apps.workflows.radio.drift_selection_cli"
_NEWKIRK = "solar_apps.workflows.radio.physical_diagnostics_cli"
_TRAJECTORY_APP = "solar_apps.frontends.radio.source_trajectory.source_app_launcher"
_TRAJECTORY_EXPORT = "solar_apps.workflows.radio.trajectory_cli"
_DEM_RADIO = "solar_apps.workflows.xray_dem.dem_radio_cli"
_RADIO_CONFIG = "solar_apps.workflows.radio.configs.radio_20250124_config"


class Phase2CAdapter:
    """Prepare confirmed work without importing a Qt or scientific stack."""

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

    def build_dart_spectrogram(self, input_dir: str | Path) -> TaskLaunch:
        selected = self.validate_directory(input_dir)
        output = self._new_output_dir("dart-spectrogram")
        port = self._free_port()
        return TaskLaunch(
            "DART Spectrogram",
            "dart-spectrogram",
            _DART,
            (
                "--input-dir",
                str(selected),
                "--output-dir",
                str(output),
                "--allowed-roots",
                self._root_list(output),
                "--port",
                str(port),
                "--browser",
            ),
            output,
            "\n".join(
                (
                    "Module: DART Spectrogram",
                    f"Input: {selected}",
                    "Parameters: retained interactive DART workflow",
                    f"Output: {output}",
                    f"Workload: {self._file_count(selected)} input file(s)",
                    f"Endpoint: http://127.0.0.1:{port}",
                )
            ),
        )

    def build_drift_rate(
        self,
        *,
        t_start: str,
        f_start_mhz: float,
        t_end: str,
        f_end_mhz: float,
    ) -> TaskLaunch:
        if not str(t_start).strip() or not str(t_end).strip():
            raise ValueError("Both drift endpoint times are required")
        line = {
            "label": "drift_001",
            "mode": "manual",
            "t_start": str(t_start).strip(),
            "f_start_mhz": float(f_start_mhz),
            "t_end": str(t_end).strip(),
            "f_end_mhz": float(f_end_mhz),
            "color": "white",
            "note": "App 1.0 Phase 2C confirmed selection",
        }
        output = self._new_output_dir("drift-rate")
        return TaskLaunch(
            "Drift Rate",
            "dart-spectrogram",
            _DRIFT,
            (
                "--drift-lines-json",
                json.dumps([line], separators=(",", ":")),
                "--output-dir",
                str(output),
            ),
            output,
            "\n".join(
                (
                    "Module: Drift Rate",
                    f"Input: endpoints {line['t_start']} / {line['t_end']}",
                    (
                        "Parameters: "
                        f"{line['f_start_mhz']:g} -> {line['f_end_mhz']:g} MHz"
                    ),
                    f"Output: {output}",
                    "Workload: one explicitly selected two-point drift line",
                )
            ),
        )

    def build_newkirk_diagnostics(
        self,
        *,
        gaussian_csv: str | Path | None = None,
        drift_csv: str | Path | None = None,
    ) -> TaskLaunch:
        gaussian = self._optional_file(gaussian_csv)
        drift = self._optional_file(drift_csv)
        if gaussian is None and drift is None:
            raise ValueError("Select a Gaussian CSV, a drift CSV, or both")
        output = self._new_output_dir("newkirk")
        arguments: list[str] = [
            "--config",
            _RADIO_CONFIG,
            "--output-dir",
            str(output),
        ]
        if gaussian is not None:
            arguments.extend(["--gaussian-csv", str(gaussian)])
        if drift is not None:
            arguments.extend(["--drift-csv", str(drift)])
        inputs = "; ".join(str(path) for path in (gaussian, drift) if path is not None)
        return TaskLaunch(
            "Newkirk Diagnostics",
            "dart-spectrogram",
            _NEWKIRK,
            tuple(arguments),
            output,
            "\n".join(
                (
                    "Module: Newkirk Diagnostics",
                    f"Input: {inputs}",
                    "Parameters: configured 2025-01-24 Newkirk assumptions",
                    f"Output: {output}",
                    "Workload: persisted diagnostic table(s); no upstream rerun",
                )
            ),
        )

    def build_source_trajectory(
        self,
        centers: str | Path,
        *,
        aia_dir: str | Path | None = None,
    ) -> TaskLaunch:
        center_path = self.validate_file(centers)
        aia = self._optional_directory(aia_dir)
        output = self._new_output_dir("source-trajectory")
        port = self._free_port()
        arguments: list[str] = [
            "--centers",
            str(center_path),
            "--allowed-roots",
            self._root_list(output),
            "--port",
            str(port),
            "--browser",
        ]
        if aia is not None:
            arguments.extend(["--aia-dir", str(aia)])
        return TaskLaunch(
            "Source Trajectory",
            "source-trajectory",
            _TRAJECTORY_APP,
            tuple(arguments),
            output,
            "\n".join(
                (
                    "Module: Source Trajectory",
                    f"Input: centers={center_path}; AIA={aia or 'none'}",
                    "Parameters: retained interactive trajectory workflow",
                    f"Output: {output}",
                    f"Workload: {self._table_rows(center_path)} table row(s)",
                    f"Endpoint: http://127.0.0.1:{port}",
                )
            ),
        )

    def build_trajectory_export(
        self,
        centers: str | Path,
        *,
        aia_dir: str | Path | None = None,
        tail_n: int = 5,
    ) -> TaskLaunch:
        center_path = self.validate_file(centers)
        aia = self._optional_directory(aia_dir)
        if tail_n <= 0:
            raise ValueError("Trajectory tail length must be positive")
        output = self._new_output_dir("source-trajectory")
        html = output / "radio_source_trajectory.html"
        arguments: list[str] = [
            "--centers",
            str(center_path),
            "--out",
            str(html),
            "--mode",
            "tail",
            "--tail-n",
            str(int(tail_n)),
        ]
        if aia is not None:
            arguments.extend(["--aia-dir", str(aia)])
        return TaskLaunch(
            "Trajectory HTML Export",
            "source-trajectory",
            _TRAJECTORY_EXPORT,
            tuple(arguments),
            output,
            "\n".join(
                (
                    "Module: Source Trajectory Export",
                    f"Input: centers={center_path}; AIA={aia or 'none'}",
                    f"Parameters: tail mode; tail_n={int(tail_n)}",
                    f"Output: {html}",
                    f"Workload: {self._table_rows(center_path)} table row(s)",
                )
            ),
        )

    def build_dem_radio_overlay(
        self,
        *,
        aia_fits: str | Path,
        tb_data: str | Path,
        radio_file: str | Path,
    ) -> TaskLaunch:
        aia = self.validate_file(aia_fits)
        tb = self.validate_file(tb_data)
        radio = self.validate_file(radio_file)
        output = self._new_output_dir("dem-radio")
        return TaskLaunch(
            "DEM Radio Overlay",
            "source-trajectory",
            _DEM_RADIO,
            (
                "--aia-fits",
                str(aia),
                "--tb-data",
                str(tb),
                "--radio-file",
                str(radio),
                "--output-dir",
                str(output),
            ),
            output,
            "\n".join(
                (
                    "Module: DEM Radio Overlay",
                    f"Input: AIA={aia}; Tb={tb}; radio={radio}",
                    "Parameters: existing DEM overlay defaults; CPU workflow",
                    f"Output: {output}",
                    "Workload: one AIA/Tb/radio triplet",
                )
            ),
        )

    def validate_directory(self, value: str | Path) -> Path:
        return self._validate_path(value, directory=True)

    def validate_file(self, value: str | Path) -> Path:
        return self._validate_path(value, directory=False)

    def _validate_path(self, value: str | Path, *, directory: bool) -> Path:
        if not self.allowed_roots:
            raise PermissionError(
                "No allowed roots are configured for App 1.0 data access."
            )
        path = Path(value).expanduser().resolve(strict=False)
        exists = path.is_dir() if directory else path.is_file()
        if not exists:
            kind = "directory" if directory else "file"
            raise FileNotFoundError(f"Input {kind} does not exist: {path}")
        if not any(path == root or root in path.parents for root in self.allowed_roots):
            raise PermissionError(
                f"Input is outside the configured allowed roots: {path}"
            )
        return path

    def _optional_file(self, value: str | Path | None) -> Path | None:
        return None if value in (None, "") else self.validate_file(value)

    def _optional_directory(self, value: str | Path | None) -> Path | None:
        return None if value in (None, "") else self.validate_directory(value)

    def _new_output_dir(self, module_id: str) -> Path:
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        return self.runtime.run_output_dir("preview", run_id, module_id)

    def _root_list(self, output: Path) -> str:
        return os.pathsep.join(map(str, (*self.allowed_roots, output)))

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            return int(probe.getsockname()[1])

    @staticmethod
    def _file_count(directory: Path) -> int:
        return sum(1 for item in directory.iterdir() if item.is_file())

    @staticmethod
    def _table_rows(path: Path) -> int:
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            return max(sum(1 for _line in handle) - 1, 0)


__all__ = ["Phase2CAdapter"]
