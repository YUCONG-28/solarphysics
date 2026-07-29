# SPDX-License-Identifier: GPL-3.0-only
"""Import-safe Phase 2C adapters for radio diagnostics and DEM workflows."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from solar_apps.platform.layout import RuntimeLayout
from solar_apps.platform.paths.allowed_roots import configured_allowed_roots

from .phase2a import TaskLaunch
from .runtime import AppV1RuntimePaths

_NATIVE_WORKER = "solar_apps.frontends.app_v1.native_science_worker"
_DRIFT = "solar_apps.workflows.radio.drift_selection_cli"
_NEWKIRK = "solar_apps.workflows.radio.physical_diagnostics_cli"
_TRAJECTORY_EXPORT = "solar_apps.workflows.radio.trajectory_cli"
_TRAJECTORY_MEDIA = "solar_apps.workflows.radio.trajectory_media_cli"
_TRAJECTORY_PREVIEW = "solar_apps.frontends.app_v1.trajectory_preview_worker"
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

    def build_dart_spectrogram(
        self,
        input_dir: str | Path,
        *,
        center_frequencies: str = "",
        bandwidth_mhz: float = 2.0,
        display_mode: str = "db",
        max_samples: int = 1200,
        dpi: int = 150,
    ) -> TaskLaunch:
        selected = self.validate_directory(input_dir)
        if bandwidth_mhz <= 0:
            raise ValueError("DART bandwidth must be positive")
        if display_mode not in {"db", "linear"}:
            raise ValueError("Unsupported DART display mode")
        if max_samples < 32:
            raise ValueError("DART sample limit must be at least 32")
        if not 72 <= dpi <= 600:
            raise ValueError("DART DPI must be between 72 and 600")
        output = self._new_output_dir("dart-spectrogram")
        return TaskLaunch(
            "DART Spectrogram",
            "dart-spectrogram",
            _NATIVE_WORKER,
            (
                "dart-render",
                "--module-id",
                "dart-spectrogram",
                "--input-dir",
                str(selected),
                "--output-dir",
                str(output),
                "--center-frequencies",
                str(center_frequencies),
                "--bandwidth-mhz",
                str(float(bandwidth_mhz)),
                "--display-mode",
                display_mode,
                "--max-frequency-samples",
                str(int(max_samples)),
                "--max-time-samples",
                str(int(max_samples)),
                "--dpi",
                str(int(dpi)),
            ),
            output,
            "\n".join(
                (
                    "Module: DART Spectrogram",
                    f"Input: {selected}",
                    (
                        "Parameters: native spectrum; "
                        f"display={display_mode}; centers={center_frequencies or 'none'}; "
                        f"bandwidth={bandwidth_mhz:g} MHz"
                    ),
                    f"Output: {output}",
                    f"Workload: {self._file_count(selected)} input file(s)",
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
        frame_mode: str = "tail",
        tail_n: int = 5,
        width: int = 960,
        height: int = 720,
        theme: str = "light",
        max_frames: int = 300,
    ) -> TaskLaunch:
        center_path = self.validate_file(centers)
        aia = self._optional_directory(aia_dir)
        if frame_mode not in {"current", "tail", "all"}:
            raise ValueError("Unsupported trajectory frame mode")
        if tail_n <= 0:
            raise ValueError("Trajectory tail length must be positive")
        if not 320 <= width <= 4096 or not 240 <= height <= 4096:
            raise ValueError("Trajectory preview size is outside the supported range")
        if theme not in {"light", "dark"}:
            raise ValueError("Trajectory preview theme must be light or dark")
        if not 1 <= max_frames <= 10_000:
            raise ValueError("Trajectory preview frame limit must be 1–10000")
        output = self._new_output_dir("source-trajectory")
        arguments: list[str] = [
            "--centers",
            str(center_path),
            "--output-dir",
            str(output),
            "--frame-mode",
            frame_mode,
            "--tail-n",
            str(int(tail_n)),
            "--width",
            str(int(width)),
            "--height",
            str(int(height)),
            "--theme",
            theme,
            "--max-frames",
            str(int(max_frames)),
        ]
        if aia is not None:
            arguments.extend(["--aia-dir", str(aia)])
        return TaskLaunch(
            "Source Trajectory Playback",
            "source-trajectory",
            _TRAJECTORY_PREVIEW,
            tuple(arguments),
            output,
            "\n".join(
                (
                    "Module: Source Trajectory Playback",
                    f"Input: centers={center_path}; AIA={aia or 'none'}",
                    (
                        f"Parameters: mode={frame_mode}; tail_n={tail_n}; "
                        f"size={width}x{height}; theme={theme}; "
                        f"max_frames={max_frames}"
                    ),
                    f"Output: {output}",
                    f"Workload: {self._table_rows(center_path)} table row(s)",
                )
            ),
        )

    def build_trajectory_media(
        self,
        centers: str | Path,
        *,
        aia_dir: str | Path | None = None,
        output_format: str = "mp4",
        frame_mode: str = "tail",
        tail_n: int = 5,
        fps: float = 6.0,
        width: int = 1280,
        height: int = 720,
        theme: str = "light",
    ) -> TaskLaunch:
        center_path = self.validate_file(centers)
        aia = self._optional_directory(aia_dir)
        if output_format not in {"mp4", "gif", "webm"}:
            raise ValueError("Trajectory media format must be MP4, GIF, or WebM")
        if frame_mode not in {"current", "tail", "all"}:
            raise ValueError("Unsupported trajectory frame mode")
        if tail_n <= 0:
            raise ValueError("Trajectory tail length must be positive")
        if not 0.2 <= fps <= 120:
            raise ValueError("Trajectory frame rate must be 0.2–120 fps")
        if not 320 <= width <= 4096 or not 240 <= height <= 4096:
            raise ValueError("Trajectory media size is outside the supported range")
        if theme not in {"light", "dark"}:
            raise ValueError("Trajectory media theme must be light or dark")
        output = self._new_output_dir("source-trajectory")
        arguments = [
            "--centers",
            str(center_path),
            "--output-dir",
            str(output),
            "--format",
            output_format,
            "--frame-mode",
            frame_mode,
            "--tail-n",
            str(int(tail_n)),
            "--fps",
            str(float(fps)),
            "--width",
            str(int(width)),
            "--height",
            str(int(height)),
            "--theme",
            theme,
        ]
        if aia is not None:
            arguments.extend(["--aia-dir", str(aia), "--use-aia"])
        return TaskLaunch(
            "Source Trajectory Media",
            "source-trajectory",
            _TRAJECTORY_MEDIA,
            tuple(arguments),
            output,
            "\n".join(
                (
                    "Module: Source Trajectory Media",
                    f"Input: centers={center_path}; AIA={aia or 'none'}",
                    (
                        f"Parameters: format={output_format}; mode={frame_mode}; "
                        f"tail_n={tail_n}; fps={fps:g}; size={width}x{height}; "
                        f"theme={theme}"
                    ),
                    f"Output: {output}",
                    f"Workload: {self._table_rows(center_path)} table row(s)",
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

    @staticmethod
    def _file_count(directory: Path) -> int:
        return sum(1 for item in directory.iterdir() if item.is_file())

    @staticmethod
    def _table_rows(path: Path) -> int:
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            return max(sum(1 for _line in handle) - 1, 0)


__all__ = ["Phase2CAdapter"]
