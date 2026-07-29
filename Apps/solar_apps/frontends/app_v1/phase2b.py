# SPDX-License-Identifier: GPL-3.0-only
"""Import-safe Phase 2B adapters for existing radio frontends and workflows."""

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
_BAD_FRAME_WORKER = "solar_apps.frontends.app_v1.bad_frame_worker"
_RADIO_COMPOSITE_WORKER = "solar_apps.frontends.app_v1.radio_composite_worker"
_GAUSSIAN = "solar_apps.workflows.radio.source_map_cli"
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

    def build_bad_frame_review(
        self,
        input_root: str | Path,
        *,
        frequencies: str = "",
        polarizations: str = "RR,LL",
        start_index: int = 0,
        end_index: int | None = None,
        strategy: str = "rules",
        scope: str = "candidates",
        sample_count: int = 1200,
    ) -> TaskLaunch:
        selected = self.validate_input_directory(input_root)
        if start_index < 0 or (end_index is not None and end_index <= start_index):
            raise ValueError("Bad-frame end index must be greater than start index")
        if strategy not in {"rules", "labeling", "shadow"}:
            raise ValueError("Unsupported bad-frame review strategy")
        if scope not in {"candidates", "all_scanned"}:
            raise ValueError("Unsupported bad-frame review scope")
        if sample_count <= 0:
            raise ValueError("Bad-frame sample count must be positive")
        output = self._new_output_dir("bad-frame-review")
        arguments: list[str] = [
            "create",
            "--input-root",
            str(selected),
            "--output-root",
            str(output),
            "--frequencies",
            frequencies,
            "--polarizations",
            polarizations,
            "--start-index",
            str(int(start_index)),
            "--strategy",
            strategy,
            "--scope",
            scope,
            "--sample-count",
            str(int(sample_count)),
        ]
        if end_index is not None:
            arguments.extend(["--end-index", str(int(end_index))])
        for root in self.allowed_roots:
            arguments.extend(["--allowed-root", str(root)])
        return TaskLaunch(
            "Bad Frame Review",
            "bad-frame-review",
            _BAD_FRAME_WORKER,
            tuple(arguments),
            output,
            "\n".join(
                (
                    "Module: Bad Frame Review",
                    f"Input: {selected}",
                    (
                        f"Parameters: frequencies={frequencies or 'all'}; "
                        f"polarizations={polarizations}; indices={start_index}:"
                        f"{end_index if end_index is not None else 'all'}; "
                        f"strategy={strategy}; scope={scope}; "
                        f"sample_count={sample_count}"
                    ),
                    f"Output: {output}",
                    f"Workload: {len(self._fits_files(selected))} FITS candidate(s)",
                )
            ),
        )

    def build_bad_frame_action(
        self,
        review_root: str | Path,
        review_id: str,
        *,
        action: str,
        target_kind: str | None = None,
        target_id: str | None = None,
        quality: str | None = None,
        event_tags: str = "",
        artifact_tags: str = "",
        final_status: str = "completed",
        cmap: str = "coolwarm",
        transform: str = "robust_asinh",
    ) -> TaskLaunch:
        root = Path(review_root).expanduser().resolve(strict=False)
        if not root.is_dir() or not root.is_relative_to(self.runtime.outputs_dir):
            raise ValueError("Review root must be an existing App 1.0 output directory")
        if action not in {"preview", "label", "finalize", "archive"}:
            raise ValueError("Unsupported bad-frame review action")
        arguments = [
            action,
            "--output-root",
            str(root),
            "--review-id",
            str(review_id),
        ]
        for allowed in self.allowed_roots:
            arguments.extend(["--allowed-root", str(allowed)])
        if target_kind is not None:
            arguments.extend(["--target-kind", target_kind])
        if target_id is not None:
            arguments.extend(["--target-id", target_id])
        if action == "preview":
            arguments.extend(["--cmap", cmap, "--transform", transform])
        elif action == "label":
            if quality is None:
                raise ValueError("Select a quality label")
            arguments.extend(
                [
                    "--quality",
                    quality,
                    "--event-tags",
                    event_tags,
                    "--artifact-tags",
                    artifact_tags,
                ]
            )
        elif action == "finalize":
            arguments.extend(["--final-status", final_status])
        output = root
        return TaskLaunch(
            f"Bad Frame Review {action.title()}",
            "bad-frame-review",
            _BAD_FRAME_WORKER,
            tuple(arguments),
            output,
            "\n".join(
                (
                    f"Module: Bad Frame Review {action.title()}",
                    f"Input: review={review_id}; target={target_id or 'review'}",
                    (
                        f"Parameters: quality={quality or 'none'}; "
                        f"event_tags={event_tags or 'none'}; "
                        f"artifact_tags={artifact_tags or 'none'}"
                    ),
                    f"Output: {root}",
                    "Workload: one persistent native review action",
                )
            ),
        )

    def build_source_map_app(self, input_root: str | Path) -> TaskLaunch:
        selected = self.validate_input_directory(input_root)
        output = self._new_output_dir("source-map")
        arguments = (
            "source-map-discover",
            "--module-id",
            "source-map",
            "--input-dir",
            str(selected),
            "--output-dir",
            str(output),
        )
        return self._native_launch(
            "Source Map",
            "source-map",
            _NATIVE_WORKER,
            arguments,
            output,
            selected,
            "discover radio FITS files for the native Source Map workspace",
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
        roi_bounds: str = "-300,-300,300,300",
        frequencies: str = "",
    ) -> TaskLaunch:
        selected = self.validate_input_directory(radio_dir)
        if polarization not in {"L+R", "LCP", "RCP", "all"}:
            raise ValueError("Unsupported ROI polarization")
        output = self._new_output_dir("roi-lightcurve")
        arguments = (
            "roi-run",
            "--module-id",
            "roi-lightcurve",
            "--input-dir",
            str(selected),
            "--output-dir",
            str(output),
            "--polarization",
            polarization,
            "--roi-bounds",
            str(roi_bounds),
            "--frequencies",
            str(frequencies),
        )
        return self._native_launch(
            "ROI Light Curve",
            "roi-lightcurve",
            _NATIVE_WORKER,
            arguments,
            output,
            selected,
            f"extract ROI {roi_bounds} with polarization {polarization}",
        )

    def build_radio_composite(
        self,
        radio_dir: str | Path,
        dart_dir: str | Path,
        *,
        frequencies: str = "",
        polarization: str = "RR+LL",
        roi_bounds: str = "-300,-300,300,300",
        dart_bandwidth_mhz: float = 2.0,
        fps: float = 10.0,
        stride: int = 1,
        dpi: int = 160,
        transform: str = "linear",
        save_video: bool = True,
        save_frames: bool = True,
    ) -> TaskLaunch:
        radio = self.validate_input_directory(radio_dir)
        dart = self.validate_input_directory(dart_dir)
        if polarization not in {"RR+LL", "RR", "LL"}:
            raise ValueError("Unsupported Radio Composite polarization")
        if dart_bandwidth_mhz <= 0:
            raise ValueError("DART bandwidth must be positive")
        if not 0.2 <= fps <= 120:
            raise ValueError("Composite frame rate must be 0.2–120 fps")
        if stride < 1:
            raise ValueError("Composite frame stride must be positive")
        if not 72 <= dpi <= 600:
            raise ValueError("Composite DPI must be 72–600")
        if transform not in {"linear", "log10"}:
            raise ValueError("Composite transform must be linear or log10")
        if not save_video and not save_frames:
            raise ValueError("Enable composite video, PNG frames, or both")
        output = self._new_output_dir("radio-composite")
        arguments = [
            "--radio-dir",
            str(radio),
            "--dart-dir",
            str(dart),
            "--output-dir",
            str(output),
            "--frequencies",
            frequencies,
            "--polarization",
            polarization,
            "--roi-bounds",
            roi_bounds,
            "--dart-bandwidth-mhz",
            str(float(dart_bandwidth_mhz)),
            "--fps",
            str(float(fps)),
            "--stride",
            str(int(stride)),
            "--dpi",
            str(int(dpi)),
            "--transform",
            transform,
            "--save-video" if save_video else "--no-save-video",
            "--save-frames" if save_frames else "--no-save-frames",
        ]
        workload = len(self._fits_files(radio)) + len(self._fits_files(dart))
        summary = "\n".join(
            (
                "Module: Radio Composite",
                f"Input: radio={radio}; DART={dart}",
                (
                    f"Parameters: frequencies={frequencies or 'all'}; "
                    f"polarization={polarization}; roi={roi_bounds}; "
                    f"bandwidth={dart_bandwidth_mhz:g} MHz; fps={fps:g}; "
                    f"stride={stride}; dpi={dpi}; transform={transform}; "
                    f"video={save_video}; frames={save_frames}"
                ),
                f"Output: {output}",
                f"Workload: {workload} FITS candidate(s)",
            )
        )
        return TaskLaunch(
            "Radio Composite",
            "radio-composite",
            _RADIO_COMPOSITE_WORKER,
            tuple(arguments),
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

    def _native_launch(
        self,
        title: str,
        module_id: str,
        python_module: str,
        arguments: tuple[str, ...],
        output: Path,
        selected: Path,
        parameters: str,
    ) -> TaskLaunch:
        workload = len(self._fits_files(selected))
        summary = "\n".join(
            (
                f"Module: {title}",
                f"Input: {selected}",
                f"Parameters: {parameters}",
                f"Output: {output}",
                f"Workload: {workload} FITS candidate(s)",
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


__all__ = ["Phase2BAdapter"]
