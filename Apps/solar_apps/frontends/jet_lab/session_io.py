"""Versioned native-frame sessions; old display assumptions stay historical."""

from copy import deepcopy
from datetime import datetime
import json
import csv
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QSignalBlocker
from PyQt6.QtWidgets import (
    QFileDialog,
)

from solar_toolkit.map.jet_annotations import (
    save_session,
    file_sha256,
    json_safe,
)


from .undo_history import undoable


class SessionStorage:
    def session_state(self):
        drafts = {}
        for digest, (mask, evidence) in self.native_drafts.items():
            edges = np.diff(np.r_[False, mask.ravel(), False].astype(np.int8))
            drafts[digest] = {
                "shape": list(mask.shape),
                "evidence": evidence,
                "runs": np.c_[
                    np.where(edges == 1)[0], np.where(edges == -1)[0]
                ].tolist(),
            }
        state = {
            "native_roi_drafts": drafts,
            "height_rsun": self.height.currentData(),
            "preview_backend": (
                "sunpy_reproject_to"
                if self.height.currentData() == 0
                else "explicit_shell_ray_mapping"
            ),
            "preview_rsun_m": 695700000.0,
            "display_mode": self.display.currentIndex(),
            "layer": self.layer.currentIndex(),
            "reference_image_sha256": self.reference_hash,
            "limits": self.limits,
            "current_pair": self.pair_metadata(),
            "frame_index": self.slider.value(),
            "master": self.master.currentIndex(),
            "band": self.band.currentText(),
            "background": self.background.isChecked(),
            "display_limits": [list(p.display_limits) for p in self.owner.panes],
            "stretch": self.owner.stretch.currentText(),
            "colour": self.owner.colour.currentText(),
            "native_display_layer": self.owner.view_layer.currentIndex(),
            "original_visible": self.original.isChecked(),
            "window_view_mode": self.owner.view_mode,
            "vertical_panel_sizes": self.owner.vertical_splitter.sizes(),
            "native_panel_sizes": self.owner.native_splitter.sizes(),
            "native_focus": self.owner.native_focus,
            "parameters_visible": self.owner.parameter_toggle.isChecked(),
            "guide_visible": not self.owner.guide.isHidden(),
            "timeline_controls_visible": self.timeline_toggle.isChecked(),
        }
        if getattr(self.owner, "native_only", False):
            current_index = (
                self.current_pair_index()
                if hasattr(self, "current_pair_index")
                else None
            )
            state.update(
                preview_backend="disabled_native_dual_view",
                reference_image_sha256=None,
                window_view_mode="native",
                historical_shared_view=deepcopy(
                    getattr(self, "historical_shared_view", {})
                ),
                event_profile=deepcopy(getattr(self, "event_profile", {})),
                frame_index=(
                    self.slider.value() if current_index is None else current_index
                ),
                current_point_number=self.owner.tie_id.value(),
                active_editor_side=self.owner.active.currentIndex(),
            )
        return state

    def pair_metadata(self):
        if self.current_pair is None:
            return None
        # Changing/restoring originals must not reuse a previous matched time pair.
        for instrument, pane in zip(("AIA", "EUVI"), self.owner.panes, strict=True):
            record = self.current_pair.get(instrument)
            expected = record.get("image_sha256") if record else None
            actual = pane.doc.sha256 if pane.doc else None
            if expected != actual:
                return None
        pair = deepcopy(self.current_pair)
        for inst in ["AIA", "EUVI"]:
            if pair[inst] is not None:
                pair[inst] = {
                    k: v
                    for k, v in pair[inst].items()
                    if not k.startswith("_") and k not in ("image", "difference")
                }
        return pair

    @undoable("保存 / 导出会话")
    def save_to(self, destination, for_compute=False):
        guard = getattr(self.owner, "ensure_point_metadata_applied", None)
        if guard and not guard("保存 / 导出"):
            return None
        # Flush pending measurement edits before recording flags for rollback.
        self.owner.update_parameters()
        self.remember()
        dirty = {key: d.dirty for key, d in self.documents.items()}
        snapshot_dirty = {key: s["dirty"] for key, s in self.document_snapshots.items()}
        missing = object()
        saved_state = {
            name: getattr(self.owner, name, missing)
            for name in (
                "last_saved",
                "last_output",
                "saved_signature",
                "saved_science_signature",
            )
        }
        try:
            return self._save_to(destination, for_compute)
        except Exception:
            for key, flag in dirty.items():
                self.documents[key].dirty = flag
            for key, flag in snapshot_dirty.items():
                self.document_snapshots[key]["dirty"] = flag
            for name, value in saved_state.items():
                if value is missing:
                    if hasattr(self.owner, name):
                        delattr(self.owner, name)
                else:
                    setattr(self.owner, name, value)
            raise

    def _save_to(self, destination, for_compute=False):
        destination = Path(destination)
        destination.mkdir(exist_ok=False)
        self.remember()
        # Immutable measurement snapshots; display assumptions are saved separately.
        index = []
        for digest in set(self.documents) | set(self.document_snapshots):
            d = self.documents.get(digest)
            if d is None:
                d = self.thaw(self.document_snapshots[digest])
            path = save_session(destination / digest, [d])
            index.append(
                {
                    "sha256": digest,
                    "annotation": path.relative_to(destination).as_posix(),
                    "undo_stack": d._undo,
                }
            )
            if digest in self.document_snapshots:
                self.document_snapshots[digest]["dirty"] = False
        state = self.session_state()
        (destination / "view_state.json").write_text(
            json.dumps(json_safe(state), ensure_ascii=False, indent=2)
        )
        records = []
        for r in self.records:
            r = dict(r)
            import os

            r["image"] = Path(os.path.relpath(r.pop("_path"), destination)).as_posix()
            if r.get("_difference"):
                r["difference"] = Path(
                    os.path.relpath(r.pop("_difference"), destination)
                ).as_posix()
            records.append(r)
        active = [p.doc.sha256 if p.doc else None for p in self.owner.panes]
        session = {
            "schema": "solarphysics.jet_lab.session",
            "version": 2 if getattr(self.owner, "native_only", False) else 1,
            "documents": index,
            "active_images": active,
            "frames": records,
            "sample_id": self.owner.sample_id,
        }
        if getattr(self.owner, "native_only", False):
            if for_compute:
                self.owner.calculate_3d()
            current = bool(self.owner.reconstruction_current())
            report = deepcopy(self.owner.reconstruction_result)
            session.update(
                reconstruction_result=report,
                reconstruction_signature=deepcopy(self.owner.reconstruction_signature),
                reconstruction_current=current,
                fit_options=deepcopy(getattr(self.owner, "fit_options", {})),
            )
            self._write_native_calculation(destination, report, current)
        (destination / "session.json").write_text(
            json.dumps(json_safe(session), ensure_ascii=False, indent=2)
        )
        if for_compute and not getattr(self.owner, "native_only", False):
            report = self.precheck()
            # Existing single-pair consumer can read this bundle directly.
            save_session(
                destination / "calculation_pair",
                [p.doc for p in self.owner.panes],
                sample_id=self.owner.sample_id,
            )
            (destination / "calculation_input.json").write_text(
                json.dumps(
                    json_safe(
                        {
                            "schema": "solarphysics.jet_lab.calculation_input",
                            "version": 1,
                            "annotation": "calculation_pair/annotations.json",
                            "pairing": self.pair_metadata(),
                            "geometry_precheck": report,
                            "display_shell_used_for_geometry": False,
                            "requires_remote_execution": True,
                            "export_kind": (
                                "calculation_input"
                                if report.get("geometry_valid")
                                else "diagnostic_only"
                            ),
                        }
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        checks = {
            p.relative_to(destination).as_posix(): file_sha256(p)
            for p in destination.rglob("*")
            if p.is_file()
        }
        (destination / "COMPLETE.json").write_text(
            json.dumps({"version": 1, "sha256": checks}, indent=2)
        )
        self.owner.last_saved = True
        self.owner.last_output = str(destination)
        from .workflow_guide import annotation_signature

        self.owner.saved_signature = annotation_signature(self.owner)
        if hasattr(self, "mark_science_saved"):
            self.mark_science_saved()
        if hasattr(self.owner, "guide"):
            self.owner.guide.refresh()
        return destination / "session.json"

    def _write_native_calculation(self, destination, report, current):
        """Freeze native measurements and candidate results independently of views."""
        report = json_safe(report)
        has_candidate = current and any(
            point.get("numerical_valid", False)
            for point in (report or {}).get("points", [])
        )
        save_session(
            destination / "calculation_pair",
            [p.doc for p in self.owner.panes],
            sample_id=self.owner.sample_id,
        )
        result_file = (
            "reconstruction.json" if current else "reconstruction_history.json"
        )
        envelope = {
            "schema": "solarphysics.jet_lab.native_reconstruction",
            "version": 1,
            "status": (
                ("current_candidate" if has_candidate else "current_diagnostic")
                if current
                else "stale_or_not_calculated"
            ),
            "signature": self.owner.reconstruction_signature,
            "display_geometry_used": False,
            "result": report,
        }
        (destination / result_file).write_text(
            json.dumps(
                json_safe(envelope), ensure_ascii=False, indent=2, allow_nan=False
            ),
            encoding="utf-8",
        )
        axis = (report or {}).get("summary", {}).get("axis")
        joint_files = {}
        for key in ("joint_fits", "joint_robust"):
            if (report or {}).get(key) is not None:
                selected = report[key].get(
                    "selected_model", (report or {}).get("selected_model")
                )
                name = key + ("" if current else "_history") + ".json"
                payload = dict(
                    schema="solarphysics.jet_lab." + key,
                    version=1,
                    status=envelope["status"],
                    signature=envelope["signature"],
                    selected_model=selected,
                    result=report[key],
                )
                (destination / name).write_text(
                    json.dumps(
                        json_safe(payload),
                        ensure_ascii=False,
                        indent=2,
                        allow_nan=False,
                    ),
                    encoding="utf-8",
                )
                joint_files[key] = name
                csv_name = key + ("" if current else "_history") + ".csv"
                fields = [
                    "model",
                    "selected_model",
                    "number",
                    "x_Rsun",
                    "y_Rsun",
                    "z_Rsun",
                    "height_Rsun",
                    "admissible_candidate",
                    "A_observed_x_px",
                    "A_observed_y_px",
                    "B_observed_x_px",
                    "B_observed_y_px",
                    "A_x_px",
                    "A_y_px",
                    "B_x_px",
                    "B_y_px",
                    "A_residual_x_px",
                    "A_residual_y_px",
                    "B_residual_x_px",
                    "B_residual_y_px",
                    "A_residual_px",
                    "B_residual_px",
                    "issues",
                ]
                with (destination / csv_name).open(
                    "w", newline="", encoding="utf-8"
                ) as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    for model_name in ("line", "curve"):
                        model = report[key].get(model_name) or {}
                        for point in model.get("fitted_points", []):
                            row = dict(
                                model=model_name,
                                selected_model=selected,
                                number=point.get("number"),
                                height_Rsun=point.get("height_Rsun"),
                                admissible_candidate=point.get("admissible_candidate"),
                                issues=";".join(point.get("issues", [])),
                            )
                            for coordinate, value in zip(
                                ("x_Rsun", "y_Rsun", "z_Rsun"),
                                point.get("xyz_Rsun") or [],
                            ):
                                row[coordinate] = value
                            for input_key, columns in (
                                (
                                    "observed_pixel_xy",
                                    [
                                        ("A_observed_x_px", "A_observed_y_px"),
                                        ("B_observed_x_px", "B_observed_y_px"),
                                    ],
                                ),
                                (
                                    "predicted_pixel_xy",
                                    [("A_x_px", "A_y_px"), ("B_x_px", "B_y_px")],
                                ),
                                (
                                    "residual_pixel_xy",
                                    [
                                        ("A_residual_x_px", "A_residual_y_px"),
                                        ("B_residual_x_px", "B_residual_y_px"),
                                    ],
                                ),
                            ):
                                for values, names in zip(
                                    point.get(input_key) or [], columns
                                ):
                                    row.update(zip(names, values))
                            for column, value in zip(
                                ("A_residual_px", "B_residual_px"),
                                point.get("residual_px") or [],
                            ):
                                row[column] = value
                            writer.writerow(row)
                joint_files[key + "_points"] = csv_name
        axis_file = None
        if axis is not None:
            axis_file = "jet_axis.json" if current else "jet_axis_history.json"
            (destination / axis_file).write_text(
                json.dumps(
                    json_safe(
                        {
                            "schema": "solarphysics.jet_lab.axis_diagnostic",
                            "version": 1,
                            "status": envelope["status"],
                            "signature": envelope["signature"],
                            "axis": axis,
                            "interpretation": "geometric_scatter_not_localization_uncertainty",
                        }
                    ),
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            residual_name = (
                "jet_axis_residuals.csv"
                if current
                else "jet_axis_history_residuals.csv"
            )
            with (destination / residual_name).open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["number", "distance_Rsun", "distance_Mm"]
                )
                writer.writeheader()
                for row in axis.get("point_residuals", []):
                    writer.writerow({key: row.get(key) for key in writer.fieldnames})
        fields = [
            "number",
            "role",
            "order",
            "endpoint",
            "feature",
            "identity_status",
            "numerical_valid",
            "admissible_candidate",
            "geometry_valid",
            "reason",
            "x_Rsun",
            "y_Rsun",
            "z_Rsun",
            "height_Rsun",
            "gap_Mm",
            "ray_angle_deg",
            "residual_A_arcsec",
            "residual_B_arcsec",
            "issues",
            "pixel_xy",
            "hpc_arcsec",
            "closest_points_Rsun",
            "ray_distances_Rsun",
            "projected_pixel_xy",
            "below_photosphere",
            "visible",
            "source_delta_emission_s",
        ]
        csv_name = (
            "reconstruction_points.csv"
            if current
            else "reconstruction_history_points.csv"
        )
        with (destination / csv_name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for point in (report or {}).get("points", []):
                row = {key: point.get(key) for key in fields}
                xyz = point.get("xyz_Rsun") or [None, None, None]
                row.update(zip(("x_Rsun", "y_Rsun", "z_Rsun"), xyz, strict=True))
                residual = point.get("residual_arcsec") or [None, None]
                row.update(residual_A_arcsec=residual[0], residual_B_arcsec=residual[1])
                for key, value in row.items():
                    if isinstance(value, (list, dict)):
                        row[key] = json.dumps(json_safe(value), ensure_ascii=False)
                writer.writerow(row)
        data = {
            "schema": "solarphysics.jet_lab.calculation_input",
            "version": 2,
            "annotation": "calculation_pair/annotations.json",
            "pairing": self.pair_metadata(),
            "reconstruction": result_file,
            "reconstruction_current": current,
            "axis_diagnostic": axis_file,
            "joint_diagnostics": joint_files,
            "fit_options": deepcopy(getattr(self.owner, "fit_options", {})),
            "display_shell_used_for_geometry": False,
            "requires_remote_execution": False,
            "export_kind": "candidate_geometry" if has_candidate else "diagnostic_only",
            "interpretation": "native_ray_candidates_not_confirmed_physical_correspondence",
        }
        (destination / "calculation_input.json").write_text(
            json.dumps(json_safe(data), ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )

    @undoable("恢复时序会话")
    def restore_session(self, path):
        guard = getattr(self.owner, "ensure_point_metadata_applied", None)
        if guard and not guard("恢复会话"):
            return
        from .session_validation import prepare_session

        prepared = prepare_session(
            path,
            self.owner.validate,
            lazy_segmentation=getattr(self.owner, "native_only", False),
        )
        return self.apply_prepared_session(prepared)

    def apply_prepared_session(self, prepared):
        """Commit only inputs whose files, identities and view fields validated."""
        s, v, documents, drafts, records = prepared
        if hasattr(self, "cancel_pending"):
            self.cancel_pending()
        else:
            self.play.stop()
            self.debounce.stop()
            self.generation += 1
            self.pending_frame = None
            if self.future:
                self.future.cancel()
        if self.roi_future:
            self.roi_future.cancel()
            self.roi_future = None
        self.documents = documents
        self.document_snapshots = {}
        for pane, digest in zip(self.owner.panes, s["active_images"]):
            pane.doc = documents.get(digest)
        native = getattr(self.owner, "native_only", False)
        if native:
            if hasattr(self, "event_profile"):
                self.event_profile = deepcopy(
                    v.get("event_profile", self.event_profile)
                )
            self.historical_shared_view = deepcopy(v.get("historical_shared_view", {}))
            if s["version"] == 1:
                self.historical_shared_view = {
                    key: deepcopy(v.get(key))
                    for key in (
                        "height_rsun",
                        "preview_backend",
                        "reference_image_sha256",
                        "limits",
                        "display_mode",
                        "window_view_mode",
                    )
                }
        self.native_drafts = drafts
        self.reference_hash = None if native else v["reference_image_sha256"]
        ref = documents.get(self.reference_hash)
        self.reference = ref.map if ref else None
        self.limits = v["limits"]
        for pane, limits in zip(
            self.owner.panes, v.get("display_limits", [[1, 99.5], [1, 99.5]])
        ):
            pane.display_limits = list(limits)
        for widget in [self.owner.stretch, self.owner.colour, self.owner.view_layer]:
            widget.blockSignals(True)
        self.owner.stretch.setCurrentText(v.get("stretch", "Asinh"))
        self.owner.colour.setCurrentText(v.get("colour", "gray"))
        self.owner.view_layer.setCurrentIndex(v.get("native_display_layer", 0))
        for widget in [self.owner.stretch, self.owner.colour, self.owner.view_layer]:
            widget.blockSignals(False)
        self.original.blockSignals(True)
        self.original.setChecked(v.get("original_visible", True))
        self.original.blockSignals(False)
        self.owner.image_priority.blockSignals(True)
        self.owner.image_priority.setChecked(False)
        self.owner.image_priority.blockSignals(False)
        self.owner._priority_previous = None
        focus = v.get("native_focus")
        self.owner.native_focus = focus if focus in (0, 1) else None
        if self.owner.native_focus is not None:
            self.owner.active.blockSignals(True)
            self.owner.active.setCurrentIndex(self.owner.native_focus)
            self.owner.active.blockSignals(False)
        self.owner.parameter_toggle.setChecked(v.get("parameters_visible", True))
        self.owner.guide.setVisible(v.get("guide_visible", True))
        self.timeline_toggle.setChecked(
            v.get("timeline_controls_visible", bool(s["frames"]))
        )
        self.owner.set_view_mode(
            "native" if native else v.get("window_view_mode", "native")
        )
        for index, pane in enumerate(self.owner.panes):
            pane.enlarge.setText(
                "返回双图" if self.owner.native_focus == index else "放大此图"
            )
        if v.get("vertical_panel_sizes"):
            self.owner.vertical_splitter.setSizes(v["vertical_panel_sizes"])
        if v.get("native_panel_sizes"):
            self.owner.native_splitter.setSizes(v["native_panel_sizes"])
        self.height.setCurrentIndex(self.height.findData(v["height_rsun"]))
        self.display.setCurrentIndex(v["display_mode"])
        self.layer.setCurrentIndex(v["layer"])
        self.records = records
        for widget in [self.band, self.master, self.background]:
            widget.blockSignals(True)
        if native and self.band.findText(str(v["band"])) < 0:
            self.band.addItem(str(v["band"]))
        self.band.setCurrentText(v["band"])
        self.master.setCurrentIndex(v["master"])
        self.background.setChecked(v["background"])
        for widget in [self.band, self.master, self.background]:
            widget.blockSignals(False)
        self.current_pair = v.get("current_pair")
        self.owner.sample_id = s.get("sample_id", self.owner.sample_id)
        if hasattr(self, "sync_sample_selection"):
            self.sync_sample_selection()
        if native:
            self.owner.fit_options = deepcopy(s.get("fit_options", {}))
            self.owner.reconstruction_result = deepcopy(s.get("reconstruction_result"))
            self.owner.reconstruction_signature = deepcopy(
                s.get("reconstruction_signature")
            )
            if hasattr(self.owner, "fit_curve"):
                self.owner.sync_fit_controls()
            self.mark_science_saved()
            self.owner.last_saved = True
            from .workflow_guide import annotation_signature

            self.owner.saved_signature = annotation_signature(self.owner)
        self.owner.sync_controls()
        if native:
            # Older sessions did not record the selected point. Keep their
            # default selection rather than guessing it from scientific data.
            if "current_point_number" in v:
                blocker = QSignalBlocker(self.owner.tie_id)
                self.owner.tie_id.setValue(v["current_point_number"])
                del blocker
            if "active_editor_side" in v:
                blocker = QSignalBlocker(self.owner.active)
                self.owner.active.setCurrentIndex(v["active_editor_side"])
                del blocker
                self.owner.sync_controls()
            self.owner.sync_point_editor()
        self.owner.redraw()
        if self.records:
            # Rebuild the index without replacing the active, already restored originals.
            self.rebuild_pairs(select=False)
            self.slider.blockSignals(True)
            self.slider.setValue(min(v["frame_index"], self.slider.maximum()))
            self.slider.blockSignals(False)
        if native:
            self.owner.refresh_reconstruction()
        self.request()

    def save_dialog(self, for_compute=False):
        folder = QFileDialog.getExistingDirectory(
            self, "选择新会话/计算输入保存父目录", str(self.owner.allowed_roots[0])
        )
        if folder:
            try:
                dest = self.owner.validate(
                    Path(folder)
                    / ("jet_session_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")),
                    "output_directory",
                )
                p = self.save_to(dest, for_compute)
                if p is not None:
                    self.status.setText(f"已保存：{p}")
            except (ValueError, OSError) as exc:
                self.status.setText(str(exc))

    def export_dialog(self):
        self.save_dialog(True)
