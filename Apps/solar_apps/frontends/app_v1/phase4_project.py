# SPDX-License-Identifier: GPL-3.0-only
"""Image Composer schema-1 persistence and dirty-project lifecycle."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFileDialog, QListWidgetItem, QMessageBox

from solar_apps.frontends.image_composer.catalog import scan_folder
from solar_apps.frontends.image_composer.models import ComposerProject
from solar_apps.frontends.image_composer.project import load_project, save_project


class ComposerProjectControls:
    """Keep save/import/discard behavior on the panel's existing model state."""

    def choose_project(self) -> None:
        if not self.confirm_discard_changes():
            return
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Import Image Composer project",
            str(self.adapter.runtime.workspaces_dir),
            "Image Composer project (*.fic.json);;JSON (*.json)",
        )
        if selected:
            self.import_project(selected)

    def import_project(self, path: str | Path) -> bool:
        candidate = Path(path).expanduser().resolve(strict=False)
        try:
            if not self.adapter._inside(candidate):
                raise PermissionError(
                    f"Project is outside configured allowed roots: {candidate}"
                )
            project = load_project(candidate)
            for folder in project.folders:
                resolved = folder.path.expanduser().resolve(strict=False)
                folder.path = resolved
                folder.records = (
                    scan_folder(resolved)
                    if self.adapter._inside(resolved) and resolved.is_dir()
                    else []
                )
                folder.resolved = bool(folder.records)
                if folder.records:
                    folder.start_index = min(
                        max(1, folder.start_index), len(folder.records)
                    )
                    folder.end_index = min(
                        max(folder.start_index, folder.end_index),
                        len(folder.records),
                    )
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Could not import project", str(exc))
            return False
        self.project = project
        self._reload_project_ui()
        self.project_path = candidate
        self._dirty = False
        return True

    def _reload_project_ui(self, *, select_folder_id: str = "") -> None:
        self._updating_controls = True
        self.folder_list.clear()
        self.thumbnail_list.clear()
        self.scene.clear()
        self._items.clear()
        selected_item: QListWidgetItem | None = None
        try:
            self.canvas_width.setValue(self.project.canvas.width)
            self.canvas_height.setValue(self.project.canvas.height)
            self.canvas_background.setText(self.project.canvas.background)
            for folder in self.project.folders:
                state = "" if folder.resolved else " — unresolved"
                item = QListWidgetItem(f"{folder.name} ({len(folder.records)}){state}")
                item.setData(Qt.ItemDataRole.UserRole, folder.id)
                self.folder_list.addItem(item)
                if folder.id == select_folder_id:
                    selected_item = item
            for slot in sorted(self.project.slots, key=lambda item: item.z_index):
                self._add_graphics_item(slot)
            self._refresh_match_folders()
            self.match_mode.setCurrentText(self.project.matching.mode)
            self.match_tolerance.setValue(self.project.matching.tolerance_seconds)
            self.match_strict.setChecked(self.project.matching.strict)
            self.output_path.setText(self.project.export.output_path)
            self.output_format.setCurrentText(self.project.export.output_format)
            self.export_fps.setValue(self.project.export.fps)
            self.export_frames.setChecked(self.project.export.save_png_frames)
            self._refresh_scene_rect()
        finally:
            self._updating_controls = False
        if selected_item is None and self.folder_list.count():
            selected_item = self.folder_list.item(0)
        if selected_item is not None:
            self.folder_list.setCurrentItem(selected_item)

    def choose_save_project(self) -> None:
        selected, _filter = QFileDialog.getSaveFileName(
            self,
            "Save Image Composer project",
            str(self.adapter.runtime.workspaces_dir / "composition.fic.json"),
            "Image Composer project (*.fic.json)",
        )
        if not selected:
            return
        self._save_project_to(selected)

    def save_current_project(self) -> bool:
        if self.project_path is None:
            self.choose_save_project()
            return not self._dirty
        return self._save_project_to(self.project_path)

    def _save_project_to(self, path: str | Path) -> bool:
        candidate = Path(path).expanduser().resolve(strict=False)
        try:
            if not self.adapter._inside(candidate):
                raise PermissionError(
                    f"Project is outside configured allowed roots: {candidate}"
                )
            saved = save_project(candidate, self.project)
        except OSError as exc:
            QMessageBox.critical(self, "Could not save project", str(exc))
            return False
        self.project_path = saved
        self._dirty = False
        QMessageBox.information(self, "Project saved", str(saved))
        return True

    def new_project(self) -> bool:
        if not self.confirm_discard_changes():
            return False
        self.project = ComposerProject()
        self.project_path = None
        self._dirty = False
        self._reload_project_ui()
        return True

    def _mark_dirty(self) -> None:
        if not self._updating_controls:
            self._dirty = True

    def confirm_discard_changes(self) -> bool:
        if not self._dirty:
            return True
        result = QMessageBox.warning(
            self,
            "Unsaved Image Composer project",
            "Save changes to the current .fic.json project?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if result == QMessageBox.StandardButton.Cancel:
            return False
        if result == QMessageBox.StandardButton.Save:
            return self.save_current_project()
        return True


__all__ = ["ComposerProjectControls"]
