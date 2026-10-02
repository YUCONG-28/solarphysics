"""Protect unapplied correspondence descriptions without changing pixel data."""

from copy import deepcopy

from PyQt6.QtCore import QSignalBlocker
from PyQt6.QtWidgets import QMessageBox


class PointMetadataEditor:
    def point_editor_values(self):
        return dict(
            feature=self.feature.text(),
            identity_status=self.tie_status.currentData(),
            **self.point_metadata(),
        )

    def reset_point_editor_baseline(self):
        self._point_editor_number = self.tie_id.value()
        self._point_editor_baseline = deepcopy(self.point_editor_values())
        self.refresh_point_metadata_pending()

    def point_metadata_pending(self):
        number = getattr(self, "_point_editor_number", self.tie_id.value())
        present = any(
            t["number"] == number
            for p in self.panes
            if p.doc
            for t in p.doc.state["tiepoints"]
        )
        return present and self.point_editor_values() != getattr(
            self, "_point_editor_baseline", self.point_editor_values()
        )

    def refresh_point_metadata_pending(self, *_):
        if hasattr(self, "metadata_pending_label"):
            pending = self.point_metadata_pending()
            self.metadata_pending_label.setText(
                f"编号 {self._point_editor_number}：说明尚未应用，请点击“应用点说明”。"
                if pending
                else "点说明已应用；修改原图标点后请重新计算。"
            )
            self.metadata_pending_label.setProperty("unapplied", bool(pending))

    def point_metadata_choice(self, action_label):
        """Return an explicit choice; kept separate for real-button tests."""
        dialog = QMessageBox(self)
        dialog.setWindowTitle("点说明尚未应用")
        dialog.setText(
            f"编号 {self._point_editor_number} 的用途、顺序或身份说明尚未应用。"
        )
        dialog.setInformativeText(
            f"在{action_label}前，选择应用到两侧已有点、放弃此次说明修改，或返回编辑。原图坐标保持不变。"
        )
        apply = dialog.addButton("应用后继续", QMessageBox.ButtonRole.AcceptRole)
        discard = dialog.addButton(
            "放弃说明修改", QMessageBox.ButtonRole.DestructiveRole
        )
        back = dialog.addButton("返回编辑", QMessageBox.ButtonRole.RejectRole)
        dialog.setDefaultButton(back)
        dialog.setEscapeButton(back)
        dialog.exec()
        return (
            "apply"
            if dialog.clickedButton() is apply
            else "discard" if dialog.clickedButton() is discard else "return"
        )

    def restore_point_editor_values(self, values):
        widgets = [
            self.feature,
            self.tie_status,
            self.point_role,
            self.point_order,
            self.point_endpoint,
        ]
        blockers = [QSignalBlocker(w) for w in widgets]
        self.feature.setText(values.get("feature", ""))
        self.tie_status.setCurrentIndex(
            self.tie_status.findData(values.get("identity_status", "possible"))
        )
        self.point_role.setCurrentIndex(
            self.point_role.findData(values.get("role", "unknown"))
        )
        self.point_order.setValue(values.get("order") or self._point_editor_number)
        self.point_endpoint.setCurrentIndex(
            self.point_endpoint.findData(values.get("endpoint"))
        )
        del blockers
        self.refresh_point_metadata_pending()

    def ensure_point_metadata_applied(self, action_label="继续"):
        """False cancels a caller before any file, frame or result mutation."""
        if not hasattr(self, "point_role") or getattr(
            getattr(self, "undo_history", None), "restoring", False
        ):
            return True
        if not self.point_metadata_pending():
            return True
        choice = self.point_metadata_choice(action_label)
        if choice == "apply":
            self.apply_point_metadata()
            return True
        if choice == "discard":
            self.restore_point_editor_values(self._point_editor_baseline)
            return True
        return False

    def point_number_changed(self, number):
        old = getattr(self, "_point_editor_number", number)
        if number != old and not self.ensure_point_metadata_applied("切换编号"):
            blocker = QSignalBlocker(self.tie_id)
            self.tie_id.setValue(old)
            del blocker
            return
        self.sync_point_editor()
