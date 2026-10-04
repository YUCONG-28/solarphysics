"""Small PyQt6 annotation workbench; views retain their own native coordinates."""

from __future__ import annotations

import json
from collections import OrderedDict
from time import monotonic

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.widgets import PolygonSelector, RectangleSelector
from PyQt6.QtGui import QPalette
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QLabel,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from solar_toolkit.map.jet_annotations import (
    pixel_hpc,
)
from .view_navigation import compact_toolbar, ViewNavigation
from .display_preview import sample_native_view, preview_region_mask


class ObservationPane(QWidget):
    def __init__(self, owner, index):
        super().__init__()
        self.owner, self.index = owner, index
        self.doc = None
        self.display_limits = [1.0, 99.5]
        self.selector = None
        self.dragging = None
        self.drag_preview = None
        layout = QVBoxLayout(self)
        self.title = QLabel("AIA / 视角 A" if index == 0 else "EUVI / 视角 B")
        self.title.setWordWrap(True)
        policy = self.title.sizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
        self.title.setSizePolicy(policy)
        title_row = QHBoxLayout()
        layout.addLayout(title_row)
        title_row.addWidget(self.title, 1)
        self.enlarge = QPushButton("放大此图")
        self.enlarge.setToolTip("只改变显示布局；再点一次返回双图")
        self.enlarge.clicked.connect(lambda: owner.focus_native(index))
        title_row.addWidget(self.enlarge)
        self.tabs = QTabBar()
        for label in ("原图", "分割叠加", "轴线与横截面"):
            self.tabs.addTab(label)
        self.tabs.setCurrentIndex(0 if owner.native_only else 1)
        self.tabs.currentChanged.connect(lambda: self.draw())
        layout.addWidget(self.tabs)
        self.figure = Figure(figsize=(5, 5), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setAccessibleName(
            "AIA 原始图像标注画布" if index == 0 else "STEREO EUVI 原始图像标注画布"
        )
        self.canvas.setAccessibleDescription(
            "在原生图像中独立标记对应特征；黄色极线仅作搜索提示。"
        )
        self.ax = self.figure.add_subplot(111)
        self._image_artist = None
        self._mask_artist = None
        self._overlay_artists = []
        self._scaled_key = None
        self._scaled_array = None
        self._scaled_source = None
        self._mask_key = None
        self._mask_sources = None
        self._epipolar_cache = OrderedDict()
        self._drawn_hash = None
        self._coordinate_time = float("-inf")
        self._coordinate_text = ""
        self._motion_time = float("-inf")
        self._normalization_key = None
        self._normalization_limits = (0.0, 1.0)
        self._preview = None
        self._preview_data = None
        self._preview_original = True
        self._drawing = False
        self._layout_pending = True
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(40)
        self._preview_timer.timeout.connect(self._refresh_preview)
        self._limit_connections = [
            self.ax.callbacks.connect(name, self._view_changed)
            for name in ("xlim_changed", "ylim_changed")
        ]
        self.toolbar = compact_toolbar(self.canvas, self)
        self.navigation = ViewNavigation(self.canvas, self.toolbar)
        layout.addWidget(self.toolbar)
        self.mark_button = QPushButton()
        self.mark_button.setVisible(owner.native_only)
        self.mark_button.setToolTip(
            "退出两侧平移 / 框选缩放，保留放大范围和已有标注，再点击图像标点"
        )
        self.mark_button.clicked.connect(lambda: owner.start_marking())
        # QAction callbacks run after Matplotlib's mode handler; the toolbar's
        # actionTriggered signal can fire before that handler changes mode.
        for action in self.toolbar.actions():
            action.triggered.connect(
                lambda _checked=False: self.update_interaction_status()
            )
        layout.addWidget(self.mark_button)
        layout.addWidget(self.canvas, 1)
        self.connections = [
            self.canvas.mpl_connect("button_press_event", self.press),
            self.canvas.mpl_connect("button_release_event", self.release),
            self.canvas.mpl_connect("motion_notify_event", self.motion),
            self.canvas.mpl_connect("resize_event", self._canvas_resized),
            self.canvas.mpl_connect("draw_event", self._after_paint),
        ]

    def draw(self, reset=False):
        if reset or (self.doc is not None and self.doc.sha256 != self._drawn_hash):
            self.figure.set_layout_engine("constrained")
            self._layout_pending = True
        self._drawing = True
        self.update_interaction_status()
        self.disconnect_selector()
        limits = (
            (self.ax.get_xlim(), self.ax.get_ylim())
            if self._image_artist is not None
            and self.doc is not None
            and self._drawn_hash == self.doc.sha256
            and not reset
            else None
        )
        for artist in self._overlay_artists:
            artist.remove()
        self._overlay_artists.clear()
        if self.drag_preview is not None:
            self.drag_preview.remove()
        self.drag_preview = None
        doc = self.doc
        if doc is None:
            self.title.setText("尚未加载原图")
            self.title.setToolTip("")
            for artist in (self._image_artist, self._mask_artist):
                if artist is not None:
                    artist.set_visible(False)
            self._keep(
                self.ax.text(
                    0.5, 0.5, "Open FITS", transform=self.ax.transAxes, ha="center"
                )
            )
            self._drawn_hash = None
            self._preview_data = None
            self._drawing = False
            self.canvas.draw_idle()
            return
        original = self.tabs.currentIndex() == 0
        if not original:
            doc.ensure_segmented()
        data = doc.raw
        if not original and self.owner.view_layer.currentIndex() == 1:
            data = (
                doc.difference
                if doc.difference is not None
                else np.full_like(doc.raw, np.nan)
            )
        h, w = data.shape
        extent = (-0.5, w - 0.5, -0.5, h - 0.5)
        self._preview_data, self._preview_original = data, original
        self._update_preview(limits or (extent[:2], extent[2:]))
        if self._drawn_hash != doc.sha256:
            self._coordinate_time = float("-inf")
        roi = doc.state["roi"]
        if roi:
            p = np.vstack([roi, roi[0]])
            self._plot(*p.T, color="cyan", lw=1)
        common = getattr(self.owner, "common", None)
        if common and doc.sha256 in common.native_drafts:
            mask, _ = common.native_drafts[doc.sha256]
            if mask.any():
                self._keep(
                    self.ax.contour(mask, levels=[0.5], colors=["cyan"], linewidths=1)
                )
        anchor = getattr(self.owner, "epipolar_anchor", None)
        if anchor and anchor[0] != self.index:
            other = self.owner.panes[anchor[0]].doc
            if other is not None and other.sha256 == anchor[1]:
                key = (
                    other.sha256,
                    doc.sha256,
                    id(other.map),
                    id(doc.map),
                    tuple(anchor[2]),
                    self.owner.native_only,
                )
                if key not in self._epipolar_cache:
                    if self.owner.native_only:
                        from solar_toolkit.map.jet_reconstruction import (
                            epipolar_native_pixels,
                        )

                        line = epipolar_native_pixels(other, doc, anchor[2])
                    else:
                        from solar_toolkit.map.jet_viewpoint import epipolar_pixels

                        line = epipolar_pixels(other.map, doc.map, anchor[2])
                    self._epipolar_cache[key] = line
                    if len(self._epipolar_cache) > 8:
                        self._epipolar_cache.popitem(last=False)
                else:
                    line = self._epipolar_cache[key]
                    self._epipolar_cache.move_to_end(key)
                if len(line):
                    self._plot(*line.T, "--", color="yellow", lw=1)
                    self._keep(
                        self.ax.text(
                            0.02,
                            0.98,
                            "Epipolar search hint",
                            transform=self.ax.transAxes,
                            color="yellow",
                            fontsize=8,
                            va="top",
                            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"),
                        )
                    )
        if self.tabs.currentIndex() == 2:
            sk = np.asarray(doc.state["original_skeleton"])
            if len(sk):
                self._keep(self.ax.scatter(*sk.T, s=2, color="white", alpha=0.5))
            axis = np.asarray(doc.state["axis"])
            if len(axis):
                self._plot(*axis.T, color="#ffae45", lw=1.8)
                self._keep(
                    self.ax.scatter(
                        *axis[[0, -1]].T, color=["lime", "red"], s=45, zorder=5
                    )
                )
                controls = np.asarray(doc.state["control_points"])
                if self.owner.mode.currentData() in {"edit", "add", "delete"} and len(
                    controls
                ):
                    self._keep(
                        self.ax.scatter(
                            *controls.T, s=9, facecolor="none", edgecolor="yellow"
                        )
                    )
            extension = np.asarray(doc.state["extension"])
            if len(extension):
                self._plot(*extension.T, "--", color="magenta", lw=1.5)
            for section in self.owner.sections.get(self.index, []):
                p, normal = (
                    np.array(section["axis_pixel"]),
                    np.array(section["normal_pixel"]),
                )
                line = np.array([p - 10 * normal, p + 10 * normal])
                self._plot(*line.T, color="cyan", alpha=0.7, lw=0.7)
        for tie in doc.state["tiepoints"]:
            x, y = tie["pixel_xy"]
            self._plot(x, y, "+", color="magenta", ms=10)
            self._keep(
                self.ax.annotate(
                    str(tie["number"]),
                    (x, y),
                    color="white",
                    xytext=(5, 5),
                    textcoords="offset points",
                )
            )
        if self.owner.native_only and self.owner.reconstruction_current():
            for point in self.owner.reconstruction_result["points"]:
                projected = point.get("projected_pixel_xy")
                if projected is not None and projected[self.index] is not None:
                    q = projected[self.index]
                    self._plot(*q, "o", mfc="none", mec="lime", ms=8)
                    native = point.get("pixel_xy")
                    if native and native[self.index] is not None:
                        p = native[self.index]
                        self._plot([p[0], q[0]], [p[1], q[1]], color="lime", lw=1)
            fits = self.owner.reconstruction_result.get("joint_fits", {})
            model = fits.get(fits.get("selected_model"), {})
            if model.get("valid"):
                # These are fitted projections, never replacement annotations.
                fitted = []
                for point in model.get("fitted_points", []):
                    projected = point.get("predicted_pixel_xy")
                    xy = projected[self.index] if projected is not None else None
                    if xy is None or not np.isfinite(xy).all():
                        fitted.append([np.nan, np.nan])
                    else:
                        fitted.append(xy)
                if fitted:
                    self._plot(
                        *np.asarray(fitted).T,
                        "-D",
                        color="#26d9ef",
                        mfc="none",
                        ms=6,
                        lw=1.2,
                    )
        self.ax.set_xlabel("Native x [px]", fontsize=9)
        self.ax.set_ylabel("Native y [px]", fontsize=9)
        self.ax.tick_params(labelsize=9)
        dark = self.owner.palette().color(QPalette.ColorRole.Window).lightness() < 128
        text_colour = "#edf1f5" if dark else "#1b2430"
        self.ax.set_facecolor("#15191f" if dark else "white")
        self.ax.tick_params(colors=text_colour)
        self.ax.xaxis.label.set_color(text_colour)
        self.ax.yaxis.label.set_color(text_colour)
        self.ax.format_coord = self.format_coordinate
        if limits:
            self.ax.set_xlim(limits[0])
            self.ax.set_ylim(limits[1])
        else:
            self.ax.set_xlim(extent[:2])
            self.ax.set_ylim(extent[2:])
        self._drawn_hash = doc.sha256
        self._drawing = False
        info = doc.info
        self.title.setText(
            f"{info['detector']} · {info['wavelength']}\nUTC {info['midpoint_utc']}"
        )
        if "preprocessing" in info:
            audit = info["preprocessing"]
            state = (
                "单位待核查，保持原值"
                if audit["exposure_action"] == "unchanged_requires_review"
                else (
                    "已按头部曝光归一化"
                    if audit["exposure_action"] == "divided_by_header_exptime"
                    else "已是计数率，未重复除曝光"
                )
            )
            if audit["exposure_action"] == "unchanged_requires_review":
                self.title.setText(self.title.text() + " · 单位待核查")
            self.title.setToolTip(
                f"{info['observatory']} · {doc.path.name}\n{state} · 非完整仪器校准\n"
                + json.dumps(audit, ensure_ascii=False, indent=2)
            )
        self.canvas.draw_idle()
        self.set_mode(self.owner.mode.currentData())

    def _display_pixels(self):
        # QtAgg figure/axes bbox dimensions already include device_pixel_ratio.
        # Small layout-rounding changes should not regenerate the preview.
        return tuple(
            max(16, int(v // 16) * 16)
            for v in (self.ax.bbox.width, self.ax.bbox.height)
        )

    def _preview_key(self, limits):
        return (
            self.doc.sha256,
            id(self._preview_data),
            tuple(self.display_limits),
            self.owner.stretch.currentText(),
            tuple(map(tuple, limits)),
            self._display_pixels(),
        )

    def _view_changed(self, *_):
        if not self._drawing and self._preview_data is not None:
            self._preview_timer.start()

    def _canvas_resized(self, *_):
        if not self._drawing and self._preview_data is not None:
            self.figure.set_layout_engine("constrained")
            self._layout_pending = True
            self._preview_timer.start()

    def _after_paint(self, *_):
        if self._layout_pending:
            # Preserve the aspect-adjusted box when layout stops managing it.
            self.ax.set_position(self.ax.get_position(), which="both")
            self.ax.set_in_layout(True)
            # Point edits retain axes geometry. Relayout only on frame/view or
            # window changes, not for every added annotation and repaint.
            self.figure.set_layout_engine("none")
            self._layout_pending = False
        if self.doc is not None and self._preview_data is not None:
            limits = (self.ax.get_xlim(), self.ax.get_ylim())
            if self._preview_key(limits) != self._scaled_key:
                self._view_changed()

    def _refresh_preview(self):
        if self.doc is None or self._preview_data is None or self._drawing:
            return
        limits = (self.ax.get_xlim(), self.ax.get_ylim())
        self._drawing = True
        try:
            self._update_preview(limits)
            # An image extent must not undo a pan, inverted axis, or zoom.
            self.ax.set_xlim(limits[0], emit=False)
            self.ax.set_ylim(limits[1], emit=False)
        finally:
            self._drawing = False
        self.canvas.draw_idle()

    def _update_preview(self, limits):
        doc, data = self.doc, self._preview_data
        normalization_key = (doc.sha256, id(data), tuple(self.display_limits))
        if normalization_key != self._normalization_key:
            finite = data[np.isfinite(data)]
            lo, hi = (
                np.percentile(finite, self.display_limits) if len(finite) else (0, 1)
            )
            # An absolute epsilon can round back to a large constant ``lo``.
            # Keep the span explicitly; a constant image displays as zero.
            span = hi - lo
            self._normalization_limits = (lo, span if span > 0 else None)
            self._normalization_key = normalization_key
        key = self._preview_key(limits)
        if key != self._scaled_key:
            preview = sample_native_view(data, limits, self._display_pixels())
            lo, span = self._normalization_limits
            scaled = (
                np.clip((preview.data - lo) / span, 0, 1)
                if span is not None
                else np.where(np.isfinite(preview.data), 0.0, np.nan)
            )
            if self.owner.stretch.currentText() == "Asinh":
                scaled = np.arcsinh(scaled * 10) / np.arcsinh(10)
            self._scaled_array = np.ma.masked_invalid(scaled.astype(np.float32))
            self._scaled_source, self._scaled_key, self._preview = data, key, preview
            if self._image_artist is None:
                self._image_artist = self.ax.imshow(
                    self._scaled_array,
                    origin="lower",
                    extent=preview.extent,
                    interpolation="nearest",
                    vmin=0,
                    vmax=1,
                )
            else:
                self._image_artist.set_data(self._scaled_array)
                self._image_artist.set_extent(preview.extent)
            self.canvas.setToolTip(
                "原生像素坐标；全图显示可抽样，放大后恢复原生细节。标注和计算始终使用原始数组。"
            )
        self._image_artist.set_visible(True)
        self._image_artist.set_cmap(self.owner.colour.currentText())
        if not self._preview_original:
            mask_key = (key, id(doc.labels), id(doc.selected))
            if mask_key != self._mask_key:
                mask = doc.selected if doc.selected is not None else doc.labels > 0
                shown = preview_region_mask(mask, self._preview)
                overlay = np.zeros((*shown.shape, 4), dtype=np.float32)
                overlay[shown] = [0.1, 0.9, 0.7, 0.32]
                if self._mask_artist is None:
                    self._mask_artist = self.ax.imshow(
                        overlay,
                        origin="lower",
                        extent=self._preview.extent,
                        interpolation="nearest",
                    )
                else:
                    self._mask_artist.set_data(overlay)
                    self._mask_artist.set_extent(self._preview.extent)
                self._mask_key, self._mask_sources = mask_key, (
                    doc.labels,
                    doc.selected,
                )
            self._mask_artist.set_visible(True)
        elif self._mask_artist is not None:
            self._mask_artist.set_visible(False)

    def _keep(self, artist):
        self._overlay_artists.append(artist)
        return artist

    def _plot(self, *args, **kwargs):
        artists = self.ax.plot(*args, **kwargs)
        self._overlay_artists.extend(artists)
        return artists

    def format_coordinate(self, x, y):
        if not self.doc:
            return ""
        now = monotonic()
        if now - self._coordinate_time < 1 / 30:
            return self._coordinate_text
        self._coordinate_time = now
        try:
            tx, ty = pixel_hpc(self.doc.map, [[x, y]])[0]
        except (ValueError, TypeError, AttributeError):
            self._coordinate_text = (
                f"native pixel ({x:.2f}, {y:.2f}) | WCS 无效，坐标不可用于测量"
            )
        else:
            self._coordinate_text = (
                f"pixel ({x:.2f}, {y:.2f}) | HPC ({tx:.2f}, {ty:.2f}) arcsec"
            )
        return self._coordinate_text

    def disconnect_selector(self):
        if self.selector:
            self.selector.set_active(False)
            self.selector.disconnect_events()
            for artist in self.selector.artists:
                artist.remove()
            self.selector = None

    def set_mode(self, mode):
        self.update_interaction_status()
        self.disconnect_selector()
        if mode == "rectangle":

            def rectangle(a, b):
                vertices = [
                    [a.xdata, a.ydata],
                    [b.xdata, a.ydata],
                    [b.xdata, b.ydata],
                    [a.xdata, b.ydata],
                ]
                self.owner.perform(lambda: self.doc.set_roi(vertices), self.index)

            self.selector = RectangleSelector(
                self.ax, rectangle, useblit=False, button=[1], minspanx=2, minspany=2
            )
        elif mode == "polygon":
            self.selector = PolygonSelector(
                self.ax,
                lambda vertices: self.owner.perform(
                    lambda: self.doc.set_roi(vertices), self.index
                ),
                useblit=False,
            )
            self.selector.validButtons = [1]

    def update_interaction_status(self):
        if not self.owner.native_only:
            return
        mode = str(self.toolbar.mode)
        if mode:
            label = "平移中" if mode == "pan/zoom" else "框选缩放中"
            text = f"{label}：当前不标点 · 点击返回标记对应点"
        elif self.owner.mode.currentData() == "tie":
            text = "标点模式：点击图像新增 / 移动当前编号"
        else:
            text = "当前为辅助操作 / 浏览 · 点击标记对应点"
        self.mark_button.setText(text)
        anchor = getattr(self.owner, "epipolar_anchor", None)
        if anchor and anchor[0] != self.index:
            self.mark_button.setToolTip(
                "黄色虚线：极线搜索提示，不是喷流方向；不自动吸附，也不确认结构身份。点击可退出平移 / 缩放并返回标点。"
            )

    def press(self, event):
        if (
            event.button != 1
            or event.inaxes != self.ax
            or event.xdata is None
            or self.doc is None
        ):
            return
        if self.toolbar.mode:
            if self.owner.native_only:
                self.update_interaction_status()
                self.owner.notice(
                    "当前图像正在平移 / 框选缩放，不会保存对应点。点击图上方“返回标记对应点”或顶部“标记对应点”后再点击；放大范围保持不变。"
                )
            return
        self.owner.active.setCurrentIndex(self.index)
        xy = [event.xdata, event.ydata]
        mode = self.owner.mode.currentData()
        if mode in {"rectangle", "polygon", "navigate"}:
            return

        def action():
            if mode == "select":
                self.doc.select(xy)
            elif mode == "inner":
                info = self.doc.trace(xy)
                self.owner.notice(info["reason"])
                self.tabs.setCurrentIndex(2)
            elif mode == "tie":
                if self.owner.native_only:
                    self.owner.mark_native_point(self.index, xy)
                    return
                self.doc.add_tiepoint(
                    xy,
                    self.owner.tie_id.value(),
                    self.owner.tie_status.currentData(),
                    self.owner.feature.text(),
                )
                self.owner.epipolar_anchor = (self.index, self.doc.sha256, xy)
            elif mode == "remove_tie":
                ties = self.doc.state["tiepoints"]
                if not ties:
                    return
                index = int(
                    np.argmin(
                        [np.linalg.norm(np.array(t["pixel_xy"]) - xy) for t in ties]
                    )
                )
                self.doc.checkpoint("remove_tiepoint")
                ties.pop(index)
            elif mode == "extension":
                if not self.doc.state["axis"]:
                    raise ValueError("先确认实测轴线，再添加推测延伸")
                self.doc.checkpoint("conditional_extension")
                self.doc.state["extension"] = [self.doc.state["axis"][-1], xy]
            else:
                controls = np.asarray(self.doc.state["control_points"])
                if len(controls) < 2:
                    raise ValueError("先选择区域并指定内端")
                i = int(np.argmin(np.linalg.norm(controls - xy, axis=1)))
                if mode == "edit":
                    self.dragging = (i, controls.copy())
                elif mode == "delete":
                    self.doc.edit(np.delete(controls, i, axis=0))
                elif mode == "add":
                    midpoints = (controls[:-1] + controls[1:]) / 2
                    j = int(np.argmin(np.linalg.norm(midpoints - xy, axis=1)))
                    self.doc.edit(np.insert(controls, j + 1, xy, axis=0))

        self.owner.perform(action, self.index)

    def motion(self, event):
        if event.inaxes == self.ax and event.xdata is not None:
            now = monotonic()
            if now - self._motion_time < 1 / 30:
                return
            self._motion_time = now
            self.owner.statusBar().showMessage(
                self.format_coordinate(event.xdata, event.ydata)
            )
            if self.dragging is not None:
                i, original = self.dragging
                controls = original.copy()
                controls[i] = [event.xdata, event.ydata]
                if self.drag_preview is None:
                    (self.drag_preview,) = self.ax.plot(
                        *controls.T, "--", color="cyan", lw=1
                    )
                else:
                    self.drag_preview.set_data(*controls.T)
                self.canvas.draw_idle()

    def release(self, event):
        if self.dragging is None:
            # The action runs on mouse-down, when SessionUndo deliberately
            # avoids snapshots. Commit on release so rapid clicks remain
            # individually undoable instead of waiting for its polling timer.
            if self.owner.native_only and self.owner.mode.currentData() == "tie":
                history = getattr(self.owner, "undo_history", None)
                if history is not None:
                    history.capture()
            return
        i, controls = self.dragging
        self.dragging = None
        if event.inaxes == self.ax and event.xdata is not None:
            controls[i] = [event.xdata, event.ydata]
            self.owner.perform(lambda: self.doc.edit(controls), self.index)

    def cleanup(self):
        self._preview_timer.stop()
        for cid in self._limit_connections:
            self.ax.callbacks.disconnect(cid)
        self._limit_connections.clear()
        self.navigation.close()
        self.disconnect_selector()
        for cid in self.connections:
            self.canvas.mpl_disconnect(cid)
        self.connections.clear()
        self._epipolar_cache.clear()
        self._scaled_array = None
        self._scaled_source = None
        self._scaled_key = None
        self._mask_sources = None
        self._preview = self._preview_data = None
        self.figure.clear()
