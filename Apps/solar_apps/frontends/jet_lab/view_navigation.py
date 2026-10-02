"""Compact display-only navigation shared by native and projected canvases."""

from PyQt6.QtCore import QSize
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT


def compact_toolbar(canvas, parent):
    toolbar = NavigationToolbar2QT(canvas, parent, coordinates=False)
    toolbar.setIconSize(QSize(18, 18))
    toolbar.setStyleSheet("QToolButton {padding: 2px; margin: 0px;}")
    return toolbar


class ViewNavigation:
    """Wheel zoom and middle-button pan alter axes limits only, never ROI."""

    def __init__(self, canvas, toolbar):
        self.canvas, self.toolbar = canvas, toolbar
        self.pan = None
        self.connections = [
            canvas.mpl_connect(name, callback)
            for name, callback in (
                ("scroll_event", self.scroll),
                ("button_press_event", self.press),
                ("motion_notify_event", self.motion),
                ("button_release_event", self.release),
            )
        ]

    def scroll(self, event):
        if (
            event.inaxes is None
            or event.xdata is None
            or event.ydata is None
            or not event.step
        ):
            return
        ax = event.inaxes
        factor = 1.2 ** (-max(-5, min(5, event.step)))
        limits = [
            (center + (a - center) * factor, center + (b - center) * factor)
            for (a, b), center in (
                (ax.get_xlim(), event.xdata),
                (ax.get_ylim(), event.ydata),
            )
        ]
        if any(abs(b - a) < 1e-3 or abs(b - a) > 1e7 for a, b in limits):
            return
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        self.canvas.draw_idle()

    def press(self, event):
        if event.button != 2 or event.inaxes is None or self.toolbar.mode:
            return
        ax = event.inaxes
        self.pan = (
            ax,
            event.x,
            event.y,
            ax.get_xlim(),
            ax.get_ylim(),
            ax.bbox.width,
            ax.bbox.height,
        )

    def motion(self, event):
        if self.pan is None or event.x is None or event.y is None:
            return
        ax, x, y, xs, ys, width, height = self.pan
        if ax not in self.canvas.figure.axes or width <= 0 or height <= 0:
            self.pan = None
            return
        dx = (event.x - x) * (xs[1] - xs[0]) / width
        dy = (event.y - y) * (ys[1] - ys[0]) / height
        ax.set_xlim(xs[0] - dx, xs[1] - dx)
        ax.set_ylim(ys[0] - dy, ys[1] - dy)
        self.canvas.draw_idle()

    def release(self, event):
        self.pan = None

    def close(self):
        for connection in self.connections:
            self.canvas.mpl_disconnect(connection)
        self.connections.clear()
        self.pan = None
