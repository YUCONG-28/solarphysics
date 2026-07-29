"""Shared plotting configuration and annotation helpers."""

from __future__ import annotations

import datetime as dt
from typing import Any


def get_aia_wavelength_config(wavelength: int) -> dict[str, Any]:
    """Return the historical colormap and logarithmic limits for an AIA band."""

    configs = {
        94: {"cmap": "sdoaia94", "vmin": 0.4, "vmax": 6666},
        131: {"cmap": "sdoaia131", "vmin": 0.7, "vmax": 6666},
        171: {"cmap": "sdoaia171", "vmin": 16, "vmax": 6666},
        193: {"cmap": "sdoaia193", "vmin": 42, "vmax": 6666},
        211: {"cmap": "sdoaia211", "vmin": 18, "vmax": 6666},
        304: {"cmap": "sdoaia304", "vmin": 0.9, "vmax": 2222},
    }
    return configs.get(wavelength, {"cmap": "sdoaia94", "vmin": 1.0, "vmax": 1e4})


def setup_chinese_font() -> None:
    """Select one installed font instead of requesting missing font families."""

    from solar_toolkit.visualization import configure_chinese_fonts

    configure_chinese_fonts()


def create_figure_with_white_background(
    figsize: tuple[float, float] = (10, 8),
):
    """Create a Matplotlib figure and axes with white backgrounds."""

    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=figsize, facecolor="white")
    axes = figure.add_subplot(111)
    axes.set_facecolor("white")
    return figure, axes


def add_frequency_highlight_lines(
    ax,
    frequencies: list[float],
    freq_range: tuple[float, float],
    time_range: tuple[dt.datetime, dt.datetime],
    color: str = "red",
) -> None:
    """Add labelled horizontal frequency highlights to a spectrogram axes."""

    import matplotlib.dates as mdates

    f_start, f_end = freq_range
    t_start, t_end = time_range
    for frequency in frequencies:
        if f_start <= frequency <= f_end:
            ax.axhline(
                y=frequency,
                color=color,
                linestyle="--",
                linewidth=1.3,
                alpha=0.6,
            )
            x_min = mdates.date2num(t_start)
            x_max = mdates.date2num(t_end)
            x_pos = x_min + 0.01 * (x_max - x_min)
            ax.text(
                x_pos,
                frequency + 0.01 * (f_end - f_start),
                f"{frequency} MHz",
                color=color,
                fontsize=8,
                verticalalignment="bottom",
                horizontalalignment="left",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "yellow", "alpha": 0.3},
            )


__all__ = [
    "add_frequency_highlight_lines",
    "create_figure_with_white_background",
    "get_aia_wavelength_config",
    "setup_chinese_font",
]
