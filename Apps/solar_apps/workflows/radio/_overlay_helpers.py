"""Private leaf helpers extracted from overlay_workflow.py.

No behavior change: these functions only use imports and builtins.
"""

from __future__ import annotations


def _parse_millisecond_suffix(ms_str: str) -> int:
    """Parse a radio filename suffix as integer milliseconds."""
    ms = int(str(ms_str).strip()[:3])
    if not 0 <= ms <= 999:
        raise ValueError(f"Invalid millisecond suffix: {ms_str!r}")
    return ms


def _slice_file_list(files: list[str], start_idx=None, end_idx=None) -> list[str]:
    start = int(start_idx) if start_idx is not None else 0
    end = int(end_idx) if end_idx is not None else None
    return files[start:end]


def _nearest_radio_entry_index(
    entries: list[tuple[tuple[str, int], object]],
    ref_key: tuple[str, int],
    used_indices: set[int],
    tolerance_ms: float,
) -> int | None:
    ref_date, ref_ms = ref_key
    best_index = None
    best_diff = float("inf")
    for index, (key, _item) in enumerate(entries):
        if index in used_indices:
            continue
        date_key, total_ms = key
        if date_key != ref_date:
            continue
        diff = abs(float(total_ms) - float(ref_ms))
        if diff < best_diff:
            best_index = index
            best_diff = diff
    if best_index is not None and best_diff <= tolerance_ms:
        return best_index
    return None


def _extent_panel_aspect(extent_arcsec: list[float]) -> float:
    left, right, bottom, top = [float(v) for v in extent_arcsec]
    width = abs(right - left)
    height = abs(top - bottom)
    return height / width if width > 0 else 1.0


def _normalise_band_key(band_label) -> str:
    text = str(band_label or "").strip()
    if not text:
        return ""
    if text.endswith("MHz"):
        return text
    try:
        value = float(text)
        if abs(value - round(value)) < 1e-6:
            return f"{int(round(value))}MHz"
        return f"{value:g}MHz"
    except Exception:
        return text


def _gaussian_diagnostics_row(
    fit_result,
    cfg_dict,
    source_file=None,
    band=None,
    polarization=None,
    radio_time=None,
    bg_diag=None,
):
    bg_diag = bg_diag or {}
    time_str = (
        radio_time.isoformat()
        if hasattr(radio_time, "isoformat")
        else (radio_time or "")
    )
    if fit_result is None:
        fail = cfg_dict.get("_last_gaussian_failure_diag", {})
        return {
            "source_file": source_file or fail.get("source_file", ""),
            "time": time_str,
            "band": band or "",
            "polarization": polarization or "",
            "quality_flag": fail.get("quality_flag", fail.get("reason", "fit_failed")),
            "quality_flag_detail": fail.get(
                "quality_flag_detail", fail.get("reason", "")
            ),
            "mask_pixel_count": fail.get("mask_pixel_count", 0),
            "fit_peak_fraction_threshold_used": fail.get(
                "fit_peak_fraction_threshold_used", ""
            ),
            "fit_peak_fraction_candidate_counts": fail.get(
                "fit_peak_fraction_candidate_counts", ""
            ),
            "background_rms_median": bg_diag.get(
                "background_rms_median", fail.get("background_rms_median", "")
            ),
            "background_level_median": bg_diag.get(
                "background_level_median", fail.get("background_level_median", "")
            ),
            "gaussian_fit_method": fail.get("gaussian_fit_method", "skipped"),
            "roi_used": fail.get("roi_used", ""),
            "roi_shape": fail.get("roi_shape", ""),
        }
    return {
        "source_file": source_file or getattr(fit_result, "source_file", ""),
        "time": time_str,
        "band": band or "",
        "polarization": polarization or "",
        "quality_flag": getattr(fit_result, "quality_flag", ""),
        "quality_flag_detail": getattr(fit_result, "quality_flag_detail", ""),
        "center_x_arcsec": fit_result.center_arcsec[0],
        "center_y_arcsec": fit_result.center_arcsec[1],
        "center_x_pixel": fit_result.center_pixel[0],
        "center_y_pixel": fit_result.center_pixel[1],
        "sigma_x_pixel": fit_result.sigma_pixel[0],
        "sigma_y_pixel": fit_result.sigma_pixel[1],
        "fwhm_major_arcsec": getattr(fit_result, "fwhm_major_arcsec", ""),
        "fwhm_minor_arcsec": getattr(fit_result, "fwhm_minor_arcsec", ""),
        "amplitude": getattr(fit_result, "amplitude", ""),
        "snr": getattr(fit_result, "snr", ""),
        "residual_rms": getattr(fit_result, "residual_rms", ""),
        "mask_pixel_count": getattr(fit_result, "mask_pixel_count", ""),
        "fit_peak_fraction_threshold_used": getattr(
            fit_result, "fit_peak_fraction_threshold_used", ""
        ),
        "fit_peak_fraction_candidate_counts": getattr(
            fit_result, "fit_peak_fraction_candidate_counts", ""
        ),
        "background_rms_median": bg_diag.get(
            "background_rms_median", getattr(fit_result, "background_rms_median", "")
        ),
        "background_level_median": bg_diag.get(
            "background_level_median",
            getattr(fit_result, "background_level_median", ""),
        ),
        "gaussian_fit_method": getattr(fit_result, "gaussian_fit_method", ""),
        "roi_used": getattr(fit_result, "roi_used", ""),
        "roi_shape": getattr(fit_result, "roi_shape", ""),
    }


def _aia_cutout_extent_arcsec(aia_cutout) -> list[float]:
    return [
        aia_cutout.bottom_left_coord.Tx.value,
        aia_cutout.top_right_coord.Tx.value,
        aia_cutout.bottom_left_coord.Ty.value,
        aia_cutout.top_right_coord.Ty.value,
    ]


def _radio_file_item_label(file_item) -> str:
    if isinstance(file_item, tuple):
        return f"{file_item[0]}|{file_item[1]}"
    return str(file_item)


def _header_unit_to_arcsec_scale(unit_value) -> float:
    unit = str(unit_value or "arcsec").strip().lower()
    if unit in {"deg", "degree", "degrees"}:
        return 3600.0
    if unit in {"arcmin", "arcminute", "arcminutes"}:
        return 60.0
    return 1.0


__all__ = [
    "_parse_millisecond_suffix",
    "_slice_file_list",
    "_nearest_radio_entry_index",
    "_extent_panel_aspect",
    "_normalise_band_key",
    "_gaussian_diagnostics_row",
    "_aia_cutout_extent_arcsec",
    "_radio_file_item_label",
    "_header_unit_to_arcsec_scale",
]
