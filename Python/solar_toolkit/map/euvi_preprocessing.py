"""Audited Python EUVI preparation; no SECCHI_PREP or instrument calibration.

Only declared DN values are divided by exposure. Header pointing is retained,
not recalibrated. Unknown processing and telemetry masks remain explicit limits.
"""

import re
from dataclasses import dataclass

import astropy.units as u
import numpy as np

VERSION = "python-euvi-v2"
__all__ = [
    "VERSION",
    "Preparation",
    "intensity_array",
    "prepare_euvi_map",
    "prepared_map",
    "unit_info",
    "unit_factor",
    "geometry_issues",
    "quantitative_issues",
]
_CCDPIX = u.def_unit("CCDPIX")


def unit_info(value):
    """Explicit detector units, with Astropy parsing for notation and scale."""
    text = str(value).strip()
    text = re.sub(r"\bdn\b", "DN", text, flags=re.I)
    text = re.sub(r"\b(pixel|pixels)\b", "pix", text, flags=re.I)
    text = re.sub(r"\bccdpix\b", "CCDPIX", text, flags=re.I)
    try:
        with u.add_enabled_units([_CCDPIX]):
            parsed = u.Unit(text, parse_strict="raise")
        for family, base in (("DN", u.DN), ("photon", u.photon), ("count", u.ct)):
            for sampling, pixel in (
                ("none", u.one),
                ("pixel", u.pix),
                ("ccdpixel", _CCDPIX),
            ):
                if parsed.is_equivalent(base / pixel / u.s):
                    return parsed, family, sampling, "rate"
                if parsed.is_equivalent(base / pixel):
                    return parsed, family, sampling, "integrated"
    except (ValueError, TypeError):
        pass
    return None


def unit_factor(source, target):
    first, second = unit_info(source), unit_info(target)
    if first is None or second is None or first[1:] != second[1:]:
        raise ValueError("incompatible_or_unknown_units")
    return float(first[0].to(second[0]))


def geometry_issues(smap):
    """Explicit metadata required for multi-observer geometry; no Earth fallback."""
    m = smap.meta
    reasons = []
    for key in ("hgln_obs", "hglt_obs", "dsun_obs"):
        try:
            value = float(m[key])
            if not np.isfinite(value) or (key == "dsun_obs" and value <= 0):
                raise ValueError()
        except (KeyError, TypeError, ValueError):
            reasons.append("missing_or_invalid_" + key)
    try:
        matrix = np.asarray(smap.wcs.pixel_scale_matrix, float)
        if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-20:
            raise ValueError()
        if not str(m.get("ctype1", "")).startswith("HPLN") or not str(
            m.get("ctype2", "")
        ).startswith("HPLT"):
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        reasons.append("invalid_helioprojective_wcs")
    return reasons


def quantitative_issues(smap, *, intensity_decision=None):
    """Audit metadata, optionally reusing an already computed exposure decision."""
    if intensity_decision is None:
        _, _, action, warnings = intensity_array(smap)
    else:
        action, warnings = intensity_decision
    reasons = list(warnings) if action == "unchanged_requires_review" else []
    try:
        missing = float(smap.meta.get("nmissing", 0) or 0)
        if not np.isfinite(missing) or missing < 0:
            raise ValueError()
        if missing > 0:
            reasons.append("telemetry_missing_blocks_not_decoded")
    except (ValueError, TypeError):
        reasons.append("invalid_missing_block_count")
    return reasons


@dataclass(frozen=True)
class Preparation:
    data: np.ndarray
    valid: np.ndarray
    unit: str
    audit: dict
    display_usable: bool = True
    quantitative_usable: bool = False
    reasons: tuple = ()


def intensity_array(smap, *, allow_unspecified_dn=False):
    """Return pixels, unit and exposure decision without mutating the input."""
    data = np.array(smap.data, dtype=float, copy=True)
    mask = getattr(smap, "mask", None)
    if mask is not None:
        data[np.asarray(mask, dtype=bool)] = np.nan
    data[~np.isfinite(data)] = np.nan
    unit = str(smap.meta.get("bunit", "")).strip()
    compact = re.sub(r"\s+", "", unit).lower()
    # Preserve rate units, including non-DN rates, without relabelling photons.
    parsed = unit_info(unit)
    rate = parsed is not None and parsed[3] == "rate"
    integrated = parsed is not None and parsed[1] == "DN" and parsed[3] == "integrated"
    warnings = []
    if not compact and allow_unspecified_dn:
        unit, integrated = "DN", True
        warnings.append("unspecified_unit_assumed_dn_for_legacy_non_euvi")
    if rate:
        action = "already_rate_no_division"
    elif integrated and not smap.meta.get("jetnorm") and not smap.meta.get("spprep"):
        try:
            exposure = float(smap.meta.get("exptime", 0))
        except (ValueError, TypeError):
            exposure = np.nan
        if not np.isfinite(exposure) or exposure <= 0:
            data.setflags(write=False)
            return (
                data,
                unit,
                "unchanged_requires_review",
                ["invalid_exposure_for_normalization"],
            )
        if parsed:
            base = (
                u.DN / {"none": u.one, "pixel": u.pix, "ccdpixel": _CCDPIX}[parsed[2]]
            )
            data *= parsed[0].to(base)
        data /= exposure
        unit = "DN/s" + (
            {"pixel": "/pix", "ccdpixel": "/CCDPIX"}.get(parsed[2], "")
            if parsed
            else ""
        )
        action = "divided_by_header_exptime"
    else:
        action = "unchanged_requires_review"
        warnings.append("unit_or_normalization_history_ambiguous")
    data.setflags(write=False)
    return data, unit, action, warnings


def prepare_euvi_map(smap):
    """Return native data, validity and an auditable limited-preparation record."""
    if str(smap.meta.get("detector", "")).strip().upper() != "EUVI":
        raise ValueError("EUVI input required")
    data, unit, action, warnings = intensity_array(smap)
    meta = smap.meta
    history = meta.get("history", [])
    if isinstance(history, str):
        history = history.splitlines()
    history = [str(line) for line in history]
    evidence = [
        line
        for line in history
        if any(
            key in line.lower()
            for key in ("point", "rectif", "roll", "secchi", "calib", "bias", "seb")
        )
    ]
    quantitative = quantitative_issues(smap, intensity_decision=(action, warnings))
    warnings.extend(r for r in quantitative if r not in warnings)
    if action == "divided_by_header_exptime":
        warnings.append(
            "header_exptime_used_without_seb_or_effective_exposure_recalibration"
        )
    if not all(k in meta for k in ("hgln_obs", "hglt_obs", "dsun_obs")):
        warnings.append("observer_metadata_requires_review")
    try:
        matrix = np.asarray(smap.rotation_matrix, dtype=float)
    except (ValueError, TypeError, AttributeError, ZeroDivisionError):
        matrix = np.full((2, 2), np.nan)
    determinant = float(np.linalg.det(matrix))
    if (
        not np.isfinite(matrix).all()
        or not np.isfinite(determinant)
        or abs(determinant) < 1e-12
    ):
        warnings.append("nonfinite_or_singular_wcs_rotation")
    keys = (
        "crpix1",
        "crpix2",
        "crval1",
        "crval2",
        "cdelt1",
        "cdelt2",
        "crota",
        "crota2",
        "pc1_1",
        "pc1_2",
        "pc2_1",
        "pc2_2",
        "hgln_obs",
        "hglt_obs",
        "dsun_obs",
        "date-obs",
        "exptime",
        "bunit",
        "calfac",
        "rectify",
        "nmissing",
        "ip_00_19",
    )
    valid = np.isfinite(data)
    valid.setflags(write=False)
    audit = dict(
        version=VERSION,
        status="limited_preprocessing",
        exposure_action=action,
        input_unit=str(meta.get("bunit", "")),
        output_unit=unit,
        finite_fraction=float(valid.mean()),
        zeros_preserved=int(np.sum(data == 0)),
        warnings=warnings,
        header_evidence=evidence,
        pointing_audit=dict(
            rotation_matrix=matrix.tolist(),
            determinant=determinant,
            additional_rotation_applied=False,
            independent_pointing_validation=False,
        ),
        header_snapshot={k: str(meta[k]) for k in keys if k in meta},
        performed=[
            "declared_unit_handling",
            "existing_mask_and_nonfinite_preservation",
            "native_wcs_preserved",
        ],
        not_performed=[
            "bias_correction",
            "seb_undo",
            "effective_exposure_recalibration",
            "flat_field",
            "radiometric_response",
            "new_pointing_or_roll_solution",
            "telemetry_block_decoding",
        ],
        idl_invoked=False,
    )
    reasons = quantitative + geometry_issues(smap)
    if not valid.any():
        reasons.append("no_valid_pixels")
    audit.update(
        display_usable=bool(valid.any()),
        quantitative_usable=not reasons,
        quantitative_scope="within_instrument_relative_analysis_not_absolute_calibration",
        reasons=reasons,
    )
    return Preparation(
        data, valid, unit, audit, bool(valid.any()), not reasons, tuple(reasons)
    )


def prepared_map(smap):
    """Build a display/measurement Map retaining the original native WCS."""
    import sunpy.map

    result = prepare_euvi_map(smap)
    meta = smap.meta.copy()
    if result.unit:
        meta["bunit"] = result.unit
    meta["eupyver"] = VERSION
    meta["eupystat"] = "limited_preprocessing"
    return sunpy.map.Map(result.data, meta, mask=~result.valid), result.audit
