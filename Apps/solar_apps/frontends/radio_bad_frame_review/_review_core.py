"""Private core implementation for radio bad-frame review.

The public facade lives in review.py; this module holds the whole
implementation body and must never be imported by external code.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from solar_toolkit.radio.quality_science import (
    AUTOMATIC_QUALITY_RULE_VERSION,
    analyze_radio_science_quality,
    write_radio_science_quality_csv,
)
from solar_toolkit.radio.raw_quality import (
    RawQualityThresholds,
    analyze_radio_raw_quality,
    read_radio_fits_image,  # noqa: F401 - re-exported via the facade
)
from solar_toolkit.map.coordinates import (
    calculate_fits_extent_from_header,
    infer_image_origin_from_header,
)
from ._review_helpers import (
    _format_frequency,
)
import solar_apps.frontends.radio_bad_frame_review.review as _review_facade  # noqa: E402

BAD_FRAME_REVIEW_SCHEMA_VERSION = 3
SUPPORTED_BAD_FRAME_REVIEW_SCHEMA_VERSIONS = frozenset({1, 2, 3})
REVIEW_STATUSES = frozenset({"draft", "completed", "skipped"})
REVIEW_DECISIONS = frozenset({"bad", "good"})
QUALITY_LABELS = frozenset({"good", "degraded", "bad", "uncertain"})
TRAINABLE_QUALITY_LABELS = frozenset({"good", "degraded", "bad"})
EVENT_TAGS = frozenset({"solar_burst", "narrowband_event", "strong_polarization"})
ARTIFACT_TAGS = frozenset(
    {
        "stripe",
        "sidelobe",
        "offset",
        "blur",
        "saturation",
        "noise",
        "invalid_data",
        "other",
    }
)
REVIEW_STRATEGIES = frozenset({"rules", "labeling", "shadow"})
REVIEW_SCOPES = frozenset({"candidates", "all_scanned"})
DEFAULT_LABELING_SAMPLE_COUNT = 1200
MINIMUM_AUTOMATIC_GOOD_FRACTION = 0.30
VIEWED_FRAMES_FILENAME = "viewed_frames.csv"
VIEWED_FRAME_FIELDS = ("file_id", "relative_path", "viewed_at")
PREVIEW_COLORMAPS = (
    "coolwarm",
    "hot",
    "inferno",
    "magma",
    "viridis",
    "plasma",
    "jet",
    "cividis",
)
PREVIEW_TRANSFORMS = ("robust_asinh", "linear")
PREVIEW_RANGE_MODES = ("auto", "fixed")
PREVIEW_ARCSEC_VIEW_LIMITS = (-3000.0, 3000.0)
PREVIEW_PERCENTILE_DEFAULTS = (50.0, 99.7)
PREVIEW_GLOBAL_SAMPLE_LIMIT = 1_000_000
PREVIEW_GLOBAL_FILE_SAMPLE_LIMIT = 4096
PREVIEW_RANGE_CACHE_VERSION = 1
PREVIEW_RANGE_CACHE_DIRNAME = "preview_range_cache"
_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
_FREQUENCY_DIR_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)?)MHz$", re.IGNORECASE)
_RAW_METRIC_FIELDS = (
    "p95",
    "p997",
    "baseline_p95",
    "baseline_p997",
    "p95_delta",
    "p997_delta",
    "bright_component_count",
    "largest_component_pixels",
    "largest_component_bbox_width",
    "largest_component_bbox_height",
    "largest_component_fill_fraction",
    "distributed_bright_fraction",
)
_SCIENCE_IDENTITY_FIELDS = frozenset(
    {
        "source_file",
        "frequency_mhz",
        "polarization",
        "file_index",
        "slot_index",
        "time",
        "legacy_quality_flag",
        "legacy_reason",
        "automatic_decision",
        "automatic_reasons",
        "hard_invalid",
        "rule_version",
    }
)
_CSV_FIELDS = (
    "candidate_id",
    "time",
    "frequency_mhz",
    "polarization",
    "relative_path",
    "source_file",
    "algorithm_flag",
    "algorithm_reason",
    "human_decision",
    "decision_source",
    "final_quality",
    "p95_delta",
    "p997_delta",
    "bright_component_count",
    "distributed_bright_fraction",
    "automatic_decision",
    "automatic_reasons",
    "automatic_rule_version",
    "selection_source",
    "sample_fingerprint",
    "quality_label",
    "event_tags",
    "artifact_tags",
    "reviewed_at",
    "ml_prediction",
)

_PREVIEW_FIGURE_FACE = "#f4f6f8"
_PREVIEW_TEXT_COLOR = "#13202b"
_PREVIEW_TICK_COLOR = "#243746"
_PREVIEW_CONTEXT_BORDER = "#43596b"
_PREVIEW_CANDIDATE_BORDER = "#007f73"
_PREVIEW_PLACEHOLDER_FACE = "#101820"
_PREVIEW_PLACEHOLDER_TEXT = "#f3f6f8"
_PREVIEW_WCS_WARNING = "#8b1d1d"
_ANGULAR_TO_ARCSEC = {
    "arcsec": 1.0,
    "arcsecond": 1.0,
    "arcseconds": 1.0,
    "asec": 1.0,
    "arcmin": 60.0,
    "arcminute": 60.0,
    "arcminutes": 60.0,
    "amin": 60.0,
    "deg": 3600.0,
    "degree": 3600.0,
    "degrees": 3600.0,
    "rad": 206264.80624709636,
    "radian": 206264.80624709636,
    "radians": 206264.80624709636,
}


@dataclass(frozen=True, slots=True)
class PreviewDisplaySettings:
    """Validated presentation-only settings for one preview triptych."""

    cmap: str = "coolwarm"
    transform: str = "robust_asinh"
    range_mode: str = "auto"
    pmin: float = PREVIEW_PERCENTILE_DEFAULTS[0]
    pmax: float = PREVIEW_PERCENTILE_DEFAULTS[1]
    vmin: float | None = None
    vmax: float | None = None

    def __post_init__(self) -> None:
        if self.cmap not in PREVIEW_COLORMAPS:
            raise ValueError("cmap must be one of: " + ", ".join(PREVIEW_COLORMAPS))
        if self.transform not in PREVIEW_TRANSFORMS:
            raise ValueError("transform must be robust_asinh or linear")
        if self.range_mode not in PREVIEW_RANGE_MODES:
            raise ValueError("range_mode must be auto or fixed")
        lower, upper = float(self.pmin), float(self.pmax)
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError("pmin and pmax must be finite")
        if not 0.0 <= lower < upper <= 100.0:
            raise ValueError("percentiles must satisfy 0 <= pmin < pmax <= 100")
        object.__setattr__(self, "pmin", lower)
        object.__setattr__(self, "pmax", upper)
        if (self.vmin is None) != (self.vmax is None):
            raise ValueError("legacy absolute range requires both vmin and vmax")
        if self.vmin is not None and self.vmax is not None:
            if not math.isfinite(float(self.vmin)) or not math.isfinite(
                float(self.vmax)
            ):
                raise ValueError("vmin and vmax must be finite")
            if float(self.vmin) >= float(self.vmax):
                raise ValueError("vmin must be less than vmax")
            object.__setattr__(self, "vmin", float(self.vmin))
            object.__setattr__(self, "vmax", float(self.vmax))

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | PreviewDisplaySettings | None
    ) -> PreviewDisplaySettings:
        if isinstance(value, cls):
            return value
        raw = value or {}
        range_mode = str(raw.get("range_mode") or "auto").strip().lower()
        pmin = _optional_finite_float(raw.get("pmin"), label="pmin")
        pmax = _optional_finite_float(raw.get("pmax"), label="pmax")
        vmin = _optional_finite_float(raw.get("vmin"), label="vmin")
        vmax = _optional_finite_float(raw.get("vmax"), label="vmax")
        return cls(
            cmap=str(raw.get("cmap") or "coolwarm").strip(),
            transform=str(raw.get("transform") or "robust_asinh").strip().lower(),
            range_mode=range_mode,
            pmin=(PREVIEW_PERCENTILE_DEFAULTS[0] if pmin is None else pmin),
            pmax=(PREVIEW_PERCENTILE_DEFAULTS[1] if pmax is None else pmax),
            vmin=vmin,
            vmax=vmax,
        )

    @property
    def has_legacy_absolute_range(self) -> bool:
        return self.vmin is not None and self.vmax is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _PreviewGeometry:
    extent_arcsec: tuple[float, float, float, float]
    origin: str


class StaleReviewError(RuntimeError):
    """Raised when a reviewed input changed after its scan."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _optional_finite_float(value: Any, *, label: str) -> float | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be a finite number")
    return numeric


def _transform_preview_data(data: Any, transform: str) -> tuple[Any, Any]:
    import numpy as np

    array = np.asarray(data, dtype=float)
    finite = np.isfinite(array)
    transformed = np.full(array.shape, np.nan, dtype=float)
    if not np.any(finite):
        return transformed, np.asarray([], dtype=float)
    finite_data = array[finite]
    if transform == "robust_asinh":
        center = float(np.median(finite_data))
        mad = float(np.median(np.abs(finite_data - center)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= np.finfo(float).eps:
            scale = float(np.std(finite_data))
        if not np.isfinite(scale) or scale <= np.finfo(float).eps:
            scale = 1.0
        transformed[finite] = np.arcsinh((finite_data - center) / scale)
    else:
        transformed[finite] = finite_data
    return transformed, transformed[finite]


def _preview_percentile_limits(
    values: Any, lower_percentile: float, upper_percentile: float
) -> tuple[float, float]:
    import numpy as np

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError("No finite intensity values are available for the preview")
    low, high = np.percentile(finite, [lower_percentile, upper_percentile])
    low, high = float(low), float(high)
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Preview percentile limits are not finite")
    if low >= high:
        delta = max(abs(low) * 1e-6, 1e-12)
        low, high = low - delta, high + delta
    return low, high


def _arcsec_factor(unit: Any, *, axis: str) -> float:
    normalized = str(unit or "").strip().lower()
    try:
        return _ANGULAR_TO_ARCSEC[normalized]
    except KeyError as exc:
        raise ValueError(f"{axis} WCS unit is not a supported angular unit") from exc


def _axis_aligned_wcs_header(header: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(header)
    cd_keys = ("CD1_1", "CD1_2", "CD2_1", "CD2_2")
    pc_keys = ("PC1_1", "PC1_2", "PC2_1", "PC2_2")
    has_cd = any(key in header for key in cd_keys)
    has_pc = any(key in header for key in pc_keys)
    if has_cd:
        if not all(key in header for key in cd_keys):
            raise ValueError("incomplete CD matrix")
        cd12 = float(header["CD1_2"])
        cd21 = float(header["CD2_1"])
        if not math.isclose(cd12, 0.0, abs_tol=1e-12) or not math.isclose(
            cd21, 0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "rotated CD matrix is not representable by an image extent"
            )
        normalized["CDELT1"] = float(header["CD1_1"])
        normalized["CDELT2"] = float(header["CD2_2"])
    elif has_pc:
        pc11 = float(header.get("PC1_1", 1.0))
        pc12 = float(header.get("PC1_2", 0.0))
        pc21 = float(header.get("PC2_1", 0.0))
        pc22 = float(header.get("PC2_2", 1.0))
        if not math.isclose(pc12, 0.0, abs_tol=1e-12) or not math.isclose(
            pc21, 0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "rotated PC matrix is not representable by an image extent"
            )
        normalized["CDELT1"] = float(header["CDELT1"]) * pc11
        normalized["CDELT2"] = float(header["CDELT2"]) * pc22
    elif not math.isclose(
        float(header.get("CROTA2", header.get("CROTA1", 0.0))),
        0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("rotated CROTA WCS is not representable by an image extent")
    return normalized


def _preview_geometry(
    header: Mapping[str, Any], image_shape: tuple[int, int]
) -> _PreviewGeometry:
    ctype1 = str(header.get("CTYPE1", "")).strip().upper()
    ctype2 = str(header.get("CTYPE2", "")).strip().upper()
    if not any(marker in ctype1 for marker in ("HPLN", "SOLX")):
        raise ValueError("CTYPE1 is not HPLN/SOLX")
    if not any(marker in ctype2 for marker in ("HPLT", "SOLY")):
        raise ValueError("CTYPE2 is not HPLT/SOLY")
    x_factor = _arcsec_factor(header.get("CUNIT1"), axis="X")
    y_factor = _arcsec_factor(header.get("CUNIT2"), axis="Y")
    normalized = _axis_aligned_wcs_header(header)
    native_extent = calculate_fits_extent_from_header(
        normalized,
        image_shape=image_shape,
        preserve_orientation=True,
    )
    extent = (
        float(native_extent[0]) * x_factor,
        float(native_extent[1]) * x_factor,
        float(native_extent[2]) * y_factor,
        float(native_extent[3]) * y_factor,
    )
    if not all(math.isfinite(value) for value in extent):
        raise ValueError("WCS extent is not finite")
    if extent[0] == extent[1] or extent[2] == extent[3]:
        raise ValueError("WCS extent is degenerate")
    return _PreviewGeometry(
        extent_arcsec=extent,
        origin=infer_image_origin_from_header(
            normalized,
            preserve_orientation=True,
        ),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(_json_safe(payload), indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_bad_frame_review(path: str | Path) -> dict[str, Any]:
    """Load and validate a versioned bad-frame review manifest."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Bad-frame review manifest must be a JSON object")
    try:
        schema_version = int(payload.get("schema_version", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("Unsupported bad-frame review schema version") from exc
    if schema_version not in SUPPORTED_BAD_FRAME_REVIEW_SCHEMA_VERSIONS:
        raise ValueError("Unsupported bad-frame review schema version")
    if payload.get("kind") != "radio-bad-frame-review":
        raise ValueError("Not a radio bad-frame review manifest")
    if payload.get("status") not in REVIEW_STATUSES:
        raise ValueError("Invalid bad-frame review status")
    if not isinstance(payload.get("candidates"), list):
        raise TypeError("Bad-frame review candidates must be a JSON array")
    if schema_version == 1:
        payload = _upgrade_v1_manifest(payload)
        schema_version = 2
    return _upgrade_v2_manifest(payload) if schema_version == 2 else payload


def _upgrade_v1_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize a legacy manifest in memory without rewriting its file."""

    upgraded = dict(payload)
    upgraded["schema_version"] = 2
    upgraded["source_schema_version"] = 1
    input_payload = dict(upgraded.get("input") or {})
    input_payload.setdefault("candidate_strategy", "rules")
    input_payload.setdefault("sample_count", DEFAULT_LABELING_SAMPLE_COUNT)
    input_payload.setdefault("minimum_good_fraction", MINIMUM_AUTOMATIC_GOOD_FRACTION)
    upgraded["input"] = input_payload
    upgraded.setdefault(
        "analysis",
        {
            "automatic_rule_version": "raw-quality-v1",
            "feature_schema_version": 0,
            "automatic_quality_csv": None,
        },
    )

    candidates = []
    for source in upgraded.get("candidates", []):
        candidate = dict(source)
        decision_source = candidate.get("decision_source")
        legacy_decision = candidate.get("human_decision")
        human_label = None
        if decision_source == "human" and legacy_decision in REVIEW_DECISIONS:
            human_label = {
                "quality_label": legacy_decision,
                "event_tags": [],
                "artifact_tags": [],
                "reviewed_at": upgraded.get("updated_at"),
                "source": "human",
                "legacy_schema_version": 1,
            }
        candidate.setdefault("automatic_decision", "bad_candidate")
        candidate.setdefault(
            "automatic_reasons",
            [str(candidate.get("algorithm_reason") or "legacy_bad")],
        )
        candidate.setdefault("automatic_rule_version", "raw-quality-v1")
        candidate.setdefault("selection_source", "automatic_bad")
        candidate.setdefault("sample_fingerprint", None)
        candidate.setdefault("features", dict(candidate.get("metrics") or {}))
        candidate.setdefault("ml_prediction", None)
        candidate["human_label"] = human_label
        candidates.append(candidate)
    upgraded["candidates"] = candidates
    upgraded.setdefault(
        "sampling",
        {
            "strategy": "rules",
            "target_sample_count": 0,
            "mandatory_count": len(candidates),
            "sampled_good_count": 0,
            "queue_count": len(candidates),
            "sampled_good_fraction": 0.0,
            "target_met": True,
            "seed": upgraded.get("input_fingerprint"),
        },
    )
    return upgraded


def _upgrade_v2_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize schema-2 reviews in memory without rewriting their files."""

    upgraded = dict(payload)
    upgraded["schema_version"] = BAD_FRAME_REVIEW_SCHEMA_VERSION
    upgraded.setdefault("source_schema_version", 2)
    input_payload = dict(upgraded.get("input") or {})
    input_payload.setdefault("review_scope", "candidates")
    upgraded["input"] = input_payload
    upgraded.setdefault(
        "audit",
        {
            "mode": "candidates",
            "progress_file": None,
            "viewed_frame_count": 0,
            "remaining_frame_count": 0,
            "coverage_fingerprint": None,
        },
    )
    summary = dict(upgraded.get("summary") or {})
    summary.setdefault("viewed_frame_count", 0)
    summary.setdefault("remaining_frame_count", 0)
    upgraded["summary"] = summary
    return upgraded


def extract_training_examples(
    review: dict[str, Any] | str | Path,
) -> list[dict[str, Any]]:
    """Return only completed, explicitly human-labelled training examples."""

    payload = (
        load_bad_frame_review(review) if isinstance(review, (str, Path)) else review
    )
    schema_version = int(payload.get("schema_version", 0))
    if schema_version == 1:
        payload = _upgrade_v1_manifest(payload)
        schema_version = 2
    if schema_version == 2:
        payload = _upgrade_v2_manifest(payload)
    if payload.get("status") != "completed":
        return []
    examples: list[dict[str, Any]] = []
    for candidate in payload.get("candidates", []):
        label = candidate.get("human_label")
        if not isinstance(label, dict) or label.get("source") != "human":
            continue
        quality_label = label.get("quality_label")
        if quality_label not in TRAINABLE_QUALITY_LABELS:
            continue
        examples.append(
            {
                "review_id": payload.get("review_id"),
                "candidate_id": candidate.get("candidate_id"),
                "file_id": candidate.get("file_id"),
                "source_file": candidate.get("source_file"),
                "relative_path": candidate.get("relative_path"),
                "sample_fingerprint": candidate.get("sample_fingerprint"),
                "frequency_mhz": candidate.get("frequency_mhz"),
                "polarization": candidate.get("polarization"),
                "time": candidate.get("time"),
                "slot_index": candidate.get("slot_index"),
                "quality_label": quality_label,
                "event_tags": list(label.get("event_tags") or []),
                "artifact_tags": list(label.get("artifact_tags") or []),
                "reviewed_at": label.get("reviewed_at"),
                "features": dict(candidate.get("features") or {}),
                "automatic_decision": candidate.get("automatic_decision"),
                "automatic_rule_version": candidate.get("automatic_rule_version"),
                "ml_prediction": candidate.get("ml_prediction"),
            }
        )
    return examples


def _validated_human_label(value: Any, *, reviewed_at: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("Each human label must be a JSON object")
    quality_label = str(value.get("quality_label", "")).strip().lower()
    if quality_label not in QUALITY_LABELS:
        raise ValueError("quality_label must be good, degraded, bad, or uncertain")
    event_tags = _validated_tags(value.get("event_tags", []), EVENT_TAGS, "event")
    artifact_tags = _validated_tags(
        value.get("artifact_tags", []), ARTIFACT_TAGS, "artifact"
    )
    return {
        "quality_label": quality_label,
        "event_tags": event_tags,
        "artifact_tags": artifact_tags,
        "reviewed_at": reviewed_at,
        "source": "human",
    }


def _validated_tags(value: Any, allowed: frozenset[str], label: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{label}_tags must be a JSON array")
    normalized = sorted({str(item).strip().lower() for item in value})
    invalid = [item for item in normalized if item not in allowed]
    if invalid:
        raise ValueError(f"Unsupported {label} tag: {invalid[0]}")
    return normalized


def _numeric_bin(value: Any, boundaries: tuple[float, ...]) -> int:
    try:
        number = float(value)
    except TypeError, ValueError:
        return -1
    if not math.isfinite(number):
        return -1
    return sum(number >= boundary for boundary in boundaries)


def final_bad_frame_paths(review: dict[str, Any] | str | Path) -> tuple[Path, ...]:
    """Return final rejected files from a completed or explicitly skipped review."""

    payload = (
        load_bad_frame_review(review) if isinstance(review, (str, Path)) else review
    )
    if payload.get("status") not in {"completed", "skipped"}:
        raise ValueError("Bad-frame review must be completed or skipped")
    paths = payload.get("final_bad_files", [])
    if not isinstance(paths, list):
        raise TypeError("final_bad_files must be a JSON array")
    return tuple(Path(str(item)) for item in paths)


class BadFrameReviewStore:
    """Manage standalone review records beneath an application-owned output root."""

    def __init__(
        self,
        output_root: str | Path,
        allowed_roots: list[str | Path] | tuple[str | Path, ...],
        *,
        shadow_predictor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        shadow_model_id: str | None = None,
    ) -> None:
        self.output_root = Path(output_root).expanduser().resolve(strict=False)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.allowed_roots = tuple(
            Path(root).expanduser().resolve(strict=False) for root in allowed_roots
        )
        if not self.allowed_roots:
            raise ValueError("At least one allowed root is required")
        self.shadow_predictor = shadow_predictor
        self.shadow_model_id = shadow_model_id
        self._audit_lock = threading.RLock()
        self._viewed_cache: dict[str, set[str]] = {}
        self._preview_range_locks_guard = threading.RLock()
        self._preview_range_locks: dict[tuple[str, str, str], threading.RLock] = {}

    def resolve_input(self, value: str | Path, *, directory: bool = False) -> Path:
        path = Path(value).expanduser().resolve(strict=True)
        if not any(path == root or root in path.parents for root in self.allowed_roots):
            raise PermissionError("Path is outside the configured allowed roots")
        if directory and not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        return path

    def list_directories(self, value: str | Path | None = None) -> dict[str, Any]:
        if value in (None, ""):
            return {
                "path": None,
                "parent": None,
                "directories": [
                    {"name": root.name or str(root), "path": str(root)}
                    for root in self.allowed_roots
                    if root.is_dir()
                ],
            }
        path = self.resolve_input(str(value), directory=True)
        containing_root = next(
            root for root in self.allowed_roots if path == root or root in path.parents
        )
        parent = str(path.parent) if path != containing_root else None
        directories = []
        for item in sorted(
            path.iterdir(), key=lambda candidate: candidate.name.casefold()
        ):
            try:
                if item.is_dir() and not item.is_symlink():
                    directories.append({"name": item.name, "path": str(item.resolve())})
            except OSError:
                continue
        return {"path": str(path), "parent": parent, "directories": directories}

    def discover(self, value: str | Path) -> dict[str, Any]:
        root = self.resolve_input(value, directory=True)
        bands: list[dict[str, Any]] = []
        for band_dir in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
            if not band_dir.is_dir() or band_dir.is_symlink():
                continue
            match = _FREQUENCY_DIR_RE.fullmatch(band_dir.name)
            if not match:
                continue
            polarizations: list[dict[str, Any]] = []
            for polarization in ("RR", "LL"):
                folder = band_dir / polarization
                if folder.is_dir() and not folder.is_symlink():
                    polarizations.append(
                        {
                            "name": polarization,
                            "file_count": sum(1 for _ in folder.glob("*.fits")),
                        }
                    )
            if polarizations:
                bands.append(
                    {
                        "frequency_mhz": float(match.group(1)),
                        "label": f"{_format_frequency(float(match.group(1)))} MHz",
                        "polarizations": polarizations,
                    }
                )
        bands.sort(key=lambda item: item["frequency_mhz"])
        return {"root": str(root), "bands": bands}

    def create_review(self, payload: dict[str, Any]) -> dict[str, Any]:
        root = self.resolve_input(str(payload.get("root", "")), directory=True)
        discovery = self.discover(root)
        discovered = {
            (float(band["frequency_mhz"]), pol["name"])
            for band in discovery["bands"]
            for pol in band["polarizations"]
        }
        frequencies = self._frequencies(payload.get("frequencies_mhz"), discovery)
        polarizations = self._polarizations(payload.get("polarizations"), discovery)
        selected_pairs = {
            (frequency, polarization)
            for frequency in frequencies
            for polarization in polarizations
        }
        missing = selected_pairs - discovered
        if missing:
            labels = ", ".join(
                f"{_format_frequency(freq)}MHz/{pol}" for freq, pol in sorted(missing)
            )
            raise ValueError(f"Selected radio directories do not exist: {labels}")

        start_index = self._index(payload.get("start_index"), default=0)
        end_index = self._optional_index(payload.get("end_index"))
        if end_index is not None and end_index <= start_index:
            raise ValueError("end_index must be greater than start_index")
        candidate_strategy = str(
            payload.get("candidate_strategy", "rules") or "rules"
        ).strip()
        if candidate_strategy not in REVIEW_STRATEGIES:
            raise ValueError(
                "candidate_strategy must be 'rules', 'labeling', or 'shadow'"
            )
        review_scope = str(
            payload.get("review_scope", "candidates") or "candidates"
        ).strip()
        if review_scope not in REVIEW_SCOPES:
            raise ValueError("review_scope must be 'candidates' or 'all_scanned'")
        sample_count = int(
            payload.get("sample_count", DEFAULT_LABELING_SAMPLE_COUNT)
            or DEFAULT_LABELING_SAMPLE_COUNT
        )
        if sample_count <= 0:
            raise ValueError("sample_count must be a positive integer")

        review_id = uuid.uuid4().hex
        staging = self.output_root / f".{review_id}.{uuid.uuid4().hex}.tmp"
        final_dir = self._review_dir(review_id, must_exist=False)
        if final_dir.exists():
            raise FileExistsError(f"Review already exists: {review_id}")
        staging.mkdir(parents=True)
        try:
            result = analyze_radio_raw_quality(
                root,
                freqs=frequencies,
                polarizations=polarizations,
                output_dir=staging,
                start_idx=start_index,
                end_idx=end_index,
            )
            diagnostics = staging / "raw_quality_diagnostics"
            for source in (result.file_csv_path, result.slot_csv_path):
                source.replace(staging / source.name)
            diagnostics.rmdir()

            file_records = self._file_records(root, result.file_rows)
            if not file_records:
                raise ValueError("No FITS files exist in the selected frame range")
            file_by_path = {record["source_file"]: record for record in file_records}
            raw_by_path = {
                str(Path(row.source_file).resolve(strict=True)): row
                for row in result.file_rows
            }
            science_rows = []
            automatic_quality_csv = None
            if candidate_strategy != "rules":
                science_rows = analyze_radio_science_quality(result.file_rows)
                automatic_quality_csv = write_radio_science_quality_csv(
                    staging / "automatic_quality.csv", science_rows
                )
            input_fingerprint = self._fingerprint(
                file_records,
                payload={
                    "frequencies_mhz": frequencies,
                    "polarizations": polarizations,
                    "start_index": start_index,
                    "end_index": end_index,
                    "candidate_strategy": candidate_strategy,
                    "review_scope": review_scope,
                    "sample_count": sample_count,
                },
            )
            if candidate_strategy == "rules":
                analysis_pool = [
                    self._raw_candidate(row, file_by_path) for row in result.file_rows
                ]
            else:
                analysis_pool = [
                    self._science_candidate(row, raw_by_path, file_by_path)
                    for row in science_rows
                ]
            analysis_warnings: list[str] = []
            if candidate_strategy != "rules" and self.shadow_predictor is not None:
                for candidate in analysis_pool:
                    try:
                        prediction = self.shadow_predictor(candidate)
                        if not isinstance(prediction, dict):
                            raise TypeError("shadow predictor returned a non-object")
                        candidate["ml_prediction"] = _json_safe(prediction)
                    except Exception as exc:  # noqa: BLE001 - rules remain available
                        if len(analysis_warnings) < 5:
                            analysis_warnings.append(
                                f"model_fallback:{type(exc).__name__}:{exc}"
                            )
                        candidate["ml_prediction"] = None
            elif candidate_strategy == "shadow":
                analysis_warnings.append(
                    "model_fallback:no verified published model is available"
                )
            self._attach_file_analysis(file_records, analysis_pool)
            review_pool = (
                [
                    candidate
                    for candidate in analysis_pool
                    if candidate["automatic_decision"] == "bad_candidate"
                ]
                if candidate_strategy == "rules"
                else analysis_pool
            )
            candidates, sampling = self._build_review_queue(
                review_pool,
                strategy=candidate_strategy,
                sample_count=sample_count,
                seed=input_fingerprint,
            )
            for candidate in candidates:
                candidate["sample_fingerprint"] = self._sample_fingerprint(
                    candidate["source_file"]
                )
            candidates.sort(key=self._candidate_sort_key)
            self._attach_context(candidates, file_records)
            now = _utc_now()
            has_pending_review = bool(candidates) or review_scope == "all_scanned"
            manifest = {
                "schema_version": BAD_FRAME_REVIEW_SCHEMA_VERSION,
                "kind": "radio-bad-frame-review",
                "review_id": review_id,
                "status": "draft" if has_pending_review else "completed",
                "created_at": now,
                "updated_at": now,
                "completed_at": None if has_pending_review else now,
                "input": {
                    "root": str(root),
                    "frequencies_mhz": frequencies,
                    "polarizations": polarizations,
                    "start_index": start_index,
                    "end_index": end_index,
                    "thresholds": asdict(RawQualityThresholds()),
                    "candidate_strategy": candidate_strategy,
                    "review_scope": review_scope,
                    "sample_count": sample_count,
                    "minimum_good_fraction": MINIMUM_AUTOMATIC_GOOD_FRACTION,
                    "model_id": self.shadow_model_id,
                },
                "analysis": {
                    "automatic_rule_version": (
                        "raw-quality-v1"
                        if candidate_strategy == "rules"
                        else AUTOMATIC_QUALITY_RULE_VERSION
                    ),
                    "feature_schema_version": (
                        0 if candidate_strategy == "rules" else 1
                    ),
                    "automatic_quality_csv": (
                        automatic_quality_csv.name if automatic_quality_csv else None
                    ),
                    "raw_quality_csv": result.file_csv_path.name,
                    "slot_quality_csv": result.slot_csv_path.name,
                    "shadow_model_id": self.shadow_model_id,
                    "warnings": analysis_warnings,
                },
                "sampling": sampling,
                "input_fingerprint": input_fingerprint,
                "files": file_records,
                "candidates": candidates,
                "final_bad_files": [],
                "audit": {
                    "mode": review_scope,
                    "progress_file": (
                        VIEWED_FRAMES_FILENAME
                        if review_scope == "all_scanned"
                        else None
                    ),
                    "viewed_frame_count": 0,
                    "remaining_frame_count": (
                        len(file_records) if review_scope == "all_scanned" else 0
                    ),
                    "coverage_fingerprint": None,
                },
                "summary": {},
            }
            self._refresh_derived(manifest)
            if review_scope == "all_scanned":
                self._write_viewed_header(staging / VIEWED_FRAMES_FILENAME)
            _atomic_json(staging / "review.json", manifest)
            self._write_candidate_csv(staging / "candidates.csv", manifest)
            staging.replace(final_dir)
            return manifest
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def list_reviews(self) -> list[dict[str, Any]]:
        summaries = []
        for folder in self.output_root.iterdir():
            if not folder.is_dir() or not _IDENTIFIER_RE.fullmatch(folder.name):
                continue
            try:
                manifest = load_bad_frame_review(folder / "review.json")
            except OSError, TypeError, ValueError, json.JSONDecodeError:
                continue
            summaries.append(self.public_payload(manifest, include_files=False))
        return sorted(summaries, key=lambda item: item["updated_at"], reverse=True)

    def load_review(self, review_id: str) -> dict[str, Any]:
        path = self._review_dir(review_id) / "review.json"
        if not path.is_file() or path.is_symlink():
            raise KeyError(f"Unknown bad-frame review: {review_id}")
        manifest = load_bad_frame_review(path)
        if manifest.get("review_id") != review_id:
            raise ValueError("Review id does not match its storage directory")
        return manifest

    def public_payload(
        self, manifest: dict[str, Any], *, include_files: bool = False
    ) -> dict[str, Any]:
        payload = dict(manifest)
        summary = dict(payload.get("summary") or {})
        audit = dict(payload.get("audit") or {})
        if self._review_scope(payload) == "all_scanned":
            viewed_ids = self._viewed_ids(str(payload["review_id"]))
            file_ids = {str(item["file_id"]) for item in payload.get("files", [])}
            viewed_count = len(viewed_ids & file_ids)
            remaining_count = max(len(file_ids) - viewed_count, 0)
            summary["viewed_frame_count"] = viewed_count
            summary["remaining_frame_count"] = remaining_count
            audit["viewed_frame_count"] = viewed_count
            audit["remaining_frame_count"] = remaining_count
        else:
            summary.setdefault("viewed_frame_count", 0)
            summary.setdefault("remaining_frame_count", 0)
        payload["summary"] = summary
        payload["audit"] = audit
        if not include_files:
            payload.pop("files", None)
        return payload

    @staticmethod
    def _review_scope(manifest: dict[str, Any]) -> str:
        value = str((manifest.get("input") or {}).get("review_scope", "candidates"))
        return value if value in REVIEW_SCOPES else "candidates"

    def update_decisions(
        self, review_id: str, decisions: dict[str, Any]
    ) -> dict[str, Any]:
        """Compatibility adapter for the legacy Bad/Keep API."""

        labels = {}
        for candidate_id, decision in decisions.items():
            if decision not in REVIEW_DECISIONS:
                raise ValueError("Review decisions must be 'bad' or 'good'")
            labels[candidate_id] = {
                "quality_label": decision,
                "event_tags": [],
                "artifact_tags": [],
            }
        return self.update_labels(review_id, labels)

    def update_labels(self, review_id: str, labels: dict[str, Any]) -> dict[str, Any]:
        """Save explicit human quality, event, and artifact labels."""

        if not isinstance(labels, dict) or not labels:
            raise ValueError("labels must be a non-empty object")
        manifest = self.load_review(review_id)
        if manifest["status"] != "draft":
            raise ValueError("Finalized reviews are read-only; create a new review")
        candidates = {item["candidate_id"]: item for item in manifest["candidates"]}
        unknown = [
            candidate_id for candidate_id in labels if candidate_id not in candidates
        ]
        if unknown:
            raise KeyError(f"Unknown bad-frame candidate: {unknown[0]}")
        self._assert_fresh(
            manifest,
            file_ids={candidates[candidate_id]["file_id"] for candidate_id in labels},
        )
        reviewed_at = _utc_now()
        for candidate_id, label in labels.items():
            candidates[candidate_id]["human_label"] = _validated_human_label(
                label, reviewed_at=reviewed_at
            )
        manifest["updated_at"] = reviewed_at
        self._refresh_derived(manifest)
        self._save(manifest)
        return manifest

    def list_frames(
        self, review_id: str, *, offset: int = 0, limit: int = 100
    ) -> dict[str, Any]:
        """Return one bounded page of frames for a full visual audit."""

        manifest = self.load_review(review_id)
        if self._review_scope(manifest) != "all_scanned":
            raise ValueError("This review was not created for all-frame browsing")
        offset = int(offset)
        limit = int(limit)
        if offset < 0:
            raise ValueError("offset cannot be negative")
        if limit < 1 or limit > 200:
            raise ValueError("limit must be between 1 and 200")
        files = self._ordered_files(manifest)
        viewed = self._viewed_ids(review_id)
        candidates = {
            str(candidate["file_id"]): candidate
            for candidate in manifest.get("candidates", [])
        }
        page = []
        for index, file_record in enumerate(files[offset : offset + limit], offset):
            page.append(
                self._public_frame(
                    file_record,
                    candidate=candidates.get(str(file_record["file_id"])),
                    viewed=str(file_record["file_id"]) in viewed,
                    index=index,
                )
            )
        first_unviewed_index = next(
            (
                index
                for index, item in enumerate(files)
                if str(item["file_id"]) not in viewed
            ),
            None,
        )
        return {
            "offset": offset,
            "limit": limit,
            "total": len(files),
            "first_unviewed_index": first_unviewed_index,
            "frames": page,
        }

    def mark_frame_viewed(self, review_id: str, file_id: str) -> dict[str, Any]:
        """Persist an idempotent visual-audit acknowledgement for one frame."""

        manifest = self.load_review(review_id)
        if manifest["status"] != "draft":
            raise ValueError("Finalized reviews are read-only; create a new review")
        if self._review_scope(manifest) != "all_scanned":
            raise ValueError("This review was not created for all-frame browsing")
        frame = self._file_for_id(manifest, file_id)
        self._assert_fresh(manifest, file_ids={str(frame["file_id"])})
        self._append_viewed_frame(review_id, frame)
        return manifest

    def update_frame_label(
        self, review_id: str, file_id: str, label: dict[str, Any]
    ) -> dict[str, Any]:
        """Save an explicit label for any frame in an all-frame review."""

        manifest = self.load_review(review_id)
        if manifest["status"] != "draft":
            raise ValueError("Finalized reviews are read-only; create a new review")
        if self._review_scope(manifest) != "all_scanned":
            raise ValueError("This review was not created for all-frame browsing")
        frame = self._file_for_id(manifest, file_id)
        self._assert_fresh(manifest, file_ids={str(frame["file_id"])})
        candidate = next(
            (
                item
                for item in manifest["candidates"]
                if item["file_id"] == frame["file_id"]
            ),
            None,
        )
        if candidate is None:
            candidate = self._candidate_from_file(manifest, frame)
            candidate["selection_source"] = "manual_full_review"
            candidate["sample_fingerprint"] = self._sample_fingerprint(
                candidate["source_file"]
            )
            self._attach_context([candidate], manifest["files"])
            manifest["candidates"].append(candidate)
            manifest["candidates"].sort(key=self._candidate_sort_key)
        reviewed_at = _utc_now()
        candidate["human_label"] = _validated_human_label(
            label, reviewed_at=reviewed_at
        )
        manifest["updated_at"] = reviewed_at
        self._append_viewed_frame(review_id, frame)
        self._refresh_derived(manifest)
        self._save(manifest)
        return manifest

    def finalize(self, review_id: str, mode: str) -> dict[str, Any]:
        manifest = self.load_review(review_id)
        if manifest["status"] != "draft":
            raise ValueError("Finalized reviews are read-only; create a new review")
        if mode not in {"completed", "skipped"}:
            raise ValueError("Finalize mode must be 'completed' or 'skipped'")
        self._assert_fresh(manifest)
        if mode == "completed":
            pending = [
                item["candidate_id"]
                for item in manifest["candidates"]
                if not isinstance(item.get("human_label"), dict)
                or item["human_label"].get("quality_label")
                not in TRAINABLE_QUALITY_LABELS
            ]
            if pending:
                raise ValueError(
                    "Every candidate requires a Good, Degraded, or Bad decision; "
                    "Uncertain labels must be resolved"
                )
            if self._review_scope(manifest) == "all_scanned":
                viewed = self._viewed_ids(review_id)
                ordered_ids = [
                    str(item["file_id"]) for item in self._ordered_files(manifest)
                ]
                remaining = [
                    file_id for file_id in ordered_ids if file_id not in viewed
                ]
                if remaining:
                    raise ValueError(
                        f"Every scanned frame must be viewed; {len(remaining)} remain"
                    )
                manifest["audit"] = {
                    **dict(manifest.get("audit") or {}),
                    "viewed_frame_count": len(ordered_ids),
                    "remaining_frame_count": 0,
                    "coverage_fingerprint": "sha256:"
                    + hashlib.sha256(
                        "\n".join(ordered_ids).encode("utf-8")
                    ).hexdigest(),
                }
        if self._review_scope(manifest) == "all_scanned" and mode == "skipped":
            viewed = self._viewed_ids(review_id)
            total = len(manifest.get("files", []))
            manifest["audit"] = {
                **dict(manifest.get("audit") or {}),
                "viewed_frame_count": min(len(viewed), total),
                "remaining_frame_count": max(total - len(viewed), 0),
            }
        manifest["status"] = mode
        manifest["updated_at"] = _utc_now()
        manifest["completed_at"] = manifest["updated_at"]
        self._refresh_derived(manifest)
        self._save(manifest)
        return manifest

    def render_candidate_preview(
        self,
        review_id: str,
        candidate_id: str,
        display: Mapping[str, Any] | PreviewDisplaySettings | None = None,
    ) -> bytes:
        manifest = self.load_review(review_id)
        candidate = next(
            (
                item
                for item in manifest["candidates"]
                if item["candidate_id"] == candidate_id
            ),
            None,
        )
        if candidate is None:
            raise KeyError(f"Unknown bad-frame candidate: {candidate_id}")
        context_file_ids = {
            file_id for file_id in candidate["context_file_ids"] if file_id
        }
        self._assert_fresh(manifest, file_ids=context_file_ids)
        files = {item["file_id"]: item for item in manifest["files"]}
        context = [
            files.get(file_id) if file_id else None
            for file_id in candidate["context_file_ids"]
        ]
        return self._render_triptych(
            manifest,
            candidate,
            context,
            display=PreviewDisplaySettings.from_mapping(display),
        )

    def render_frame_preview(
        self,
        review_id: str,
        file_id: str,
        display: Mapping[str, Any] | PreviewDisplaySettings | None = None,
    ) -> bytes:
        manifest = self.load_review(review_id)
        if self._review_scope(manifest) != "all_scanned":
            raise ValueError("This review was not created for all-frame browsing")
        frame = self._file_for_id(manifest, file_id)
        context = self._frame_context(manifest, frame)
        self._assert_fresh(
            manifest,
            file_ids={str(item["file_id"]) for item in context if item is not None},
        )
        return self._render_triptych(
            manifest,
            frame,
            context,
            display=PreviewDisplaySettings.from_mapping(display),
        )

    @staticmethod
    def _candidate_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
        return (
            item.get("time") or "",
            int(item.get("file_index", 0)),
            float(item.get("frequency_mhz", 0.0)),
            str(item.get("polarization", "")),
            str(item.get("relative_path", "")),
        )

    @staticmethod
    def _ordered_files(manifest: dict[str, Any]) -> list[dict[str, Any]]:
        return sorted(
            manifest.get("files", []),
            key=lambda item: (
                int(item.get("slot_index", 0)),
                float(item.get("frequency_mhz", 0.0)),
                str(item.get("polarization", "")),
                str(item.get("relative_path", "")),
            ),
        )

    @staticmethod
    def _public_frame(
        file_record: dict[str, Any],
        *,
        candidate: dict[str, Any] | None,
        viewed: bool,
        index: int,
    ) -> dict[str, Any]:
        source = candidate or file_record
        return {
            "index": index,
            "ordinal": index + 1,
            "file_id": file_record["file_id"],
            "relative_path": file_record["relative_path"],
            "frequency_mhz": file_record["frequency_mhz"],
            "polarization": file_record["polarization"],
            "file_index": file_record["file_index"],
            "slot_index": file_record["slot_index"],
            "time": file_record.get("time", ""),
            "viewed": viewed,
            "candidate_id": candidate.get("candidate_id") if candidate else None,
            "automatic_decision": source.get("automatic_decision"),
            "automatic_reasons": list(source.get("automatic_reasons") or []),
            "automatic_rule_version": source.get("automatic_rule_version"),
            "human_label": candidate.get("human_label") if candidate else None,
            "ml_prediction": candidate.get("ml_prediction") if candidate else None,
        }

    @staticmethod
    def _file_for_id(manifest: dict[str, Any], file_id: str) -> dict[str, Any]:
        normalized = str(file_id).strip()
        frame = next(
            (
                item
                for item in manifest.get("files", [])
                if str(item.get("file_id")) == normalized
            ),
            None,
        )
        if frame is None:
            raise KeyError(f"Unknown bad-frame review file: {normalized}")
        return frame

    @staticmethod
    def _frame_context(
        manifest: dict[str, Any], frame: dict[str, Any]
    ) -> list[dict[str, Any] | None]:
        channel = sorted(
            (
                item
                for item in manifest.get("files", [])
                if float(item["frequency_mhz"]) == float(frame["frequency_mhz"])
                and str(item["polarization"]) == str(frame["polarization"])
            ),
            key=lambda item: (int(item["file_index"]), str(item["relative_path"])),
        )
        position = next(
            index
            for index, item in enumerate(channel)
            if item["file_id"] == frame["file_id"]
        )
        return [
            channel[position - 1] if position > 0 else None,
            channel[position],
            channel[position + 1] if position + 1 < len(channel) else None,
        ]

    @staticmethod
    def _write_viewed_header(path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=VIEWED_FRAME_FIELDS)
            writer.writeheader()

    def _viewed_ids(self, review_id: str) -> set[str]:
        with self._audit_lock:
            if review_id not in self._viewed_cache:
                path = self._review_dir(review_id) / VIEWED_FRAMES_FILENAME
                viewed: set[str] = set()
                if path.is_file() and not path.is_symlink():
                    with path.open(newline="", encoding="utf-8") as handle:
                        for row in csv.DictReader(handle):
                            file_id = str(row.get("file_id") or "").strip()
                            if file_id:
                                viewed.add(file_id)
                self._viewed_cache[review_id] = viewed
            return set(self._viewed_cache[review_id])

    def _append_viewed_frame(self, review_id: str, frame: dict[str, Any]) -> None:
        file_id = str(frame["file_id"])
        with self._audit_lock:
            viewed = self._viewed_ids(review_id)
            if file_id in viewed:
                return
            path = self._review_dir(review_id) / VIEWED_FRAMES_FILENAME
            if not path.is_file():
                self._write_viewed_header(path)
            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=VIEWED_FRAME_FIELDS)
                writer.writerow(
                    {
                        "file_id": file_id,
                        "relative_path": frame["relative_path"],
                        "viewed_at": _utc_now(),
                    }
                )
            self._viewed_cache.setdefault(review_id, set()).add(file_id)

    def _candidate_from_file(
        self, manifest: dict[str, Any], frame: dict[str, Any]
    ) -> dict[str, Any]:
        folder = self._review_dir(str(manifest["review_id"]))
        analysis = dict(manifest.get("analysis") or {})
        raw_row = self._find_csv_row(
            folder / str(analysis.get("raw_quality_csv") or ""),
            str(frame["source_file"]),
        )
        science_name = analysis.get("automatic_quality_csv")
        science_row = (
            self._find_csv_row(folder / str(science_name), str(frame["source_file"]))
            if science_name
            else None
        )
        features = {
            key: self._csv_scalar(value)
            for key, value in (science_row or {}).items()
            if key not in _SCIENCE_IDENTITY_FIELDS
        }
        metrics = {
            key: self._csv_scalar((raw_row or {}).get(key))
            for key in _RAW_METRIC_FIELDS
        }
        automatic_decision = str(
            (science_row or {}).get("automatic_decision")
            or frame.get("automatic_decision")
            or "good_candidate"
        )
        automatic_reasons = [
            item
            for item in str(
                (science_row or {}).get("automatic_reasons")
                or ";".join(frame.get("automatic_reasons") or [])
                or (raw_row or {}).get("reason")
                or "manual_full_review"
            ).split(";")
            if item
        ]
        candidate = {
            "candidate_id": f"candidate-{str(frame['file_id']).split('-', 1)[-1]}",
            "file_id": frame["file_id"],
            "source_file": frame["source_file"],
            "relative_path": frame["relative_path"],
            "frequency_mhz": float(frame["frequency_mhz"]),
            "polarization": str(frame["polarization"]),
            "file_index": int(frame["file_index"]),
            "slot_index": int(frame["slot_index"]),
            "time": str(frame.get("time") or ""),
            "algorithm_flag": str((raw_row or {}).get("quality_flag") or "ok"),
            "algorithm_reason": str((raw_row or {}).get("reason") or "ok"),
            "metrics": metrics,
            "features": features,
            "automatic_decision": automatic_decision,
            "automatic_reasons": automatic_reasons,
            "automatic_rule_version": str(
                (science_row or {}).get("rule_version")
                or frame.get("automatic_rule_version")
                or "raw-quality-v1"
            ),
            "selection_source": "manual_full_review",
            "sample_fingerprint": None,
            "ml_prediction": None,
            "human_label": None,
            "human_decision": None,
            "decision_source": None,
            "final_quality": None,
        }
        if self.shadow_predictor is not None and science_row is not None:
            try:
                candidate["ml_prediction"] = _json_safe(
                    self.shadow_predictor(candidate)
                )
            except Exception:  # noqa: BLE001 - human review remains available
                candidate["ml_prediction"] = None
        return candidate

    @staticmethod
    def _find_csv_row(path: Path, source_file: str) -> dict[str, str] | None:
        if not path.is_file() or path.is_symlink():
            return None
        expected = os.path.normcase(str(Path(source_file).resolve(strict=False)))
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                actual = os.path.normcase(
                    str(Path(str(row.get("source_file") or "")).resolve(strict=False))
                )
                if actual == expected:
                    return dict(row)
        return None

    @staticmethod
    def _csv_scalar(value: Any) -> Any:
        if value in (None, ""):
            return None
        text = str(value).strip()
        if text.casefold() in {"true", "false"}:
            return text.casefold() == "true"
        try:
            number = float(text)
        except ValueError:
            return text
        return int(number) if number.is_integer() else number

    def download_path(self, review_id: str, filename: str) -> Path:
        if filename not in {"review.json", "candidates.csv", VIEWED_FRAMES_FILENAME}:
            raise KeyError(f"Unknown bad-frame review download: {filename}")
        path = self._review_dir(review_id) / filename
        if not path.is_file() or path.is_symlink():
            raise KeyError(f"Unknown bad-frame review download: {filename}")
        return path.resolve(strict=True)

    def _save(self, manifest: dict[str, Any]) -> None:
        folder = self._review_dir(manifest["review_id"])
        _atomic_json(folder / "review.json", manifest)
        self._write_candidate_csv(folder / "candidates.csv", manifest)

    def _review_dir(self, review_id: str, *, must_exist: bool = True) -> Path:
        normalized = str(review_id).strip()
        if not _IDENTIFIER_RE.fullmatch(normalized):
            raise ValueError("Invalid bad-frame review id")
        path = self.output_root / normalized
        if must_exist and (not path.is_dir() or path.is_symlink()):
            raise KeyError(f"Unknown bad-frame review: {normalized}")
        return path

    @staticmethod
    def _index(value: Any, *, default: int) -> int:
        if value in (None, ""):
            return default
        number = int(value)
        if number < 0:
            raise ValueError("Frame indices cannot be negative")
        return number

    def _optional_index(self, value: Any) -> int | None:
        return None if value in (None, "") else self._index(value, default=0)

    @staticmethod
    def _frequencies(value: Any, discovery: dict[str, Any]) -> list[float]:
        raw = (
            value
            if value not in (None, [])
            else [band["frequency_mhz"] for band in discovery["bands"]]
        )
        if not isinstance(raw, list):
            raise TypeError("frequencies_mhz must be a JSON array")
        frequencies = sorted({float(item) for item in raw})
        if not frequencies or any(
            not math.isfinite(item) or item <= 0 for item in frequencies
        ):
            raise ValueError("Select at least one positive finite frequency")
        return frequencies

    @staticmethod
    def _polarizations(value: Any, discovery: dict[str, Any]) -> list[str]:
        discovered = {
            pol["name"] for band in discovery["bands"] for pol in band["polarizations"]
        }
        raw = value if value not in (None, []) else sorted(discovered)
        if not isinstance(raw, list):
            raise TypeError("polarizations must be a JSON array")
        polarizations = sorted({str(item).upper() for item in raw})
        if not polarizations or any(item not in {"RR", "LL"} for item in polarizations):
            raise ValueError("Select RR, LL, or both polarizations")
        return polarizations

    def _file_records(self, root: Path, rows: list[Any]) -> list[dict[str, Any]]:
        records = []
        seen = set()
        for row in rows:
            path = self.resolve_input(row.source_file)
            key = os.path.normcase(str(path))
            if key in seen:
                continue
            seen.add(key)
            stat = path.stat()
            relative = path.relative_to(root).as_posix()
            digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:20]
            records.append(
                {
                    "file_id": f"file-{digest}",
                    "source_file": str(path),
                    "relative_path": relative,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "frequency_mhz": float(row.frequency_mhz),
                    "polarization": str(row.polarization),
                    "file_index": int(row.file_index),
                    "slot_index": int(row.slot_index),
                    "time": str(row.time),
                }
            )
        return records

    @staticmethod
    def _attach_file_analysis(
        file_records: list[dict[str, Any]], candidates: list[dict[str, Any]]
    ) -> None:
        by_file_id = {candidate["file_id"]: candidate for candidate in candidates}
        for record in file_records:
            candidate = by_file_id.get(record["file_id"])
            if candidate is None:
                continue
            record["automatic_decision"] = candidate.get("automatic_decision")
            record["automatic_reasons"] = list(candidate.get("automatic_reasons") or [])
            record["automatic_rule_version"] = candidate.get("automatic_rule_version")

    @staticmethod
    def _raw_candidate(
        row: Any, file_by_path: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        source_file = str(Path(row.source_file).resolve(strict=True))
        file_record = file_by_path[source_file]
        row_payload = _json_safe(asdict(row))
        automatic_decision = (
            "bad_candidate" if str(row.quality_flag) == "bad" else "good_candidate"
        )
        return {
            "candidate_id": f"candidate-{file_record['file_id'].split('-', 1)[1]}",
            "file_id": file_record["file_id"],
            "source_file": source_file,
            "relative_path": file_record["relative_path"],
            "frequency_mhz": float(row.frequency_mhz),
            "polarization": str(row.polarization),
            "file_index": int(row.file_index),
            "slot_index": int(row.slot_index),
            "time": str(row.time),
            "algorithm_flag": str(row.quality_flag),
            "algorithm_reason": str(row.reason),
            "metrics": {key: row_payload.get(key) for key in _RAW_METRIC_FIELDS},
            "features": {},
            "automatic_decision": automatic_decision,
            "automatic_reasons": [str(row.reason)],
            "automatic_rule_version": "raw-quality-v1",
            "selection_source": None,
            "sample_fingerprint": None,
            "ml_prediction": None,
            "human_label": None,
            "human_decision": None,
            "decision_source": None,
            "final_quality": None,
        }

    @classmethod
    def _science_candidate(
        cls,
        row: Any,
        raw_by_path: dict[str, Any],
        file_by_path: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        source_file = str(Path(row.source_file).resolve(strict=True))
        raw = raw_by_path[source_file]
        candidate = cls._raw_candidate(raw, file_by_path)
        science_payload = _json_safe(asdict(row))
        candidate.update(
            {
                "automatic_decision": str(row.automatic_decision),
                "automatic_reasons": [
                    item for item in str(row.automatic_reasons).split(";") if item
                ],
                "automatic_rule_version": str(row.rule_version),
                "features": {
                    key: value
                    for key, value in science_payload.items()
                    if key not in _SCIENCE_IDENTITY_FIELDS
                },
            }
        )
        return candidate

    @classmethod
    def _build_review_queue(
        cls,
        review_pool: list[dict[str, Any]],
        *,
        strategy: str,
        sample_count: int,
        seed: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if strategy == "rules":
            selected = list(review_pool)
            for item in selected:
                item["selection_source"] = "automatic_bad"
            return selected, {
                "strategy": strategy,
                "target_sample_count": 0,
                "mandatory_count": len(selected),
                "sampled_good_count": 0,
                "queue_count": len(selected),
                "sampled_good_fraction": 0.0,
                "target_met": True,
                "seed": seed,
            }

        mandatory_reasons: dict[str, str] = {}
        mandatory = []
        for item in review_pool:
            automatic_decision = item.get("automatic_decision")
            if automatic_decision == "bad_candidate":
                reason = "automatic_bad"
            elif automatic_decision == "uncertain":
                reason = "automatic_uncertain"
            else:
                reason = cls._model_review_reason(item.get("ml_prediction"))
            if reason:
                mandatory.append(item)
                mandatory_reasons[item["candidate_id"]] = reason
        mandatory_ids = set(mandatory_reasons)
        good_pool = [
            item
            for item in review_pool
            if item.get("automatic_decision") == "good_candidate"
            and item["candidate_id"] not in mandatory_ids
        ]
        minimum_for_fraction = math.ceil(
            len(mandatory)
            * MINIMUM_AUTOMATIC_GOOD_FRACTION
            / (1.0 - MINIMUM_AUTOMATIC_GOOD_FRACTION)
            - 1.0e-12
        )
        desired_good = max(sample_count - len(mandatory), minimum_for_fraction, 0)
        selected_good = cls._stratified_good_sample(
            good_pool, min(desired_good, len(good_pool)), seed=seed
        )
        for item in mandatory:
            item["selection_source"] = mandatory_reasons[item["candidate_id"]]
        for item in selected_good:
            item["selection_source"] = "automatic_good_sample"
        selected = mandatory + selected_good
        good_fraction = (
            float(len(selected_good)) / float(len(selected)) if selected else 0.0
        )
        target_met = (
            not selected
            or not mandatory
            or good_fraction + 1.0e-12 >= MINIMUM_AUTOMATIC_GOOD_FRACTION
        )
        return selected, {
            "strategy": strategy,
            "target_sample_count": sample_count,
            "mandatory_count": len(mandatory),
            "sampled_good_count": len(selected_good),
            "queue_count": len(selected),
            "sampled_good_fraction": good_fraction,
            "target_met": target_met,
            "seed": seed,
        }

    @staticmethod
    def _model_review_reason(prediction: Any) -> str | None:
        if not isinstance(prediction, dict):
            return None
        if prediction.get("ood") is True:
            return "model_ood"
        probabilities = prediction.get("probabilities")
        if not isinstance(probabilities, dict):
            return "model_uncertain"
        finite_probabilities = []
        for value in probabilities.values():
            try:
                number = float(value)
            except TypeError, ValueError:
                continue
            if math.isfinite(number):
                finite_probabilities.append(number)
        if not finite_probabilities or max(finite_probabilities) < 0.60:
            return "model_uncertain"
        predicted = str(prediction.get("predicted_label") or "")
        if predicted == "bad":
            return "rule_model_conflict"
        if predicted == "degraded":
            return "model_degraded"
        return None

    @classmethod
    def _stratified_good_sample(
        cls,
        candidates: list[dict[str, Any]],
        count: int,
        *,
        seed: str,
    ) -> list[dict[str, Any]]:
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for candidate in candidates:
            groups.setdefault(cls._good_stratum(candidate), []).append(candidate)
        for group in groups.values():
            group.sort(
                key=lambda item: hashlib.sha256(
                    f"{seed}|{item['candidate_id']}".encode("utf-8")
                ).hexdigest()
            )
        selected: list[dict[str, Any]] = []
        active = sorted(groups)
        while active and len(selected) < count:
            next_active = []
            for key in active:
                group = groups[key]
                if group and len(selected) < count:
                    selected.append(group.pop(0))
                if group:
                    next_active.append(key)
            active = next_active
        return selected

    @staticmethod
    def _good_stratum(candidate: dict[str, Any]) -> tuple[Any, ...]:
        features = candidate.get("features") or {}
        time_value = str(candidate.get("time") or "")
        date_bucket = time_value.split("T", 1)[0] if "T" in time_value else "unknown"
        return (
            float(candidate.get("frequency_mhz", 0.0)),
            str(candidate.get("polarization") or ""),
            date_bucket,
            _numeric_bin(features.get("dynamic_range_z"), (10.0, 50.0)),
            _numeric_bin(features.get("stripe_score"), (0.10, 0.25)),
        )

    def _sample_fingerprint(self, value: str | Path) -> str:
        path = self.resolve_input(value)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"

    @staticmethod
    def _attach_context(
        candidates: list[dict[str, Any]], files: list[dict[str, Any]]
    ) -> None:
        grouped: dict[tuple[float, str], list[dict[str, Any]]] = {}
        for item in files:
            grouped.setdefault(
                (item["frequency_mhz"], item["polarization"]), []
            ).append(item)
        positions: dict[str, tuple[list[dict[str, Any]], int]] = {}
        for group in grouped.values():
            group.sort(key=lambda item: (item["file_index"], item["relative_path"]))
            for index, item in enumerate(group):
                positions[item["file_id"]] = (group, index)
        for candidate in candidates:
            group, index = positions[candidate["file_id"]]
            candidate["context_file_ids"] = [
                group[index - 1]["file_id"] if index > 0 else None,
                candidate["file_id"],
                group[index + 1]["file_id"] if index + 1 < len(group) else None,
            ]

    @staticmethod
    def _fingerprint(files: list[dict[str, Any]], *, payload: dict[str, Any]) -> str:
        canonical_files = [
            {
                "relative_path": item["relative_path"],
                "size": item["size"],
                "mtime_ns": item["mtime_ns"],
            }
            for item in sorted(files, key=lambda row: row["relative_path"])
        ]
        encoded = json.dumps(
            {"selection": payload, "files": canonical_files},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def _assert_fresh(
        self,
        manifest: dict[str, Any],
        *,
        file_ids: set[str] | None = None,
    ) -> None:
        for item in manifest["files"]:
            if file_ids is not None and item["file_id"] not in file_ids:
                continue
            try:
                path = self.resolve_input(item["source_file"])
                stat = path.stat()
            except (FileNotFoundError, OSError, PermissionError) as exc:
                raise StaleReviewError("Review inputs changed; scan again") from exc
            if stat.st_size != item["size"] or stat.st_mtime_ns != item["mtime_ns"]:
                raise StaleReviewError("Review inputs changed; scan again")

    @staticmethod
    def _refresh_derived(manifest: dict[str, Any]) -> None:
        status = manifest["status"]
        counts = {label: 0 for label in QUALITY_LABELS}
        counts["pending"] = 0
        automatic_counts = {
            decision: 0 for decision in ("good_candidate", "bad_candidate", "uncertain")
        }
        final_bad_files = []
        for candidate in manifest["candidates"]:
            automatic_decision = candidate.get("automatic_decision")
            if automatic_decision in automatic_counts:
                automatic_counts[automatic_decision] += 1
            human_label = candidate.get("human_label")
            quality_label = (
                human_label.get("quality_label")
                if isinstance(human_label, dict)
                and human_label.get("source") == "human"
                else None
            )
            if quality_label in QUALITY_LABELS:
                counts[quality_label] += 1
                candidate["decision_source"] = "human"
                candidate["human_decision"] = (
                    "bad"
                    if quality_label == "bad"
                    else "good" if quality_label in {"good", "degraded"} else None
                )
                candidate["final_quality"] = (
                    "bad"
                    if quality_label == "bad"
                    else "ok" if quality_label in {"good", "degraded"} else None
                )
            elif status == "skipped":
                candidate["decision_source"] = "automatic_on_skip"
                candidate["human_decision"] = None
                candidate["final_quality"] = (
                    "bad"
                    if automatic_decision == "bad_candidate"
                    else "ok" if automatic_decision == "good_candidate" else None
                )
            else:
                candidate["decision_source"] = None
                candidate["human_decision"] = None
                candidate["final_quality"] = None
            if quality_label not in TRAINABLE_QUALITY_LABELS:
                counts["pending"] += 1
            if (
                status in {"completed", "skipped"}
                and candidate["final_quality"] == "bad"
            ):
                final_bad_files.append(candidate["source_file"])
        manifest["final_bad_files"] = final_bad_files
        manifest["summary"] = {
            "scanned_file_count": len(manifest["files"]),
            "candidate_count": len(manifest["candidates"]),
            "pending_count": counts["pending"] if status == "draft" else 0,
            "confirmed_bad_count": counts["bad"],
            "kept_count": counts["good"] + counts["degraded"],
            "good_count": counts["good"],
            "degraded_count": counts["degraded"],
            "uncertain_count": counts["uncertain"],
            "automatic_good_count": automatic_counts["good_candidate"],
            "automatic_bad_count": automatic_counts["bad_candidate"],
            "automatic_uncertain_count": automatic_counts["uncertain"],
            "sampled_good_count": int(
                (manifest.get("sampling") or {}).get("sampled_good_count", 0)
            ),
            "training_eligible_count": (
                counts["good"] + counts["degraded"] + counts["bad"]
                if status == "completed"
                else 0
            ),
            "final_bad_count": len(final_bad_files),
            "viewed_frame_count": int(
                (manifest.get("audit") or {}).get("viewed_frame_count", 0)
            ),
            "remaining_frame_count": int(
                (manifest.get("audit") or {}).get("remaining_frame_count", 0)
            ),
        }

    @staticmethod
    def _write_candidate_csv(path: Path, manifest: dict[str, Any]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            for candidate in manifest["candidates"]:
                metrics = candidate.get("metrics", {})
                human_label = candidate.get("human_label") or {}
                writer.writerow(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "time": candidate["time"],
                        "frequency_mhz": candidate["frequency_mhz"],
                        "polarization": candidate["polarization"],
                        "relative_path": candidate["relative_path"],
                        "source_file": candidate["source_file"],
                        "algorithm_flag": candidate["algorithm_flag"],
                        "algorithm_reason": candidate["algorithm_reason"],
                        "human_decision": candidate.get("human_decision") or "",
                        "decision_source": candidate.get("decision_source") or "",
                        "final_quality": candidate.get("final_quality") or "",
                        "p95_delta": metrics.get("p95_delta"),
                        "p997_delta": metrics.get("p997_delta"),
                        "bright_component_count": metrics.get("bright_component_count"),
                        "distributed_bright_fraction": metrics.get(
                            "distributed_bright_fraction"
                        ),
                        "automatic_decision": candidate.get("automatic_decision", ""),
                        "automatic_reasons": ";".join(
                            candidate.get("automatic_reasons") or []
                        ),
                        "automatic_rule_version": candidate.get(
                            "automatic_rule_version", ""
                        ),
                        "selection_source": candidate.get("selection_source", ""),
                        "sample_fingerprint": candidate.get("sample_fingerprint", ""),
                        "quality_label": human_label.get("quality_label", ""),
                        "event_tags": json.dumps(
                            human_label.get("event_tags", []), separators=(",", ":")
                        ),
                        "artifact_tags": json.dumps(
                            human_label.get("artifact_tags", []),
                            separators=(",", ":"),
                        ),
                        "reviewed_at": human_label.get("reviewed_at", ""),
                        "ml_prediction": json.dumps(
                            candidate.get("ml_prediction"), separators=(",", ":")
                        ),
                    }
                )

    def _preview_range_lock(
        self, review_id: str, frequency_mhz: float, transform: str
    ) -> threading.RLock:
        key = (review_id, f"{frequency_mhz:.12g}", transform)
        with self._preview_range_locks_guard:
            return self._preview_range_locks.setdefault(key, threading.RLock())

    def _frequency_preview_sample(
        self,
        manifest: dict[str, Any],
        *,
        frequency_mhz: float,
        transform: str,
    ) -> tuple[Any, str | None]:
        import numpy as np

        records = [
            item
            for item in manifest.get("files", [])
            if float(item.get("frequency_mhz", 0.0)) == float(frequency_mhz)
        ]
        if not records:
            raise ValueError(f"No review files are available for {frequency_mhz:g} MHz")
        file_ids = {str(item["file_id"]) for item in records}
        self._assert_fresh(manifest, file_ids=file_ids)
        fingerprint_payload = {
            "version": PREVIEW_RANGE_CACHE_VERSION,
            "frequency_mhz": float(frequency_mhz),
            "transform": transform,
            "files": [
                [
                    str(item["file_id"]),
                    int(item["size"]),
                    int(item["mtime_ns"]),
                ]
                for item in sorted(records, key=lambda value: str(value["file_id"]))
            ],
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        frequency_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{frequency_mhz:.12g}")
        cache_dir = (
            self._review_dir(str(manifest["review_id"])) / PREVIEW_RANGE_CACHE_DIRNAME
        )
        cache_path = cache_dir / f"{frequency_key}MHz-{transform}.npz"
        lock = self._preview_range_lock(
            str(manifest["review_id"]), frequency_mhz, transform
        )
        with lock:
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    cached_fingerprint = str(cached["fingerprint"].item())
                    if cached_fingerprint == fingerprint:
                        sample = np.asarray(cached["sample"], dtype=float)
                        unit = str(cached["unit"].item()).strip() or None
                        if sample.size:
                            return sample, unit
            except FileNotFoundError, KeyError, OSError, TypeError, ValueError:
                pass

            quota = max(
                1,
                min(
                    _review_facade.PREVIEW_GLOBAL_FILE_SAMPLE_LIMIT,
                    _review_facade.PREVIEW_GLOBAL_SAMPLE_LIMIT // len(records),
                ),
            )
            samples: list[Any] = []
            units: dict[str, str] = {}
            for record in sorted(records, key=lambda value: str(value["file_id"])):
                try:
                    path = self.resolve_input(str(record["source_file"]))
                    data, header = _review_facade.read_radio_fits_image(path)
                    _transformed, finite_values = _transform_preview_data(
                        data, transform
                    )
                except Exception:  # noqa: BLE001 - partial cache remains useful
                    continue
                if not finite_values.size:
                    continue
                unit = str(header.get("BUNIT") or "").strip()
                if unit:
                    units.setdefault(unit.casefold(), unit)
                if finite_values.size > quota:
                    indices = np.linspace(
                        0, finite_values.size - 1, num=quota, dtype=np.int64
                    )
                    finite_values = finite_values[indices]
                samples.append(np.asarray(finite_values, dtype=float))
            if len(units) > 1:
                labels = ", ".join(sorted(units.values(), key=str.casefold))
                raise ValueError(
                    f"Fixed range requires one BUNIT per frequency; found: {labels}"
                )
            if not samples:
                raise ValueError(
                    f"No finite intensity values are available for {frequency_mhz:g} MHz"
                )
            sample = np.concatenate(samples)
            if sample.size > _review_facade.PREVIEW_GLOBAL_SAMPLE_LIMIT:
                indices = np.linspace(
                    0,
                    sample.size - 1,
                    num=_review_facade.PREVIEW_GLOBAL_SAMPLE_LIMIT,
                    dtype=np.int64,
                )
                sample = sample[indices]
            unit = next(iter(units.values()), None)
            temporary: Path | None = None
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                temporary = cache_dir / f".{cache_path.name}.{uuid.uuid4().hex}.tmp.npz"
                np.savez_compressed(
                    temporary,
                    sample=sample,
                    fingerprint=np.asarray(fingerprint),
                    unit=np.asarray(unit or ""),
                )
                temporary.replace(cache_path)
            except OSError:
                pass
            finally:
                if temporary is not None and temporary.exists():
                    try:
                        temporary.unlink()
                    except OSError:
                        pass
            return sample, unit

    def _render_triptych(
        self,
        manifest: dict[str, Any],
        candidate: dict[str, Any],
        context: list[dict[str, Any] | None],
        *,
        display: PreviewDisplaySettings,
    ) -> bytes:
        from matplotlib import colormaps
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        images: list[
            tuple[Any | None, str, _PreviewGeometry | None, str | None, Any | None]
        ] = []
        for item in context:
            if item is None:
                images.append((None, "Frame unavailable", None, None, None))
                continue
            path = self.resolve_input(item["source_file"])
            try:
                data, header = _review_facade.read_radio_fits_image(path)
                transformed, finite_values = _transform_preview_data(
                    data, display.transform
                )
                if finite_values.size:
                    try:
                        geometry = _preview_geometry(header, transformed.shape)
                    except KeyError, TypeError, ValueError:
                        geometry = None
                    unit = str(header.get("BUNIT") or "").strip() or None
                    images.append(
                        (transformed, path.name, geometry, unit, finite_values)
                    )
                else:
                    images.append((None, "No finite pixels", None, None, None))
            except Exception as exc:  # noqa: BLE001 - show unreadable candidates
                images.append(
                    (
                        None,
                        f"Unreadable frame\n{type(exc).__name__}",
                        None,
                        None,
                        None,
                    )
                )

        shared_limits: tuple[float, float] | None = None
        shared_unit: str | None = None
        if display.has_legacy_absolute_range:
            shared_limits = (float(display.vmin), float(display.vmax))
        elif display.range_mode == "fixed":
            sample, shared_unit = self._frequency_preview_sample(
                manifest,
                frequency_mhz=float(candidate["frequency_mhz"]),
                transform=display.transform,
            )
            shared_limits = _preview_percentile_limits(
                sample, display.pmin, display.pmax
            )
        panel_limits = [
            (
                (
                    shared_limits
                    if shared_limits is not None
                    else _preview_percentile_limits(values, display.pmin, display.pmax)
                )
                if image is not None and values is not None
                else None
            )
            for image, _detail, _geometry, _unit, values in images
        ]

        figure = Figure(
            figsize=(14.2 if shared_limits is None else 12, 4.55),
            dpi=120,
            layout="constrained",
        )
        figure.patch.set_facecolor(_PREVIEW_FIGURE_FACE)
        FigureCanvasAgg(figure)
        axes = figure.subplots(1, 3)
        labels = ("Previous", "Candidate", "Next")
        image_artists: list[tuple[Any, Any, str | None]] = []
        cmap = colormaps.get_cmap(display.cmap).with_extremes(
            bad=_PREVIEW_PLACEHOLDER_FACE
        )
        for index, (axis, panel, limits) in enumerate(
            zip(axes, images, panel_limits, strict=True)
        ):
            image, detail, geometry, unit, _finite_values = panel
            axis.set_facecolor(_PREVIEW_PLACEHOLDER_FACE)
            if image is None:
                axis.text(
                    0.5,
                    0.5,
                    detail,
                    color=_PREVIEW_PLACEHOLDER_TEXT,
                    fontsize=10,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
                axis.set_xticks([])
                axis.set_yticks([])
            else:
                image_kwargs: dict[str, Any] = {
                    "origin": geometry.origin if geometry is not None else "lower",
                    "cmap": cmap,
                    "vmin": limits[0],
                    "vmax": limits[1],
                    "interpolation": "nearest",
                }
                if geometry is not None:
                    image_kwargs["extent"] = geometry.extent_arcsec
                image_artist = axis.imshow(
                    image,
                    **image_kwargs,
                )
                image_artists.append((image_artist, axis, unit))
                if geometry is not None:
                    axis.set_xlabel(
                        "HPLN / arcsec",
                        fontsize=9,
                        color=_PREVIEW_TICK_COLOR,
                        fontweight="bold",
                    )
                    axis.set_ylabel(
                        "HPLT / arcsec",
                        fontsize=9,
                        color=_PREVIEW_TICK_COLOR,
                        fontweight="bold",
                    )
                    axis.set_xlim(*PREVIEW_ARCSEC_VIEW_LIMITS)
                    axis.set_ylim(*PREVIEW_ARCSEC_VIEW_LIMITS)
                    axis.ticklabel_format(axis="both", style="plain", useOffset=False)
                else:
                    axis.set_xlabel(
                        "Pixel X — WCS unavailable",
                        fontsize=9,
                        color=_PREVIEW_WCS_WARNING,
                        fontweight="bold",
                    )
                    axis.set_ylabel(
                        "Pixel Y — WCS unavailable",
                        fontsize=9,
                        color=_PREVIEW_WCS_WARNING,
                        fontweight="bold",
                    )
                    axis.text(
                        0.02,
                        0.98,
                        "WCS unavailable",
                        transform=axis.transAxes,
                        ha="left",
                        va="top",
                        fontsize=8.5,
                        color=_PREVIEW_WCS_WARNING,
                        fontweight="bold",
                        bbox={
                            "facecolor": _PREVIEW_FIGURE_FACE,
                            "edgecolor": _PREVIEW_WCS_WARNING,
                            "alpha": 0.92,
                            "pad": 2.0,
                        },
                    )
                axis.tick_params(
                    labelsize=8.5,
                    colors=_PREVIEW_TICK_COLOR,
                    width=1.0,
                    length=3.5,
                )
            axis.set_title(
                f"{labels[index]}\n{detail}",
                fontsize=10,
                color=_PREVIEW_TEXT_COLOR,
                fontweight="bold",
                pad=8,
            )
            border = (
                _PREVIEW_CANDIDATE_BORDER if index == 1 else _PREVIEW_CONTEXT_BORDER
            )
            width = 3.0 if index == 1 else 1.5
            for spine in axis.spines.values():
                spine.set_color(border)
                spine.set_linewidth(width)

        def colorbar_label(unit: str | None) -> str:
            if display.transform == "robust_asinh":
                return "signed asinh robust intensity"
            return f"Intensity [{unit}]" if unit else "Raw FITS intensity"

        def style_colorbar(colorbar: Any, label: str) -> None:
            colorbar.set_label(
                label,
                fontsize=9,
                color=_PREVIEW_TICK_COLOR,
                fontweight="bold",
            )
            colorbar.ax.tick_params(
                labelsize=8.5,
                colors=_PREVIEW_TICK_COLOR,
                width=1.0,
            )
            colorbar.outline.set_edgecolor(_PREVIEW_CONTEXT_BORDER)

        if image_artists and shared_limits is not None:
            units = {
                unit for _image, _detail, _geometry, unit, _values in images if unit
            }
            unit = shared_unit or (next(iter(units)) if len(units) == 1 else None)
            colorbar = figure.colorbar(
                image_artists[-1][0],
                ax=[artist_axis for _artist, artist_axis, _unit in image_artists],
                shrink=0.78,
                pad=0.02,
            )
            style_colorbar(colorbar, colorbar_label(unit))
        elif image_artists:
            for image_artist, axis, unit in image_artists:
                colorbar = figure.colorbar(image_artist, ax=axis, shrink=0.76, pad=0.02)
                style_colorbar(colorbar, colorbar_label(unit))
        figure.suptitle(
            f"{_format_frequency(candidate['frequency_mhz'])} MHz  |  "
            f"{candidate['polarization']}  |  {candidate['time'] or 'unknown time'}",
            fontsize=12,
            color=_PREVIEW_TEXT_COLOR,
            fontweight="bold",
        )
        buffer = io.BytesIO()
        figure.savefig(
            buffer,
            format="png",
            dpi=120,
            facecolor=_PREVIEW_FIGURE_FACE,
        )
        figure.clear()
        return buffer.getvalue()


__all__ = [
    "ARTIFACT_TAGS",
    "BAD_FRAME_REVIEW_SCHEMA_VERSION",
    "BadFrameReviewStore",
    "EVENT_TAGS",
    "QUALITY_LABELS",
    "REVIEW_SCOPES",
    "StaleReviewError",
    "extract_training_examples",
    "final_bad_frame_paths",
    "load_bad_frame_review",
]
