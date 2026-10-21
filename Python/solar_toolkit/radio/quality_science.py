"""Scientific, non-learning quality features for radio FITS images.

This module deliberately contains no fitted model.  It provides the first
stage of the bad-frame workflow: deterministic validation, robust signed
statistics, spatial morphology, and contextual comparisons.  Human review
and optional machine learning are separate downstream stages.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from scipy import ndimage, stats
from skimage.transform import resize

from solar_toolkit._utils.numeric import numeric_data

from .raw_quality import RawFileQualityRow, read_radio_fits_image

__all__ = [
    "AUTOMATIC_QUALITY_RULE_VERSION",
    "AutomaticQualityDecision",
    "RadioScienceFeatureConfig",
    "RadioScienceFeatures",
    "RadioScienceQualityRow",
    "analyze_radio_science_quality",
    "classify_scientific_quality",
    "compute_context_similarity",
    "compute_scientific_quality_features",
    "write_radio_science_quality_csv",
]


AUTOMATIC_QUALITY_RULE_VERSION = "radio-science-v1"
AUTOMATIC_DECISIONS = frozenset({"good_candidate", "bad_candidate", "uncertain"})


@dataclass(frozen=True)
class RadioScienceFeatureConfig:
    """Conservative thresholds for deterministic radio-image screening."""

    thumbnail_size: int = 16
    spatial_size: int = 128
    tail_sigma: float = 6.0
    component_sigma: float = 5.0
    hard_min_finite_fraction: float = 0.50
    bad_min_finite_fraction: float = 0.98
    uncertain_min_finite_fraction: float = 0.999
    bad_stripe_score: float = 0.42
    uncertain_stripe_score: float = 0.28
    bad_stripe_group_z: float = 5.0
    uncertain_stripe_group_z: float = 3.5
    bad_tail_fraction: float = 0.075
    uncertain_tail_fraction: float = 0.035
    bad_component_count: int = 12
    uncertain_component_count: int = 6
    context_bad_correlation: float = -0.10
    context_uncertain_correlation: float = 0.20
    context_bad_residual_mad: float = 3.5
    context_uncertain_residual_mad: float = 1.75

    def __post_init__(self) -> None:
        if self.thumbnail_size < 4:
            raise ValueError("thumbnail_size must be at least 4")
        if self.spatial_size < self.thumbnail_size:
            raise ValueError("spatial_size must be >= thumbnail_size")
        if self.tail_sigma <= 0 or self.component_sigma <= 0:
            raise ValueError("sigma thresholds must be positive")
        for name in (
            "hard_min_finite_fraction",
            "bad_min_finite_fraction",
            "uncertain_min_finite_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")


@dataclass(frozen=True)
class RadioScienceFeatures:
    """Signed robust statistics and morphology for one two-dimensional image."""

    total_pixel_count: int
    image_height: int
    image_width: int
    finite_pixel_count: int
    finite_fraction: float
    zero_fraction: float
    robust_median: float
    robust_mad: float
    robust_scale: float
    p001_z: float
    p01_z: float
    p05_z: float
    p50_z: float
    p95_z: float
    p99_z: float
    p999_z: float
    dynamic_range_z: float
    skewness: float
    excess_kurtosis: float
    positive_tail_fraction: float
    negative_tail_fraction: float
    positive_component_count: int
    negative_component_count: int
    positive_component_count_3sigma: int
    negative_component_count_3sigma: int
    positive_component_count_7sigma: int
    negative_component_count_7sigma: int
    largest_positive_component_fraction: float
    largest_negative_component_fraction: float
    largest_component_bbox_fill: float
    largest_component_bbox_width_fraction: float
    largest_component_bbox_height_fraction: float
    largest_component_compactness: float
    negative_to_positive_peak_ratio: float
    negative_to_positive_tail_energy_ratio: float
    sidelobe_to_main_energy_ratio: float
    source_centroid_x: float
    source_centroid_y: float
    source_extent_fraction: float
    gradient_orientation_entropy: float
    gradient_orientation_dominance: float
    fft_axis_energy_fraction: float
    fft_peak_energy_fraction: float
    stripe_score: float
    wcs_valid: bool
    observation_time_valid: bool
    frequency_metadata_valid: bool
    data_valid: bool
    invalid_reason: str


@dataclass(frozen=True)
class AutomaticQualityDecision:
    """Deterministic pre-review decision for one frame."""

    automatic_decision: str
    automatic_reasons: tuple[str, ...]
    hard_invalid: bool
    rule_version: str = AUTOMATIC_QUALITY_RULE_VERSION

    def __post_init__(self) -> None:
        if self.automatic_decision not in AUTOMATIC_DECISIONS:
            raise ValueError(
                f"Unsupported automatic decision: {self.automatic_decision}"
            )


@dataclass(frozen=True)
class RadioScienceQualityRow:
    """File identity, scientific features, context, and automatic decision."""

    source_file: str
    frequency_mhz: float
    polarization: str
    file_index: int
    slot_index: int
    time: str
    legacy_quality_flag: str
    legacy_reason: str
    automatic_decision: str
    automatic_reasons: str
    hard_invalid: bool
    rule_version: str
    temporal_correlation: float
    temporal_residual_mad: float
    polarization_correlation: float
    polarization_residual_mad: float
    adjacent_frequency_correlation: float
    adjacent_frequency_residual_mad: float
    stripe_score_group_z: float
    tail_fraction_group_z: float
    shape_consistent_with_group: bool
    total_pixel_count: int
    image_height: int
    image_width: int
    finite_pixel_count: int
    finite_fraction: float
    zero_fraction: float
    robust_median: float
    robust_mad: float
    robust_scale: float
    p001_z: float
    p01_z: float
    p05_z: float
    p50_z: float
    p95_z: float
    p99_z: float
    p999_z: float
    dynamic_range_z: float
    skewness: float
    excess_kurtosis: float
    positive_tail_fraction: float
    negative_tail_fraction: float
    positive_component_count: int
    negative_component_count: int
    positive_component_count_3sigma: int
    negative_component_count_3sigma: int
    positive_component_count_7sigma: int
    negative_component_count_7sigma: int
    largest_positive_component_fraction: float
    largest_negative_component_fraction: float
    largest_component_bbox_fill: float
    largest_component_bbox_width_fraction: float
    largest_component_bbox_height_fraction: float
    largest_component_compactness: float
    negative_to_positive_peak_ratio: float
    negative_to_positive_tail_energy_ratio: float
    sidelobe_to_main_energy_ratio: float
    source_centroid_x: float
    source_centroid_y: float
    source_extent_fraction: float
    gradient_orientation_entropy: float
    gradient_orientation_dominance: float
    fft_axis_energy_fraction: float
    fft_peak_energy_fraction: float
    stripe_score: float
    wcs_valid: bool
    observation_time_valid: bool
    frequency_metadata_valid: bool
    data_valid: bool
    invalid_reason: str


@dataclass
class _PendingScienceRow:
    raw: RawFileQualityRow
    features: RadioScienceFeatures
    thumbnail: np.ndarray | None
    temporal_correlation: float = math.nan
    temporal_residual_mad: float = math.nan
    polarization_correlation: float = math.nan
    polarization_residual_mad: float = math.nan
    adjacent_frequency_correlation: float = math.nan
    adjacent_frequency_residual_mad: float = math.nan
    stripe_score_group_z: float = 0.0
    tail_fraction_group_z: float = 0.0
    shape_consistent_with_group: bool = True


def compute_scientific_quality_features(
    data: np.ndarray,
    header: fits.Header | None = None,
    config: RadioScienceFeatureConfig | None = None,
    *,
    expected_frequency_mhz: float | None = None,
    expected_time: str | None = None,
) -> tuple[RadioScienceFeatures, np.ndarray | None]:
    """Extract robust features, treating masked samples exactly like NaNs."""

    cfg = config or RadioScienceFeatureConfig()
    arr = np.squeeze(numeric_data(data))
    if arr.ndim != 2:
        return (
            _invalid_features(
                int(arr.size), 0, math.nan, f"expected_2d_got_{arr.ndim}d"
            ),
            None,
        )

    total = int(arr.size)
    finite = np.isfinite(arr)
    finite_count = int(finite.sum())
    finite_fraction = _safe_fraction(finite_count, total)
    if finite_count == 0:
        return (
            _invalid_features(total, finite_count, finite_fraction, "no_finite_pixels"),
            None,
        )

    values = arr[finite]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = float(1.4826 * mad)
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        q25, q75 = np.percentile(values, [25.0, 75.0])
        scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        return (
            _invalid_features(
                total, finite_count, finite_fraction, "constant_or_zero_variance"
            ),
            None,
        )

    z_values = (values - median) / scale
    p001, p01, p05, p50, p95, p99, p999 = np.percentile(
        z_values, [0.1, 1.0, 5.0, 50.0, 95.0, 99.0, 99.9]
    )
    clipped = np.clip(z_values, -50.0, 50.0)
    skewness = _finite_float(stats.skew(clipped, bias=False), 0.0)
    kurtosis = _finite_float(stats.kurtosis(clipped, fisher=True, bias=False), 0.0)

    z_image = np.zeros(arr.shape, dtype=np.float64)
    z_image[finite] = np.arcsinh((arr[finite] - median) / scale)
    spatial = _resize_image(z_image, cfg.spatial_size)
    thumbnail = _resize_image(z_image, cfg.thumbnail_size).astype(np.float32)
    thumbnail -= float(np.median(thumbnail))
    thumb_scale = float(np.median(np.abs(thumbnail))) * 1.4826
    if np.isfinite(thumb_scale) and thumb_scale > np.finfo(float).eps:
        thumbnail /= thumb_scale

    component_z = np.zeros(arr.shape, dtype=np.float64)
    component_z[finite] = (arr[finite] - median) / scale
    positive_mask = component_z >= cfg.component_sigma
    negative_mask = component_z <= -cfg.component_sigma
    positive_stats = _component_stats(positive_mask, np.maximum(component_z, 0.0))
    negative_stats = _component_stats(negative_mask, np.maximum(-component_z, 0.0))
    positive_stats_3 = _component_stats(component_z >= 3.0)
    negative_stats_3 = _component_stats(component_z <= -3.0)
    positive_stats_7 = _component_stats(component_z >= 7.0)
    negative_stats_7 = _component_stats(component_z <= -7.0)
    largest = max(
        positive_stats + negative_stats,
        key=lambda item: item["size"],
        default=None,
    )
    source_mask = component_z >= 3.0
    centroid_x, centroid_y, source_extent = _source_geometry(component_z, source_mask)
    gradient_entropy, gradient_dominance = _gradient_orientation_stats(spatial)
    axis_fraction, peak_fraction, stripe_score = _fft_stats(spatial)
    positive_peak = float(np.max(component_z[finite]))
    negative_peak = abs(float(np.min(component_z[finite])))
    positive_tail_energy = float(np.sum(np.square(np.maximum(component_z, 0.0))))
    negative_tail_energy = float(np.sum(np.square(np.minimum(component_z, 0.0))))
    positive_components_by_size = sorted(
        positive_stats, key=lambda item: item["size"], reverse=True
    )
    main_energy = (
        float(positive_components_by_size[0]["energy"])
        if positive_components_by_size
        else 0.0
    )
    sidelobe_energy = sum(
        float(item["energy"]) for item in positive_components_by_size[1:]
    ) + sum(float(item["energy"]) for item in negative_stats)

    features = RadioScienceFeatures(
        total_pixel_count=total,
        image_height=int(arr.shape[0]),
        image_width=int(arr.shape[1]),
        finite_pixel_count=finite_count,
        finite_fraction=finite_fraction,
        zero_fraction=float(np.mean(values == 0.0)),
        robust_median=median,
        robust_mad=mad,
        robust_scale=scale,
        p001_z=float(p001),
        p01_z=float(p01),
        p05_z=float(p05),
        p50_z=float(p50),
        p95_z=float(p95),
        p99_z=float(p99),
        p999_z=float(p999),
        dynamic_range_z=float(p999 - p001),
        skewness=skewness,
        excess_kurtosis=kurtosis,
        positive_tail_fraction=float(np.mean(z_values >= cfg.tail_sigma)),
        negative_tail_fraction=float(np.mean(z_values <= -cfg.tail_sigma)),
        positive_component_count=len(positive_stats),
        negative_component_count=len(negative_stats),
        positive_component_count_3sigma=len(positive_stats_3),
        negative_component_count_3sigma=len(negative_stats_3),
        positive_component_count_7sigma=len(positive_stats_7),
        negative_component_count_7sigma=len(negative_stats_7),
        largest_positive_component_fraction=_largest_fraction(positive_stats, total),
        largest_negative_component_fraction=_largest_fraction(negative_stats, total),
        largest_component_bbox_fill=(
            float(largest["fill_fraction"]) if largest else 0.0
        ),
        largest_component_bbox_width_fraction=(
            float(largest["width"]) / float(arr.shape[1]) if largest else 0.0
        ),
        largest_component_bbox_height_fraction=(
            float(largest["height"]) / float(arr.shape[0]) if largest else 0.0
        ),
        largest_component_compactness=(
            float(largest["compactness"]) if largest else 0.0
        ),
        negative_to_positive_peak_ratio=_safe_ratio(negative_peak, positive_peak),
        negative_to_positive_tail_energy_ratio=_safe_ratio(
            negative_tail_energy, positive_tail_energy
        ),
        sidelobe_to_main_energy_ratio=_safe_ratio(sidelobe_energy, main_energy),
        source_centroid_x=centroid_x,
        source_centroid_y=centroid_y,
        source_extent_fraction=source_extent,
        gradient_orientation_entropy=gradient_entropy,
        gradient_orientation_dominance=gradient_dominance,
        fft_axis_energy_fraction=axis_fraction,
        fft_peak_energy_fraction=peak_fraction,
        stripe_score=stripe_score,
        wcs_valid=_valid_wcs(header, arr.shape),
        observation_time_valid=_valid_observation_time(header, expected_time),
        frequency_metadata_valid=_valid_frequency_metadata(
            header, expected_frequency_mhz
        ),
        data_valid=True,
        invalid_reason="",
    )
    return features, thumbnail


def compute_context_similarity(
    current: np.ndarray | None,
    references: Iterable[np.ndarray | None],
) -> tuple[float, float]:
    """Return context statistics from pairwise finite, unmasked samples."""

    if current is None:
        return math.nan, math.nan
    correlations: list[float] = []
    residuals: list[float] = []
    current_array = numeric_data(current)
    current_flat = current_array.ravel()
    for reference in references:
        if reference is None:
            continue
        reference_array = numeric_data(reference)
        if reference_array.shape != current_array.shape:
            continue
        ref_flat = reference_array.ravel()
        finite = np.isfinite(current_flat) & np.isfinite(ref_flat)
        if not finite.any():
            continue
        current_values = current_flat[finite]
        ref_values = ref_flat[finite]
        current_std = float(np.std(current_values))
        ref_std = float(np.std(ref_values))
        if current_std > 0.0 and ref_std > 0.0:
            correlation = float(np.corrcoef(current_values, ref_values)[0, 1])
            if np.isfinite(correlation):
                correlations.append(correlation)
        difference = current_values - ref_values
        residual = float(np.median(np.abs(difference - np.median(difference))))
        if np.isfinite(residual):
            residuals.append(1.4826 * residual)
    if not correlations and not residuals:
        return math.nan, math.nan
    return (
        float(np.median(correlations)) if correlations else math.nan,
        float(np.median(residuals)) if residuals else math.nan,
    )


def classify_scientific_quality(
    features: RadioScienceFeatures,
    *,
    legacy_quality_flag: str = "ok",
    legacy_reason: str = "",
    temporal_correlation: float = math.nan,
    temporal_residual_mad: float = math.nan,
    polarization_correlation: float = math.nan,
    polarization_residual_mad: float = math.nan,
    adjacent_frequency_correlation: float = math.nan,
    adjacent_frequency_residual_mad: float = math.nan,
    stripe_score_group_z: float = 0.0,
    tail_fraction_group_z: float = 0.0,
    shape_consistent_with_group: bool = True,
    config: RadioScienceFeatureConfig | None = None,
) -> AutomaticQualityDecision:
    """Apply deterministic, conservative rules before any human or ML stage."""

    cfg = config or RadioScienceFeatureConfig()
    if not features.data_valid:
        return AutomaticQualityDecision(
            automatic_decision="bad_candidate",
            automatic_reasons=(f"invalid_data:{features.invalid_reason}",),
            hard_invalid=True,
        )
    if features.finite_fraction < cfg.hard_min_finite_fraction:
        return AutomaticQualityDecision(
            automatic_decision="bad_candidate",
            automatic_reasons=(
                f"insufficient_finite_pixels:{features.finite_fraction:.4f}",
            ),
            hard_invalid=True,
        )

    bad_reasons: list[str] = []
    uncertain_reasons: list[str] = []
    if legacy_quality_flag == "bad":
        bad_reasons.append(f"legacy_rule:{legacy_reason or 'bad'}")
    if features.finite_fraction < cfg.bad_min_finite_fraction:
        bad_reasons.append(f"finite_fraction:{features.finite_fraction:.4f}")
    elif features.finite_fraction < cfg.uncertain_min_finite_fraction:
        uncertain_reasons.append(
            f"finite_fraction_borderline:{features.finite_fraction:.4f}"
        )

    tail_fraction = features.positive_tail_fraction + features.negative_tail_fraction
    component_count = (
        features.positive_component_count + features.negative_component_count
    )
    if (
        features.stripe_score >= cfg.bad_stripe_score
        and stripe_score_group_z >= cfg.bad_stripe_group_z
    ):
        bad_reasons.append(
            "directional_stripes:"
            f"score={features.stripe_score:.3f},z={stripe_score_group_z:.2f}"
        )
    elif (
        features.stripe_score >= cfg.uncertain_stripe_score
        and stripe_score_group_z >= cfg.uncertain_stripe_group_z
    ):
        uncertain_reasons.append(
            "directional_structure:"
            f"score={features.stripe_score:.3f},z={stripe_score_group_z:.2f}"
        )

    if (
        tail_fraction >= cfg.bad_tail_fraction
        and component_count >= cfg.bad_component_count
        and tail_fraction_group_z >= cfg.bad_stripe_group_z
    ):
        bad_reasons.append(
            "distributed_signed_tail:"
            f"fraction={tail_fraction:.4f},components={component_count}"
        )
    elif (
        tail_fraction >= cfg.uncertain_tail_fraction
        and component_count >= cfg.uncertain_component_count
        and tail_fraction_group_z >= cfg.uncertain_stripe_group_z
    ):
        uncertain_reasons.append(
            "signed_tail_borderline:"
            f"fraction={tail_fraction:.4f},components={component_count}"
        )

    context_pairs = (
        ("temporal", temporal_correlation, temporal_residual_mad),
        (
            "polarization",
            polarization_correlation,
            polarization_residual_mad,
        ),
        (
            "adjacent_frequency",
            adjacent_frequency_correlation,
            adjacent_frequency_residual_mad,
        ),
    )
    for label, correlation, residual in context_pairs:
        if (
            np.isfinite(correlation)
            and np.isfinite(residual)
            and correlation <= cfg.context_bad_correlation
            and residual >= cfg.context_bad_residual_mad
        ):
            uncertain_reasons.append(
                f"{label}_strong_disagreement:corr={correlation:.3f},"
                f"residual={residual:.3f}"
            )
        elif (
            np.isfinite(correlation)
            and np.isfinite(residual)
            and correlation <= cfg.context_uncertain_correlation
            and residual >= cfg.context_uncertain_residual_mad
        ):
            uncertain_reasons.append(
                f"{label}_disagreement:corr={correlation:.3f},residual={residual:.3f}"
            )

    if not features.wcs_valid:
        uncertain_reasons.append("missing_or_inconsistent_wcs")
    if not features.observation_time_valid:
        uncertain_reasons.append("missing_or_invalid_observation_time")
    if not features.frequency_metadata_valid:
        uncertain_reasons.append("missing_or_inconsistent_frequency")
    if not shape_consistent_with_group:
        uncertain_reasons.append("image_shape_differs_from_channel_group")
    if bad_reasons:
        return AutomaticQualityDecision(
            automatic_decision="bad_candidate",
            automatic_reasons=tuple(bad_reasons + uncertain_reasons),
            hard_invalid=False,
        )
    if uncertain_reasons:
        return AutomaticQualityDecision(
            automatic_decision="uncertain",
            automatic_reasons=tuple(uncertain_reasons),
            hard_invalid=False,
        )
    return AutomaticQualityDecision(
        automatic_decision="good_candidate",
        automatic_reasons=("scientific_rules_ok",),
        hard_invalid=False,
    )


def analyze_radio_science_quality(
    file_rows: Sequence[RawFileQualityRow],
    config: RadioScienceFeatureConfig | None = None,
) -> list[RadioScienceQualityRow]:
    """Evaluate all file rows and derive temporal/cross-channel context."""

    cfg = config or RadioScienceFeatureConfig()
    pending: list[_PendingScienceRow] = []
    for raw in file_rows:
        try:
            data, header = read_radio_fits_image(raw.source_file)
            features, thumbnail = compute_scientific_quality_features(
                data,
                header,
                cfg,
                expected_frequency_mhz=raw.frequency_mhz,
                expected_time=raw.time,
            )
        except Exception as exc:  # noqa: BLE001 - invalid files become candidates
            features = _invalid_features(0, 0, math.nan, f"{type(exc).__name__}:{exc}")
            thumbnail = None
        pending.append(
            _PendingScienceRow(raw=raw, features=features, thumbnail=thumbnail)
        )

    _attach_group_deviations(pending)
    _attach_temporal_context(pending)
    _attach_cross_channel_context(pending)
    rows: list[RadioScienceQualityRow] = []
    for item in pending:
        decision = classify_scientific_quality(
            item.features,
            legacy_quality_flag=item.raw.quality_flag,
            legacy_reason=item.raw.reason,
            temporal_correlation=item.temporal_correlation,
            temporal_residual_mad=item.temporal_residual_mad,
            polarization_correlation=item.polarization_correlation,
            polarization_residual_mad=item.polarization_residual_mad,
            adjacent_frequency_correlation=item.adjacent_frequency_correlation,
            adjacent_frequency_residual_mad=(item.adjacent_frequency_residual_mad),
            stripe_score_group_z=item.stripe_score_group_z,
            tail_fraction_group_z=item.tail_fraction_group_z,
            shape_consistent_with_group=item.shape_consistent_with_group,
            config=cfg,
        )
        rows.append(_quality_row(item, decision))
    return rows


def write_radio_science_quality_csv(
    path: str | Path, rows: Sequence[RadioScienceQualityRow]
) -> Path:
    """Write a complete automatic-quality feature table."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[field.name for field in fields(RadioScienceQualityRow)]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return target


def _attach_group_deviations(pending: Sequence[_PendingScienceRow]) -> None:
    grouped: dict[tuple[float, str], list[_PendingScienceRow]] = defaultdict(list)
    for item in pending:
        grouped[(item.raw.frequency_mhz, item.raw.polarization)].append(item)
    for group in grouped.values():
        stripe_values = np.asarray(
            [item.features.stripe_score for item in group], dtype=float
        )
        tail_values = np.asarray(
            [
                item.features.positive_tail_fraction
                + item.features.negative_tail_fraction
                for item in group
            ],
            dtype=float,
        )
        stripe_z = _robust_group_z(stripe_values)
        tail_z = _robust_group_z(tail_values)
        for index, item in enumerate(group):
            item.stripe_score_group_z = float(stripe_z[index])
            item.tail_fraction_group_z = float(tail_z[index])
        valid_shapes = [
            (item.features.image_height, item.features.image_width)
            for item in group
            if item.features.data_valid
        ]
        if valid_shapes:
            counts: dict[tuple[int, int], int] = defaultdict(int)
            for shape in valid_shapes:
                counts[shape] += 1
            expected_shape = max(counts, key=lambda shape: (counts[shape], shape))
            for item in group:
                item.shape_consistent_with_group = (
                    not item.features.data_valid
                    or (item.features.image_height, item.features.image_width)
                    == expected_shape
                )


def _attach_temporal_context(pending: Sequence[_PendingScienceRow]) -> None:
    grouped: dict[tuple[float, str], list[_PendingScienceRow]] = defaultdict(list)
    for item in pending:
        grouped[(item.raw.frequency_mhz, item.raw.polarization)].append(item)
    for group in grouped.values():
        group.sort(key=lambda item: (item.raw.file_index, item.raw.source_file))
        for index, item in enumerate(group):
            references = [
                group[neighbor].thumbnail
                for neighbor in range(max(0, index - 2), min(len(group), index + 3))
                if neighbor != index
            ]
            (
                item.temporal_correlation,
                item.temporal_residual_mad,
            ) = compute_context_similarity(item.thumbnail, references)


def _attach_cross_channel_context(pending: Sequence[_PendingScienceRow]) -> None:
    by_slot: dict[str, list[_PendingScienceRow]] = defaultdict(list)
    for item in pending:
        by_slot[_observation_slot_key(item.raw)].append(item)
    for group in by_slot.values():
        by_channel = {
            (item.raw.frequency_mhz, item.raw.polarization): item for item in group
        }
        frequencies = sorted({item.raw.frequency_mhz for item in group})
        for item in group:
            peer_pol = "LL" if item.raw.polarization == "RR" else "RR"
            peer = by_channel.get((item.raw.frequency_mhz, peer_pol))
            (
                item.polarization_correlation,
                item.polarization_residual_mad,
            ) = compute_context_similarity(
                item.thumbnail, [peer.thumbnail if peer else None]
            )

            try:
                freq_index = frequencies.index(item.raw.frequency_mhz)
            except ValueError:
                continue
            neighbor_items: list[_PendingScienceRow | None] = []
            if freq_index > 0:
                neighbor_items.append(
                    by_channel.get((frequencies[freq_index - 1], item.raw.polarization))
                )
            if freq_index + 1 < len(frequencies):
                neighbor_items.append(
                    by_channel.get((frequencies[freq_index + 1], item.raw.polarization))
                )
            (
                item.adjacent_frequency_correlation,
                item.adjacent_frequency_residual_mad,
            ) = compute_context_similarity(
                item.thumbnail,
                [
                    neighbor.thumbnail if neighbor else None
                    for neighbor in neighbor_items
                ],
            )


def _quality_row(
    item: _PendingScienceRow, decision: AutomaticQualityDecision
) -> RadioScienceQualityRow:
    payload: dict[str, Any] = asdict(item.features)
    return RadioScienceQualityRow(
        source_file=item.raw.source_file,
        frequency_mhz=float(item.raw.frequency_mhz),
        polarization=str(item.raw.polarization),
        file_index=int(item.raw.file_index),
        slot_index=int(item.raw.slot_index),
        time=str(item.raw.time),
        legacy_quality_flag=str(item.raw.quality_flag),
        legacy_reason=str(item.raw.reason),
        automatic_decision=decision.automatic_decision,
        automatic_reasons=";".join(decision.automatic_reasons),
        hard_invalid=decision.hard_invalid,
        rule_version=decision.rule_version,
        temporal_correlation=item.temporal_correlation,
        temporal_residual_mad=item.temporal_residual_mad,
        polarization_correlation=item.polarization_correlation,
        polarization_residual_mad=item.polarization_residual_mad,
        adjacent_frequency_correlation=item.adjacent_frequency_correlation,
        adjacent_frequency_residual_mad=item.adjacent_frequency_residual_mad,
        stripe_score_group_z=item.stripe_score_group_z,
        tail_fraction_group_z=item.tail_fraction_group_z,
        shape_consistent_with_group=item.shape_consistent_with_group,
        **payload,
    )


def _resize_image(image: np.ndarray, size: int) -> np.ndarray:
    if image.shape == (size, size):
        return np.asarray(image, dtype=np.float64)
    return np.asarray(
        resize(
            image,
            (size, size),
            order=1,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        ),
        dtype=np.float64,
    )


def _gradient_orientation_stats(image: np.ndarray) -> tuple[float, float]:
    grad_y, grad_x = np.gradient(image)
    magnitude = np.hypot(grad_x, grad_y)
    valid = np.isfinite(magnitude) & (magnitude > np.finfo(float).eps)
    if not np.any(valid):
        return 1.0, 0.0
    angles = np.mod(np.arctan2(grad_y[valid], grad_x[valid]), np.pi)
    histogram, _ = np.histogram(
        angles,
        bins=18,
        range=(0.0, np.pi),
        weights=magnitude[valid],
    )
    total = float(histogram.sum())
    if total <= 0.0:
        return 1.0, 0.0
    probabilities = histogram / total
    nonzero = probabilities > 0.0
    entropy = -float(np.sum(probabilities[nonzero] * np.log(probabilities[nonzero])))
    entropy /= math.log(len(histogram))
    return entropy, float(np.max(probabilities))


def _fft_stats(image: np.ndarray) -> tuple[float, float, float]:
    centered = np.asarray(image, dtype=float) - float(np.median(image))
    height, width = centered.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered * window))) ** 2
    cy, cx = height // 2, width // 2
    radius = max(2, min(height, width) // 32)
    yy, xx = np.ogrid[:height, :width]
    center_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    power[center_mask] = 0.0
    total = float(power.sum())
    if total <= np.finfo(float).eps:
        return 0.0, 0.0, 0.0
    band = max(1, min(height, width) // 64)
    axis_mask = np.zeros_like(power, dtype=bool)
    axis_mask[max(0, cy - band) : cy + band + 1, :] = True
    axis_mask[:, max(0, cx - band) : cx + band + 1] = True
    axis_mask[center_mask] = False
    axis_fraction = float(power[axis_mask].sum() / total)
    flat = power.ravel()
    top_count = max(1, int(math.ceil(flat.size * 0.01)))
    peak_fraction = float(np.partition(flat, -top_count)[-top_count:].sum() / total)
    stripe_score = float(max(axis_fraction, peak_fraction))
    return axis_fraction, peak_fraction, stripe_score


def _component_stats(
    mask: np.ndarray, weights: np.ndarray | None = None
) -> list[dict[str, float]]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.int8))
    stats_rows: list[dict[str, float]] = []
    for label_index in range(1, int(count) + 1):
        ys, xs = np.where(labels == label_index)
        if ys.size == 0:
            continue
        height = int(ys.max() - ys.min() + 1)
        width = int(xs.max() - xs.min() + 1)
        area = max(height * width, 1)
        component = labels == label_index
        boundary = component & ~ndimage.binary_erosion(component)
        perimeter = max(int(np.count_nonzero(boundary)), 1)
        stats_rows.append(
            {
                "size": float(ys.size),
                "width": float(width),
                "height": float(height),
                "fill_fraction": float(ys.size) / float(area),
                "compactness": min(
                    1.0, 4.0 * math.pi * float(ys.size) / float(perimeter**2)
                ),
                "energy": (
                    float(np.sum(weights[component]))
                    if weights is not None
                    else float(ys.size)
                ),
            }
        )
    return stats_rows


def _source_geometry(
    z_image: np.ndarray, source_mask: np.ndarray
) -> tuple[float, float, float]:
    weights = np.where(source_mask, np.maximum(z_image, 0.0), 0.0)
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        return math.nan, math.nan, 0.0
    yy, xx = np.indices(z_image.shape)
    centroid_x = float((weights * xx).sum() / total_weight)
    centroid_y = float((weights * yy).sum() / total_weight)
    return (
        centroid_x,
        centroid_y,
        float(np.count_nonzero(source_mask)) / float(source_mask.size),
    )


def _largest_fraction(stats_rows: Sequence[dict[str, float]], total: int) -> float:
    if not stats_rows or total <= 0:
        return 0.0
    return float(max(row["size"] for row in stats_rows)) / float(total)


def _valid_wcs(header: fits.Header | None, shape: tuple[int, int]) -> bool:
    if header is None:
        return False
    ctype1 = str(header.get("CTYPE1", "")).strip()
    ctype2 = str(header.get("CTYPE2", "")).strip()
    if not ctype1 or not ctype2:
        return False
    naxis1 = int(header.get("NAXIS1", shape[1]))
    naxis2 = int(header.get("NAXIS2", shape[0]))
    if (naxis2, naxis1) != tuple(shape):
        return False
    for key in ("CDELT1", "CDELT2"):
        value = header.get(key)
        if value is None or not np.isfinite(float(value)) or float(value) == 0.0:
            return False
    return True


def _valid_observation_time(
    header: fits.Header | None, expected_time: str | None
) -> bool:
    if expected_time and str(expected_time).strip():
        return True
    if header is None:
        return False
    return any(
        str(header.get(key, "")).strip()
        for key in ("DATE-OBS", "DATEOBS", "TIME-OBS", "TIMEOBS")
    )


def _valid_frequency_metadata(
    header: fits.Header | None, expected_frequency_mhz: float | None
) -> bool:
    expected = _finite_float(expected_frequency_mhz, math.nan)
    if header is None:
        return np.isfinite(expected)
    observed = math.nan
    for key in ("FREQ", "FREQUENCY", "RESTFRQ", "CRVAL3"):
        value = _finite_float(header.get(key), math.nan)
        if np.isfinite(value):
            observed = value / 1.0e6 if abs(value) > 1.0e5 else value
            break
    if not np.isfinite(observed):
        # The directory contract already supplies a checked channel frequency.
        return np.isfinite(expected)
    if not np.isfinite(expected):
        return observed > 0.0
    tolerance = max(0.5, abs(expected) * 0.01)
    return abs(observed - expected) <= tolerance


def _observation_slot_key(row: RawFileQualityRow) -> str:
    if row.time and str(row.time).strip():
        return f"time:{str(row.time).strip()}"
    # Compatibility fallback only.  Callers must keep an entire observation
    # under one dataset split when real timestamps are unavailable.
    return f"fallback-slot:{int(row.slot_index)}"


def _robust_group_z(values: np.ndarray) -> np.ndarray:
    result = np.zeros(values.shape, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return result
    center = float(np.median(values[finite]))
    mad = float(np.median(np.abs(values[finite] - center)))
    scale = 1.4826 * mad
    if scale <= np.finfo(float).eps:
        return result
    result[finite] = (values[finite] - center) / scale
    return result


def _invalid_features(
    total: int,
    finite_count: int,
    finite_fraction: float,
    reason: str,
) -> RadioScienceFeatures:
    payload: dict[str, Any] = {
        field.name: math.nan for field in fields(RadioScienceFeatures)
    }
    payload.update(
        {
            "total_pixel_count": int(total),
            "image_height": 0,
            "image_width": 0,
            "finite_pixel_count": int(finite_count),
            "finite_fraction": float(finite_fraction),
            "zero_fraction": math.nan,
            "positive_component_count": 0,
            "negative_component_count": 0,
            "positive_component_count_3sigma": 0,
            "negative_component_count_3sigma": 0,
            "positive_component_count_7sigma": 0,
            "negative_component_count_7sigma": 0,
            "largest_positive_component_fraction": 0.0,
            "largest_negative_component_fraction": 0.0,
            "largest_component_bbox_fill": 0.0,
            "largest_component_bbox_width_fraction": 0.0,
            "largest_component_bbox_height_fraction": 0.0,
            "largest_component_compactness": 0.0,
            "negative_to_positive_peak_ratio": math.nan,
            "negative_to_positive_tail_energy_ratio": math.nan,
            "sidelobe_to_main_energy_ratio": math.nan,
            "source_extent_fraction": 0.0,
            "gradient_orientation_entropy": 0.0,
            "gradient_orientation_dominance": 0.0,
            "fft_axis_energy_fraction": 0.0,
            "fft_peak_energy_fraction": 0.0,
            "stripe_score": 0.0,
            "wcs_valid": False,
            "observation_time_valid": False,
            "frequency_metadata_valid": False,
            "data_valid": False,
            "invalid_reason": str(reason),
        }
    )
    return RadioScienceFeatures(**payload)


def _safe_fraction(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return math.nan
    return float(numerator) / float(denominator)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return math.nan
    if denominator <= np.finfo(float).eps:
        return math.nan if numerator <= np.finfo(float).eps else math.inf
    return float(numerator) / float(denominator)


def _finite_float(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return number if np.isfinite(number) else float(fallback)
