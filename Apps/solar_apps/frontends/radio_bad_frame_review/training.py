"""Gold-label collection and explicit radio-quality model training."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from solar_toolkit.radio.quality_ml import (
    QualityModelBundle,
    QualitySample,
    QualityTrainingConfig,
    load_quality_model_bundle,
    predict_quality_model,
    save_quality_model_bundle,
    split_quality_samples,
    train_quality_model,
)

from .model_registry import QualityModelRegistry

__all__ = [
    "GoldQualityDataset",
    "collect_gold_quality_dataset",
    "load_published_quality_model",
    "predict_review_candidate",
    "train_review_quality_model",
]


@dataclass(frozen=True)
class GoldQualityDataset:
    """Eligible human labels plus rule outputs needed for fair comparison."""

    samples: tuple[QualitySample, ...]
    automatic_decisions: dict[str, str]
    excluded_counts: dict[str, int]
    review_ids: tuple[str, ...]
    data_fingerprint: str


def collect_gold_quality_dataset(review_root: str | Path) -> GoldQualityDataset:
    """Collect completed, explicit human labels from review manifests only."""

    root = Path(review_root).expanduser().resolve(strict=True)
    manifests = sorted(root.glob("*/review.json"))
    if not manifests and (root / "review.json").is_file():
        manifests = [root / "review.json"]
    samples: list[QualitySample] = []
    automatic_decisions: dict[str, str] = {}
    excluded: Counter[str] = Counter()
    review_ids: list[str] = []

    for path in manifests:
        manifest = _read_json_object(path)
        candidates = manifest.get("candidates", [])
        if not isinstance(candidates, list):
            excluded["invalid_review"] += 1
            continue
        if manifest.get("status") != "completed":
            excluded[f"review_{manifest.get('status', 'unknown')}"] += len(candidates)
            continue
        review_id = str(manifest.get("review_id") or path.parent.name)
        observation_id = _observation_id(manifest, review_id)
        review_ids.append(review_id)
        for candidate in candidates:
            if not isinstance(candidate, dict):
                excluded["invalid_candidate"] += 1
                continue
            human = candidate.get("human_label")
            if not isinstance(human, dict):
                human = _legacy_human_label(candidate)
            label = str(human.get("quality_label", "")).strip().casefold()
            source = str(human.get("source", "")).strip().casefold()
            if source not in {"human", "human_legacy_v1"}:
                excluded["non_human"] += 1
                continue
            if label == "uncertain":
                excluded["uncertain"] += 1
                continue
            if label not in {"good", "degraded", "bad"}:
                excluded["unlabeled"] += 1
                continue
            feature_payload = candidate.get("features")
            if not isinstance(feature_payload, dict):
                feature_payload = candidate.get("feature_values")
            features = _numeric_features(feature_payload)
            if not features:
                excluded["missing_features"] += 1
                continue
            sample_id = str(
                candidate.get("sample_sha256")
                or candidate.get("sample_id")
                or candidate.get("sample_fingerprint")
                or candidate.get("candidate_id")
                or ""
            ).strip()
            if not sample_id:
                excluded["missing_sample_id"] += 1
                continue
            time_value = str(candidate.get("time") or "").strip()
            slot_id = time_value or f"slot:{candidate.get('slot_index', 'unknown')}"
            shape = _shape(candidate, features)
            sample = QualitySample(
                sample_id=sample_id,
                observation_id=observation_id,
                slot_id=slot_id,
                observed_at=(
                    time_value or str(manifest.get("created_at") or "").strip() or None
                ),
                frequency_mhz=_optional_float(candidate.get("frequency_mhz")),
                polarization=str(candidate.get("polarization") or "").strip() or None,
                shape=shape,
                feature_values=features,
                quality_label=label,
                label_source="human",
                event_tags=tuple(sorted(_string_list(human.get("event_tags")))),
            )
            samples.append(sample)
            automatic_decisions[sample_id] = str(
                candidate.get("automatic_decision")
                or (
                    "bad_candidate"
                    if candidate.get("algorithm_flag") == "bad"
                    else "good_candidate"
                )
            )

    ordered = tuple(sorted(samples, key=lambda item: item.sample_id))
    fingerprint_rows = [
        {
            "sample_id": sample.sample_id,
            "observation_id": sample.observation_id,
            "slot_id": sample.slot_id,
            "quality_label": sample.quality_label,
            "event_tags": list(sample.event_tags),
            "feature_values": dict(sorted(sample.feature_values.items())),
        }
        for sample in ordered
    ]
    fingerprint = hashlib.sha256(
        (
            json.dumps(fingerprint_rows, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
    ).hexdigest()
    return GoldQualityDataset(
        samples=ordered,
        automatic_decisions=automatic_decisions,
        excluded_counts=dict(sorted(excluded.items())),
        review_ids=tuple(sorted(set(review_ids))),
        data_fingerprint=fingerprint,
    )


def train_review_quality_model(
    review_root: str | Path,
    registry: QualityModelRegistry,
    *,
    model_id: str,
    config: QualityTrainingConfig | None = None,
) -> dict[str, Any]:
    """Train and register a candidate; publication is always a later action."""

    dataset = collect_gold_quality_dataset(review_root)
    bundle = train_quality_model(dataset.samples, config=config)
    if not model_id or any(
        not (character.isalnum() or character in "-_.") for character in model_id
    ):
        raise ValueError("model_id contains unsupported characters")
    bundle_directory = (registry.root / model_id).resolve(strict=False)
    try:
        bundle_directory.relative_to(registry.root)
    except ValueError as exc:
        raise ValueError("model_id escapes the registry root") from exc
    saved = save_quality_model_bundle(bundle, bundle_directory)
    rule_metrics = _locked_rule_metrics(dataset, config=config)
    ml_metrics = dict(bundle.manifest["locked_test"]["metrics"])
    metrics = {
        **ml_metrics,
        "locked_test_event_rejections": int(
            ml_metrics.get("event_false_rejection_count") or 0
        ),
        "rule_bad_recall": rule_metrics["bad_recall"],
        "rule_good_false_rejection_rate": rule_metrics["good_false_rejection_rate"],
        "rule_event_false_rejection_rate": rule_metrics["event_false_rejection_rate"],
        "outperforms_rule_at_constraint": bool(
            _finite_le(ml_metrics.get("good_false_rejection_rate"), 0.005)
            and _finite_gt(ml_metrics.get("bad_recall"), rule_metrics["bad_recall"])
        ),
    }
    _write_model_card(
        bundle_directory / "MODEL_CARD.md",
        model_id=model_id,
        bundle=bundle,
        dataset=dataset,
        metrics=metrics,
        rule_metrics=rule_metrics,
    )
    entry = registry.register_candidate(
        model_id,
        bundle_directory,
        metrics=metrics,
        feature_schema_version=int(bundle.manifest["feature_schema_version"]),
        data_fingerprint=dataset.data_fingerprint,
    )
    return {
        "model": entry,
        "manifest_sha256": saved.manifest_sha256,
        "model_sha256": saved.model_sha256,
        "eligible_human_samples": len(dataset.samples),
        "excluded_counts": dataset.excluded_counts,
        "review_ids": list(dataset.review_ids),
        "rule_metrics": rule_metrics,
    }


def load_published_quality_model(
    registry: QualityModelRegistry,
) -> QualityModelBundle:
    """Hash-verify the active package, then and only then deserialize it."""

    state = registry.list_models()
    model_id = state.get("active_model_id")
    if not model_id:
        raise RuntimeError("no published radio-quality model is active")
    entry = state["models"][model_id]
    directory = registry.resolve_verified_bundle()
    return load_quality_model_bundle(
        directory,
        expected_manifest_sha256=str(entry["manifest_sha256"]),
    )


def predict_review_candidate(
    bundle: QualityModelBundle,
    candidate: dict[str, Any],
    *,
    model_id: str,
    bundle_sha256: str,
) -> dict[str, Any]:
    """Create a read-only shadow prediction for one review candidate."""

    features = _numeric_features(candidate.get("features"))
    sample = QualitySample(
        sample_id=str(
            candidate.get("sample_fingerprint")
            or candidate.get("candidate_id")
            or "shadow-sample"
        ),
        observation_id="shadow",
        slot_id=str(candidate.get("time") or candidate.get("slot_index") or "unknown"),
        observed_at=str(candidate.get("time") or "") or None,
        frequency_mhz=_optional_float(candidate.get("frequency_mhz")),
        polarization=str(candidate.get("polarization") or "") or None,
        shape=_shape(candidate, features),
        feature_values=features,
    )
    prediction = predict_quality_model(bundle, sample)
    return {
        "model_id": model_id,
        "bundle_sha256": bundle_sha256,
        "mode": "shadow",
        "probabilities": dict(prediction.probabilities),
        "predicted_label": prediction.predicted_label,
        "ood": prediction.ood,
        "ood_reasons": list(prediction.ood_reasons),
        "top_anomalous_features": list(prediction.top_anomalous_features),
    }


def _locked_rule_metrics(
    dataset: GoldQualityDataset,
    *,
    config: QualityTrainingConfig | None,
) -> dict[str, float | int]:
    minimum_batches = (config or QualityTrainingConfig()).min_observation_batches
    split = split_quality_samples(
        dataset.samples, min_observation_batches=minimum_batches
    )
    test = split.test
    true_bad = [sample.quality_label == "bad" for sample in test]
    rule_bad = [
        dataset.automatic_decisions.get(sample.sample_id) == "bad_candidate"
        for sample in test
    ]
    bad_count = sum(true_bad)
    good_indexes = [
        index for index, sample in enumerate(test) if sample.quality_label == "good"
    ]
    event_indexes = [
        index
        for index, sample in enumerate(test)
        if sample.event_tags and sample.quality_label in {"good", "degraded"}
    ]
    return {
        "bad_recall": (
            sum(predicted and truth for predicted, truth in zip(rule_bad, true_bad))
            / bad_count
            if bad_count
            else math.nan
        ),
        "good_false_rejection_rate": (
            sum(rule_bad[index] for index in good_indexes) / len(good_indexes)
            if good_indexes
            else math.nan
        ),
        "event_false_rejection_rate": (
            sum(rule_bad[index] for index in event_indexes) / len(event_indexes)
            if event_indexes
            else math.nan
        ),
        "event_false_rejection_count": sum(rule_bad[index] for index in event_indexes),
        "test_sample_count": len(test),
    }


def _write_model_card(
    path: Path,
    *,
    model_id: str,
    bundle: QualityModelBundle,
    dataset: GoldQualityDataset,
    metrics: dict[str, Any],
    rule_metrics: dict[str, Any],
) -> None:
    training = bundle.manifest["training"]
    lines = [
        f"# Radio quality model: {model_id}",
        "",
        "Status: candidate / shadow only. This file does not publish the model.",
        "",
        "## Training data",
        "",
        f"- Human-labelled samples: {len(dataset.samples)}",
        f"- Independent reviews: {len(dataset.review_ids)}",
        f"- Data fingerprint: `{dataset.data_fingerprint}`",
        f"- Training observations: {', '.join(training['train_observation_ids'])}",
        f"- Calibration observation: {training['calibration_observation_id']}",
        f"- Locked test observation: {training['test_observation_id']}",
        "- Automatic and skipped labels were excluded from supervision.",
        "",
        "## Locked-test metrics",
        "",
        "```json",
        json.dumps(metrics, indent=2, sort_keys=True),
        "```",
        "",
        "## Existing-rule comparison",
        "",
        "```json",
        json.dumps(rule_metrics, indent=2, sort_keys=True),
        "```",
        "",
        "## Intended use and limitations",
        "",
        "The model supplies calibrated shadow predictions and OOD flags after the",
        "deterministic FITS gate and scientific rules. It never deletes a frame,",
        "overwrites a human label, or becomes active automatically. New batches",
        "must still be reviewed by a person, especially real solar events.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _legacy_human_label(candidate: dict[str, Any]) -> dict[str, Any]:
    decision = str(candidate.get("human_decision") or "").strip().casefold()
    source = str(candidate.get("decision_source") or "").strip().casefold()
    if decision in {"good", "bad"} and source == "human":
        return {
            "quality_label": decision,
            "source": "human_legacy_v1",
            "event_tags": [],
        }
    return {}


def _observation_id(manifest: dict[str, Any], review_id: str) -> str:
    explicit = str(manifest.get("observation_id") or "").strip()
    if explicit:
        return explicit
    input_payload = manifest.get("input")
    root = input_payload.get("root") if isinstance(input_payload, dict) else ""
    digest = hashlib.sha256(str(root).casefold().encode("utf-8")).hexdigest()[:16]
    return f"review-root-{digest or review_id}"


def _numeric_features(payload: Any) -> dict[str, float | None]:
    if not isinstance(payload, dict):
        return {}
    features: dict[str, float | None] = {}
    for key, value in payload.items():
        if isinstance(value, bool):
            features[str(key)] = float(value)
        elif value is None:
            features[str(key)] = None
        elif isinstance(value, (int, float)):
            number = float(value)
            features[str(key)] = number if not math.isinf(number) else None
    return features


def _shape(
    candidate: dict[str, Any], features: dict[str, float | None]
) -> tuple[int, int] | None:
    shape = candidate.get("shape")
    if isinstance(shape, (list, tuple)) and len(shape) == 2:
        return int(shape[0]), int(shape[1])
    height = features.get("image_height")
    width = features.get("image_width")
    if height and width:
        return int(height), int(width)
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"review manifest must be a JSON object: {path}")
    return payload


def _finite_le(value: Any, limit: float) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number <= limit


def _finite_gt(value: Any, baseline: Any) -> bool:
    try:
        number = float(value)
        comparison = float(baseline)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and math.isfinite(comparison) and number > comparison
