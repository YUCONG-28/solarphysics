"""Good-frame-only convolutional autoencoder features for radio images.

This optional second-stage model is intentionally not a classifier.  It is
trained exclusively on explicitly human-confirmed ``good`` frames and emits
only reconstruction and latent-space diagnostics that a separate reviewed
quality model may consume as additional features.

PyTorch is imported lazily by training or inference calls so the public radio
package remains usable in installations without the deep-learning extra.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from importlib import import_module
from typing import Any

import numpy as np

from solar_toolkit._utils.numeric import numeric_data

__all__ = [
    "DEFAULT_MIN_CONFIRMED_GOOD_FRAMES",
    "QUALITY_AUTOENCODER_FEATURE_NAMES",
    "QUALITY_AUTOENCODER_SCHEMA_VERSION",
    "AutoencoderFeatureResult",
    "AutoencoderTrainingConfig",
    "ConfirmedGoodFrameSelection",
    "QualityAutoencoderBundle",
    "QualityAutoencoderDependencyError",
    "QualityAutoencoderSample",
    "QualityAutoencoderValidationError",
    "extract_autoencoder_quality_features",
    "prepare_confirmed_good_frames",
    "train_good_frame_autoencoder",
]

QUALITY_AUTOENCODER_SCHEMA_VERSION = 1
DEFAULT_MIN_CONFIRMED_GOOD_FRAMES = 500
QUALITY_AUTOENCODER_FEATURE_NAMES = (
    "ae_reconstruction_median_abs_error",
    "ae_reconstruction_p95_abs_error",
    "ae_reconstruction_robust_z",
    "ae_latent_robust_distance",
)


class QualityAutoencoderDependencyError(RuntimeError):
    """Raised when an autoencoder operation requires unavailable PyTorch."""


class QualityAutoencoderValidationError(ValueError):
    """Raised when good-frame data or training configuration is invalid."""


@dataclass(frozen=True)
class QualityAutoencoderSample:
    """One image and its review provenance."""

    sample_id: str
    image: np.ndarray
    quality_label: str | None
    label_source: str | None


@dataclass(frozen=True)
class ConfirmedGoodFrameSelection:
    """Eligible human-good frames and auditable exclusion counts."""

    samples: tuple[QualityAutoencoderSample, ...]
    excluded_counts: Mapping[str, int]


@dataclass(frozen=True)
class AutoencoderTrainingConfig:
    """Small convolutional autoencoder training controls."""

    min_confirmed_good_frames: int = DEFAULT_MIN_CONFIRMED_GOOD_FRAMES
    input_size: int = 64
    latent_dim: int = 16
    base_channels: int = 8
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    asinh_clip: float = 5.0
    seed: int = 0
    cpu_threads: int = 1


@dataclass(frozen=True)
class QualityAutoencoderBundle:
    """In-memory autoencoder and robust good-frame reference statistics."""

    model: Any
    config: AutoencoderTrainingConfig
    feature_schema_version: int
    latent_center: np.ndarray
    latent_scale: np.ndarray
    residual_p95_center: float
    residual_p95_scale: float
    training_sample_count: int
    training_fingerprint: str
    epoch_losses: tuple[float, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AutoencoderFeatureResult:
    """Additional numerical features; deliberately contains no class decision."""

    reconstruction_median_abs_error: float
    reconstruction_p95_abs_error: float
    reconstruction_robust_z: float
    latent_robust_distance: float

    def as_feature_values(self) -> dict[str, float]:
        """Return stable names suitable for appending to a feature table."""

        return {
            QUALITY_AUTOENCODER_FEATURE_NAMES[0]: (
                self.reconstruction_median_abs_error
            ),
            QUALITY_AUTOENCODER_FEATURE_NAMES[1]: self.reconstruction_p95_abs_error,
            QUALITY_AUTOENCODER_FEATURE_NAMES[2]: self.reconstruction_robust_z,
            QUALITY_AUTOENCODER_FEATURE_NAMES[3]: self.latent_robust_distance,
        }


def prepare_confirmed_good_frames(
    samples: Iterable[QualityAutoencoderSample | Mapping[str, Any]],
) -> ConfirmedGoodFrameSelection:
    """Select only explicit ``label_source=human`` and ``quality_label=good``.

    Automatic, skipped, pending, degraded, uncertain, and bad rows are never
    promoted to a good-frame training target.  Exclusions are counted so the
    caller can persist the exact selection audit.
    """

    accepted: list[QualityAutoencoderSample] = []
    excluded: Counter[str] = Counter()
    seen_ids: set[str] = set()

    for raw_sample in samples:
        sample = _coerce_sample(raw_sample)
        source = _normalize_text(sample.label_source)
        label = _normalize_text(sample.quality_label)

        if source == "automatic_on_skip":
            excluded["automatic_on_skip"] += 1
            continue
        if source != "human":
            excluded["non_human"] += 1
            continue
        if label != "good":
            excluded["quality_not_good"] += 1
            continue
        if not sample.sample_id.strip():
            raise QualityAutoencoderValidationError("sample_id is required")
        if sample.sample_id in seen_ids:
            raise QualityAutoencoderValidationError(
                f"duplicate confirmed-good sample_id {sample.sample_id!r}"
            )
        _validated_image(sample.image, sample_id=sample.sample_id)
        seen_ids.add(sample.sample_id)
        accepted.append(replace(sample, quality_label="good", label_source="human"))

    return ConfirmedGoodFrameSelection(
        samples=tuple(accepted),
        excluded_counts=dict(sorted(excluded.items())),
    )


def train_good_frame_autoencoder(
    samples: Iterable[QualityAutoencoderSample | Mapping[str, Any]],
    *,
    config: AutoencoderTrainingConfig | None = None,
) -> QualityAutoencoderBundle:
    """Fit a deterministic CPU autoencoder using confirmed good frames only."""

    cfg = config or AutoencoderTrainingConfig()
    _validate_training_config(cfg)
    selection = prepare_confirmed_good_frames(samples)
    if len(selection.samples) < cfg.min_confirmed_good_frames:
        raise QualityAutoencoderValidationError(
            "good-frame autoencoder requires at least "
            f"{cfg.min_confirmed_good_frames} confirmed human-good frames; got "
            f"{len(selection.samples)}"
        )

    torch = _require_torch()
    previous_threads = int(torch.get_num_threads())
    previous_deterministic = bool(torch.are_deterministic_algorithms_enabled())
    try:
        torch.set_num_threads(cfg.cpu_threads)
        torch.manual_seed(cfg.seed)
        torch.use_deterministic_algorithms(True)
        model = _build_model(torch, cfg).to("cpu")
        training_tensor = _image_tensor(
            torch,
            [sample.image for sample in selection.samples],
            cfg,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(cfg.seed)
        epoch_losses: list[float] = []

        model.train()
        for _ in range(cfg.epochs):
            permutation = torch.randperm(
                len(training_tensor),
                generator=generator,
            )
            total_loss = 0.0
            total_items = 0
            for start in range(0, len(permutation), cfg.batch_size):
                indices = permutation[start : start + cfg.batch_size]
                batch = training_tensor[indices]
                optimizer.zero_grad(set_to_none=True)
                reconstruction, _ = model(batch)
                loss = torch.nn.functional.smooth_l1_loss(
                    reconstruction,
                    batch,
                    reduction="mean",
                )
                loss.backward()
                optimizer.step()
                batch_size = int(batch.shape[0])
                total_loss += float(loss.detach().cpu()) * batch_size
                total_items += batch_size
            epoch_losses.append(total_loss / max(total_items, 1))

        model.eval()
        with torch.no_grad():
            reconstruction, latent = model(training_tensor)
            absolute_residual = torch.abs(reconstruction - training_tensor)
            flattened = absolute_residual.flatten(start_dim=1)
            residual_p95 = torch.quantile(flattened, 0.95, dim=1)
        latent_values = latent.detach().cpu().numpy().astype(np.float64, copy=False)
        residual_values = (
            residual_p95.detach().cpu().numpy().astype(np.float64, copy=False)
        )
        latent_center, latent_scale = _robust_axis_reference(latent_values)
        residual_center, residual_scale = _robust_scalar_reference(residual_values)
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.set_num_threads(previous_threads)

    return QualityAutoencoderBundle(
        model=model,
        config=cfg,
        feature_schema_version=QUALITY_AUTOENCODER_SCHEMA_VERSION,
        latent_center=latent_center,
        latent_scale=latent_scale,
        residual_p95_center=residual_center,
        residual_p95_scale=residual_scale,
        training_sample_count=len(selection.samples),
        training_fingerprint=_training_fingerprint(selection.samples, cfg),
        epoch_losses=tuple(epoch_losses),
    )


def extract_autoencoder_quality_features(
    bundle: QualityAutoencoderBundle,
    image: np.ndarray,
) -> AutoencoderFeatureResult:
    """Compute reconstruction and latent diagnostics without classifying."""

    _validate_bundle(bundle)
    torch = _require_torch()
    tensor = _image_tensor(torch, [image], bundle.config)
    bundle.model.eval()
    with torch.no_grad():
        reconstruction, latent = bundle.model(tensor)
        residual = torch.abs(reconstruction - tensor).flatten(start_dim=1)[0]
        residual_median = float(torch.median(residual).cpu())
        residual_p95 = float(torch.quantile(residual, 0.95).cpu())
    latent_values = latent[0].detach().cpu().numpy().astype(np.float64, copy=False)
    latent_z = (latent_values - bundle.latent_center) / bundle.latent_scale
    latent_distance = float(np.sqrt(np.mean(np.square(latent_z))))
    robust_residual_z = float(
        (residual_p95 - bundle.residual_p95_center) / bundle.residual_p95_scale
    )
    values = (
        residual_median,
        residual_p95,
        robust_residual_z,
        latent_distance,
    )
    if not np.isfinite(values).all():
        raise QualityAutoencoderValidationError(
            "autoencoder produced non-finite quality features"
        )
    return AutoencoderFeatureResult(
        reconstruction_median_abs_error=residual_median,
        reconstruction_p95_abs_error=residual_p95,
        reconstruction_robust_z=robust_residual_z,
        latent_robust_distance=latent_distance,
    )


def _coerce_sample(
    raw: QualityAutoencoderSample | Mapping[str, Any],
) -> QualityAutoencoderSample:
    if isinstance(raw, QualityAutoencoderSample):
        return raw
    if not isinstance(raw, Mapping):
        raise TypeError(
            "autoencoder samples must be QualityAutoencoderSample or mappings"
        )
    if "image" not in raw:
        raise QualityAutoencoderValidationError("sample image is required")
    return QualityAutoencoderSample(
        sample_id=str(raw.get("sample_id", "")).strip(),
        image=numeric_data(raw["image"]),
        quality_label=raw.get("quality_label", raw.get("human_label")),
        label_source=raw.get("label_source", raw.get("decision_source")),
    )


def _validate_training_config(config: AutoencoderTrainingConfig) -> None:
    integer_values = {
        "min_confirmed_good_frames": config.min_confirmed_good_frames,
        "input_size": config.input_size,
        "latent_dim": config.latent_dim,
        "base_channels": config.base_channels,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "cpu_threads": config.cpu_threads,
    }
    if any(value <= 0 for value in integer_values.values()):
        raise QualityAutoencoderValidationError(
            "autoencoder integer settings must be positive"
        )
    if config.input_size < 8 or config.input_size % 4:
        raise QualityAutoencoderValidationError(
            "input_size must be at least 8 and divisible by 4"
        )
    if config.learning_rate <= 0 or config.weight_decay < 0:
        raise QualityAutoencoderValidationError(
            "invalid autoencoder optimization settings"
        )
    if config.asinh_clip <= 0:
        raise QualityAutoencoderValidationError("asinh_clip must be positive")


def _validated_image(image: np.ndarray, *, sample_id: str = "inference") -> np.ndarray:
    # Keep masks and input precision until the explicit model normalization.
    array = np.squeeze(numeric_data(image))
    if array.ndim != 2:
        raise QualityAutoencoderValidationError(
            f"sample {sample_id!r} must contain one 2D image"
        )
    if array.size == 0 or not np.isfinite(array).any():
        raise QualityAutoencoderValidationError(
            f"sample {sample_id!r} has no finite image pixels"
        )
    return array


def _normalize_image(image: np.ndarray, *, clip: float) -> np.ndarray:
    array = _validated_image(image).astype(np.float64, copy=False)
    finite = np.isfinite(array)
    finite_values = array[finite]
    center = float(np.median(finite_values))
    scale = float(1.4826 * np.median(np.abs(finite_values - center)))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(finite_values))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    filled = np.where(finite, array, center)
    normalized = np.arcsinh((filled - center) / scale)
    return (np.clip(normalized, -clip, clip) / clip).astype(
        np.float32,
        copy=False,
    )


def _image_tensor(
    torch: Any,
    images: Sequence[np.ndarray],
    config: AutoencoderTrainingConfig,
) -> Any:
    normalized = np.stack(
        [_normalize_image(image, clip=config.asinh_clip) for image in images]
    )
    tensor = torch.from_numpy(normalized[:, None, :, :]).to(
        device="cpu",
        dtype=torch.float32,
    )
    if tuple(tensor.shape[-2:]) != (config.input_size, config.input_size):
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(config.input_size, config.input_size),
            mode="bilinear",
            align_corners=False,
        )
    return tensor.contiguous()


def _build_model(torch: Any, config: AutoencoderTrainingConfig) -> Any:
    channels = config.base_channels
    reduced_size = config.input_size // 4
    encoded_values = channels * 4 * reduced_size * reduced_size

    class LightweightConvAutoencoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(1, channels, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(
                    channels,
                    channels * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(
                    channels * 2,
                    channels * 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                torch.nn.ReLU(inplace=True),
            )
            self.to_latent = torch.nn.Linear(encoded_values, config.latent_dim)
            self.from_latent = torch.nn.Linear(config.latent_dim, encoded_values)
            self.decoder = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(
                    channels * 4,
                    channels * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                torch.nn.ReLU(inplace=True),
                torch.nn.ConvTranspose2d(
                    channels * 2,
                    channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(channels, 1, kernel_size=3, padding=1),
                torch.nn.Tanh(),
            )

        def encode(self, value: Any) -> Any:
            encoded = self.encoder(value)
            return self.to_latent(encoded.flatten(start_dim=1))

        def forward(self, value: Any) -> tuple[Any, Any]:
            latent = self.encode(value)
            decoded = self.from_latent(latent).reshape(
                -1,
                channels * 4,
                reduced_size,
                reduced_size,
            )
            return self.decoder(decoded), latent

    return LightweightConvAutoencoder()


def _robust_axis_reference(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - center), axis=0)
    standard_deviation = np.std(values, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, standard_deviation)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return center.astype(np.float64), scale.astype(np.float64)


def _robust_scalar_reference(values: np.ndarray) -> tuple[float, float]:
    center = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - center)))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return center, scale


def _training_fingerprint(
    samples: Sequence[QualityAutoencoderSample],
    config: AutoencoderTrainingConfig,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        (
            f"schema={QUALITY_AUTOENCODER_SCHEMA_VERSION};"
            f"size={config.input_size};latent={config.latent_dim};"
            f"seed={config.seed}"
        ).encode()
    )
    for sample in sorted(samples, key=lambda item: item.sample_id):
        normalized = _normalize_image(sample.image, clip=config.asinh_clip)
        digest.update(sample.sample_id.encode("utf-8"))
        digest.update(str(normalized.shape).encode("ascii"))
        digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def _validate_bundle(bundle: QualityAutoencoderBundle) -> None:
    if not isinstance(bundle, QualityAutoencoderBundle):
        raise TypeError("bundle must be a QualityAutoencoderBundle")
    if bundle.feature_schema_version != QUALITY_AUTOENCODER_SCHEMA_VERSION:
        raise QualityAutoencoderValidationError(
            "unsupported autoencoder feature schema"
        )
    if bundle.latent_center.shape != (bundle.config.latent_dim,):
        raise QualityAutoencoderValidationError("invalid latent center shape")
    if bundle.latent_scale.shape != (bundle.config.latent_dim,):
        raise QualityAutoencoderValidationError("invalid latent scale shape")
    if not np.isfinite(bundle.latent_center).all():
        raise QualityAutoencoderValidationError("latent center is not finite")
    if not np.isfinite(bundle.latent_scale).all() or np.any(bundle.latent_scale <= 0):
        raise QualityAutoencoderValidationError(
            "latent scale must be finite and positive"
        )
    if not math.isfinite(bundle.residual_p95_center):
        raise QualityAutoencoderValidationError("residual reference is not finite")
    if not math.isfinite(bundle.residual_p95_scale) or bundle.residual_p95_scale <= 0:
        raise QualityAutoencoderValidationError(
            "residual reference scale must be finite and positive"
        )
    if not callable(getattr(bundle.model, "forward", None)):
        raise QualityAutoencoderValidationError("autoencoder bundle model is invalid")


def _require_torch() -> Any:
    try:
        return import_module("torch")
    except ImportError as exc:
        raise QualityAutoencoderDependencyError(
            "radio quality autoencoder requires the optional PyTorch dependency"
        ) from exc


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().casefold()
