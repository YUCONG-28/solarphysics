"""Configuration objects for the visual simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class MHDConfig:
    """Normalized two-dimensional reduced-MHD parameters."""

    nx: int = 96
    ny: int = 96
    lx: float = 4.0 * 3.141592653589793
    ly: float = 2.0 * 3.141592653589793
    sheet_half_width: float = 0.20
    sheet_center_fraction: float = 0.25
    perturbation_amplitude: float = 0.04
    perturbation_width: float = 0.45
    resistivity: float = 0.002
    viscosity: float = 0.002
    dt: float = 0.005
    steps: int = 400
    snapshot_stride: int = 10
    lorentz_convention: str = "physical"

    def __post_init__(self) -> None:
        if self.nx < 8 or self.ny < 8:
            raise ValueError("nx and ny must both be at least 8.")
        if self.nx % 2 or self.ny % 2:
            raise ValueError("nx and ny must be even for the spectral grid.")
        if self.dt <= 0.0 or self.steps < 1 or self.snapshot_stride < 1:
            raise ValueError("dt, steps, and snapshot_stride must be positive.")
        if self.lorentz_convention not in {"physical", "legacy"}:
            raise ValueError("lorentz_convention must be 'physical' or 'legacy'.")


@dataclass(frozen=True)
class RadioConfig:
    """Physical proxy parameters used after the normalized MHD run."""

    start_frequency_mhz: float = 300.0
    min_frequency_mhz: float = 20.0
    max_frequency_mhz: float = 350.0
    density_scale_height_mm: float = 50.0
    beam_speed_fraction_c: float = 0.20
    duration_s: float = 4.5
    time_samples: int = 180
    frequency_samples: int = 256
    spike_count: int = 12
    spike_onset_start_s: float = 0.08
    spike_onset_fraction: float = 0.25
    spike_onset_cap_s: float = 0.75
    spike_frequency_offset_min_mhz: float = 5.0
    spike_frequency_offset_max_mhz: float = 40.0

    def __post_init__(self) -> None:
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive.")
        if not 0.0 < self.spike_onset_fraction <= 1.0:
            raise ValueError("spike_onset_fraction must be in (0, 1].")
        if self.spike_onset_start_s < 0.0:
            raise ValueError("spike_onset_start_s must be non-negative.")
        if self.spike_onset_cap_s <= 0.0:
            raise ValueError("spike_onset_cap_s must be positive.")
        onset_end = min(
            self.spike_onset_cap_s,
            self.duration_s * self.spike_onset_fraction,
        )
        if self.spike_onset_start_s >= onset_end:
            raise ValueError(
                "Spike onset window is empty; require spike_onset_start_s "
                "< min(spike_onset_cap_s, duration_s * spike_onset_fraction)."
            )
        if self.spike_frequency_offset_min_mhz <= 0.0:
            raise ValueError("spike_frequency_offset_min_mhz must be positive.")
        if self.spike_frequency_offset_max_mhz <= self.spike_frequency_offset_min_mhz:
            raise ValueError(
                "spike_frequency_offset_max_mhz must exceed "
                "spike_frequency_offset_min_mhz."
            )
        if not (
            self.min_frequency_mhz < self.start_frequency_mhz < self.max_frequency_mhz
        ):
            raise ValueError(
                "start_frequency_mhz must lie strictly inside the frequency band."
            )


@dataclass(frozen=True)
class JetConfig:
    """Sheet-localized bidirectional-outflow diagnostic parameters."""

    sheet_half_width_factor: float = 2.0
    velocity_quantile: float = 0.95
    jet_threshold: float = 0.60
    reconnection_threshold: float = 0.60
    consecutive_snapshots: int = 3
    xpoint_half_window_fraction: float = 0.125

    def __post_init__(self) -> None:
        if self.sheet_half_width_factor <= 0.0:
            raise ValueError("sheet_half_width_factor must be positive.")
        if not 0.5 < self.velocity_quantile < 1.0:
            raise ValueError("velocity_quantile must be in (0.5, 1).")
        if not 0.0 <= self.jet_threshold <= 1.0:
            raise ValueError("jet_threshold must be in [0, 1].")
        if not 0.0 <= self.reconnection_threshold <= 1.0:
            raise ValueError("reconnection_threshold must be in [0, 1].")
        if self.consecutive_snapshots < 1:
            raise ValueError("consecutive_snapshots must be at least 1.")
        if not 0.0 < self.xpoint_half_window_fraction <= 0.25:
            raise ValueError("xpoint_half_window_fraction must be in (0, 0.25].")


@dataclass(frozen=True)
class TimeCalibrationConfig:
    """Conversion from normalized MHD time to radio-proxy seconds."""

    mode: str = "proxy"
    length_scale_mm: float | None = None
    magnetic_field_gauss: float | None = None
    electron_density_cm3: float | None = None
    mean_mass_per_electron: float = 1.2

    def __post_init__(self) -> None:
        if self.mode not in {"proxy", "alfven", "event"}:
            raise ValueError(
                "time calibration mode must be 'proxy', 'alfven', or 'event'."
            )
        scales = (
            self.length_scale_mm,
            self.magnetic_field_gauss,
            self.electron_density_cm3,
        )
        if self.mode == "alfven":
            if any(value is None or value <= 0.0 for value in scales):
                raise ValueError(
                    "Alfvén calibration requires positive L0, B0, and ne0."
                )
        elif any(value is not None for value in scales):
            raise ValueError(
                "Physical scales are accepted only with Alfvén calibration."
            )
        if self.mean_mass_per_electron <= 0.0:
            raise ValueError("mean_mass_per_electron must be positive.")


@dataclass(frozen=True)
class RunConfig:
    """Complete deterministic run configuration."""

    profile: str
    seed: int
    mhd: MHDConfig
    radio: RadioConfig
    jet: JetConfig
    time_calibration: TimeCalibrationConfig = field(
        default_factory=TimeCalibrationConfig
    )
    spike_coupling: str = "jet"

    def __post_init__(self) -> None:
        if self.spike_coupling not in {"jet", "uniform"}:
            raise ValueError("spike_coupling must be 'jet' or 'uniform'.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_config(
    profile: str,
    seed: int,
    *,
    lorentz_convention: str = "physical",
    spike_coupling: str = "jet",
    jet: JetConfig | None = None,
    time_calibration: TimeCalibrationConfig | None = None,
) -> RunConfig:
    """Return a named deterministic resolution profile."""

    key = profile.strip().lower()
    # The historical ``cuda-*`` names describe the Windows provenance, not a
    # hardware requirement.  Generic ``rmhd-*`` names select identical
    # scientific configurations on CPU or CUDA.
    legacy_aliases = {
        "rmhd-coarse": "cuda-coarse",
        "rmhd-medium": "cuda-medium",
        "rmhd-fine": "cuda-fine",
        "rmhd-medium-event": "cuda-medium-event",
        "rmhd-fine-event": "cuda-fine-event",
        "rmhd-fine-control": "cuda-fine-control",
    }
    resolved_key = legacy_aliases.get(key, key)
    if resolved_key == "standard":
        mhd = MHDConfig(lorentz_convention=lorentz_convention)
        radio = RadioConfig()
    elif resolved_key == "quick":
        mhd = MHDConfig(
            nx=48,
            ny=48,
            steps=80,
            snapshot_stride=2,
            lorentz_convention=lorentz_convention,
        )
        radio = RadioConfig(
            time_samples=96,
            frequency_samples=128,
            spike_count=8,
        )
    elif resolved_key == "cuda-coarse":
        mhd = MHDConfig(
            nx=256,
            ny=128,
            dt=0.005,
            steps=400,
            snapshot_stride=1,
            lorentz_convention=lorentz_convention,
        )
        radio = RadioConfig(time_samples=1441, frequency_samples=1024)
    elif resolved_key == "cuda-medium":
        mhd = MHDConfig(
            nx=512,
            ny=256,
            dt=0.0025,
            steps=800,
            snapshot_stride=2,
            lorentz_convention=lorentz_convention,
        )
        radio = RadioConfig(time_samples=1441, frequency_samples=1024)
    elif resolved_key == "cuda-fine":
        mhd = MHDConfig(
            nx=1024,
            ny=512,
            dt=0.00125,
            steps=1600,
            snapshot_stride=4,
            lorentz_convention=lorentz_convention,
        )
        radio = RadioConfig(time_samples=1441, frequency_samples=1024)
    elif resolved_key == "cuda-medium-event":
        mhd = MHDConfig(
            nx=512,
            ny=256,
            dt=0.0025,
            steps=3200,
            snapshot_stride=8,
            lorentz_convention=lorentz_convention,
        )
        radio = RadioConfig(time_samples=1441, frequency_samples=1024)
    elif resolved_key == "cuda-fine-event":
        mhd = MHDConfig(
            nx=1024,
            ny=512,
            dt=0.00125,
            steps=6400,
            snapshot_stride=16,
            lorentz_convention=lorentz_convention,
        )
        radio = RadioConfig(time_samples=1441, frequency_samples=1024)
    elif resolved_key == "cuda-fine-control":
        mhd = MHDConfig(
            nx=1024,
            ny=512,
            dt=0.00125,
            steps=6400,
            snapshot_stride=16,
            perturbation_amplitude=0.0,
            lorentz_convention=lorentz_convention,
        )
        radio = RadioConfig(time_samples=1441, frequency_samples=1024)
    else:
        raise ValueError(
            f"Unknown profile {profile!r}; use quick, standard, "
            "cuda-coarse, cuda-medium, cuda-fine, cuda-medium-event, "
            "cuda-fine-event, cuda-fine-control, or the corresponding "
            "rmhd-* hardware-neutral name."
        )
    return RunConfig(
        profile=key,
        seed=int(seed),
        mhd=mhd,
        radio=radio,
        jet=jet or JetConfig(),
        time_calibration=time_calibration or TimeCalibrationConfig(),
        spike_coupling=spike_coupling,
    )
