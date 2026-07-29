"""Explicit SI normalization for event-constrained coronal MHD cases."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

MU_0 = 4.0e-7 * np.pi
PROTON_MASS_KG = 1.67262192595e-27


@dataclass(frozen=True)
class PhysicalNormalization:
    """Reference scales used to convert Athena code units to SI.

    No physical scale is implicit: every event-calibrated run must persist
    these values in its case configuration and bridge metadata.
    """

    length_m: float
    magnetic_field_t: float
    electron_density_m3: float
    mean_mass_per_electron: float = 1.2

    def __post_init__(self) -> None:
        for name in ("length_m", "magnetic_field_t", "electron_density_m3"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if self.mean_mass_per_electron <= 0.0:
            raise ValueError("mean_mass_per_electron must be positive.")

    @classmethod
    def from_solar_units(
        cls,
        *,
        length_mm: float,
        magnetic_field_gauss: float,
        electron_density_cm3: float,
        mean_mass_per_electron: float = 1.2,
    ) -> PhysicalNormalization:
        """Construct from the units used in the solar-jet README."""

        return cls(
            length_m=length_mm * 1.0e6,
            magnetic_field_t=magnetic_field_gauss * 1.0e-4,
            electron_density_m3=electron_density_cm3 * 1.0e6,
            mean_mass_per_electron=mean_mass_per_electron,
        )

    @property
    def mass_density_kg_m3(self) -> float:
        return (
            self.mean_mass_per_electron
            * PROTON_MASS_KG
            * self.electron_density_m3
        )

    @property
    def alfven_speed_m_s(self) -> float:
        return self.magnetic_field_t / np.sqrt(
            MU_0 * self.mass_density_kg_m3
        )

    @property
    def time_s(self) -> float:
        return self.length_m / self.alfven_speed_m_s

    @property
    def pressure_pa(self) -> float:
        return self.magnetic_field_t**2 / MU_0

    def to_metadata(self) -> dict[str, float]:
        result = asdict(self)
        result.update(
            {
                "mass_density_kg_m3": self.mass_density_kg_m3,
                "alfven_speed_m_s": self.alfven_speed_m_s,
                "time_s": self.time_s,
                "pressure_pa": self.pressure_pa,
            }
        )
        return result
