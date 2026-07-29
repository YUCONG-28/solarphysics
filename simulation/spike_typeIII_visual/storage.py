"""Chunked, float64 scientific storage for reduced-MHD runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import MHDConfig
from .physics.rmhd import MHDResult, SpectralGrid


class HDF5ArrayProxy:
    """Path-backed array facade that opens only the requested slice."""

    def __init__(self, path: Path, dataset: str, shape: tuple[int, ...]):
        self.path = Path(path)
        self.dataset = dataset
        self.shape = shape
        self.ndim = len(shape)
        self._extrema_cache: tuple[float, float] | None = None

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index):
        h5py = _h5py_module()
        with h5py.File(self.path, "r") as handle:
            return handle[self.dataset][index].astype(np.float64)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self[:], dtype=dtype)

    def _extrema(self) -> tuple[float, float]:
        """Return global extrema using one cached, frame-streamed HDF5 scan."""

        if self._extrema_cache is not None:
            return self._extrema_cache
        if not self.shape or any(size == 0 for size in self.shape):
            raise ValueError("zero-size array has no minimum or maximum")
        h5py = _h5py_module()
        minimum = np.inf
        maximum = -np.inf
        with h5py.File(self.path, "r") as handle:
            dataset = handle[self.dataset]
            for index in range(self.shape[0]):
                frame = dataset[index]
                minimum = min(minimum, float(np.min(frame)))
                maximum = max(maximum, float(np.max(frame)))
        self._extrema_cache = (minimum, maximum)
        return self._extrema_cache

    def min(self) -> float:
        """Return the global minimum without materializing the full array."""

        return self._extrema()[0]

    def max(self) -> float:
        """Return the global maximum without materializing the full array."""

        return self._extrema()[1]


def _h5py_module():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "HDF5 output requires h5py. Install it in solar_simulation or "
            "torch-cuda."
        ) from exc
    return h5py


def write_rmhd_hdf5(
    result: MHDResult,
    config: MHDConfig,
    path: Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write one authoritative float64 RMHD dataset."""

    h5py = _h5py_module()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    spatial_chunk = (
        1,
        min(result.psi.shape[1], 128),
        min(result.psi.shape[2], 256),
    )
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = "spike-typeiii-rmhd-v1"
        handle.attrs["model"] = "2-D incompressible resistive reduced MHD"
        handle.attrs["proxy_boundary"] = (
            "electron beam and radio intensity are kinematic/phenomenological"
        )
        handle.attrs["config_json"] = json.dumps(
            config.__dict__,
            sort_keys=True,
            separators=(",", ":"),
        )
        handle.attrs["execution_backend"] = result.execution_backend
        handle.attrs["execution_device"] = result.execution_device
        handle.attrs["execution_precision"] = result.execution_precision
        handle.attrs["peak_device_memory_bytes"] = result.peak_device_memory_bytes
        if metadata:
            handle.attrs["metadata_json"] = json.dumps(
                metadata,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        grid_group = handle.create_group("grid")
        grid_group.create_dataset("x", data=np.asarray(result.grid.x, dtype=np.float64))
        grid_group.create_dataset("y", data=np.asarray(result.grid.y, dtype=np.float64))
        state_group = handle.create_group("state")
        state_group.create_dataset(
            "time",
            data=np.asarray(result.times, dtype=np.float64),
        )
        for name, values in (("psi", result.psi), ("omega", result.omega)):
            state_group.create_dataset(
                name,
                data=np.asarray(values, dtype=np.float64),
                dtype="f8",
                chunks=spatial_chunk,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
        diagnostics = handle.create_group("diagnostics")
        diagnostic_values = {
            "magnetic_energy": result.magnetic_energy,
            "kinetic_energy": result.kinetic_energy,
            "max_current": result.max_current,
            "max_speed": result.max_speed,
            "reconnection_proxy": result.reconnection_proxy,
            "flux_difference": result.flux_difference,
            "xpoint_electric_field": result.xpoint_electric_field,
            "island_width_proxy": result.island_width_proxy,
            "ohmic_dissipation": result.ohmic_dissipation,
            "viscous_dissipation": result.viscous_dissipation,
            "energy_budget_residual": result.energy_budget_residual,
        }
        for name, values in diagnostic_values.items():
            diagnostics.create_dataset(name, data=np.asarray(values, dtype=np.float64))
        diagnostics.attrs["divergence_normalized_rms"] = result.divergence_rms
    return path


def read_rmhd_hdf5(
    path: Path, *, lazy: bool = False
) -> tuple[MHDResult, MHDConfig, dict[str, Any]]:
    """Load an RMHD HDF5 dataset for deterministic re-rendering."""

    h5py = _h5py_module()
    with h5py.File(Path(path), "r") as handle:
        if handle.attrs.get("schema") != "spike-typeiii-rmhd-v1":
            raise ValueError("Unsupported RMHD HDF5 schema.")
        config = MHDConfig(**json.loads(str(handle.attrs["config_json"])))
        grid = SpectralGrid.from_config(config)
        times = handle["state/time"][...].astype(np.float64)
        if lazy:
            psi = HDF5ArrayProxy(
                Path(path), "state/psi", tuple(handle["state/psi"].shape)
            )
            omega = HDF5ArrayProxy(
                Path(path), "state/omega", tuple(handle["state/omega"].shape)
            )
        else:
            psi = handle["state/psi"][...].astype(np.float64)
            omega = handle["state/omega"][...].astype(np.float64)
        diagnostics = handle["diagnostics"]

        def values(name: str) -> np.ndarray:
            return (
                diagnostics[name][...].astype(np.float64)
                if name in diagnostics
                else np.empty(0, dtype=float)
            )

        result = MHDResult(
            grid=grid,
            times=times,
            psi=psi,
            omega=omega,
            magnetic_energy=values("magnetic_energy"),
            kinetic_energy=values("kinetic_energy"),
            max_current=values("max_current"),
            max_speed=values("max_speed"),
            reconnection_proxy=values("reconnection_proxy"),
            divergence_rms=float(diagnostics.attrs["divergence_normalized_rms"]),
            flux_difference=values("flux_difference"),
            xpoint_electric_field=values("xpoint_electric_field"),
            island_width_proxy=values("island_width_proxy"),
            ohmic_dissipation=values("ohmic_dissipation"),
            viscous_dissipation=values("viscous_dissipation"),
            energy_budget_residual=values("energy_budget_residual"),
            execution_backend=str(handle.attrs.get("execution_backend", "numpy")),
            execution_device=str(handle.attrs.get("execution_device", "cpu")),
            execution_precision=str(
                handle.attrs.get("execution_precision", "float64")
            ),
            peak_device_memory_bytes=int(
                handle.attrs.get("peak_device_memory_bytes", 0)
            ),
        )
        metadata = json.loads(str(handle.attrs.get("metadata_json", "{}")))
    return result, config, metadata
