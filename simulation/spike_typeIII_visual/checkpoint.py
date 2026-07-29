"""Atomic, configuration-bound restart files for long RMHD calculations."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import MHDConfig


def configuration_hash(config: MHDConfig, *, engine: str, precision: str) -> str:
    """Return a stable hash for all state-evolution choices."""

    payload = {
        "config": asdict(config),
        "engine": engine,
        "precision": precision,
        "schema": 1,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def save_checkpoint(
    path: Path,
    *,
    config: MHDConfig,
    engine: str,
    precision: str,
    step: int,
    psi: np.ndarray,
    omega: np.ndarray,
    history: dict[str, Any],
) -> None:
    """Write a complete restart state and atomically publish it."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    payload = {
        "schema": np.asarray("spike-typeiii-rmhd-checkpoint-v1"),
        "config_hash": np.asarray(
            configuration_hash(config, engine=engine, precision=precision)
        ),
        "step": np.asarray(step, dtype=np.int64),
        "psi": np.asarray(psi),
        "omega": np.asarray(omega),
    }
    payload.update({key: np.asarray(value) for key, value in history.items()})
    try:
        with temporary_path.open("wb") as stream:
            np.savez(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_checkpoint(
    path: Path,
    *,
    config: MHDConfig,
    engine: str,
    precision: str,
) -> dict[str, np.ndarray | int]:
    """Load a checkpoint only when its schema and configuration match."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path.name}")
    expected = configuration_hash(config, engine=engine, precision=precision)
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["schema"]) != "spike-typeiii-rmhd-checkpoint-v1":
            raise ValueError("Unsupported RMHD checkpoint schema.")
        actual = str(archive["config_hash"])
        if actual != expected:
            raise ValueError(
                "Checkpoint configuration hash does not match this run; "
                "refusing an unsafe resume."
            )
        return {
            key: int(archive[key]) if key == "step" else archive[key].copy()
            for key in archive.files
            if key not in {"schema", "config_hash"}
        }
