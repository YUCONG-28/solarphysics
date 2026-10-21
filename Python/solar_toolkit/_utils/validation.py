"""Internal validation and legacy configuration helpers."""

from __future__ import annotations

import datetime as dt
import json
import math
from typing import Any

import yaml

from ..time import normalize_time


def validate_time_range(start_time: dt.datetime, end_time: dt.datetime) -> bool:
    """Validate strictly increasing complete times on the integer TAI clock.

    Naive datetimes use UTC. Aware datetimes and explicitly scaled timestamps
    use the shared time normalizer rather than mixed timezone arithmetic.
    """

    start_ns = normalize_time(start_time).tai_ns
    end_ns = normalize_time(end_time).tai_ns
    if start_ns >= end_ns:
        raise ValueError(
            f"Start time ({start_time}) must precede end time ({end_time})"
        )
    return True


def validate_frequency_range(f_start: float, f_end: float) -> bool:
    """Validate a strictly increasing, finite positive frequency range in MHz."""

    try:
        if isinstance(f_start, bool) or isinstance(f_end, bool):
            raise ValueError("boolean frequency")
        f_start, f_end = float(f_start), float(f_end)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Frequencies must be finite positive numbers in MHz") from exc
    if not all(math.isfinite(value) and value > 0 for value in (f_start, f_end)):
        raise ValueError("Frequencies must be finite positive numbers in MHz")
    if f_start >= f_end:
        raise ValueError(
            f"Start frequency ({f_start} MHz) must be less than end frequency "
            f"({f_end} MHz)"
        )
    return True


class SolarDataConfig:
    """Legacy workflow configuration; callers must supply paths and ROI bounds.

    Data/output directories and the scientific ROI default to ``None``. This
    compatibility container preserves explicit caller settings without guessing
    a machine layout or reusing a historical event's region.
    """

    def __init__(self, config_dict: dict[str, Any] | None = None):
        self.defaults = {
            "data_dir": None,
            "output_dir": None,
            "roi_bounds": None,
            "dpi": 300,
            "fig_width": 10.0,
            "use_parallel": True,
            "max_workers": None,
            "chunk_mem_mb": 50,
            "save_images": True,
            "show_images": False,
        }
        if config_dict:
            self.defaults.update(config_dict)
        for key, value in self.defaults.items():
            setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        """Return the current configuration as a dictionary."""

        return {key: getattr(self, key) for key in self.defaults}

    def save_to_file(self, filepath: str) -> None:
        """Save configuration to a JSON or YAML file."""

        config_dict = self.to_dict()
        if filepath.endswith(".json"):
            with open(filepath, "w", encoding="utf-8") as handle:
                json.dump(config_dict, handle, indent=2, ensure_ascii=False)
        elif filepath.endswith((".yaml", ".yml")):
            with open(filepath, "w", encoding="utf-8") as handle:
                yaml.safe_dump(
                    config_dict,
                    handle,
                    default_flow_style=False,
                    allow_unicode=True,
                )
        else:
            raise ValueError(f"Unsupported configuration file format: {filepath}")

    @classmethod
    def load_from_file(cls, filepath: str) -> SolarDataConfig:
        """Load configuration from a JSON or YAML file."""

        if filepath.endswith(".json"):
            with open(filepath, encoding="utf-8") as handle:
                config_dict = json.load(handle)
        elif filepath.endswith((".yaml", ".yml")):
            with open(filepath, encoding="utf-8") as handle:
                config_dict = yaml.safe_load(handle)
        else:
            raise ValueError(f"Unsupported configuration file format: {filepath}")
        return cls(config_dict)


__all__ = ["SolarDataConfig", "validate_frequency_range", "validate_time_range"]
