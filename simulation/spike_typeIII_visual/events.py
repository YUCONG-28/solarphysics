"""Build and validate privacy-safe event constraints for forward modelling."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_WINDOWS_PATH = re.compile(r"\b[A-Za-z]:[\\/]")
_HOME_PATH = re.compile(r"(?:^|[\"'\s])/(?:Users|home)/[^/\s]+/")


def _contains_private_locator(value: object) -> bool:
    if isinstance(value, str):
        return bool(
            _EMAIL.search(value)
            or _WINDOWS_PATH.search(value)
            or _HOME_PATH.search(value)
        )
    if isinstance(value, dict):
        return any(
            _contains_private_locator(key) or _contains_private_locator(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_private_locator(item) for item in value)
    return False


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class DriftConstraint:
    """One reviewed dynamic-spectrum drift segment."""

    drift_id: str
    status: str
    start_utc: str
    end_utc: str
    frequency_start_mhz: float
    frequency_end_mhz: float
    speed_fraction_c: float | None = None
    uncertainty_fraction_c: float | None = None

    def __post_init__(self) -> None:
        if self.status not in {"confirmed", "candidate", "excluded"}:
            raise ValueError("drift status must be confirmed, candidate, or excluded.")
        if min(self.frequency_start_mhz, self.frequency_end_mhz) <= 0.0:
            raise ValueError("drift frequencies must be positive.")
        if self.speed_fraction_c is not None and self.speed_fraction_c <= 0.0:
            raise ValueError("electron-beam speed must be positive.")


@dataclass(frozen=True)
class EventBundle:
    """Sanitized event constraints; never contains source file paths."""

    event_id: str
    core_start_utc: str
    core_end_utc: str
    frequency_range_mhz: tuple[float, float]
    cadence_s: float
    roi: dict[str, float | str]
    data_ids: tuple[str, ...]
    drifts: tuple[DriftConstraint, ...] = ()
    aia_jet: dict[str, Any] = field(default_factory=dict)
    radio_sources: tuple[dict[str, Any], ...] = ()
    notes: tuple[str, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Only EventBundle schema 1 is supported.")
        if not self.event_id or not self.data_ids:
            raise ValueError("event_id and at least one logical data ID are required.")
        low, high = self.frequency_range_mhz
        if low <= 0.0 or high <= low:
            raise ValueError("frequency_range_mhz must be positive and increasing.")
        if self.cadence_s <= 0.0:
            raise ValueError("cadence_s must be positive.")
        _ = self.duration_s
        if _contains_private_locator(asdict(self)):
            raise ValueError(
                "EventBundle contains an email address or a personal absolute path."
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["frequency_range_mhz"] = list(self.frequency_range_mhz)
        payload["data_ids"] = list(self.data_ids)
        payload["notes"] = list(self.notes)
        payload["bundle_sha256"] = _sha256_json(payload)
        return payload

    @property
    def duration_s(self) -> float:
        """Duration of the reviewed UTC core window."""

        start = datetime.fromisoformat(self.core_start_utc)
        end = datetime.fromisoformat(self.core_end_utc)
        duration = (end - start).total_seconds()
        if duration <= 0.0:
            raise ValueError("Event core UTC window must be strictly increasing.")
        return duration

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EventBundle:
        clean = dict(payload)
        expected_hash = clean.pop("bundle_sha256", None)
        drifts = tuple(DriftConstraint(**item) for item in clean.pop("drifts", []))
        frequency_range = tuple(clean.pop("frequency_range_mhz"))
        data_ids = tuple(clean.pop("data_ids"))
        radio_sources = tuple(clean.pop("radio_sources", []))
        notes = tuple(clean.pop("notes", []))
        bundle = cls(
            **clean,
            frequency_range_mhz=frequency_range,
            data_ids=data_ids,
            drifts=drifts,
            radio_sources=radio_sources,
            notes=notes,
        )
        if expected_hash is not None:
            actual = bundle.to_dict()["bundle_sha256"]
            if actual != expected_hash:
                raise ValueError("EventBundle SHA-256 does not match its content.")
        return bundle


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "YAML event configs require PyYAML; JSON works without it."
            ) from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise TypeError("Event configuration must contain one mapping.")
    return data


def build_event_bundle(config_path: Path, output_path: Path) -> EventBundle:
    """Create a reviewed, path-free EventBundle from a public configuration."""

    bundle = EventBundle.from_dict(_load_config(config_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--event-config", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        bundle = build_event_bundle(args.event_config, args.output)
        print(
            json.dumps(
                {
                    "event_id": bundle.event_id,
                    "output": args.output.name,
                    "bundle_sha256": bundle.to_dict()["bundle_sha256"],
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
