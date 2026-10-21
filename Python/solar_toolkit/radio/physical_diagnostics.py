"""Pure physical diagnostics for already-computed radio measurements."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from .newkirk import _positive_parameter, extrapolate_drift_line_with_newkirk

__all__ = [
    "build_drift_newkirk_table",
    "filter_accepted_drift_rows",
]


def filter_accepted_drift_rows(
    drift_rows,
    *,
    quality_column: str = "quality_flag",
    accepted_quality: Iterable[str] = ("ok",),
) -> pd.DataFrame:
    """Return a copy containing only accepted drift-quality rows.

    Tables without a quality column are retained unchanged, matching the
    existing diagnostics behavior for legacy inputs.
    """

    source = pd.DataFrame(drift_rows).copy()
    if quality_column not in source.columns:
        return source
    accepted = {str(value).strip().lower() for value in accepted_quality}
    quality = source[quality_column].astype(str).str.strip().str.lower()
    return source.loc[quality.isin(accepted)].copy()


def build_drift_newkirk_table(
    drift_rows,
    newkirk_config: Mapping[str, Any],
) -> pd.DataFrame:
    """Expand accepted drift rows across finite positive Newkirk model cases.

    Model settings are validated even for an empty or quality-rejected table.
    Invalid observed samples retain the model diagnostic's NaN/reason contract.
    Input tables and configuration mappings are not modified.
    """

    source = filter_accepted_drift_rows(drift_rows)
    multipliers = _model_values(newkirk_config, "multipliers", (1,))
    harmonics = _model_values(newkirk_config, "harmonics", (1,))
    rows: list[dict[str, Any]] = []
    for _, row in source.iterrows():
        for multiplier in multipliers:
            for harmonic in harmonics:
                rows.append(
                    extrapolate_drift_line_with_newkirk(
                        row.to_dict(),
                        multiplier=multiplier,
                        harmonic=harmonic,
                    )
                )
    return pd.DataFrame(rows)


def _model_values(
    config: Mapping[str, Any],
    key: str,
    default: tuple[float, ...],
) -> tuple[Any, ...]:
    values = config.get(key, default)
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        values = (values,)
    return tuple(_positive_parameter(value, key) for value in values)
