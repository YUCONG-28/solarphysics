"""Legacy CSO workflow entry points share the scientific numerical kernel."""

import numpy as np
import pytest
from astropy import units as u

from solar_apps.workflows.radio import cso_workflow


def test_ratio_delegates_and_retains_positional_eps(monkeypatch):
    calls = []
    sentinel = np.array([0.25], dtype=np.float32)

    def ratio(right, left, *, eps):
        calls.append((right, left, eps))
        return sentinel

    monkeypatch.setattr(cso_workflow._canonical_cso, "calc_polarization_ratio", ratio)
    assert cso_workflow.calc_polarization_ratio([3], [1], 1e-12) is sentinel
    assert calls == [([3], [1], 1e-12)]


def test_workflow_ratio_has_no_integer_overflow_and_respects_units_masks():
    result = cso_workflow.calc_polarization_ratio(
        np.array([10], dtype=np.uint8), np.array([20], dtype=np.uint8)
    )
    np.testing.assert_allclose(result, [-1 / 3], rtol=1e-7)
    np.testing.assert_equal(
        cso_workflow.calc_polarization_ratio([1] * u.Jy, [1000] * u.mJy), [0]
    )
    with pytest.raises(u.UnitConversionError):
        cso_workflow.calc_polarization_ratio([1] * u.Jy, [1] * u.s)
    np.testing.assert_equal(
        cso_workflow.calc_polarization_ratio(np.ma.array([9], mask=True), [1]),
        [np.nan],
    )


def test_workflow_log_delegates_mask_handling_and_keeps_invalid_values_missing(
    monkeypatch,
):
    data = np.ma.array([1, 1e20, -1], mask=[0, 1, 0])
    np.testing.assert_equal(cso_workflow._safe_log10(data), [0, np.nan, np.nan])
    sentinel = np.array([123])
    monkeypatch.setattr(
        cso_workflow._canonical_cso, "safe_log10", lambda value: sentinel
    )
    assert cso_workflow._safe_log10(data) is sentinel
