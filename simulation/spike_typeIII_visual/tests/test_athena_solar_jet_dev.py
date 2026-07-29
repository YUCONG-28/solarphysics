from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.integrate import quad

SIMULATION_ROOT = Path(__file__).resolve().parents[2]
ATHENA = SIMULATION_ROOT / "fluxrope_demo" / "athena4.2"


def _temperature(y: float, a: float, b: float, ytr: float, width: float) -> float:
    return a + 0.5 * (b - a) * (1.0 + np.tanh((y - ytr) / width))


def _primitive(y: float, a: float, b: float, ytr: float, width: float) -> float:
    u = 2.0 * (y - ytr) / width
    coefficient = (a - b) / (a * b)
    if u >= 0.0:
        return 0.5 * width * (
            u / b + coefficient * (np.log(b) + np.log1p((a / b) * np.exp(-u)))
        )
    return 0.5 * width * (
        u / a + coefficient * (np.log(a) + np.log1p((b / a) * np.exp(u)))
    )


def test_closed_form_hydrostatic_integral() -> None:
    a, b, ytr, width = 0.013333333333, 1.0, 0.25, 0.04
    for y in np.linspace(-0.1, 20.0, 31):
        numerical = quad(
            lambda value: 1.0 / _temperature(value, a, b, ytr, width),
            0.0,
            float(y),
            epsabs=1e-13,
            epsrel=1e-13,
        )[0]
        analytic = _primitive(float(y), a, b, ytr, width) - _primitive(
            0.0, a, b, ytr, width
        )
        np.testing.assert_allclose(analytic, numerical, rtol=1e-12, atol=1e-12)


def test_anomalous_resistivity_and_case_two_are_wired() -> None:
    source = (
        ATHENA / "src" / "prob" / "spike_topping_solar_jet.c"
    ).read_text(encoding="utf-8")
    athinput = (
        ATHENA / "tst" / "2D-mhd" / "athinput.spike_topping_solar_jet"
    ).read_text(encoding="utf-8")
    assert "void get_eta_user" in source
    assert "CASE             = 2" in athinput
    eta_background, eta_anomalous = 1e-5, 2e-4
    threshold = 5.0
    below = eta_background + (eta_anomalous - eta_background) * (
        1.0 - np.exp(-max(4.0 - threshold, 0.0) ** 2)
    )
    above = eta_background + (eta_anomalous - eta_background) * (
        1.0 - np.exp(-max(7.0 - threshold, 0.0) ** 2)
    )
    assert below == eta_background
    assert above > eta_background
    assert above < eta_anomalous


def test_2p5d_conduction_projection_contains_guide_field() -> None:
    source = (ATHENA / "src" / "microphysics" / "conduction.c").read_text(
        encoding="utf-8"
    )
    marker = "A two-dimensional mesh may still evolve all three field"
    assert marker in source
    section = source[source.index(marker) : source.index(marker) + 900]
    assert "SQR(Bz)" in section
