from __future__ import annotations

from pathlib import Path


def test_observation_file_inventory_rows():
    from solar_toolkit.data import ObservationFile, build_inventory

    inventory = build_inventory(
        [
            ObservationFile(
                path=Path("aia.fits"),
                instrument="AIA",
                wavelength="211",
                obs_time="2000-01-01T04:48:37",
            )
        ]
    )

    assert inventory.to_dict("records") == [
        {
            "path": "aia.fits",
            "instrument": "AIA",
            "wavelength": "211",
            "obs_time": "2000-01-01T04:48:37",
        }
    ]
