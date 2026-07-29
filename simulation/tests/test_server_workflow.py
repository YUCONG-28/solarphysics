from __future__ import annotations

import tarfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from spike_typeIII_visual.config import profile_config
from spike_typeIII_visual.physics.rmhd import solve_rmhd
from spike_typeIII_visual.server import package, render_scripts
from spike_typeIII_visual.storage import (
    HDF5ArrayProxy,
    read_rmhd_hdf5,
    write_rmhd_hdf5,
)


def _tiny_config():
    return replace(
        profile_config("quick", 20260726).mhd,
        steps=4,
        snapshot_stride=1,
    )


def test_hardware_neutral_profiles_match_legacy_cuda_profiles():
    for generic, legacy in (
        ("rmhd-medium-event", "cuda-medium-event"),
        ("rmhd-fine-event", "cuda-fine-event"),
        ("rmhd-fine-control", "cuda-fine-control"),
    ):
        assert profile_config(generic, 7).mhd == profile_config(legacy, 7).mhd
        assert profile_config(generic, 7).profile == generic


def test_numpy_checkpoint_resume_and_configuration_rejection(tmp_path: Path):
    config = _tiny_config()
    checkpoint = tmp_path / "restart.npz"
    solve_rmhd(
        config,
        checkpoint_path=checkpoint,
        checkpoint_every=2,
        stop_after_step=2,
    )
    resumed = solve_rmhd(
        config,
        checkpoint_path=checkpoint,
        checkpoint_every=2,
        resume=True,
    )
    uninterrupted = solve_rmhd(config)
    np.testing.assert_allclose(resumed.psi, uninterrupted.psi, rtol=0, atol=0)
    np.testing.assert_allclose(resumed.omega, uninterrupted.omega, rtol=0, atol=0)
    with pytest.raises(ValueError, match="configuration hash"):
        solve_rmhd(
            replace(config, resistivity=config.resistivity * 2),
            checkpoint_path=checkpoint,
            resume=True,
        )


def test_hdf5_lazy_reader_loads_one_snapshot(tmp_path: Path):
    config = _tiny_config()
    result = solve_rmhd(config)
    path = write_rmhd_hdf5(result, config, tmp_path / "fields.h5")
    eager, _, _ = read_rmhd_hdf5(path)
    lazy, _, _ = read_rmhd_hdf5(path, lazy=True)
    assert isinstance(lazy.psi, HDF5ArrayProxy)
    assert lazy.psi.shape == eager.psi.shape
    np.testing.assert_array_equal(lazy.psi[2], eager.psi[2])
    assert lazy.psi.min() == np.min(eager.psi)
    assert lazy.psi.max() == np.max(eager.psi)


def test_gridview_templates_are_single_process_and_private_neutral(tmp_path: Path):
    render_scripts(tmp_path)
    combined = b"\n".join(path.read_bytes() for path in sorted(tmp_path.iterdir()))
    assert b"mpirun" not in combined
    assert b"/Users/" not in combined
    assert b":\\Users\\" not in combined
    assert b"192.168." not in combined
    assert b"$HOME" in combined or b"${HOME}" in combined


def test_report_lite_excludes_hdf5_logs_and_private_configuration(tmp_path: Path):
    run = tmp_path / "run"
    (run / "figures").mkdir(parents=True)
    (run / "data").mkdir()
    (run / "figures" / "plot.png").write_bytes(b"png")
    (run / "data" / "summary.json").write_text('{"passed":true}\n')
    (run / "data" / "fields.h5").write_bytes(b"large")
    (run / "raw.log").write_text("private runtime log\n")
    output = tmp_path / "report-lite.tar.gz"
    assert package(run, "report-lite", output) == 0
    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
    assert "figures/plot.png" in names
    assert "data/summary.json" in names
    assert "data/fields.h5" not in names
    assert "raw.log" not in names
    assert "PACKAGE_SHA256SUMS.txt" in names
