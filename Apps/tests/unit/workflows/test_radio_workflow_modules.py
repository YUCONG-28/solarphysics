"""Preserve worker imports and product ordering across workflow module splits."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from solar_apps.workflows.radio import pipeline_workflow as pipeline
from solar_apps.workflows.radio import source_map_workflow as source_map


def test_pipeline_discovery_does_not_import_scientific_dependencies() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from solar_apps.workflows.radio import pipeline_workflow; "
            "assert not {'numpy', 'pandas', 'matplotlib'} & set(sys.modules); "
            "assert pipeline_workflow._parse_args(['--config', 'event']).config == 'event'",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_selection_module_can_load_first_and_pickled_functions_remain_callable(
    tmp_path: Path,
) -> None:
    (tmp_path / "frame.fits").touch()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pickle, sys; from pathlib import Path; "
            "from solar_apps.workflows.radio import _source_map_selection as selection; "
            "from solar_apps.workflows.radio import source_map_workflow as facade; "
            "assert facade.get_sorted_fits is selection.get_sorted_fits; "
            "fn = pickle.loads(pickle.dumps(facade.get_sorted_fits)); "
            "assert fn(sys.argv[1], 0, None, study_mode='exploratory') == "
            "[str(Path(sys.argv[1]) / 'frame.fits')]",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("newkirk_enabled", [False, True])
def test_pipeline_tables_and_provenance_survive_a_later_plot_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    newkirk_enabled: bool,
) -> None:
    event = {
        "output": {"output_dir": str(tmp_path)},
        "features": {"spectrogram_panel": False},
    }
    newkirk = {"enabled": newkirk_enabled, "multipliers": [1], "harmonics": [1]}
    monkeypatch.setattr(
        pipeline, "load_radio_user_config", lambda name: (event, newkirk)
    )
    monkeypatch.setattr(pipeline, "load_radio_output_config", lambda name: {})
    monkeypatch.setattr(
        pipeline, "load_newkirk_height_comparison_config", lambda name: {}
    )
    monkeypatch.setattr(
        pipeline, "load_drift_selection_product_config", lambda name: {}
    )
    monkeypatch.setattr(
        pipeline, "load_radio_diagnostic_presentation_config", lambda name: {}
    )
    requested_argv = [
        "--config",
        "synthetic-event",
        "--analysis-subdir",
        "products",
        "--gaussian-csv",
        "gaussian.csv",
        "--valid-centers-csv",
        "valid.csv",
        "--newkirk-csv",
        "newkirk.csv",
        "--drift-speed-csv",
        "drift.csv",
    ]

    def source(config, *, argv):
        assert argv == requested_argv
        assert config["enable_gaussian_overlay"] is True
        assert config["save_gaussian_diagnostics"] is True
        output = pipeline.resolve_analysis_dir(config)
        output.mkdir(parents=True)
        pd.DataFrame(
            {
                "time": ["20000101044830", "20000101044831"],
                "freq": [149.0, 164.0],
                "center_x_arcsec": [100.0, 200.0],
                "center_y_arcsec": [200.0, 300.0],
                "quality_flag": ["ok", "rejected"],
                "overlay_valid": [True, False],
                "trajectory_valid": [True, False],
            }
        ).to_csv(output / config["gaussian_diagnostics_csv"], index=False)

    monkeypatch.setattr(source_map, "_run_source_map_config", source)

    def fail_after_tables(frame, output):
        assert len(frame) == 1
        assert output.suffix == ".png"
        assert (tmp_path / "products" / "valid.csv").is_file()
        assert (tmp_path / "products" / "drift.csv").is_file()
        assert (tmp_path / "products" / "newkirk.csv").exists() == newkirk_enabled
        raise RuntimeError("diagnostic figure failed")

    monkeypatch.setattr(pipeline, "_plot_gaussian_center_trajectory", fail_after_tables)
    with pytest.raises(RuntimeError, match="diagnostic figure failed"):
        pipeline.run_pipeline(requested_argv)
    valid = pd.read_csv(tmp_path / "products" / "valid.csv")
    assert valid["freq"].tolist() == [149.0]
    provenance = json.loads(
        (tmp_path / "products" / "radio_run_provenance.json").read_text()
    )
    assert provenance["config_source"] == "synthetic-event"
    assert provenance["cli_overrides"]["gaussian_csv"] == "gaussian.csv"
    if newkirk_enabled:
        extrapolated = pd.read_csv(tmp_path / "products" / "newkirk.csv")
        assert extrapolated["newkirk_multiplier"].tolist() == [1.0]
        assert extrapolated["newkirk_harmonic"].tolist() == [1]
