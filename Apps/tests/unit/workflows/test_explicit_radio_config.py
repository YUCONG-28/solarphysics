"""Radio commands require explicit configuration and keep JSON data-only."""

import json

import pytest

from solar_apps.workflows.radio import (
    overlay_cli,
    pipeline_cli,
    quicklook,
    raw_quality_cli,
    source_map_cli,
)
from solar_apps.workflows.radio.entrypoint_utils import (
    build_common_parser,
    load_json_config,
    resolve_config_source,
)
from solar_toolkit.radio.config import load_radio_user_config


def test_missing_configuration_does_not_select_an_observation():
    args = build_common_parser("Synthetic command").parse_args([])
    assert args.config is None
    with pytest.raises(ValueError, match="Provide --config-file"):
        resolve_config_source(args)


def test_json_config_is_defensive_and_preserves_algorithm_overrides(tmp_path):
    payload = {
        "user": {"data": {"multi_band_freqs": [100.0]}},
        "newkirk": {"multipliers": [1]},
    }
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(payload))
    args = build_common_parser("Synthetic command").parse_args(
        ["--config-file", str(path)]
    )
    source = resolve_config_source(args)
    user, model = load_radio_user_config(source)
    user["data"]["multi_band_freqs"].append(200.0)
    assert source == payload
    assert model["multipliers"] == [1]


@pytest.mark.parametrize("payload", [[], 1, "synthetic"])
def test_json_config_rejects_non_objects(tmp_path, payload):
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(TypeError, match="object"):
        load_json_config(path)


def test_source_map_json_reaches_runner_without_loading_an_event_module(tmp_path):
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps({"user": {"data": {"synthetic": True}}}))
    calls = []
    assert (
        source_map_cli.main(
            ["--config-file", str(path)],
            runner=lambda config: calls.append(config) or 1,
        )
        == 1
    )
    assert calls[0]["data"]["synthetic"]


def test_pipeline_validates_config_before_starting_runner(tmp_path):
    calls = []
    with pytest.raises(FileNotFoundError):
        pipeline_cli.main(
            ["--config-file", str(tmp_path / "missing.json")],
            runner=lambda argv: calls.append(argv),
        )
    assert calls == []


def test_overlay_accepts_direct_or_section_json(tmp_path):
    for payload in ({"paths": {}}, {"aia_radio_hmi": {"paths": {}}}):
        path = tmp_path / "synthetic.json"
        path.write_text(json.dumps(payload))
        calls = []
        assert (
            overlay_cli.main(
                ["--config-file", str(path)],
                runner=lambda config: calls.append(config) or 1,
            )
            == 1
        )
        assert calls == [{"paths": {}}]


def test_explicit_workspace_mapping_has_no_module_dependency():
    args = build_common_parser("Synthetic command").parse_args(
        ["--workspace-config-json", '{"data":{"synthetic":true}}']
    )
    assert resolve_config_source(args) == {"user": {"data": {"synthetic": True}}}


@pytest.mark.parametrize("module", [quicklook, raw_quality_cli])
def test_workspace_adapters_can_supply_explicit_empty_algorithm_config(module):
    args = module.build_parser().parse_args(["--workspace-config-json", "{}"])
    assert resolve_config_source(args) == {"user": {}}


def test_workspace_mapping_preserves_model_sections():
    args = build_common_parser("Synthetic command").parse_args(
        [
            "--workspace-config-json",
            '{"data":{"synthetic":true},"newkirk":{"multipliers":[8]}}',
        ]
    )
    source = resolve_config_source(args)
    assert source["user"] == {"data": {"synthetic": True}}
    assert load_radio_user_config(source)[1]["multipliers"] == [8]
