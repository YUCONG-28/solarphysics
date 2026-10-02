"""Read validated annotation bundles without loading Qt."""

import importlib.util
from pathlib import Path

import pytest

from solar_apps.workflows.jet_lab.evaluate import evaluate
from solar_toolkit.map.jet_annotations import JetDocument, save_session


def test_bundle_comparison_export_and_self_repeat_rejection(tmp_path):
    # Reuse the deterministic FITS fixture, not a private observation or snapshot.
    source = Path(__file__).resolve().parents[4] / "Python/tests/test_jet_extraction.py"
    spec = importlib.util.spec_from_file_location("jet_fixture", source)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    path = fixture.make_fits(tmp_path / "image.fits")
    docs = []
    for y in (48, 48.3):
        doc = JetDocument(path)
        doc.parameters(method="manual", value=25, min_area=1)
        doc.select([20, 48])
        doc.trace([20, 48])
        doc.edit([[20, y], [100, y]])
        docs.append(doc)
    first = save_session(
        tmp_path / "first", [docs[0]], sample_id="synthetic", role="test_fixture"
    )
    repeat = save_session(
        tmp_path / "repeat",
        [docs[1]],
        sample_id="synthetic__repeat",
        role="test_fixture",
    )
    result = evaluate(
        first,
        repeat,
        tmp_path / "comparison",
        allowed_roots=[tmp_path],
        same_structure=True,
    )
    assert result["views"][0]["repeat_comparison"]["eligible_for_repeatability"]
    assert result["views"][1]["repeat_comparison"]["reason"] == "missing_view"
    assert (tmp_path / "comparison/COMPLETE.json").is_file()
    with pytest.raises(ValueError, match="independent repeat"):
        evaluate(first, first, tmp_path / "bad", allowed_roots=[tmp_path])
    assert not (tmp_path / "bad").exists()
