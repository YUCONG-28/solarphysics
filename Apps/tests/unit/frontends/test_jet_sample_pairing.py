"""Pilot manifests preserve actual evidence without inventing sparse cadence."""

from types import SimpleNamespace

import pytest

from solar_apps.frontends.jet_lab.native_frames import sample_pairing


def example():
    docs = [
        SimpleNamespace(
            sha256=key, info=dict(midpoint_utc=f"2000-01-01T12:00:0{i}", dsun_m=1.49e11)
        )
        for i, key in enumerate(("a", "b"))
    ]
    evidence = dict(
        status="matched",
        tolerance_s=6.0,
        delta_emission_s=-1.0,
        cadence_source="complete_header_inventory",
        repeated_other=False,
    )
    for name, doc in zip(("AIA", "EUVI"), docs):
        evidence[name] = dict(image_sha256=doc.sha256, **doc.info)
    return docs, evidence


def test_structured_pair_evidence_is_preserved_not_aliased():
    docs, evidence = example()
    restored = sample_pairing({"pairing": evidence}, docs)
    assert restored == evidence
    restored["AIA"]["image_sha256"] = "changed"
    assert evidence["AIA"]["image_sha256"] == "a"


def test_old_sample_method_does_not_become_a_verified_pair():
    docs, _ = example()
    assert (
        sample_pairing(
            {"pairing": "nearest_solar_emission_time_no_interpolation"}, docs
        )
        is None
    )
    assert sample_pairing({}, docs) is None


@pytest.mark.parametrize("change", ["hash", "time", "tolerance", "delta"])
def test_mismatched_evidence_is_rejected(change):
    docs, evidence = example()
    if change == "hash":
        evidence["AIA"]["image_sha256"] = "old_product"
    elif change == "time":
        evidence["EUVI"]["midpoint_utc"] = "2000-01-01T12:02:00"
    elif change == "tolerance":
        evidence["tolerance_s"] = 0.1
    else:
        evidence["delta_emission_s"] = 0
    with pytest.raises(ValueError):
        sample_pairing({"pairing": evidence}, docs)
