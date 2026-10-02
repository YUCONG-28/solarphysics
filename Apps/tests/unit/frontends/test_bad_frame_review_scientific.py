"""Scientific queue, schema-v2, and gold-label tests for bad-frame review."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from solar_apps.frontends.radio_bad_frame_review.review import (
    BAD_FRAME_REVIEW_SCHEMA_VERSION,
    BadFrameReviewStore,
    extract_training_examples,
    load_bad_frame_review,
)
from solar_apps.frontends.radio_bad_frame_review.server import create_app


def _queue_candidate(index: int, decision: str) -> dict:
    return {
        "candidate_id": f"candidate-{index:02d}",
        "file_id": f"file-{index:02d}",
        "source_file": f"frame-{index:02d}.fits",
        "relative_path": f"149MHz/RR/frame-{index:02d}.fits",
        "frequency_mhz": 149.0 if index % 2 == 0 else 164.0,
        "polarization": "RR" if index % 3 else "LL",
        "file_index": index,
        "slot_index": index,
        "time": f"2000-05-03T07:20:{index:02d}Z",
        "automatic_decision": decision,
        "features": {
            "dynamic_range_z": 5.0 + index * 7.0,
            "stripe_score": 0.04 * (index % 7),
        },
        "selection_source": None,
    }


def test_labeling_queue_keeps_all_mandatory_and_samples_thirty_percent_good() -> None:
    mandatory = [
        _queue_candidate(index, "bad_candidate" if index < 4 else "uncertain")
        for index in range(7)
    ]
    good = [_queue_candidate(index + 7, "good_candidate") for index in range(12)]

    selected, sampling = BadFrameReviewStore._build_review_queue(
        mandatory + good,
        strategy="labeling",
        sample_count=5,
        seed="sha256:stable",
    )
    repeated, repeated_sampling = BadFrameReviewStore._build_review_queue(
        [
            _queue_candidate(index, item["automatic_decision"])
            for index, item in enumerate(mandatory + good)
        ],
        strategy="labeling",
        sample_count=5,
        seed="sha256:stable",
    )

    selected_ids = {item["candidate_id"] for item in selected}
    assert {item["candidate_id"] for item in mandatory} <= selected_ids
    assert sampling["sampled_good_fraction"] >= 0.30
    assert sampling["mandatory_count"] == 7
    assert sampling["sampled_good_count"] == math.ceil(7 * 3 / 7)
    assert sampling["target_met"] is True
    assert [item["candidate_id"] for item in selected] == [
        item["candidate_id"] for item in repeated
    ]
    assert repeated_sampling == sampling
    assert all(
        item["selection_source"]
        in {"automatic_bad", "automatic_uncertain", "automatic_good_sample"}
        for item in selected
    )


def test_legacy_schemas_are_read_as_v3_and_automatic_skip_is_not_human(
    tmp_path: Path,
) -> None:
    base_candidate = {
        "candidate_id": "candidate-one",
        "file_id": "file-one",
        "source_file": str(tmp_path / "one.fits"),
        "relative_path": "149MHz/RR/one.fits",
        "frequency_mhz": 149.0,
        "polarization": "RR",
        "file_index": 0,
        "slot_index": 0,
        "time": "2000-05-03T07:20:00Z",
        "algorithm_flag": "bad",
        "algorithm_reason": "legacy-rule",
        "metrics": {},
        "human_decision": "good",
        "final_quality": "ok",
    }
    completed_path = tmp_path / "completed.json"
    completed_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "radio-bad-frame-review",
                "review_id": "legacy-completed",
                "status": "completed",
                "updated_at": "2000-05-03T08:00:00Z",
                "candidates": [{**base_candidate, "decision_source": "human"}],
                "files": [],
                "final_bad_files": [],
            }
        ),
        encoding="utf-8",
    )
    completed = load_bad_frame_review(completed_path)
    assert completed["schema_version"] == BAD_FRAME_REVIEW_SCHEMA_VERSION == 3
    assert completed["source_schema_version"] == 1
    assert completed["candidates"][0]["automatic_decision"] == "bad_candidate"
    assert completed["candidates"][0]["human_label"]["quality_label"] == "good"
    assert [item["quality_label"] for item in extract_training_examples(completed)] == [
        "good"
    ]

    skipped_path = tmp_path / "skipped.json"
    skipped_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "radio-bad-frame-review",
                "review_id": "legacy-skipped",
                "status": "skipped",
                "updated_at": "2000-05-03T08:00:00Z",
                "candidates": [
                    {**base_candidate, "decision_source": "automatic_on_skip"}
                ],
                "files": [],
                "final_bad_files": [base_candidate["source_file"]],
            }
        ),
        encoding="utf-8",
    )
    skipped = load_bad_frame_review(skipped_path)
    assert skipped["candidates"][0]["human_label"] is None
    assert extract_training_examples(skipped) == []

    schema_two_path = tmp_path / "schema-two.json"
    schema_two_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "kind": "radio-bad-frame-review",
                "review_id": "schema-two",
                "status": "completed",
                "input": {"candidate_strategy": "rules"},
                "candidates": [],
                "files": [],
                "summary": {},
                "final_bad_files": [],
            }
        ),
        encoding="utf-8",
    )
    schema_two = load_bad_frame_review(schema_two_path)
    assert schema_two["schema_version"] == 3
    assert schema_two["source_schema_version"] == 2
    assert schema_two["input"]["review_scope"] == "candidates"
    assert schema_two["audit"]["mode"] == "candidates"


def _draft_manifest(review_id: str, paths: list[Path]) -> dict:
    files = []
    candidates = []
    for index, path in enumerate(paths):
        stat = path.stat()
        file_id = f"file-{index}"
        files.append(
            {
                "file_id": file_id,
                "source_file": str(path.resolve()),
                "relative_path": path.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "frequency_mhz": 149.0,
                "polarization": "RR",
                "file_index": index,
                "slot_index": index,
                "time": f"2000-05-03T07:20:{index:02d}Z",
            }
        )
        candidates.append(
            {
                "candidate_id": f"candidate-{index}",
                "file_id": file_id,
                "source_file": str(path.resolve()),
                "relative_path": path.name,
                "frequency_mhz": 149.0,
                "polarization": "RR",
                "file_index": index,
                "slot_index": index,
                "time": f"2000-05-03T07:20:{index:02d}Z",
                "algorithm_flag": "bad",
                "algorithm_reason": "scientific-test",
                "metrics": {},
                "features": {"stripe_score": 0.5 + index},
                "automatic_decision": "bad_candidate",
                "automatic_reasons": ["scientific-test"],
                "automatic_rule_version": "radio-science-v1",
                "selection_source": "automatic_bad",
                "sample_fingerprint": f"sha256:{index:064x}",
                "ml_prediction": (
                    {
                        "predicted_label": "bad",
                        "probabilities": {"good": 0.01, "degraded": 0.04, "bad": 0.95},
                        "ood": False,
                        "mode": "shadow",
                    }
                    if index == 0
                    else None
                ),
                "human_label": None,
                "human_decision": None,
                "decision_source": None,
                "final_quality": None,
            }
        )
    manifest = {
        "schema_version": 2,
        "kind": "radio-bad-frame-review",
        "review_id": review_id,
        "status": "draft",
        "created_at": "2000-05-03T08:00:00Z",
        "updated_at": "2000-05-03T08:00:00Z",
        "completed_at": None,
        "input": {"candidate_strategy": "labeling"},
        "input_fingerprint": "sha256:test",
        "files": files,
        "candidates": candidates,
        "final_bad_files": [],
        "summary": {},
    }
    BadFrameReviewStore._refresh_derived(manifest)
    return manifest


def test_human_axes_uncertain_completion_skip_and_training_gate(tmp_path: Path) -> None:
    paths = [tmp_path / "one.fits", tmp_path / "two.fits"]
    for index, path in enumerate(paths):
        path.write_bytes(f"frame-{index}".encode())
    output = tmp_path / "reviews"
    store = BadFrameReviewStore(output, [tmp_path])
    review_id = "scientific-review"
    review_dir = output / review_id
    review_dir.mkdir()
    (review_dir / "review.json").write_text(
        json.dumps(_draft_manifest(review_id, paths)), encoding="utf-8"
    )

    updated = store.update_labels(
        review_id,
        {
            "candidate-0": {
                "quality_label": "degraded",
                "event_tags": ["solar_burst"],
                "artifact_tags": ["sidelobe"],
            },
            "candidate-1": {
                "quality_label": "uncertain",
                "event_tags": [],
                "artifact_tags": ["stripe"],
            },
        },
    )
    assert updated["summary"]["degraded_count"] == 1
    assert updated["summary"]["uncertain_count"] == 1
    assert updated["summary"]["pending_count"] == 1
    with pytest.raises(ValueError, match="Uncertain labels must be resolved"):
        store.finalize(review_id, "completed")

    store.update_decisions(review_id, {"candidate-1": "bad"})
    completed = store.finalize(review_id, "completed")
    assert completed["summary"]["final_bad_count"] == 1
    assert completed["candidates"][0]["ml_prediction"]["predicted_label"] == "bad"
    assert completed["candidates"][0]["final_quality"] == "ok"
    examples = extract_training_examples(completed)
    assert {item["quality_label"] for item in examples} == {"degraded", "bad"}
    degraded = next(item for item in examples if item["quality_label"] == "degraded")
    assert degraded["event_tags"] == ["solar_burst"]
    assert degraded["artifact_tags"] == ["sidelobe"]

    skipped_id = "scientific-skipped"
    skipped_dir = output / skipped_id
    skipped_dir.mkdir()
    (skipped_dir / "review.json").write_text(
        json.dumps(_draft_manifest(skipped_id, paths[:1])), encoding="utf-8"
    )
    store.update_decisions(skipped_id, {"candidate-0": "good"})
    skipped = store.finalize(skipped_id, "skipped")
    assert skipped["candidates"][0]["decision_source"] == "human"
    assert skipped["candidates"][0]["final_quality"] == "ok"
    assert extract_training_examples(skipped) == []


def _write_science_fits(path: Path, second: int) -> None:
    yy, xx = np.indices((32, 32))
    image = 10.0 + 1000.0 * np.exp(-((xx - 16) ** 2 + (yy - 16) ** 2) / 18.0)
    hdu = fits.ImageHDU(data=image.astype(np.float32))
    hdu.header["DATE-OBS"] = f"2000-05-03T07:20:{second:02d}Z"
    hdu.header["CTYPE1"] = "HPLN-TAN"
    hdu.header["CTYPE2"] = "HPLT-TAN"
    hdu.header["CDELT1"] = 1.0
    hdu.header["CDELT2"] = 1.0
    hdu.header["FREQ"] = 149.0e6
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)


def test_labeling_review_writes_scientific_csv_and_sample_fingerprints(
    tmp_path: Path,
) -> None:
    root = tmp_path / "radio"
    for index in range(4):
        _write_science_fits(
            root / "149MHz" / "RR" / f"149MHz_20000503_0720{index:02d}.fits",
            index,
        )
    store = BadFrameReviewStore(tmp_path / "reviews", [tmp_path])
    review = store.create_review(
        {
            "root": str(root),
            "frequencies_mhz": [149],
            "polarizations": ["RR"],
            "candidate_strategy": "labeling",
            "sample_count": 2,
        }
    )
    review_dir = tmp_path / "reviews" / review["review_id"]
    assert review["schema_version"] == 3
    assert review["analysis"]["automatic_quality_csv"] == "automatic_quality.csv"
    assert (review_dir / "automatic_quality.csv").is_file()
    assert review["candidates"]
    assert all(
        str(item["sample_fingerprint"]).startswith("sha256:")
        for item in review["candidates"]
    )


def test_v2_http_labels_are_distinct_from_rules_and_ml(tmp_path: Path) -> None:
    root = tmp_path / "radio"
    for index in range(3):
        _write_science_fits(
            root / "149MHz" / "RR" / f"frame-{index}.fits",
            index,
        )
    app = create_app(
        [tmp_path],
        output_root=tmp_path / "reviews",
        stop_on_client_close=False,
    )
    client = app.test_client()
    created = client.post(
        "/api/reviews",
        json={
            "root": str(root),
            "frequencies_mhz": [149],
            "polarizations": ["RR"],
            "candidate_strategy": "labeling",
            "sample_count": 2,
        },
    )
    assert created.status_code == 201
    review = created.get_json()["review"]
    candidate = review["candidates"][0]
    original_automatic = candidate["automatic_decision"]

    patched = client.patch(
        f"/api/reviews/{review['review_id']}",
        json={
            "labels": {
                candidate["candidate_id"]: {
                    "quality_label": "degraded",
                    "event_tags": ["solar_burst"],
                    "artifact_tags": ["sidelobe"],
                }
            }
        },
    )
    assert patched.status_code == 200
    updated = patched.get_json()["review"]["candidates"][0]
    assert updated["automatic_decision"] == original_automatic
    assert updated["human_label"]["quality_label"] == "degraded"
    assert updated["human_label"]["event_tags"] == ["solar_burst"]
    assert updated["ml_prediction"] is None

    ambiguous = client.patch(
        f"/api/reviews/{review['review_id']}",
        json={
            "decisions": {candidate["candidate_id"]: "good"},
            "labels": {
                candidate["candidate_id"]: {
                    "quality_label": "good",
                    "event_tags": [],
                    "artifact_tags": [],
                }
            },
        },
    )
    assert ambiguous.status_code == 400
    models = client.get("/api/models")
    assert models.status_code == 200
    assert models.get_json()["active_model_id"] is None
