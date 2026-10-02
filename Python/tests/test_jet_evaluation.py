"""Known differences and ineligible cases for annotation comparison."""

import numpy as np
import pytest
from test_jet_extraction import make_fits

from solar_toolkit.map.jet_annotations import JetDocument
from solar_toolkit.map.jet_evaluation import (
    angle_difference_deg,
    compare_axes,
    compare_documents,
    compare_saved_automatic,
    width_summary,
)


def document(path, offset=0):
    doc = JetDocument(path)
    doc.parameters(method="manual", value=25, min_area=1)
    doc.select([20, 48])
    doc.trace([20, 48])
    doc.edit([[20, 48 + offset], [100, 48 + offset]])
    return doc


def test_known_translation_and_angle():
    a = np.array([[0, 0], [100, 0]])
    result = compare_axes(a, a + [2, 3])
    assert result["inner_endpoint_difference_arcsec"] == pytest.approx(np.sqrt(13))
    assert result["length_difference_arcsec"] == 0
    assert all(d["difference_deg"] == 0 for d in result["directions"])
    radians = np.deg2rad(7)
    b = [[0, 0], [100 * np.cos(radians), 100 * np.sin(radians)]]
    result = compare_axes(a, b)
    assert result["directions"][1]["difference_deg"] == pytest.approx(7)
    assert result["directions"][1]["within_5deg_engineering_target"] is False


def test_angle_wrap():
    assert angle_difference_deg(179, -179) == 2
    assert angle_difference_deg(-179, 179) == -2


def test_reversal_is_not_silently_corrected():
    result = compare_axes([[0, 0], [100, 0]], [[100, 0], [0, 0]])
    assert result["orientation_maybe_reversed"]
    assert abs(result["directions"][1]["difference_deg"]) == 180
    assert result["directions"][1]["within_5deg_engineering_target"] is None


def test_short_segments_do_not_pass_requested_length():
    result = compare_axes([[0, 0], [10, 0]], [[0, 0], [10, 0]])
    assert all(
        d["within_5deg_engineering_target"] is None for d in result["directions"]
    )


@pytest.mark.parametrize("axis", [[], [[0, 0], [0, 0]], [[0, 0], [np.nan, 0]]])
def test_invalid_axis(axis):
    with pytest.raises(ValueError):
        compare_axes(axis, [[0, 0], [10, 0]])


def test_identity_required_and_readonly(tmp_path):
    path = make_fits(tmp_path / "image.fits")
    first, second = document(path), document(path, 0.3)
    before = second.state["axis"].copy()
    result = compare_documents(first, second)
    assert result["status"] == "descriptive_only"
    assert not result["eligible_for_repeatability"]
    assert all(d["eligible_engineering_pass"] is None for d in result["directions"])
    result = compare_documents(first, second, same_structure=True)
    assert result["eligible_for_repeatability"]
    assert result["directions"][1]["eligible_engineering_pass"] is True
    assert result["outer_endpoint_difference_arcsec"] == pytest.approx(0.18, abs=1e-4)
    assert second.state["axis"] == before
    assert result["mask_iou"] == 1
    assert (
        compare_saved_automatic(second)["comparison"]
        == "selected_branch_automatic_vs_saved_axis"
    )
    second.state["axis_status"] = "pending"
    assert not compare_documents(first, second, same_structure=True)[
        "eligible_for_repeatability"
    ]


def test_different_image_and_missing_view(tmp_path):
    first = document(make_fits(tmp_path / "one.fits"))
    second = document(make_fits(tmp_path / "two.fits", exposure=4))
    assert (
        compare_documents(first, second)["reason"] == "different_image_or_observation"
    )
    assert compare_documents(None, second)["reason"] == "missing_view"


def test_invalid_width_is_not_zero():
    result = width_summary(
        {
            "widths": [
                {
                    "gaussian": {"status": "multiple_peaks"},
                    "mask_status": "boundary_truncated",
                    "mask_width_arcsec": None,
                }
            ]
        }
    )
    assert result["median_fwhm_arcsec"] is None
    assert result["valid_gaussian_count"] == 0
    assert result["gaussian_status_counts"] == {"multiple_peaks": 1}
