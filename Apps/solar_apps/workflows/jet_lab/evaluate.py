"""Read two completed Jet Lab bundles and export repeatability diagnostics."""

import argparse
import csv
import json

from solar_apps.platform.paths.allowed_roots import configured_allowed_roots
from solar_apps.platform.paths.native_dialog import validate_allowed_path
from solar_toolkit.map.jet_annotations import file_sha256, json_safe, load_session
from solar_toolkit.map.jet_evaluation import compare_documents, compare_saved_automatic


def evaluate(first_path, repeat_path, output, *, allowed_roots, same_structure=False):
    def validate(path):
        return validate_allowed_path(path, allowed_roots=allowed_roots, kind="file")

    first_path, repeat_path = validate(first_path), validate(repeat_path)
    if first_path == repeat_path or file_sha256(first_path) == file_sha256(repeat_path):
        raise ValueError("The same annotation cannot serve as its independent repeat")
    first, first_meta = load_session(first_path, validate)
    repeat, repeat_meta = load_session(repeat_path, validate)
    result = {
        "schema": "solarphysics.jet_lab.evaluation",
        "version": 1,
        "first_annotation": str(first_path),
        "repeat_annotation": str(repeat_path),
        "first_annotation_sha256": file_sha256(first_path),
        "repeat_annotation_sha256": file_sha256(repeat_path),
        "first_sample_id": first_meta["sample_id"],
        "repeat_sample_id": repeat_meta["sample_id"],
        "first_role": first_meta["role"],
        "repeat_role": repeat_meta["role"],
        "warning": "descriptive_repeatability_not_accuracy_or_3d_validation",
        "views": [],
    }
    for index, (a, b) in enumerate(zip(first, repeat)):
        result["views"].append(
            {
                "view": index,
                "repeat_comparison": compare_documents(
                    a, b, same_structure=same_structure
                ),
                "first_automatic_vs_saved": compare_saved_automatic(a),
                "repeat_automatic_vs_saved": compare_saved_automatic(b),
            }
        )
    output = validate_allowed_path(
        output, allowed_roots=allowed_roots, kind="output_directory"
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "evaluation.json").write_text(
        json.dumps(json_safe(result), indent=2, ensure_ascii=False, allow_nan=False)
    )
    fields = [
        "view",
        "status",
        "reason",
        "inner_endpoint_difference_arcsec",
        "outer_endpoint_difference_arcsec",
        "length_difference_arcsec",
        "orientation_maybe_reversed",
        "direction30_difference_deg",
        "eligible_30arcsec_pass",
    ]
    with (output / "repeat_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for view in result["views"]:
            comparison = view["repeat_comparison"]
            row = {k: comparison.get(k) for k in fields}
            row["view"] = view["view"]
            direction = next(
                (
                    d
                    for d in comparison.get("directions", [])
                    if d["requested_arcsec"] == 30
                ),
                None,
            )
            if direction:
                row.update(
                    direction30_difference_deg=direction["difference_deg"],
                    eligible_30arcsec_pass=direction["eligible_engineering_pass"],
                )
            writer.writerow(row)
    (output / "COMPLETE.json").write_text(
        json.dumps(
            {
                "sha256": {
                    p.name: file_sha256(p) for p in output.iterdir() if p.is_file()
                }
            },
            indent=2,
        )
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True)
    parser.add_argument("--repeat", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allowed-roots", required=True)
    parser.add_argument(
        "--same-structure",
        action="store_true",
        help="Explicitly assert the same structure was annotated; does not verify 3D",
    )
    args = parser.parse_args(argv)
    roots = configured_allowed_roots(cli_value=args.allowed_roots)
    evaluate(
        args.first,
        args.repeat,
        args.output,
        allowed_roots=roots,
        same_structure=args.same_structure,
    )
    print(f"Saved evaluation: {args.output}")


if __name__ == "__main__":
    main()
