#!/usr/bin/env python3
"""Cross-platform validation and deterministic rendering for papers.json."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote


FIELDS = (
    "citekey", "previous_citekeys", "title", "authors", "year", "journal",
    "doi", "arxiv_id", "ads_url", "pdf_url", "local_pdf_path",
    "local_pdf_sha256", "summary", "summary_cn", "topic_tags", "method_tags",
    "relevance_level", "recommended_priority", "key_methods",
    "why_relevant_to_current_project", "date_added", "last_checked", "direction",
    "instrument", "radio_frequency_range", "target_event_type", "gaussian_model",
    "centroid_definition", "source_size_definition", "beam_handling",
    "background_handling", "uncertainty_method", "quality_control",
    "applicability_to_DART_DRAT", "notes_for_current_code",
)
ARRAY_FIELDS = {"previous_citekeys", "authors", "topic_tags", "method_tags"}
CITEKEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._:-]*$")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
ADS_URL_RE = re.compile(r"^https://ui\.adsabs\.harvard\.edu/abs/[^/?#]+/?$")


class CatalogError(ValueError):
    pass


def normalize_title(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def normalize_doi(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value)
    value = re.sub(r"^doi:\s*", "", value)
    return value.strip()


def normalize_arxiv(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", value)
    value = re.sub(r"\.pdf$", "", value)
    value = re.sub(r"^arxiv:\s*", "", value)
    value = re.sub(r"v\d+$", "", value)
    return value.strip("/")


def load_catalog(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"cannot read catalog {path}: {exc}") from exc
    if not isinstance(value, list) or not value:
        raise CatalogError("literature catalog must be a non-empty JSON array")
    return value


def validate_catalog(papers: list[dict[str, Any]]) -> None:
    seen: dict[str, set[str]] = {
        "citekey": set(), "title": set(), "doi": set(), "arxiv": set(), "ads": set()
    }
    all_citekeys: set[str] = set()
    for index, paper in enumerate(papers, 1):
        if not isinstance(paper, dict):
            raise CatalogError(f"record {index} must be an object")
        title = str(paper.get("title", ""))
        actual = set(paper)
        missing = [field for field in FIELDS if field not in actual]
        extra = sorted(actual - set(FIELDS))
        if missing or extra:
            raise CatalogError(f"record {title!r} must use exactly 35 fields; missing={missing}; extra={extra}")
        for field in FIELDS:
            value = paper[field]
            if field in ARRAY_FIELDS:
                if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                    raise CatalogError(f"record {title!r} field {field!r} must be an array of strings")
            elif not isinstance(value, str):
                raise CatalogError(f"record {title!r} field {field!r} must be a string")

        citekey = paper["citekey"].strip()
        if not CITEKEY_RE.fullmatch(citekey):
            raise CatalogError(f"record {title!r} has invalid citekey {citekey!r}")
        for candidate in [citekey, *paper["previous_citekeys"]]:
            if not CITEKEY_RE.fullmatch(candidate):
                raise CatalogError(f"record {title!r} has invalid previous citekey {candidate!r}")
            folded = candidate.lower()
            if folded in all_citekeys:
                raise CatalogError(f"duplicate current or previous citekey: {candidate}")
            all_citekeys.add(folded)

        if paper["relevance_level"] not in {"A", "B", "C", "D"}:
            raise CatalogError(f"record {title!r} has invalid relevance_level")
        if paper["recommended_priority"] not in {"high", "medium", "background", "hold"}:
            raise CatalogError(f"record {title!r} has invalid recommended_priority")
        for field in ("date_added", "last_checked"):
            try:
                datetime.strptime(paper[field], "%Y-%m-%d")
            except ValueError as exc:
                raise CatalogError(f"record {title!r} field {field!r} must use yyyy-mm-dd") from exc

        path = paper["local_pdf_path"].strip()
        digest = paper["local_pdf_sha256"].strip()
        if bool(path) != bool(digest):
            raise CatalogError(f"record {title!r} must set local_pdf_path and local_pdf_sha256 together")
        if path and not path.startswith("data://literature-catalog/"):
            raise CatalogError(
                f"record {title!r} local_pdf_path must use data://literature-catalog/"
            )
        if digest and not SHA256_RE.fullmatch(digest):
            raise CatalogError(f"record {title!r} has invalid local_pdf_sha256")
        ads_url = paper["ads_url"].strip()
        if ads_url and not ADS_URL_RE.fullmatch(ads_url):
            raise CatalogError(
                f"record {title!r} ads_url must be an ADS abstract URL; "
                "use doi/arxiv_id for other identities"
            )

        identities = {
            "title": normalize_title(title),
            "doi": normalize_doi(paper["doi"]),
            "arxiv": normalize_arxiv(paper["arxiv_id"]),
            "ads": ads_url.lower(),
        }
        is_explicit_identity_hold = (
            paper["recommended_priority"] == "hold"
            and "UNVERIFIED_IDENTITY" in paper["notes_for_current_code"]
        )
        if is_explicit_identity_hold:
            if paper["authors"] or any(identities[key] for key in ("doi", "arxiv", "ads")):
                raise CatalogError(
                    f"record {title!r} identity hold must not assert authors or identifiers"
                )
            claim_fields = (
                "key_methods", "instrument", "radio_frequency_range", "target_event_type",
                "gaussian_model", "centroid_definition", "source_size_definition",
                "beam_handling", "background_handling", "uncertainty_method",
            )
            if paper["method_tags"] or any(paper[field].strip() for field in claim_fields):
                raise CatalogError(
                    f"record {title!r} identity hold must not assert methods or scientific scope"
                )
        if not any(identities[key] for key in ("doi", "arxiv", "ads")) and not is_explicit_identity_hold:
            raise CatalogError(f"record {title!r} requires a DOI, arXiv ID, or ADS URL")
        for key, identity in identities.items():
            if not identity:
                continue
            if identity in seen[key]:
                raise CatalogError(f"duplicate {key}: {identity}")
            seen[key].add(identity)


def markdown_cell(value: str) -> str:
    return value.replace("\r", " ").replace("\n", " ").replace("|", r"\|")


def render_markdown(papers: list[dict[str, Any]]) -> str:
    rank = {"A": 4, "B": 3, "C": 2, "D": 1}
    ordered = sorted(
        papers,
        key=lambda paper: (
            -rank[paper["relevance_level"]],
            -int(paper["year"] or 0),
            paper["title"].casefold(),
        ),
    )
    lines = [
        "# Paper Master Index", "",
        "| Citekey | Title | Year | Relevance | Priority | Key methods | Why relevant |",
        "|---|---|---|---|---|---|---|",
    ]
    for paper in ordered:
        fields = (
            paper["citekey"], paper["title"], paper["year"], paper["relevance_level"],
            paper["recommended_priority"], paper["key_methods"],
            paper["why_relevant_to_current_project"],
        )
        lines.append("| " + " | ".join(markdown_cell(value) for value in fields) + " |")
    return "\n".join(lines) + "\n"


def write_utf8_bom(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8-sig")


def check_local_hashes(papers: list[dict[str, Any]], data_root: Path | None) -> None:
    if data_root is None:
        return
    prefix = "data://literature-catalog/"
    for paper in papers:
        logical = paper["local_pdf_path"]
        if not logical:
            continue
        if not logical.startswith(prefix):
            raise CatalogError(f"{paper['citekey']}: local_pdf_path must use {prefix}")
        path = data_root / unquote(logical.removeprefix(prefix))
        if not path.is_file():
            raise CatalogError(f"{paper['citekey']}: local PDF not found: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual.lower() != paper["local_pdf_sha256"].lower():
            raise CatalogError(f"{paper['citekey']}: local PDF SHA-256 mismatch")


def default_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[2]
    return root / "Paper/catalog/papers.json", root / "Paper/catalog/papers.md"


def main(argv: list[str] | None = None) -> int:
    default_json, default_markdown = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=default_json)
    parser.add_argument("--markdown", type=Path, default=default_markdown)
    parser.add_argument("--data-root", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check")
    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    try:
        papers = load_catalog(args.catalog)
        validate_catalog(papers)
        check_local_hashes(papers, args.data_root)
        expected = render_markdown(papers)
        actual = args.markdown.read_text(encoding="utf-8-sig") if args.markdown.exists() else ""
        if args.command == "check":
            if actual.replace("\r\n", "\n") != expected:
                raise CatalogError("papers.md does not match deterministic papers.json rendering")
            print(f"catalog check passed: {len(papers)} papers")
            return 0
        if args.apply:
            write_utf8_bom(args.markdown, expected)
            print(f"rendered {len(papers)} papers to {args.markdown}")
        else:
            diff = difflib.unified_diff(
                actual.splitlines(), expected.splitlines(),
                fromfile=str(args.markdown), tofile=f"{args.markdown} (expected)", lineterm="",
            )
            print("\n".join(diff) or "no changes")
        return 0
    except CatalogError as exc:
        print(f"catalog error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
