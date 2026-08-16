from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).parents[1] / "catalog_cli.py"
SPEC = importlib.util.spec_from_file_location("catalog_cli", MODULE_PATH)
assert SPEC and SPEC.loader
catalog_cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(catalog_cli)


class CatalogCliTests(unittest.TestCase):
    def test_repository_catalog_is_valid_and_deterministic(self) -> None:
        catalog, markdown = catalog_cli.default_paths()
        papers = catalog_cli.load_catalog(catalog)
        catalog_cli.validate_catalog(papers)
        self.assertEqual(
            catalog_cli.render_markdown(papers),
            markdown.read_text(encoding="utf-8-sig"),
        )

    def test_duplicate_previous_citekey_is_rejected(self) -> None:
        catalog, _ = catalog_cli.default_paths()
        papers = catalog_cli.load_catalog(catalog)
        duplicate = dict(papers[0])
        duplicate["citekey"] = "UniqueReplacement"
        duplicate["previous_citekeys"] = [papers[0]["citekey"]]
        duplicate["title"] = "Unique replacement title"
        duplicate["doi"] = "10.1000/unique-replacement"
        with self.assertRaisesRegex(catalog_cli.CatalogError, "citekey"):
            catalog_cli.validate_catalog([papers[0], duplicate])

    def test_local_pdf_path_requires_sha256(self) -> None:
        catalog, _ = catalog_cli.default_paths()
        paper = dict(catalog_cli.load_catalog(catalog)[0])
        paper["local_pdf_path"] = "data://literature-catalog/example.pdf"
        paper["local_pdf_sha256"] = ""
        with self.assertRaisesRegex(catalog_cli.CatalogError, "together"):
            catalog_cli.validate_catalog([paper])

    def test_local_pdf_path_requires_logical_catalog_uri(self) -> None:
        catalog, _ = catalog_cli.default_paths()
        paper = dict(catalog_cli.load_catalog(catalog)[0])
        paper["local_pdf_path"] = "/Users/example/paper.pdf"
        paper["local_pdf_sha256"] = "a" * 64
        with self.assertRaisesRegex(catalog_cli.CatalogError, "data://literature-catalog/"):
            catalog_cli.validate_catalog([paper])

    def test_identity_hold_cannot_assert_methods(self) -> None:
        catalog, _ = catalog_cli.default_paths()
        paper = next(
            item for item in catalog_cli.load_catalog(catalog)
            if item["recommended_priority"] == "hold"
            and "UNVERIFIED_IDENTITY" in item["notes_for_current_code"]
        )
        asserted = dict(paper)
        asserted["key_methods"] = "elliptical Gaussian fitting"
        with self.assertRaisesRegex(catalog_cli.CatalogError, "must not assert methods"):
            catalog_cli.validate_catalog([asserted])

    def test_ads_url_cannot_contain_a_publisher_or_doi_url(self) -> None:
        catalog, _ = catalog_cli.default_paths()
        paper = dict(catalog_cli.load_catalog(catalog)[0])
        paper["ads_url"] = "https://doi.org/10.1000/not-ads"
        with self.assertRaisesRegex(catalog_cli.CatalogError, "ADS abstract URL"):
            catalog_cli.validate_catalog([paper])


if __name__ == "__main__":
    unittest.main()
