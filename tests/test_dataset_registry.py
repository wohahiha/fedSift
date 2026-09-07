from __future__ import annotations
import unittest
from pathlib import Path
from fedsift.dataset_registry import (
    DEBRECEN,
    PIMA,
    DatasetRegistryError,
    load_registered_dataset,
    registered_dataset,
)

ROOT = Path(__file__).resolve().parents[1]


class DatasetRegistryTests(unittest.TestCase):

    def test_exact_real_sources_load_with_frozen_content(self) -> None:
        pima = load_registered_dataset(ROOT, "pima")
        retinopathy = load_registered_dataset(ROOT, "retinopathy")
        self.assertEqual((len(pima.row_ids), sum(pima.labels)), (768, 268))
        self.assertEqual((len(retinopathy.row_ids), sum(retinopathy.labels)), (1151, 611))
        self.assertEqual(pima.source_sha256, PIMA.file_sha256)
        self.assertEqual(retinopathy.source_sha256, DEBRECEN.file_sha256)
        self.assertEqual(pima.source_path, f"registered://pima/{PIMA.file_sha256}")
        self.assertEqual(
            retinopathy.source_path, f"registered://retinopathy/{DEBRECEN.file_sha256}"
        )
        self.assertTrue(pima.predictor_allowlist_explicit)
        self.assertTrue(retinopathy.predictor_allowlist_explicit)

    def test_registry_records_authoritative_provenance_and_license(self) -> None:
        for dataset_id in ("pima", "retinopathy"):
            spec = registered_dataset(dataset_id)
            self.assertTrue(spec.authoritative_metadata_url.startswith("https://"))
            self.assertTrue(spec.authoritative_download_url.startswith("https://"))
            self.assertEqual(len(spec.source_artifact_sha256), 64)
            self.assertIn("verified", spec.provenance_status)
            self.assertTrue(spec.license_statement)

    def test_unknown_or_wrong_workspace_fails_closed(self) -> None:
        with self.assertRaises(DatasetRegistryError):
            registered_dataset("synthetic")
        with self.assertRaises(DatasetRegistryError):
            load_registered_dataset(ROOT / "code", "pima")


if __name__ == "__main__":
    unittest.main()
