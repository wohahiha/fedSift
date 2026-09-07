from __future__ import annotations
import copy
import hashlib
import math
import unittest
from unittest import mock
from fedsift.candidate_space import canonical_sha256
from fedsift.hpo_attempt_receipt import build_hpo_attempt_receipt
from fedsift.hpo_capability import _build_capability_payload, materialize_capability_row_ids
from fedsift.hpo_output import (
    HpoOutputError,
    build_hpo_output_manifest,
    validate_hpo_output_catalog,
    validate_hpo_output_manifest,
)
from fedsift.resource_accounting import build_round_resource_receipt
from tests.test_candidate_hpo_plan import SMALL_CANDIDATE_COUNT, cached_plan, hpo_rows_fixture


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _rehash_manifest(manifest: dict[str, object]) -> None:
    payload = copy.deepcopy(manifest)
    payload.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = canonical_sha256(payload)


def _rehash_receipt(receipt: dict[str, object]) -> None:
    payload = copy.deepcopy(receipt)
    payload.pop("attempt_receipt_sha256", None)
    receipt["attempt_receipt_sha256"] = canonical_sha256(payload)


class HpoOutputTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.space, cls.nested, cls.group_manifest, cls.plan = cached_plan(SMALL_CANDIDATE_COUNT)
        cls.rows = hpo_rows_fixture()
        cls.unit = next(
            (
                unit
                for unit in cls.plan["units"]
                if unit["method"] == "dp_fedavg"
                and unit["outer_repeat"] == 0
                and (unit["outer_fold"] == 0)
                and (unit["inner_fold"] == 0)
                and (unit["candidate_id"] == "candidate_0000")
            )
        )
        cls.capability = _build_capability_payload(
            cls.plan, cls.space, cls.nested, cls.group_manifest, cls.unit["unit_id"]
        )
        cls.row_ids = materialize_capability_row_ids(
            cls.capability,
            role="inner_validation",
            expected_capability_sha256=cls.capability["capability_sha256"],
        )
        cls.probabilities = [0.8 if cls.rows.labels[row_id] else 0.2 for row_id in cls.row_ids]
        cls.hashes = {
            name: _hash(name)
            for name in (
                "preprocessing",
                "model",
                "training",
                "code",
                "environment",
                "dependencies",
                "privacy-report",
                "privacy-schedule",
            )
        }
        cls.round_receipt = build_round_resource_receipt(
            method_id="dp_fedavg",
            server_round=1,
            participating_client_ids=["client_0", "client_1"],
            total_client_count=5,
            model_payload_bytes=128,
            model_manifest_sha256=cls.hashes["model"],
            local_optimizer_steps=7,
            sampled_record_gradient_evaluations=29,
        )
        cls.manifest = cls._build()
        cls.attempt_receipt = build_hpo_attempt_receipt(
            cls.capability,
            expected_capability_sha256=cls.capability["capability_sha256"],
            attempt_index=0,
            outcome="complete",
            code_sha256=cls.hashes["code"],
            environment_sha256=cls.hashes["environment"],
            dependency_lock_sha256=cls.hashes["dependencies"],
            privacy_accounting_report_sha256=cls.hashes["privacy-report"],
            privacy_schedule_sha256=cls.hashes["privacy-schedule"],
            exclusive_output_artifact_manifest_sha256=cls.manifest["manifest_sha256"],
        )

    @classmethod
    def _build(cls, **overrides):
        arguments = {
            "expected_capability_sha256": cls.capability["capability_sha256"],
            "attempt_index": 0,
            "authoritative_rows": cls.rows,
            "probabilities": cls.probabilities,
            "preprocessing_manifest_sha256": cls.hashes["preprocessing"],
            "model_manifest_sha256": cls.hashes["model"],
            "training_trace_sha256": cls.hashes["training"],
            "code_sha256": cls.hashes["code"],
            "environment_sha256": cls.hashes["environment"],
            "dependency_lock_sha256": cls.hashes["dependencies"],
            "privacy_accounting_report_sha256": cls.hashes["privacy-report"],
            "privacy_schedule_sha256": cls.hashes["privacy-schedule"],
            "round_resource_receipts": [cls.round_receipt],
            "expected_rounds": 1,
        }
        arguments.update(overrides)
        return build_hpo_output_manifest(cls.capability, **arguments)

    def _validate(self, manifest, receipt=None, **overrides):
        arguments = {
            "expected_capability_sha256": self.capability["capability_sha256"],
            "authoritative_rows": self.rows,
            "attempt_receipt": receipt,
        }
        arguments.update(overrides)
        return validate_hpo_output_manifest(manifest, self.capability, **arguments)

    def test_hard_coded_golden_and_complete_binding_chain(self) -> None:
        self.assertEqual(
            self.manifest["manifest_sha256"],
            "c30ddc1eef1ea033033ee2b23a487ce6e1bce7110bdd08af2833c53903d95761",
        )
        validation = self._validate(self.manifest, self.attempt_receipt)
        self.assertEqual(validation["row_count"], len(self.row_ids))
        self.assertEqual(validation["communication_bytes"], 512)
        self.assertEqual(
            self.manifest["predictions"][0]["row_token"], f"row_id:{self.row_ids[0]:012d}"
        )
        self.assertEqual(
            self.manifest["predictions"][0]["probability_hex"], self.probabilities[0].hex()
        )
        artifact = self.manifest["artifact_binding"]
        self.assertEqual(
            set(artifact),
            {
                "preprocessing_manifest_sha256",
                "model_manifest_sha256",
                "training_trace_sha256",
                "privacy_accounting_report_sha256",
                "privacy_schedule_sha256",
                "resource_summary_sha256",
            },
        )

    def test_catalog_accepts_one_shot_iterable_without_retaining_manifests(self) -> None:
        plan = {
            "hpo_plan_sha256": "1" * 64,
            "units": [
                {"unit_id": "unit_a", "method": "dp_fedavg", "outer_repeat": 0, "outer_fold": 0},
                {"unit_id": "unit_b", "method": "dp_fedavg", "outer_repeat": 0, "outer_fold": 0},
            ],
        }
        receipts = [
            {
                "outcome": "complete",
                "unit_id": "unit_a",
                "attempt_index": 0,
                "attempt_receipt_sha256": "6" * 64,
                "exclusive_output_artifact_manifest_sha256": "2" * 64,
            },
            {
                "outcome": "complete",
                "unit_id": "unit_b",
                "attempt_index": 0,
                "attempt_receipt_sha256": "7" * 64,
                "exclusive_output_artifact_manifest_sha256": "3" * 64,
            },
        ]
        manifests = [
            {"unit_id": "unit_b", "attempt_index": 0, "manifest_sha256": "3" * 64},
            {"unit_id": "unit_a", "attempt_index": 0, "manifest_sha256": "2" * 64},
        ]
        common = {
            "candidate_space": {},
            "nested_plan": {},
            "group_manifest": {},
            "authoritative_rows": {},
        }
        with (
            mock.patch(
                "fedsift.hpo_output.validate_hpo_attempt_receipt_catalog",
                return_value={"catalog_validation_sha256": "4" * 64},
            ),
            mock.patch("fedsift.hpo_output._validate_shape"),
            mock.patch(
                "fedsift.hpo_output._build_capability_payload",
                return_value={"capability_sha256": "5" * 64},
            ),
            mock.patch("fedsift.hpo_output.validate_hpo_output_manifest"),
        ):
            sequence_result = validate_hpo_output_catalog(plan, [], receipts, manifests, **common)
            iterable_result = validate_hpo_output_catalog(
                plan, [], receipts, (manifest for manifest in manifests), **common
            )
        self.assertEqual(iterable_result, sequence_result)
        self.assertEqual(iterable_result["output_manifest_count"], 2)

    def test_nonfinite_out_of_range_and_missing_authority_fail(self) -> None:
        for bad in (math.nan, math.inf, -0.01, 1.01):
            values = list(self.probabilities)
            values[0] = bad
            with self.subTest(probability=bad), self.assertRaises(HpoOutputError):
                self._build(probabilities=values)
        labels = {row_id: self.rows.labels[row_id] for row_id in self.row_ids[1:]}
        with self.assertRaises(HpoOutputError):
            self._build(
                authoritative_rows=labels, authoritative_dataset_sha256=self.rows.source_sha256
            )
        full_labels = dict(enumerate(self.rows.labels))
        with self.assertRaises(HpoOutputError):
            self._build(authoritative_rows=full_labels)
        with self.assertRaises(HpoOutputError):
            self._build(authoritative_rows=full_labels, authoritative_dataset_sha256="f" * 64)

    def test_missing_duplicate_reordered_label_and_noncanonical_hex_fail(self) -> None:
        cases: list[dict[str, object]] = []
        missing = copy.deepcopy(self.manifest)
        missing["predictions"].pop()
        _rehash_manifest(missing)
        cases.append(missing)
        duplicate = copy.deepcopy(self.manifest)
        duplicate["predictions"][1] = copy.deepcopy(duplicate["predictions"][0])
        _rehash_manifest(duplicate)
        cases.append(duplicate)
        reordered = copy.deepcopy(self.manifest)
        reordered["predictions"][0], reordered["predictions"][1] = (
            reordered["predictions"][1],
            reordered["predictions"][0],
        )
        _rehash_manifest(reordered)
        cases.append(reordered)
        wrong_label = copy.deepcopy(self.manifest)
        wrong_label["predictions"][0]["label"] ^= 1
        _rehash_manifest(wrong_label)
        cases.append(wrong_label)
        alternate_hex = copy.deepcopy(self.manifest)
        alternate_hex["predictions"][0]["probability_hex"] = alternate_hex["predictions"][0][
            "probability_hex"
        ].upper()
        _rehash_manifest(alternate_hex)
        cases.append(alternate_hex)
        for case in cases:
            with self.subTest(case=case["predictions"][:2]):
                with self.assertRaises(HpoOutputError):
                    self._validate(case)

    def test_resource_totals_are_rebuilt_and_caller_totals_cannot_win(self) -> None:
        forged = copy.deepcopy(self.manifest)
        forged["resource_evidence"]["communication_bytes"] = 1
        _rehash_manifest(forged)
        with self.assertRaises(HpoOutputError):
            self._validate(forged)
        forged_summary = copy.deepcopy(self.manifest)
        forged_summary["resource_evidence"]["resource_summary"]["totals"][
            "round_total_federated_bytes"
        ] = 1
        forged_summary["resource_evidence"]["resource_summary"]["report_sha256"] = canonical_sha256(
            {
                key: value
                for (key, value) in forged_summary["resource_evidence"]["resource_summary"].items()
                if key != "report_sha256"
            }
        )
        forged_summary["artifact_binding"]["resource_summary_sha256"] = forged_summary[
            "resource_evidence"
        ]["resource_summary"]["report_sha256"]
        _rehash_manifest(forged_summary)
        with self.assertRaises(HpoOutputError):
            self._validate(forged_summary)

    def test_attempt_output_execution_and_privacy_bindings_fail_closed(self) -> None:
        wrong_output = copy.deepcopy(self.attempt_receipt)
        wrong_output["exclusive_output_artifact_manifest_sha256"] = "f" * 64
        _rehash_receipt(wrong_output)
        with self.assertRaises(HpoOutputError):
            self._validate(self.manifest, wrong_output)
        wrong_execution = copy.deepcopy(self.attempt_receipt)
        wrong_execution["execution_binding"]["code_sha256"] = "e" * 64
        _rehash_receipt(wrong_execution)
        with self.assertRaises(HpoOutputError):
            self._validate(self.manifest, wrong_execution)
        wrong_privacy = copy.deepcopy(self.attempt_receipt)
        wrong_privacy["privacy_binding"]["privacy_accounting_report_sha256"] = "d" * 64
        _rehash_receipt(wrong_privacy)
        with self.assertRaises(HpoOutputError):
            self._validate(self.manifest, wrong_privacy)


if __name__ == "__main__":
    unittest.main()
