from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import copy
import hashlib
import unittest
from fedsift.candidate_space import canonical_sha256
from fedsift.hpo_attempt_receipt import (
    HpoAttemptReceiptError,
    NONPRIVATE_PRIVACY_NOT_APPLICABLE,
    build_hpo_attempt_receipt,
    validate_hpo_attempt_receipt,
    validate_hpo_attempt_receipt_catalog,
)
from fedsift.hpo_capability import _build_capability_payload
from fedsift.hpo_plan import HpoPlanError, close_hpo_ledger, validate_selection_authorization
from tests.test_candidate_hpo_plan import (
    SMALL_CANDIDATE_COUNT,
    cached_plan,
    cached_success_evidence,
)


def _hash(label: str, unit_id: str, attempt_index: int) -> str:
    return hashlib.sha256(f"{label}:{unit_id}:{attempt_index}".encode("ascii")).hexdigest()


def _rehash(receipt: dict[str, object]) -> None:
    payload = copy.deepcopy(receipt)
    payload.pop("attempt_receipt_sha256", None)
    receipt["attempt_receipt_sha256"] = canonical_sha256(payload)


class HpoAttemptReceiptTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.space, cls.nested, cls.manifest, cls.plan = cached_plan(SMALL_CANDIDATE_COUNT)
        cls.nonprivate_unit = next(
            (unit for unit in cls.plan["units"] if unit["method"] == "fedavg_nonprivate")
        )
        cls.private_unit = next(
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
        cls.cross_outer_unit = next(
            (
                unit
                for unit in cls.plan["units"]
                if unit["method"] == cls.private_unit["method"]
                and unit["candidate_id"] == cls.private_unit["candidate_id"]
                and (unit["inner_fold"] == cls.private_unit["inner_fold"])
                and (unit["hpo_seed"] == cls.private_unit["hpo_seed"])
                and (unit["outer_repeat"] == cls.private_unit["outer_repeat"])
                and (unit["outer_fold"] != cls.private_unit["outer_fold"])
            )
        )
        cls.cross_candidate_unit = next(
            (
                unit
                for unit in cls.plan["units"]
                if unit["method"] == cls.private_unit["method"]
                and unit["outer_repeat"] == cls.private_unit["outer_repeat"]
                and (unit["outer_fold"] == cls.private_unit["outer_fold"])
                and (unit["inner_fold"] == cls.private_unit["inner_fold"])
                and (unit["hpo_seed"] == cls.private_unit["hpo_seed"])
                and (unit["candidate_id"] != cls.private_unit["candidate_id"])
            )
        )
        cls.capabilities = {
            unit["unit_id"]: _build_capability_payload(
                cls.plan, cls.space, cls.nested, cls.manifest, unit["unit_id"]
            )
            for unit in (
                cls.nonprivate_unit,
                cls.private_unit,
                cls.cross_outer_unit,
                cls.cross_candidate_unit,
            )
        }

    def _receipt(
        self,
        unit,
        *,
        attempt_index: int = 0,
        outcome: str = "complete",
        failure_code: str | None = None,
    ):
        capability = self.capabilities[unit["unit_id"]]
        nonprivate = unit["method"] == "fedavg_nonprivate"
        privacy_report = (
            NONPRIVATE_PRIVACY_NOT_APPLICABLE
            if nonprivate
            else _hash("privacy-report", unit["unit_id"], attempt_index)
        )
        privacy_schedule = (
            NONPRIVATE_PRIVACY_NOT_APPLICABLE
            if nonprivate
            else _hash("privacy-schedule", unit["unit_id"], attempt_index)
        )
        return build_hpo_attempt_receipt(
            capability,
            expected_capability_sha256=capability["capability_sha256"],
            attempt_index=attempt_index,
            outcome=outcome,
            failure_code=failure_code,
            code_sha256=_hash("code", unit["unit_id"], attempt_index),
            environment_sha256=_hash("environment", unit["unit_id"], attempt_index),
            dependency_lock_sha256=_hash("dependencies", unit["unit_id"], attempt_index),
            privacy_accounting_report_sha256=privacy_report,
            privacy_schedule_sha256=privacy_schedule,
            exclusive_output_artifact_manifest_sha256=(
                _hash("output", unit["unit_id"], attempt_index) if outcome == "complete" else None
            ),
            failure_incident_sha256=(
                _hash("incident", unit["unit_id"], attempt_index) if outcome == "failed" else None
            ),
        )

    @staticmethod
    def _ledger(receipts):
        entries = []
        for receipt in receipts:
            attempt = {
                "attempt_index": receipt["attempt_index"],
                "outcome": receipt["outcome"],
                "attempt_receipt_sha256": receipt["attempt_receipt_sha256"],
            }
            if receipt["outcome"] == "failed":
                attempt["failure_code"] = receipt["failure_code"]
            entries.append({"unit_id": receipt["unit_id"], "attempts": [attempt]})
        return entries

    def _validate_catalog(self, ledger, receipts):
        return validate_hpo_attempt_receipt_catalog(
            self.plan,
            ledger,
            receipts,
            candidate_space=self.space,
            nested_plan=self.nested,
            group_manifest=self.manifest,
        )

    def test_complete_and_failure_receipts_are_exact_and_result_blind(self) -> None:
        complete = self._receipt(self.nonprivate_unit)
        infrastructure = self._receipt(
            self.private_unit, outcome="failed", failure_code="infrastructure_failure"
        )
        for receipt in (complete, infrastructure):
            validate_hpo_attempt_receipt(receipt)
            self.assertNotIn("average_precision", receipt)
            self.assertNotIn("metrics", receipt)
            self.assertNotIn("predictions", receipt)
        self.assertIn("exclusive_output_artifact_manifest_sha256", complete)
        self.assertNotIn("failure_incident_sha256", complete)
        self.assertIn("failure_incident_sha256", infrastructure)
        self.assertNotIn("exclusive_output_artifact_manifest_sha256", infrastructure)
        self.assertEqual(infrastructure["failure_class"], "infrastructure")
        ledger = self._ledger([complete, infrastructure])
        validation = self._validate_catalog(ledger, [infrastructure, complete])
        self.assertEqual(validation["ledger_attempt_count"], 2)
        self.assertEqual(validation["receipt_count"], 2)

    def test_fake_missing_orphan_and_duplicate_receipts_fail(self) -> None:
        first = self._receipt(self.nonprivate_unit)
        second = self._receipt(self.private_unit)
        ledger = self._ledger([first, second])
        fake_hash = copy.deepcopy(ledger)
        fake_hash[0]["attempts"][0]["attempt_receipt_sha256"] = "f" * 64
        with self.assertRaises(HpoAttemptReceiptError):
            self._validate_catalog(fake_hash, [first, second])
        with self.assertRaises(HpoAttemptReceiptError):
            self._validate_catalog(ledger, [first])
        orphan = self._receipt(self.cross_outer_unit)
        with self.assertRaises(HpoAttemptReceiptError):
            self._validate_catalog(ledger, [first, second, orphan])
        with self.assertRaises(HpoAttemptReceiptError):
            self._validate_catalog(ledger, [first, second, second])
        duplicate_identity = copy.deepcopy(second)
        duplicate_identity["execution_binding"]["code_sha256"] = "e" * 64
        _rehash(duplicate_identity)
        with self.assertRaises(HpoAttemptReceiptError):
            self._validate_catalog(ledger, [first, second, duplicate_identity])

    def test_swapped_cross_unit_candidate_and_outer_receipts_fail(self) -> None:
        first = self._receipt(self.private_unit)
        second = self._receipt(self.cross_outer_unit)
        swapped = self._ledger([first, second])
        (
            swapped[0]["attempts"][0]["attempt_receipt_sha256"],
            swapped[1]["attempts"][0]["attempt_receipt_sha256"],
        ) = (
            swapped[1]["attempts"][0]["attempt_receipt_sha256"],
            swapped[0]["attempts"][0]["attempt_receipt_sha256"],
        )
        with self.assertRaises(HpoAttemptReceiptError):
            self._validate_catalog(swapped, [first, second])
        cases = []
        cross_unit = copy.deepcopy(first)
        cross_unit["unit_id"] = self.cross_outer_unit["unit_id"]
        cases.append((cross_unit, self.cross_outer_unit))
        cross_candidate = copy.deepcopy(first)
        cross_candidate["unit_binding"]["candidate_id"] = self.cross_candidate_unit["candidate_id"]
        cross_candidate["unit_binding"]["candidate_sha256"] = self.cross_candidate_unit[
            "candidate_sha256"
        ]
        cases.append((cross_candidate, self.private_unit))
        cross_outer = copy.deepcopy(first)
        cross_outer["unit_binding"]["outer_fold"] = self.cross_outer_unit["outer_fold"]
        cases.append((cross_outer, self.private_unit))
        for forged, ledger_unit in cases:
            with self.subTest(binding=forged["unit_binding"]):
                _rehash(forged)
                ledger = self._ledger([forged])
                ledger[0]["unit_id"] = ledger_unit["unit_id"]
                with self.assertRaises(HpoAttemptReceiptError):
                    self._validate_catalog(ledger, [forged])

    def test_unknown_fields_performance_fields_and_outcome_drift_fail(self) -> None:
        receipt = self._receipt(self.private_unit)
        unknown = copy.deepcopy(receipt)
        unknown["extra"] = "forbidden"
        _rehash(unknown)
        with self.assertRaises(HpoAttemptReceiptError):
            validate_hpo_attempt_receipt(unknown)
        polluted = copy.deepcopy(receipt)
        polluted["average_precision"] = 0.9
        _rehash(polluted)
        with self.assertRaises(HpoAttemptReceiptError):
            validate_hpo_attempt_receipt(polluted)
        ledger = self._ledger([receipt])
        ledger[0]["attempts"][0]["outcome"] = "failed"
        ledger[0]["attempts"][0]["failure_code"] = "nonfinite_update"
        with self.assertRaises(HpoAttemptReceiptError):
            self._validate_catalog(ledger, [receipt])

    def test_private_requires_accounting_and_nonprivate_requires_explicit_na(self) -> None:
        private = self._receipt(self.private_unit)
        private_na = copy.deepcopy(private)
        private_na["privacy_binding"][
            "privacy_accounting_report_sha256"
        ] = NONPRIVATE_PRIVACY_NOT_APPLICABLE
        private_na["privacy_binding"]["privacy_schedule_sha256"] = NONPRIVATE_PRIVACY_NOT_APPLICABLE
        _rehash(private_na)
        with self.assertRaises(HpoAttemptReceiptError):
            validate_hpo_attempt_receipt(private_na)
        nonprivate = self._receipt(self.nonprivate_unit)
        fake_accounting = copy.deepcopy(nonprivate)
        fake_accounting["privacy_binding"]["privacy_accounting_report_sha256"] = "a" * 64
        fake_accounting["privacy_binding"]["privacy_schedule_sha256"] = "b" * 64
        _rehash(fake_accounting)
        with self.assertRaises(HpoAttemptReceiptError):
            validate_hpo_attempt_receipt(fake_accounting)
        capability = self.capabilities[self.private_unit["unit_id"]]
        with self.assertRaises(HpoAttemptReceiptError):
            build_hpo_attempt_receipt(
                capability,
                expected_capability_sha256=capability["capability_sha256"],
                attempt_index=0,
                outcome="complete",
                code_sha256="a" * 64,
                environment_sha256="b" * 64,
                dependency_lock_sha256="c" * 64,
                privacy_accounting_report_sha256=NONPRIVATE_PRIVACY_NOT_APPLICABLE,
                privacy_schedule_sha256=NONPRIVATE_PRIVACY_NOT_APPLICABLE,
                exclusive_output_artifact_manifest_sha256="d" * 64,
            )

    def test_complete_and_failure_branch_whitelists_fail_closed(self) -> None:
        capability = self.capabilities[self.private_unit["unit_id"]]
        common = {
            "expected_capability_sha256": capability["capability_sha256"],
            "attempt_index": 0,
            "code_sha256": "a" * 64,
            "environment_sha256": "b" * 64,
            "dependency_lock_sha256": "c" * 64,
            "privacy_accounting_report_sha256": "d" * 64,
            "privacy_schedule_sha256": "e" * 64,
        }
        with self.assertRaises(HpoAttemptReceiptError):
            build_hpo_attempt_receipt(
                capability,
                outcome="complete",
                failure_incident_sha256="f" * 64,
                exclusive_output_artifact_manifest_sha256="1" * 64,
                **common,
            )
        with self.assertRaises(HpoAttemptReceiptError):
            build_hpo_attempt_receipt(
                capability,
                outcome="failed",
                failure_code="infrastructure_failure",
                failure_incident_sha256="f" * 64,
                exclusive_output_artifact_manifest_sha256="1" * 64,
                **common,
            )
        with self.assertRaises(HpoAttemptReceiptError):
            build_hpo_attempt_receipt(
                capability,
                outcome="failed",
                failure_code="made_up_failure",
                failure_incident_sha256="f" * 64,
                **common,
            )

    def test_closure_and_selection_revalidate_the_original_receipt_catalog(self) -> None:
        ledger, receipts = cached_success_evidence(SMALL_CANDIDATE_COUNT)
        ledger = copy.deepcopy(ledger)
        receipts = copy.deepcopy(receipts)
        closure = close_hpo_ledger(
            self.plan,
            ledger,
            attempt_receipts=receipts,
            candidate_space=self.space,
            nested_plan=self.nested,
            group_manifest=self.manifest,
        )
        self.assertEqual(closure["schema"], _identity("hpo_ledger_closure"))
        self.assertEqual(closure["verified_attempt_receipt_count"], len(receipts))
        validate_selection_authorization(
            self.plan,
            closure,
            ledger,
            attempt_receipts=receipts,
            candidate_space=self.space,
            nested_plan=self.nested,
            group_manifest=self.manifest,
        )
        tampered = copy.deepcopy(receipts)
        tampered[0]["execution_binding"]["environment_sha256"] = "0" * 64
        _rehash(tampered[0])
        with self.assertRaises(HpoPlanError):
            validate_selection_authorization(
                self.plan,
                closure,
                ledger,
                attempt_receipts=tampered,
                candidate_space=self.space,
                nested_plan=self.nested,
                group_manifest=self.manifest,
            )


if __name__ == "__main__":
    unittest.main()
