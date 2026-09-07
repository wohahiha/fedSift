from __future__ import annotations
import copy
import hashlib
import math
import unittest
from functools import lru_cache
from fedsift.candidate_space import canonical_sha256
from fedsift.evaluation import compute_hpo_probability_metrics
from fedsift.hpo_attempt_receipt import (
    NONPRIVATE_PRIVACY_NOT_APPLICABLE,
    build_hpo_attempt_receipt,
    validate_hpo_attempt_receipt_catalog,
)
from fedsift.hpo_capability import (
    _build_capability_payload,
    build_hpo_unit_index,
    materialize_capability_row_ids,
)
from fedsift.hpo_output import (
    CATALOG_SCHEMA,
    HpoOutputError,
    build_hpo_output_manifest,
    validate_hpo_output_catalog,
)
from fedsift.hpo_plan import close_hpo_ledger
from fedsift.hpo_select import (
    HpoSelectionError,
    _seed_evaluation,
    _sort_key,
    build_hpo_selection_decision,
    build_hpo_selection_decision_from_catalog_commitment,
    build_hpo_selection_decision_from_stage_commitment,
    build_hpo_selection_stage_commitment,
    outer_selection_receipt_arguments,
    validate_hpo_selection_decision,
)
from fedsift.outer_refit import build_outer_selection_receipt
from fedsift.resource_accounting import build_round_resource_receipt
from tests.test_candidate_hpo_plan import SMALL_CANDIDATE_COUNT, cached_plan, hpo_rows_fixture

TARGET_METHOD = "dp_fedavg"
TARGET_REPEAT = 0
TARGET_FOLD = 0


def _hash(label: str, unit_id: str = "") -> str:
    return hashlib.sha256(f"{label}:{unit_id}".encode("ascii")).hexdigest()


def _probabilities_for_loss(labels, loss: float):
    correct = math.exp(-loss)
    return [correct if label == 1 else 1.0 - correct for label in labels]


def _rehash_decision(decision: dict[str, object]) -> None:
    payload = copy.deepcopy(decision)
    payload.pop("selection_decision_manifest_sha256", None)
    decision["selection_decision_manifest_sha256"] = canonical_sha256(payload)


def _rehash_output(manifest: dict[str, object]) -> None:
    payload = copy.deepcopy(manifest)
    payload.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = canonical_sha256(payload)


def _rehash_receipt(receipt: dict[str, object]) -> None:
    payload = copy.deepcopy(receipt)
    payload.pop("attempt_receipt_sha256", None)
    receipt["attempt_receipt_sha256"] = canonical_sha256(payload)


@lru_cache(maxsize=1)
def _complete_evidence():
    space, nested, group_manifest, plan = cached_plan(SMALL_CANDIDATE_COUNT)
    rows = hpo_rows_fixture()
    target_units = [
        unit
        for unit in plan["units"]
        if unit["method"] == TARGET_METHOD
        and unit["outer_repeat"] == TARGET_REPEAT
        and (unit["outer_fold"] == TARGET_FOLD)
    ]
    fold_counts = {}
    for unit in target_units:
        if unit["candidate_id"] != "candidate_0000":
            continue
        capability = _build_capability_payload(plan, space, nested, group_manifest, unit["unit_id"])
        fold_counts[int(unit["inner_fold"])] = int(
            capability["data_slices"]["inner_validation"]["row_count"]
        )
    largest_fold = max(fold_counts, key=lambda value: (fold_counts[value], -value))
    small_loss = 1e-06
    poor_loss = 1.0
    unweighted_candidate0 = (poor_loss + 2.0 * small_loss) / 3.0
    total_rows = sum(fold_counts.values())
    pooled_candidate0 = (
        fold_counts[largest_fold] * poor_loss
        + (total_rows - fold_counts[largest_fold]) * small_loss
    ) / total_rows
    if not pooled_candidate0 > unweighted_candidate0:
        raise AssertionError("test split lacks the required unequal-fold counterexample")
    candidate1_loss = (pooled_candidate0 + unweighted_candidate0) / 2.0
    ledger = []
    attempt_receipts = []
    output_manifests = []
    target_capabilities = {}
    for unit in plan["units"]:
        unit_id = unit["unit_id"]
        method = unit["method"]
        capability = _build_capability_payload(plan, space, nested, group_manifest, unit_id)
        if (
            method == TARGET_METHOD
            and unit["outer_repeat"] == TARGET_REPEAT
            and (unit["outer_fold"] == TARGET_FOLD)
        ):
            target_capabilities[unit_id] = capability
        row_ids = materialize_capability_row_ids(
            capability,
            role="inner_validation",
            expected_capability_sha256=capability["capability_sha256"],
        )
        labels = [rows.labels[row_id] for row_id in row_ids]
        if (
            method == TARGET_METHOD
            and unit["outer_repeat"] == TARGET_REPEAT
            and (unit["outer_fold"] == TARGET_FOLD)
            and (unit["candidate_id"] == "candidate_0000")
        ):
            loss = poor_loss if unit["inner_fold"] == largest_fold else small_loss
            probabilities = _probabilities_for_loss(labels, loss)
        elif (
            method == TARGET_METHOD
            and unit["outer_repeat"] == TARGET_REPEAT
            and (unit["outer_fold"] == TARGET_FOLD)
            and (unit["candidate_id"] == "candidate_0001")
        ):
            probabilities = _probabilities_for_loss(labels, candidate1_loss)
        else:
            probabilities = _probabilities_for_loss(labels, 2.0)
        model_hash = _hash("model", unit_id)
        control_bytes = 64 if method == "dp_scaffold_adapted" else 0
        round_receipt = build_round_resource_receipt(
            method_id=method,
            server_round=1,
            participating_client_ids=["client_0", "client_1", "client_2", "client_3", "client_4"],
            total_client_count=5,
            model_payload_bytes=64,
            model_manifest_sha256=model_hash,
            local_optimizer_steps=5,
            sampled_record_gradient_evaluations=25,
            control_payload_bytes=control_bytes,
        )
        nonprivate = method == "fedavg_nonprivate"
        privacy_report = (
            NONPRIVATE_PRIVACY_NOT_APPLICABLE if nonprivate else _hash("privacy-report", unit_id)
        )
        privacy_schedule = (
            NONPRIVATE_PRIVACY_NOT_APPLICABLE if nonprivate else _hash("privacy-schedule", unit_id)
        )
        execution = {
            "code_sha256": _hash("code", unit_id),
            "environment_sha256": _hash("environment", unit_id),
            "dependency_lock_sha256": _hash("dependencies", unit_id),
        }
        output = build_hpo_output_manifest(
            capability,
            expected_capability_sha256=capability["capability_sha256"],
            attempt_index=0,
            authoritative_rows=rows,
            probabilities=probabilities,
            preprocessing_manifest_sha256=_hash("preprocessing", unit_id),
            model_manifest_sha256=model_hash,
            training_trace_sha256=_hash("training", unit_id),
            privacy_accounting_report_sha256=privacy_report,
            privacy_schedule_sha256=privacy_schedule,
            round_resource_receipts=[round_receipt],
            expected_rounds=1,
            **execution,
        )
        attempt = build_hpo_attempt_receipt(
            capability,
            expected_capability_sha256=capability["capability_sha256"],
            attempt_index=0,
            outcome="complete",
            privacy_accounting_report_sha256=privacy_report,
            privacy_schedule_sha256=privacy_schedule,
            exclusive_output_artifact_manifest_sha256=output["manifest_sha256"],
            **execution,
        )
        ledger.append(
            {
                "unit_id": unit_id,
                "attempts": [
                    {
                        "attempt_index": 0,
                        "outcome": "complete",
                        "attempt_receipt_sha256": attempt["attempt_receipt_sha256"],
                    }
                ],
            }
        )
        attempt_receipts.append(attempt)
        output_manifests.append(output)
    closure = close_hpo_ledger(
        plan,
        ledger,
        attempt_receipts=attempt_receipts,
        candidate_space=space,
        nested_plan=nested,
        group_manifest=group_manifest,
    )
    output_catalog = validate_hpo_output_catalog(
        plan,
        ledger,
        attempt_receipts,
        output_manifests,
        candidate_space=space,
        nested_plan=nested,
        group_manifest=group_manifest,
        authoritative_rows=rows,
    )
    decision = build_hpo_selection_decision(
        plan,
        closure,
        ledger,
        attempt_receipts,
        output_manifests,
        candidate_space=space,
        nested_plan=nested,
        group_manifest=group_manifest,
        authoritative_rows=rows,
        method=TARGET_METHOD,
        outer_repeat=TARGET_REPEAT,
        outer_fold=TARGET_FOLD,
    )
    return {
        "space": space,
        "nested": nested,
        "group_manifest": group_manifest,
        "plan": plan,
        "rows": rows,
        "ledger": ledger,
        "attempt_receipts": attempt_receipts,
        "output_manifests": output_manifests,
        "closure": closure,
        "output_catalog": output_catalog,
        "decision": decision,
        "target_units": target_units,
        "target_capabilities": target_capabilities,
        "largest_fold": largest_fold,
        "candidate1_loss": candidate1_loss,
    }


class HpoSelectionTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.evidence = _complete_evidence()

    def test_hard_coded_golden_and_outer_receipt_integration(self) -> None:
        decision = self.evidence["decision"]
        self.assertEqual(decision["selected_candidate"]["candidate_id"], "candidate_0001")
        self.assertEqual(
            decision["selection_decision_manifest_sha256"],
            "0ba300b94b2f2de5895101d9b52af5ea175b8a62145582705b27175e67ce9c40",
        )
        arguments = outer_selection_receipt_arguments(decision)
        receipt = build_outer_selection_receipt(
            self.evidence["plan"],
            self.evidence["closure"],
            self.evidence["ledger"],
            attempt_receipts=self.evidence["attempt_receipts"],
            candidate_space=self.evidence["space"],
            nested_plan=self.evidence["nested"],
            group_manifest=self.evidence["group_manifest"],
            **arguments,
        )
        self.assertEqual(
            receipt["selection_decision_manifest_sha256"],
            decision["selection_decision_manifest_sha256"],
        )
        self.assertEqual(receipt["candidate_id"], "candidate_0001")

    def test_scope_local_selection_matches_full_catalog_oracle(self) -> None:
        evidence = self.evidence
        catalog = evidence["output_catalog"]
        self.assertEqual(
            catalog["catalog_validation_sha256"],
            evidence["decision"]["output_catalog_validation_sha256"],
        )
        target_ids = {unit["unit_id"] for unit in evidence["target_units"]}
        scope_outputs = [
            manifest
            for manifest in evidence["output_manifests"]
            if manifest["unit_id"] in target_ids
        ]
        stage = build_hpo_selection_stage_commitment(
            evidence["plan"],
            evidence["closure"],
            evidence["ledger"],
            evidence["attempt_receipts"],
            catalog,
            candidate_space=evidence["space"],
            nested_plan=evidence["nested"],
            group_manifest=evidence["group_manifest"],
            expected_catalog_validation_sha256=catalog["catalog_validation_sha256"],
        )
        scope_receipts = [
            receipt for receipt in evidence["attempt_receipts"] if receipt["unit_id"] in target_ids
        ]
        local = build_hpo_selection_decision_from_stage_commitment(
            evidence["plan"],
            evidence["closure"],
            stage,
            scope_receipts,
            scope_outputs,
            candidate_space=evidence["space"],
            nested_plan=evidence["nested"],
            group_manifest=evidence["group_manifest"],
            authoritative_rows=evidence["rows"],
            expected_stage_commitment_sha256=stage["stage_commitment_sha256"],
            method=TARGET_METHOD,
            outer_repeat=TARGET_REPEAT,
            outer_fold=TARGET_FOLD,
            unit_index=build_hpo_unit_index(evidence["plan"]),
        )
        self.assertEqual(local, evidence["decision"])
        forged = copy.deepcopy(stage)
        forged["output_manifest_catalog_sha256"] = "0" * 64
        forged.pop("stage_commitment_sha256")
        forged["stage_commitment_sha256"] = canonical_sha256(forged)
        with self.assertRaises(HpoSelectionError):
            build_hpo_selection_decision_from_stage_commitment(
                evidence["plan"],
                evidence["closure"],
                forged,
                scope_receipts,
                scope_outputs,
                candidate_space=evidence["space"],
                nested_plan=evidence["nested"],
                group_manifest=evidence["group_manifest"],
                authoritative_rows=evidence["rows"],
                expected_stage_commitment_sha256=stage["stage_commitment_sha256"],
                method=TARGET_METHOD,
                outer_repeat=TARGET_REPEAT,
                outer_fold=TARGET_FOLD,
            )
        forged_outputs = copy.deepcopy(scope_outputs)
        forged_receipts = copy.deepcopy(scope_receipts)
        forged_outputs[0]["predictions"][0]["probability_hex"] = (0.51).hex()
        _rehash_output(forged_outputs[0])
        receipt_by_unit = {receipt["unit_id"]: receipt for receipt in forged_receipts}
        forged_receipt = receipt_by_unit[forged_outputs[0]["unit_id"]]
        forged_receipt["exclusive_output_artifact_manifest_sha256"] = forged_outputs[0][
            "manifest_sha256"
        ]
        _rehash_receipt(forged_receipt)
        with self.assertRaises(HpoSelectionError):
            build_hpo_selection_decision_from_stage_commitment(
                evidence["plan"],
                evidence["closure"],
                stage,
                forged_receipts,
                forged_outputs,
                candidate_space=evidence["space"],
                nested_plan=evidence["nested"],
                group_manifest=evidence["group_manifest"],
                authoritative_rows=evidence["rows"],
                expected_stage_commitment_sha256=stage["stage_commitment_sha256"],
                method=TARGET_METHOD,
                outer_repeat=TARGET_REPEAT,
                outer_fold=TARGET_FOLD,
            )
        cross_scope = next(
            (
                manifest
                for manifest in evidence["output_manifests"]
                if manifest["unit_id"] not in target_ids
            )
        )
        crossed = list(scope_outputs)
        crossed[-1] = cross_scope
        with self.assertRaises(HpoSelectionError):
            build_hpo_selection_decision_from_stage_commitment(
                evidence["plan"],
                evidence["closure"],
                stage,
                scope_receipts,
                crossed,
                candidate_space=evidence["space"],
                nested_plan=evidence["nested"],
                group_manifest=evidence["group_manifest"],
                authoritative_rows=evidence["rows"],
                expected_stage_commitment_sha256=stage["stage_commitment_sha256"],
                method=TARGET_METHOD,
                outer_repeat=TARGET_REPEAT,
                outer_fold=TARGET_FOLD,
            )

    def test_concatenate_then_metric_reverses_naive_fold_metric_average(self) -> None:
        outputs = {manifest["unit_id"]: manifest for manifest in self.evidence["output_manifests"]}
        fold_logloss = {"candidate_0000": [], "candidate_0001": []}
        for unit in self.evidence["target_units"]:
            if unit["candidate_id"] not in fold_logloss:
                continue
            manifest = outputs[unit["unit_id"]]
            rows = []
            labels = []
            probabilities = []
            for prediction in manifest["predictions"]:
                rows.append(int(prediction["row_token"][7:]))
                labels.append(prediction["label"])
                probabilities.append(float.fromhex(prediction["probability_hex"]))
            report = compute_hpo_probability_metrics(rows, labels, probabilities)
            fold_logloss[unit["candidate_id"]].append(report["metrics"]["log_loss"])
        naive = {
            candidate: math.fsum(values) / len(values)
            for (candidate, values) in fold_logloss.items()
        }
        self.assertLess(naive["candidate_0000"], naive["candidate_0001"])
        evaluations = {
            row["candidate_id"]: row for row in self.evidence["decision"]["candidate_evaluations"]
        }
        self.assertGreater(
            evaluations["candidate_0000"]["aggregate_metrics"]["log_loss"],
            evaluations["candidate_0001"]["aggregate_metrics"]["log_loss"],
        )
        self.assertEqual(
            self.evidence["decision"]["aggregation_contract"]["fold_metric_averaging"], "forbidden"
        )

    def test_decision_validator_recomputes_catalog_and_rejects_cross_scope(self) -> None:
        validate_hpo_selection_decision(
            self.evidence["decision"],
            self.evidence["plan"],
            self.evidence["closure"],
            self.evidence["ledger"],
            self.evidence["attempt_receipts"],
            self.evidence["output_manifests"],
            candidate_space=self.evidence["space"],
            nested_plan=self.evidence["nested"],
            group_manifest=self.evidence["group_manifest"],
            authoritative_rows=self.evidence["rows"],
        )
        cross_scope = copy.deepcopy(self.evidence["decision"])
        cross_scope["outer_fold"] = 1
        _rehash_decision(cross_scope)
        with self.assertRaises(HpoSelectionError):
            validate_hpo_selection_decision(
                cross_scope,
                self.evidence["plan"],
                self.evidence["closure"],
                self.evidence["ledger"],
                self.evidence["attempt_receipts"],
                self.evidence["output_manifests"],
                candidate_space=self.evidence["space"],
                nested_plan=self.evidence["nested"],
                group_manifest=self.evidence["group_manifest"],
                authoritative_rows=self.evidence["rows"],
            )

    def test_missing_duplicate_and_tampered_output_catalog_fail(self) -> None:
        common = {
            "candidate_space": self.evidence["space"],
            "nested_plan": self.evidence["nested"],
            "group_manifest": self.evidence["group_manifest"],
            "authoritative_rows": self.evidence["rows"],
        }
        with self.assertRaises(HpoOutputError):
            validate_hpo_output_catalog(
                self.evidence["plan"],
                self.evidence["ledger"],
                self.evidence["attempt_receipts"],
                self.evidence["output_manifests"][:-1],
                **common,
            )
        duplicate = list(self.evidence["output_manifests"])
        duplicate[-1] = duplicate[0]
        with self.assertRaises(HpoOutputError):
            validate_hpo_output_catalog(
                self.evidence["plan"],
                self.evidence["ledger"],
                self.evidence["attempt_receipts"],
                duplicate,
                **common,
            )
        tampered = copy.deepcopy(self.evidence["output_manifests"])
        tampered[0]["predictions"][0]["label"] ^= 1
        with self.assertRaises(HpoOutputError):
            validate_hpo_output_catalog(
                self.evidence["plan"],
                self.evidence["ledger"],
                self.evidence["attempt_receipts"],
                tampered,
                **common,
            )

    def test_single_class_overlap_and_cross_outer_seed_units_fail(self) -> None:
        units = [
            unit
            for unit in self.evidence["target_units"]
            if unit["candidate_id"] == "candidate_0000"
        ]
        output_by_unit = {
            manifest["unit_id"]: copy.deepcopy(manifest)
            for manifest in self.evidence["output_manifests"]
            if manifest["unit_id"] in {unit["unit_id"] for unit in units}
        }
        single_class = copy.deepcopy(output_by_unit)
        for manifest in single_class.values():
            for prediction in manifest["predictions"]:
                prediction["label"] = 0
        with self.assertRaises(HpoSelectionError):
            _seed_evaluation(units, single_class, hpo_seed=11, inner_folds=(0, 1, 2))
        overlap = copy.deepcopy(output_by_unit)
        fold0 = next((unit for unit in units if unit["inner_fold"] == 0))
        fold1 = next((unit for unit in units if unit["inner_fold"] == 1))
        overlap[fold1["unit_id"]]["predictions"][0]["row_token"] = overlap[fold0["unit_id"]][
            "predictions"
        ][0]["row_token"]
        with self.assertRaises(HpoSelectionError):
            _seed_evaluation(units, overlap, hpo_seed=11, inner_folds=(0, 1, 2))
        cross_outer = copy.deepcopy(output_by_unit)
        cross_outer[fold1["unit_id"]]["capability_binding"]["outer_fold"] = 1
        with self.assertRaises(HpoSelectionError):
            _seed_evaluation(units, cross_outer, hpo_seed=11, inner_folds=(0, 1, 2))

    def test_exact_no_tolerance_tie_uses_candidate_id_last(self) -> None:
        metrics = {
            "log_loss": 0.5,
            "average_precision": 0.7,
            "auroc": 0.8,
            "brier_score": 0.2,
            "communication_bytes": 100.0,
        }
        first = {"candidate_id": "candidate_0000", "aggregate_metrics": metrics}
        second = {"candidate_id": "candidate_0001", "aggregate_metrics": copy.deepcopy(metrics)}
        self.assertEqual(sorted([second, first], key=_sort_key)[0], first)
        second["aggregate_metrics"]["log_loss"] = math.nextafter(0.5, 0.0)
        self.assertEqual(sorted([first, second], key=_sort_key)[0], second)


if __name__ == "__main__":
    unittest.main()
