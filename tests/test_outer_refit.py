from __future__ import annotations
import copy
import hashlib
import inspect
import json
import unittest
from fedsift.candidate_space import candidate_by_id, canonical_sha256
from fedsift.group_manifest import group_records
from fedsift.hpo_plan import close_hpo_ledger, inherit_matched_ablation
from fedsift.outer_refit import (
    ForbiddenOuterTestRefitAccess,
    OuterRefitError,
    OuterTestGateError,
    build_outer_refit_capability,
    build_outer_selection_receipt,
    issue_outer_test_gate,
    materialize_outer_refit_row_ids,
    open_outer_test_once,
    outer_refit_candidate_configuration,
    seal_outer_refit_model_manifest,
    seal_outer_refit_prediction_manifest,
    seal_outer_refit_threshold_manifest,
    validate_outer_refit_artifact_chain,
    validate_outer_refit_capability,
    validate_outer_selection_receipt,
    validate_outer_test_access_receipt,
    validate_outer_test_gate,
)
from tests.test_candidate_hpo_plan import (
    SMALL_CANDIDATE_COUNT,
    cached_plan,
    cached_success_evidence,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _rehash(value: dict[str, object], field: str) -> None:
    payload = copy.deepcopy(value)
    payload.pop(field, None)
    value[field] = canonical_sha256(payload)


class OuterRefitGateTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.space, cls.nested, cls.manifest, cls.plan = cached_plan(SMALL_CANDIDATE_COUNT)
        cls.ledger, cls.receipts = cached_success_evidence(SMALL_CANDIDATE_COUNT)
        cls.closure = close_hpo_ledger(
            cls.plan,
            cls.ledger,
            attempt_receipts=cls.receipts,
            candidate_space=cls.space,
            nested_plan=cls.nested,
            group_manifest=cls.manifest,
        )
        cls.main_candidate = candidate_by_id(
            cls.space, "public_argmin_time_dpfedadam", "candidate_0000"
        )
        cls.main_selection = build_outer_selection_receipt(
            cls.plan,
            cls.closure,
            cls.ledger,
            attempt_receipts=cls.receipts,
            candidate_space=cls.space,
            nested_plan=cls.nested,
            group_manifest=cls.manifest,
            method="public_argmin_time_dpfedadam",
            outer_repeat=0,
            outer_fold=0,
            candidate_id=cls.main_candidate["candidate_id"],
            candidate_sha256=cls.main_candidate["candidate_sha256"],
            selection_decision_manifest_sha256=_hash("main-selection-decision"),
        )
        cls.main_capability = build_outer_refit_capability(
            cls.plan,
            cls.closure,
            cls.ledger,
            attempt_receipts=cls.receipts,
            candidate_space=cls.space,
            nested_plan=cls.nested,
            group_manifest=cls.manifest,
            selection_receipt=cls.main_selection,
            eval_seed=701,
        )

    @classmethod
    def _chain(cls, capability, suffix: str):
        cap_hash = capability["capability_sha256"]
        model = seal_outer_refit_model_manifest(
            capability,
            expected_capability_sha256=cap_hash,
            model_artifact_manifest_sha256=_hash(f"model:{suffix}"),
        )
        prediction = seal_outer_refit_prediction_manifest(
            capability,
            model,
            expected_capability_sha256=cap_hash,
            prediction_artifact_manifest_sha256=_hash(f"prediction:{suffix}"),
        )
        threshold = seal_outer_refit_threshold_manifest(
            capability,
            prediction,
            expected_capability_sha256=cap_hash,
            threshold_artifact_manifest_sha256=_hash(f"threshold:{suffix}"),
        )
        return (model, prediction, threshold)

    @classmethod
    def _issue_main(cls, suffix: str):
        model, prediction, threshold = cls._chain(cls.main_capability, suffix)
        gate = issue_outer_test_gate(
            cls.main_capability,
            model,
            prediction,
            threshold,
            cls.plan,
            cls.closure,
            cls.ledger,
            attempt_receipts=cls.receipts,
            candidate_space=cls.space,
            nested_plan=cls.nested,
            group_manifest=cls.manifest,
            selection_receipt=cls.main_selection,
            one_time_access_authorization_sha256=_hash(f"gate:{suffix}"),
        )
        return (gate, model, prediction, threshold)

    def test_selection_and_refit_capability_rebuild_from_exact_upstreams(self) -> None:
        validate_outer_selection_receipt(
            self.main_selection,
            self.plan,
            self.closure,
            self.ledger,
            attempt_receipts=self.receipts,
            candidate_space=self.space,
            nested_plan=self.nested,
            group_manifest=self.manifest,
        )
        validate_outer_refit_capability(
            self.main_capability,
            self.plan,
            self.closure,
            self.ledger,
            attempt_receipts=self.receipts,
            candidate_space=self.space,
            nested_plan=self.nested,
            group_manifest=self.manifest,
            selection_receipt=self.main_selection,
        )
        self.assertEqual(
            self.main_capability["scope"]["method_identity"], "public_argmin_time_dpfedadam"
        )
        self.assertIsNone(self.main_capability["matched_ablation_inheritance"])
        self.assertEqual(
            self.main_capability["bindings"]["preprocessing_profile_sha256"],
            self.plan["bindings"]["preprocessing_profile_sha256"],
        )
        self.assertEqual(
            self.main_capability["bindings"]["client_partition_policy_sha256"],
            self.plan["nested_design_binding"]["client_partition_policy_sha256"],
        )
        rows = materialize_outer_refit_row_ids(
            self.main_capability,
            role="client_0",
            expected_capability_sha256=self.main_capability["capability_sha256"],
        )
        self.assertTrue(rows)

    def test_refit_capability_contains_no_outer_test_and_has_no_manual_loader(self) -> None:
        capability_text = json.dumps(self.main_capability, ensure_ascii=False, sort_keys=True)
        fold = self.nested["repetitions"][0]["outer_folds"][0]
        outer_membership = fold["outer_test"]
        records = {record.group_id: record for record in group_records(self.manifest)}
        outer_rows = sorted(
            (
                row_id
                for group_id in outer_membership["group_ids"]
                for row_id in records[group_id].row_ids
            )
        )
        for group_id in outer_membership["group_ids"]:
            self.assertNotIn(json.dumps(group_id), capability_text)
        for row_id in outer_rows:
            self.assertNotIn(f"row_id:{row_id:012d}", capability_text)
        self.assertNotIn("outer_test", self.main_capability["split_bindings"])
        self.assertNotIn("outer_test", self.main_capability["data_slices"])
        with self.assertRaises(ForbiddenOuterTestRefitAccess):
            materialize_outer_refit_row_ids(
                self.main_capability,
                role="outer_test",
                expected_capability_sha256=self.main_capability["capability_sha256"],
            )
        self.assertNotIn("row_ids", inspect.signature(materialize_outer_refit_row_ids).parameters)
        with self.assertRaises(TypeError):
            materialize_outer_refit_row_ids(
                self.main_capability,
                role="v_ctrl",
                expected_capability_sha256=self.main_capability["capability_sha256"],
                row_ids=(outer_rows[0],),
            )

    def test_tamper_and_cross_scope_selection_fail_closed(self) -> None:
        tampered = copy.deepcopy(self.main_capability)
        tampered["bindings"]["selection_receipt_sha256"] = "e" * 64
        _rehash(tampered, "capability_sha256")
        with self.assertRaises(OuterRefitError):
            validate_outer_refit_capability(
                tampered,
                self.plan,
                self.closure,
                self.ledger,
                attempt_receipts=self.receipts,
                candidate_space=self.space,
                nested_plan=self.nested,
                group_manifest=self.manifest,
                selection_receipt=self.main_selection,
            )
        receipt = copy.deepcopy(self.main_selection)
        receipt["outer_fold"] = 1
        _rehash(receipt, "selection_receipt_sha256")
        with self.assertRaises(OuterRefitError):
            validate_outer_selection_receipt(
                receipt,
                self.plan,
                self.closure,
                self.ledger,
                attempt_receipts=self.receipts,
                candidate_space=self.space,
                nested_plan=self.nested,
                group_manifest=self.manifest,
            )

    def test_closed_chain_gate_cross_scope_open_once_and_replay(self) -> None:
        gate, model, prediction, threshold = self._issue_main("normal")
        validate_outer_refit_artifact_chain(
            self.main_capability,
            model,
            prediction,
            threshold,
            expected_capability_sha256=self.main_capability["capability_sha256"],
        )
        manifest = gate.public_manifest()
        validate_outer_test_gate(
            manifest,
            self.main_capability,
            model,
            prediction,
            threshold,
            self.plan,
            self.closure,
            self.ledger,
            attempt_receipts=self.receipts,
            candidate_space=self.space,
            nested_plan=self.nested,
            group_manifest=self.manifest,
            selection_receipt=self.main_selection,
            one_time_access_authorization_sha256=_hash("gate:normal"),
        )
        with self.assertRaises(OuterTestGateError) as cross_fold:
            open_outer_test_once(
                gate,
                self.plan,
                self.space,
                self.nested,
                self.manifest,
                expected_gate_sha256=manifest["gate_sha256"],
                outer_repeat=0,
                outer_fold=1,
                eval_seed=701,
                access_attempt_id="cross-fold",
            )
        self.assertEqual(cross_fold.exception.attempt_receipt["failure_code"], "cross_outer_scope")
        with self.assertRaises(OuterTestGateError) as cross_seed:
            open_outer_test_once(
                gate,
                self.plan,
                self.space,
                self.nested,
                self.manifest,
                expected_gate_sha256=manifest["gate_sha256"],
                outer_repeat=0,
                outer_fold=0,
                eval_seed=702,
                access_attempt_id="cross-seed",
            )
        self.assertEqual(cross_seed.exception.attempt_receipt["failure_code"], "cross_eval_seed")
        cross_plan = copy.deepcopy(self.plan)
        cross_plan["hpo_plan_sha256"] = "f" * 64
        with self.assertRaises(OuterTestGateError) as wrong_plan:
            open_outer_test_once(
                gate,
                cross_plan,
                self.space,
                self.nested,
                self.manifest,
                expected_gate_sha256=manifest["gate_sha256"],
                outer_repeat=0,
                outer_fold=0,
                eval_seed=701,
                access_attempt_id="cross-plan",
            )
        self.assertEqual(wrong_plan.exception.attempt_receipt["failure_code"], "cross_hpo_plan")
        self.assertFalse(gate.consumed)
        opened = open_outer_test_once(
            gate,
            self.plan,
            self.space,
            self.nested,
            self.manifest,
            expected_gate_sha256=manifest["gate_sha256"],
            outer_repeat=0,
            outer_fold=0,
            eval_seed=701,
            access_attempt_id="normal-open",
        )
        self.assertTrue(opened.row_ids)
        self.assertTrue(gate.consumed)
        self.assertEqual(opened.access_receipt["outcome"], "opened")
        validate_outer_test_access_receipt(opened.access_receipt, manifest)
        with self.assertRaises(OuterTestGateError) as second:
            open_outer_test_once(
                gate,
                self.plan,
                self.space,
                self.nested,
                self.manifest,
                expected_gate_sha256=manifest["gate_sha256"],
                outer_repeat=0,
                outer_fold=0,
                eval_seed=701,
                access_attempt_id="second-open",
            )
        self.assertEqual(second.exception.attempt_receipt["failure_code"], "replay_or_second_open")
        with self.assertRaises(OuterTestGateError) as replay:
            open_outer_test_once(
                gate,
                self.plan,
                self.space,
                self.nested,
                self.manifest,
                expected_gate_sha256=manifest["gate_sha256"],
                outer_repeat=0,
                outer_fold=0,
                eval_seed=701,
                access_attempt_id="normal-open",
            )
        self.assertEqual(
            replay.exception.attempt_receipt["failure_code"], "duplicate_access_attempt_id"
        )
        tampered_gate = gate.public_manifest()
        tampered_gate["outer_test_seal"]["membership_sha256"] = "a" * 64
        _rehash(tampered_gate, "gate_sha256")
        with self.assertRaises(OuterRefitError):
            validate_outer_test_gate(
                tampered_gate,
                self.main_capability,
                model,
                prediction,
                threshold,
                self.plan,
                self.closure,
                self.ledger,
                attempt_receipts=self.receipts,
                candidate_space=self.space,
                nested_plan=self.nested,
                group_manifest=self.manifest,
                selection_receipt=self.main_selection,
                one_time_access_authorization_sha256=_hash("gate:normal"),
            )

    def test_failed_open_is_recorded_and_consumes_gate(self) -> None:
        gate, _, _, _ = self._issue_main("failed")
        manifest = gate.public_manifest()
        failed = open_outer_test_once(
            gate,
            self.plan,
            self.space,
            self.nested,
            self.manifest,
            expected_gate_sha256=manifest["gate_sha256"],
            outer_repeat=0,
            outer_fold=0,
            eval_seed=701,
            access_attempt_id="failed-open",
            loader_failure_incident_sha256=_hash("loader-incident"),
        )
        self.assertEqual(failed.row_ids, ())
        self.assertEqual(failed.access_receipt["outcome"], "failed")
        self.assertFalse(failed.access_receipt["rows_released"])
        self.assertTrue(gate.consumed)
        validate_outer_test_access_receipt(failed.access_receipt, manifest)
        with self.assertRaises(OuterTestGateError):
            open_outer_test_once(
                gate,
                self.plan,
                self.space,
                self.nested,
                self.manifest,
                expected_gate_sha256=manifest["gate_sha256"],
                outer_repeat=0,
                outer_fold=0,
                eval_seed=701,
                access_attempt_id="after-failed-open",
            )

    def test_public_argmin_ablation_inherits_fedsift_parent_without_hpo(self) -> None:
        parent = candidate_by_id(self.space, "fedsift", "candidate_0000")
        selection = build_outer_selection_receipt(
            self.plan,
            self.closure,
            self.ledger,
            attempt_receipts=self.receipts,
            candidate_space=self.space,
            nested_plan=self.nested,
            group_manifest=self.manifest,
            method="fedsift",
            outer_repeat=0,
            outer_fold=0,
            candidate_id=parent["candidate_id"],
            candidate_sha256=parent["candidate_sha256"],
            selection_decision_manifest_sha256=_hash("fedsift-parent-selection"),
        )
        inheritance = inherit_matched_ablation(
            self.plan,
            self.closure,
            self.ledger,
            attempt_receipts=self.receipts,
            candidate_space=self.space,
            nested_plan=self.nested,
            group_manifest=self.manifest,
            ablation_id="fedsift_public_argmin_rule",
            outer_repeat=0,
            outer_fold=0,
            parent_candidate_id=parent["candidate_id"],
            parent_candidate_sha256=parent["candidate_sha256"],
            parent_selection_receipt_sha256=selection["selection_receipt_sha256"],
        )
        capability = build_outer_refit_capability(
            self.plan,
            self.closure,
            self.ledger,
            attempt_receipts=self.receipts,
            candidate_space=self.space,
            nested_plan=self.nested,
            group_manifest=self.manifest,
            selection_receipt=selection,
            eval_seed=701,
            matched_ablation_inheritance=inheritance,
        )
        configuration = outer_refit_candidate_configuration(
            capability, expected_capability_sha256=capability["capability_sha256"]
        )
        self.assertEqual(configuration["method_identity"], "fedsift_public_argmin_rule")
        self.assertEqual(configuration["parent_method"], "fedsift")
        self.assertEqual(configuration["candidate_sha256"], parent["candidate_sha256"])
        for field in ("local_optimizer", "backend", "privacy_schedule"):
            self.assertEqual(configuration["parameters"][field], parent["parameters"][field])
        for field in ("records", "labels_visible", "query_every_rounds", "step_candidates"):
            self.assertEqual(
                configuration["parameters"]["control_rule"][field],
                parent["parameters"]["control_rule"][field],
            )
        self.assertEqual(
            configuration["mechanism_switch"]["replacement_value"], "public_argmin_mean_logloss"
        )
        self.assertEqual(
            configuration["mechanism_switch"]["changed_component"], "control_rule.selection_rule"
        )
        self.assertEqual(inheritance["additional_hpo_units"], 0)
        self.assertFalse(inheritance["independent_candidate_roster"])
        self.assertEqual(
            inheritance["parent_candidate_space_sha256"], self.space["manifest_sha256"]
        )
        self.assertEqual(
            inheritance["parent_roster_sha256"], self.space["methods"]["fedsift"]["roster_sha256"]
        )
        self.assertNotEqual(self.main_candidate["candidate_sha256"], parent["candidate_sha256"])


if __name__ == "__main__":
    unittest.main()
