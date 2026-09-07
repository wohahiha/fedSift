from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import copy
import hashlib
import math
import unittest
from dataclasses import replace
from unittest.mock import patch
from fedsift.candidate_space import MAIN_METHODS, _base_parameters, canonical_sha256
from fedsift.hpo_capability import _build_capability_payload
from fedsift.hpo_plan import MATCHED_ABLATIONS
from fedsift.method_dispatch import build_method_execution_spec
from fedsift.outer_refit import REFIT_CAPABILITY_SCHEMA, _WORKER_POLICY as OUTER_REFIT_WORKER_POLICY
from fedsift.preprocessing import (
    PreprocessedRoles,
    _matrix_sha256,
    preprocessing_artifact_fingerprint,
)
from fedsift.train_unit import (
    ExplicitClientBatchPlan,
    ForbiddenTrainingRoleAccess,
    ResourceOnlyTrainingResult,
    SealedUnitData,
    TrainingBudget,
    TrainingUnitError,
    TrainingUnitFailure,
    _cached_minimal_noise_calibration,
    _full_private_schedule,
    execute_training_unit,
    execute_training_unit_resource_only,
    materialize_training_role,
    seal_preprocessed_unit_data,
    validate_resource_only_training_result,
    validate_training_unit_result,
)
from tests.test_candidate_hpo_plan import SMALL_CANDIDATE_COUNT, cached_plan

ALL_METHODS = (*MAIN_METHODS, *tuple(MATCHED_ABLATIONS))
TWO_ROUND_FINAL_STATE_GOLDEN = {
    "fedavg_nonprivate": "54fa773904d6694a3eb980277ef48fdbba28b0f2c61651b774b8ea0477a94ba2",
    "dp_fedavg": "8518bf9d15b259b0d76b40789048b2aaf3d5bd98b247100f30e849d1b8685357",
    "dp_fedprox_adapted": "28238845b3506ece57ba592a5619b86099c77261a2f7d72fd44705375d635c0a",
    "dp_scaffold_adapted": "9f0bb3ea50623e995cc501c1b887db393fb850e5d645499124470a9d051af2c9",
    "dp_fedadam": "c0e8228fa0a3916c2763d87a14884c7a6a97eff1a757a1edd9567f65edd7d87f",
    "dp_fedyogi": "f3512e998aee1a58eafccac7966b713d747ef9e15435fd922c3456b33009f868",
    "dp_fedsofim_delta_proxy_adapted": "767607dc4237fe9aebb52480fa2a511c5d6e5b83fe0a841891c013fc4a642be6",
    "time_dpfedadam": "b4182c42370f1013a9bd4cadbc85c5e6a30187f76b8b38eb7f69b90ec55f7b19",
    "public_argmin_time_dpfedadam": "35cb952caeac972ae84eba045a894b7cefb0e013fa2692e0f4b1c5f7ffa252cb",
    "fedsift": "1261f47c65d08c9e7f71781f3a5f4c93d041b22e5ef361f5ad3037071dff0dce",
    "fedsift_uniform_schedule": "05a3aadecfbfb24c572dd3b4ff4e59e6083931a0c32c3e8bdf6752c97df92dd8",
    "fedsift_without_sift": "04966e4193f7e0f786bbb0900be9fb0159e739acb697282dea0e08e6ec4f8ba5",
    "fedsift_public_argmin_rule": "5ec8c48b60fc1bc10f436f23af263ecca2b8500e72a6b141e903a416f8498129",
}


def _rehash_capability(value: dict[str, object]) -> None:
    payload = copy.deepcopy(value)
    payload.pop("capability_sha256", None)
    value["capability_sha256"] = canonical_sha256(payload)


def _method_capability(base: dict[str, object], method: str) -> dict[str, object]:
    capability = copy.deepcopy(base)
    parent = (
        str(MATCHED_ABLATIONS[method]["parent_method"]) if method in MATCHED_ABLATIONS else method
    )
    parameters = _base_parameters(parent)
    if parent in {"public_argmin_time_dpfedadam", "fedsift"}:
        parameters["control_rule"]["query_every_rounds"] = 1
    identity = capability["unit_identity"]
    candidate = capability["candidate"]
    identity["method"] = method
    identity["max_steps"] = 2
    candidate["candidate_id"] = f"toy_{method}"
    candidate["parameters"] = parameters
    candidate["parameters_sha256"] = canonical_sha256(parameters)
    candidate["candidate_sha256"] = hashlib.sha256(
        f"toy-candidate:{method}".encode("ascii")
    ).hexdigest()
    _rehash_capability(capability)
    return capability


def _outer_refit_capability(base: dict[str, object]) -> dict[str, object]:
    parameters = _base_parameters("fedavg_nonprivate")
    hpo_bindings = base["bindings"]
    hpo_splits = base["split_bindings"]
    hpo_slices = base["data_slices"]
    capability = {
        "schema": REFIT_CAPABILITY_SCHEMA,
        "status": "SEALED_OUTER_REFIT_OUTER_TEST_EXCLUDED",
        "study_id": base["study_id"],
        "dataset_id": base["dataset_id"],
        "bindings": {
            "hpo_plan_sha256": hpo_bindings["hpo_plan_sha256"],
            "hpo_closure_sha256": "1" * 64,
            "terminal_ledger_sha256": "2" * 64,
            "attempt_receipt_catalog_sha256": "3" * 64,
            "outer_authorization_sha256": "4" * 64,
            "selection_receipt_sha256": "5" * 64,
            "selection_decision_manifest_sha256": "6" * 64,
            "candidate_space_sha256": hpo_bindings["candidate_space_sha256"],
            "nested_plan_sha256": hpo_bindings["nested_plan_sha256"],
            "nested_freeze_binding_sha256": hpo_bindings["nested_freeze_binding_sha256"],
            "source_sha256": hpo_bindings["dataset_sha256"],
            "group_manifest_sha256": hpo_bindings["group_manifest_sha256"],
            "row_to_group_sha256": hpo_bindings["row_to_group_sha256"],
            "protocol_sha256": hpo_bindings["protocol_sha256"],
            "implementation_contract_sha256": hpo_bindings["implementation_contract_sha256"],
            "client_partition_policy": "toy_fixed_five_client_partition",
            "client_partition_policy_sha256": "7" * 64,
            "preprocessing_profile_name": hpo_bindings["preprocessing_profile_name"],
            "preprocessing_profile_sha256": hpo_bindings["preprocessing_profile_sha256"],
            "preprocessing_profile": copy.deepcopy(hpo_bindings["preprocessing_profile"]),
        },
        "scope": {
            "method_identity": "fedavg_nonprivate",
            "parent_method": "fedavg_nonprivate",
            "outer_repeat": 0,
            "outer_fold": 0,
            "eval_seed": 17,
        },
        "candidate": {
            "candidate_id": "toy_outer_fedavg",
            "candidate_sha256": hashlib.sha256(b"toy-outer-fedavg").hexdigest(),
            "parameters_sha256": canonical_sha256(parameters),
            "parameters": parameters,
            "mechanism_switch": None,
        },
        "matched_ablation_inheritance": None,
        "split_bindings": {
            "outer_partition_sha256": "8" * 64,
            "outer_train_membership_sha256": "9" * 64,
            "role_partition_sha256": hpo_splits["role_partition_sha256"],
            "private_membership_sha256": "a" * 64,
            "v_ctrl_membership_sha256": hpo_splits["v_ctrl_membership_sha256"],
            "v_sel_membership_sha256": hpo_splits["v_sel_membership_sha256"],
            "client_partition_sha256": hpo_splits["client_partition_sha256"],
            "client_membership_sha256": copy.deepcopy(hpo_splits["client_membership_sha256"]),
        },
        "data_slices": {
            "private_clients": copy.deepcopy(hpo_slices["private_clients"]),
            "v_ctrl": copy.deepcopy(hpo_slices["v_ctrl"]),
            "v_sel": copy.deepcopy(hpo_slices["v_sel"]),
        },
        "worker_policy": copy.deepcopy(OUTER_REFIT_WORKER_POLICY),
    }
    _rehash_capability(capability)
    return capability


def _toy_preprocessed(capability: dict[str, object]) -> PreprocessedRoles:
    cap_hash = capability["capability_sha256"]
    roles = {}
    matrices = {}
    role_names = [*tuple((f"client_{index}" for index in range(5))), "v_ctrl", "v_sel"]
    if capability["schema"] != REFIT_CAPABILITY_SCHEMA:
        role_names.append("inner_validation")
    for role in role_names:
        row_ids = materialize_training_role(
            capability, role=role, expected_capability_sha256=cap_hash
        )
        matrix = tuple(
            (
                tuple((((row_id + 3 * column) % 19 - 9) / 10.0 for column in range(8)))
                for row_id in row_ids
            )
        )
        matrices[role] = matrix
        roles[role] = {
            "row_count": len(row_ids),
            "row_ids": list(row_ids),
            "matrix_sha256": _matrix_sha256(role, row_ids, matrix),
        }
    artifact = {
        "schema": _identity("toy_preprocessed_capability"),
        "status": "SEALED_TEST_COMPONENT_INTEGRATION",
        "input_binding": {"capability_sha256": cap_hash},
        "roles": roles,
    }
    artifact["artifact_sha256"] = preprocessing_artifact_fingerprint(artifact)
    return PreprocessedRoles(artifact=artifact, matrices=matrices)


def _sealed_data(capability: dict[str, object]):
    preprocessed = _toy_preprocessed(capability)
    labels = {
        row_id: row_id % 2
        for role in preprocessed.artifact["roles"].values()
        for row_id in role["row_ids"]
    }
    return seal_preprocessed_unit_data(
        capability, preprocessed, labels, expected_capability_sha256=capability["capability_sha256"]
    )


def _budget(data: SealedUnitData, capability: dict[str, object]) -> TrainingBudget:
    client_rows = data.role("client_0").row_ids[:4]
    plan = ExplicitClientBatchPlan(
        "client_0", epochs=((tuple(client_rows[:2]), tuple(client_rows[2:])),)
    )
    if capability["schema"] == REFIT_CAPABILITY_SCHEMA:
        method = capability["scope"]["method_identity"]
    else:
        method = capability["unit_identity"]["method"]
    if method == "fedavg_nonprivate":
        calibrated_base = 1.0
    else:
        parameters = capability["candidate"]["parameters"]
        if (
            method == "fedsift_uniform_schedule"
            or parameters["privacy_schedule"]["kind"] == "uniform"
        ):
            first_factor = second_factor = 1.0
        else:
            first_factor = float(parameters["privacy_schedule"]["saving_sigma_factor"])
            second_factor = float(parameters["privacy_schedule"]["spending_sigma_factor"])
        calibrated_base = _cached_minimal_noise_calibration(
            0.03, 2, 1, first_factor, second_factor, 100.0, 1e-05
        ).base_noise_multiplier
    return TrainingBudget(
        server_rounds=2,
        dp_optimizer_steps_by_round=(1, 1),
        nonprivate_batch_plans_by_round=((plan,), (plan,)),
        participating_clients_by_round=(("client_0",), ("client_0",)),
        poisson_sample_rate=0.03,
        base_noise_multiplier=calibrated_base,
        clip_norm=1.0,
        target_epsilon=100.0,
        target_delta=1e-05,
        model_family="logistic_screening",
        initialization_seed=193,
        partition_mode="fixed_label_driven_auxiliary_condition",
        fixed_auxiliary_partition_condition_sha256="d" * 64,
    )


class TrainingUnitIntegrationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        space, nested, manifest, plan = cached_plan(SMALL_CANDIDATE_COUNT)
        del space, nested, manifest
        base_unit = plan["units"][0]
        space, nested, manifest, plan = cached_plan(SMALL_CANDIDATE_COUNT)
        cls.base = _build_capability_payload(plan, space, nested, manifest, base_unit["unit_id"])

    def test_all_ten_main_methods_and_three_ablations_execute_two_real_rounds(self) -> None:
        observed = {}
        for method in ALL_METHODS:
            with self.subTest(method=method):
                capability = _method_capability(self.base, method)
                data = _sealed_data(capability)
                budget = _budget(data, capability)
                result = execute_training_unit(
                    capability,
                    data,
                    budget,
                    expected_capability_sha256=capability["capability_sha256"],
                    expected_budget_sha256=budget.budget_sha256,
                )
                validate_training_unit_result(
                    result,
                    capability,
                    data,
                    budget,
                    expected_capability_sha256=capability["capability_sha256"],
                    expected_budget_sha256=budget.budget_sha256,
                )
                artifact = result.artifact
                self.assertEqual(artifact["method_id"], method)
                self.assertEqual(len(artifact["round_evidence"]), 2)
                self.assertEqual(
                    [value["role"] for value in artifact["raw_native_predictions"]],
                    ["v_sel", "inner_validation"],
                )
                self.assertFalse(artifact["evaluation_metrics_computed"])
                self.assertFalse(artifact["candidate_selection_performed"])
                self.assertFalse(artifact["outer_test_accessed"])
                self.assertNotIn("fallback", result.dispatch_spec.server_backend_handler)
                self.assertTrue(
                    all(
                        (
                            row["server_kernel_handler"]
                            == result.dispatch_spec.server_backend_handler
                            for row in artifact["round_evidence"]
                        )
                    )
                )
                if method == "fedavg_nonprivate":
                    self.assertIsNone(result.parallel_privacy_report)
                    self.assertFalse(result.local_step_receipts)
                else:
                    self.assertIsNotNone(result.parallel_privacy_report)
                    self.assertEqual(len(result.sequential_client_reports), 1)
                    self.assertEqual(
                        result.sequential_client_reports[0].composed_report.accounted_steps, 2
                    )
                if method == "dp_fedprox_adapted":
                    self.assertEqual(
                        {value.correction_kind for value in result.local_step_receipts}, {"fedprox"}
                    )
                if method == "dp_scaffold_adapted":
                    self.assertEqual(
                        {value.correction_kind for value in result.local_step_receipts},
                        {"scaffold"},
                    )
                    self.assertEqual(
                        result.dispatch_spec.scaffold_variant,
                        "option_ii_batched_weighted_registered_client_weights",
                    )
                    self.assertEqual(
                        {row["resolved_kernel_name"] for row in artifact["round_evidence"]},
                        {"scaffold_server_batched_weighted"},
                    )
                if method == "dp_fedsofim_delta_proxy_adapted":
                    self.assertEqual(len(result.sofim_receipt_pairs), 2)
                    for proxy, _ in result.sofim_receipt_pairs:
                        self.assertEqual(proxy["clients"][0]["actual_local_steps"], 1)
                        self.assertEqual(
                            proxy["clients"][0]["local_learning_rate_hex"],
                            float(
                                _base_parameters(method)["local_optimizer"]["learning_rate"]
                            ).hex(),
                        )
                query_methods = {
                    "public_argmin_time_dpfedadam",
                    "fedsift",
                    "fedsift_uniform_schedule",
                    "fedsift_public_argmin_rule",
                }
                if method in query_methods:
                    self.assertEqual(len(result.control_evidence), 2)
                    self.assertTrue(
                        all((len(value.candidate_table) == 5 for value in result.control_evidence))
                    )
                else:
                    self.assertFalse(result.control_evidence)
                if method == "fedsift_without_sift":
                    self.assertFalse(result.dispatch_spec.public_control_queries_enabled)
                    self.assertTrue(
                        all(
                            (
                                row["selected_alpha_hex"] == (1.0).hex()
                                for row in artifact["round_evidence"]
                            )
                        )
                    )
                observed[method] = artifact["final_model_state_sha256"]
                self.assertEqual(
                    artifact["final_model_state_sha256"], TWO_ROUND_FINAL_STATE_GOLDEN[method]
                )
        self.assertEqual(observed, TWO_ROUND_FINAL_STATE_GOLDEN)
        self.assertEqual(
            len(
                {
                    _method_capability(self.base, method)["candidate"]["candidate_sha256"]
                    for method in ALL_METHODS
                }
            ),
            13,
        )

    def test_fedavg_nonprivate_resource_unit_executes_shuffled_explicit_batches(self) -> None:
        capability = _method_capability(self.base, "fedavg_nonprivate")
        data = _sealed_data(capability)
        ordered_rows = tuple(sorted(data.role("client_0").row_ids))
        self.assertGreaterEqual(len(ordered_rows), 4)
        plan = ExplicitClientBatchPlan(
            "client_0",
            epochs=(((ordered_rows[2], ordered_rows[0]), (ordered_rows[3], ordered_rows[1])),),
        )
        self.assertNotEqual(plan.epochs[0][0], tuple(sorted(plan.epochs[0][0])))
        budget = replace(
            _budget(data, capability), nonprivate_batch_plans_by_round=((plan,), (plan,))
        )
        result = execute_training_unit_resource_only(
            capability,
            data,
            budget,
            expected_capability_sha256=capability["capability_sha256"],
            expected_budget_sha256=budget.budget_sha256,
        )
        validate_resource_only_training_result(
            result,
            capability,
            data,
            budget,
            expected_capability_sha256=capability["capability_sha256"],
            expected_budget_sha256=budget.budget_sha256,
        )
        self.assertEqual(result.resource_summary["totals"]["local_optimizer_steps"], 4)
        self.assertEqual(
            result.resource_summary["totals"]["sampled_record_gradient_evaluations"], 8
        )

    def test_outer_access_budget_data_and_dispatch_tampering_fail_atomically(self) -> None:
        capability = _method_capability(self.base, "dp_fedavg")
        data = _sealed_data(capability)
        budget = _budget(data, capability)
        with self.assertRaises(ForbiddenTrainingRoleAccess):
            materialize_training_role(
                capability,
                role="outer_test",
                expected_capability_sha256=capability["capability_sha256"],
            )
        drifted_budget = replace(budget, clip_norm=2.0)
        with self.assertRaises(TrainingUnitFailure) as budget_failure:
            execute_training_unit(
                capability,
                data,
                drifted_budget,
                expected_capability_sha256=capability["capability_sha256"],
                expected_budget_sha256=budget.budget_sha256,
            )
        self.assertFalse(budget_failure.exception.failure_receipt["partial_model_released"])
        self.assertFalse(budget_failure.exception.failure_receipt["partial_prediction_released"])
        oversized_noise_budget = replace(
            budget, base_noise_multiplier=budget.base_noise_multiplier * 2.0
        )
        with self.assertRaises(TrainingUnitFailure) as fairness_failure:
            execute_training_unit(
                capability,
                data,
                oversized_noise_budget,
                expected_capability_sha256=capability["capability_sha256"],
                expected_budget_sha256=oversized_noise_budget.budget_sha256,
            )
        self.assertEqual(fairness_failure.exception.failure_receipt["stage"], "preflight")
        self.assertFalse(fairness_failure.exception.failure_receipt["performance_metric_consumed"])
        changed_role = replace(
            data.roles[0],
            features=(
                (data.roles[0].features[0][0] + 0.25, *data.roles[0].features[0][1:]),
                *data.roles[0].features[1:],
            ),
        )
        changed_data = replace(data, roles=(changed_role, *data.roles[1:]))
        with self.assertRaises(TrainingUnitFailure):
            execute_training_unit(
                capability,
                changed_data,
                budget,
                expected_capability_sha256=capability["capability_sha256"],
                expected_budget_sha256=budget.budget_sha256,
            )
        broken = copy.deepcopy(capability)
        broken["candidate"]["parameters"]["backend"]["name"] = "hidden_fedavg"
        broken["candidate"]["parameters_sha256"] = canonical_sha256(
            broken["candidate"]["parameters"]
        )
        _rehash_capability(broken)
        broken_data = _sealed_data(broken)
        broken_budget = _budget(broken_data, broken)
        with self.assertRaises(TrainingUnitFailure) as dispatch_failure:
            execute_training_unit(
                broken,
                broken_data,
                broken_budget,
                expected_capability_sha256=broken["capability_sha256"],
                expected_budget_sha256=broken_budget.budget_sha256,
            )
        self.assertEqual(dispatch_failure.exception.failure_receipt["stage"], "preflight")
        self.assertFalse(dispatch_failure.exception.failure_receipt["performance_metric_consumed"])

    def test_outer_refit_capability_executes_without_inner_or_outer_predictions(self) -> None:
        capability = _outer_refit_capability(self.base)
        data = _sealed_data(capability)
        budget = _budget(data, capability)
        result = execute_training_unit(
            capability,
            data,
            budget,
            expected_capability_sha256=capability["capability_sha256"],
            expected_budget_sha256=budget.budget_sha256,
        )
        self.assertEqual(result.artifact["capability_kind"], "outer_refit")
        self.assertEqual(
            [value["role"] for value in result.artifact["raw_native_predictions"]], ["v_sel"]
        )
        serialized = str(result.artifact["raw_native_predictions"])
        self.assertNotIn("inner_validation", serialized)
        self.assertNotIn("outer_test", serialized)
        with self.assertRaises(ForbiddenTrainingRoleAccess):
            materialize_training_role(
                capability,
                role="outer_test",
                expected_capability_sha256=capability["capability_sha256"],
            )

    def test_resource_only_path_reuses_core_and_stops_before_prediction_payload(self) -> None:
        capability = _method_capability(self.base, "fedsift")
        data = _sealed_data(capability)
        budget = _budget(data, capability)
        with patch(
            "fedsift.train_unit._prediction_payload",
            side_effect=AssertionError("resource-only path attempted prediction inference"),
        ):
            result = execute_training_unit_resource_only(
                capability,
                data,
                budget,
                expected_capability_sha256=capability["capability_sha256"],
                expected_budget_sha256=budget.budget_sha256,
            )
            validate_resource_only_training_result(
                result,
                capability,
                data,
                budget,
                expected_capability_sha256=capability["capability_sha256"],
                expected_budget_sha256=budget.budget_sha256,
            )
        self.assertIs(type(result), ResourceOnlyTrainingResult)
        self.assertFalse(hasattr(result, "model"))
        self.assertFalse(hasattr(result, "model_state"))
        self.assertFalse(hasattr(result, "control_evidence"))
        self.assertFalse(hasattr(result, "local_step_receipts"))
        self.assertEqual(len(result.round_resource_receipts), 2)
        self.assertEqual(result.resource_summary["totals"]["local_optimizer_steps"], 2)
        self.assertGreater(
            result.resource_summary["totals"]["public_candidate_forward_record_evaluations"], 0
        )
        serialized_audit = str(result.audit_commitment).lower()
        for forbidden in (
            "alpha",
            "candidate_table",
            "control",
            "model",
            "prediction",
            "loss",
            "metric",
            "outer",
        ):
            self.assertNotIn(forbidden, serialized_audit)
        tampered = copy.deepcopy(result.audit_commitment)
        tampered["resource_report_sha256"] = "0" * 64
        with self.assertRaises(TrainingUnitError):
            validate_resource_only_training_result(
                replace(result, audit_commitment=tampered),
                capability,
                data,
                budget,
                expected_capability_sha256=capability["capability_sha256"],
                expected_budget_sha256=budget.budget_sha256,
            )

    def test_time_schedule_fraction_is_applied_to_rounds_before_step_expansion(self) -> None:
        capability = _method_capability(self.base, "time_dpfedadam")
        data = _sealed_data(capability)
        budget = replace(_budget(data, capability), dp_optimizer_steps_by_round=(3, 7))
        parameters = capability["candidate"]["parameters"]
        spec = build_method_execution_spec(
            method_id="time_dpfedadam", candidate_parameters=parameters, mechanism_switch=None
        )
        schedule = _full_private_schedule(spec, parameters, budget)
        saving_rounds = math.ceil(
            budget.server_rounds * float(parameters["privacy_schedule"]["saving_round_fraction"])
        )
        self.assertEqual(schedule[0].steps, sum((3, 7)[:saving_rounds]))
        self.assertEqual(schedule[1].steps, sum((3, 7)[saving_rounds:]))
        self.assertNotEqual(schedule[0].steps, math.ceil(sum((3, 7)) * 0.4))


if __name__ == "__main__":
    unittest.main()
