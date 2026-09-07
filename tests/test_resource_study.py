from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import copy
import unittest
from pathlib import Path
from unittest import mock
from fedsift.candidate_space import MAIN_METHODS
from fedsift.resource_protocol import (
    assert_resource_protocol_result_blind,
    build_resource_protocol_order_manifest,
)
from fedsift.preprocessing import PIMA_PRIMARY_PROFILE
from fedsift.resource_study import (
    FORMAL_CANDIDATE_ID,
    MINIMUM_QUERY_COVERAGE_ROUNDS,
    REGISTERED_CLIENTS,
    ResourceStudyBoundaryError,
    ResourceStudyError,
    ResourceStudyScope,
    build_resource_study,
    run_real_data_cost_benchmark,
    validate_resource_study_preparation,
)
from fedsift.study_factory import build_study_construction
from fedsift.training_budget_factory import SharedTrainingBudgetPolicy

ROOT = Path(__file__).resolve().parents[1]
STUDY_ID = _identity("real_data_factory_test")
PROTOCOL_SHA256 = "a" * 64
IMPLEMENTATION_SHA256 = "b" * 64
CLIENT_POLICY_SHA256 = "356fac82397469a91b4da89e27d7557590236d4fdf0df20259881c6bbff423c7"
CANDIDATE_COUNT = 10
HPO_SEEDS = (101,)
MAX_STEPS = 9
PIMA_BUNDLE_SHA256 = "1acbf5cbd7e299b5c358bfef2ce3adf7103254829f4fd3cebda83ea47f7b26fb"


def _construction():
    return build_study_construction(
        ROOT,
        "pima",
        study_id=STUDY_ID,
        protocol_sha256=PROTOCOL_SHA256,
        implementation_sha256=IMPLEMENTATION_SHA256,
        client_policy_sha256=CLIENT_POLICY_SHA256,
        candidate_count=CANDIDATE_COUNT,
        hpo_seeds=HPO_SEEDS,
        max_steps=MAX_STEPS,
        preprocessing_profile_name=PIMA_PRIMARY_PROFILE,
        expected_bundle_sha256=PIMA_BUNDLE_SHA256,
    )


def _policy(
    *,
    server_rounds: int = MAX_STEPS,
    clients: tuple[str, ...] = REGISTERED_CLIENTS,
    condition_sha256: str = CLIENT_POLICY_SHA256,
) -> SharedTrainingBudgetPolicy:
    return SharedTrainingBudgetPolicy(
        server_rounds=server_rounds,
        local_epochs=1,
        poisson_sample_rate=0.5,
        clip_norm=1.0,
        target_epsilon=20.0,
        target_delta=1e-05,
        model_family="logistic_screening",
        participating_client_ids=clients,
        paired_initialization_seed=20260901,
        nonprivate_order_seed=_identity("test_real_data_resource_smoke_nonprivate_order"),
        partition_mode="fixed_label_driven_auxiliary_condition",
        fixed_auxiliary_partition_condition_sha256=condition_sha256,
    )


def _scope() -> ResourceStudyScope:
    return ResourceStudyScope(outer_repeat=0, outer_fold=0, inner_fold=0, hpo_seed=HPO_SEEDS[0])


def _build(construction, policy=None):
    fixed_policy = _policy() if policy is None else policy
    scope = _scope()
    return build_resource_study(
        construction,
        ROOT,
        expected_bundle_sha256=PIMA_BUNDLE_SHA256,
        scope=scope,
        expected_scope_sha256=scope.scope_sha256,
        shared_policy=fixed_policy,
        expected_policy_sha256=fixed_policy.policy_sha256,
        warmup_passes=1,
        latin_square_repetitions=1,
        order_seed=_identity("test_real_data_resource_smoke_order"),
    )


def _executor_request(preparation, method_id: str) -> dict[str, object]:
    capability = preparation.cost_capability
    input_binding = preparation.input_binding
    order = build_resource_protocol_order_manifest(
        capability,
        synthetic_input=input_binding,
        expected_capability_sha256=str(capability["capability_sha256"]),
    )
    sequence_index = 0
    for phase, field in (("warmup", "warmup_orders"), ("measured", "measured_orders")):
        for entry in order[field]:
            for position, observed_method in enumerate(entry["method_ids"]):
                sequence_index += 1
                if observed_method == method_id:
                    return {
                        "method_id": method_id,
                        "candidate_id": 0,
                        "phase": phase,
                        "sequence_index": sequence_index,
                        "block_index": entry["block_index"],
                        "row_index": entry["row_index"],
                        "order_position": position,
                        "synthetic_input": copy.deepcopy(input_binding),
                    }
    raise AssertionError("method was absent from the frozen cost schedule")


class RealRegisteredDataResourceSmokeTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.construction = _construction()
        cls.policy = _policy()
        cls.preparation = _build(cls.construction, cls.policy)

    def test_all_ten_main_methods_are_prepared_with_one_neutral_policy(self) -> None:
        preparation = self.preparation
        validate_resource_study_preparation(
            preparation, expected_preparation_sha256=preparation.preparation_sha256
        )
        bindings = preparation.method_bindings
        self.assertEqual(tuple((row["method_id"] for row in bindings)), MAIN_METHODS)
        self.assertEqual(len(bindings), 10)
        self.assertEqual({row["candidate_roster_id"] for row in bindings}, {FORMAL_CANDIDATE_ID})
        self.assertEqual(
            {row["shared_policy_sha256"] for row in bindings}, {self.policy.policy_sha256}
        )
        self.assertEqual(
            {row["model_manifest_sha256"] for row in bindings},
            {self.preparation.cost_capability["model_manifest_sha256"]},
        )
        self.assertEqual(self.preparation.cost_capability["expected_rounds"], MAX_STEPS)
        self.assertGreaterEqual(MAX_STEPS, MINIMUM_QUERY_COVERAGE_ROUNDS)
        for prepared in self.preparation._prepared_methods:
            with self.subTest(method_id=prepared.method_id):
                budget = prepared.budget_result.budget
                self.assertEqual(budget.server_rounds, MAX_STEPS)
                self.assertEqual(
                    budget.participating_clients_by_round, (REGISTERED_CLIENTS,) * MAX_STEPS
                )
                self.assertEqual(
                    budget.fixed_auxiliary_partition_condition_sha256, CLIENT_POLICY_SHA256
                )
                candidate = prepared.capability["candidate"]
                self.assertEqual(candidate["candidate_id"], FORMAL_CANDIDATE_ID)
                parameters = candidate["parameters"]
                if "control_rule" in parameters:
                    self.assertEqual(parameters["control_rule"]["query_every_rounds"], 5)
                    self.assertGreaterEqual(
                        budget.server_rounds, parameters["control_rule"]["query_every_rounds"]
                    )
        assert_resource_protocol_result_blind(preparation.input_binding)
        assert_resource_protocol_result_blind(preparation.cost_capability)
        assert_resource_protocol_result_blind(preparation.manifest())

    def test_fresh_execution_twice_returns_only_deep_copied_round_receipts(self) -> None:
        request = _executor_request(self.preparation, "dp_fedavg")
        with mock.patch(
            "fedsift.resource_study.execute_training_unit_resource_only",
            wraps=__import__(
                "fedsift.resource_study", fromlist=["execute_training_unit_resource_only"]
            ).execute_training_unit_resource_only,
        ) as fresh_runner:
            first = self.preparation.resource_executor(**request)
            second = self.preparation.resource_executor(**request)
        self.assertEqual(fresh_runner.call_count, 2)
        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        self.assertEqual(len(first), MAX_STEPS)
        for left, right in zip(first, second):
            self.assertIsNot(left, right)
            self.assertEqual(left["schema"], _identity("round_resource_receipt"))
            self.assertEqual(left["method_id"], "dp_fedavg")
            self.assertNotIn("resource_summary", left)
            self.assertNotIn("audit_commitment", left)
            self.assertNotIn("model_state", left)
            self.assertNotIn("predictions", left)
        assert_resource_protocol_result_blind(first)

    def test_candidate_context_input_and_cached_tampering_fail_closed(self) -> None:
        request = _executor_request(self.preparation, "fedavg_nonprivate")
        executor = self.preparation.resource_executor
        nonzero = dict(request)
        nonzero["candidate_id"] = 1
        with self.assertRaises(ResourceStudyBoundaryError):
            executor(**nonzero)
        wrong_context = dict(request)
        wrong_context["order_position"] = int(request["order_position"]) + 1
        with self.assertRaises(ResourceStudyBoundaryError):
            executor(**wrong_context)
        wrong_input = copy.deepcopy(request)
        wrong_input["synthetic_input"]["dataset_id"] = "retinopathy"
        with self.assertRaises(ResourceStudyBoundaryError):
            executor(**wrong_input)
        original = self.preparation._input_binding["dataset_id"]
        try:
            self.preparation._input_binding["dataset_id"] = "tampered"
            with self.assertRaises(ResourceStudyError):
                validate_resource_study_preparation(
                    self.preparation,
                    expected_preparation_sha256=self.preparation.preparation_sha256,
                )
        finally:
            self.preparation._input_binding["dataset_id"] = original

    def test_outer_authority_wrong_rounds_participants_and_condition_fail_early(self) -> None:
        scope = _scope()
        with self.assertRaises(ResourceStudyBoundaryError):
            build_resource_study(
                self.construction,
                ROOT,
                expected_bundle_sha256=PIMA_BUNDLE_SHA256,
                scope=scope,
                expected_scope_sha256=scope.scope_sha256,
                shared_policy=self.policy,
                expected_policy_sha256=self.policy.policy_sha256,
                outer_gate=object(),
            )
        invalid_policies = (
            _policy(server_rounds=MINIMUM_QUERY_COVERAGE_ROUNDS),
            _policy(clients=REGISTERED_CLIENTS[:-1]),
            _policy(condition_sha256="0" * 64),
        )
        for policy in invalid_policies:
            with self.subTest(policy_sha256=policy.policy_sha256):
                with self.assertRaises(ResourceStudyError):
                    build_resource_study(
                        self.construction,
                        ROOT,
                        expected_bundle_sha256=PIMA_BUNDLE_SHA256,
                        scope=scope,
                        expected_scope_sha256=scope.scope_sha256,
                        shared_policy=policy,
                        expected_policy_sha256=policy.policy_sha256,
                    )
        with self.assertRaises(ResourceStudyError):
            build_resource_study(
                self.construction,
                ROOT,
                expected_bundle_sha256="0" * 64,
                scope=scope,
                expected_scope_sha256=scope.scope_sha256,
                shared_policy=self.policy,
                expected_policy_sha256=self.policy.policy_sha256,
            )

    def test_cost_benchmark_wrapper_forwards_only_bound_resource_executor(self) -> None:
        sentinel = {"artifact_sha256": "f" * 64}
        with mock.patch(
            "fedsift.resource_study.run_cost_benchmark", return_value=sentinel
        ) as benchmark:
            result = run_real_data_cost_benchmark(
                self.preparation,
                expected_preparation_sha256=self.preparation.preparation_sha256,
                expected_capability_sha256=str(
                    self.preparation.cost_capability["capability_sha256"]
                ),
            )
        self.assertIs(result, sentinel)
        kwargs = benchmark.call_args.kwargs
        self.assertEqual(kwargs["synthetic_input"], self.preparation.input_binding)
        self.assertEqual(
            kwargs["expected_capability_sha256"],
            self.preparation.cost_capability["capability_sha256"],
        )
        self.assertIs(kwargs["resource_executor"].__self__, self.preparation)


if __name__ == "__main__":
    unittest.main()
