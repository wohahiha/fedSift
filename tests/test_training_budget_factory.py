from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import copy
import math
import unittest
from dataclasses import replace
from fedsift.candidate_space import (
    MAIN_METHODS,
    _base_parameters,
    assert_no_performance_fields,
    canonical_sha256,
)
from fedsift.hpo_plan import MATCHED_ABLATIONS
from fedsift.method_dispatch import build_method_execution_spec
from fedsift.train_unit import _require_minimal_noise_calibration
from fedsift.training_budget_factory import (
    SharedTrainingBudgetPolicy,
    TrainingBudgetFactoryError,
    TrainingBudgetFactoryResult,
    build_training_budget,
    validate_training_budget_factory_result,
)

ALL_METHODS = (*MAIN_METHODS, *tuple(MATCHED_ABLATIONS))


def _policy() -> SharedTrainingBudgetPolicy:
    return SharedTrainingBudgetPolicy(
        server_rounds=4,
        local_epochs=2,
        poisson_sample_rate=0.3,
        clip_norm=1.25,
        target_epsilon=20.0,
        target_delta=1e-05,
        model_family="logistic_screening",
        participating_client_ids=tuple((f"client_{index}" for index in range(5))),
        paired_initialization_seed=20260831,
        nonprivate_order_seed=_identity("test_fixed_result_blind_order"),
        partition_mode="fixed_nonprivate",
    )


def _rows() -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {}
    offset = 0
    for client_index in range(5):
        count = 10 + client_index
        result[f"client_{client_index}"] = tuple(range(offset, offset + count))
        offset += count
    return result


def _method_inputs(method_id: str):
    parent = (
        str(MATCHED_ABLATIONS[method_id]["parent_method"])
        if method_id in MATCHED_ABLATIONS
        else method_id
    )
    parameters = _base_parameters(parent)
    switch = (
        copy.deepcopy(MATCHED_ABLATIONS[method_id]["mechanism_switch"])
        if method_id in MATCHED_ABLATIONS
        else None
    )
    spec = build_method_execution_spec(
        method_id=method_id, candidate_parameters=parameters, mechanism_switch=switch
    )
    return (parameters, spec)


def _build(method_id: str, policy: SharedTrainingBudgetPolicy | None = None):
    fixed_policy = _policy() if policy is None else policy
    parameters, spec = _method_inputs(method_id)
    result = build_training_budget(
        spec,
        parameters,
        _rows(),
        fixed_policy,
        expected_dispatch_sha256=spec.dispatch_sha256,
        expected_candidate_parameters_sha256=canonical_sha256(parameters),
        expected_policy_sha256=fixed_policy.policy_sha256,
    )
    return (parameters, spec, fixed_policy, result)


class TrainingBudgetFactoryTests(unittest.TestCase):

    def test_candidate_zero_for_all_ten_methods_and_three_ablations_is_fair_and_executable(
        self,
    ) -> None:
        policy = _policy()
        rows = _rows()
        results = {}
        for method_id in ALL_METHODS:
            with self.subTest(method_id=method_id):
                parameters, spec = _method_inputs(method_id)
                result = build_training_budget(
                    spec,
                    parameters,
                    rows,
                    policy,
                    expected_dispatch_sha256=spec.dispatch_sha256,
                    expected_candidate_parameters_sha256=canonical_sha256(parameters),
                    expected_policy_sha256=policy.policy_sha256,
                )
                validate_training_budget_factory_result(
                    result,
                    spec,
                    parameters,
                    rows,
                    policy,
                    expected_dispatch_sha256=spec.dispatch_sha256,
                    expected_candidate_parameters_sha256=canonical_sha256(parameters),
                    expected_policy_sha256=policy.policy_sha256,
                )
                assert_no_performance_fields(result.artifact)
                budget = result.budget
                expected_steps = math.ceil(policy.local_epochs / policy.poisson_sample_rate)
                self.assertEqual(
                    budget.dp_optimizer_steps_by_round, (expected_steps,) * policy.server_rounds
                )
                self.assertEqual(
                    budget.participating_clients_by_round,
                    (policy.participating_client_ids,) * policy.server_rounds,
                )
                self.assertEqual(budget.clip_norm, policy.clip_norm)
                self.assertEqual(budget.target_epsilon, policy.target_epsilon)
                self.assertEqual(budget.target_delta, policy.target_delta)
                self.assertEqual(budget.model_family, policy.model_family)
                self.assertEqual(budget.initialization_seed, policy.paired_initialization_seed)
                self.assertEqual(result.artifact["shared_policy_sha256"], policy.policy_sha256)
                self.assertFalse(result.artifact["observed_outputs_consumed"])
                self.assertEqual(
                    result.artifact["test_partition_authority"], "forbidden_and_not_accepted"
                )
                if method_id in MATCHED_ABLATIONS:
                    self.assertTrue(result.artifact["matched_parent_candidate_rules"])
                    self.assertEqual(spec.parent_method, "fedsift")
                else:
                    self.assertFalse(result.artifact["matched_parent_candidate_rules"])
                if spec.privacy_mode == "nonprivate_explicit_minibatch":
                    self.assertEqual(budget.base_noise_multiplier, 1.0)
                else:
                    calibration, _ = _require_minimal_noise_calibration(spec, parameters, budget)
                    self.assertEqual(
                        budget.base_noise_multiplier.hex(),
                        float(calibration.base_noise_multiplier).hex(),
                    )
                results[method_id] = result
        shared_budget_fields = {
            (
                value.budget.server_rounds,
                value.budget.dp_optimizer_steps_by_round,
                value.budget.participating_clients_by_round,
                value.budget.poisson_sample_rate,
                value.budget.clip_norm,
                value.budget.target_epsilon,
                value.budget.target_delta,
                value.budget.model_family,
                value.budget.initialization_seed,
                value.artifact["nonprivate_explicit_plan_sha256"],
            )
            for value in results.values()
        }
        self.assertEqual(len(shared_budget_fields), 1)
        uniform_base = results["dp_fedavg"].budget.base_noise_multiplier.hex()
        for method_id in (
            "dp_fedprox_adapted",
            "dp_scaffold_adapted",
            "dp_fedadam",
            "dp_fedyogi",
            "dp_fedsofim_delta_proxy_adapted",
            "fedsift_uniform_schedule",
        ):
            self.assertEqual(results[method_id].budget.base_noise_multiplier.hex(), uniform_base)
        time_base = results["time_dpfedadam"].budget.base_noise_multiplier.hex()
        for method_id in (
            "public_argmin_time_dpfedadam",
            "fedsift",
            "fedsift_without_sift",
            "fedsift_public_argmin_rule",
        ):
            self.assertEqual(results[method_id].budget.base_noise_multiplier.hex(), time_base)

    def test_nonprivate_explicit_epochs_are_complete_batched_and_round_domain_separated(
        self,
    ) -> None:
        _, _, policy, result = _build("fedavg_nonprivate")
        rows = _rows()
        budget = result.budget
        for client_position, client_id in enumerate(policy.participating_client_ids):
            expected_rows = rows[client_id]
            previous_by_epoch = [None] * policy.local_epochs
            n = len(expected_rows)
            nearest_batch_size = max(1, min(n, math.floor(policy.poisson_sample_rate * n + 0.5)))
            for server_round in range(policy.server_rounds):
                plan = budget.nonprivate_batch_plans_by_round[server_round][client_position]
                self.assertEqual(plan.client_id, client_id)
                self.assertEqual(len(plan.epochs), policy.local_epochs)
                for epoch_index, batches in enumerate(plan.epochs):
                    flattened = tuple((row_id for batch in batches for row_id in batch))
                    self.assertEqual(set(flattened), set(expected_rows))
                    self.assertEqual(len(flattened), len(expected_rows))
                    self.assertTrue(
                        all((len(batch) == nearest_batch_size for batch in batches[:-1]))
                    )
                    self.assertGreaterEqual(len(batches[-1]), 1)
                    self.assertLessEqual(len(batches[-1]), nearest_batch_size)
                    if previous_by_epoch[epoch_index] is not None and n > 1:
                        self.assertNotEqual(flattened, previous_by_epoch[epoch_index])
                    previous_by_epoch[epoch_index] = flattened
        for row in result.artifact["client_traversal_contract"]:
            client_id = row["client_id"]
            n = len(rows[client_id])
            self.assertEqual(
                row["nonprivate_exact_record_evaluations_per_round"], policy.local_epochs * n
            )
            private_expected = float.fromhex(
                row["private_expected_record_evaluations_per_round_hex"]
            )
            self.assertEqual(
                private_expected,
                policy.poisson_sample_rate
                * math.ceil(policy.local_epochs / policy.poisson_sample_rate)
                * n,
            )

    def test_time_schedule_is_calibrated_by_server_round_phase(self) -> None:
        parameters, spec, policy, result = _build("time_dpfedadam")
        evidence = result.artifact["noise_calibration"]
        saving_rounds = math.ceil(
            policy.server_rounds * parameters["privacy_schedule"]["saving_round_fraction"]
        )
        steps_per_round = math.ceil(policy.local_epochs / policy.poisson_sample_rate)
        self.assertEqual(evidence["saving_round_count"], saving_rounds)
        self.assertEqual(evidence["phase_one_steps"], saving_rounds * steps_per_round)
        self.assertEqual(evidence["total_steps"], policy.server_rounds * steps_per_round)
        self.assertEqual(
            evidence["phase_one_factor_hex"],
            float(parameters["privacy_schedule"]["saving_sigma_factor"]).hex(),
        )
        self.assertEqual(
            evidence["phase_two_factor_hex"],
            float(parameters["privacy_schedule"]["spending_sigma_factor"]).hex(),
        )
        _require_minimal_noise_calibration(spec, parameters, result.budget)
        _, _, _, uniform_ablation = _build("fedsift_uniform_schedule", policy)
        uniform_evidence = uniform_ablation.artifact["noise_calibration"]
        self.assertIsNone(uniform_evidence["saving_round_count"])
        self.assertEqual(uniform_evidence["phase_one_factor_hex"], (1.0).hex())
        self.assertEqual(uniform_evidence["phase_two_factor_hex"], (1.0).hex())

    def test_candidate_hyperparameters_do_not_break_seed_or_batch_pairing(self) -> None:
        policy = _policy()
        rows = _rows()
        for method_id in ("fedavg_nonprivate", "dp_fedavg"):
            with self.subTest(method_id=method_id):
                reference_parameters, reference_spec = _method_inputs(method_id)
                reference = build_training_budget(
                    reference_spec,
                    reference_parameters,
                    rows,
                    policy,
                    expected_dispatch_sha256=reference_spec.dispatch_sha256,
                    expected_candidate_parameters_sha256=canonical_sha256(reference_parameters),
                    expected_policy_sha256=policy.policy_sha256,
                )
                changed_parameters = copy.deepcopy(reference_parameters)
                changed_parameters["local_optimizer"]["learning_rate"] = 0.071
                changed_spec = build_method_execution_spec(
                    method_id=method_id,
                    candidate_parameters=changed_parameters,
                    mechanism_switch=None,
                )
                changed = build_training_budget(
                    changed_spec,
                    changed_parameters,
                    rows,
                    policy,
                    expected_dispatch_sha256=changed_spec.dispatch_sha256,
                    expected_candidate_parameters_sha256=canonical_sha256(changed_parameters),
                    expected_policy_sha256=policy.policy_sha256,
                )
                self.assertNotEqual(
                    reference_spec.candidate_parameters_sha256,
                    changed_spec.candidate_parameters_sha256,
                )
                self.assertEqual(reference.budget, changed.budget)
                self.assertEqual(
                    reference.artifact["nonprivate_explicit_plan_sha256"],
                    changed.artifact["nonprivate_explicit_plan_sha256"],
                )
                self.assertEqual(
                    changed.budget.initialization_seed, policy.paired_initialization_seed
                )

    def test_outer_result_fields_commitment_drift_overlap_and_forgery_fail_closed(self) -> None:
        parameters, spec, policy, result = _build("dp_fedavg")
        with self.assertRaises(TrainingBudgetFactoryError):
            SharedTrainingBudgetPolicy(
                server_rounds=4,
                local_epochs=2,
                poisson_sample_rate=0.3,
                clip_norm=1.25,
                target_epsilon=20.0,
                target_delta=1e-05,
                model_family="logistic_screening",
                participating_client_ids=("outer_test",),
                paired_initialization_seed=20260831,
                nonprivate_order_seed="fixed",
            )
        outer_rows = _rows()
        outer_rows["outer_test"] = (999,)
        with self.assertRaises(TrainingBudgetFactoryError):
            build_training_budget(
                spec,
                parameters,
                outer_rows,
                policy,
                expected_dispatch_sha256=spec.dispatch_sha256,
                expected_candidate_parameters_sha256=canonical_sha256(parameters),
                expected_policy_sha256=policy.policy_sha256,
            )
        contaminated = copy.deepcopy(parameters)
        contaminated["metrics"] = {"auroc": 1.0}
        with self.assertRaises(TrainingBudgetFactoryError):
            build_training_budget(
                spec,
                contaminated,
                _rows(),
                policy,
                expected_dispatch_sha256=spec.dispatch_sha256,
                expected_candidate_parameters_sha256=canonical_sha256(contaminated),
                expected_policy_sha256=policy.policy_sha256,
            )
        overlapping = _rows()
        overlapping["client_1"] = (overlapping["client_0"][0], *overlapping["client_1"][1:])
        with self.assertRaises(TrainingBudgetFactoryError):
            build_training_budget(
                spec,
                parameters,
                overlapping,
                policy,
                expected_dispatch_sha256=spec.dispatch_sha256,
                expected_candidate_parameters_sha256=canonical_sha256(parameters),
                expected_policy_sha256=policy.policy_sha256,
            )
        forged_spec = replace(spec, dispatch_sha256="0" * 64)
        with self.assertRaises(TrainingBudgetFactoryError):
            build_training_budget(
                forged_spec,
                parameters,
                _rows(),
                policy,
                expected_dispatch_sha256=forged_spec.dispatch_sha256,
                expected_candidate_parameters_sha256=canonical_sha256(parameters),
                expected_policy_sha256=policy.policy_sha256,
            )
        with self.assertRaises(TrainingBudgetFactoryError):
            build_training_budget(
                spec,
                parameters,
                _rows(),
                policy,
                expected_dispatch_sha256=spec.dispatch_sha256,
                expected_candidate_parameters_sha256="0" * 64,
                expected_policy_sha256=policy.policy_sha256,
            )
        with self.assertRaises(TrainingBudgetFactoryError):
            TrainingBudgetFactoryResult(
                budget=result.budget,
                artifact=copy.deepcopy(result.artifact),
                _factory_seal=object(),
            )
        tampered_artifact = copy.deepcopy(result.artifact)
        tampered_artifact["dp_optimizer_steps_per_round"] += 1
        with self.assertRaises(TrainingBudgetFactoryError):
            validate_training_budget_factory_result(
                replace(result, artifact=tampered_artifact),
                spec,
                parameters,
                _rows(),
                policy,
                expected_dispatch_sha256=spec.dispatch_sha256,
                expected_candidate_parameters_sha256=canonical_sha256(parameters),
                expected_policy_sha256=policy.policy_sha256,
            )


if __name__ == "__main__":
    unittest.main()
