from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import hashlib
import json
import math
import unittest
import warnings
from dataclasses import replace
from unittest.mock import patch
import opacus
from opacus.accountants import PRVAccountant, RDPAccountant
from scipy.integrate import IntegrationWarning
from fedsift.privacy_accounting import (
    EXPECTED_OPACUS_VERSION,
    ParallelCompositionConditions,
    PoissonDPStage,
    PrivacyAccountingError,
    RuntimePrefixAccountant,
    account_poisson_dpsgd,
    calibrate_two_phase_noise,
    client_run_manifest_fingerprint,
    conditionally_compose_label_driven_record_partitions,
    make_two_phase_schedule,
    parallel_compose_disjoint_record_partitions,
    schedule_fingerprint,
    sequential_compose_runtime_reports,
)

CANDIDATE_SHA256 = hashlib.sha256(b"privacy-candidate").hexdigest()


def _evidence(step_index: int, tag: str = "mechanism") -> str:
    return hashlib.sha256(f"{tag}-step-{step_index}".encode("ascii")).hexdigest()


def _stream(step_index: int, purpose: str, tag: str = "mechanism") -> str:
    return hashlib.sha256(f"{tag}-{purpose}-stream-{step_index}".encode("ascii")).hexdigest()


def _population_hash(row_ids: tuple[int, ...] = (0, 1)) -> str:
    payload = json.dumps(
        {"schema": _identity("row_id_sequence"), "row_ids": row_ids},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _context_hash(
    *,
    client_id: str = "client_0",
    federated_round: int = 0,
    method_id: str = "dp_fedavg",
    candidate_sha256: str = CANDIDATE_SHA256,
) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "schema": "privacy-accounting-test-context-v1",
                "client_id": client_id,
                "federated_round": federated_round,
                "method_id": method_id,
                "candidate_sha256": candidate_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _runtime_identity(
    *,
    client_id: str = "client_0",
    federated_round: int = 0,
    method_id: str = "dp_fedavg",
    candidate_sha256: str = CANDIDATE_SHA256,
) -> dict[str, object]:
    return {
        "local_training_context_sha256": _context_hash(
            client_id=client_id,
            federated_round=federated_round,
            method_id=method_id,
            candidate_sha256=candidate_sha256,
        ),
        "client_id": client_id,
        "federated_round": federated_round,
        "method_id": method_id,
        "candidate_sha256": candidate_sha256,
    }


def _conditions(**changes: bool) -> ParallelCompositionConditions:
    values = {
        "partitions_fixed_before_private_training": True,
        "assignment_uses_no_unprotected_private_values": True,
        "assignment_uses_private_labels_or_outcomes": False,
        "mechanisms_read_only_their_assigned_partition": True,
        "records_never_migrate_or_recur_across_partitions": True,
        "no_raw_record_cross_partition_state": True,
        "client_rng_streams_domain_separated_and_nonreused": True,
        "client_randomness_independent_or_joint_dp_proved": True,
    }
    values.update(changes)
    return ParallelCompositionConditions(**values)


def _runtime_report(
    schedule: tuple[PoissonDPStage, ...],
    *,
    delta: float,
    evidence_tag: str = "mechanism",
    population_ids: tuple[int, ...] = (0, 1),
    client_id: str = "client_0",
    federated_round: int = 0,
    method_id: str = "dp_fedavg",
    candidate_sha256: str = CANDIDATE_SHA256,
):
    context_hash = _context_hash(
        client_id=client_id,
        federated_round=federated_round,
        method_id=method_id,
        candidate_sha256=candidate_sha256,
    )
    runtime = RuntimePrefixAccountant(schedule, delta=delta)
    step_index = 0
    for stage in schedule:
        for _ in range(stage.steps):
            step_index += 1
            runtime.record_optimizer_step(
                optimizer_step_index=step_index,
                sample_rate=stage.sample_rate,
                noise_multiplier=stage.noise_multiplier,
                sampled_record_count=1,
                gaussian_mechanism_executed=True,
                mechanism_evidence_sha256=_evidence(step_index, evidence_tag),
                population_row_ids_sha256=_population_hash(population_ids),
                sampling_rng_stream_sha256=_stream(step_index, "sampling", evidence_tag),
                noise_rng_stream_sha256=_stream(step_index, "noise", evidence_tag),
                local_training_context_sha256=context_hash,
                client_id=client_id,
                federated_round=federated_round,
                method_id=method_id,
                candidate_sha256=candidate_sha256,
            )
    return runtime.close(executed_optimizer_steps=step_index)


def _client_run(*reports, delta: float):
    first = reports[0].execution_history[0]
    expected_contexts = {
        report.execution_history[0]
        .federated_round: report.execution_history[0]
        .local_training_context_sha256
        for report in reports
    }
    manifest_hash = client_run_manifest_fingerprint(
        client_id=first.client_id,
        method_id=first.method_id,
        candidate_sha256=first.candidate_sha256,
        population_row_ids_sha256=first.population_row_ids_sha256,
        expected_round_context_sha256=expected_contexts,
    )
    return sequential_compose_runtime_reports(
        reports,
        expected_round_context_sha256=expected_contexts,
        client_run_manifest_sha256=manifest_hash,
        delta=delta,
        eps_error=0.01,
        delta_error=delta / 1000.0,
    )


class RecordPrivacyAccountingGoldenTests(unittest.TestCase):

    def test_numerical_warning_and_rdp_order_boundary_fail_closed(self) -> None:
        schedule = (PoissonDPStage("fixture", 0.1, 1.2, 2),)
        original_prv = PRVAccountant.get_epsilon

        def warned_prv(accountant, *args, **kwargs):
            warnings.warn("forced quadrature warning", IntegrationWarning)
            return original_prv(accountant, *args, **kwargs)

        with patch.object(PRVAccountant, "get_epsilon", warned_prv):
            with self.assertRaisesRegex(PrivacyAccountingError, "IntegrationWarning"):
                account_poisson_dpsgd(schedule, delta=1e-05)
        original_rdp = RDPAccountant.get_privacy_spent

        def boundary_rdp(accountant, *, delta, alphas=None):
            if alphas is None:
                return original_rdp(accountant, delta=delta, alphas=alphas)
            return (1.0, max(alphas))

        with patch.object(RDPAccountant, "get_privacy_spent", boundary_rdp):
            with self.assertRaisesRegex(PrivacyAccountingError, "largest registered order"):
                account_poisson_dpsgd(schedule, delta=1e-05)

    def test_fixed_opacus_154_prv_golden_and_rdp_secondary_bound(self) -> None:
        self.assertEqual(opacus.__version__, EXPECTED_OPACUS_VERSION)
        report = account_poisson_dpsgd(
            (PoissonDPStage(phase="train", sample_rate=0.01, noise_multiplier=1.0, steps=100),),
            delta=1e-05,
            eps_error=0.01,
            delta_error=1e-08,
        )
        self.assertAlmostEqual(report.epsilon_prv_upper, 0.7281339329885865, places=11)
        self.assertAlmostEqual(report.epsilon_rdp, 1.2141452107864754, places=11)
        self.assertEqual(report.accounted_steps, 100)
        self.assertEqual(report.registered_steps, 100)
        self.assertTrue(report.complete)
        self.assertEqual(report.rdp_evidence_status, "same_history_secondary_finite_bound_only")
        self.assertEqual(report.report_kind, "STATIC_PLAN_ACCOUNTING_NOT_EXECUTION_EVIDENCE")
        self.assertEqual(report.execution_history, ())
        self.assertEqual(
            report.evidence_confidentiality, "public_plan_accounting_no_execution_history"
        )
        self.assertFalse(report.performance_metrics_consumed)
        self.assertIn("client_level_dp", report.claims_not_made)
        self.assertIn("end_to_end_system_dp", report.claims_not_made)

    def test_heterogeneous_schedule_matches_manual_opacus_composition(self) -> None:
        schedule = (
            PoissonDPStage("phase_a", 0.02, 1.3, 2),
            PoissonDPStage("phase_b", 0.04, 0.9, 3),
            PoissonDPStage("phase_c", 0.02, 1.3, 1),
        )
        observed = account_poisson_dpsgd(schedule, delta=1e-06, eps_error=0.01, delta_error=1e-09)
        prv = PRVAccountant()
        rdp = RDPAccountant()
        for stage in schedule:
            for _ in range(stage.steps):
                prv.step(noise_multiplier=stage.noise_multiplier, sample_rate=stage.sample_rate)
                rdp.step(noise_multiplier=stage.noise_multiplier, sample_rate=stage.sample_rate)
        expected_prv = prv.get_epsilon(delta=1e-06, eps_error=0.01, delta_error=1e-09)
        expected_rdp = rdp.get_epsilon(delta=1e-06)
        self.assertAlmostEqual(observed.epsilon_prv_upper, expected_prv, places=12)
        self.assertAlmostEqual(observed.epsilon_rdp, expected_rdp, places=12)
        self.assertEqual(observed.accounted_steps, 6)

    def test_schedule_fingerprint_binds_phase_q_sigma_and_steps(self) -> None:
        schedule = make_two_phase_schedule(
            sample_rate=0.02,
            base_noise_multiplier=1.5,
            total_steps=7,
            phase_one_steps=3,
            phase_one_factor=1.2,
            phase_two_factor=0.8,
        )
        self.assertAlmostEqual(schedule[0].noise_multiplier, 1.8)
        self.assertAlmostEqual(schedule[1].noise_multiplier, 1.2)
        self.assertEqual((schedule[0].steps, schedule[1].steps), (3, 4))
        self.assertEqual(schedule_fingerprint(schedule), schedule_fingerprint(schedule))
        changed = (schedule[0], PoissonDPStage("phase_two", 0.02, 1.2, 5))
        self.assertNotEqual(schedule_fingerprint(schedule), schedule_fingerprint(changed))


class PrivacyAccountingMonotonicityTests(unittest.TestCase):

    def _epsilon(self, *, q: float, sigma: float, steps: int) -> float:
        return account_poisson_dpsgd(
            (PoissonDPStage("train", q, sigma, steps),),
            delta=1e-05,
            eps_error=0.01,
            delta_error=1e-08,
        ).epsilon_prv_upper

    def test_more_steps_larger_q_and_less_noise_do_not_improve_privacy(self) -> None:
        base = self._epsilon(q=0.02, sigma=1.5, steps=4)
        self.assertGreater(self._epsilon(q=0.02, sigma=1.5, steps=8), base)
        self.assertGreater(self._epsilon(q=0.04, sigma=1.5, steps=4), base)
        self.assertGreater(self._epsilon(q=0.02, sigma=1.0, steps=4), base)
        self.assertLess(self._epsilon(q=0.02, sigma=2.0, steps=4), base)

    def test_two_phase_target_calibration_is_result_blind_and_feasible(self) -> None:
        calibrated = calibrate_two_phase_noise(
            sample_rate=0.02,
            total_steps=6,
            phase_one_steps=2,
            phase_one_factor=1.25,
            phase_two_factor=0.75,
            target_epsilon=1.0,
            delta=1e-05,
            eps_error=0.01,
            delta_error=1e-08,
            initial_noise_upper=1.0,
            noise_relative_tolerance=0.01,
            max_bisection_iterations=20,
        )
        self.assertTrue(calibrated.result_blind)
        self.assertFalse(calibrated.performance_metrics_consumed)
        self.assertLessEqual(calibrated.report.epsilon_prv_upper, 1.0)
        self.assertEqual(calibrated.base_noise_multiplier, calibrated.upper_feasible_bound)
        self.assertLess(calibrated.lower_infeasible_bound, calibrated.upper_feasible_bound)
        self.assertEqual(calibrated.report.accounted_steps, 6)


class RuntimePrefixAccountingTests(unittest.TestCase):

    def test_empty_poisson_draw_still_advances_one_accounted_step(self) -> None:
        schedule = (PoissonDPStage("train", 0.1, 2.0, 2),)
        runtime = RuntimePrefixAccountant(schedule, delta=1e-05, eps_error=0.01, delta_error=1e-08)
        runtime.record_optimizer_step(
            optimizer_step_index=1,
            sample_rate=0.1,
            noise_multiplier=2.0,
            sampled_record_count=0,
            gaussian_mechanism_executed=True,
            mechanism_evidence_sha256=_evidence(1),
            population_row_ids_sha256=_population_hash(),
            sampling_rng_stream_sha256=_stream(1, "sampling"),
            noise_rng_stream_sha256=_stream(1, "noise"),
            **_runtime_identity(),
        )
        prefix = runtime.prefix_report()
        self.assertEqual(prefix.accounted_steps, 1)
        self.assertEqual(prefix.empty_sample_steps, 1)
        self.assertFalse(prefix.complete)
        self.assertGreater(prefix.epsilon_prv_upper, 0.0)
        runtime.record_optimizer_step(
            optimizer_step_index=2,
            sample_rate=0.1,
            noise_multiplier=2.0,
            sampled_record_count=3,
            gaussian_mechanism_executed=True,
            mechanism_evidence_sha256=_evidence(2),
            population_row_ids_sha256=_population_hash(),
            sampling_rng_stream_sha256=_stream(2, "sampling"),
            noise_rng_stream_sha256=_stream(2, "noise"),
            **_runtime_identity(),
        )
        closed = runtime.close(executed_optimizer_steps=2)
        self.assertTrue(closed.complete)
        self.assertEqual(closed.empty_sample_steps, 1)
        self.assertEqual(closed.accounted_steps, 2)
        self.assertEqual(closed.report_kind, "RUNTIME_HISTORY_WITH_STEP_EVIDENCE_REACCOUNTABLE")
        self.assertEqual(closed.schema, _identity("record_dp_runtime_report"))
        self.assertTrue(closed.evidence_confidentiality.endswith("do_not_publish"))
        self.assertEqual(closed.accounted_schedule, schedule)
        self.assertEqual(closed.registered_schedule, schedule)
        self.assertEqual(len(closed.execution_history), 2)
        self.assertEqual([step.sampled_record_count for step in closed.execution_history], [0, 3])
        self.assertEqual(
            [step.mechanism_evidence_sha256 for step in closed.execution_history],
            [_evidence(1), _evidence(2)],
        )
        with self.assertRaises(PrivacyAccountingError):
            runtime.record_optimizer_step(
                optimizer_step_index=3,
                sample_rate=0.1,
                noise_multiplier=2.0,
                sampled_record_count=1,
                gaussian_mechanism_executed=True,
                mechanism_evidence_sha256=_evidence(3),
                population_row_ids_sha256=_population_hash(),
                sampling_rng_stream_sha256=_stream(3, "sampling"),
                noise_rng_stream_sha256=_stream(3, "noise"),
                **_runtime_identity(),
            )

    def test_execution_accounting_or_mechanism_mismatch_fails_closed(self) -> None:
        schedule = make_two_phase_schedule(
            sample_rate=0.05,
            base_noise_multiplier=1.0,
            total_steps=3,
            phase_one_steps=1,
            phase_one_factor=2.0,
            phase_two_factor=1.0,
        )
        runtime = RuntimePrefixAccountant(schedule, delta=1e-05)
        with self.assertRaises(PrivacyAccountingError):
            runtime.record_optimizer_step(
                optimizer_step_index=1,
                sample_rate=0.05,
                noise_multiplier=2.0,
                sampled_record_count=0,
                gaussian_mechanism_executed=False,
                mechanism_evidence_sha256=_evidence(1),
                population_row_ids_sha256=_population_hash(),
                sampling_rng_stream_sha256=_stream(1, "sampling"),
                noise_rng_stream_sha256=_stream(1, "noise"),
                **_runtime_identity(),
            )
        with self.assertRaisesRegex(PrivacyAccountingError, "mechanism evidence hash"):
            runtime.record_optimizer_step(
                optimizer_step_index=1,
                sample_rate=0.05,
                noise_multiplier=2.0,
                sampled_record_count=1,
                gaussian_mechanism_executed=True,
                mechanism_evidence_sha256="forged",
                population_row_ids_sha256=_population_hash(),
                sampling_rng_stream_sha256=_stream(1, "sampling"),
                noise_rng_stream_sha256=_stream(1, "noise"),
                **_runtime_identity(),
            )
        with self.assertRaises(PrivacyAccountingError):
            runtime.record_optimizer_step(
                optimizer_step_index=1,
                sample_rate=0.06,
                noise_multiplier=2.0,
                sampled_record_count=1,
                gaussian_mechanism_executed=True,
                mechanism_evidence_sha256=_evidence(1),
                population_row_ids_sha256=_population_hash(),
                sampling_rng_stream_sha256=_stream(1, "sampling"),
                noise_rng_stream_sha256=_stream(1, "noise"),
                **_runtime_identity(),
            )
        self.assertEqual(runtime.recorded_steps, 0)
        runtime.record_optimizer_step(
            optimizer_step_index=1,
            sample_rate=0.05,
            noise_multiplier=2.0,
            sampled_record_count=1,
            gaussian_mechanism_executed=True,
            mechanism_evidence_sha256=_evidence(1),
            population_row_ids_sha256=_population_hash(),
            sampling_rng_stream_sha256=_stream(1, "sampling"),
            noise_rng_stream_sha256=_stream(1, "noise"),
            **_runtime_identity(),
        )
        with self.assertRaises(PrivacyAccountingError):
            runtime.close(executed_optimizer_steps=1)
        with self.assertRaises(PrivacyAccountingError):
            runtime.close(executed_optimizer_steps=3)
        with self.assertRaises(PrivacyAccountingError):
            runtime.record_optimizer_step(
                optimizer_step_index=3,
                sample_rate=0.05,
                noise_multiplier=1.0,
                sampled_record_count=1,
                gaussian_mechanism_executed=True,
                mechanism_evidence_sha256=_evidence(3),
                population_row_ids_sha256=_population_hash(),
                sampling_rng_stream_sha256=_stream(3, "sampling"),
                noise_rng_stream_sha256=_stream(3, "noise"),
                **_runtime_identity(),
            )


class SequentialRuntimeCompositionTests(unittest.TestCase):

    def test_same_client_rounds_are_reaccounted_sequentially_before_parallel_use(self) -> None:
        first = _runtime_report(
            (PoissonDPStage("round_1", 0.05, 1.5, 1),),
            delta=1e-05,
            evidence_tag="round-one",
            federated_round=0,
        )
        second = _runtime_report(
            (PoissonDPStage("round_2", 0.05, 1.5, 1),),
            delta=1e-05,
            evidence_tag="round-two",
            federated_round=1,
        )
        client_run = _client_run(first, second, delta=1e-05)
        combined = client_run.composed_report
        self.assertEqual(client_run.expected_federated_rounds, (0, 1))
        self.assertEqual(combined.accounted_steps, 2)
        self.assertEqual(combined.registered_steps, 2)
        self.assertEqual(
            tuple((step.mechanism_evidence_sha256 for step in combined.execution_history)),
            (
                first.execution_history[0].mechanism_evidence_sha256,
                second.execution_history[0].mechanism_evidence_sha256,
            ),
        )
        self.assertGreater(combined.epsilon_prv_upper, first.epsilon_prv_upper)
        self.assertTrue(combined.complete)
        self.assertEqual(len(client_run.component_report_sha256), 2)

    def test_missing_duplicate_gap_and_mixed_identity_fail_closed(self) -> None:
        first = _runtime_report(
            (PoissonDPStage("round_0", 0.05, 1.5, 1),),
            delta=1e-05,
            evidence_tag="round-zero",
            federated_round=0,
        )
        second = _runtime_report(
            (PoissonDPStage("round_1", 0.05, 1.5, 1),),
            delta=1e-05,
            evidence_tag="round-one",
            federated_round=1,
        )
        expected = {
            0: first.execution_history[0].local_training_context_sha256,
            1: second.execution_history[0].local_training_context_sha256,
        }
        manifest = client_run_manifest_fingerprint(
            client_id="client_0",
            method_id="dp_fedavg",
            candidate_sha256=CANDIDATE_SHA256,
            population_row_ids_sha256=_population_hash(),
            expected_round_context_sha256=expected,
        )
        with self.assertRaisesRegex(PrivacyAccountingError, "omit, duplicate"):
            sequential_compose_runtime_reports(
                (first,),
                expected_round_context_sha256=expected,
                client_run_manifest_sha256=manifest,
                delta=1e-05,
            )
        with self.assertRaisesRegex(PrivacyAccountingError, "omit, duplicate"):
            sequential_compose_runtime_reports(
                (first, first),
                expected_round_context_sha256=expected,
                client_run_manifest_sha256=manifest,
                delta=1e-05,
            )
        with self.assertRaisesRegex(PrivacyAccountingError, "contiguous"):
            client_run_manifest_fingerprint(
                client_id="client_0",
                method_id="dp_fedavg",
                candidate_sha256=CANDIDATE_SHA256,
                population_row_ids_sha256=_population_hash(),
                expected_round_context_sha256={0: expected[0], 2: expected[1]},
            )
        mixed_client = _runtime_report(
            (PoissonDPStage("round_1", 0.05, 1.5, 1),),
            delta=1e-05,
            evidence_tag="mixed-client",
            client_id="client_other",
            federated_round=1,
        )
        with self.assertRaisesRegex(PrivacyAccountingError, "mixes client"):
            sequential_compose_runtime_reports(
                (first, mixed_client),
                expected_round_context_sha256=expected,
                client_run_manifest_sha256=manifest,
                delta=1e-05,
            )
        for label, changed in (
            (
                "method",
                _runtime_report(
                    (PoissonDPStage("round_1", 0.05, 1.5, 1),),
                    delta=1e-05,
                    evidence_tag="mixed-method",
                    federated_round=1,
                    method_id="dp_fedadam",
                ),
            ),
            (
                "candidate",
                _runtime_report(
                    (PoissonDPStage("round_1", 0.05, 1.5, 1),),
                    delta=1e-05,
                    evidence_tag="mixed-candidate",
                    federated_round=1,
                    candidate_sha256=hashlib.sha256(b"other-candidate").hexdigest(),
                ),
            ),
            (
                "population",
                _runtime_report(
                    (PoissonDPStage("round_1", 0.05, 1.5, 1),),
                    delta=1e-05,
                    evidence_tag="mixed-population",
                    federated_round=1,
                    population_ids=(0, 2),
                ),
            ),
        ):
            with (
                self.subTest(label=label),
                self.assertRaisesRegex(PrivacyAccountingError, "mixes client"),
            ):
                sequential_compose_runtime_reports(
                    (first, changed),
                    expected_round_context_sha256=expected,
                    client_run_manifest_sha256=manifest,
                    delta=1e-05,
                )

    def test_duplicate_execution_and_static_plan_fail_closed(self) -> None:
        runtime = _runtime_report(
            (PoissonDPStage("round", 0.05, 1.5, 1),), delta=1e-05, evidence_tag="unique-round"
        )
        context = runtime.execution_history[0].local_training_context_sha256
        manifest = client_run_manifest_fingerprint(
            client_id="client_0",
            method_id="dp_fedavg",
            candidate_sha256=CANDIDATE_SHA256,
            population_row_ids_sha256=_population_hash(),
            expected_round_context_sha256={0: context},
        )
        with self.assertRaisesRegex(PrivacyAccountingError, "omit, duplicate"):
            sequential_compose_runtime_reports(
                (runtime, runtime),
                expected_round_context_sha256={0: context},
                client_run_manifest_sha256=manifest,
                delta=1e-05,
            )
        static = account_poisson_dpsgd((PoissonDPStage("round", 0.05, 1.5, 1),), delta=1e-05)
        with self.assertRaisesRegex(PrivacyAccountingError, "static plan"):
            sequential_compose_runtime_reports(
                (static,),
                expected_round_context_sha256={0: context},
                client_run_manifest_sha256=manifest,
                delta=1e-05,
            )


class DisjointClientParallelCompositionTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.gate_a = _runtime_report(
            (PoissonDPStage("client_train", 0.02, 1.5, 2),),
            delta=1e-05,
            evidence_tag="client-a",
            population_ids=(0, 1),
            client_id="client_a",
        )
        cls.gate_b = _runtime_report(
            (PoissonDPStage("client_train", 0.03, 1.2, 2),),
            delta=2e-05,
            evidence_tag="client-b",
            population_ids=(2, 3),
            client_id="client_b",
        )
        cls.report_a = _client_run(cls.gate_a, delta=1e-05)
        cls.report_b = _client_run(cls.gate_b, delta=2e-05)

    def test_max_rule_requires_and_binds_fixed_disjoint_record_partitions(self) -> None:
        combined = parallel_compose_disjoint_record_partitions(
            {"client_a": self.report_a, "client_b": self.report_b},
            {"client_a": [0, 1], "client_b": [2, 3]},
            conditions=_conditions(),
        )
        self.assertEqual(
            combined.epsilon,
            max(
                self.report_a.composed_report.epsilon_prv_upper,
                self.report_b.composed_report.epsilon_prv_upper,
            ),
        )
        self.assertEqual(combined.delta, 2e-05)
        self.assertTrue(combined.conditions_attested_and_membership_checked)
        self.assertEqual(combined.client_count, 2)
        self.assertIn("client_level_dp", combined.claims_not_made)
        self.assertIn("end_to_end_system_dp", combined.claims_not_made)
        self.assertEqual(combined.conditioning_status, "unconditional_fixed_nonprivate_partition")

    def test_overlap_or_unproved_condition_fails_closed(self) -> None:
        reports = {"client_a": self.report_a, "client_b": self.report_b}
        with self.assertRaises(PrivacyAccountingError):
            parallel_compose_disjoint_record_partitions(
                reports, {"client_a": [0, 1], "client_b": [1, 2]}, conditions=_conditions()
            )
        with self.assertRaises(PrivacyAccountingError):
            parallel_compose_disjoint_record_partitions(
                reports,
                {"client_a": [0, 1], "client_b": [2, 3]},
                conditions=_conditions(assignment_uses_no_unprotected_private_values=False),
            )
        with self.assertRaises(PrivacyAccountingError):
            parallel_compose_disjoint_record_partitions(
                reports,
                {"client_a": [0, 1], "client_b": [2, 3]},
                conditions=_conditions(client_randomness_independent_or_joint_dp_proved=False),
            )

    def test_single_gate_static_and_forged_runtime_cannot_compose(self) -> None:
        with self.assertRaisesRegex(PrivacyAccountingError, "complete sequential"):
            parallel_compose_disjoint_record_partitions(
                {"client_a": self.gate_a}, {"client_a": [0, 1]}, conditions=_conditions()
            )
        static = account_poisson_dpsgd((PoissonDPStage("client_train", 0.02, 1.5, 2),), delta=1e-05)
        with self.assertRaisesRegex(PrivacyAccountingError, "complete sequential"):
            parallel_compose_disjoint_record_partitions(
                {"client_a": static}, {"client_a": [0, 1]}, conditions=_conditions()
            )
        forged = replace(
            self.report_a,
            composed_report=replace(self.report_a.composed_report, epsilon_prv_upper=0.0),
        )
        with self.assertRaisesRegex(PrivacyAccountingError, "re-accounting"):
            parallel_compose_disjoint_record_partitions(
                {"client_a": forged}, {"client_a": [0, 1]}, conditions=_conditions()
            )
        changed_step = replace(
            self.report_a.composed_report.execution_history[0], mechanism_evidence_sha256="f" * 64
        )
        forged_history = replace(
            self.report_a,
            composed_report=replace(
                self.report_a.composed_report,
                execution_history=(
                    changed_step,
                    *self.report_a.composed_report.execution_history[1:],
                ),
            ),
        )
        with self.assertRaisesRegex(PrivacyAccountingError, "history hash"):
            parallel_compose_disjoint_record_partitions(
                {"client_a": forged_history}, {"client_a": [0, 1]}, conditions=_conditions()
            )

    def test_membership_identity_rng_and_record_id_tampering_fail_closed(self) -> None:
        reports = {"client_a": self.report_a, "client_b": self.report_b}
        with self.assertRaisesRegex(PrivacyAccountingError, "population hash"):
            parallel_compose_disjoint_record_partitions(
                reports, {"client_a": [0, 4], "client_b": [2, 3]}, conditions=_conditions()
            )
        with self.assertRaisesRegex(PrivacyAccountingError, "exact Python integers"):
            parallel_compose_disjoint_record_partitions(
                reports, {"client_a": [0, 1], "client_b": [2, "1"]}, conditions=_conditions()
            )
        with self.assertRaisesRegex(PrivacyAccountingError, "nonnegative"):
            parallel_compose_disjoint_record_partitions(
                reports, {"client_a": [0, 1], "client_b": [-1, 3]}, conditions=_conditions()
            )
        with self.assertRaisesRegex(PrivacyAccountingError, "mapping key"):
            parallel_compose_disjoint_record_partitions(
                {"renamed": self.report_a}, {"renamed": [0, 1]}, conditions=_conditions()
            )
        shared_a = _client_run(
            _runtime_report(
                (PoissonDPStage("train", 0.02, 1.5, 1),),
                delta=1e-05,
                evidence_tag="shared-stream",
                client_id="client_a",
                population_ids=(0, 1),
            ),
            delta=1e-05,
        )
        shared_b = _client_run(
            _runtime_report(
                (PoissonDPStage("train", 0.02, 1.5, 1),),
                delta=1e-05,
                evidence_tag="shared-stream",
                client_id="client_b",
                population_ids=(2, 3),
            ),
            delta=1e-05,
        )
        with self.assertRaisesRegex(PrivacyAccountingError, "RNG stream"):
            parallel_compose_disjoint_record_partitions(
                {"client_a": shared_a, "client_b": shared_b},
                {"client_a": [0, 1], "client_b": [2, 3]},
                conditions=_conditions(),
            )

    def test_label_driven_partition_is_only_conditionally_reportable(self) -> None:
        reports = {"client_a": self.report_a, "client_b": self.report_b}
        with self.assertRaisesRegex(PrivacyAccountingError, "label- or outcome-driven"):
            parallel_compose_disjoint_record_partitions(
                reports,
                {"client_a": [0, 1], "client_b": [2, 3]},
                conditions=_conditions(
                    assignment_uses_no_unprotected_private_values=True,
                    assignment_uses_private_labels_or_outcomes=True,
                ),
            )
        partition_hash = hashlib.sha256(b"frozen-label-skew-partition").hexdigest()
        conditional = conditionally_compose_label_driven_record_partitions(
            reports,
            {"client_a": [0, 1], "client_b": [2, 3]},
            conditions=_conditions(
                assignment_uses_no_unprotected_private_values=False,
                assignment_uses_private_labels_or_outcomes=True,
            ),
            fixed_auxiliary_partition_condition_sha256=partition_hash,
        )
        self.assertEqual(conditional.fixed_auxiliary_partition_condition_sha256, partition_hash)
        self.assertIn("not_end_to_end", conditional.conditioning_status)
        self.assertIn("conditional_record_level_dp", conditional.guarantee_scope)
        self.assertIn("end_to_end_system_dp", conditional.claims_not_made)


class PrivacyAccountingInputValidationTests(unittest.TestCase):

    def test_invalid_stage_parameters_fail_closed(self) -> None:
        invalid_builders = (
            lambda: PoissonDPStage("", 0.1, 1.0, 1),
            lambda: PoissonDPStage(" train", 0.1, 1.0, 1),
            lambda: PoissonDPStage("train", 0.0, 1.0, 1),
            lambda: PoissonDPStage("train", 1.1, 1.0, 1),
            lambda: PoissonDPStage("train", math.nan, 1.0, 1),
            lambda: PoissonDPStage("train", 0.1, 0.0, 1),
            lambda: PoissonDPStage("train", 0.1, math.inf, 1),
            lambda: PoissonDPStage("train", 0.1, 1.0, 0),
            lambda: PoissonDPStage("train", 0.1, 1.0, True),
            lambda: PoissonDPStage("train", 0.1, 1.0, 1.5),
        )
        for build in invalid_builders:
            with self.subTest(build=build), self.assertRaises(PrivacyAccountingError):
                build()

    def test_invalid_privacy_and_schedule_parameters_fail_closed(self) -> None:
        stage = PoissonDPStage("train", 0.1, 1.0, 1)
        invalid_calls = (
            lambda: account_poisson_dpsgd((), delta=1e-05),
            lambda: account_poisson_dpsgd((stage,), delta=0.0),
            lambda: account_poisson_dpsgd((stage,), delta=1.0),
            lambda: account_poisson_dpsgd((stage,), delta=math.nan),
            lambda: account_poisson_dpsgd((stage,), delta=1e-05, eps_error=0.0),
            lambda: account_poisson_dpsgd((stage,), delta=1e-05, eps_error=1.0),
            lambda: account_poisson_dpsgd((stage,), delta=1e-05, delta_error=1e-05),
            lambda: make_two_phase_schedule(
                sample_rate=0.1,
                base_noise_multiplier=1.0,
                total_steps=2,
                phase_one_steps=2,
                phase_one_factor=1.0,
                phase_two_factor=1.0,
            ),
            lambda: make_two_phase_schedule(
                sample_rate=0.1,
                base_noise_multiplier=1.0,
                total_steps=2,
                phase_one_steps=1,
                phase_one_factor=0.0,
                phase_two_factor=1.0,
            ),
        )
        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises(PrivacyAccountingError):
                call()

    def test_parallel_composition_rejects_duplicate_and_mismatched_partitions(self) -> None:
        report = _client_run(
            _runtime_report(
                (PoissonDPStage("train", 0.01, 2.0, 1),),
                delta=1e-05,
                evidence_tag="input-validation",
                population_ids=(1, 2),
                client_id="a",
            ),
            delta=1e-05,
        )
        with self.assertRaises(PrivacyAccountingError):
            parallel_compose_disjoint_record_partitions(
                {"a": report}, {"a": [1, 1]}, conditions=_conditions()
            )
        with self.assertRaises(PrivacyAccountingError):
            parallel_compose_disjoint_record_partitions(
                {"a": report}, {"b": [1]}, conditions=_conditions()
            )


if __name__ == "__main__":
    unittest.main()
