from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import hashlib
import json
import unittest
from dataclasses import replace
from unittest.mock import patch
from opacus.accountants import PRVAccountant, RDPAccountant
from fedsift import privacy_accounting as accounting
from fedsift.privacy_accounting import (
    EXPECTED_OPACUS_VERSION,
    ParallelCompositionConditions,
    PoissonDPStage,
    PrivacyAccountingError,
    RuntimePrefixAccountant,
    account_poisson_dpsgd,
    accountant_value_cache_info,
    clear_accountant_value_cache,
    client_run_manifest_fingerprint,
    parallel_compose_disjoint_record_partitions,
    sequential_compose_runtime_reports,
)

CANDIDATE_SHA256 = hashlib.sha256(b"accounting-cache-candidate").hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _population_hash(row_ids: tuple[int, ...]) -> str:
    return _digest({"schema": _identity("row_id_sequence"), "row_ids": row_ids})


def _runtime_report(
    *, client_id: str, federated_round: int, population_ids: tuple[int, ...], tag: str
):
    schedule = (PoissonDPStage("local_train", 0.25, 2.0, 1),)
    context_hash = _digest(
        {
            "client_id": client_id,
            "federated_round": federated_round,
            "method_id": "dp_fedavg",
            "candidate_sha256": CANDIDATE_SHA256,
        }
    )
    runtime = RuntimePrefixAccountant(schedule, delta=1e-05, eps_error=0.01, delta_error=1e-08)
    runtime.record_optimizer_step(
        optimizer_step_index=1,
        sample_rate=0.25,
        noise_multiplier=2.0,
        sampled_record_count=1,
        gaussian_mechanism_executed=True,
        mechanism_evidence_sha256=_digest([tag, "mechanism"]),
        population_row_ids_sha256=_population_hash(population_ids),
        sampling_rng_stream_sha256=_digest([tag, "sampling"]),
        noise_rng_stream_sha256=_digest([tag, "noise"]),
        local_training_context_sha256=context_hash,
        client_id=client_id,
        federated_round=federated_round,
        method_id="dp_fedavg",
        candidate_sha256=CANDIDATE_SHA256,
    )
    return runtime.close(executed_optimizer_steps=1)


def _client_run(*, client_id: str, population_ids: tuple[int, ...]):
    reports = tuple(
        (
            _runtime_report(
                client_id=client_id,
                federated_round=round_index,
                population_ids=population_ids,
                tag=f"{client_id}/round-{round_index}",
            )
            for round_index in (1, 2)
        )
    )
    first = reports[0].execution_history[0]
    contexts = {
        report.execution_history[0]
        .federated_round: report.execution_history[0]
        .local_training_context_sha256
        for report in reports
    }
    manifest = client_run_manifest_fingerprint(
        client_id=client_id,
        method_id="dp_fedavg",
        candidate_sha256=CANDIDATE_SHA256,
        population_row_ids_sha256=first.population_row_ids_sha256,
        expected_round_context_sha256=contexts,
    )
    return sequential_compose_runtime_reports(
        reports,
        expected_round_context_sha256=contexts,
        client_run_manifest_sha256=manifest,
        delta=1e-05,
        eps_error=0.01,
        delta_error=1e-08,
    )


def _parallel_report():
    reports = {
        "client_0": _client_run(client_id="client_0", population_ids=(0, 1)),
        "client_1": _client_run(client_id="client_1", population_ids=(2, 3)),
    }
    memberships = {"client_0": (0, 1), "client_1": (2, 3)}
    conditions = ParallelCompositionConditions(
        partitions_fixed_before_private_training=True,
        assignment_uses_no_unprotected_private_values=True,
        assignment_uses_private_labels_or_outcomes=False,
        mechanisms_read_only_their_assigned_partition=True,
        records_never_migrate_or_recur_across_partitions=True,
        no_raw_record_cross_partition_state=True,
        client_rng_streams_domain_separated_and_nonreused=True,
        client_randomness_independent_or_joint_dp_proved=True,
    )
    return parallel_compose_disjoint_record_partitions(reports, memberships, conditions=conditions)


class AccountingValueCacheTests(unittest.TestCase):

    def setUp(self) -> None:
        clear_accountant_value_cache()

    def tearDown(self) -> None:
        clear_accountant_value_cache()

    def test_cold_and_hot_static_reports_are_fieldwise_identical_and_solve_once(self) -> None:
        schedule = (PoissonDPStage("uniform", 0.25, 2.0, 3),)
        with patch.object(
            accounting, "_solve_accountant_values", wraps=accounting._solve_accountant_values
        ) as solve:
            cold = account_poisson_dpsgd(schedule, delta=1e-05, eps_error=0.01, delta_error=1e-08)
            hot = account_poisson_dpsgd(schedule, delta=1e-05, eps_error=0.01, delta_error=1e-08)
        self.assertEqual(cold, hot)
        self.assertEqual(solve.call_count, 1)
        info = accountant_value_cache_info()
        self.assertEqual((info.misses, info.hits, info.currsize), (1, 1, 1))

    def test_key_compresses_only_identical_phase_q_sigma_and_binds_parameters(self) -> None:
        split = (PoissonDPStage("uniform", 0.25, 2.0, 1), PoissonDPStage("uniform", 0.25, 2.0, 2))
        merged = (PoissonDPStage("uniform", 0.25, 2.0, 3),)
        phase_changed = (
            PoissonDPStage("saving", 0.25, 2.0, 1),
            PoissonDPStage("spending", 0.25, 2.0, 2),
        )
        account_poisson_dpsgd(split, delta=1e-05, delta_error=1e-08)
        account_poisson_dpsgd(merged, delta=1e-05, delta_error=1e-08)
        after_compressed_hit = accountant_value_cache_info()
        self.assertEqual((after_compressed_hit.misses, after_compressed_hit.hits), (1, 1))
        account_poisson_dpsgd(phase_changed, delta=1e-05, delta_error=1e-08)
        account_poisson_dpsgd(merged, delta=2e-05, delta_error=1e-08)
        account_poisson_dpsgd(merged, delta=1e-05, eps_error=0.02, delta_error=1e-08)
        account_poisson_dpsgd(merged, delta=1e-05, delta_error=2e-08)
        account_poisson_dpsgd(
            (PoissonDPStage("uniform", 0.3, 2.0, 3),), delta=1e-05, delta_error=1e-08
        )
        account_poisson_dpsgd(
            (PoissonDPStage("uniform", 0.25, 2.1, 3),), delta=1e-05, delta_error=1e-08
        )
        account_poisson_dpsgd(
            (PoissonDPStage("uniform", 0.25, 2.0, 4),), delta=1e-05, delta_error=1e-08
        )
        info = accountant_value_cache_info()
        self.assertEqual(info.misses, 8)
        with self.assertRaisesRegex(PrivacyAccountingError, "unexpected Opacus"):
            accounting._cached_accountant_values(
                accounting._accounting_schedule_cache_key(merged),
                1e-05,
                0.01,
                1e-08,
                EXPECTED_OPACUS_VERSION + "-tampered",
            )

    def test_history_is_checked_before_a_primed_cache_can_be_read(self) -> None:
        schedule = (PoissonDPStage("uniform", 0.25, 2.0, 1),)
        account_poisson_dpsgd(schedule, delta=1e-05, delta_error=1e-08)
        before = accountant_value_cache_info()
        prv = PRVAccountant()
        rdp = RDPAccountant()
        prv.step(noise_multiplier=2.0, sample_rate=0.25)
        rdp.step(noise_multiplier=3.0, sample_rate=0.25)
        with self.assertRaisesRegex(PrivacyAccountingError, "registered identical history"):
            accounting._accountant_values(
                prv, rdp, schedule, delta=1e-05, eps_error=0.01, delta_error=1e-08
            )
        after = accountant_value_cache_info()
        self.assertEqual(after, before)

    def test_cold_and_hot_sequential_parallel_artifacts_and_hashes_are_identical(self) -> None:
        cold = _parallel_report()
        cold_info = accountant_value_cache_info()
        hot = _parallel_report()
        hot_info = accountant_value_cache_info()
        self.assertEqual(cold, hot)
        self.assertEqual((cold_info.misses, cold_info.currsize), (2, 2))
        self.assertEqual(hot_info.misses, cold_info.misses)
        self.assertGreater(hot_info.hits, cold_info.hits)
        clear_accountant_value_cache()
        cold_client = _client_run(client_id="client_0", population_ids=(0, 1))
        cold_hash = cold_client.artifact_sha256
        hot_client = _client_run(client_id="client_0", population_ids=(0, 1))
        self.assertEqual(cold_client, hot_client)
        self.assertEqual(cold_hash, hot_client.artifact_sha256)
        self.assertEqual(cold_client.component_report_sha256, hot_client.component_report_sha256)

    def test_primed_cache_does_not_accept_schedule_or_private_history_tampering(self) -> None:
        report = _runtime_report(
            client_id="client_0", federated_round=1, population_ids=(0, 1), tag="tamper-source"
        )
        accounting._reconstruct_runtime_report(report)
        altered_stage = PoissonDPStage("local_train", 0.25, 2.0, 2)
        with self.assertRaises(PrivacyAccountingError):
            accounting._reconstruct_runtime_report(
                replace(report, registered_schedule=(altered_stage,))
            )
        step = report.execution_history[0]
        mutations = (
            replace(step, sampled_record_count=step.sampled_record_count + 1),
            replace(step, sampling_rng_stream_sha256=_digest("changed-rng")),
            replace(step, local_training_context_sha256=_digest("changed-context")),
            replace(step, candidate_sha256=_digest("changed-candidate")),
        )
        for altered in mutations:
            with self.subTest(field=altered), self.assertRaises(PrivacyAccountingError):
                accounting._reconstruct_runtime_report(
                    replace(report, execution_history=(altered,))
                )


if __name__ == "__main__":
    unittest.main()
