from __future__ import annotations
import copy
import unittest
from fedsift.candidate_space import canonical_sha256
from fedsift.cost_benchmark import (
    CostBenchmarkError,
    run_cost_benchmark,
    validate_cost_benchmark_artifact,
)
from fedsift.resource_protocol import (
    ResourceProtocolBoundaryError,
    build_resource_protocol_capability,
)
from fedsift.resource_accounting import build_round_resource_receipt

METHODS = ("dp_fedavg", "dp_fedprox_adapted", "fedsift")
MODEL_HASH = "a" * 64
INPUT = {"dataset_id": "registered-fixture", "record_count": 24}


def _capability() -> dict[str, object]:
    return build_resource_protocol_capability(
        method_ids=METHODS,
        synthetic_input=INPUT,
        expected_rounds=2,
        model_manifest_sha256=MODEL_HASH,
        warmup_passes=1,
        latin_square_repetitions=1,
    )


class ResourceExecutor:

    def __call__(self, **kwargs: object) -> list[dict[str, object]]:
        method = str(kwargs["method_id"])
        return [
            build_round_resource_receipt(
                method_id=method,
                server_round=round_index,
                participating_client_ids=("client-0", "client-1"),
                total_client_count=2,
                model_payload_bytes=32,
                model_manifest_sha256=MODEL_HASH,
                local_optimizer_steps=4,
                sampled_record_gradient_evaluations=16,
                public_control_query_executed=method == "fedsift",
                public_control_record_count=4 if method == "fedsift" else 0,
                public_control_candidate_count=3 if method == "fedsift" else 0,
            )
            for round_index in (1, 2)
        ]


class DeterministicMeasurement:

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, operation, context):
        self.calls += 1
        returned = operation()
        method_offset = METHODS.index(str(context["method_id"])) * 100
        elapsed = 1000 + method_offset + self.calls
        before = 10000 + self.calls
        peak = before + 200 + method_offset
        return (
            returned,
            {
                "backend_id": "deterministic_test_backend_v1",
                "clock_id": "deterministic_test_clock",
                "elapsed_ns": elapsed,
                "process_cpu_ns": elapsed - 100,
                "rss_before_bytes": before,
                "rss_after_bytes": before + 10,
                "peak_rss_bytes": peak,
                "peak_rss_delta_bytes": peak - before,
                "rss_sample_count": 3,
                "rss_sample_interval_ns": 5000000,
            },
        )


class CostBenchmarkTests(unittest.TestCase):

    def test_balanced_benchmark_is_result_blind_and_exactly_replayable(self) -> None:
        capability = _capability()
        artifact = run_cost_benchmark(
            capability,
            synthetic_input=INPUT,
            expected_capability_sha256=str(capability["capability_sha256"]),
            resource_executor=ResourceExecutor(),
            measurement_backend=DeterministicMeasurement(),
        )
        self.assertEqual(len(artifact["execution_ledger"]), 12)
        for summary in artifact["method_summaries"]:
            self.assertEqual(summary["measured_run_count"], 3)
            self.assertEqual(len(summary["elapsed_ns_sorted"]), 3)
        validate_cost_benchmark_artifact(
            artifact,
            capability,
            synthetic_input=INPUT,
            expected_capability_sha256=str(capability["capability_sha256"]),
            expected_artifact_sha256=str(artifact["artifact_sha256"]),
        )

    def test_executor_output_with_observed_predictions_is_rejected(self) -> None:
        capability = _capability()

        def invalid_executor(**kwargs):
            del kwargs
            return {"predictions": [0.2, 0.8]}

        with self.assertRaises(ResourceProtocolBoundaryError):
            run_cost_benchmark(
                capability,
                synthetic_input=INPUT,
                expected_capability_sha256=str(capability["capability_sha256"]),
                resource_executor=invalid_executor,
                measurement_backend=DeterministicMeasurement(),
            )

    def test_outer_authority_is_rejected(self) -> None:
        capability = _capability()
        with self.assertRaisesRegex(ResourceProtocolBoundaryError, "outer authority"):
            run_cost_benchmark(
                capability,
                synthetic_input=INPUT,
                expected_capability_sha256=str(capability["capability_sha256"]),
                resource_executor=ResourceExecutor(),
                measurement_backend=DeterministicMeasurement(),
                outer_gate={"opened": True},
            )

    def test_measurement_backend_cannot_skip_or_repeat_execution(self) -> None:
        capability = _capability()

        def skipped_backend(operation, context):
            del operation, context
            return (
                [],
                {
                    "backend_id": "invalid_skip",
                    "clock_id": "invalid_skip",
                    "elapsed_ns": 1,
                    "process_cpu_ns": 1,
                    "rss_before_bytes": 1,
                    "rss_after_bytes": 1,
                    "peak_rss_bytes": 1,
                    "peak_rss_delta_bytes": 0,
                    "rss_sample_count": 1,
                    "rss_sample_interval_ns": 1,
                },
            )

        with self.assertRaisesRegex(CostBenchmarkError, "exactly once"):
            run_cost_benchmark(
                capability,
                synthetic_input=INPUT,
                expected_capability_sha256=str(capability["capability_sha256"]),
                resource_executor=ResourceExecutor(),
                measurement_backend=skipped_backend,
            )

        def repeated_backend(operation, context):
            del context
            operation()
            return (operation(), {})

        with self.assertRaisesRegex(CostBenchmarkError, "exactly once"):
            run_cost_benchmark(
                capability,
                synthetic_input=INPUT,
                expected_capability_sha256=str(capability["capability_sha256"]),
                resource_executor=ResourceExecutor(),
                measurement_backend=repeated_backend,
            )

    def test_tampered_measurement_fails_after_rehash(self) -> None:
        capability = _capability()
        artifact = run_cost_benchmark(
            capability,
            synthetic_input=INPUT,
            expected_capability_sha256=str(capability["capability_sha256"]),
            resource_executor=ResourceExecutor(),
            measurement_backend=DeterministicMeasurement(),
        )
        tampered = copy.deepcopy(artifact)
        tampered["execution_ledger"][0]["measurement"]["peak_rss_delta_bytes"] += 1
        tampered["execution_ledger_sha256"] = canonical_sha256(tampered["execution_ledger"])
        tampered["artifact_sha256"] = canonical_sha256(
            {key: value for (key, value) in tampered.items() if key != "artifact_sha256"}
        )
        with self.assertRaisesRegex(CostBenchmarkError, "delta differs"):
            validate_cost_benchmark_artifact(
                tampered,
                capability,
                synthetic_input=INPUT,
                expected_capability_sha256=str(capability["capability_sha256"]),
                expected_artifact_sha256=str(tampered["artifact_sha256"]),
            )


if __name__ == "__main__":
    unittest.main()
