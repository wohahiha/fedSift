"""Prewarmed, order-balanced wall-clock and process-RSS benchmark.

This module is deliberately separate from :mod:`resource_accounting`, whose
receipts contain deterministic protocol work only.  Benchmark runs reuse the
candidate-zero cost-smoke capability and its cyclic Latin-square schedule, but
measure the executor from outside.  An executor can return only round resource
receipts; predictions, losses, HPO decisions, OOF output, and outer authority
remain outside the benchmark boundary.

The production memory backend samples Linux ``/proc/self/status`` and is meant
for the frozen WSL CPU environment.  Absolute process RSS and the within-run
increase are both retained.  They are descriptive capacity evidence, not a
claim about algorithmic memory complexity, because native allocators may keep
buffers between runs.  Cyclic order balance reduces that allocator/order
confounding symmetrically across methods.
"""

from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import copy
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from .candidate_space import canonical_sha256, require_sha256
from .resource_protocol import (
    ResourceProtocolBoundaryError,
    assert_resource_protocol_result_blind,
    build_resource_protocol_order_manifest,
    validate_resource_protocol_capability,
)
from .resource_accounting import ResourceAccountingError, aggregate_resource_receipts


class CostBenchmarkError(ValueError):
    """Raised when the timing/RSS benchmark or its artifact is malformed."""


BENCHMARK_ARTIFACT_SCHEMA = _identity("cost_benchmark_artifact")
MEASUREMENT_BACKEND_ID = "linux_proc_status_rss_monotonic_ns_v1"
DEFAULT_SAMPLE_INTERVAL_NS = 5000000
BenchmarkExecutor = Callable[..., object]
MeasurementBackend = Callable[
    [Callable[[], object], Mapping[str, object]], tuple[object, Mapping[str, object]]
]
_MEASUREMENT_FIELDS = {
    "backend_id",
    "clock_id",
    "elapsed_ns",
    "process_cpu_ns",
    "rss_before_bytes",
    "rss_after_bytes",
    "peak_rss_bytes",
    "peak_rss_delta_bytes",
    "rss_sample_count",
    "rss_sample_interval_ns",
}
_RUN_FIELDS = {
    "sequence_index",
    "phase",
    "block_index",
    "row_index",
    "order_position",
    "method_id",
    "candidate_id",
    "included_in_cost_comparison",
    "round_receipt_sha256",
    "resource_report_sha256",
    "measurement",
}
_METHOD_SUMMARY_FIELDS = {
    "method_id",
    "measured_run_count",
    "elapsed_ns_sorted",
    "median_elapsed_ns",
    "p95_elapsed_ns_nearest_rank",
    "process_cpu_ns_sorted",
    "median_process_cpu_ns",
    "p95_process_cpu_ns_nearest_rank",
    "peak_rss_bytes_sorted",
    "median_peak_rss_bytes",
    "p95_peak_rss_bytes_nearest_rank",
    "peak_rss_delta_bytes_sorted",
    "median_peak_rss_delta_bytes",
    "p95_peak_rss_delta_bytes_nearest_rank",
}
_ARTIFACT_FIELDS = {
    "schema",
    "status",
    "candidate_id",
    "capability_sha256",
    "input_sha256",
    "order_manifest_sha256",
    "measurement_scope",
    "execution_ledger",
    "execution_ledger_sha256",
    "method_summaries",
    "method_summaries_sha256",
    "interpretation",
    "artifact_sha256",
}


def _exact_int(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CostBenchmarkError(f"{field} must be an exact integer >= {minimum}")
    return value


def _sha256(value: object, field: str) -> str:
    try:
        return require_sha256(value, field)
    except Exception as exc:
        raise CostBenchmarkError(str(exc)) from exc


def _rehash_without(value: Mapping[str, object], field: str) -> str:
    payload = copy.deepcopy(dict(value))
    payload.pop(field, None)
    return canonical_sha256(payload)


def _expected_schedule(order_manifest: Mapping[str, object]) -> list[dict[str, object]]:
    schedule: list[dict[str, object]] = []
    for phase, field, included in (
        ("warmup", "warmup_orders", False),
        ("measured", "measured_orders", True),
    ):
        entries = order_manifest[field]
        if not isinstance(entries, list):
            raise CostBenchmarkError("order manifest schedule is malformed")
        for entry in entries:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("method_ids"), list):
                raise CostBenchmarkError("order manifest entry is malformed")
            for position, method_id in enumerate(entry["method_ids"]):
                schedule.append(
                    {
                        "sequence_index": len(schedule) + 1,
                        "phase": phase,
                        "block_index": entry["block_index"],
                        "row_index": entry["row_index"],
                        "order_position": position,
                        "method_id": method_id,
                        "candidate_id": 0,
                        "included_in_cost_comparison": included,
                    }
                )
    return schedule


def _read_linux_rss_bytes() -> int:
    """Read current resident bytes without adding a third-party dependency."""
    try:
        with open("/proc/self/status", "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    fields = line.split()
                    if len(fields) != 3 or fields[2] != "kB":
                        break
                    return int(fields[1]) * 1024
    except (OSError, ValueError) as exc:
        raise CostBenchmarkError(
            "formal RSS measurement requires readable Linux /proc/self/status"
        ) from exc
    raise CostBenchmarkError("formal RSS measurement requires VmRSS in Linux /proc/self/status")


def measure_linux_process_rss(
    operation: Callable[[], object],
    context: Mapping[str, object],
    *,
    sample_interval_ns: int = DEFAULT_SAMPLE_INTERVAL_NS,
) -> tuple[object, Mapping[str, object]]:
    """Measure one call using monotonic clocks and sampled process RSS."""
    del context
    interval = _exact_int(sample_interval_ns, "sample_interval_ns", minimum=1000000)
    rss_before = _read_linux_rss_bytes()
    samples = [rss_before]
    sampler_errors: list[BaseException] = []
    stop = threading.Event()

    def sample() -> None:
        try:
            while not stop.wait(interval / 1000000000):
                samples.append(_read_linux_rss_bytes())
        except BaseException as exc:
            sampler_errors.append(exc)
            stop.set()

    sampler = threading.Thread(target=sample, name=_identity("rss_sampler"), daemon=True)
    sampler.start()
    wall_start = time.perf_counter_ns()
    cpu_start = time.process_time_ns()
    try:
        returned = operation()
    finally:
        cpu_end = time.process_time_ns()
        wall_end = time.perf_counter_ns()
        stop.set()
        sampler.join()
        rss_after = _read_linux_rss_bytes()
        samples.append(rss_after)
    if sampler_errors:
        raise CostBenchmarkError(
            "RSS sampler failed during benchmark execution"
        ) from sampler_errors[0]
    measurement = {
        "backend_id": MEASUREMENT_BACKEND_ID,
        "clock_id": "time.perf_counter_ns+time.process_time_ns",
        "elapsed_ns": wall_end - wall_start,
        "process_cpu_ns": cpu_end - cpu_start,
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "peak_rss_bytes": max(samples),
        "peak_rss_delta_bytes": max(0, max(samples) - rss_before),
        "rss_sample_count": len(samples),
        "rss_sample_interval_ns": interval,
    }
    _validate_measurement(measurement)
    return (returned, measurement)


def _validate_measurement(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _MEASUREMENT_FIELDS:
        raise CostBenchmarkError("measurement fields differ from exact schema")
    result = copy.deepcopy(dict(value))
    if type(result["backend_id"]) is not str or not result["backend_id"]:
        raise CostBenchmarkError("measurement backend_id must be a non-empty string")
    if type(result["clock_id"]) is not str or not result["clock_id"]:
        raise CostBenchmarkError("measurement clock_id must be a non-empty string")
    for field in (
        "elapsed_ns",
        "process_cpu_ns",
        "rss_before_bytes",
        "rss_after_bytes",
        "peak_rss_bytes",
        "peak_rss_delta_bytes",
        "rss_sample_count",
        "rss_sample_interval_ns",
    ):
        _exact_int(result[field], field, minimum=1 if field != "peak_rss_delta_bytes" else 0)
    if result["peak_rss_bytes"] < max(result["rss_before_bytes"], result["rss_after_bytes"]):
        raise CostBenchmarkError("peak RSS is below an endpoint RSS observation")
    if result["peak_rss_delta_bytes"] != max(
        0, result["peak_rss_bytes"] - result["rss_before_bytes"]
    ):
        raise CostBenchmarkError("peak RSS delta differs from exact reconstruction")
    assert_resource_protocol_result_blind(result, "measurement")
    return result


def _validated_receipt_hashes(
    returned: object, *, method_id: str, expected_rounds: int, model_manifest_sha256: str
) -> tuple[list[str], str]:
    assert_resource_protocol_result_blind(returned, "benchmark_executor_output")
    if isinstance(returned, (str, bytes, Mapping)) or not isinstance(returned, Sequence):
        raise CostBenchmarkError("benchmark executor must return round receipts only")
    receipts: list[dict[str, object]] = []
    for receipt in returned:
        if not isinstance(receipt, Mapping):
            raise CostBenchmarkError("benchmark executor returned a non-mapping receipt")
        receipts.append(copy.deepcopy(dict(receipt)))
    try:
        report = aggregate_resource_receipts(
            receipts,
            expected_method_id=method_id,
            expected_rounds=expected_rounds,
            expected_model_manifest_sha256=model_manifest_sha256,
        )
    except ResourceAccountingError as exc:
        raise CostBenchmarkError("benchmark receipts failed exact reconstruction") from exc
    return ([str(receipt["receipt_sha256"]) for receipt in receipts], str(report["report_sha256"]))


def _median_exact(sorted_values: Sequence[int]) -> int | float:
    count = len(sorted_values)
    if count == 0:
        raise CostBenchmarkError("cannot summarize an empty measured sample")
    midpoint = count // 2
    if count % 2:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2


def _nearest_rank_p95(sorted_values: Sequence[int]) -> int:
    if not sorted_values:
        raise CostBenchmarkError("cannot summarize an empty measured sample")
    return sorted_values[math.ceil(0.95 * len(sorted_values)) - 1]


def _build_method_summaries(
    ledger: Sequence[Mapping[str, object]], method_ids: Sequence[str]
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    metric_fields = (
        ("elapsed_ns", "elapsed_ns"),
        ("process_cpu_ns", "process_cpu_ns"),
        ("peak_rss_bytes", "peak_rss_bytes"),
        ("peak_rss_delta_bytes", "peak_rss_delta_bytes"),
    )
    for method_id in sorted(method_ids):
        records = [
            record
            for record in ledger
            if record["method_id"] == method_id and record["included_in_cost_comparison"] is True
        ]
        summary: dict[str, object] = {"method_id": method_id, "measured_run_count": len(records)}
        for source, label in metric_fields:
            values = sorted((int(record["measurement"][source]) for record in records))
            summary[f"{label}_sorted"] = values
            summary[f"median_{label}"] = _median_exact(values)
            summary[f"p95_{label}_nearest_rank"] = _nearest_rank_p95(values)
        if set(summary) != _METHOD_SUMMARY_FIELDS:
            raise CostBenchmarkError("internal method-summary schema differs")
        summaries.append(summary)
    return summaries


def run_cost_benchmark(
    capability: Mapping[str, object],
    *,
    synthetic_input: object,
    expected_capability_sha256: str,
    resource_executor: BenchmarkExecutor,
    measurement_backend: MeasurementBackend = measure_linux_process_rss,
    outer_capability: object | None = None,
    outer_gate: object | None = None,
    outer_test_gate: object | None = None,
) -> dict[str, object]:
    """Run the frozen prewarm/Latin-square schedule and measure externally."""
    if any((value is not None for value in (outer_capability, outer_gate, outer_test_gate))):
        raise ResourceProtocolBoundaryError(
            "the result-blind cost benchmark cannot receive outer authority"
        )
    if not callable(resource_executor) or not callable(measurement_backend):
        raise CostBenchmarkError("executor and measurement backend must be callable")
    frozen_input = copy.deepcopy(synthetic_input)
    assert_resource_protocol_result_blind(frozen_input, "synthetic_input")
    validate_resource_protocol_capability(
        capability,
        synthetic_input=frozen_input,
        expected_capability_sha256=expected_capability_sha256,
    )
    order_manifest = build_resource_protocol_order_manifest(
        capability,
        synthetic_input=frozen_input,
        expected_capability_sha256=expected_capability_sha256,
    )
    expected_rounds = int(capability["expected_rounds"])
    model_hash = str(capability["model_manifest_sha256"])
    ledger: list[dict[str, object]] = []
    for run in _expected_schedule(order_manifest):
        context = copy.deepcopy(run)
        operation_calls = 0

        def operation() -> object:
            nonlocal operation_calls
            operation_calls += 1
            if operation_calls != 1:
                raise CostBenchmarkError(
                    "measurement backend must invoke the benchmark operation exactly once"
                )
            return resource_executor(
                method_id=run["method_id"],
                candidate_id=0,
                phase=run["phase"],
                sequence_index=run["sequence_index"],
                block_index=run["block_index"],
                row_index=run["row_index"],
                order_position=run["order_position"],
                synthetic_input=copy.deepcopy(frozen_input),
            )

        returned, raw_measurement = measurement_backend(operation, context)
        if operation_calls != 1:
            raise CostBenchmarkError(
                "measurement backend must invoke the benchmark operation exactly once"
            )
        measurement = _validate_measurement(raw_measurement)
        receipt_hashes, report_hash = _validated_receipt_hashes(
            returned,
            method_id=str(run["method_id"]),
            expected_rounds=expected_rounds,
            model_manifest_sha256=model_hash,
        )
        ledger.append(
            {
                **run,
                "round_receipt_sha256": receipt_hashes,
                "resource_report_sha256": report_hash,
                "measurement": measurement,
            }
        )
    method_summaries = _build_method_summaries(ledger, capability["method_ids"])
    artifact: dict[str, object] = {
        "schema": BENCHMARK_ARTIFACT_SCHEMA,
        "status": "complete_result_blind_descriptive_cost_only",
        "candidate_id": 0,
        "capability_sha256": capability["capability_sha256"],
        "input_sha256": capability["input_sha256"],
        "order_manifest_sha256": order_manifest["order_manifest_sha256"],
        "measurement_scope": "external_executor_wall_clock_process_cpu_and_sampled_absolute_process_rss",
        "execution_ledger": ledger,
        "execution_ledger_sha256": canonical_sha256(ledger),
        "method_summaries": method_summaries,
        "method_summaries_sha256": canonical_sha256(method_summaries),
        "interpretation": "candidate_zero_capacity_evidence_only_not_predictive_performance_or_algorithmic_memory_complexity; absolute_RSS_may_include_retained_native_allocator_buffers; prewarm_and_cyclic_order_balance_apply_symmetrically",
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    validate_cost_benchmark_artifact(
        artifact,
        capability,
        synthetic_input=frozen_input,
        expected_capability_sha256=expected_capability_sha256,
        expected_artifact_sha256=str(artifact["artifact_sha256"]),
    )
    return artifact


def validate_cost_benchmark_artifact(
    artifact: Mapping[str, object],
    capability: Mapping[str, object],
    *,
    synthetic_input: object,
    expected_capability_sha256: str,
    expected_artifact_sha256: str,
) -> None:
    """Validate schedule binding, measurements, summaries, and external hash."""
    validate_resource_protocol_capability(
        capability,
        synthetic_input=synthetic_input,
        expected_capability_sha256=expected_capability_sha256,
    )
    if not isinstance(artifact, Mapping) or set(artifact) != _ARTIFACT_FIELDS:
        raise CostBenchmarkError("cost benchmark fields differ from exact schema")
    assert_resource_protocol_result_blind(artifact, "cost_benchmark_artifact")
    if (
        artifact.get("schema") != BENCHMARK_ARTIFACT_SCHEMA
        or artifact.get("status") != "complete_result_blind_descriptive_cost_only"
        or artifact.get("candidate_id") != 0
    ):
        raise CostBenchmarkError("cost benchmark identity differs")
    if artifact.get("capability_sha256") != capability.get("capability_sha256"):
        raise CostBenchmarkError("cost benchmark capability binding differs")
    if artifact.get("input_sha256") != capability.get("input_sha256"):
        raise CostBenchmarkError("cost benchmark input binding differs")
    expected_order = build_resource_protocol_order_manifest(
        capability,
        synthetic_input=synthetic_input,
        expected_capability_sha256=expected_capability_sha256,
    )
    if artifact.get("order_manifest_sha256") != expected_order.get("order_manifest_sha256"):
        raise CostBenchmarkError("cost benchmark order binding differs")
    ledger = artifact.get("execution_ledger")
    if not isinstance(ledger, list) or len(ledger) != expected_order["execution_count"]:
        raise CostBenchmarkError("cost benchmark execution ledger is incomplete")
    if artifact.get("execution_ledger_sha256") != canonical_sha256(ledger):
        raise CostBenchmarkError("cost benchmark execution ledger hash differs")
    schedule = _expected_schedule(expected_order)
    for expected, record in zip(schedule, ledger):
        if not isinstance(record, Mapping) or set(record) != _RUN_FIELDS:
            raise CostBenchmarkError("cost benchmark run fields differ")
        for field, value in expected.items():
            if record.get(field) != value:
                raise CostBenchmarkError("cost benchmark execution order differs")
        receipt_hashes = record.get("round_receipt_sha256")
        if (
            not isinstance(receipt_hashes, list)
            or len(receipt_hashes) != capability["expected_rounds"]
        ):
            raise CostBenchmarkError("cost benchmark receipt hash list is incomplete")
        for index, value in enumerate(receipt_hashes):
            _sha256(value, f"round_receipt_sha256[{index}]")
        _sha256(record.get("resource_report_sha256"), "resource_report_sha256")
        _validate_measurement(record.get("measurement"))
    expected_summaries = _build_method_summaries(ledger, capability["method_ids"])
    if artifact.get("method_summaries") != expected_summaries:
        raise CostBenchmarkError("cost benchmark method summaries differ")
    if artifact.get("method_summaries_sha256") != canonical_sha256(expected_summaries):
        raise CostBenchmarkError("cost benchmark method summary hash differs")
    committed = _sha256(artifact.get("artifact_sha256"), "artifact_sha256")
    expected = _sha256(expected_artifact_sha256, "expected_artifact_sha256")
    if committed != _rehash_without(artifact, "artifact_sha256"):
        raise CostBenchmarkError("cost benchmark artifact content hash differs")
    if committed != expected:
        raise CostBenchmarkError("cost benchmark differs from external commitment")
