"""Result-blind, candidate-zero resource smoke for FedSift.

The smoke runner deliberately has a much smaller authority surface than an
HPO or outer-refit worker.  Its executor receives only a synthetic input copy
and a candidate-zero method identity, and may return only the existing
per-round resource receipts.  Derived resource values are rebuilt by
``aggregate_resource_receipts``; executor-supplied summaries are never
accepted.

Warm-up executions precede measured executions.  Measured method order is a
deterministic cyclic Latin square, so every method occupies every order
position equally often in each complete square.  The input, capability, order
manifest, execution ledger, and final artifact are all hash committed.
"""

from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import copy
from collections.abc import Callable, Mapping, Sequence
from .candidate_space import (
    CandidateSpaceError,
    assert_no_performance_fields,
    canonical_sha256,
    require_sha256,
)
from .resource_accounting import ResourceAccountingError, aggregate_resource_receipts


class ResourceProtocolError(ValueError):
    """Raised when a cost-smoke capability or artifact is malformed."""


class ResourceProtocolBoundaryError(ResourceProtocolError):
    """Raised when result, selection, OOF, or outer authority crosses the boundary."""


class ResourceProtocolExecutionError(ResourceProtocolError):
    """Raised when an executor does not return a valid resource-receipt ledger."""


CAPABILITY_SCHEMA = _identity("cost_smoke_capability")
ORDER_MANIFEST_SCHEMA = _identity("cost_smoke_order_manifest")
ARTIFACT_SCHEMA = _identity("cost_smoke_artifact")
DEFAULT_ORDER_SEED = _identity("candidate_zero_resource_cost_smoke")
RESOURCE_METHODS: tuple[str, ...] = (
    "fedavg_nonprivate",
    "dp_fedavg",
    "dp_fedprox_adapted",
    "dp_scaffold_adapted",
    "dp_fedadam",
    "dp_fedyogi",
    "dp_fedsofim_delta_proxy_adapted",
    "time_dpfedadam",
    "public_argmin_time_dpfedadam",
    "fedsift",
    "fedsift_uniform_schedule",
    "fedsift_without_sift",
    "fedsift_public_argmin_rule",
)
ResourceExecutor = Callable[..., object]
_CAPABILITY_FIELDS = {
    "schema",
    "status",
    "candidate_id",
    "method_ids",
    "expected_rounds",
    "model_manifest_sha256",
    "input_sha256",
    "ordering",
    "capability_sha256",
}
_ORDERING_FIELDS = {"kind", "order_seed", "warmup_passes", "latin_square_repetitions"}
_ORDER_MANIFEST_FIELDS = {
    "schema",
    "status",
    "candidate_id",
    "capability_sha256",
    "input_sha256",
    "method_ids",
    "ordering_rule",
    "warmup_orders",
    "measured_orders",
    "position_balance",
    "execution_count",
    "order_manifest_sha256",
}
_ORDER_ENTRY_FIELDS = {"block_index", "row_index", "method_ids"}
_ARTIFACT_FIELDS = {
    "schema",
    "status",
    "candidate_id",
    "capability_sha256",
    "input_sha256",
    "order_manifest",
    "execution_ledger",
    "execution_ledger_sha256",
    "artifact_sha256",
}
_EXECUTION_FIELDS = {
    "sequence_index",
    "phase",
    "block_index",
    "row_index",
    "order_position",
    "method_id",
    "candidate_id",
    "included_in_cost_comparison",
    "round_resource_receipts",
    "resource_summary",
}
_EXPLICIT_FORBIDDEN_COMPACT_KEYS = frozenset(
    {
        "alpha",
        "alphas",
        "candidatealphas",
        "chosenalpha",
        "selectedalpha",
        "selectedalphahex",
        "candidatetable",
        "controltable",
        "controlcandidatetable",
        "rawcandidatetable",
        "innervalidation",
        "innervalidationmetrics",
        "hposelection",
        "selectedcandidate",
        "selectiondecision",
        "oof",
        "oofprediction",
        "oofpredictions",
        "outercapability",
        "outerrefitcapability",
        "outergate",
        "outertest",
        "outertestgate",
    }
)
_OBSERVED_SUFFIXES = (
    "loss",
    "metric",
    "metrics",
    "performance",
    "prediction",
    "predictions",
    "score",
    "scores",
)


def _compact_key(value: str) -> str:
    return "".join((character for character in value.casefold() if character.isalnum()))


def _assert_explicit_boundary_fields(value: object, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise ResourceProtocolBoundaryError(
                    f"{path} contains a non-string result-blind artifact key"
                )
            compact = _compact_key(raw_key)
            is_control_table = "control" in compact and "table" in compact
            is_candidate_table = "candidate" in compact and "table" in compact
            is_inner_result = "innervalidation" in compact
            is_hpo_selection = "hpo" in compact and "select" in compact
            is_oof = compact.startswith("oof") or "outoffold" in compact
            is_outer_authority = "outer" in compact and any(
                (token in compact for token in ("capability", "gate", "test"))
            )
            is_observed_output = compact.endswith(_OBSERVED_SUFFIXES)
            if (
                compact in _EXPLICIT_FORBIDDEN_COMPACT_KEYS
                or is_control_table
                or is_candidate_table
                or is_inner_result
                or is_hpo_selection
                or is_oof
                or is_outer_authority
                or is_observed_output
            ):
                raise ResourceProtocolBoundaryError(
                    f"result-blind cost artifact contains forbidden field {path}.{raw_key}"
                )
            _assert_explicit_boundary_fields(child, f"{path}.{raw_key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_explicit_boundary_fields(child, f"{path}[{index}]")


def assert_resource_protocol_result_blind(value: object, path: str = "root") -> None:
    """Apply the shared performance-field gate plus smoke-only boundary gates."""
    try:
        assert_no_performance_fields(value, path)
    except CandidateSpaceError as exc:
        raise ResourceProtocolBoundaryError(str(exc)) from exc
    _assert_explicit_boundary_fields(value, path)


def _reject_outer_authority(
    *, outer_capability: object | None, outer_gate: object | None, outer_test_gate: object | None
) -> None:
    if any((value is not None for value in (outer_capability, outer_gate, outer_test_gate))):
        raise ResourceProtocolBoundaryError(
            "the resource-cost smoke cannot receive or retain outer authority"
        )


def _exact_int(value: object, field: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ResourceProtocolError(f"{field} must be an exact integer >= {minimum}")
    return value


def _candidate_zero(value: object) -> int:
    candidate_id = _exact_int(value, "candidate_id", minimum=0)
    if candidate_id != 0:
        raise ResourceProtocolBoundaryError(
            "the resource-cost smoke is restricted to candidate_id=0"
        )
    return candidate_id


def _sha256(value: object, field: str) -> str:
    try:
        return require_sha256(value, field)
    except CandidateSpaceError as exc:
        raise ResourceProtocolError(str(exc)) from exc


def _method_ids(values: object) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ResourceProtocolError("method_ids must be a non-empty sequence")
    methods = tuple(values)
    if not methods:
        raise ResourceProtocolError("method_ids must be a non-empty sequence")
    if any((type(method) is not str or not method for method in methods)):
        raise ResourceProtocolError("method_ids must contain canonical strings")
    if len(set(methods)) != len(methods):
        raise ResourceProtocolError("method_ids must be unique")
    unknown = set(methods).difference(RESOURCE_METHODS)
    if unknown:
        raise ResourceProtocolError("method_ids contains a method outside resource accounting")
    return tuple(sorted(methods))


def _safe_input(value: object) -> object:
    frozen = copy.deepcopy(value)
    assert_resource_protocol_result_blind(frozen, "synthetic_input")
    try:
        canonical_sha256(frozen)
    except CandidateSpaceError as exc:
        raise ResourceProtocolError("synthetic_input is not strict canonical JSON") from exc
    return frozen


def _rehash_without(value: Mapping[str, object], field: str) -> str:
    payload = copy.deepcopy(dict(value))
    payload.pop(field, None)
    return canonical_sha256(payload)


def build_resource_protocol_capability(
    *,
    method_ids: Sequence[str],
    synthetic_input: object,
    expected_rounds: int,
    model_manifest_sha256: str,
    candidate_id: int = 0,
    warmup_passes: int = 1,
    latin_square_repetitions: int = 1,
    order_seed: str = DEFAULT_ORDER_SEED,
    outer_capability: object | None = None,
    outer_gate: object | None = None,
    outer_test_gate: object | None = None,
) -> dict[str, object]:
    """Build the minimal candidate-zero capability for a synthetic cost smoke."""
    _reject_outer_authority(
        outer_capability=outer_capability, outer_gate=outer_gate, outer_test_gate=outer_test_gate
    )
    candidate = _candidate_zero(candidate_id)
    methods = _method_ids(method_ids)
    rounds = _exact_int(expected_rounds, "expected_rounds", minimum=1)
    warmups = _exact_int(warmup_passes, "warmup_passes", minimum=1)
    repetitions = _exact_int(latin_square_repetitions, "latin_square_repetitions", minimum=1)
    if type(order_seed) is not str or not order_seed or order_seed.strip() != order_seed:
        raise ResourceProtocolError("order_seed must be a non-empty canonical string")
    manifest_hash = _sha256(model_manifest_sha256, "model_manifest_sha256")
    frozen_input = _safe_input(synthetic_input)
    capability: dict[str, object] = {
        "schema": CAPABILITY_SCHEMA,
        "status": "ready_result_blind",
        "candidate_id": candidate,
        "method_ids": list(methods),
        "expected_rounds": rounds,
        "model_manifest_sha256": manifest_hash,
        "input_sha256": canonical_sha256(frozen_input),
        "ordering": {
            "kind": "hash_ranked_cyclic_latin_square_after_warmup",
            "order_seed": order_seed,
            "warmup_passes": warmups,
            "latin_square_repetitions": repetitions,
        },
    }
    capability["capability_sha256"] = canonical_sha256(capability)
    validate_resource_protocol_capability(
        capability,
        synthetic_input=frozen_input,
        expected_capability_sha256=str(capability["capability_sha256"]),
    )
    return capability


def validate_resource_protocol_capability(
    capability: Mapping[str, object], *, synthetic_input: object, expected_capability_sha256: str
) -> None:
    """Validate the exact minimal capability and its independent commitments."""
    if not isinstance(capability, Mapping):
        raise ResourceProtocolError("cost-smoke capability must be a mapping")
    assert_resource_protocol_result_blind(capability, "capability")
    if set(capability) != _CAPABILITY_FIELDS:
        raise ResourceProtocolError("cost-smoke capability fields differ from exact schema")
    if (
        capability.get("schema") != CAPABILITY_SCHEMA
        or capability.get("status") != "ready_result_blind"
    ):
        raise ResourceProtocolError("cost-smoke capability identity differs")
    _candidate_zero(capability.get("candidate_id"))
    methods = _method_ids(capability.get("method_ids"))
    if list(methods) != capability.get("method_ids"):
        raise ResourceProtocolError("cost-smoke method roster is not canonical")
    _exact_int(capability.get("expected_rounds"), "expected_rounds", minimum=1)
    _sha256(capability.get("model_manifest_sha256"), "model_manifest_sha256")
    frozen_input = _safe_input(synthetic_input)
    input_hash = _sha256(capability.get("input_sha256"), "input_sha256")
    if input_hash != canonical_sha256(frozen_input):
        raise ResourceProtocolError("synthetic input differs from capability commitment")
    ordering = capability.get("ordering")
    if not isinstance(ordering, Mapping) or set(ordering) != _ORDERING_FIELDS:
        raise ResourceProtocolError("cost-smoke ordering fields differ")
    if ordering.get("kind") != "hash_ranked_cyclic_latin_square_after_warmup":
        raise ResourceProtocolError("cost-smoke ordering kind differs")
    seed = ordering.get("order_seed")
    if type(seed) is not str or not seed or seed.strip() != seed:
        raise ResourceProtocolError("order_seed must be a non-empty canonical string")
    _exact_int(ordering.get("warmup_passes"), "warmup_passes", minimum=1)
    _exact_int(ordering.get("latin_square_repetitions"), "latin_square_repetitions", minimum=1)
    committed = _sha256(capability.get("capability_sha256"), "capability_sha256")
    expected = _sha256(expected_capability_sha256, "expected_capability_sha256")
    if committed != _rehash_without(capability, "capability_sha256"):
        raise ResourceProtocolError("cost-smoke capability content hash differs")
    if committed != expected:
        raise ResourceProtocolError("cost-smoke capability differs from external commitment")


def _rotate(values: Sequence[str], offset: int) -> list[str]:
    split = offset % len(values)
    return list(values[split:]) + list(values[:split])


def build_resource_protocol_order_manifest(
    capability: Mapping[str, object], *, synthetic_input: object, expected_capability_sha256: str
) -> dict[str, object]:
    """Build the deterministic warm-up and cyclic Latin-square order manifest."""
    validate_resource_protocol_capability(
        capability,
        synthetic_input=synthetic_input,
        expected_capability_sha256=expected_capability_sha256,
    )
    methods = tuple(capability["method_ids"])
    ordering = capability["ordering"]
    assert isinstance(ordering, Mapping)
    input_hash = str(capability["input_sha256"])
    seed = str(ordering["order_seed"])
    base_order = tuple(
        sorted(
            methods,
            key=lambda method: (
                canonical_sha256(
                    {"order_seed": seed, "input_sha256": input_hash, "method_id": method}
                ),
                method,
            ),
        )
    )
    warmup_passes = int(ordering["warmup_passes"])
    repetitions = int(ordering["latin_square_repetitions"])
    warmup_orders = [
        {
            "block_index": pass_index,
            "row_index": 0,
            "method_ids": _rotate(base_order, pass_index - 1),
        }
        for pass_index in range(1, warmup_passes + 1)
    ]
    measured_orders: list[dict[str, object]] = []
    for block_index in range(1, repetitions + 1):
        for row_index in range(1, len(base_order) + 1):
            measured_orders.append(
                {
                    "block_index": block_index,
                    "row_index": row_index,
                    "method_ids": _rotate(base_order, row_index - 1 + block_index - 1),
                }
            )
    position_balance: list[dict[str, object]] = []
    for position in range(len(methods)):
        counts = {method: 0 for method in methods}
        for entry in measured_orders:
            row = entry["method_ids"]
            assert isinstance(row, list)
            counts[str(row[position])] += 1
        position_balance.append({"order_position": position, "method_counts": counts})
    manifest: dict[str, object] = {
        "schema": ORDER_MANIFEST_SCHEMA,
        "status": "complete_result_blind",
        "candidate_id": 0,
        "capability_sha256": capability["capability_sha256"],
        "input_sha256": input_hash,
        "method_ids": list(methods),
        "ordering_rule": "prewarm_then_hash_ranked_cyclic_latin_square",
        "warmup_orders": warmup_orders,
        "measured_orders": measured_orders,
        "position_balance": position_balance,
        "execution_count": len(methods) * (warmup_passes + len(methods) * repetitions),
    }
    manifest["order_manifest_sha256"] = canonical_sha256(manifest)
    assert_resource_protocol_result_blind(manifest, "order_manifest")
    return manifest


def _expected_schedule(order_manifest: Mapping[str, object]) -> list[dict[str, object]]:
    schedule: list[dict[str, object]] = []
    for phase, field, included in (
        ("warmup", "warmup_orders", False),
        ("measured", "measured_orders", True),
    ):
        entries = order_manifest[field]
        assert isinstance(entries, list)
        for entry in entries:
            assert isinstance(entry, Mapping)
            methods = entry["method_ids"]
            assert isinstance(methods, list)
            for position, method_id in enumerate(methods):
                schedule.append(
                    {
                        "sequence_index": len(schedule) + 1,
                        "phase": phase,
                        "block_index": entry["block_index"],
                        "row_index": entry["row_index"],
                        "order_position": position,
                        "method_id": method_id,
                        "included_in_cost_comparison": included,
                    }
                )
    return schedule


def _validated_resource_execution(
    returned: object, *, method_id: str, expected_rounds: int, model_manifest_sha256: str
) -> tuple[list[dict[str, object]], dict[str, object]]:
    assert_resource_protocol_result_blind(returned, "resource_executor_output")
    if isinstance(returned, (str, bytes, Mapping)) or not isinstance(returned, Sequence):
        raise ResourceProtocolExecutionError(
            "resource executor must return only a sequence of round receipts"
        )
    receipts: list[dict[str, object]] = []
    for value in returned:
        if not isinstance(value, Mapping):
            raise ResourceProtocolExecutionError(
                "resource executor returned a non-mapping round receipt"
            )
        receipts.append(copy.deepcopy(dict(value)))
    try:
        summary = aggregate_resource_receipts(
            receipts,
            expected_method_id=method_id,
            expected_rounds=expected_rounds,
            expected_model_manifest_sha256=model_manifest_sha256,
        )
    except ResourceAccountingError as exc:
        raise ResourceProtocolExecutionError(
            "resource executor ledger failed ResourceReceipt reconstruction"
        ) from exc
    assert_resource_protocol_result_blind(summary, "rebuilt_resource_summary")
    return (receipts, summary)


def run_resource_protocol(
    capability: Mapping[str, object],
    *,
    synthetic_input: object,
    expected_capability_sha256: str,
    resource_executor: ResourceExecutor,
    outer_capability: object | None = None,
    outer_gate: object | None = None,
    outer_test_gate: object | None = None,
) -> dict[str, object]:
    """Execute warm-ups and balanced measured runs using resource evidence only.

    ``resource_executor`` is called with keyword-only run context and a fresh
    deep copy of ``synthetic_input``.  It must return only the ordered sequence
    of existing round resource receipts for that one run.
    """
    _reject_outer_authority(
        outer_capability=outer_capability, outer_gate=outer_gate, outer_test_gate=outer_test_gate
    )
    if not callable(resource_executor):
        raise ResourceProtocolError("resource_executor must be callable")
    frozen_input = _safe_input(synthetic_input)
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
    execution_ledger: list[dict[str, object]] = []
    for run in _expected_schedule(order_manifest):
        returned = resource_executor(
            method_id=run["method_id"],
            candidate_id=0,
            phase=run["phase"],
            sequence_index=run["sequence_index"],
            block_index=run["block_index"],
            row_index=run["row_index"],
            order_position=run["order_position"],
            synthetic_input=copy.deepcopy(frozen_input),
        )
        receipts, summary = _validated_resource_execution(
            returned,
            method_id=str(run["method_id"]),
            expected_rounds=expected_rounds,
            model_manifest_sha256=model_hash,
        )
        execution_ledger.append(
            {
                **run,
                "candidate_id": 0,
                "round_resource_receipts": receipts,
                "resource_summary": summary,
            }
        )
    artifact: dict[str, object] = {
        "schema": ARTIFACT_SCHEMA,
        "status": "complete_result_blind",
        "candidate_id": 0,
        "capability_sha256": capability["capability_sha256"],
        "input_sha256": capability["input_sha256"],
        "order_manifest": order_manifest,
        "execution_ledger": execution_ledger,
        "execution_ledger_sha256": canonical_sha256(execution_ledger),
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    assert_resource_protocol_result_blind(artifact, "cost_smoke_artifact")
    validate_resource_protocol_artifact(
        artifact,
        capability,
        synthetic_input=frozen_input,
        expected_capability_sha256=expected_capability_sha256,
        expected_artifact_sha256=str(artifact["artifact_sha256"]),
    )
    return artifact


def validate_resource_protocol_artifact(
    artifact: Mapping[str, object],
    capability: Mapping[str, object],
    *,
    synthetic_input: object,
    expected_capability_sha256: str,
    expected_artifact_sha256: str,
    outer_capability: object | None = None,
    outer_gate: object | None = None,
    outer_test_gate: object | None = None,
) -> None:
    """Rebuild order and resource summaries and verify all external hashes."""
    _reject_outer_authority(
        outer_capability=outer_capability, outer_gate=outer_gate, outer_test_gate=outer_test_gate
    )
    validate_resource_protocol_capability(
        capability,
        synthetic_input=synthetic_input,
        expected_capability_sha256=expected_capability_sha256,
    )
    if not isinstance(artifact, Mapping):
        raise ResourceProtocolError("cost-smoke artifact must be a mapping")
    assert_resource_protocol_result_blind(artifact, "cost_smoke_artifact")
    if set(artifact) != _ARTIFACT_FIELDS:
        raise ResourceProtocolError("cost-smoke artifact fields differ from exact schema")
    if (
        artifact.get("schema") != ARTIFACT_SCHEMA
        or artifact.get("status") != "complete_result_blind"
    ):
        raise ResourceProtocolError("cost-smoke artifact identity differs")
    _candidate_zero(artifact.get("candidate_id"))
    if artifact.get("capability_sha256") != capability.get("capability_sha256") or artifact.get(
        "input_sha256"
    ) != capability.get("input_sha256"):
        raise ResourceProtocolError("cost-smoke artifact capability or input binding differs")
    committed_artifact = _sha256(artifact.get("artifact_sha256"), "artifact_sha256")
    expected_artifact = _sha256(expected_artifact_sha256, "expected_artifact_sha256")
    if committed_artifact != _rehash_without(artifact, "artifact_sha256"):
        raise ResourceProtocolError("cost-smoke artifact content hash differs")
    if committed_artifact != expected_artifact:
        raise ResourceProtocolError("cost-smoke artifact differs from external commitment")
    expected_order = build_resource_protocol_order_manifest(
        capability,
        synthetic_input=synthetic_input,
        expected_capability_sha256=expected_capability_sha256,
    )
    order_manifest = artifact.get("order_manifest")
    if not isinstance(order_manifest, Mapping) or set(order_manifest) != _ORDER_MANIFEST_FIELDS:
        raise ResourceProtocolError("cost-smoke order manifest fields differ")
    if dict(order_manifest) != expected_order:
        raise ResourceProtocolError("cost-smoke order manifest differs from deterministic replay")
    if order_manifest.get("order_manifest_sha256") != _rehash_without(
        order_manifest, "order_manifest_sha256"
    ):
        raise ResourceProtocolError("cost-smoke order manifest hash differs")
    for field in ("warmup_orders", "measured_orders"):
        entries = order_manifest.get(field)
        if not isinstance(entries, list) or any(
            (
                not isinstance(entry, Mapping) or set(entry) != _ORDER_ENTRY_FIELDS
                for entry in entries
            )
        ):
            raise ResourceProtocolError("cost-smoke order entry fields differ")
    ledger = artifact.get("execution_ledger")
    if not isinstance(ledger, list):
        raise ResourceProtocolError("cost-smoke execution ledger must be a list")
    if artifact.get("execution_ledger_sha256") != canonical_sha256(ledger):
        raise ResourceProtocolError("cost-smoke execution ledger hash differs")
    expected_schedule = _expected_schedule(expected_order)
    if len(ledger) != len(expected_schedule):
        raise ResourceProtocolError("cost-smoke execution ledger is incomplete")
    expected_rounds = int(capability["expected_rounds"])
    model_hash = str(capability["model_manifest_sha256"])
    for expected_run, record in zip(expected_schedule, ledger):
        if not isinstance(record, Mapping) or set(record) != _EXECUTION_FIELDS:
            raise ResourceProtocolError("cost-smoke execution record fields differ")
        for field, expected_value in expected_run.items():
            if record.get(field) != expected_value:
                raise ResourceProtocolError("cost-smoke execution order differs from manifest")
        _candidate_zero(record.get("candidate_id"))
        receipts = record.get("round_resource_receipts")
        if isinstance(receipts, (str, bytes, Mapping)) or not isinstance(receipts, Sequence):
            raise ResourceProtocolError("cost-smoke round resource receipts are malformed")
        copied_receipts: list[dict[str, object]] = []
        for receipt in receipts:
            if not isinstance(receipt, Mapping):
                raise ResourceProtocolError("cost-smoke round resource receipt is malformed")
            copied_receipts.append(copy.deepcopy(dict(receipt)))
        try:
            rebuilt = aggregate_resource_receipts(
                copied_receipts,
                expected_method_id=str(expected_run["method_id"]),
                expected_rounds=expected_rounds,
                expected_model_manifest_sha256=model_hash,
            )
        except ResourceAccountingError as exc:
            raise ResourceProtocolError(
                "cost-smoke resource evidence fails exact reconstruction"
            ) from exc
        if record.get("resource_summary") != rebuilt:
            raise ResourceProtocolError("cost-smoke resource summary differs from receipts")


__all__ = [
    "ARTIFACT_SCHEMA",
    "CAPABILITY_SCHEMA",
    "DEFAULT_ORDER_SEED",
    "ORDER_MANIFEST_SCHEMA",
    "RESOURCE_METHODS",
    "ResourceProtocolBoundaryError",
    "ResourceProtocolError",
    "ResourceProtocolExecutionError",
    "assert_resource_protocol_result_blind",
    "build_resource_protocol_capability",
    "build_resource_protocol_order_manifest",
    "run_resource_protocol",
    "validate_resource_protocol_artifact",
    "validate_resource_protocol_capability",
]
