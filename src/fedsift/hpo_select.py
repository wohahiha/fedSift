"""Outer-local, complete-inner-OOF selection for FedSift.

Selection is deliberately downstream of the closed attempt/output catalogs.
For each candidate and frozen HPO seed, the three inner-validation prediction
payloads are concatenated *before* any nonlinear probability metric is
computed.  Candidate metrics are then arithmetic means across all frozen
seeds.  No fold-metric average and no cross-outer evidence path exists.
"""

from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import copy
import hmac
import math
from numbers import Integral, Real
from typing import Any, Mapping, Sequence
from .candidate_space import (
    MAIN_METHODS,
    CandidateSpaceError,
    candidate_by_id,
    canonical_sha256,
    require_sha256,
)
from .evaluation import EvaluationError, compute_hpo_probability_metrics
from .group_manifest import DatasetRows
from .hpo_attempt_receipt import HpoAttemptReceiptError, validate_hpo_attempt_receipt
from .hpo_capability import HpoUnitIndex, _build_capability_payload, build_hpo_unit_index
from .hpo_output import (
    HpoOutputError,
    probability_from_canonical_hex,
    validate_hpo_output_catalog,
    validate_hpo_output_catalog_commitment,
    validate_hpo_output_manifest,
)
from .hpo_plan import HpoPlanError, validate_selection_authorization


class HpoSelectionError(ValueError):
    """Raised when the frozen outer-local HPO selection cannot be reproduced."""


SCHEMA = _identity("hpo_selection_decision")
STATUS = "SEALED_OUTER_LOCAL_LEXICOGRAPHIC_DECISION"
STAGE_SCHEMA = _identity("hpo_selection_stage_commitment")
STAGE_STATUS = "COMMITTED_GLOBAL_CATALOG_SCOPE_LOCAL_SELECTION_READY"
AGGREGATION_CONTRACT: dict[str, object] = {
    "inner_fold_count": 3,
    "inner_fold_operation": "concatenate_exact_raw_native_rows_across_inner_folds_before_metrics",
    "fold_metric_averaging": "forbidden",
    "hpo_seed_operation": "arithmetic_mean_across_all_frozen_hpo_seeds",
    "communication_per_seed": "sum_rebuilt_inner_fold_communication_bytes",
    "communication_across_seeds": "arithmetic_mean",
    "outer_scope_pooling": "forbidden",
}
TIE_CONTRACT: dict[str, object] = {
    "comparison_order": [
        {"criterion": "log_loss", "direction": "minimize"},
        {"criterion": "average_precision", "direction": "maximize"},
        {"criterion": "auroc", "direction": "maximize"},
        {"criterion": "brier_score", "direction": "minimize"},
        {"criterion": "communication_bytes", "direction": "minimize"},
        {"criterion": "candidate_id", "direction": "lexicographic_min"},
    ],
    "numeric_tie": "exact_equal_canonical_binary64_no_tolerance",
    "candidate_id_order": "python_unicode_codepoint_lexicographic",
    "rounding_before_comparison": False,
}
_METRIC_NAMES = ("log_loss", "average_precision", "auroc", "brier_score")
_TOP_LEVEL_FIELDS = {
    "schema",
    "status",
    "study_id",
    "dataset_id",
    "hpo_plan_sha256",
    "hpo_closure_sha256",
    "terminal_ledger_sha256",
    "attempt_receipt_catalog_sha256",
    "output_catalog_validation_sha256",
    "output_manifest_catalog_sha256",
    "selection_scope",
    "method",
    "outer_repeat",
    "outer_fold",
    "outer_authorization_sha256",
    "frozen_hpo_seeds",
    "inner_folds",
    "aggregation_contract",
    "tie_contract",
    "candidate_evaluations",
    "selected_candidate",
    "selection_decision_manifest_sha256",
}
_CANDIDATE_EVALUATION_FIELDS = {
    "candidate_id",
    "candidate_sha256",
    "selection_eligibility",
    "seed_evaluations",
    "aggregate_metrics",
    "aggregate_metric_hex",
    "comparison_key",
    "comparison_rank",
}
_SEED_EVALUATION_FIELDS = {
    "hpo_seed",
    "inner_fold_order",
    "output_manifest_sha256",
    "concatenated_row_count",
    "concatenated_row_ids_sha256",
    "concatenated_labels_sha256",
    "concatenated_probability_hex_sha256",
    "probability_report_sha256",
    "metrics",
    "metric_hex",
    "communication_bytes",
}
_SELECTED_FIELDS = {
    "candidate_id",
    "candidate_sha256",
    "comparison_rank",
    "aggregate_metrics_sha256",
}
_STAGE_FIELDS = {
    "schema",
    "status",
    "hpo_plan_sha256",
    "hpo_closure_sha256",
    "terminal_ledger_sha256",
    "attempt_receipt_catalog_sha256",
    "receipt_catalog_validation_sha256",
    "output_catalog_validation_sha256",
    "output_manifest_catalog_sha256",
    "scope_catalogs",
    "complete_attempt_count",
    "selection_scope",
    "stage_commitment_sha256",
}
_STAGE_SCOPE_FIELDS = {
    "method",
    "outer_repeat",
    "outer_fold",
    "complete_attempt_count",
    "attempt_receipt_scope_sha256",
    "output_manifest_scope_sha256",
}


def _artifact_hash(value: Mapping[str, object], field: str) -> str:
    payload = copy.deepcopy(dict(value))
    payload.pop(field, None)
    return canonical_sha256(payload)


def _hash(value: object, field: str) -> str:
    try:
        return require_sha256(value, field)
    except CandidateSpaceError as exc:
        raise HpoSelectionError(str(exc)) from exc


def _exact_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise HpoSelectionError(f"{field} must be an exact integer")
    result = int(value)
    if result < minimum:
        raise HpoSelectionError(f"{field} must be >= {minimum}")
    return result


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise HpoSelectionError(f"{field} must be a non-empty canonical string")
    return value


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise HpoSelectionError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise HpoSelectionError(f"{field} must be finite")
    return 0.0 if result == 0.0 else result


def _outer_authorization(
    closure: Mapping[str, object], outer_repeat: int, outer_fold: int
) -> Mapping[str, object]:
    rows = closure.get("outer_authorizations")
    if not isinstance(rows, list):
        raise HpoSelectionError("HPO closure has no outer authorizations")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("outer_repeat") == outer_repeat
        and (row.get("outer_fold") == outer_fold)
    ]
    if len(matches) != 1:
        raise HpoSelectionError("outer authorization is not unique")
    return matches[0]


def _frozen_design(hpo_plan: Mapping[str, object]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    equal = hpo_plan.get("equal_budget_contract")
    nested = hpo_plan.get("nested_design_binding")
    if not isinstance(equal, Mapping) or not isinstance(nested, Mapping):
        raise HpoSelectionError("HPO plan lacks frozen design bindings")
    raw_seeds = equal.get("hpo_seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise HpoSelectionError("frozen HPO seeds are missing")
    seeds = tuple(
        (_exact_int(value, f"hpo_seeds[{index}]") for (index, value) in enumerate(raw_seeds))
    )
    if len(seeds) != len(set(seeds)):
        raise HpoSelectionError("frozen HPO seeds are duplicated")
    inner_count = _exact_int(nested.get("inner_folds"), "inner_folds", minimum=2)
    if inner_count != 3:
        raise HpoSelectionError(_identity("n1_selection_requires_exactly_three_inner_folds"))
    return (seeds, tuple(range(inner_count)))


def _complete_output_by_unit(
    attempt_receipts: Sequence[Mapping[str, object]],
    output_manifests: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    manifests_by_hash: dict[str, Mapping[str, object]] = {}
    for manifest in output_manifests:
        if not isinstance(manifest, Mapping):
            raise HpoSelectionError("output catalog entry is not a mapping")
        manifest_hash = str(manifest.get("manifest_sha256"))
        if manifest_hash in manifests_by_hash:
            raise HpoSelectionError("output catalog repeats a manifest hash")
        manifests_by_hash[manifest_hash] = manifest
    result: dict[str, Mapping[str, object]] = {}
    for receipt in attempt_receipts:
        if not isinstance(receipt, Mapping) or receipt.get("outcome") != "complete":
            continue
        output_hash = str(receipt.get("exclusive_output_artifact_manifest_sha256"))
        manifest = manifests_by_hash.get(output_hash)
        if manifest is None:
            raise HpoSelectionError("completed unit has no validated output manifest")
        unit_id = str(receipt.get("unit_id"))
        if unit_id in result:
            raise HpoSelectionError("one HPO unit has multiple completed outputs")
        result[unit_id] = manifest
    return result


def _candidate_units(
    hpo_plan: Mapping[str, object],
    *,
    method: str,
    outer_repeat: int,
    outer_fold: int,
    candidate_id: str,
) -> list[Mapping[str, object]]:
    units = hpo_plan.get("units")
    if not isinstance(units, list):
        raise HpoSelectionError("HPO plan unit inventory is missing")
    return [
        unit
        for unit in units
        if isinstance(unit, Mapping)
        and unit.get("method") == method
        and (unit.get("outer_repeat") == outer_repeat)
        and (unit.get("outer_fold") == outer_fold)
        and (unit.get("candidate_id") == candidate_id)
    ]


def _prediction_payload(
    manifest: Mapping[str, object],
) -> tuple[list[int], list[int], list[float], list[str]]:
    predictions = manifest.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise HpoSelectionError("validated output has no predictions")
    row_ids: list[int] = []
    labels: list[int] = []
    probabilities: list[float] = []
    probability_hex: list[str] = []
    for row in predictions:
        if not isinstance(row, Mapping):
            raise HpoSelectionError("validated prediction row is malformed")
        token = row.get("row_token")
        if (
            not isinstance(token, str)
            or len(token) != 19
            or (not token.startswith("row_id:"))
            or (not token[7:].isdigit())
        ):
            raise HpoSelectionError("validated prediction row token is malformed")
        row_id = int(token[7:])
        label = _exact_int(row.get("label"), "prediction label")
        if label not in (0, 1):
            raise HpoSelectionError("prediction label is not binary")
        encoded = row.get("probability_hex")
        probability = probability_from_canonical_hex(encoded)
        row_ids.append(row_id)
        labels.append(label)
        probabilities.append(probability)
        probability_hex.append(str(encoded))
    return (row_ids, labels, probabilities, probability_hex)


def _seed_evaluation(
    units: Sequence[Mapping[str, object]],
    output_by_unit: Mapping[str, Mapping[str, object]],
    *,
    hpo_seed: int,
    inner_folds: tuple[int, ...],
) -> dict[str, object]:
    by_fold = {int(unit["inner_fold"]): unit for unit in units if int(unit["hpo_seed"]) == hpo_seed}
    if set(by_fold) != set(inner_folds) or len(by_fold) != len(inner_folds):
        raise HpoSelectionError("candidate lacks one exact unit per inner fold and seed")
    row_ids: list[int] = []
    labels: list[int] = []
    probabilities: list[float] = []
    probability_hex: list[str] = []
    output_hashes: list[str] = []
    communication_bytes = 0
    for inner_fold in inner_folds:
        unit = by_fold[inner_fold]
        unit_id = str(unit["unit_id"])
        manifest = output_by_unit.get(unit_id)
        if manifest is None:
            raise HpoSelectionError("eligible candidate has no completed fold output")
        binding = manifest.get("capability_binding")
        if (
            not isinstance(binding, Mapping)
            or binding.get("inner_fold") != inner_fold
            or binding.get("hpo_seed") != hpo_seed
            or (binding.get("outer_repeat") != unit.get("outer_repeat"))
            or (binding.get("outer_fold") != unit.get("outer_fold"))
            or (binding.get("candidate_id") != unit.get("candidate_id"))
            or (binding.get("method") != unit.get("method"))
        ):
            raise HpoSelectionError("output crossed a candidate/seed/fold/outer scope")
        fold_rows, fold_labels, fold_probabilities, fold_hex = _prediction_payload(manifest)
        row_ids.extend(fold_rows)
        labels.extend(fold_labels)
        probabilities.extend(fold_probabilities)
        probability_hex.extend(fold_hex)
        output_hashes.append(str(manifest["manifest_sha256"]))
        resource = manifest.get("resource_evidence")
        if not isinstance(resource, Mapping):
            raise HpoSelectionError("validated output lacks resource evidence")
        communication_bytes += _exact_int(
            resource.get("communication_bytes"), "communication_bytes"
        )
    if len(row_ids) != len(set(row_ids)):
        raise HpoSelectionError("inner folds overlap in row membership")
    try:
        report = compute_hpo_probability_metrics(row_ids, labels, probabilities)
    except EvaluationError as exc:
        raise HpoSelectionError(
            "concatenated inner-validation evidence is not a valid two-class metric unit"
        ) from exc
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != {
        "average_precision",
        "auroc",
        "brier_score",
        "log_loss",
    }:
        raise HpoSelectionError("probability report metric schema differs")
    canonical_metrics = {name: _finite_float(metrics[name], name) for name in _METRIC_NAMES}
    return {
        "hpo_seed": hpo_seed,
        "inner_fold_order": list(inner_folds),
        "output_manifest_sha256": output_hashes,
        "concatenated_row_count": len(row_ids),
        "concatenated_row_ids_sha256": canonical_sha256(row_ids),
        "concatenated_labels_sha256": canonical_sha256(labels),
        "concatenated_probability_hex_sha256": canonical_sha256(probability_hex),
        "probability_report_sha256": report["report_sha256"],
        "metrics": canonical_metrics,
        "metric_hex": {name: value.hex() for (name, value) in canonical_metrics.items()},
        "communication_bytes": communication_bytes,
    }


def _mean(values: Sequence[float], field: str) -> float:
    if not values:
        raise HpoSelectionError(f"{field} has no frozen-seed values")
    result = math.fsum(values) / len(values)
    return _finite_float(result, field)


def _candidate_evaluation(
    hpo_plan: Mapping[str, object],
    candidate_space: Mapping[str, object],
    output_by_unit: Mapping[str, Mapping[str, object]],
    *,
    method: str,
    outer_repeat: int,
    outer_fold: int,
    candidate_id: str,
    hpo_seeds: tuple[int, ...],
    inner_folds: tuple[int, ...],
    candidate_units: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    units = (
        list(candidate_units)
        if candidate_units is not None
        else _candidate_units(
            hpo_plan,
            method=method,
            outer_repeat=outer_repeat,
            outer_fold=outer_fold,
            candidate_id=candidate_id,
        )
    )
    expected_count = len(hpo_seeds) * len(inner_folds)
    if len(units) != expected_count:
        raise HpoSelectionError("eligible candidate unit inventory is incomplete")
    identities = {(int(unit["hpo_seed"]), int(unit["inner_fold"])) for unit in units}
    if identities != {(seed, inner_fold) for seed in hpo_seeds for inner_fold in inner_folds}:
        raise HpoSelectionError("eligible candidate unit identities are duplicated or missing")
    candidate_hashes = {str(unit["candidate_sha256"]) for unit in units}
    if len(candidate_hashes) != 1:
        raise HpoSelectionError("candidate hash changes inside one outer scope")
    try:
        roster_candidate = candidate_by_id(candidate_space, method, candidate_id)
    except CandidateSpaceError as exc:
        raise HpoSelectionError(str(exc)) from exc
    candidate_hash = next(iter(candidate_hashes))
    if roster_candidate.get("candidate_sha256") != candidate_hash:
        raise HpoSelectionError("candidate hash differs from frozen roster")
    seed_evaluations = [
        _seed_evaluation(units, output_by_unit, hpo_seed=seed, inner_folds=inner_folds)
        for seed in hpo_seeds
    ]
    aggregate_metrics = {
        name: _mean([float(seed["metrics"][name]) for seed in seed_evaluations], name)
        for name in _METRIC_NAMES
    }
    aggregate_metrics["communication_bytes"] = _mean(
        [float(seed["communication_bytes"]) for seed in seed_evaluations], "communication_bytes"
    )
    comparison_key = [
        {
            "criterion": name,
            "direction": "maximize" if name in {"average_precision", "auroc"} else "minimize",
            "value_hex": aggregate_metrics[name].hex(),
        }
        for name in ("log_loss", "average_precision", "auroc", "brier_score", "communication_bytes")
    ]
    comparison_key.append(
        {"criterion": "candidate_id", "direction": "lexicographic_min", "value": candidate_id}
    )
    return {
        "candidate_id": candidate_id,
        "candidate_sha256": candidate_hash,
        "selection_eligibility": "fully_closed_in_this_outer_scope",
        "seed_evaluations": seed_evaluations,
        "aggregate_metrics": aggregate_metrics,
        "aggregate_metric_hex": {name: value.hex() for (name, value) in aggregate_metrics.items()},
        "comparison_key": comparison_key,
        "comparison_rank": 0,
    }


def _sort_key(evaluation: Mapping[str, object]) -> tuple[object, ...]:
    metrics = evaluation["aggregate_metrics"]
    assert isinstance(metrics, Mapping)
    return (
        float(metrics["log_loss"]),
        -float(metrics["average_precision"]),
        -float(metrics["auroc"]),
        float(metrics["brier_score"]),
        float(metrics["communication_bytes"]),
        str(evaluation["candidate_id"]),
    )


def _validate_decision_shape(decision: Mapping[str, object]) -> None:
    if not isinstance(decision, Mapping) or set(decision) != _TOP_LEVEL_FIELDS:
        raise HpoSelectionError("selection decision fields differ from exact schema")
    if decision.get("schema") != SCHEMA or decision.get("status") != STATUS:
        raise HpoSelectionError("selection decision schema or status differs")
    for field in ("study_id", "dataset_id", "method"):
        _identifier(decision.get(field), field)
    if decision.get("method") not in MAIN_METHODS:
        raise HpoSelectionError("selection decision method is not in the main roster")
    _exact_int(decision.get("outer_repeat"), "outer_repeat")
    _exact_int(decision.get("outer_fold"), "outer_fold")
    for field in (
        "hpo_plan_sha256",
        "hpo_closure_sha256",
        "terminal_ledger_sha256",
        "attempt_receipt_catalog_sha256",
        "output_catalog_validation_sha256",
        "output_manifest_catalog_sha256",
        "outer_authorization_sha256",
        "selection_decision_manifest_sha256",
    ):
        _hash(decision.get(field), field)
    if decision.get("selection_scope") != "within_outer_repeat_outer_fold_only":
        raise HpoSelectionError("selection decision scope differs")
    if decision.get("aggregation_contract") != AGGREGATION_CONTRACT:
        raise HpoSelectionError("selection aggregation contract differs")
    if decision.get("tie_contract") != TIE_CONTRACT:
        raise HpoSelectionError("selection tie contract differs")
    seeds = decision.get("frozen_hpo_seeds")
    folds = decision.get("inner_folds")
    if not isinstance(seeds, list) or not seeds or len(seeds) != len(set(seeds)):
        raise HpoSelectionError("selection frozen seeds are invalid")
    if folds != [0, 1, 2]:
        raise HpoSelectionError("selection inner folds differ")
    evaluations = decision.get("candidate_evaluations")
    if not isinstance(evaluations, list) or not evaluations:
        raise HpoSelectionError("selection candidate evaluations are missing")
    for evaluation in evaluations:
        if not isinstance(evaluation, Mapping) or set(evaluation) != _CANDIDATE_EVALUATION_FIELDS:
            raise HpoSelectionError("candidate evaluation fields differ")
        seed_evaluations = evaluation.get("seed_evaluations")
        if not isinstance(seed_evaluations, list) or len(seed_evaluations) != len(seeds):
            raise HpoSelectionError("candidate seed evaluations are incomplete")
        for seed in seed_evaluations:
            if not isinstance(seed, Mapping) or set(seed) != _SEED_EVALUATION_FIELDS:
                raise HpoSelectionError("seed evaluation fields differ")
    selected = decision.get("selected_candidate")
    if not isinstance(selected, Mapping) or set(selected) != _SELECTED_FIELDS:
        raise HpoSelectionError("selected candidate fields differ")
    if selected.get("comparison_rank") != 1:
        raise HpoSelectionError("selected candidate must have rank one")
    stored = str(decision["selection_decision_manifest_sha256"])
    if not hmac.compare_digest(
        stored, _artifact_hash(decision, "selection_decision_manifest_sha256")
    ):
        raise HpoSelectionError("selection decision canonical hash differs")


def _selection_inventory(
    hpo_plan: Mapping[str, object],
    closure: Mapping[str, object],
    *,
    method: str,
    outer_repeat: int,
    outer_fold: int,
) -> tuple[int, int, Mapping[str, object], list[str], tuple[int, ...], tuple[int, ...]]:
    if method not in MAIN_METHODS:
        raise HpoSelectionError("selection method is not in the frozen main roster")
    repeat = _exact_int(outer_repeat, "outer_repeat")
    fold = _exact_int(outer_fold, "outer_fold")
    if closure.get("selection_scope") != "within_outer_repeat_outer_fold_only":
        raise HpoSelectionError("HPO closure does not authorize outer-local selection")
    outer = _outer_authorization(closure, repeat, fold)
    methods = outer.get("methods")
    if not isinstance(methods, Mapping):
        raise HpoSelectionError("outer authorization has no method inventories")
    inventory = methods.get(method)
    if not isinstance(inventory, Mapping):
        raise HpoSelectionError("method has no outer-local authorization")
    eligible = inventory.get("eligible_candidate_ids")
    if not isinstance(eligible, list) or not eligible:
        raise HpoSelectionError("method has no eligible candidates in this outer scope")
    if any((not isinstance(value, str) or not value for value in eligible)):
        raise HpoSelectionError("eligible candidate inventory is malformed")
    if len(eligible) != len(set(eligible)):
        raise HpoSelectionError("eligible candidate inventory is duplicated")
    hpo_seeds, inner_folds = _frozen_design(hpo_plan)
    return (repeat, fold, outer, eligible, hpo_seeds, inner_folds)


def _assemble_selection_decision(
    hpo_plan: Mapping[str, object],
    closure: Mapping[str, object],
    candidate_space: Mapping[str, object],
    output_catalog: Mapping[str, object],
    output_by_unit: Mapping[str, Mapping[str, object]],
    *,
    method: str,
    repeat: int,
    fold: int,
    outer: Mapping[str, object],
    eligible: Sequence[str],
    hpo_seeds: tuple[int, ...],
    inner_folds: tuple[int, ...],
    scope_units_by_candidate: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
) -> dict[str, object]:
    evaluations = [
        _candidate_evaluation(
            hpo_plan,
            candidate_space,
            output_by_unit,
            method=method,
            outer_repeat=repeat,
            outer_fold=fold,
            candidate_id=candidate_id,
            hpo_seeds=hpo_seeds,
            inner_folds=inner_folds,
            candidate_units=(
                None
                if scope_units_by_candidate is None
                else scope_units_by_candidate.get(candidate_id)
            ),
        )
        for candidate_id in eligible
    ]
    ranked = sorted(evaluations, key=_sort_key)
    rank_by_id = {
        str(evaluation["candidate_id"]): rank for (rank, evaluation) in enumerate(ranked, start=1)
    }
    for evaluation in evaluations:
        evaluation["comparison_rank"] = rank_by_id[str(evaluation["candidate_id"])]
    evaluations.sort(key=lambda value: str(value["candidate_id"]))
    winner = ranked[0]
    selected_metrics = winner["aggregate_metrics"]
    assert isinstance(selected_metrics, Mapping)
    decision: dict[str, object] = {
        "schema": SCHEMA,
        "status": STATUS,
        "study_id": hpo_plan["study_id"],
        "dataset_id": hpo_plan["dataset_id"],
        "hpo_plan_sha256": hpo_plan["hpo_plan_sha256"],
        "hpo_closure_sha256": closure["closure_sha256"],
        "terminal_ledger_sha256": closure["ledger_sha256"],
        "attempt_receipt_catalog_sha256": closure["attempt_receipt_catalog_sha256"],
        "output_catalog_validation_sha256": output_catalog["catalog_validation_sha256"],
        "output_manifest_catalog_sha256": output_catalog["output_manifest_catalog_sha256"],
        "selection_scope": "within_outer_repeat_outer_fold_only",
        "method": method,
        "outer_repeat": repeat,
        "outer_fold": fold,
        "outer_authorization_sha256": outer["outer_authorization_sha256"],
        "frozen_hpo_seeds": list(hpo_seeds),
        "inner_folds": list(inner_folds),
        "aggregation_contract": copy.deepcopy(AGGREGATION_CONTRACT),
        "tie_contract": copy.deepcopy(TIE_CONTRACT),
        "candidate_evaluations": evaluations,
        "selected_candidate": {
            "candidate_id": winner["candidate_id"],
            "candidate_sha256": winner["candidate_sha256"],
            "comparison_rank": 1,
            "aggregate_metrics_sha256": canonical_sha256(selected_metrics),
        },
    }
    decision["selection_decision_manifest_sha256"] = _artifact_hash(
        decision, "selection_decision_manifest_sha256"
    )
    _validate_decision_shape(decision)
    return decision


def build_hpo_selection_decision(
    hpo_plan: Mapping[str, object],
    closure: Mapping[str, object],
    ledger_entries: Sequence[Mapping[str, object]],
    attempt_receipts: Sequence[Mapping[str, object]],
    output_manifests: Sequence[Mapping[str, object]],
    *,
    candidate_space: Mapping[str, object],
    nested_plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
    authoritative_rows: DatasetRows | Mapping[int, int],
    method: str,
    outer_repeat: int,
    outer_fold: int,
    authoritative_dataset_sha256: str | None = None,
) -> dict[str, object]:
    """Apply the frozen lexicographic rule inside exactly one outer scope."""
    try:
        validate_selection_authorization(
            hpo_plan,
            closure,
            ledger_entries,
            attempt_receipts=attempt_receipts,
            candidate_space=candidate_space,
            nested_plan=nested_plan,
            group_manifest=group_manifest,
        )
        output_catalog = validate_hpo_output_catalog(
            hpo_plan,
            ledger_entries,
            attempt_receipts,
            output_manifests,
            candidate_space=candidate_space,
            nested_plan=nested_plan,
            group_manifest=group_manifest,
            authoritative_rows=authoritative_rows,
            authoritative_dataset_sha256=authoritative_dataset_sha256,
        )
    except (HpoPlanError, HpoOutputError) as exc:
        raise HpoSelectionError(str(exc)) from exc
    repeat, fold, outer, eligible, hpo_seeds, inner_folds = _selection_inventory(
        hpo_plan, closure, method=method, outer_repeat=outer_repeat, outer_fold=outer_fold
    )
    output_by_unit = _complete_output_by_unit(attempt_receipts, output_manifests)
    return _assemble_selection_decision(
        hpo_plan,
        closure,
        candidate_space,
        output_catalog,
        output_by_unit,
        method=method,
        repeat=repeat,
        fold=fold,
        outer=outer,
        eligible=eligible,
        hpo_seeds=hpo_seeds,
        inner_folds=inner_folds,
    )


def build_hpo_selection_decision_from_catalog_commitment(
    hpo_plan: Mapping[str, object],
    closure: Mapping[str, object],
    ledger_entries: Sequence[Mapping[str, object]],
    attempt_receipts: Sequence[Mapping[str, object]],
    output_catalog_validation: Mapping[str, object],
    scope_output_manifests: Sequence[Mapping[str, object]],
    *,
    candidate_space: Mapping[str, object],
    nested_plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
    authoritative_rows: DatasetRows | Mapping[int, int],
    expected_catalog_validation_sha256: str,
    method: str,
    outer_repeat: int,
    outer_fold: int,
    authoritative_dataset_sha256: str | None = None,
) -> dict[str, object]:
    """Select from one outer/method scope after a streamed global close.

    Only the eligible manifests for this exact scope are retained in memory.
    The global catalog remains bound by its separately committed validation
    hash and the complete ledger/attempt receipt catalog.
    """
    try:
        validate_selection_authorization(
            hpo_plan,
            closure,
            ledger_entries,
            attempt_receipts=attempt_receipts,
            candidate_space=candidate_space,
            nested_plan=nested_plan,
            group_manifest=group_manifest,
        )
        output_catalog = validate_hpo_output_catalog_commitment(
            output_catalog_validation,
            hpo_plan,
            ledger_entries,
            attempt_receipts,
            candidate_space=candidate_space,
            nested_plan=nested_plan,
            group_manifest=group_manifest,
            expected_catalog_validation_sha256=expected_catalog_validation_sha256,
        )
    except (HpoPlanError, HpoOutputError) as exc:
        raise HpoSelectionError(str(exc)) from exc
    repeat, fold, outer, eligible, hpo_seeds, inner_folds = _selection_inventory(
        hpo_plan, closure, method=method, outer_repeat=outer_repeat, outer_fold=outer_fold
    )
    plan_units = hpo_plan.get("units")
    if not isinstance(plan_units, list):
        raise HpoSelectionError("HPO plan unit inventory is missing")
    eligible_set = set(eligible)
    expected_units = [
        unit
        for unit in plan_units
        if isinstance(unit, Mapping)
        and unit.get("method") == method
        and (unit.get("outer_repeat") == repeat)
        and (unit.get("outer_fold") == fold)
        and (unit.get("candidate_id") in eligible_set)
    ]
    expected_unit_ids = {str(unit["unit_id"]) for unit in expected_units}
    expected_count = len(eligible) * len(hpo_seeds) * len(inner_folds)
    if len(expected_units) != expected_count or len(expected_unit_ids) != expected_count:
        raise HpoSelectionError("eligible scope unit inventory is incomplete or duplicated")
    receipt_by_unit = {
        str(receipt["unit_id"]): receipt
        for receipt in attempt_receipts
        if isinstance(receipt, Mapping)
        and receipt.get("outcome") == "complete"
        and (str(receipt.get("unit_id")) in expected_unit_ids)
    }
    if set(receipt_by_unit) != expected_unit_ids:
        raise HpoSelectionError("eligible scope lacks one complete receipt per unit")
    if isinstance(scope_output_manifests, (str, bytes)) or not isinstance(
        scope_output_manifests, Sequence
    ):
        raise HpoSelectionError("scope output manifests must be a sequence")
    output_by_unit: dict[str, Mapping[str, object]] = {}
    unit_index = build_hpo_unit_index(hpo_plan)
    try:
        for manifest in scope_output_manifests:
            if not isinstance(manifest, Mapping):
                raise HpoOutputError("scope output entry is not a mapping")
            unit_id = str(manifest.get("unit_id"))
            if unit_id not in expected_unit_ids or unit_id in output_by_unit:
                raise HpoOutputError("scope output is duplicated or crosses its authorization")
            receipt = receipt_by_unit[unit_id]
            if manifest.get("manifest_sha256") != receipt.get(
                "exclusive_output_artifact_manifest_sha256"
            ):
                raise HpoOutputError("scope output differs from its complete attempt receipt")
            capability = _build_capability_payload(
                hpo_plan,
                candidate_space,
                nested_plan,
                group_manifest,
                unit_id,
                unit_index=unit_index,
            )
            validate_hpo_output_manifest(
                manifest,
                capability,
                expected_capability_sha256=str(capability["capability_sha256"]),
                authoritative_rows=authoritative_rows,
                authoritative_dataset_sha256=authoritative_dataset_sha256,
                attempt_receipt=receipt,
            )
            output_by_unit[unit_id] = manifest
    except (HpoAttemptReceiptError, HpoOutputError) as exc:
        raise HpoSelectionError(str(exc)) from exc
    if set(output_by_unit) != expected_unit_ids:
        raise HpoSelectionError("scope output catalog is missing an eligible unit")
    scope_units_by_candidate = {
        candidate_id: [unit for unit in expected_units if unit.get("candidate_id") == candidate_id]
        for candidate_id in eligible
    }
    return _assemble_selection_decision(
        hpo_plan,
        closure,
        candidate_space,
        output_catalog,
        output_by_unit,
        method=method,
        repeat=repeat,
        fold=fold,
        outer=outer,
        eligible=eligible,
        hpo_seeds=hpo_seeds,
        inner_folds=inner_folds,
        scope_units_by_candidate=scope_units_by_candidate,
    )


def build_hpo_selection_stage_commitment(
    hpo_plan: Mapping[str, object],
    closure: Mapping[str, object],
    ledger_entries: Sequence[Mapping[str, object]],
    attempt_receipts: Sequence[Mapping[str, object]],
    output_catalog_validation: Mapping[str, object],
    *,
    candidate_space: Mapping[str, object],
    nested_plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
    expected_catalog_validation_sha256: str,
) -> dict[str, object]:
    """Validate global evidence once and commit it for repeated local selection."""
    try:
        validate_selection_authorization(
            hpo_plan,
            closure,
            ledger_entries,
            attempt_receipts=attempt_receipts,
            candidate_space=candidate_space,
            nested_plan=nested_plan,
            group_manifest=group_manifest,
        )
        output_catalog = validate_hpo_output_catalog_commitment(
            output_catalog_validation,
            hpo_plan,
            ledger_entries,
            attempt_receipts,
            candidate_space=candidate_space,
            nested_plan=nested_plan,
            group_manifest=group_manifest,
            expected_catalog_validation_sha256=expected_catalog_validation_sha256,
        )
    except (HpoPlanError, HpoOutputError) as exc:
        raise HpoSelectionError(str(exc)) from exc
    stage: dict[str, object] = {
        "schema": STAGE_SCHEMA,
        "status": STAGE_STATUS,
        "hpo_plan_sha256": hpo_plan["hpo_plan_sha256"],
        "hpo_closure_sha256": closure["closure_sha256"],
        "terminal_ledger_sha256": closure["ledger_sha256"],
        "attempt_receipt_catalog_sha256": closure["attempt_receipt_catalog_sha256"],
        "receipt_catalog_validation_sha256": closure["receipt_catalog_validation_sha256"],
        "output_catalog_validation_sha256": output_catalog["catalog_validation_sha256"],
        "output_manifest_catalog_sha256": output_catalog["output_manifest_catalog_sha256"],
        "scope_catalogs": copy.deepcopy(output_catalog["scope_catalogs"]),
        "complete_attempt_count": output_catalog["complete_attempt_count"],
        "selection_scope": "within_outer_repeat_outer_fold_only",
    }
    stage["stage_commitment_sha256"] = _artifact_hash(stage, "stage_commitment_sha256")
    return stage


def _validate_hpo_selection_stage_commitment(
    stage: Mapping[str, object],
    hpo_plan: Mapping[str, object],
    closure: Mapping[str, object],
    *,
    expected_stage_commitment_sha256: str,
) -> None:
    expected_hash = _hash(expected_stage_commitment_sha256, "expected_stage_commitment_sha256")
    if not isinstance(stage, Mapping) or set(stage) != _STAGE_FIELDS:
        raise HpoSelectionError("selection stage commitment fields differ")
    if (
        stage.get("schema") != STAGE_SCHEMA
        or stage.get("status") != STAGE_STATUS
        or stage.get("selection_scope") != "within_outer_repeat_outer_fold_only"
        or (stage.get("hpo_plan_sha256") != hpo_plan.get("hpo_plan_sha256"))
        or (stage.get("hpo_closure_sha256") != closure.get("closure_sha256"))
        or (stage.get("terminal_ledger_sha256") != closure.get("ledger_sha256"))
        or (
            stage.get("attempt_receipt_catalog_sha256")
            != closure.get("attempt_receipt_catalog_sha256")
        )
        or (
            stage.get("receipt_catalog_validation_sha256")
            != closure.get("receipt_catalog_validation_sha256")
        )
    ):
        raise HpoSelectionError("selection stage commitment identity differs")
    for field in (
        "output_catalog_validation_sha256",
        "output_manifest_catalog_sha256",
        "stage_commitment_sha256",
    ):
        _hash(stage.get(field), field)
    _exact_int(stage.get("complete_attempt_count"), "complete_attempt_count", minimum=1)
    scope_catalogs = stage.get("scope_catalogs")
    if not isinstance(scope_catalogs, list):
        raise HpoSelectionError("selection stage scope catalogs must be a list")
    plan_units = hpo_plan.get("units")
    if not isinstance(plan_units, list):
        raise HpoSelectionError("HPO plan unit inventory is missing")
    expected_scope_counts: dict[tuple[str, int, int], int] = {}
    for unit in plan_units:
        if not isinstance(unit, Mapping):
            raise HpoSelectionError("HPO plan unit inventory contains a non-mapping")
        scope = (
            str(unit.get("method")),
            _exact_int(unit.get("outer_repeat"), "outer_repeat", minimum=0),
            _exact_int(unit.get("outer_fold"), "outer_fold", minimum=0),
        )
        expected_scope_counts[scope] = expected_scope_counts.get(scope, 0) + 1
    observed_scopes: dict[tuple[str, int, int], Mapping[str, object]] = {}
    for catalog in scope_catalogs:
        if not isinstance(catalog, Mapping) or set(catalog) != _STAGE_SCOPE_FIELDS:
            raise HpoSelectionError("selection stage scope catalog fields differ")
        method = catalog.get("method")
        if not isinstance(method, str) or not method:
            raise HpoSelectionError("selection stage scope method is invalid")
        scope = (
            method,
            _exact_int(catalog.get("outer_repeat"), "outer_repeat", minimum=0),
            _exact_int(catalog.get("outer_fold"), "outer_fold", minimum=0),
        )
        count = _exact_int(
            catalog.get("complete_attempt_count"), "complete_attempt_count", minimum=1
        )
        if scope in observed_scopes or expected_scope_counts.get(scope) != count:
            raise HpoSelectionError("selection stage scope inventory differs")
        _hash(catalog.get("attempt_receipt_scope_sha256"), "attempt_receipt_scope_sha256")
        _hash(catalog.get("output_manifest_scope_sha256"), "output_manifest_scope_sha256")
        observed_scopes[scope] = catalog
    if set(observed_scopes) != set(expected_scope_counts):
        raise HpoSelectionError("selection stage scope inventory is incomplete")
    if sum(expected_scope_counts.values()) != stage.get("complete_attempt_count"):
        raise HpoSelectionError("selection stage complete attempt count differs")
    stored = str(stage["stage_commitment_sha256"])
    if not hmac.compare_digest(stored, expected_hash) or not hmac.compare_digest(
        stored, _artifact_hash(stage, "stage_commitment_sha256")
    ):
        raise HpoSelectionError("selection stage commitment hash differs")


def build_hpo_selection_decision_from_stage_commitment(
    hpo_plan: Mapping[str, object],
    closure: Mapping[str, object],
    stage: Mapping[str, object],
    scope_attempt_receipts: Sequence[Mapping[str, object]],
    scope_output_manifests: Sequence[Mapping[str, object]],
    *,
    candidate_space: Mapping[str, object],
    nested_plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
    authoritative_rows: DatasetRows | Mapping[int, int],
    expected_stage_commitment_sha256: str,
    method: str,
    outer_repeat: int,
    outer_fold: int,
    authoritative_dataset_sha256: str | None = None,
    unit_index: HpoUnitIndex | None = None,
) -> dict[str, object]:
    """Build one decision after global evidence has crossed a committed gate."""
    _validate_hpo_selection_stage_commitment(
        stage, hpo_plan, closure, expected_stage_commitment_sha256=expected_stage_commitment_sha256
    )
    repeat, fold, outer, eligible, hpo_seeds, inner_folds = _selection_inventory(
        hpo_plan, closure, method=method, outer_repeat=outer_repeat, outer_fold=outer_fold
    )
    plan_units = hpo_plan.get("units")
    if not isinstance(plan_units, list):
        raise HpoSelectionError("HPO plan unit inventory is missing")
    eligible_set = set(eligible)
    expected_units = [
        unit
        for unit in plan_units
        if isinstance(unit, Mapping)
        and unit.get("method") == method
        and (unit.get("outer_repeat") == repeat)
        and (unit.get("outer_fold") == fold)
        and (unit.get("candidate_id") in eligible_set)
    ]
    expected_count = len(eligible) * len(hpo_seeds) * len(inner_folds)
    expected_unit_ids = {str(unit["unit_id"]) for unit in expected_units}
    if len(expected_units) != expected_count or len(expected_unit_ids) != expected_count:
        raise HpoSelectionError("eligible scope unit inventory is incomplete or duplicated")
    if isinstance(scope_attempt_receipts, (str, bytes)) or not isinstance(
        scope_attempt_receipts, Sequence
    ):
        raise HpoSelectionError("scope attempt receipts must be a sequence")
    receipt_by_unit: dict[str, Mapping[str, object]] = {}
    try:
        for receipt in scope_attempt_receipts:
            if not isinstance(receipt, Mapping):
                raise HpoOutputError("scope attempt receipt is not a mapping")
            validate_hpo_attempt_receipt(receipt)
            unit_id = str(receipt.get("unit_id"))
            if (
                receipt.get("outcome") != "complete"
                or unit_id not in expected_unit_ids
                or unit_id in receipt_by_unit
            ):
                raise HpoOutputError(
                    "scope attempt receipt is incomplete, duplicated, or unauthorized"
                )
            receipt_by_unit[unit_id] = receipt
    except (HpoAttemptReceiptError, HpoOutputError) as exc:
        raise HpoSelectionError(str(exc)) from exc
    if set(receipt_by_unit) != expected_unit_ids:
        raise HpoSelectionError("scope lacks one complete receipt per eligible unit")
    scope_commitments = [
        catalog
        for catalog in stage["scope_catalogs"]
        if catalog.get("method") == method
        and catalog.get("outer_repeat") == repeat
        and (catalog.get("outer_fold") == fold)
    ]
    if len(scope_commitments) != 1:
        raise HpoSelectionError("selection stage lacks the exact requested scope")
    scope_commitment = scope_commitments[0]
    normalized_receipts = sorted(
        [
            {
                "unit_id": receipt["unit_id"],
                "attempt_index": receipt["attempt_index"],
                "attempt_receipt_sha256": receipt["attempt_receipt_sha256"],
            }
            for receipt in receipt_by_unit.values()
        ],
        key=lambda row: (
            str(row["unit_id"]),
            int(row["attempt_index"]),
            str(row["attempt_receipt_sha256"]),
        ),
    )
    if (
        scope_commitment["complete_attempt_count"] != expected_count
        or canonical_sha256(normalized_receipts) != scope_commitment["attempt_receipt_scope_sha256"]
    ):
        raise HpoSelectionError("scope receipts differ from the global commitment")
    if isinstance(scope_output_manifests, (str, bytes)) or not isinstance(
        scope_output_manifests, Sequence
    ):
        raise HpoSelectionError("scope output manifests must be a sequence")
    bound_index = unit_index if unit_index is not None else build_hpo_unit_index(hpo_plan)
    output_by_unit: dict[str, Mapping[str, object]] = {}
    try:
        for manifest in scope_output_manifests:
            if not isinstance(manifest, Mapping):
                raise HpoOutputError("scope output entry is not a mapping")
            unit_id = str(manifest.get("unit_id"))
            if unit_id not in expected_unit_ids or unit_id in output_by_unit:
                raise HpoOutputError("scope output is duplicated or crosses its authorization")
            receipt = receipt_by_unit[unit_id]
            if manifest.get("manifest_sha256") != receipt.get(
                "exclusive_output_artifact_manifest_sha256"
            ):
                raise HpoOutputError("scope output differs from its complete attempt receipt")
            capability = _build_capability_payload(
                hpo_plan,
                candidate_space,
                nested_plan,
                group_manifest,
                unit_id,
                unit_index=bound_index,
            )
            validate_hpo_output_manifest(
                manifest,
                capability,
                expected_capability_sha256=str(capability["capability_sha256"]),
                authoritative_rows=authoritative_rows,
                authoritative_dataset_sha256=authoritative_dataset_sha256,
                attempt_receipt=receipt,
            )
            output_by_unit[unit_id] = manifest
    except HpoOutputError as exc:
        raise HpoSelectionError(str(exc)) from exc
    if set(output_by_unit) != expected_unit_ids:
        raise HpoSelectionError("scope output catalog is missing an eligible unit")
    normalized_outputs = sorted(
        [
            {
                "unit_id": manifest["unit_id"],
                "attempt_index": manifest["attempt_index"],
                "manifest_sha256": manifest["manifest_sha256"],
            }
            for manifest in output_by_unit.values()
        ],
        key=lambda row: (
            str(row["unit_id"]),
            int(row["attempt_index"]),
            str(row["manifest_sha256"]),
        ),
    )
    if canonical_sha256(normalized_outputs) != scope_commitment["output_manifest_scope_sha256"]:
        raise HpoSelectionError("scope outputs differ from the global commitment")
    scope_units_by_candidate = {
        candidate_id: [unit for unit in expected_units if unit.get("candidate_id") == candidate_id]
        for candidate_id in eligible
    }
    output_catalog = {
        "catalog_validation_sha256": stage["output_catalog_validation_sha256"],
        "output_manifest_catalog_sha256": stage["output_manifest_catalog_sha256"],
    }
    return _assemble_selection_decision(
        hpo_plan,
        closure,
        candidate_space,
        output_catalog,
        output_by_unit,
        method=method,
        repeat=repeat,
        fold=fold,
        outer=outer,
        eligible=eligible,
        hpo_seeds=hpo_seeds,
        inner_folds=inner_folds,
        scope_units_by_candidate=scope_units_by_candidate,
    )


def validate_hpo_selection_decision(
    decision: Mapping[str, object],
    hpo_plan: Mapping[str, object],
    closure: Mapping[str, object],
    ledger_entries: Sequence[Mapping[str, object]],
    attempt_receipts: Sequence[Mapping[str, object]],
    output_manifests: Sequence[Mapping[str, object]],
    *,
    candidate_space: Mapping[str, object],
    nested_plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
    authoritative_rows: DatasetRows | Mapping[int, int],
    authoritative_dataset_sha256: str | None = None,
) -> None:
    """Recompute the full decision and compare it byte-for-byte in JSON space."""
    _validate_decision_shape(decision)
    expected = build_hpo_selection_decision(
        hpo_plan,
        closure,
        ledger_entries,
        attempt_receipts,
        output_manifests,
        candidate_space=candidate_space,
        nested_plan=nested_plan,
        group_manifest=group_manifest,
        authoritative_rows=authoritative_rows,
        method=str(decision["method"]),
        outer_repeat=int(decision["outer_repeat"]),
        outer_fold=int(decision["outer_fold"]),
        authoritative_dataset_sha256=authoritative_dataset_sha256,
    )
    if dict(decision) != expected:
        raise HpoSelectionError("selection decision differs from exact upstream evidence")


def outer_selection_receipt_arguments(decision: Mapping[str, object]) -> dict[str, object]:
    """Return the exact arguments consumed by ``build_outer_selection_receipt``."""
    _validate_decision_shape(decision)
    selected = decision["selected_candidate"]
    assert isinstance(selected, Mapping)
    return {
        "method": decision["method"],
        "outer_repeat": decision["outer_repeat"],
        "outer_fold": decision["outer_fold"],
        "candidate_id": selected["candidate_id"],
        "candidate_sha256": selected["candidate_sha256"],
        "selection_decision_manifest_sha256": decision["selection_decision_manifest_sha256"],
    }


__all__ = [
    "AGGREGATION_CONTRACT",
    "HpoSelectionError",
    "SCHEMA",
    "STATUS",
    "TIE_CONTRACT",
    "build_hpo_selection_decision",
    "build_hpo_selection_decision_from_catalog_commitment",
    "build_hpo_selection_decision_from_stage_commitment",
    "build_hpo_selection_stage_commitment",
    "outer_selection_receipt_arguments",
    "validate_hpo_selection_decision",
]
