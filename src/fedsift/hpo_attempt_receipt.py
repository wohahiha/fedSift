"""Result-blind evidence receipts for formal HPO attempts.

An attempt receipt contains identities, reproducibility hashes, privacy-
accounting bindings, and exactly one terminal artifact hash.  It deliberately
contains no predictions, metrics, losses, or selection values.  Ledger closure
uses :func:`validate_hpo_attempt_receipt_catalog` to prove that every ledger
attempt names one canonical receipt derived from its exact planned unit and
sealed worker capability.
"""

from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import copy
import hmac
from typing import Any, Mapping, Sequence
from .candidate_space import (
    MAIN_METHODS,
    CandidateSpaceError,
    assert_no_performance_fields,
    canonical_sha256,
    require_exact_int,
    require_sha256,
)
from .hpo_capability import (
    HpoCapabilityError,
    _build_capability_payload,
    _require_expected_capability_hash,
    build_hpo_unit_index,
)


class HpoAttemptReceiptError(ValueError):
    """Raised when an HPO attempt receipt or catalog is not exact."""


SCHEMA = _identity("hpo_attempt_receipt")
COMPLETE_STATUS = "SEALED_COMPLETE_ATTEMPT"
FAILED_STATUS = "SEALED_FAILED_ATTEMPT"
COMPLETE_FAILURE_NOT_APPLICABLE = "NOT_APPLICABLE_COMPLETE"
NONPRIVATE_PRIVACY_NOT_APPLICABLE = "NOT_APPLICABLE_NONPRIVATE"
_COMMON_FIELDS = {
    "schema",
    "status",
    "unit_id",
    "attempt_index",
    "outcome",
    "failure_code",
    "unit_binding",
    "execution_binding",
    "privacy_binding",
    "attempt_receipt_sha256",
}
_COMPLETE_FIELDS = _COMMON_FIELDS | {"exclusive_output_artifact_manifest_sha256"}
_FAILED_FIELDS = _COMMON_FIELDS | {"failure_class", "failure_incident_sha256"}
_UNIT_BINDING_FIELDS = {
    "hpo_plan_sha256",
    "unit_capability_sha256",
    "candidate_id",
    "candidate_sha256",
    "method",
    "outer_repeat",
    "outer_fold",
    "inner_fold",
    "hpo_seed",
    "max_steps",
}
_EXECUTION_BINDING_FIELDS = {"code_sha256", "environment_sha256", "dependency_lock_sha256"}
_PRIVACY_BINDING_FIELDS = {
    "applicability",
    "privacy_accounting_report_sha256",
    "privacy_schedule_sha256",
}
_CATALOG_VALIDATION_FIELDS = {
    "schema",
    "status",
    "hpo_plan_sha256",
    "ledger_attempt_count",
    "receipt_count",
    "attempt_receipt_catalog_sha256",
    "catalog_validation_sha256",
}


def _artifact_hash(value: Mapping[str, object], field: str) -> str:
    payload = copy.deepcopy(dict(value))
    payload.pop(field, None)
    return canonical_sha256(payload)


def _failure_sets() -> tuple[frozenset[str], frozenset[str]]:
    from .hpo_plan import INFRASTRUCTURE_FAILURE_CODES, NONRETRYABLE_FAILURE_CODES

    return (INFRASTRUCTURE_FAILURE_CODES, NONRETRYABLE_FAILURE_CODES)


def _require_hash(value: object, field: str) -> str:
    try:
        return require_sha256(value, field)
    except CandidateSpaceError as exc:
        raise HpoAttemptReceiptError(str(exc)) from exc


def _require_index(value: object, field: str, minimum: int = 0) -> int:
    try:
        return require_exact_int(value, field, minimum=minimum)
    except CandidateSpaceError as exc:
        raise HpoAttemptReceiptError(str(exc)) from exc


def _privacy_binding(
    method: str, privacy_accounting_report_sha256: object, privacy_schedule_sha256: object
) -> dict[str, str]:
    if method not in MAIN_METHODS:
        raise HpoAttemptReceiptError("receipt method is not in the frozen roster")
    if method == "fedavg_nonprivate":
        if (
            privacy_accounting_report_sha256 != NONPRIVATE_PRIVACY_NOT_APPLICABLE
            or privacy_schedule_sha256 != NONPRIVATE_PRIVACY_NOT_APPLICABLE
        ):
            raise HpoAttemptReceiptError(
                "non-private attempts must mark both privacy bindings not applicable"
            )
        return {
            "applicability": "not_applicable_nonprivate",
            "privacy_accounting_report_sha256": NONPRIVATE_PRIVACY_NOT_APPLICABLE,
            "privacy_schedule_sha256": NONPRIVATE_PRIVACY_NOT_APPLICABLE,
        }
    return {
        "applicability": "record_dp_required",
        "privacy_accounting_report_sha256": _require_hash(
            privacy_accounting_report_sha256, "privacy_accounting_report_sha256"
        ),
        "privacy_schedule_sha256": _require_hash(
            privacy_schedule_sha256, "privacy_schedule_sha256"
        ),
    }


def build_hpo_attempt_receipt(
    capability: Mapping[str, object],
    *,
    expected_capability_sha256: str,
    attempt_index: int,
    outcome: str,
    failure_code: str | None = None,
    code_sha256: str,
    environment_sha256: str,
    dependency_lock_sha256: str,
    privacy_accounting_report_sha256: str,
    privacy_schedule_sha256: str,
    exclusive_output_artifact_manifest_sha256: str | None = None,
    failure_incident_sha256: str | None = None,
) -> dict[str, object]:
    """Build one canonical worker receipt from a sealed unit capability."""
    try:
        _require_expected_capability_hash(capability, expected_capability_sha256)
    except HpoCapabilityError as exc:
        raise HpoAttemptReceiptError(str(exc)) from exc
    index = _require_index(attempt_index, "attempt_index")
    if outcome not in {"complete", "failed"}:
        raise HpoAttemptReceiptError("attempt outcome must be complete or failed")
    identity = capability["unit_identity"]
    candidate = capability["candidate"]
    bindings = capability["bindings"]
    if not all((isinstance(value, Mapping) for value in (identity, candidate, bindings))):
        raise HpoAttemptReceiptError("sealed capability bindings are malformed")
    method = str(identity["method"])
    unit_binding: dict[str, object] = {
        "hpo_plan_sha256": bindings["hpo_plan_sha256"],
        "unit_capability_sha256": capability["capability_sha256"],
        "candidate_id": candidate["candidate_id"],
        "candidate_sha256": candidate["candidate_sha256"],
        "method": method,
        "outer_repeat": identity["outer_repeat"],
        "outer_fold": identity["outer_fold"],
        "inner_fold": identity["inner_fold"],
        "hpo_seed": identity["hpo_seed"],
        "max_steps": identity["max_steps"],
    }
    execution_binding = {
        "code_sha256": _require_hash(code_sha256, "code_sha256"),
        "environment_sha256": _require_hash(environment_sha256, "environment_sha256"),
        "dependency_lock_sha256": _require_hash(dependency_lock_sha256, "dependency_lock_sha256"),
    }
    privacy_binding = _privacy_binding(
        method, privacy_accounting_report_sha256, privacy_schedule_sha256
    )
    receipt: dict[str, object] = {
        "schema": SCHEMA,
        "status": COMPLETE_STATUS if outcome == "complete" else FAILED_STATUS,
        "unit_id": capability["unit_id"],
        "attempt_index": index,
        "outcome": outcome,
        "failure_code": COMPLETE_FAILURE_NOT_APPLICABLE if outcome == "complete" else failure_code,
        "unit_binding": unit_binding,
        "execution_binding": execution_binding,
        "privacy_binding": privacy_binding,
    }
    infrastructure, scientific = _failure_sets()
    if outcome == "complete":
        if failure_code is not None or failure_incident_sha256 is not None:
            raise HpoAttemptReceiptError("complete attempts cannot carry failure evidence")
        receipt["exclusive_output_artifact_manifest_sha256"] = _require_hash(
            exclusive_output_artifact_manifest_sha256, "exclusive_output_artifact_manifest_sha256"
        )
    else:
        if exclusive_output_artifact_manifest_sha256 is not None:
            raise HpoAttemptReceiptError("failed attempts cannot carry a completed output manifest")
        if failure_code in infrastructure:
            failure_class = "infrastructure"
        elif failure_code in scientific:
            failure_class = "scientific_nonretryable"
        else:
            raise HpoAttemptReceiptError("failure code is not preregistered")
        receipt["failure_class"] = failure_class
        receipt["failure_incident_sha256"] = _require_hash(
            failure_incident_sha256, "failure_incident_sha256"
        )
    receipt["attempt_receipt_sha256"] = _artifact_hash(receipt, "attempt_receipt_sha256")
    validate_hpo_attempt_receipt(receipt)
    return receipt


def validate_hpo_attempt_receipt(receipt: Mapping[str, object]) -> None:
    """Validate an exact receipt locally, including its canonical hash."""
    if not isinstance(receipt, Mapping):
        raise HpoAttemptReceiptError("attempt receipt must be a mapping")
    try:
        assert_no_performance_fields(receipt)
    except CandidateSpaceError as exc:
        raise HpoAttemptReceiptError(str(exc)) from exc
    outcome = receipt.get("outcome")
    expected_fields = (
        _COMPLETE_FIELDS
        if outcome == "complete"
        else _FAILED_FIELDS if outcome == "failed" else set()
    )
    if set(receipt) != expected_fields:
        raise HpoAttemptReceiptError("attempt receipt fields differ from the exact outcome schema")
    expected_status = COMPLETE_STATUS if outcome == "complete" else FAILED_STATUS
    if receipt.get("schema") != SCHEMA or receipt.get("status") != expected_status:
        raise HpoAttemptReceiptError("attempt receipt schema or status differs")
    if not isinstance(receipt.get("unit_id"), str) or not receipt.get("unit_id"):
        raise HpoAttemptReceiptError("attempt receipt unit id is invalid")
    _require_index(receipt.get("attempt_index"), "attempt_index")
    unit_binding = receipt.get("unit_binding")
    execution = receipt.get("execution_binding")
    privacy = receipt.get("privacy_binding")
    if not isinstance(unit_binding, Mapping) or set(unit_binding) != _UNIT_BINDING_FIELDS:
        raise HpoAttemptReceiptError("receipt unit binding differs from the exact schema")
    if not isinstance(execution, Mapping) or set(execution) != _EXECUTION_BINDING_FIELDS:
        raise HpoAttemptReceiptError("receipt execution binding differs from the exact schema")
    if not isinstance(privacy, Mapping) or set(privacy) != _PRIVACY_BINDING_FIELDS:
        raise HpoAttemptReceiptError("receipt privacy binding differs from the exact schema")
    for field in ("hpo_plan_sha256", "unit_capability_sha256", "candidate_sha256"):
        _require_hash(unit_binding.get(field), field)
    for field, minimum in (
        ("outer_repeat", 0),
        ("outer_fold", 0),
        ("inner_fold", 0),
        ("hpo_seed", 0),
        ("max_steps", 1),
    ):
        _require_index(unit_binding.get(field), field, minimum)
    for field in ("candidate_id", "method"):
        if not isinstance(unit_binding.get(field), str) or not unit_binding.get(field):
            raise HpoAttemptReceiptError(f"receipt {field} is invalid")
    for field in _EXECUTION_BINDING_FIELDS:
        _require_hash(execution.get(field), field)
    expected_privacy = _privacy_binding(
        str(unit_binding["method"]),
        privacy.get("privacy_accounting_report_sha256"),
        privacy.get("privacy_schedule_sha256"),
    )
    if dict(privacy) != expected_privacy:
        raise HpoAttemptReceiptError("receipt privacy applicability differs")
    infrastructure, scientific = _failure_sets()
    if outcome == "complete":
        if receipt.get("failure_code") != COMPLETE_FAILURE_NOT_APPLICABLE:
            raise HpoAttemptReceiptError("complete receipt failure code differs")
        _require_hash(
            receipt.get("exclusive_output_artifact_manifest_sha256"),
            "exclusive_output_artifact_manifest_sha256",
        )
    else:
        failure_code = receipt.get("failure_code")
        if failure_code in infrastructure:
            expected_class = "infrastructure"
        elif failure_code in scientific:
            expected_class = "scientific_nonretryable"
        else:
            raise HpoAttemptReceiptError("receipt failure code is not preregistered")
        if receipt.get("failure_class") != expected_class:
            raise HpoAttemptReceiptError("receipt failure class differs")
        _require_hash(receipt.get("failure_incident_sha256"), "failure_incident_sha256")
    stored = _require_hash(receipt.get("attempt_receipt_sha256"), "attempt_receipt_sha256")
    if not hmac.compare_digest(stored, _artifact_hash(receipt, "attempt_receipt_sha256")):
        raise HpoAttemptReceiptError("attempt receipt canonical hash differs")


def _ledger_attempt_references(
    ledger_entries: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if isinstance(ledger_entries, (str, bytes)):
        raise HpoAttemptReceiptError("HPO ledger must be a sequence")
    try:
        assert_no_performance_fields(ledger_entries)
    except CandidateSpaceError as exc:
        raise HpoAttemptReceiptError(str(exc)) from exc
    seen_units: set[str] = set()
    references: list[dict[str, object]] = []
    for entry in ledger_entries:
        if not isinstance(entry, Mapping) or set(entry) != {"unit_id", "attempts"}:
            raise HpoAttemptReceiptError("ledger entry fields differ from the exact schema")
        unit_id = entry.get("unit_id")
        if not isinstance(unit_id, str) or not unit_id or unit_id in seen_units:
            raise HpoAttemptReceiptError("ledger unit id is invalid or duplicated")
        seen_units.add(unit_id)
        attempts = entry.get("attempts")
        if not isinstance(attempts, list) or not 1 <= len(attempts) <= 2:
            raise HpoAttemptReceiptError("ledger attempts must contain one or two entries")
        for index, attempt in enumerate(attempts):
            if not isinstance(attempt, Mapping):
                raise HpoAttemptReceiptError("ledger attempt must be a mapping")
            outcome = attempt.get("outcome")
            expected_fields = {"attempt_index", "outcome", "attempt_receipt_sha256"}
            if outcome == "failed":
                expected_fields.add("failure_code")
            if set(attempt) != expected_fields:
                raise HpoAttemptReceiptError(
                    "ledger attempt fields differ from the exact outcome schema"
                )
            if attempt.get("attempt_index") != index:
                raise HpoAttemptReceiptError("ledger attempt indices must be contiguous from zero")
            receipt_sha256 = _require_hash(
                attempt.get("attempt_receipt_sha256"), "attempt_receipt_sha256"
            )
            references.append(
                {
                    "unit_id": unit_id,
                    "attempt_index": index,
                    "outcome": outcome,
                    "failure_code": attempt.get("failure_code", COMPLETE_FAILURE_NOT_APPLICABLE),
                    "attempt_receipt_sha256": receipt_sha256,
                }
            )
    receipt_hashes = [str(row["attempt_receipt_sha256"]) for row in references]
    if len(receipt_hashes) != len(set(receipt_hashes)):
        raise HpoAttemptReceiptError(
            "one attempt receipt hash cannot authorize multiple ledger attempts"
        )
    return references


def validate_hpo_attempt_receipt_catalog(
    plan: Mapping[str, object],
    ledger_entries: Sequence[Mapping[str, object]],
    attempt_receipts: Sequence[Mapping[str, object]],
    *,
    candidate_space: Mapping[str, object],
    nested_plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
) -> dict[str, object]:
    """Match every ledger attempt to one exact, upstream-bound receipt."""
    from .hpo_plan import HpoPlanError, validate_hpo_plan

    try:
        validate_hpo_plan(plan, candidate_space, nested_plan, group_manifest)
    except HpoPlanError as exc:
        raise HpoAttemptReceiptError(str(exc)) from exc
    references = _ledger_attempt_references(ledger_entries)
    if isinstance(attempt_receipts, (str, bytes)):
        raise HpoAttemptReceiptError("receipt catalog must be a sequence")
    try:
        assert_no_performance_fields(attempt_receipts)
    except CandidateSpaceError as exc:
        raise HpoAttemptReceiptError(str(exc)) from exc
    by_hash: dict[str, Mapping[str, object]] = {}
    by_identity: dict[tuple[str, int], str] = {}
    normalized: list[dict[str, object]] = []
    for receipt in attempt_receipts:
        if not isinstance(receipt, Mapping):
            raise HpoAttemptReceiptError("receipt catalog entries must be mappings")
        validate_hpo_attempt_receipt(receipt)
        receipt_hash = str(receipt["attempt_receipt_sha256"])
        identity = (str(receipt["unit_id"]), int(receipt["attempt_index"]))
        if receipt_hash in by_hash:
            raise HpoAttemptReceiptError("receipt catalog repeats a canonical hash")
        if identity in by_identity:
            raise HpoAttemptReceiptError("receipt catalog repeats an attempt identity")
        by_hash[receipt_hash] = receipt
        by_identity[identity] = receipt_hash
        normalized.append(copy.deepcopy(dict(receipt)))
    referenced_hashes = {str(reference["attempt_receipt_sha256"]) for reference in references}
    catalog_hashes = set(by_hash)
    if referenced_hashes - catalog_hashes:
        raise HpoAttemptReceiptError("ledger references a missing attempt receipt")
    if catalog_hashes - referenced_hashes:
        raise HpoAttemptReceiptError("receipt catalog contains an orphan receipt")
    units = plan.get("units")
    if not isinstance(units, list):
        raise HpoAttemptReceiptError("HPO plan unit inventory is missing")
    planned = {str(unit["unit_id"]): unit for unit in units if isinstance(unit, Mapping)}
    unit_index = build_hpo_unit_index(plan)
    expected_capabilities: dict[str, Mapping[str, object]] = {}
    for reference in references:
        receipt = by_hash[str(reference["attempt_receipt_sha256"])]
        unit_id = str(reference["unit_id"])
        unit = planned.get(unit_id)
        if unit is None:
            raise HpoAttemptReceiptError("ledger receipt references an unknown unit")
        if (
            receipt.get("unit_id") != unit_id
            or receipt.get("attempt_index") != reference["attempt_index"]
            or receipt.get("outcome") != reference["outcome"]
            or (receipt.get("failure_code") != reference["failure_code"])
        ):
            raise HpoAttemptReceiptError(
                "ledger attempt identity or outcome differs from its receipt"
            )
        binding = receipt["unit_binding"]
        assert isinstance(binding, Mapping)
        if unit_id not in expected_capabilities:
            expected_capabilities[unit_id] = _build_capability_payload(
                plan, candidate_space, nested_plan, group_manifest, unit_id, unit_index=unit_index
            )
        expected_capability = expected_capabilities[unit_id]
        expected_binding = {
            "hpo_plan_sha256": plan["hpo_plan_sha256"],
            "unit_capability_sha256": expected_capability["capability_sha256"],
            "candidate_id": unit["candidate_id"],
            "candidate_sha256": unit["candidate_sha256"],
            "method": unit["method"],
            "outer_repeat": unit["outer_repeat"],
            "outer_fold": unit["outer_fold"],
            "inner_fold": unit["inner_fold"],
            "hpo_seed": unit["hpo_seed"],
            "max_steps": unit["max_steps"],
        }
        if dict(binding) != expected_binding:
            raise HpoAttemptReceiptError(
                "receipt unit/candidate/capability binding differs from the exact plan"
            )
    normalized.sort(
        key=lambda receipt: (
            str(receipt["unit_id"]),
            int(receipt["attempt_index"]),
            str(receipt["attempt_receipt_sha256"]),
        )
    )
    validation: dict[str, object] = {
        "schema": _identity("hpo_attempt_receipt_catalog_validation"),
        "status": "VALIDATED_EXACT_RECEIPT_CATALOG",
        "hpo_plan_sha256": plan["hpo_plan_sha256"],
        "ledger_attempt_count": len(references),
        "receipt_count": len(normalized),
        "attempt_receipt_catalog_sha256": canonical_sha256(normalized),
    }
    validation["catalog_validation_sha256"] = _artifact_hash(
        validation, "catalog_validation_sha256"
    )
    if set(validation) != _CATALOG_VALIDATION_FIELDS:
        raise HpoAttemptReceiptError("internal catalog validation schema differs")
    return validation
