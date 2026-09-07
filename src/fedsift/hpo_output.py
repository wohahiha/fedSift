"""Content-addressed, row-authoritative outputs for formal FedSift HPO units.

The result-blind attempt receipt intentionally contains only the hash of a
completed output artifact.  This module defines that artifact and closes the
other half of the evidence chain: the artifact is tied to one sealed worker
capability, one attempt, exact execution/privacy commitments, exact
``inner_validation`` rows and labels, and a resource total rebuilt from the
round receipts rather than accepted from the caller.

Native probabilities are retained without calibration or clipping.  Each
binary64 value is serialized with :meth:`float.hex`, so the stored evidence is
losslessly round-trippable.  Clipping, where required for log loss, happens
only in the common evaluation contract.
"""

from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import copy
import hmac
import math
from numbers import Integral, Real
from collections.abc import Iterable
from typing import Any, Mapping, Sequence
from .candidate_space import CandidateSpaceError, canonical_sha256, require_sha256
from .group_manifest import DatasetRows
from .hpo_attempt_receipt import (
    COMPLETE_STATUS,
    NONPRIVATE_PRIVACY_NOT_APPLICABLE,
    HpoAttemptReceiptError,
    validate_hpo_attempt_receipt,
    validate_hpo_attempt_receipt_catalog,
)
from .hpo_capability import (
    HpoCapabilityError,
    _build_capability_payload,
    _require_expected_capability_hash,
    build_hpo_unit_index,
    materialize_capability_row_ids,
)
from .resource_accounting import ResourceAccountingError, aggregate_resource_receipts


class HpoOutputError(ValueError):
    """Raised when an HPO output artifact is incomplete or not authoritative."""


SCHEMA = _identity("hpo_output_manifest")
STATUS = "SEALED_RAW_NATIVE_INNER_VALIDATION_OUTPUT"
CATALOG_SCHEMA = _identity("hpo_output_catalog_validation")
PREDICTION_CONTRACT: dict[str, object] = {
    "role": "inner_validation",
    "scale": "probability_of_binary_label_one",
    "state": "raw_native_uncalibrated",
    "calibration_applied": False,
    "clipping_applied": False,
    "serialization": "canonical_python_binary64_float_hex",
    "row_order": "sealed_capability_inner_validation_row_order",
    "outer_test_access": "forbidden",
}
_TOP_LEVEL_FIELDS = {
    "schema",
    "status",
    "study_id",
    "dataset_id",
    "unit_id",
    "attempt_index",
    "capability_binding",
    "execution_binding",
    "privacy_binding",
    "artifact_binding",
    "prediction_contract",
    "label_authority",
    "predictions",
    "resource_evidence",
    "manifest_sha256",
}
_CAPABILITY_BINDING_FIELDS = {
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
    "inner_validation_membership_sha256",
    "inner_validation_row_ids_sha256",
    "preprocessing_profile_sha256",
}
_EXECUTION_BINDING_FIELDS = {"code_sha256", "environment_sha256", "dependency_lock_sha256"}
_PRIVACY_BINDING_FIELDS = {
    "applicability",
    "privacy_accounting_report_sha256",
    "privacy_schedule_sha256",
}
_ARTIFACT_BINDING_FIELDS = {
    "preprocessing_manifest_sha256",
    "model_manifest_sha256",
    "training_trace_sha256",
    "privacy_accounting_report_sha256",
    "privacy_schedule_sha256",
    "resource_summary_sha256",
}
_LABEL_AUTHORITY_FIELDS = {
    "authority_kind",
    "dataset_sha256",
    "authority_row_count",
    "authority_row_label_mapping_sha256",
    "selected_labels_sha256",
}
_PREDICTION_FIELDS = {"row_token", "label", "probability_hex"}
_RESOURCE_EVIDENCE_FIELDS = {
    "expected_rounds",
    "round_resource_receipts",
    "resource_summary",
    "communication_bytes_definition",
    "communication_bytes",
}
_CATALOG_FIELDS = {
    "schema",
    "status",
    "hpo_plan_sha256",
    "attempt_receipt_catalog_validation_sha256",
    "complete_attempt_count",
    "output_manifest_count",
    "output_manifest_catalog_sha256",
    "scope_catalogs",
    "catalog_validation_sha256",
}
_SCOPE_CATALOG_FIELDS = {
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


def _unit_scope(unit_index: object, unit_id: str) -> tuple[str, int, int]:
    try:
        unit = unit_index.units_by_id[unit_id]
        method = unit["method"]
        outer_repeat = unit["outer_repeat"]
        outer_fold = unit["outer_fold"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise HpoOutputError("catalog entry references an unknown HPO unit") from exc
    if not isinstance(method, str) or not method:
        raise HpoOutputError("HPO unit method is invalid")
    if type(outer_repeat) is not int or type(outer_fold) is not int:
        raise HpoOutputError("HPO unit outer scope is invalid")
    return (method, outer_repeat, outer_fold)


def _scope_catalog_rows(
    complete_receipts: Sequence[Mapping[str, object]],
    normalized_outputs: Sequence[Mapping[str, object]],
    unit_index: object,
) -> list[dict[str, object]]:
    receipt_rows: dict[tuple[str, int, int], list[dict[str, object]]] = {}
    output_rows: dict[tuple[str, int, int], list[dict[str, object]]] = {}
    for receipt in complete_receipts:
        unit_id = str(receipt["unit_id"])
        scope = _unit_scope(unit_index, unit_id)
        receipt_rows.setdefault(scope, []).append(
            {
                "unit_id": unit_id,
                "attempt_index": receipt["attempt_index"],
                "attempt_receipt_sha256": receipt["attempt_receipt_sha256"],
            }
        )
    for output in normalized_outputs:
        unit_id = str(output["unit_id"])
        scope = _unit_scope(unit_index, unit_id)
        output_rows.setdefault(scope, []).append(dict(output))
    if set(receipt_rows) != set(output_rows):
        raise HpoOutputError("receipt and output scope inventories differ")
    catalogs: list[dict[str, object]] = []
    for method, outer_repeat, outer_fold in sorted(receipt_rows):
        scope = (method, outer_repeat, outer_fold)
        receipts = sorted(
            receipt_rows[scope],
            key=lambda row: (
                str(row["unit_id"]),
                int(row["attempt_index"]),
                str(row["attempt_receipt_sha256"]),
            ),
        )
        outputs = sorted(
            output_rows[scope],
            key=lambda row: (
                str(row["unit_id"]),
                int(row["attempt_index"]),
                str(row["manifest_sha256"]),
            ),
        )
        if len(receipts) != len(outputs):
            raise HpoOutputError("receipt and output counts differ within an HPO scope")
        catalogs.append(
            {
                "method": method,
                "outer_repeat": outer_repeat,
                "outer_fold": outer_fold,
                "complete_attempt_count": len(receipts),
                "attempt_receipt_scope_sha256": canonical_sha256(receipts),
                "output_manifest_scope_sha256": canonical_sha256(outputs),
            }
        )
    return catalogs


def _validate_scope_catalogs(
    value: object, complete_receipts: Sequence[Mapping[str, object]], unit_index: object
) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise HpoOutputError("scope catalogs must be a list")
    expected_receipt_rows: dict[tuple[str, int, int], list[dict[str, object]]] = {}
    for receipt in complete_receipts:
        unit_id = str(receipt["unit_id"])
        scope = _unit_scope(unit_index, unit_id)
        expected_receipt_rows.setdefault(scope, []).append(
            {
                "unit_id": unit_id,
                "attempt_index": receipt["attempt_index"],
                "attempt_receipt_sha256": receipt["attempt_receipt_sha256"],
            }
        )
    observed: dict[tuple[str, int, int], dict[str, object]] = {}
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != _SCOPE_CATALOG_FIELDS:
            raise HpoOutputError("scope catalog fields differ")
        method = raw.get("method")
        repeat = raw.get("outer_repeat")
        fold = raw.get("outer_fold")
        count = raw.get("complete_attempt_count")
        if (
            not isinstance(method, str)
            or not method
            or type(repeat) is not int
            or (type(fold) is not int)
            or (type(count) is not int)
            or (count < 1)
        ):
            raise HpoOutputError("scope catalog identity or count is invalid")
        scope = (method, repeat, fold)
        if scope in observed or scope not in expected_receipt_rows:
            raise HpoOutputError("scope catalog is duplicated or unplanned")
        receipt_hash = _hash(
            raw.get("attempt_receipt_scope_sha256"), "attempt_receipt_scope_sha256"
        )
        output_hash = _hash(raw.get("output_manifest_scope_sha256"), "output_manifest_scope_sha256")
        receipts = sorted(
            expected_receipt_rows[scope],
            key=lambda row: (
                str(row["unit_id"]),
                int(row["attempt_index"]),
                str(row["attempt_receipt_sha256"]),
            ),
        )
        if count != len(receipts) or not hmac.compare_digest(
            receipt_hash, canonical_sha256(receipts)
        ):
            raise HpoOutputError("scope receipt commitment differs")
        observed[scope] = {
            "method": method,
            "outer_repeat": repeat,
            "outer_fold": fold,
            "complete_attempt_count": count,
            "attempt_receipt_scope_sha256": receipt_hash,
            "output_manifest_scope_sha256": output_hash,
        }
    if set(observed) != set(expected_receipt_rows):
        raise HpoOutputError("scope catalog inventory is incomplete")
    return [observed[scope] for scope in sorted(observed)]


def _hash(value: object, field: str) -> str:
    try:
        return require_sha256(value, field)
    except CandidateSpaceError as exc:
        raise HpoOutputError(str(exc)) from exc


def _exact_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise HpoOutputError(f"{field} must be an exact integer")
    result = int(value)
    if result < minimum:
        raise HpoOutputError(f"{field} must be >= {minimum}")
    return result


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise HpoOutputError(f"{field} must be a non-empty canonical string")
    return value


def _sequence(value: object, field: str) -> list[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise HpoOutputError(f"{field} must be a sequence")
    return list(value)


def _iterable(value: object, field: str) -> Iterable[object]:
    if isinstance(value, (str, bytes, Mapping)):
        raise HpoOutputError(f"{field} must be an iterable of mappings")
    try:
        return iter(value)
    except TypeError as exc:
        raise HpoOutputError(f"{field} must be an iterable of mappings") from exc


def _row_token(row_id: int) -> str:
    if row_id < 0 or row_id > 999999999999:
        raise HpoOutputError("row id cannot be represented by the capability token")
    return f"row_id:{row_id:012d}"


def _canonical_probability(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise HpoOutputError(f"{field} must be a real number")
    probability = float(value)
    if not math.isfinite(probability):
        raise HpoOutputError(f"{field} must be finite")
    if probability < 0.0 or probability > 1.0:
        raise HpoOutputError(f"{field} must be in [0, 1]")
    return 0.0 if probability == 0.0 else probability


def probability_from_canonical_hex(value: object) -> float:
    """Decode one exact native probability and reject alternate encodings."""
    if not isinstance(value, str) or not value:
        raise HpoOutputError("probability_hex must be a non-empty string")
    try:
        probability = float.fromhex(value)
    except ValueError as exc:
        raise HpoOutputError("probability_hex is not binary64 hexadecimal") from exc
    probability = _canonical_probability(probability, "probability_hex")
    if probability.hex() != value:
        raise HpoOutputError("probability_hex is not the canonical binary64 encoding")
    return probability


def _authority_rows(
    authority: DatasetRows | Mapping[int, int],
    *,
    dataset_id: str,
    dataset_sha256: str,
    authoritative_dataset_sha256: str | None,
) -> tuple[dict[int, int], dict[str, object]]:
    if isinstance(authority, DatasetRows):
        if authority.dataset != dataset_id:
            raise HpoOutputError("DatasetRows dataset identity differs from capability")
        if authority.source_sha256 != dataset_sha256:
            raise HpoOutputError("DatasetRows source hash differs from capability")
        if (
            authoritative_dataset_sha256 is not None
            and authoritative_dataset_sha256 != dataset_sha256
        ):
            raise HpoOutputError("external dataset commitment differs from capability")
        row_labels = dict(zip(authority.row_ids, authority.labels))
        kind = "DatasetRows"
    elif isinstance(authority, Mapping):
        if authoritative_dataset_sha256 is None:
            raise HpoOutputError("a row-label mapping requires authoritative_dataset_sha256")
        if _hash(authoritative_dataset_sha256, "authoritative_dataset_sha256") != dataset_sha256:
            raise HpoOutputError("row-label mapping dataset hash differs from capability")
        row_labels = {}
        for raw_row_id, raw_label in authority.items():
            row_id = _exact_int(raw_row_id, "authority row id")
            label = _exact_int(raw_label, f"authority label for row {row_id}")
            if label not in (0, 1):
                raise HpoOutputError("authoritative labels must be exact binary values")
            row_labels[row_id] = label
        if not row_labels:
            raise HpoOutputError("authoritative row-label mapping must be non-empty")
        kind = "row_label_mapping"
    else:
        raise HpoOutputError("label authority must be DatasetRows or a row-label mapping")
    ordered = [{"row_id": row_id, "label": row_labels[row_id]} for row_id in sorted(row_labels)]
    metadata: dict[str, object] = {
        "authority_kind": kind,
        "dataset_sha256": dataset_sha256,
        "authority_row_count": len(ordered),
        "authority_row_label_mapping_sha256": canonical_sha256(ordered),
    }
    return (row_labels, metadata)


def _privacy_binding(
    method: str, privacy_accounting_report_sha256: object, privacy_schedule_sha256: object
) -> dict[str, str]:
    if method == "fedavg_nonprivate":
        if (
            privacy_accounting_report_sha256 != NONPRIVATE_PRIVACY_NOT_APPLICABLE
            or privacy_schedule_sha256 != NONPRIVATE_PRIVACY_NOT_APPLICABLE
        ):
            raise HpoOutputError(
                "non-private output must use the explicit privacy-not-applicable markers"
            )
        return {
            "applicability": "not_applicable_nonprivate",
            "privacy_accounting_report_sha256": NONPRIVATE_PRIVACY_NOT_APPLICABLE,
            "privacy_schedule_sha256": NONPRIVATE_PRIVACY_NOT_APPLICABLE,
        }
    return {
        "applicability": "record_dp_required",
        "privacy_accounting_report_sha256": _hash(
            privacy_accounting_report_sha256, "privacy_accounting_report_sha256"
        ),
        "privacy_schedule_sha256": _hash(privacy_schedule_sha256, "privacy_schedule_sha256"),
    }


def _capability_binding(capability: Mapping[str, object]) -> dict[str, object]:
    identity = capability["unit_identity"]
    candidate = capability["candidate"]
    bindings = capability["bindings"]
    split = capability["split_bindings"]
    inner = capability["data_slices"]["inner_validation"]
    assert all(
        (isinstance(value, Mapping) for value in (identity, candidate, bindings, split, inner))
    )
    return {
        "hpo_plan_sha256": bindings["hpo_plan_sha256"],
        "unit_capability_sha256": capability["capability_sha256"],
        "candidate_id": candidate["candidate_id"],
        "candidate_sha256": candidate["candidate_sha256"],
        "method": identity["method"],
        "outer_repeat": identity["outer_repeat"],
        "outer_fold": identity["outer_fold"],
        "inner_fold": identity["inner_fold"],
        "hpo_seed": identity["hpo_seed"],
        "max_steps": identity["max_steps"],
        "inner_validation_membership_sha256": split["inner_validation_membership_sha256"],
        "inner_validation_row_ids_sha256": inner["row_ids_sha256"],
        "preprocessing_profile_sha256": bindings["preprocessing_profile_sha256"],
    }


def _build_manifest(
    capability: Mapping[str, object],
    *,
    expected_capability_sha256: str,
    attempt_index: int,
    authoritative_rows: DatasetRows | Mapping[int, int],
    authoritative_dataset_sha256: str | None,
    probabilities: Sequence[float],
    preprocessing_manifest_sha256: str,
    model_manifest_sha256: str,
    training_trace_sha256: str,
    code_sha256: str,
    environment_sha256: str,
    dependency_lock_sha256: str,
    privacy_accounting_report_sha256: str,
    privacy_schedule_sha256: str,
    round_resource_receipts: Sequence[Mapping[str, object]],
    expected_rounds: int,
) -> dict[str, object]:
    try:
        _require_expected_capability_hash(capability, expected_capability_sha256)
        row_ids = materialize_capability_row_ids(
            capability,
            role="inner_validation",
            expected_capability_sha256=expected_capability_sha256,
        )
    except HpoCapabilityError as exc:
        raise HpoOutputError(str(exc)) from exc
    dataset_id = _identifier(capability.get("dataset_id"), "capability dataset_id")
    bindings = capability["bindings"]
    identity = capability["unit_identity"]
    assert isinstance(bindings, Mapping) and isinstance(identity, Mapping)
    dataset_sha256 = _hash(bindings.get("dataset_sha256"), "dataset_sha256")
    row_labels, authority_metadata = _authority_rows(
        authoritative_rows,
        dataset_id=dataset_id,
        dataset_sha256=dataset_sha256,
        authoritative_dataset_sha256=authoritative_dataset_sha256,
    )
    missing = [row_id for row_id in row_ids if row_id not in row_labels]
    if missing:
        raise HpoOutputError("label authority is missing inner-validation rows")
    raw_probabilities = _sequence(probabilities, "probabilities")
    if len(raw_probabilities) != len(row_ids):
        raise HpoOutputError("probability count differs from sealed inner-validation membership")
    canonical_probabilities = [
        _canonical_probability(value, f"probabilities[{index}]")
        for (index, value) in enumerate(raw_probabilities)
    ]
    prediction_rows = [
        {
            "row_token": _row_token(row_id),
            "label": row_labels[row_id],
            "probability_hex": probability.hex(),
        }
        for (row_id, probability) in zip(row_ids, canonical_probabilities)
    ]
    authority_metadata["selected_labels_sha256"] = canonical_sha256(
        [row["label"] for row in prediction_rows]
    )
    method = _identifier(identity.get("method"), "capability method")
    rounds = _exact_int(expected_rounds, "expected_rounds", minimum=1)
    model_hash = _hash(model_manifest_sha256, "model_manifest_sha256")
    try:
        resource_summary = aggregate_resource_receipts(
            round_resource_receipts,
            expected_method_id=method,
            expected_rounds=rounds,
            expected_model_manifest_sha256=model_hash,
        )
    except ResourceAccountingError as exc:
        raise HpoOutputError(str(exc)) from exc
    totals = resource_summary.get("totals")
    if not isinstance(totals, Mapping):
        raise HpoOutputError("rebuilt resource summary has no totals")
    communication_bytes = _exact_int(
        totals.get("round_total_federated_bytes"), "rebuilt communication bytes"
    )
    privacy = _privacy_binding(method, privacy_accounting_report_sha256, privacy_schedule_sha256)
    execution = {
        "code_sha256": _hash(code_sha256, "code_sha256"),
        "environment_sha256": _hash(environment_sha256, "environment_sha256"),
        "dependency_lock_sha256": _hash(dependency_lock_sha256, "dependency_lock_sha256"),
    }
    artifact_binding = {
        "preprocessing_manifest_sha256": _hash(
            preprocessing_manifest_sha256, "preprocessing_manifest_sha256"
        ),
        "model_manifest_sha256": model_hash,
        "training_trace_sha256": _hash(training_trace_sha256, "training_trace_sha256"),
        "privacy_accounting_report_sha256": privacy["privacy_accounting_report_sha256"],
        "privacy_schedule_sha256": privacy["privacy_schedule_sha256"],
        "resource_summary_sha256": resource_summary["report_sha256"],
    }
    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "status": STATUS,
        "study_id": capability["study_id"],
        "dataset_id": dataset_id,
        "unit_id": capability["unit_id"],
        "attempt_index": _exact_int(attempt_index, "attempt_index"),
        "capability_binding": _capability_binding(capability),
        "execution_binding": execution,
        "privacy_binding": privacy,
        "artifact_binding": artifact_binding,
        "prediction_contract": copy.deepcopy(PREDICTION_CONTRACT),
        "label_authority": authority_metadata,
        "predictions": prediction_rows,
        "resource_evidence": {
            "expected_rounds": rounds,
            "round_resource_receipts": [
                copy.deepcopy(dict(receipt)) for receipt in round_resource_receipts
            ],
            "resource_summary": resource_summary,
            "communication_bytes_definition": "sum_of_rebuilt_round_total_federated_bytes",
            "communication_bytes": communication_bytes,
        },
    }
    manifest["manifest_sha256"] = _artifact_hash(manifest, "manifest_sha256")
    return manifest


def build_hpo_output_manifest(
    capability: Mapping[str, object],
    *,
    expected_capability_sha256: str,
    attempt_index: int,
    authoritative_rows: DatasetRows | Mapping[int, int],
    probabilities: Sequence[float],
    preprocessing_manifest_sha256: str,
    model_manifest_sha256: str,
    training_trace_sha256: str,
    code_sha256: str,
    environment_sha256: str,
    dependency_lock_sha256: str,
    privacy_accounting_report_sha256: str,
    privacy_schedule_sha256: str,
    round_resource_receipts: Sequence[Mapping[str, object]],
    expected_rounds: int,
    authoritative_dataset_sha256: str | None = None,
) -> dict[str, object]:
    """Seal one complete, raw-native inner-validation output manifest."""
    manifest = _build_manifest(
        capability,
        expected_capability_sha256=expected_capability_sha256,
        attempt_index=attempt_index,
        authoritative_rows=authoritative_rows,
        authoritative_dataset_sha256=authoritative_dataset_sha256,
        probabilities=probabilities,
        preprocessing_manifest_sha256=preprocessing_manifest_sha256,
        model_manifest_sha256=model_manifest_sha256,
        training_trace_sha256=training_trace_sha256,
        code_sha256=code_sha256,
        environment_sha256=environment_sha256,
        dependency_lock_sha256=dependency_lock_sha256,
        privacy_accounting_report_sha256=privacy_accounting_report_sha256,
        privacy_schedule_sha256=privacy_schedule_sha256,
        round_resource_receipts=round_resource_receipts,
        expected_rounds=expected_rounds,
    )
    validate_hpo_output_manifest(
        manifest,
        capability,
        expected_capability_sha256=expected_capability_sha256,
        authoritative_rows=authoritative_rows,
        authoritative_dataset_sha256=authoritative_dataset_sha256,
    )
    return manifest


def _validate_shape(manifest: Mapping[str, object]) -> None:
    if not isinstance(manifest, Mapping) or set(manifest) != _TOP_LEVEL_FIELDS:
        raise HpoOutputError("output manifest fields differ from the exact schema")
    if manifest.get("schema") != SCHEMA or manifest.get("status") != STATUS:
        raise HpoOutputError("output manifest schema or status differs")
    for field in ("study_id", "dataset_id", "unit_id"):
        _identifier(manifest.get(field), field)
    _exact_int(manifest.get("attempt_index"), "attempt_index")
    for field, exact in (
        ("capability_binding", _CAPABILITY_BINDING_FIELDS),
        ("execution_binding", _EXECUTION_BINDING_FIELDS),
        ("privacy_binding", _PRIVACY_BINDING_FIELDS),
        ("artifact_binding", _ARTIFACT_BINDING_FIELDS),
        ("label_authority", _LABEL_AUTHORITY_FIELDS),
        ("resource_evidence", _RESOURCE_EVIDENCE_FIELDS),
    ):
        value = manifest.get(field)
        if not isinstance(value, Mapping) or set(value) != exact:
            raise HpoOutputError(f"{field} differs from the exact schema")
    if manifest.get("prediction_contract") != PREDICTION_CONTRACT:
        raise HpoOutputError("prediction contract differs")
    predictions = manifest.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise HpoOutputError("predictions must be a non-empty list")
    if any((not isinstance(row, Mapping) or set(row) != _PREDICTION_FIELDS for row in predictions)):
        raise HpoOutputError("prediction rows differ from the exact schema")
    stored = _hash(manifest.get("manifest_sha256"), "manifest_sha256")
    if not hmac.compare_digest(stored, _artifact_hash(manifest, "manifest_sha256")):
        raise HpoOutputError("output manifest canonical hash differs")


def validate_hpo_output_manifest(
    manifest: Mapping[str, object],
    capability: Mapping[str, object],
    *,
    expected_capability_sha256: str,
    authoritative_rows: DatasetRows | Mapping[int, int],
    authoritative_dataset_sha256: str | None = None,
    attempt_receipt: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate content, authority, resources, and optional attempt receipt."""
    _validate_shape(manifest)
    try:
        _require_expected_capability_hash(capability, expected_capability_sha256)
        expected_rows = materialize_capability_row_ids(
            capability,
            role="inner_validation",
            expected_capability_sha256=expected_capability_sha256,
        )
    except HpoCapabilityError as exc:
        raise HpoOutputError(str(exc)) from exc
    if (
        manifest.get("study_id") != capability.get("study_id")
        or manifest.get("dataset_id") != capability.get("dataset_id")
        or manifest.get("unit_id") != capability.get("unit_id")
        or (manifest.get("capability_binding") != _capability_binding(capability))
    ):
        raise HpoOutputError("output manifest differs from its sealed capability")
    bindings = capability["bindings"]
    assert isinstance(bindings, Mapping)
    row_labels, authority_metadata = _authority_rows(
        authoritative_rows,
        dataset_id=str(capability["dataset_id"]),
        dataset_sha256=str(bindings["dataset_sha256"]),
        authoritative_dataset_sha256=authoritative_dataset_sha256,
    )
    predictions = manifest["predictions"]
    assert isinstance(predictions, list)
    observed_rows: list[int] = []
    observed_labels: list[int] = []
    for index, (expected_row_id, row) in enumerate(zip(expected_rows, predictions)):
        assert isinstance(row, Mapping)
        expected_token = _row_token(expected_row_id)
        if row.get("row_token") != expected_token:
            raise HpoOutputError("prediction rows differ from exact capability membership/order")
        if expected_row_id not in row_labels:
            raise HpoOutputError("label authority is missing an inner-validation row")
        label = _exact_int(row.get("label"), f"predictions[{index}].label")
        if label not in (0, 1) or label != row_labels[expected_row_id]:
            raise HpoOutputError("prediction label differs from authoritative data")
        probability_from_canonical_hex(row.get("probability_hex"))
        observed_rows.append(expected_row_id)
        observed_labels.append(label)
    if len(predictions) != len(expected_rows):
        raise HpoOutputError("prediction count differs from exact capability membership")
    if len(observed_rows) != len(set(observed_rows)):
        raise HpoOutputError("prediction rows are duplicated")
    authority_metadata["selected_labels_sha256"] = canonical_sha256(observed_labels)
    if manifest.get("label_authority") != authority_metadata:
        raise HpoOutputError("label-authority commitment differs")
    execution = manifest["execution_binding"]
    privacy = manifest["privacy_binding"]
    artifact = manifest["artifact_binding"]
    resource = manifest["resource_evidence"]
    assert all((isinstance(value, Mapping) for value in (execution, privacy, artifact, resource)))
    for field in _EXECUTION_BINDING_FIELDS:
        _hash(execution.get(field), field)
    expected_privacy = _privacy_binding(
        str(manifest["capability_binding"]["method"]),
        privacy.get("privacy_accounting_report_sha256"),
        privacy.get("privacy_schedule_sha256"),
    )
    if dict(privacy) != expected_privacy:
        raise HpoOutputError("output privacy binding differs")
    for field in (
        "preprocessing_manifest_sha256",
        "model_manifest_sha256",
        "training_trace_sha256",
        "resource_summary_sha256",
    ):
        _hash(artifact.get(field), field)
    if (
        artifact.get("privacy_accounting_report_sha256")
        != privacy["privacy_accounting_report_sha256"]
        or artifact.get("privacy_schedule_sha256") != privacy["privacy_schedule_sha256"]
    ):
        raise HpoOutputError("artifact and privacy bindings disagree")
    rounds = _exact_int(resource.get("expected_rounds"), "expected_rounds", minimum=1)
    round_receipts = resource.get("round_resource_receipts")
    if not isinstance(round_receipts, list):
        raise HpoOutputError("round resource receipts must be a list")
    try:
        rebuilt = aggregate_resource_receipts(
            round_receipts,
            expected_method_id=str(manifest["capability_binding"]["method"]),
            expected_rounds=rounds,
            expected_model_manifest_sha256=str(artifact["model_manifest_sha256"]),
        )
    except ResourceAccountingError as exc:
        raise HpoOutputError(str(exc)) from exc
    if resource.get("resource_summary") != rebuilt:
        raise HpoOutputError("resource summary was not rebuilt from exact round receipts")
    if artifact.get("resource_summary_sha256") != rebuilt["report_sha256"]:
        raise HpoOutputError("resource summary hash binding differs")
    totals = rebuilt["totals"]
    assert isinstance(totals, Mapping)
    rebuilt_bytes = _exact_int(
        totals.get("round_total_federated_bytes"), "rebuilt communication bytes"
    )
    if (
        resource.get("communication_bytes_definition")
        != "sum_of_rebuilt_round_total_federated_bytes"
        or resource.get("communication_bytes") != rebuilt_bytes
    ):
        raise HpoOutputError("communication bytes differ from rebuilt round receipts")
    if attempt_receipt is not None:
        try:
            validate_hpo_attempt_receipt(attempt_receipt)
        except HpoAttemptReceiptError as exc:
            raise HpoOutputError(str(exc)) from exc
        if (
            attempt_receipt.get("status") != COMPLETE_STATUS
            or attempt_receipt.get("outcome") != "complete"
        ):
            raise HpoOutputError("only a complete attempt can bind an output manifest")
        if (
            attempt_receipt.get("unit_id") != manifest.get("unit_id")
            or attempt_receipt.get("attempt_index") != manifest.get("attempt_index")
            or attempt_receipt.get("unit_binding")
            != {
                key: manifest["capability_binding"][key]
                for key in (
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
                )
            }
            or (attempt_receipt.get("execution_binding") != execution)
            or (attempt_receipt.get("privacy_binding") != privacy)
        ):
            raise HpoOutputError("attempt execution/privacy/unit binding differs")
        if not hmac.compare_digest(
            str(attempt_receipt.get("exclusive_output_artifact_manifest_sha256")),
            str(manifest["manifest_sha256"]),
        ):
            raise HpoOutputError("attempt receipt names a different output manifest")
    return {
        "schema": _identity("hpo_output_validation"),
        "status": "VALIDATED_CONTENT_ADDRESSABLE_OUTPUT",
        "unit_id": manifest["unit_id"],
        "attempt_index": manifest["attempt_index"],
        "manifest_sha256": manifest["manifest_sha256"],
        "row_count": len(predictions),
        "communication_bytes": rebuilt_bytes,
    }


def validate_hpo_output_catalog(
    hpo_plan: Mapping[str, object],
    ledger_entries: Sequence[Mapping[str, object]],
    attempt_receipts: Sequence[Mapping[str, object]],
    output_manifests: Iterable[Mapping[str, object]],
    *,
    candidate_space: Mapping[str, object],
    nested_plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
    authoritative_rows: DatasetRows | Mapping[int, int],
    authoritative_dataset_sha256: str | None = None,
) -> dict[str, object]:
    """Validate one and only one output for every completed ledger attempt."""
    try:
        receipt_catalog = validate_hpo_attempt_receipt_catalog(
            hpo_plan,
            ledger_entries,
            attempt_receipts,
            candidate_space=candidate_space,
            nested_plan=nested_plan,
            group_manifest=group_manifest,
        )
    except HpoAttemptReceiptError as exc:
        raise HpoOutputError(str(exc)) from exc
    manifests = _iterable(output_manifests, "output_manifests")
    complete_receipts = [
        receipt
        for receipt in attempt_receipts
        if isinstance(receipt, Mapping) and receipt.get("outcome") == "complete"
    ]
    expected_by_hash: dict[str, Mapping[str, object]] = {}
    for receipt in complete_receipts:
        output_hash = str(receipt["exclusive_output_artifact_manifest_sha256"])
        if output_hash in expected_by_hash:
            raise HpoOutputError(
                "one output manifest hash cannot satisfy multiple complete attempts"
            )
        expected_by_hash[output_hash] = receipt
    observed_hashes: set[str] = set()
    observed_identity: set[tuple[str, int]] = set()
    normalized: list[dict[str, object]] = []
    unit_index = build_hpo_unit_index(hpo_plan)
    for raw_manifest in manifests:
        if not isinstance(raw_manifest, Mapping):
            raise HpoOutputError("output catalog entries must be mappings")
        _validate_shape(raw_manifest)
        manifest_hash = str(raw_manifest["manifest_sha256"])
        identity = (str(raw_manifest["unit_id"]), int(raw_manifest["attempt_index"]))
        if manifest_hash in observed_hashes:
            raise HpoOutputError("output catalog repeats a manifest hash")
        if identity in observed_identity:
            raise HpoOutputError("output catalog repeats an attempt identity")
        receipt = expected_by_hash.get(manifest_hash)
        if receipt is None:
            raise HpoOutputError("output catalog contains an orphan or forged output")
        capability = _build_capability_payload(
            hpo_plan,
            candidate_space,
            nested_plan,
            group_manifest,
            str(receipt["unit_id"]),
            unit_index=unit_index,
        )
        validate_hpo_output_manifest(
            raw_manifest,
            capability,
            expected_capability_sha256=str(capability["capability_sha256"]),
            authoritative_rows=authoritative_rows,
            authoritative_dataset_sha256=authoritative_dataset_sha256,
            attempt_receipt=receipt,
        )
        observed_hashes.add(manifest_hash)
        observed_identity.add(identity)
        normalized.append(
            {
                "unit_id": raw_manifest["unit_id"],
                "attempt_index": raw_manifest["attempt_index"],
                "manifest_sha256": raw_manifest["manifest_sha256"],
            }
        )
    if set(expected_by_hash) - observed_hashes:
        raise HpoOutputError("output catalog is missing a completed attempt output")
    normalized.sort(
        key=lambda value: (
            str(value["unit_id"]),
            int(value["attempt_index"]),
            str(value["manifest_sha256"]),
        )
    )
    validation: dict[str, object] = {
        "schema": CATALOG_SCHEMA,
        "status": "VALIDATED_COMPLETE_OUTPUT_CATALOG",
        "hpo_plan_sha256": hpo_plan["hpo_plan_sha256"],
        "attempt_receipt_catalog_validation_sha256": receipt_catalog["catalog_validation_sha256"],
        "complete_attempt_count": len(complete_receipts),
        "output_manifest_count": len(normalized),
        "output_manifest_catalog_sha256": canonical_sha256(normalized),
        "scope_catalogs": _scope_catalog_rows(complete_receipts, normalized, unit_index),
    }
    validation["catalog_validation_sha256"] = _artifact_hash(
        validation, "catalog_validation_sha256"
    )
    if set(validation) != _CATALOG_FIELDS:
        raise HpoOutputError("internal output-catalog validation schema differs")
    return validation


def validate_hpo_output_catalog_commitment(
    validation: Mapping[str, object],
    hpo_plan: Mapping[str, object],
    ledger_entries: Sequence[Mapping[str, object]],
    attempt_receipts: Sequence[Mapping[str, object]],
    *,
    candidate_space: Mapping[str, object],
    nested_plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
    expected_catalog_validation_sha256: str,
) -> dict[str, object]:
    """Validate a previously streamed catalog before scope-local selection.

    The externally supplied hash is the stage boundary: without it a caller
    could replace both a catalog commitment and its self-hash.  Raw prediction
    manifests remain outside this compact validation and are revalidated only
    for the selected outer/method scope downstream.
    """
    expected_hash = _hash(expected_catalog_validation_sha256, "expected_catalog_validation_sha256")
    if not isinstance(validation, Mapping) or set(validation) != _CATALOG_FIELDS:
        raise HpoOutputError("output catalog validation fields differ")
    if (
        validation.get("schema") != CATALOG_SCHEMA
        or validation.get("status") != "VALIDATED_COMPLETE_OUTPUT_CATALOG"
        or validation.get("hpo_plan_sha256") != hpo_plan.get("hpo_plan_sha256")
    ):
        raise HpoOutputError("output catalog validation identity differs")
    stored_hash = _hash(validation.get("catalog_validation_sha256"), "catalog_validation_sha256")
    if not hmac.compare_digest(stored_hash, expected_hash) or not hmac.compare_digest(
        stored_hash, _artifact_hash(validation, "catalog_validation_sha256")
    ):
        raise HpoOutputError("output catalog validation commitment differs")
    output_catalog_hash = _hash(
        validation.get("output_manifest_catalog_sha256"), "output_manifest_catalog_sha256"
    )
    try:
        receipt_catalog = validate_hpo_attempt_receipt_catalog(
            hpo_plan,
            ledger_entries,
            attempt_receipts,
            candidate_space=candidate_space,
            nested_plan=nested_plan,
            group_manifest=group_manifest,
        )
    except HpoAttemptReceiptError as exc:
        raise HpoOutputError(str(exc)) from exc
    complete_count = sum(
        (
            1
            for receipt in attempt_receipts
            if isinstance(receipt, Mapping) and receipt.get("outcome") == "complete"
        )
    )
    complete_receipts = [
        receipt
        for receipt in attempt_receipts
        if isinstance(receipt, Mapping) and receipt.get("outcome") == "complete"
    ]
    scope_catalogs = _validate_scope_catalogs(
        validation.get("scope_catalogs"), complete_receipts, build_hpo_unit_index(hpo_plan)
    )
    expected = {
        "schema": CATALOG_SCHEMA,
        "status": "VALIDATED_COMPLETE_OUTPUT_CATALOG",
        "hpo_plan_sha256": hpo_plan["hpo_plan_sha256"],
        "attempt_receipt_catalog_validation_sha256": receipt_catalog["catalog_validation_sha256"],
        "complete_attempt_count": complete_count,
        "output_manifest_count": complete_count,
        "output_manifest_catalog_sha256": output_catalog_hash,
        "scope_catalogs": scope_catalogs,
    }
    expected["catalog_validation_sha256"] = _artifact_hash(expected, "catalog_validation_sha256")
    if dict(validation) != expected:
        raise HpoOutputError("output catalog validation differs from the closed attempt catalog")
    return copy.deepcopy(expected)


__all__ = [
    "CATALOG_SCHEMA",
    "HpoOutputError",
    "PREDICTION_CONTRACT",
    "SCHEMA",
    "STATUS",
    "build_hpo_output_manifest",
    "probability_from_canonical_hex",
    "validate_hpo_output_catalog",
    "validate_hpo_output_catalog_commitment",
    "validate_hpo_output_manifest",
]
