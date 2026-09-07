from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
from fedsift.experiment_io import (
    registered_study,
    results_root,
    resolve_record_path,
    verify_release_sources,
)
import copy
import hmac
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from fedsift.candidate_space import canonical_sha256
from fedsift.group_manifest import DatasetRows
from fedsift.outer_refit import materialize_outer_refit_row_ids, validate_outer_test_access_receipt
from fedsift.preprocessing import (
    PreprocessedRoles,
    PreprocessingError,
    _construct,
    _matrix_sha256,
    _transform_matrix,
    _validate_exact_root_group_manifest,
    preprocessing_artifact_fingerprint,
    validate_preprocessing_profile_spec,
)

REFIT_SCHEMA = _identity("outer_refit_preprocessing")
OPENED_OUTER_SCHEMA = _identity("opened_outer_preprocessing")
REFIT_ROLES = ("client_0", "client_1", "client_2", "client_3", "client_4", "v_ctrl", "v_sel")


@dataclass(frozen=True)
class OpenedOuterMatrix:
    artifact: Mapping[str, Any]
    matrix: tuple[tuple[float, ...], ...]


def _require_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise PreprocessingError(f"{name} must be a sha256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise PreprocessingError(f"{name} must be a sha256 hex digest") from exc
    return value


def _row_tokens(row_ids: Sequence[int]) -> list[str]:
    return [f"row_id:{int(row_id):012d}" for row_id in row_ids]


def preprocess_outer_refit_capability(
    capability: Mapping[str, object],
    rows: DatasetRows,
    group_manifest: Mapping[str, Any],
    *,
    expected_capability_sha256: str,
) -> PreprocessedRoles:
    """Fit on V_ctrl and transform only the sealed outer-refit roles."""
    expected = _require_hash(expected_capability_sha256, "capability hash")
    if capability.get("capability_sha256") != expected:
        raise PreprocessingError("outer-refit capability hash differs")
    if capability.get("status") != "SEALED_OUTER_REFIT_OUTER_TEST_EXCLUDED":
        raise PreprocessingError("outer-refit capability is not sealed")
    manifest = _validate_exact_root_group_manifest(rows, group_manifest)
    bindings = capability.get("bindings")
    if not isinstance(bindings, Mapping):
        raise PreprocessingError("outer-refit capability bindings are missing")
    if capability.get("dataset_id") != rows.dataset:
        raise PreprocessingError("outer-refit dataset identity differs")
    for field, observed in (
        ("source_sha256", rows.source_sha256),
        ("group_manifest_sha256", manifest["group_manifest_sha256"]),
        ("row_to_group_sha256", manifest["row_to_group_sha256"]),
    ):
        if bindings.get(field) != observed:
            raise PreprocessingError(f"outer-refit {field} differs")
    profile = bindings.get("preprocessing_profile")
    if not isinstance(profile, Mapping):
        raise PreprocessingError("outer-refit preprocessing profile is missing")
    validate_preprocessing_profile_spec(
        profile, dataset=rows.dataset, feature_names=rows.feature_names
    )
    if bindings.get("preprocessing_profile_name") != profile.get("name") or bindings.get(
        "preprocessing_profile_sha256"
    ) != profile.get("profile_sha256"):
        raise PreprocessingError("outer-refit preprocessing profile binding differs")
    role_rows = {
        role: materialize_outer_refit_row_ids(
            capability, role=role, expected_capability_sha256=expected
        )
        for role in REFIT_ROLES
    }
    result = _construct(
        rows,
        role_rows,
        profile_spec=profile,
        group_manifest_sha256=str(manifest["group_manifest_sha256"]),
        role_names=REFIT_ROLES,
    )
    artifact = copy.deepcopy(result.artifact)
    artifact["schema"] = REFIT_SCHEMA
    artifact["status"] = "SEALED_OUTER_REFIT_PREPROCESSING_OUTER_TEST_UNREAD"
    artifact["input_binding"].update(
        {
            "capability_sha256": expected,
            "selection_receipt_sha256": bindings["selection_receipt_sha256"],
            "nested_plan_sha256": bindings["nested_plan_sha256"],
        }
    )
    artifact["fit"].update(
        {
            "outer_test_features_used": False,
            "outer_test_labels_used": False,
            "parameters_frozen_before_outer_open": True,
        }
    )
    artifact["privacy_boundary"].update(
        {
            "outer_test_features_read_only_for_exact_manifest_reconstruction": True,
            "outer_test_features_used_for_parameter_fit": False,
        }
    )
    artifact["artifact_sha256"] = preprocessing_artifact_fingerprint(artifact)
    return PreprocessedRoles(artifact=artifact, matrices=result.matrices)


def validate_outer_refit_preprocessing(
    value: PreprocessedRoles,
    capability: Mapping[str, object],
    rows: DatasetRows,
    group_manifest: Mapping[str, Any],
    *,
    expected_capability_sha256: str,
) -> None:
    if not isinstance(value, PreprocessedRoles):
        raise PreprocessingError("outer-refit preprocessing has the wrong type")
    stored = _require_hash(value.artifact.get("artifact_sha256"), "artifact hash")
    if not hmac.compare_digest(stored, preprocessing_artifact_fingerprint(value.artifact)):
        raise PreprocessingError("outer-refit preprocessing fingerprint differs")
    expected = preprocess_outer_refit_capability(
        capability, rows, group_manifest, expected_capability_sha256=expected_capability_sha256
    )
    if value.artifact != expected.artifact or value.matrices != expected.matrices:
        raise PreprocessingError("outer-refit preprocessing differs from reconstruction")


def apply_frozen_preprocessing_to_opened_outer_test(
    refit: PreprocessedRoles,
    rows: DatasetRows,
    outer_row_ids: Sequence[int],
    access_receipt: Mapping[str, object],
    gate_manifest: Mapping[str, object],
    *,
    expected_refit_artifact_sha256: str,
) -> OpenedOuterMatrix:
    """Apply already-frozen parameters after a successful one-time open."""
    if not isinstance(refit, PreprocessedRoles):
        raise PreprocessingError("outer-refit preprocessing has the wrong type")
    expected = _require_hash(expected_refit_artifact_sha256, "refit artifact hash")
    if refit.artifact.get("artifact_sha256") != expected:
        raise PreprocessingError("outer-refit preprocessing hash differs")
    if refit.artifact.get("schema") != REFIT_SCHEMA:
        raise PreprocessingError("outer-refit preprocessing schema differs")
    if not hmac.compare_digest(expected, preprocessing_artifact_fingerprint(refit.artifact)):
        raise PreprocessingError("outer-refit preprocessing fingerprint differs")
    validate_outer_test_access_receipt(access_receipt, gate_manifest)
    if access_receipt.get("outcome") != "opened" or access_receipt.get("rows_released") is not True:
        raise PreprocessingError("outer-test access receipt is not a successful open")
    ids = tuple((int(value) for value in outer_row_ids))
    if not ids or ids != tuple(sorted(set(ids))):
        raise PreprocessingError("opened outer-test row ids are not sorted and unique")
    if access_receipt.get("outer_test_row_ids_sha256") != canonical_sha256(_row_tokens(ids)):
        raise PreprocessingError("opened outer-test row ids differ from access receipt")
    refit_ids = {
        int(row_id) for role in REFIT_ROLES for row_id in refit.artifact["roles"][role]["row_ids"]
    }
    if refit_ids.intersection(ids):
        raise PreprocessingError("outer-test rows overlap outer-refit roles")
    parameters = refit.artifact.get("parameters")
    if not isinstance(parameters, list):
        raise PreprocessingError("frozen preprocessing parameters are missing")
    matrix = _transform_matrix(rows, ids, parameters)
    artifact: dict[str, Any] = {
        "schema": OPENED_OUTER_SCHEMA,
        "status": "FROZEN_PARAMETERS_APPLIED_AFTER_ONE_TIME_OUTER_OPEN",
        "dataset_id": rows.dataset,
        "source_sha256": rows.source_sha256,
        "refit_preprocessing_artifact_sha256": expected,
        "parameters_sha256": refit.artifact["parameters_sha256"],
        "access_receipt_sha256": access_receipt["access_receipt_sha256"],
        "gate_sha256": gate_manifest["gate_sha256"],
        "outer_test_row_ids_sha256": access_receipt["outer_test_row_ids_sha256"],
        "row_count": len(ids),
        "matrix_shape": [len(ids), len(rows.feature_names)],
        "matrix_sha256": _matrix_sha256("outer_test", ids, matrix),
        "parameters_refit_after_outer_open": False,
        "outer_test_labels_used_for_preprocessing": False,
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return OpenedOuterMatrix(artifact=artifact, matrix=matrix)


__all__ = [
    "OPENED_OUTER_SCHEMA",
    "REFIT_ROLES",
    "REFIT_SCHEMA",
    "OpenedOuterMatrix",
    "apply_frozen_preprocessing_to_opened_outer_test",
    "preprocess_outer_refit_capability",
    "validate_outer_refit_preprocessing",
]
