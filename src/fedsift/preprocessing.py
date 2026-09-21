"""Capability-scoped result-blind preprocessing fitted only on ``V_ctrl``.

The formal API accepts one sealed HPO capability, its independently delivered
expected hash, the exact :class:`DatasetRows`, and the exact root group
manifest.  Callers cannot provide role row IDs, a nested plan, or an outer-test
role.  Five private-client slices plus ``V_ctrl``, ``V_sel`` and
``inner_validation`` are derived exclusively from the capability.

The leading-underscore row-map helper remains for numerical unit tests.  It is
not a formal membership or provenance proof and must not be used by an HPO
runner.

When ``V_ctrl`` comes from the same sensitive benchmark, this preprocessing is
only a fixed auxiliary condition in a conditional record-level DP statement;
it is not an end-to-end DP transformation from every raw patient row to the
released model.
"""

from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import hashlib
import hmac
import json
import math
import struct
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Mapping, Sequence
from .group_manifest import DatasetRows, GroupManifestError, build_group_manifest, group_records


class PreprocessingError(RuntimeError):
    """Raised when preprocessing input, output or provenance fails closed."""


PIMA_PREDICTORS = (
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
)
PIMA_ZERO_AS_MISSING = ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")
DEBRECEN_PREDICTORS = tuple((str(index) for index in range(19)))
PIMA_PRIMARY_PROFILE = "pima_primary_zero_as_missing_v1"
PIMA_SENSITIVITY_PROFILE = "pima_sensitivity_zero_preserved_v1"
DEBRECEN_PROFILE = "debrecen_numeric_median_standardize_v1"
ROLE_NAMES = ("private", "v_ctrl", "v_sel", "inner_validation")
HPO_CLIENT_NAMES = tuple((f"client_{index}" for index in range(5)))
HPO_ROLE_NAMES = (*HPO_CLIENT_NAMES, "v_ctrl", "v_sel", "inner_validation")
ARTIFACT_SCHEMA = _identity("preprocessing_artifact")
HPO_ARTIFACT_SCHEMA = _identity("hpo_preprocessing_artifact")
_PROFILE_FIELDS = {
    "schema",
    "name",
    "dataset",
    "predictor_allowlist",
    "zero_as_missing_columns",
    "ordinary_nan_handling",
    "imputation_fit_role",
    "standardization_fit_role",
    "scale_definition",
    "zero_scale_policy",
    "transformed_roles",
    "profile_selection",
    "target_column_access",
    "outer_test_access",
    "profile_sha256",
}


@dataclass(frozen=True, slots=True)
class PreprocessedRoles:
    """In-memory transformed matrices plus a JSON-safe audit artifact."""

    artifact: dict[str, Any]
    matrices: dict[str, tuple[tuple[float, ...], ...]]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any((character not in "0123456789abcdef" for character in value))
    ):
        raise PreprocessingError(f"{name} must be a lowercase SHA-256")
    return value


def _profile_spec(profile: str) -> dict[str, Any]:
    if profile == PIMA_PRIMARY_PROFILE:
        dataset = "pima"
        predictors = PIMA_PREDICTORS
        zero_missing = PIMA_ZERO_AS_MISSING
    elif profile == PIMA_SENSITIVITY_PROFILE:
        dataset = "pima"
        predictors = PIMA_PREDICTORS
        zero_missing = ()
    elif profile == DEBRECEN_PROFILE:
        dataset = "retinopathy"
        predictors = DEBRECEN_PREDICTORS
        zero_missing = ()
    else:
        raise PreprocessingError(f"unknown preprocessing profile: {profile!r}")
    spec = {
        "schema": _identity("preprocessing_profile"),
        "name": profile,
        "dataset": dataset,
        "predictor_allowlist": list(predictors),
        "zero_as_missing_columns": list(zero_missing),
        "ordinary_nan_handling": "median_imputation",
        "imputation_fit_role": "v_ctrl_features_only",
        "standardization_fit_role": "v_ctrl_after_imputation_only",
        "scale_definition": "population_standard_deviation_ddof_0",
        "zero_scale_policy": "replace_with_one_and_record",
        "transformed_roles": list(HPO_ROLE_NAMES),
        "profile_selection": "pre_registered_without_model_performance",
        "target_column_access": "forbidden",
        "outer_test_access": "no_api_and_rejected_role_schema",
    }
    spec["profile_sha256"] = _sha256_json(spec)
    return spec


def preprocessing_profile(profile: str) -> dict[str, Any]:
    """Return the exact registered profile and its fingerprint."""
    return json.loads(_canonical_json(_profile_spec(profile)))


def preprocessing_profile_fingerprint(spec: Mapping[str, Any]) -> str:
    """Fingerprint every normative profile field except its stored hash."""
    if not isinstance(spec, Mapping):
        raise PreprocessingError("preprocessing profile must be a mapping")
    return _sha256_json(
        {str(key): value for (key, value) in spec.items() if key != "profile_sha256"}
    )


def validate_preprocessing_profile_spec(
    spec: Mapping[str, Any],
    *,
    dataset: str | None = None,
    feature_names: Sequence[str] | None = None,
) -> None:
    """Validate a complete, result-blind preprocessing profile specification."""
    if not isinstance(spec, Mapping) or set(spec) != _PROFILE_FIELDS:
        raise PreprocessingError("preprocessing profile fields differ from the exact schema")
    if spec.get("schema") != _identity("preprocessing_profile"):
        raise PreprocessingError("preprocessing profile schema differs")
    for field in ("name", "dataset"):
        if not isinstance(spec.get(field), str) or not str(spec.get(field)).strip():
            raise PreprocessingError(f"preprocessing profile {field} is invalid")
    predictors = spec.get("predictor_allowlist")
    zero_missing = spec.get("zero_as_missing_columns")
    if (
        not isinstance(predictors, list)
        or not predictors
        or any((not isinstance(value, str) or not value for value in predictors))
        or (len(predictors) != len(set(predictors)))
    ):
        raise PreprocessingError("profile predictor allowlist is invalid")
    if (
        not isinstance(zero_missing, list)
        or any((not isinstance(value, str) for value in zero_missing))
        or len(zero_missing) != len(set(zero_missing))
        or (not set(zero_missing).issubset(set(predictors)))
    ):
        raise PreprocessingError("profile zero-as-missing columns are invalid")
    normative = {
        "ordinary_nan_handling": "median_imputation",
        "imputation_fit_role": "v_ctrl_features_only",
        "standardization_fit_role": "v_ctrl_after_imputation_only",
        "scale_definition": "population_standard_deviation_ddof_0",
        "zero_scale_policy": "replace_with_one_and_record",
        "transformed_roles": list(HPO_ROLE_NAMES),
        "profile_selection": "pre_registered_without_model_performance",
        "target_column_access": "forbidden",
        "outer_test_access": "no_api_and_rejected_role_schema",
    }
    for field, expected in normative.items():
        if spec.get(field) != expected:
            raise PreprocessingError(f"preprocessing profile {field} differs")
    stored = _require_sha256(spec.get("profile_sha256"), "profile hash")
    if not hmac.compare_digest(stored, preprocessing_profile_fingerprint(spec)):
        raise PreprocessingError("preprocessing profile hash differs")
    if dataset is not None and spec.get("dataset") != dataset:
        raise PreprocessingError("preprocessing profile dataset differs")
    if feature_names is not None and list(feature_names) != predictors:
        raise PreprocessingError(
            "preprocessing profile predictor allowlist differs from exact features"
        )


def _normalize_roles(
    role_row_ids: Mapping[str, Sequence[int]],
    *,
    row_count: int,
    role_names: Sequence[str] = ROLE_NAMES,
) -> dict[str, tuple[int, ...]]:
    if not isinstance(role_row_ids, Mapping):
        raise PreprocessingError("role_row_ids must be an explicit role mapping")
    exact_roles = tuple((str(value) for value in role_names))
    if set(role_row_ids) != set(exact_roles):
        forbidden = sorted(
            (
                str(name)
                for name in role_row_ids
                if str(name).lower().replace("-", "_") in {"outer_test", "test"}
            )
        )
        if forbidden:
            raise PreprocessingError("outer-test role is forbidden in preprocessing")
        raise PreprocessingError("role schema differs from the exact preprocessing scope")
    normalized: dict[str, tuple[int, ...]] = {}
    seen: dict[int, str] = {}
    for role in exact_roles:
        values = role_row_ids[role]
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise PreprocessingError(f"{role} row IDs must be an explicit sequence")
        ids: list[int] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise PreprocessingError(f"{role} contains a non-integer row ID")
            row_id = int(value)
            if not 0 <= row_id < row_count:
                raise PreprocessingError(f"{role} contains an out-of-range row ID")
            ids.append(row_id)
        if not ids:
            raise PreprocessingError(f"{role} row membership must be non-empty")
        if len(ids) != len(set(ids)):
            raise PreprocessingError(f"{role} contains a duplicate row ID")
        ordered = tuple(sorted(ids))
        for row_id in ordered:
            if row_id in seen:
                raise PreprocessingError(f"row {row_id} overlaps roles {seen[row_id]} and {role}")
            seen[row_id] = role
        normalized[role] = ordered
    return normalized


def _validate_rows(rows: DatasetRows, spec: Mapping[str, Any]) -> None:
    if not isinstance(rows, DatasetRows):
        raise PreprocessingError("preprocessing requires DatasetRows, not a plan")
    validate_preprocessing_profile_spec(spec)
    _require_sha256(rows.source_sha256, "DatasetRows source hash")
    if rows.dataset != spec.get("dataset"):
        raise PreprocessingError("profile is not registered for this dataset")
    if rows.predictor_allowlist_explicit is not True:
        raise PreprocessingError("predictor allowlist must be explicit")
    expected = tuple((str(value) for value in spec["predictor_allowlist"]))
    if tuple(rows.feature_names) != expected:
        raise PreprocessingError("predictor columns or order differ from the profile")
    if not rows.target_name:
        raise PreprocessingError("declared target name is required for exclusion audit")
    target_folded = rows.target_name.casefold()
    if any((name.casefold() == target_folded for name in rows.feature_names)):
        raise PreprocessingError("declared target appears in predictor features")
    if len(rows.features) != len(rows.row_ids):
        raise PreprocessingError("DatasetRows feature length differs")


def _numeric_value(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PreprocessingError(f"{name} must be a real number")
    number = float(value)
    if math.isinf(number):
        raise PreprocessingError(f"{name} is infinite")
    return number


def _is_missing(value: float, feature_name: str, zero_missing: set[str]) -> bool:
    return math.isnan(value) or (feature_name in zero_missing and value == 0.0)


def _median(values: Sequence[float], feature_name: str) -> float:
    if not values:
        raise PreprocessingError(f"V_ctrl feature {feature_name!r} is entirely missing")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        result = ordered[middle]
    else:
        result = ordered[middle - 1] / 2.0 + ordered[middle] / 2.0
    if not math.isfinite(result):
        raise PreprocessingError(f"V_ctrl median for {feature_name!r} is non-finite")
    return result


def _raw_fit_matrix_sha256(
    rows: DatasetRows, row_ids: Sequence[int], *, zero_missing: set[str]
) -> str:
    digest = hashlib.sha256(_identity("control_fit_matrix_domain").encode("utf-8"))
    for row_id in row_ids:
        digest.update(struct.pack(">Q", row_id))
        for feature_name, raw in zip(rows.feature_names, rows.features[row_id]):
            value = _numeric_value(raw, name=f"row {row_id} feature {feature_name}")
            if _is_missing(value, feature_name, zero_missing):
                digest.update(b"M")
            else:
                digest.update(b"V")
                digest.update(struct.pack(">d", value))
    return digest.hexdigest()


def _fit_parameters(
    rows: DatasetRows, v_ctrl_ids: Sequence[int], *, zero_missing: set[str]
) -> list[dict[str, Any]]:
    parameters: list[dict[str, Any]] = []
    for column_index, feature_name in enumerate(rows.feature_names):
        observed: list[float] = []
        missing_count = 0
        for row_id in v_ctrl_ids:
            value = _numeric_value(
                rows.features[row_id][column_index],
                name=f"V_ctrl row {row_id} feature {feature_name}",
            )
            if _is_missing(value, feature_name, zero_missing):
                missing_count += 1
            else:
                observed.append(value)
        median = _median(observed, feature_name)
        imputed = [
            (
                median
                if _is_missing(
                    _numeric_value(
                        rows.features[row_id][column_index],
                        name=f"V_ctrl row {row_id} feature {feature_name}",
                    ),
                    feature_name,
                    zero_missing,
                )
                else float(rows.features[row_id][column_index])
            )
            for row_id in v_ctrl_ids
        ]
        try:
            mean = math.fsum(imputed) / len(imputed)
            variance = math.fsum(((value - mean) ** 2 for value in imputed)) / len(imputed)
        except (OverflowError, ValueError) as error:
            raise PreprocessingError(
                f"V_ctrl standardization arithmetic for {feature_name!r} failed"
            ) from error
        if not math.isfinite(mean) or not math.isfinite(variance) or variance < 0.0:
            raise PreprocessingError(
                f"V_ctrl standardization statistics for {feature_name!r} are invalid"
            )
        raw_scale = math.sqrt(variance)
        if not math.isfinite(raw_scale):
            raise PreprocessingError(f"V_ctrl scale for {feature_name!r} is invalid")
        scale_replaced = raw_scale == 0.0
        scale = 1.0 if scale_replaced else raw_scale
        parameters.append(
            {
                "feature_name": feature_name,
                "column_index": column_index,
                "zero_as_missing": feature_name in zero_missing,
                "v_ctrl_row_count": len(v_ctrl_ids),
                "v_ctrl_observed_count": len(observed),
                "v_ctrl_missing_count": missing_count,
                "median": median,
                "post_imputation_mean": mean,
                "population_scale_before_zero_replacement": raw_scale,
                "standardization_scale": scale,
                "zero_scale_replaced_with_one": scale_replaced,
            }
        )
    return parameters


def _transform_matrix(
    rows: DatasetRows, row_ids: Sequence[int], parameters: Sequence[Mapping[str, Any]]
) -> tuple[tuple[float, ...], ...]:
    matrix: list[tuple[float, ...]] = []
    for row_id in row_ids:
        transformed: list[float] = []
        for parameter in parameters:
            column_index = int(parameter["column_index"])
            feature_name = str(parameter["feature_name"])
            value = _numeric_value(
                rows.features[row_id][column_index], name=f"row {row_id} feature {feature_name}"
            )
            missing = math.isnan(value) or (bool(parameter["zero_as_missing"]) and value == 0.0)
            imputed = float(parameter["median"]) if missing else value
            try:
                standardized = (imputed - float(parameter["post_imputation_mean"])) / float(
                    parameter["standardization_scale"]
                )
            except (OverflowError, ZeroDivisionError, ValueError) as error:
                raise PreprocessingError(
                    f"transformed row {row_id} feature {feature_name!r} arithmetic failed"
                ) from error
            if not math.isfinite(standardized):
                raise PreprocessingError(
                    f"transformed row {row_id} feature {feature_name!r} is non-finite"
                )
            transformed.append(standardized)
        matrix.append(tuple(transformed))
    return tuple(matrix)


def _matrix_sha256(role: str, row_ids: Sequence[int], matrix: Sequence[Sequence[float]]) -> str:
    digest = hashlib.sha256(_identity("preprocessed_matrix_domain").encode("utf-8"))
    digest.update(role.encode("utf-8"))
    digest.update(struct.pack(">Q", len(row_ids)))
    for row_id, row in zip(row_ids, matrix):
        digest.update(struct.pack(">Q", row_id))
        digest.update(struct.pack(">I", len(row)))
        for value in row:
            number = float(value)
            if not math.isfinite(number):
                raise PreprocessingError("matrix hash received a non-finite value")
            digest.update(struct.pack(">d", number))
    return digest.hexdigest()


def preprocessing_artifact_fingerprint(artifact: Mapping[str, Any]) -> str:
    """Fingerprint every artifact field except the fingerprint itself."""
    if not isinstance(artifact, Mapping):
        raise PreprocessingError("preprocessing artifact must be a mapping")
    return _sha256_json(
        {str(key): value for (key, value) in artifact.items() if key != "artifact_sha256"}
    )


def _construct(
    rows: DatasetRows,
    role_row_ids: Mapping[str, Sequence[int]],
    *,
    profile: str | None = None,
    profile_spec: Mapping[str, Any] | None = None,
    group_manifest_sha256: str,
    role_names: Sequence[str] = ROLE_NAMES,
) -> PreprocessedRoles:
    if profile_spec is None:
        if not isinstance(profile, str):
            raise PreprocessingError("registered preprocessing profile is required")
        spec = _profile_spec(profile)
    else:
        spec = json.loads(_canonical_json(profile_spec))
        validate_preprocessing_profile_spec(spec)
        if profile is not None and profile != spec.get("name"):
            raise PreprocessingError("profile name and full specification differ")
    profile_name = str(spec["name"])
    _validate_rows(rows, spec)
    group_manifest_sha256 = _require_sha256(group_manifest_sha256, "group-manifest hash")
    exact_roles = tuple((str(value) for value in role_names))
    roles = _normalize_roles(role_row_ids, row_count=len(rows.row_ids), role_names=exact_roles)
    zero_missing = set((str(value) for value in spec["zero_as_missing_columns"]))
    parameters = _fit_parameters(rows, roles["v_ctrl"], zero_missing=zero_missing)
    parameter_sha256 = _sha256_json(parameters)
    matrices = {role: _transform_matrix(rows, roles[role], parameters) for role in exact_roles}
    role_bindings = {
        role: {
            "row_count": len(roles[role]),
            "row_ids": list(roles[role]),
            "membership_sha256": _sha256_json({"role": role, "row_ids": list(roles[role])}),
            "matrix_shape": [len(roles[role]), len(rows.feature_names)],
            "matrix_sha256": _matrix_sha256(role, roles[role], matrices[role]),
        }
        for role in exact_roles
    }
    role_membership_set_sha256 = _sha256_json(
        {role: role_bindings[role]["membership_sha256"] for role in exact_roles}
    )
    artifact: dict[str, Any] = {
        "schema": ARTIFACT_SCHEMA,
        "status": "RESULT_BLIND_V_CTRL_FEATURES_ONLY",
        "input_binding": {
            "dataset": rows.dataset,
            "source_sha256": rows.source_sha256,
            "group_manifest_sha256": group_manifest_sha256,
            "predictor_allowlist": list(rows.feature_names),
            "predictor_allowlist_sha256": _sha256_json(list(rows.feature_names)),
            "predictor_allowlist_explicit": True,
            "declared_target_name": rows.target_name,
            "declared_target_excluded": True,
            "profile_name": profile_name,
            "profile_sha256": spec["profile_sha256"],
            "role_membership_set_sha256": role_membership_set_sha256,
        },
        "profile": spec,
        "fit": {
            "fit_role": "v_ctrl",
            "fit_row_count": len(roles["v_ctrl"]),
            "fit_membership_sha256": role_bindings["v_ctrl"]["membership_sha256"],
            "fit_raw_feature_matrix_sha256": _raw_fit_matrix_sha256(
                rows, roles["v_ctrl"], zero_missing=zero_missing
            ),
            "labels_read": False,
            "non_v_ctrl_features_used_for_fit": False,
            "outer_test_features_used": False,
        },
        "parameters": parameters,
        "parameters_sha256": parameter_sha256,
        "roles": role_bindings,
        "privacy_boundary": {
            "claim": "conditional_record_level_dp_given_fixed_auxiliary_Z",
            "auxiliary_component": "V_ctrl_and_derived_preprocessing_parameters",
            "if_v_ctrl_is_sensitive_benchmark_data": "not_end_to_end_dp_from_all_raw_patient_rows_to_final_model",
            "labels_used_for_preprocessing": False,
            "security_or_privacy_proof_from_preprocessing_alone": False,
        },
    }
    artifact["artifact_sha256"] = preprocessing_artifact_fingerprint(artifact)
    return PreprocessedRoles(artifact=artifact, matrices=matrices)


def _preprocess_roles(
    rows: DatasetRows,
    role_row_ids: Mapping[str, Sequence[int]],
    *,
    profile: str,
    group_manifest_sha256: str,
) -> PreprocessedRoles:
    """Numerical test helper; does not prove nested or group membership."""
    return _construct(
        rows, role_row_ids, profile=profile, group_manifest_sha256=group_manifest_sha256
    )


def _validate_preprocessed_roles(
    value: PreprocessedRoles,
    rows: DatasetRows,
    role_row_ids: Mapping[str, Sequence[int]],
    *,
    profile: str,
    group_manifest_sha256: str,
) -> None:
    """Rebuild the numerical helper output; not a formal membership gate."""
    if not isinstance(value, PreprocessedRoles):
        raise PreprocessingError("preprocessed value has the wrong result type")
    if value.artifact.get("schema") != ARTIFACT_SCHEMA:
        raise PreprocessingError("preprocessing artifact schema differs")
    stored = _require_sha256(value.artifact.get("artifact_sha256"), "preprocessing artifact hash")
    if not hmac.compare_digest(stored, preprocessing_artifact_fingerprint(value.artifact)):
        raise PreprocessingError("preprocessing artifact fingerprint mismatch")
    expected = _construct(
        rows, role_row_ids, profile=profile, group_manifest_sha256=group_manifest_sha256
    )
    if value.artifact != expected.artifact or value.matrices != expected.matrices:
        raise PreprocessingError("preprocessed roles differ from exact V_ctrl-only reconstruction")


def preprocess_roles(
    rows: DatasetRows,
    role_row_ids: Mapping[str, Sequence[int]],
    *,
    profile: str,
    group_manifest_sha256: str,
) -> PreprocessedRoles:
    """Compatibility alias for tests; formal HPO code must not call this API."""
    return _preprocess_roles(
        rows, role_row_ids, profile=profile, group_manifest_sha256=group_manifest_sha256
    )


def validate_preprocessed_roles(
    value: PreprocessedRoles,
    rows: DatasetRows,
    role_row_ids: Mapping[str, Sequence[int]],
    *,
    profile: str,
    group_manifest_sha256: str,
) -> None:
    """Compatibility validator for the numerical test helper only."""
    _validate_preprocessed_roles(
        value, rows, role_row_ids, profile=profile, group_manifest_sha256=group_manifest_sha256
    )


def _validate_exact_root_group_manifest(
    rows: DatasetRows, group_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(group_manifest, Mapping):
        raise PreprocessingError("formal preprocessing requires a group manifest")
    try:
        group_records(group_manifest)
    except GroupManifestError as exc:
        raise PreprocessingError(str(exc)) from exc
    if group_manifest.get("group_rule") == "salted_subject_id_sha256":
        raise PreprocessingError(
            "subject-id manifest cannot be reconstructed from DatasetRows alone"
        )
    if (
        group_manifest.get("schema") != _identity("group_manifest")
        or group_manifest.get("group_rule")
        != "label_excluded_canonical_float64_exact_feature_sha256"
        or group_manifest.get("group_id_hash_inputs")
        != "loaded_predictor_matrix_excluding_declared_target"
    ):
        raise PreprocessingError("formal preprocessing requires the exact-feature root manifest")
    try:
        rebuilt = build_group_manifest(rows)
    except GroupManifestError as exc:
        raise PreprocessingError(str(exc)) from exc
    if dict(group_manifest) != rebuilt:
        raise PreprocessingError("DatasetRows do not reconstruct the exact supplied group manifest")
    return rebuilt


def _formal_capability_roles(
    capability: Mapping[str, object],
    *,
    expected_capability_sha256: str,
    group_manifest: Mapping[str, Any],
) -> tuple[dict[str, tuple[int, ...]], dict[str, dict[str, object]]]:
    from .hpo_capability import (
        HpoCapabilityError,
        _require_expected_capability_hash,
        materialize_capability_row_ids,
    )

    try:
        _require_expected_capability_hash(capability, expected_capability_sha256)
    except HpoCapabilityError as exc:
        raise PreprocessingError(str(exc)) from exc
    data_slices = capability.get("data_slices")
    if not isinstance(data_slices, Mapping):
        raise PreprocessingError("capability data slices are missing")
    private_clients = data_slices.get("private_clients")
    if not isinstance(private_clients, Mapping):
        raise PreprocessingError("capability private-client slices are missing")
    roles: dict[str, tuple[int, ...]] = {}
    slice_bindings: dict[str, Mapping[str, object]] = {}
    for role in HPO_ROLE_NAMES:
        try:
            roles[role] = materialize_capability_row_ids(
                capability, role=role, expected_capability_sha256=expected_capability_sha256
            )
        except HpoCapabilityError as exc:
            raise PreprocessingError(str(exc)) from exc
        raw_binding = private_clients[role] if role in HPO_CLIENT_NAMES else data_slices[role]
        if not isinstance(raw_binding, Mapping):
            raise PreprocessingError("capability role binding is malformed")
        slice_bindings[role] = raw_binding
    records = group_records(group_manifest)
    row_to_group = {row_id: record.group_id for record in records for row_id in record.row_ids}
    group_rows = {record.group_id: set(record.row_ids) for record in records}
    seen_groups: dict[str, str] = {}
    formal_bindings: dict[str, dict[str, object]] = {}
    for role in HPO_ROLE_NAMES:
        row_set = set(roles[role])
        try:
            group_ids = sorted({row_to_group[row_id] for row_id in row_set})
        except KeyError as exc:
            raise PreprocessingError(
                "capability role references a row outside the exact manifest"
            ) from exc
        for group_id in group_ids:
            if not group_rows[group_id].issubset(row_set):
                raise PreprocessingError(
                    "one exact group is split across capability roles or scope"
                )
            previous = seen_groups.get(group_id)
            if previous is not None:
                raise PreprocessingError(
                    f"exact group overlaps capability roles {previous} and {role}"
                )
            seen_groups[group_id] = role
        raw = slice_bindings[role]
        formal_bindings[role] = {
            "membership_sha256": raw["membership_sha256"],
            "parent_split_sha256": raw["parent_split_sha256"],
            "row_ids_sha256": raw["row_ids_sha256"],
            "row_count": raw["row_count"],
            "group_count": len(group_ids),
            "group_membership_sha256": _sha256_json(group_ids),
        }
    return (roles, formal_bindings)


def _construct_hpo_preprocessed_roles(
    capability: Mapping[str, object],
    rows: DatasetRows,
    group_manifest: Mapping[str, Any],
    *,
    expected_capability_sha256: str,
) -> PreprocessedRoles:
    manifest = _validate_exact_root_group_manifest(rows, group_manifest)
    bindings = capability.get("bindings")
    if not isinstance(bindings, Mapping):
        raise PreprocessingError("capability bindings are missing")
    for field, observed in (
        ("dataset_sha256", rows.source_sha256),
        ("group_manifest_sha256", manifest["group_manifest_sha256"]),
        ("row_to_group_sha256", manifest["row_to_group_sha256"]),
    ):
        if bindings.get(field) != observed:
            raise PreprocessingError(f"capability {field} differs from exact input")
    if capability.get("dataset_id") != rows.dataset:
        raise PreprocessingError("capability dataset identity differs from DatasetRows")
    profile_spec = bindings.get("preprocessing_profile")
    if not isinstance(profile_spec, Mapping):
        raise PreprocessingError("capability preprocessing profile is missing")
    validate_preprocessing_profile_spec(
        profile_spec, dataset=rows.dataset, feature_names=rows.feature_names
    )
    if bindings.get("preprocessing_profile_name") != profile_spec.get("name") or bindings.get(
        "preprocessing_profile_sha256"
    ) != profile_spec.get("profile_sha256"):
        raise PreprocessingError("capability preprocessing profile bindings differ")
    roles, capability_role_bindings = _formal_capability_roles(
        capability, expected_capability_sha256=expected_capability_sha256, group_manifest=manifest
    )
    result = _construct(
        rows,
        roles,
        profile_spec=profile_spec,
        group_manifest_sha256=str(manifest["group_manifest_sha256"]),
        role_names=HPO_ROLE_NAMES,
    )
    artifact = result.artifact
    artifact["schema"] = HPO_ARTIFACT_SCHEMA
    artifact["status"] = "SEALED_CAPABILITY_SCOPED_V_CTRL_ONLY"
    artifact["input_binding"].update(
        {
            "capability_sha256": capability["capability_sha256"],
            "hpo_plan_sha256": bindings["hpo_plan_sha256"],
            "dataset_sha256": bindings["dataset_sha256"],
            "row_to_group_sha256": bindings["row_to_group_sha256"],
            "preprocessing_profile_name": bindings["preprocessing_profile_name"],
            "preprocessing_profile_sha256": bindings["preprocessing_profile_sha256"],
            "labels_read_only_for_exact_manifest_reconstruction": True,
        }
    )
    artifact["privacy_boundary"].update(
        {
            "labels_used_for_parameter_fit": False,
            "labels_read_only_for_exact_manifest_reconstruction": True,
        }
    )
    artifact["capability_role_bindings"] = capability_role_bindings
    role_matrix_sha256 = {role: artifact["roles"][role]["matrix_sha256"] for role in HPO_ROLE_NAMES}
    artifact["output_binding"] = {
        "parameters_sha256": artifact["parameters_sha256"],
        "role_matrix_sha256": role_matrix_sha256,
        "role_matrix_set_sha256": _sha256_json(role_matrix_sha256),
    }
    artifact["artifact_sha256"] = preprocessing_artifact_fingerprint(artifact)
    return PreprocessedRoles(artifact=artifact, matrices=result.matrices)


def preprocess_hpo_capability(
    capability: Mapping[str, object],
    rows: DatasetRows,
    group_manifest: Mapping[str, Any],
    *,
    expected_capability_sha256: str,
) -> PreprocessedRoles:
    """Formally preprocess exactly one sealed HPO capability scope."""
    return _construct_hpo_preprocessed_roles(
        capability, rows, group_manifest, expected_capability_sha256=expected_capability_sha256
    )


def validate_hpo_preprocessed_roles(
    value: PreprocessedRoles,
    capability: Mapping[str, object],
    rows: DatasetRows,
    group_manifest: Mapping[str, Any],
    *,
    expected_capability_sha256: str,
) -> None:
    """Rebuild a formal capability-scoped artifact and every output float."""
    if not isinstance(value, PreprocessedRoles):
        raise PreprocessingError("preprocessed value has the wrong result type")
    if value.artifact.get("schema") != HPO_ARTIFACT_SCHEMA:
        raise PreprocessingError("formal preprocessing artifact schema differs")
    stored = _require_sha256(value.artifact.get("artifact_sha256"), "preprocessing artifact hash")
    if not hmac.compare_digest(stored, preprocessing_artifact_fingerprint(value.artifact)):
        raise PreprocessingError("preprocessing artifact fingerprint mismatch")
    expected = _construct_hpo_preprocessed_roles(
        capability, rows, group_manifest, expected_capability_sha256=expected_capability_sha256
    )
    if value.artifact != expected.artifact or value.matrices != expected.matrices:
        raise PreprocessingError(
            "formal preprocessing differs from exact capability reconstruction"
        )


__all__ = [
    "DEBRECEN_PROFILE",
    "PIMA_PRIMARY_PROFILE",
    "PIMA_SENSITIVITY_PROFILE",
    "PreprocessedRoles",
    "PreprocessingError",
    "preprocess_hpo_capability",
    "preprocessing_artifact_fingerprint",
    "preprocessing_profile",
    "preprocessing_profile_fingerprint",
    "validate_hpo_preprocessed_roles",
    "validate_preprocessing_profile_spec",
]
