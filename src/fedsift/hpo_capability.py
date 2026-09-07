"""Sealed, outer-test-free capabilities for one formal HPO worker unit.

The orchestrator validates the complete frozen study objects and emits one
exact-schema capability.  A worker receives only that capability: one candidate
configuration plus row tokens for the unit's private clients, ``V_ctrl``,
``V_sel`` and inner-validation slice.  No group identifier, outer-test row,
complete nested plan, or other HPO unit is serialized.
"""

from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import copy
import hmac
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from .candidate_space import CandidateSpaceError, canonical_sha256, require_sha256
from .group_manifest import GroupRecord, group_records
from .hpo_plan import HpoPlanError, validate_hpo_plan
from .nested_plan import CLIENT_NAMES


class HpoCapabilityError(RuntimeError):
    """Raised when a unit capability is malformed, forged, or mismatched."""


class OuterTestCapabilityError(HpoCapabilityError):
    """Raised before a worker can request any outer-test materialization."""


@dataclass(frozen=True)
class HpoUnitIndex:
    """Opaque, read-only index tied to one already validated HPO plan object."""

    hpo_plan_sha256: str
    unit_count: int
    units_by_id: Mapping[str, Mapping[str, object]] = field(repr=False)
    _plan: Mapping[str, object] = field(repr=False, compare=False)


SCHEMA = _identity("hpo_unit_capability")
STATUS = "SEALED_OUTER_TEST_EXCLUDED"
ROW_TOKEN_PREFIX = "row_id:"
_ROW_TOKEN = re.compile("^row_id:([0-9]{12})$")
_TOP_LEVEL_FIELDS = {
    "schema",
    "status",
    "study_id",
    "dataset_id",
    "unit_id",
    "bindings",
    "unit_identity",
    "candidate",
    "split_bindings",
    "data_slices",
    "worker_policy",
    "capability_sha256",
}
_BINDING_FIELDS = {
    "hpo_plan_sha256",
    "candidate_space_sha256",
    "nested_plan_sha256",
    "nested_freeze_binding_sha256",
    "dataset_sha256",
    "group_manifest_sha256",
    "row_to_group_sha256",
    "protocol_sha256",
    "implementation_contract_sha256",
    "preprocessing_profile_name",
    "preprocessing_profile_sha256",
    "preprocessing_profile",
}
_IDENTITY_FIELDS = {"method", "outer_repeat", "outer_fold", "inner_fold", "hpo_seed", "max_steps"}
_CANDIDATE_FIELDS = {"candidate_id", "candidate_sha256", "parameters_sha256", "parameters"}
_SPLIT_BINDING_FIELDS = {
    "outer_partition_sha256",
    "inner_partition_sha256",
    "inner_train_membership_sha256",
    "inner_validation_membership_sha256",
    "role_partition_sha256",
    "private_membership_sha256",
    "v_ctrl_membership_sha256",
    "v_sel_membership_sha256",
    "client_partition_sha256",
    "client_membership_sha256",
}
_SLICE_FIELDS = {
    "membership_sha256",
    "parent_split_sha256",
    "row_count",
    "row_ids_sha256",
    "row_ids",
}
_DATA_SLICE_FIELDS = {"private_clients", "v_ctrl", "v_sel", "inner_validation"}
_WORKER_POLICY = {
    "artifact_scope": "one_hpo_unit_only",
    "outer_test_materialization": "not_present_and_no_worker_api",
    "group_identifiers_serialized": False,
    "complete_nested_plan_serialized": False,
    "other_hpo_units_serialized": False,
    "row_identifier_encoding": "row_id_colon_12_digit_decimal",
}


def _artifact_hash(value: Mapping[str, object], field: str) -> str:
    payload = copy.deepcopy(dict(value))
    payload.pop(field, None)
    return canonical_sha256(payload)


def _root_group_map(group_manifest: Mapping[str, Any]) -> dict[str, GroupRecord]:
    return {record.group_id: record for record in group_records(group_manifest)}


def _row_token(row_id: int) -> str:
    if isinstance(row_id, bool) or not isinstance(row_id, int) or row_id < 0:
        raise HpoCapabilityError("row id must be a nonnegative exact integer")
    if row_id > 999999999999:
        raise HpoCapabilityError("row id exceeds the registered token width")
    return f"{ROW_TOKEN_PREFIX}{row_id:012d}"


def _decode_row_token(value: object) -> int:
    if not isinstance(value, str):
        raise HpoCapabilityError("serialized row identifiers must be strings")
    match = _ROW_TOKEN.fullmatch(value)
    if match is None:
        raise HpoCapabilityError("serialized row identifier is not canonical")
    return int(match.group(1))


def _membership_rows(
    membership: Mapping[str, Any], root_groups: Mapping[str, GroupRecord]
) -> list[int]:
    group_ids = membership.get("group_ids")
    if not isinstance(group_ids, list) or not group_ids:
        raise HpoCapabilityError("nested membership lacks non-empty group ids")
    if len(group_ids) != len(set(group_ids)):
        raise HpoCapabilityError("nested membership repeats a group id")
    try:
        rows = sorted(
            (row_id for group_id in group_ids for row_id in root_groups[str(group_id)].row_ids)
        )
    except KeyError as exc:
        raise HpoCapabilityError("nested membership references an unknown group") from exc
    if len(rows) != len(set(rows)):
        raise HpoCapabilityError("nested membership expands to duplicate row ids")
    if membership.get("row_count") != len(rows):
        raise HpoCapabilityError("nested membership row count differs")
    return rows


def _slice(
    membership: Mapping[str, Any],
    *,
    parent_split_sha256: object,
    root_groups: Mapping[str, GroupRecord],
) -> dict[str, object]:
    try:
        membership_sha256 = require_sha256(membership.get("membership_sha256"), "membership_sha256")
        parent_sha256 = require_sha256(parent_split_sha256, "parent_split_sha256")
    except CandidateSpaceError as exc:
        raise HpoCapabilityError(str(exc)) from exc
    tokens = [_row_token(row_id) for row_id in _membership_rows(membership, root_groups)]
    return {
        "membership_sha256": membership_sha256,
        "parent_split_sha256": parent_sha256,
        "row_count": len(tokens),
        "row_ids_sha256": canonical_sha256(tokens),
        "row_ids": tokens,
    }


def build_hpo_unit_index(hpo_plan: Mapping[str, object]) -> HpoUnitIndex:
    """Build a linear-time lookup index once for a frozen plan."""
    units = hpo_plan.get("units")
    if not isinstance(units, list):
        raise HpoCapabilityError("HPO plan has no unit inventory")
    try:
        plan_hash = require_sha256(hpo_plan.get("hpo_plan_sha256"), "hpo_plan_sha256")
    except CandidateSpaceError as exc:
        raise HpoCapabilityError(str(exc)) from exc
    indexed: dict[str, Mapping[str, object]] = {}
    for unit in units:
        if not isinstance(unit, Mapping):
            raise HpoCapabilityError("HPO plan unit inventory contains a non-mapping")
        unit_id = unit.get("unit_id")
        if not isinstance(unit_id, str) or not unit_id or unit_id in indexed:
            raise HpoCapabilityError("unit_id is not unique in the exact HPO plan")
        indexed[unit_id] = unit
    return HpoUnitIndex(
        hpo_plan_sha256=plan_hash,
        unit_count=len(units),
        units_by_id=MappingProxyType(indexed),
        _plan=hpo_plan,
    )


def _unit_by_id(
    hpo_plan: Mapping[str, object], unit_id: str, *, unit_index: HpoUnitIndex | None = None
) -> Mapping[str, object]:
    if not isinstance(unit_id, str) or not unit_id:
        raise HpoCapabilityError("unit_id must be a non-empty string")
    units = hpo_plan.get("units")
    if not isinstance(units, list):
        raise HpoCapabilityError("HPO plan has no unit inventory")
    if unit_index is not None:
        if (
            not isinstance(unit_index, HpoUnitIndex)
            or unit_index._plan is not hpo_plan
            or unit_index.hpo_plan_sha256 != hpo_plan.get("hpo_plan_sha256")
            or (unit_index.unit_count != len(units))
        ):
            raise HpoCapabilityError("unit index is not bound to the exact HPO plan")
        unit = unit_index.units_by_id.get(unit_id)
        if unit is None:
            raise HpoCapabilityError("unit_id is not unique in the exact HPO plan")
        return unit
    matches = [
        unit for unit in units if isinstance(unit, Mapping) and unit.get("unit_id") == unit_id
    ]
    if len(matches) != 1:
        raise HpoCapabilityError("unit_id is not unique in the exact HPO plan")
    return matches[0]


def _candidate_for_unit(
    candidate_space: Mapping[str, object], unit: Mapping[str, object]
) -> Mapping[str, object]:
    methods = candidate_space.get("methods")
    if not isinstance(methods, Mapping):
        raise HpoCapabilityError("candidate space has no method rosters")
    roster = methods.get(unit.get("method"))
    if not isinstance(roster, Mapping):
        raise HpoCapabilityError("unit method has no candidate roster")
    candidates = roster.get("candidates")
    if not isinstance(candidates, list):
        raise HpoCapabilityError("unit method candidate roster is missing")
    matches = [
        candidate
        for candidate in candidates
        if isinstance(candidate, Mapping)
        and candidate.get("candidate_id") == unit.get("candidate_id")
        and (candidate.get("candidate_sha256") == unit.get("candidate_sha256"))
    ]
    if len(matches) != 1:
        raise HpoCapabilityError("unit candidate does not match the frozen roster")
    return matches[0]


def _nested_unit(
    nested_plan: Mapping[str, Any], unit: Mapping[str, object]
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    try:
        repeat = nested_plan["repetitions"][int(unit["outer_repeat"])]
        fold = repeat["outer_folds"][int(unit["outer_fold"])]
        inner = fold["inner_folds"][int(unit["inner_fold"])]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise HpoCapabilityError("unit indexes do not resolve in the nested plan") from exc
    if not all((isinstance(value, Mapping) for value in (repeat, fold, inner))):
        raise HpoCapabilityError("resolved nested unit is malformed")
    return (repeat, fold, inner)


def _build_capability_payload(
    hpo_plan: Mapping[str, object],
    candidate_space: Mapping[str, object],
    nested_plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
    unit_id: str,
    *,
    unit_index: HpoUnitIndex | None = None,
) -> dict[str, object]:
    unit = _unit_by_id(hpo_plan, unit_id, unit_index=unit_index)
    candidate = _candidate_for_unit(candidate_space, unit)
    _, fold, inner = _nested_unit(nested_plan, unit)
    root_groups = _root_group_map(group_manifest)
    role_parent = inner["role_partition"]["split_sha256"]
    client_parent = inner["client_partition"]["split_sha256"]
    inner_parent = fold["inner_partition"]["split_sha256"]
    clients = inner.get("clients")
    roles = inner.get("roles")
    if not isinstance(clients, Mapping) or not isinstance(roles, Mapping):
        raise HpoCapabilityError("nested unit roles or clients are missing")
    private_clients = {
        client: _slice(clients[client], parent_split_sha256=client_parent, root_groups=root_groups)
        for client in CLIENT_NAMES
    }
    data_slices: dict[str, object] = {
        "private_clients": private_clients,
        "v_ctrl": _slice(roles["v_ctrl"], parent_split_sha256=role_parent, root_groups=root_groups),
        "v_sel": _slice(roles["v_sel"], parent_split_sha256=role_parent, root_groups=root_groups),
        "inner_validation": _slice(
            inner["inner_validation"], parent_split_sha256=inner_parent, root_groups=root_groups
        ),
    }
    private_rows = [
        token for client in CLIENT_NAMES for token in private_clients[client]["row_ids"]
    ]
    role_rows = [
        private_rows,
        list(data_slices["v_ctrl"]["row_ids"]),
        list(data_slices["v_sel"]["row_ids"]),
        list(data_slices["inner_validation"]["row_ids"]),
    ]
    flat_rows = [token for values in role_rows for token in values]
    if len(flat_rows) != len(set(flat_rows)):
        raise HpoCapabilityError("worker data slices are not row-exclusive")
    outer_rows = set(_membership_rows(fold["outer_test"], root_groups))
    worker_rows = {_decode_row_token(token) for token in flat_rows}
    if outer_rows & worker_rows:
        raise HpoCapabilityError("outer-test row entered an HPO unit capability")
    plan_bindings = hpo_plan.get("bindings")
    if not isinstance(plan_bindings, Mapping):
        raise HpoCapabilityError("HPO plan bindings are missing")
    parameters = copy.deepcopy(candidate["parameters"])
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "status": STATUS,
        "study_id": unit["study_id"],
        "dataset_id": unit["dataset_id"],
        "unit_id": unit_id,
        "bindings": {
            "hpo_plan_sha256": hpo_plan["hpo_plan_sha256"],
            "candidate_space_sha256": unit["candidate_space_sha256"],
            "nested_plan_sha256": unit["nested_plan_sha256"],
            "nested_freeze_binding_sha256": plan_bindings["nested_freeze_binding_sha256"],
            "dataset_sha256": unit["dataset_sha256"],
            "group_manifest_sha256": plan_bindings["group_manifest_sha256"],
            "row_to_group_sha256": plan_bindings["row_to_group_sha256"],
            "protocol_sha256": plan_bindings["protocol_sha256"],
            "implementation_contract_sha256": plan_bindings["implementation_contract_sha256"],
            "preprocessing_profile_name": unit["preprocessing_profile_name"],
            "preprocessing_profile_sha256": unit["preprocessing_profile_sha256"],
            "preprocessing_profile": copy.deepcopy(unit["preprocessing_profile"]),
        },
        "unit_identity": {
            "method": unit["method"],
            "outer_repeat": unit["outer_repeat"],
            "outer_fold": unit["outer_fold"],
            "inner_fold": unit["inner_fold"],
            "hpo_seed": unit["hpo_seed"],
            "max_steps": unit["max_steps"],
        },
        "candidate": {
            "candidate_id": unit["candidate_id"],
            "candidate_sha256": unit["candidate_sha256"],
            "parameters_sha256": canonical_sha256(parameters),
            "parameters": parameters,
        },
        "split_bindings": copy.deepcopy(unit["split_bindings"]),
        "data_slices": data_slices,
        "worker_policy": copy.deepcopy(_WORKER_POLICY),
    }
    payload["capability_sha256"] = _artifact_hash(payload, "capability_sha256")
    return payload


def build_hpo_unit_capability(
    hpo_plan: Mapping[str, object],
    candidate_space: Mapping[str, object],
    nested_plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
    *,
    unit_id: str,
) -> dict[str, object]:
    """Build one sealed worker capability from exact frozen upstreams."""
    try:
        validate_hpo_plan(hpo_plan, candidate_space, nested_plan, group_manifest)
    except HpoPlanError as exc:
        raise HpoCapabilityError(str(exc)) from exc
    capability = _build_capability_payload(
        hpo_plan, candidate_space, nested_plan, group_manifest, unit_id
    )
    validate_hpo_unit_capability(
        capability, hpo_plan, candidate_space, nested_plan, group_manifest, unit_id=unit_id
    )
    return capability


def build_prevalidated_hpo_unit_capability(
    hpo_plan: Mapping[str, object],
    candidate_space: Mapping[str, object],
    nested_plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
    *,
    unit_id: str,
    unit_index: HpoUnitIndex,
    expected_hpo_plan_sha256: str,
) -> dict[str, object]:
    """Build one unit after the orchestrator has validated the plan once."""
    try:
        expected = require_sha256(expected_hpo_plan_sha256, "expected_hpo_plan_sha256")
        stored = require_sha256(hpo_plan.get("hpo_plan_sha256"), "hpo_plan_sha256")
    except CandidateSpaceError as exc:
        raise HpoCapabilityError(str(exc)) from exc
    if not hmac.compare_digest(expected, stored):
        raise HpoCapabilityError("prevalidated HPO plan hash differs")
    capability = _build_capability_payload(
        hpo_plan, candidate_space, nested_plan, group_manifest, unit_id, unit_index=unit_index
    )
    _validate_sealed_shape(capability)
    return capability


def _validate_slice_shape(value: object, name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != _SLICE_FIELDS:
        raise HpoCapabilityError(f"{name} slice fields differ from the exact schema")
    try:
        require_sha256(value.get("membership_sha256"), f"{name}.membership_sha256")
        require_sha256(value.get("parent_split_sha256"), f"{name}.parent_split_sha256")
        require_sha256(value.get("row_ids_sha256"), f"{name}.row_ids_sha256")
    except CandidateSpaceError as exc:
        raise HpoCapabilityError(str(exc)) from exc
    row_ids = value.get("row_ids")
    if not isinstance(row_ids, list) or not row_ids:
        raise HpoCapabilityError(f"{name} row ids must be a non-empty list")
    decoded = [_decode_row_token(token) for token in row_ids]
    if decoded != sorted(decoded) or len(decoded) != len(set(decoded)):
        raise HpoCapabilityError(f"{name} row ids must be sorted and unique")
    if value.get("row_count") != len(row_ids):
        raise HpoCapabilityError(f"{name} row count differs")
    if value.get("row_ids_sha256") != canonical_sha256(row_ids):
        raise HpoCapabilityError(f"{name} row-id hash differs")


def _validate_sealed_shape(capability: Mapping[str, object]) -> None:
    if not isinstance(capability, Mapping) or set(capability) != _TOP_LEVEL_FIELDS:
        raise HpoCapabilityError("capability top-level fields differ from the exact schema")
    if capability.get("schema") != SCHEMA or capability.get("status") != STATUS:
        raise HpoCapabilityError("capability schema or status differs")
    for name in ("study_id", "dataset_id", "unit_id"):
        if not isinstance(capability.get(name), str) or not capability.get(name):
            raise HpoCapabilityError(f"capability {name} is invalid")
    bindings = capability.get("bindings")
    identity = capability.get("unit_identity")
    candidate = capability.get("candidate")
    split_bindings = capability.get("split_bindings")
    data_slices = capability.get("data_slices")
    if not isinstance(bindings, Mapping) or set(bindings) != _BINDING_FIELDS:
        raise HpoCapabilityError("capability bindings differ from the exact schema")
    if not isinstance(identity, Mapping) or set(identity) != _IDENTITY_FIELDS:
        raise HpoCapabilityError("capability unit identity differs from the exact schema")
    if not isinstance(candidate, Mapping) or set(candidate) != _CANDIDATE_FIELDS:
        raise HpoCapabilityError("capability candidate differs from the exact schema")
    if not isinstance(split_bindings, Mapping) or set(split_bindings) != _SPLIT_BINDING_FIELDS:
        raise HpoCapabilityError("capability split bindings differ from the exact schema")
    if not isinstance(data_slices, Mapping) or set(data_slices) != _DATA_SLICE_FIELDS:
        raise HpoCapabilityError("capability data slices differ from the exact schema")
    if capability.get("worker_policy") != _WORKER_POLICY:
        raise HpoCapabilityError("capability worker policy differs")
    try:
        for field, value in bindings.items():
            if field not in {"preprocessing_profile_name", "preprocessing_profile"}:
                require_sha256(value, str(field))
        require_sha256(candidate.get("candidate_sha256"), "candidate_sha256")
        require_sha256(candidate.get("parameters_sha256"), "parameters_sha256")
        for field, value in split_bindings.items():
            if field != "client_membership_sha256":
                require_sha256(value, str(field))
    except CandidateSpaceError as exc:
        raise HpoCapabilityError(str(exc)) from exc
    profile_spec = bindings.get("preprocessing_profile")
    if not isinstance(profile_spec, Mapping):
        raise HpoCapabilityError("capability preprocessing profile is missing")
    try:
        from .preprocessing import PreprocessingError, validate_preprocessing_profile_spec

        validate_preprocessing_profile_spec(profile_spec, dataset=str(capability["dataset_id"]))
    except PreprocessingError as exc:
        raise HpoCapabilityError(str(exc)) from exc
    if bindings.get("preprocessing_profile_name") != profile_spec.get("name") or bindings.get(
        "preprocessing_profile_sha256"
    ) != profile_spec.get("profile_sha256"):
        raise HpoCapabilityError("capability preprocessing profile bindings differ")
    client_membership = split_bindings.get("client_membership_sha256")
    if not isinstance(client_membership, Mapping) or set(client_membership) != set(CLIENT_NAMES):
        raise HpoCapabilityError(
            "capability client-membership bindings differ from the exact schema"
        )
    try:
        for client in CLIENT_NAMES:
            require_sha256(client_membership.get(client), f"client_membership_sha256.{client}")
    except CandidateSpaceError as exc:
        raise HpoCapabilityError(str(exc)) from exc
    method = identity.get("method")
    if not isinstance(method, str) or not method:
        raise HpoCapabilityError("capability method is invalid")
    for field, minimum in (
        ("outer_repeat", 0),
        ("outer_fold", 0),
        ("inner_fold", 0),
        ("hpo_seed", 0),
        ("max_steps", 1),
    ):
        value = identity.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise HpoCapabilityError(f"capability {field} is invalid")
    if (
        not isinstance(candidate.get("candidate_id"), str)
        or not candidate.get("candidate_id")
        or (not isinstance(candidate.get("parameters"), Mapping))
    ):
        raise HpoCapabilityError("capability candidate identity or parameters are invalid")
    if candidate.get("parameters_sha256") != canonical_sha256(candidate.get("parameters")):
        raise HpoCapabilityError("candidate parameter hash differs")
    private_clients = data_slices.get("private_clients")
    if not isinstance(private_clients, Mapping) or set(private_clients) != set(CLIENT_NAMES):
        raise HpoCapabilityError("private client slices differ from the exact schema")
    for client in CLIENT_NAMES:
        _validate_slice_shape(private_clients[client], client)
    for role in ("v_ctrl", "v_sel", "inner_validation"):
        _validate_slice_shape(data_slices[role], role)
    all_tokens: list[str] = []
    for client in CLIENT_NAMES:
        all_tokens.extend(private_clients[client]["row_ids"])
    for role in ("v_ctrl", "v_sel", "inner_validation"):
        all_tokens.extend(data_slices[role]["row_ids"])
    if len(all_tokens) != len(set(all_tokens)):
        raise HpoCapabilityError("capability data slices are not row-exclusive")
    stored = capability.get("capability_sha256")
    try:
        stored = require_sha256(stored, "capability_sha256")
    except CandidateSpaceError as exc:
        raise HpoCapabilityError(str(exc)) from exc
    if not hmac.compare_digest(stored, _artifact_hash(capability, "capability_sha256")):
        raise HpoCapabilityError("capability hash differs")


def validate_hpo_unit_capability(
    capability: Mapping[str, object],
    hpo_plan: Mapping[str, object],
    candidate_space: Mapping[str, object],
    nested_plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
    *,
    unit_id: str,
) -> None:
    """Rebuild a capability and compare it to all exact upstream objects."""
    _validate_sealed_shape(capability)
    try:
        validate_hpo_plan(hpo_plan, candidate_space, nested_plan, group_manifest)
    except HpoPlanError as exc:
        raise HpoCapabilityError(str(exc)) from exc
    expected = _build_capability_payload(
        hpo_plan, candidate_space, nested_plan, group_manifest, unit_id
    )
    if dict(capability) != expected:
        raise HpoCapabilityError("capability differs from the exact requested HPO unit")


def _require_expected_capability_hash(
    capability: Mapping[str, object], expected_capability_sha256: str
) -> None:
    _validate_sealed_shape(capability)
    try:
        expected = require_sha256(expected_capability_sha256, "expected_capability_sha256")
    except CandidateSpaceError as exc:
        raise HpoCapabilityError(str(exc)) from exc
    if not hmac.compare_digest(str(capability["capability_sha256"]), expected):
        raise HpoCapabilityError("worker received an unexpected capability hash")


def materialize_capability_row_ids(
    capability: Mapping[str, object], *, role: str, expected_capability_sha256: str
) -> tuple[int, ...]:
    """Worker-only row accessor; outer-test roles fail before any lookup."""
    if not isinstance(role, str):
        raise HpoCapabilityError("role must be a string")
    if role in {"outer_test", "outer-test", "test"}:
        raise OuterTestCapabilityError("HPO worker capabilities cannot materialize outer-test rows")
    allowed = {"v_ctrl", "v_sel", "inner_validation", *CLIENT_NAMES}
    if role not in allowed:
        raise HpoCapabilityError("role is not present in an HPO unit capability")
    _require_expected_capability_hash(capability, expected_capability_sha256)
    data_slices = capability["data_slices"]
    assert isinstance(data_slices, Mapping)
    if role in CLIENT_NAMES:
        private_clients = data_slices["private_clients"]
        assert isinstance(private_clients, Mapping)
        value = private_clients[role]
    else:
        value = data_slices[role]
    assert isinstance(value, Mapping)
    return tuple((_decode_row_token(token) for token in value["row_ids"]))


def capability_candidate_parameters(
    capability: Mapping[str, object], *, expected_capability_sha256: str
) -> dict[str, object]:
    """Return only the sealed unit candidate parameters to a worker."""
    _require_expected_capability_hash(capability, expected_capability_sha256)
    candidate = capability["candidate"]
    assert isinstance(candidate, Mapping)
    parameters = candidate["parameters"]
    if not isinstance(parameters, Mapping):
        raise HpoCapabilityError("sealed candidate parameters are not a mapping")
    return copy.deepcopy(dict(parameters))
