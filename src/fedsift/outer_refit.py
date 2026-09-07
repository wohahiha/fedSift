"""Result-blind outer-refit capabilities and one-time outer-test access.

The trusted orchestrator validates the complete frozen study and emits a
single outer-refit capability.  The capability contains only the selected
configuration and row tokens for the five private clients, ``V_ctrl`` and
``V_sel``.  It contains no outer-test membership, group identifier, or row
token.  Model, refit-validation prediction, and threshold outputs must then be
closed in order before an outer-test gate can be issued.

The row-token seals and the in-process open-once state are integrity controls
for an honest-but-fallible experiment worker.  They are not encryption and do
not isolate outer-test data from a malicious process that already possesses
the complete nested plan or dataset.  Open receipts must therefore be
persisted by the trusted experiment orchestrator for cross-process audit.
"""

from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import copy
import hmac
import re
import threading
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from .candidate_space import (
    MAIN_METHODS,
    CandidateSpaceError,
    assert_no_performance_fields,
    candidate_by_id,
    canonical_sha256,
    require_exact_int,
    require_sha256,
)
from .group_manifest import GroupRecord, group_records
from .hpo_plan import (
    MATCHED_ABLATIONS,
    HpoPlanError,
    inherit_matched_ablation,
    validate_hpo_plan,
    validate_selection_authorization,
)
from .nested_plan import CLIENT_NAMES


class OuterRefitError(RuntimeError):
    """Raised when an outer-refit artifact is malformed or mismatched."""


class ForbiddenOuterTestRefitAccess(OuterRefitError):
    """Raised before a refit worker can materialize an outer-test row."""


class OuterTestGateError(OuterRefitError):
    """Raised when a one-time outer-test gate rejects an access attempt."""

    def __init__(
        self, message: str, *, attempt_receipt: Mapping[str, object] | None = None
    ) -> None:
        super().__init__(message)
        self.attempt_receipt = (
            None if attempt_receipt is None else copy.deepcopy(dict(attempt_receipt))
        )


SELECTION_SCHEMA = _identity("outer_selection_receipt")
REFIT_CAPABILITY_SCHEMA = _identity("outer_refit_capability")
REFIT_STAGE_SCHEMA = _identity("outer_refit_stage_manifest")
OUTER_TEST_GATE_SCHEMA = _identity("outer_test_gate")
OUTER_TEST_ACCESS_SCHEMA = _identity("outer_test_access_receipt")
ROW_TOKEN_PREFIX = "row_id:"
_ROW_TOKEN = re.compile("^row_id:([0-9]{12})$")
_SELECTION_FIELDS = {
    "schema",
    "status",
    "study_id",
    "hpo_plan_sha256",
    "hpo_closure_sha256",
    "terminal_ledger_sha256",
    "attempt_receipt_catalog_sha256",
    "outer_authorization_sha256",
    "selection_decision_manifest_sha256",
    "method",
    "outer_repeat",
    "outer_fold",
    "candidate_id",
    "candidate_sha256",
    "selection_receipt_sha256",
}
_CAPABILITY_FIELDS = {
    "schema",
    "status",
    "study_id",
    "dataset_id",
    "bindings",
    "scope",
    "candidate",
    "matched_ablation_inheritance",
    "split_bindings",
    "data_slices",
    "worker_policy",
    "capability_sha256",
}
_CAPABILITY_BINDING_FIELDS = {
    "hpo_plan_sha256",
    "hpo_closure_sha256",
    "terminal_ledger_sha256",
    "attempt_receipt_catalog_sha256",
    "outer_authorization_sha256",
    "selection_receipt_sha256",
    "selection_decision_manifest_sha256",
    "candidate_space_sha256",
    "nested_plan_sha256",
    "nested_freeze_binding_sha256",
    "source_sha256",
    "group_manifest_sha256",
    "row_to_group_sha256",
    "protocol_sha256",
    "implementation_contract_sha256",
    "client_partition_policy",
    "client_partition_policy_sha256",
    "preprocessing_profile_name",
    "preprocessing_profile_sha256",
    "preprocessing_profile",
}
_SCOPE_FIELDS = {"method_identity", "parent_method", "outer_repeat", "outer_fold", "eval_seed"}
_CANDIDATE_FIELDS = {
    "candidate_id",
    "candidate_sha256",
    "parameters_sha256",
    "parameters",
    "mechanism_switch",
}
_SPLIT_FIELDS = {
    "outer_partition_sha256",
    "outer_train_membership_sha256",
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
_DATA_SLICE_FIELDS = {"private_clients", "v_ctrl", "v_sel"}
_WORKER_POLICY = {
    "artifact_scope": "one_selected_outer_refit_only",
    "allowed_roles": [*CLIENT_NAMES, "v_ctrl", "v_sel"],
    "manual_row_ids": "forbidden",
    "outer_test_membership": "not_serialized",
    "outer_test_materialization": "not_present_and_no_refit_worker_api",
    "complete_nested_plan_serialized": False,
    "group_identifiers_serialized": False,
    "row_identifier_encoding": "row_id_colon_12_digit_decimal",
    "security_boundary": "row_tokens_and_seals_are_integrity_controls_not_cryptographic_isolation_from_a_malicious_process",
}
_STAGE_FIELDS = {
    "schema",
    "status",
    "stage",
    "bindings",
    "scope",
    "role_bindings",
    "predecessor",
    "output_artifact_manifest_sha256",
    "stage_manifest_sha256",
}
_STAGE_BINDING_FIELDS = {
    "refit_capability_sha256",
    "hpo_plan_sha256",
    "nested_plan_sha256",
    "source_sha256",
    "group_manifest_sha256",
    "preprocessing_profile_sha256",
    "candidate_sha256",
    "selection_receipt_sha256",
}
_ROLE_BINDING_FIELDS = {
    "private_client_membership_sha256",
    "v_ctrl_membership_sha256",
    "v_sel_membership_sha256",
}
_STAGE_ORDER = ("refit_model", "refit_validation_predictions", "decision_threshold")
_GATE_FIELDS = {
    "schema",
    "status",
    "bindings",
    "scope",
    "outer_test_seal",
    "access_policy",
    "gate_sha256",
}
_GATE_BINDING_FIELDS = {
    "refit_capability_sha256",
    "model_stage_manifest_sha256",
    "prediction_stage_manifest_sha256",
    "threshold_stage_manifest_sha256",
    "hpo_plan_sha256",
    "nested_plan_sha256",
    "source_sha256",
    "group_manifest_sha256",
    "preprocessing_profile_sha256",
    "candidate_sha256",
    "selection_receipt_sha256",
    "one_time_access_authorization_sha256",
}
_OUTER_TEST_SEAL_FIELDS = {"membership_sha256"}
_ACCESS_POLICY = {
    "maximum_successful_or_failed_opens": 1,
    "failed_open_consumes_gate": True,
    "manual_row_ids": "forbidden",
    "refit_capability_can_rebuild_outer_test": False,
    "replay_scope": "one_trusted_orchestrator_process",
    "receipt_persistence": "required_for_cross_process_audit",
    "security_boundary": "membership_seal_and_open_state_are_integrity_controls_not_cryptographic_isolation",
}
_ACCESS_FIELDS = {
    "schema",
    "status",
    "gate_sha256",
    "access_attempt_id",
    "scope",
    "outcome",
    "failure_code",
    "failure_incident_sha256",
    "outer_test_row_ids_sha256",
    "rows_released",
    "access_receipt_sha256",
}
_REJECT_FAILURE_CODES = frozenset(
    {
        "replay_or_second_open",
        "duplicate_access_attempt_id",
        "cross_outer_scope",
        "cross_eval_seed",
        "cross_hpo_plan",
        "cross_nested_plan",
        "cross_group_manifest",
        "upstream_validation_failure",
    }
)
_FAILED_OPEN_CODES = frozenset({"outer_test_loader_failure", "internal_materialization_failure"})


def _artifact_hash(value: Mapping[str, object], field: str) -> str:
    payload = copy.deepcopy(dict(value))
    payload.pop(field, None)
    return canonical_sha256(payload)


def _require_hash(value: object, field: str) -> str:
    try:
        return require_sha256(value, field)
    except CandidateSpaceError as exc:
        raise OuterRefitError(str(exc)) from exc


def _require_int(value: object, field: str, *, minimum: int = 0) -> int:
    try:
        result = require_exact_int(value, field, minimum=minimum)
    except CandidateSpaceError as exc:
        raise OuterRefitError(str(exc)) from exc
    if result > 2**63 - 1:
        raise OuterRefitError(f"{field} must fit a signed 64-bit integer")
    return result


def _root_group_map(group_manifest: Mapping[str, Any]) -> dict[str, GroupRecord]:
    return {record.group_id: record for record in group_records(group_manifest)}


def _row_token(row_id: int) -> str:
    if isinstance(row_id, bool) or not isinstance(row_id, int) or row_id < 0:
        raise OuterRefitError("row id must be a nonnegative exact integer")
    if row_id > 999999999999:
        raise OuterRefitError("row id exceeds the registered token width")
    return f"{ROW_TOKEN_PREFIX}{row_id:012d}"


def _decode_row_token(value: object) -> int:
    if not isinstance(value, str):
        raise OuterRefitError("serialized row identifiers must be strings")
    match = _ROW_TOKEN.fullmatch(value)
    if match is None:
        raise OuterRefitError("serialized row identifier is not canonical")
    return int(match.group(1))


def _membership_rows(
    membership: Mapping[str, Any], root_groups: Mapping[str, GroupRecord]
) -> list[int]:
    group_ids = membership.get("group_ids")
    if not isinstance(group_ids, list) or not group_ids:
        raise OuterRefitError("nested membership lacks non-empty group ids")
    if any((not isinstance(group_id, str) for group_id in group_ids)):
        raise OuterRefitError("nested membership contains a non-string group id")
    if group_ids != sorted(group_ids) or len(group_ids) != len(set(group_ids)):
        raise OuterRefitError("nested membership group ids are not sorted and unique")
    try:
        rows = sorted(
            (row_id for group_id in group_ids for row_id in root_groups[group_id].row_ids)
        )
    except KeyError as exc:
        raise OuterRefitError("nested membership references an unknown group") from exc
    if len(rows) != len(set(rows)) or membership.get("row_count") != len(rows):
        raise OuterRefitError("nested membership row expansion differs")
    return rows


def _slice(
    membership: Mapping[str, Any],
    *,
    parent_split_sha256: object,
    root_groups: Mapping[str, GroupRecord],
) -> dict[str, object]:
    tokens = [_row_token(row_id) for row_id in _membership_rows(membership, root_groups)]
    return {
        "membership_sha256": _require_hash(
            membership.get("membership_sha256"), "membership_sha256"
        ),
        "parent_split_sha256": _require_hash(parent_split_sha256, "parent_split_sha256"),
        "row_count": len(tokens),
        "row_ids_sha256": canonical_sha256(tokens),
        "row_ids": tokens,
    }


def _outer_authorization(
    closure: Mapping[str, object], outer_repeat: int, outer_fold: int
) -> Mapping[str, object]:
    scopes = closure.get("outer_authorizations")
    if not isinstance(scopes, list):
        raise OuterRefitError("HPO closure lacks outer authorizations")
    matches = [
        scope
        for scope in scopes
        if isinstance(scope, Mapping)
        and scope.get("outer_repeat") == outer_repeat
        and (scope.get("outer_fold") == outer_fold)
    ]
    if len(matches) != 1:
        raise OuterRefitError("outer authorization is not unique")
    return matches[0]


def _validate_closed_hpo(
    hpo_plan: Mapping[str, object],
    closure: Mapping[str, object],
    ledger_entries: Sequence[Mapping[str, object]],
    attempt_receipts: Sequence[Mapping[str, object]],
    candidate_space: Mapping[str, object],
    nested_plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
) -> None:
    try:
        validate_hpo_plan(hpo_plan, candidate_space, nested_plan, group_manifest)
        validate_selection_authorization(
            hpo_plan,
            closure,
            ledger_entries,
            attempt_receipts=attempt_receipts,
            candidate_space=candidate_space,
            nested_plan=nested_plan,
            group_manifest=group_manifest,
        )
    except HpoPlanError as exc:
        raise OuterRefitError(str(exc)) from exc


def _planned_candidate_hashes(
    hpo_plan: Mapping[str, object],
    *,
    method: str,
    outer_repeat: int,
    outer_fold: int,
    candidate_id: str,
) -> set[str]:
    units = hpo_plan.get("units")
    if not isinstance(units, list):
        raise OuterRefitError("HPO plan unit inventory is missing")
    return {
        str(unit.get("candidate_sha256"))
        for unit in units
        if isinstance(unit, Mapping)
        and unit.get("method") == method
        and (unit.get("outer_repeat") == outer_repeat)
        and (unit.get("outer_fold") == outer_fold)
        and (unit.get("candidate_id") == candidate_id)
    }


def build_outer_selection_receipt(
    hpo_plan: Mapping[str, object],
    closure: Mapping[str, object],
    ledger_entries: Sequence[Mapping[str, object]],
    *,
    attempt_receipts: Sequence[Mapping[str, object]],
    candidate_space: Mapping[str, object],
    nested_plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
    method: str,
    outer_repeat: int,
    outer_fold: int,
    candidate_id: str,
    candidate_sha256: str,
    selection_decision_manifest_sha256: str,
) -> dict[str, object]:
    """Seal one externally selected, outer-local eligible HPO candidate.

    This function does not read metrics or choose a candidate.  The result
    selection engine remains responsible for the registered lexicographic
    rule; this receipt binds its immutable decision-manifest hash to the
    recomputed terminal-ledger authorization.
    """
    _validate_closed_hpo(
        hpo_plan,
        closure,
        ledger_entries,
        attempt_receipts,
        candidate_space,
        nested_plan,
        group_manifest,
    )
    if method not in MAIN_METHODS:
        raise OuterRefitError("outer selection method is not a main HPO method")
    repeat = _require_int(outer_repeat, "outer_repeat")
    fold = _require_int(outer_fold, "outer_fold")
    candidate_hash = _require_hash(candidate_sha256, "candidate_sha256")
    decision_hash = _require_hash(
        selection_decision_manifest_sha256, "selection_decision_manifest_sha256"
    )
    if not isinstance(candidate_id, str) or not candidate_id:
        raise OuterRefitError("candidate_id must be a non-empty string")
    outer = _outer_authorization(closure, repeat, fold)
    methods = outer.get("methods")
    if not isinstance(methods, Mapping):
        raise OuterRefitError("outer authorization lacks method inventories")
    inventory = methods.get(method)
    if not isinstance(inventory, Mapping) or candidate_id not in inventory.get(
        "eligible_candidate_ids", []
    ):
        raise OuterRefitError("selected candidate is not eligible in this outer scope")
    try:
        candidate = candidate_by_id(candidate_space, method, candidate_id)
    except CandidateSpaceError as exc:
        raise OuterRefitError(str(exc)) from exc
    if candidate.get("candidate_sha256") != candidate_hash:
        raise OuterRefitError("selected candidate hash differs from the frozen roster")
    if _planned_candidate_hashes(
        hpo_plan, method=method, outer_repeat=repeat, outer_fold=fold, candidate_id=candidate_id
    ) != {candidate_hash}:
        raise OuterRefitError("selected candidate is not exact in this outer HPO plan")
    receipt: dict[str, object] = {
        "schema": SELECTION_SCHEMA,
        "status": "OUTER_LOCAL_HPO_SELECTION_AUTHORIZED",
        "study_id": hpo_plan["study_id"],
        "hpo_plan_sha256": hpo_plan["hpo_plan_sha256"],
        "hpo_closure_sha256": closure["closure_sha256"],
        "terminal_ledger_sha256": closure["ledger_sha256"],
        "attempt_receipt_catalog_sha256": closure["attempt_receipt_catalog_sha256"],
        "outer_authorization_sha256": outer["outer_authorization_sha256"],
        "selection_decision_manifest_sha256": decision_hash,
        "method": method,
        "outer_repeat": repeat,
        "outer_fold": fold,
        "candidate_id": candidate_id,
        "candidate_sha256": candidate_hash,
    }
    receipt["selection_receipt_sha256"] = _artifact_hash(receipt, "selection_receipt_sha256")
    _validate_selection_shape(receipt)
    return receipt


def _validate_selection_shape(receipt: Mapping[str, object]) -> None:
    if not isinstance(receipt, Mapping) or set(receipt) != _SELECTION_FIELDS:
        raise OuterRefitError("selection receipt fields differ from the exact schema")
    if (
        receipt.get("schema") != SELECTION_SCHEMA
        or receipt.get("status") != "OUTER_LOCAL_HPO_SELECTION_AUTHORIZED"
    ):
        raise OuterRefitError("selection receipt schema or status differs")
    if not isinstance(receipt.get("study_id"), str) or not receipt.get("study_id"):
        raise OuterRefitError("selection receipt study_id is invalid")
    if receipt.get("method") not in MAIN_METHODS:
        raise OuterRefitError("selection receipt method is invalid")
    if not isinstance(receipt.get("candidate_id"), str) or not receipt.get("candidate_id"):
        raise OuterRefitError("selection receipt candidate_id is invalid")
    _require_int(receipt.get("outer_repeat"), "outer_repeat")
    _require_int(receipt.get("outer_fold"), "outer_fold")
    for field in (
        "hpo_plan_sha256",
        "hpo_closure_sha256",
        "terminal_ledger_sha256",
        "attempt_receipt_catalog_sha256",
        "outer_authorization_sha256",
        "selection_decision_manifest_sha256",
        "candidate_sha256",
        "selection_receipt_sha256",
    ):
        _require_hash(receipt.get(field), field)
    if not hmac.compare_digest(
        str(receipt["selection_receipt_sha256"]),
        _artifact_hash(receipt, "selection_receipt_sha256"),
    ):
        raise OuterRefitError("selection receipt hash differs")


def validate_outer_selection_receipt(
    receipt: Mapping[str, object],
    hpo_plan: Mapping[str, object],
    closure: Mapping[str, object],
    ledger_entries: Sequence[Mapping[str, object]],
    *,
    attempt_receipts: Sequence[Mapping[str, object]],
    candidate_space: Mapping[str, object],
    nested_plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
) -> None:
    """Rebuild a selection receipt from its exact closed-HPO upstreams."""
    _validate_selection_shape(receipt)
    expected = build_outer_selection_receipt(
        hpo_plan,
        closure,
        ledger_entries,
        attempt_receipts=attempt_receipts,
        candidate_space=candidate_space,
        nested_plan=nested_plan,
        group_manifest=group_manifest,
        method=str(receipt["method"]),
        outer_repeat=int(receipt["outer_repeat"]),
        outer_fold=int(receipt["outer_fold"]),
        candidate_id=str(receipt["candidate_id"]),
        candidate_sha256=str(receipt["candidate_sha256"]),
        selection_decision_manifest_sha256=str(receipt["selection_decision_manifest_sha256"]),
    )
    if dict(receipt) != expected:
        raise OuterRefitError("selection receipt differs from exact upstreams")


def _resolved_outer_fold(
    nested_plan: Mapping[str, Any], outer_repeat: int, outer_fold: int
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    try:
        repeat = nested_plan["repetitions"][outer_repeat]
        fold = repeat["outer_folds"][outer_fold]
        refit = fold["outer_refit"]
    except (KeyError, IndexError, TypeError) as exc:
        raise OuterRefitError("outer-refit scope does not resolve in nested plan") from exc
    if not all((isinstance(value, Mapping) for value in (repeat, fold, refit))):
        raise OuterRefitError("resolved outer-refit scope is malformed")
    return (repeat, fold, refit)


def _validate_or_rebuild_inheritance(
    inheritance: Mapping[str, object] | None,
    selection_receipt: Mapping[str, object],
    hpo_plan: Mapping[str, object],
    closure: Mapping[str, object],
    ledger_entries: Sequence[Mapping[str, object]],
    attempt_receipts: Sequence[Mapping[str, object]],
    candidate_space: Mapping[str, object],
    nested_plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
) -> tuple[str, Mapping[str, object] | None, Mapping[str, object]]:
    parent_method = str(selection_receipt["method"])
    try:
        candidate = candidate_by_id(
            candidate_space, parent_method, str(selection_receipt["candidate_id"])
        )
    except CandidateSpaceError as exc:
        raise OuterRefitError(str(exc)) from exc
    if inheritance is None:
        return (parent_method, None, candidate)
    if not isinstance(inheritance, Mapping):
        raise OuterRefitError("matched ablation inheritance must be a mapping")
    ablation_id = inheritance.get("ablation_id")
    if not isinstance(ablation_id, str) or ablation_id not in MATCHED_ABLATIONS:
        raise OuterRefitError("matched ablation identity is not preregistered")
    spec = MATCHED_ABLATIONS[ablation_id]
    if spec.get("parent_method") != parent_method:
        raise OuterRefitError("matched ablation parent differs from selection receipt")
    try:
        expected = inherit_matched_ablation(
            hpo_plan,
            closure,
            ledger_entries,
            attempt_receipts=attempt_receipts,
            candidate_space=candidate_space,
            nested_plan=nested_plan,
            group_manifest=group_manifest,
            ablation_id=ablation_id,
            outer_repeat=int(selection_receipt["outer_repeat"]),
            outer_fold=int(selection_receipt["outer_fold"]),
            parent_candidate_id=str(selection_receipt["candidate_id"]),
            parent_candidate_sha256=str(selection_receipt["candidate_sha256"]),
            parent_selection_receipt_sha256=str(selection_receipt["selection_receipt_sha256"]),
        )
    except HpoPlanError as exc:
        raise OuterRefitError(str(exc)) from exc
    if dict(inheritance) != expected:
        raise OuterRefitError("matched ablation inheritance differs from exact parent")
    if inheritance.get("method_identity") != ablation_id:
        raise OuterRefitError("matched ablation method identity differs")
    if (
        inheritance.get("independent_candidate_roster") is not False
        or inheritance.get("additional_hpo_units") != 0
    ):
        raise OuterRefitError("matched ablation attempted independent HPO")
    return (ablation_id, expected, candidate)


def _build_refit_payload(
    hpo_plan: Mapping[str, object],
    closure: Mapping[str, object],
    candidate_space: Mapping[str, object],
    nested_plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
    selection_receipt: Mapping[str, object],
    *,
    eval_seed: int,
    inheritance: Mapping[str, object] | None,
    method_identity: str,
    candidate: Mapping[str, object],
) -> dict[str, object]:
    repeat_index = int(selection_receipt["outer_repeat"])
    fold_index = int(selection_receipt["outer_fold"])
    repeat, fold, refit = _resolved_outer_fold(nested_plan, repeat_index, fold_index)
    roles = refit.get("roles")
    clients = refit.get("clients")
    role_partition = refit.get("role_partition")
    client_partition = refit.get("client_partition")
    if not all(
        (isinstance(value, Mapping) for value in (roles, clients, role_partition, client_partition))
    ):
        raise OuterRefitError("outer-refit roles or client partitions are missing")
    assert isinstance(roles, Mapping)
    assert isinstance(clients, Mapping)
    assert isinstance(role_partition, Mapping)
    assert isinstance(client_partition, Mapping)
    if set(clients) != set(CLIENT_NAMES):
        raise OuterRefitError("outer-refit client roster differs")
    root_groups = _root_group_map(group_manifest)
    role_parent = role_partition.get("split_sha256")
    client_parent = client_partition.get("split_sha256")
    private_clients = {
        client: _slice(clients[client], parent_split_sha256=client_parent, root_groups=root_groups)
        for client in CLIENT_NAMES
    }
    data_slices: dict[str, object] = {
        "private_clients": private_clients,
        "v_ctrl": _slice(roles["v_ctrl"], parent_split_sha256=role_parent, root_groups=root_groups),
        "v_sel": _slice(roles["v_sel"], parent_split_sha256=role_parent, root_groups=root_groups),
    }
    worker_tokens = [
        token for client in CLIENT_NAMES for token in private_clients[client]["row_ids"]
    ]
    worker_tokens.extend(data_slices["v_ctrl"]["row_ids"])
    worker_tokens.extend(data_slices["v_sel"]["row_ids"])
    if len(worker_tokens) != len(set(worker_tokens)):
        raise OuterRefitError("outer-refit capability rows are not exclusive")
    outer_test = fold.get("outer_test")
    if not isinstance(outer_test, Mapping):
        raise OuterRefitError("nested plan lacks outer-test membership")
    outer_test_rows = set(_membership_rows(outer_test, root_groups))
    if outer_test_rows & {_decode_row_token(token) for token in worker_tokens}:
        raise OuterRefitError("outer-test row entered outer-refit capability")
    plan_bindings = hpo_plan.get("bindings")
    design_binding = hpo_plan.get("nested_design_binding")
    if not isinstance(plan_bindings, Mapping) or not isinstance(design_binding, Mapping):
        raise OuterRefitError("HPO plan frozen bindings are missing")
    parameters = copy.deepcopy(candidate["parameters"])
    mechanism_switch = (
        None if inheritance is None else copy.deepcopy(inheritance["mechanism_switch"])
    )
    payload: dict[str, object] = {
        "schema": REFIT_CAPABILITY_SCHEMA,
        "status": "SEALED_OUTER_REFIT_OUTER_TEST_EXCLUDED",
        "study_id": hpo_plan["study_id"],
        "dataset_id": hpo_plan["dataset_id"],
        "bindings": {
            "hpo_plan_sha256": hpo_plan["hpo_plan_sha256"],
            "hpo_closure_sha256": closure["closure_sha256"],
            "terminal_ledger_sha256": closure["ledger_sha256"],
            "attempt_receipt_catalog_sha256": closure["attempt_receipt_catalog_sha256"],
            "outer_authorization_sha256": selection_receipt["outer_authorization_sha256"],
            "selection_receipt_sha256": selection_receipt["selection_receipt_sha256"],
            "selection_decision_manifest_sha256": selection_receipt[
                "selection_decision_manifest_sha256"
            ],
            "candidate_space_sha256": candidate_space["manifest_sha256"],
            "nested_plan_sha256": nested_plan["nested_plan_sha256"],
            "nested_freeze_binding_sha256": plan_bindings["nested_freeze_binding_sha256"],
            "source_sha256": plan_bindings["dataset_sha256"],
            "group_manifest_sha256": plan_bindings["group_manifest_sha256"],
            "row_to_group_sha256": plan_bindings["row_to_group_sha256"],
            "protocol_sha256": plan_bindings["protocol_sha256"],
            "implementation_contract_sha256": plan_bindings["implementation_contract_sha256"],
            "client_partition_policy": design_binding["client_partition_policy"],
            "client_partition_policy_sha256": design_binding["client_partition_policy_sha256"],
            "preprocessing_profile_name": plan_bindings["preprocessing_profile_name"],
            "preprocessing_profile_sha256": plan_bindings["preprocessing_profile_sha256"],
            "preprocessing_profile": copy.deepcopy(plan_bindings["preprocessing_profile"]),
        },
        "scope": {
            "method_identity": method_identity,
            "parent_method": selection_receipt["method"],
            "outer_repeat": repeat_index,
            "outer_fold": fold_index,
            "eval_seed": eval_seed,
        },
        "candidate": {
            "candidate_id": candidate["candidate_id"],
            "candidate_sha256": candidate["candidate_sha256"],
            "parameters_sha256": canonical_sha256(parameters),
            "parameters": parameters,
            "mechanism_switch": mechanism_switch,
        },
        "matched_ablation_inheritance": (
            None if inheritance is None else copy.deepcopy(dict(inheritance))
        ),
        "split_bindings": {
            "outer_partition_sha256": repeat["outer_partition"]["split_sha256"],
            "outer_train_membership_sha256": fold["outer_train"]["membership_sha256"],
            "role_partition_sha256": role_partition["split_sha256"],
            "private_membership_sha256": roles["private"]["membership_sha256"],
            "v_ctrl_membership_sha256": roles["v_ctrl"]["membership_sha256"],
            "v_sel_membership_sha256": roles["v_sel"]["membership_sha256"],
            "client_partition_sha256": client_partition["split_sha256"],
            "client_membership_sha256": {
                client: clients[client]["membership_sha256"] for client in CLIENT_NAMES
            },
        },
        "data_slices": data_slices,
        "worker_policy": copy.deepcopy(_WORKER_POLICY),
    }
    payload["capability_sha256"] = _artifact_hash(payload, "capability_sha256")
    return payload


def build_outer_refit_capability(
    hpo_plan: Mapping[str, object],
    closure: Mapping[str, object],
    ledger_entries: Sequence[Mapping[str, object]],
    *,
    attempt_receipts: Sequence[Mapping[str, object]],
    candidate_space: Mapping[str, object],
    nested_plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
    selection_receipt: Mapping[str, object],
    eval_seed: int,
    matched_ablation_inheritance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build one outer-test-free capability for a selected outer refit."""
    validate_outer_selection_receipt(
        selection_receipt,
        hpo_plan,
        closure,
        ledger_entries,
        attempt_receipts=attempt_receipts,
        candidate_space=candidate_space,
        nested_plan=nested_plan,
        group_manifest=group_manifest,
    )
    seed = _require_int(eval_seed, "eval_seed")
    method_identity, inheritance, candidate = _validate_or_rebuild_inheritance(
        matched_ablation_inheritance,
        selection_receipt,
        hpo_plan,
        closure,
        ledger_entries,
        attempt_receipts,
        candidate_space,
        nested_plan,
        group_manifest,
    )
    capability = _build_refit_payload(
        hpo_plan,
        closure,
        candidate_space,
        nested_plan,
        group_manifest,
        selection_receipt,
        eval_seed=seed,
        inheritance=inheritance,
        method_identity=method_identity,
        candidate=candidate,
    )
    _validate_capability_shape(capability)
    return capability


def _validate_slice_shape(value: object, name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != _SLICE_FIELDS:
        raise OuterRefitError(f"{name} slice fields differ from exact schema")
    for field in ("membership_sha256", "parent_split_sha256", "row_ids_sha256"):
        _require_hash(value.get(field), f"{name}.{field}")
    tokens = value.get("row_ids")
    if not isinstance(tokens, list) or not tokens:
        raise OuterRefitError(f"{name} row tokens must be non-empty")
    decoded = [_decode_row_token(token) for token in tokens]
    if decoded != sorted(decoded) or len(decoded) != len(set(decoded)):
        raise OuterRefitError(f"{name} row tokens are not sorted and unique")
    if value.get("row_count") != len(tokens):
        raise OuterRefitError(f"{name} row count differs")
    if value.get("row_ids_sha256") != canonical_sha256(tokens):
        raise OuterRefitError(f"{name} row-token hash differs")


def _validate_capability_shape(capability: Mapping[str, object]) -> None:
    if not isinstance(capability, Mapping) or set(capability) != _CAPABILITY_FIELDS:
        raise OuterRefitError("outer-refit capability fields differ from exact schema")
    if (
        capability.get("schema") != REFIT_CAPABILITY_SCHEMA
        or capability.get("status") != "SEALED_OUTER_REFIT_OUTER_TEST_EXCLUDED"
    ):
        raise OuterRefitError("outer-refit capability schema or status differs")
    try:
        assert_no_performance_fields(capability)
    except CandidateSpaceError as exc:
        raise OuterRefitError(str(exc)) from exc
    bindings = capability.get("bindings")
    scope = capability.get("scope")
    candidate = capability.get("candidate")
    splits = capability.get("split_bindings")
    slices = capability.get("data_slices")
    if not isinstance(bindings, Mapping) or set(bindings) != _CAPABILITY_BINDING_FIELDS:
        raise OuterRefitError("outer-refit bindings differ from exact schema")
    if not isinstance(scope, Mapping) or set(scope) != _SCOPE_FIELDS:
        raise OuterRefitError("outer-refit scope differs from exact schema")
    if not isinstance(candidate, Mapping) or set(candidate) != _CANDIDATE_FIELDS:
        raise OuterRefitError("outer-refit candidate differs from exact schema")
    if not isinstance(splits, Mapping) or set(splits) != _SPLIT_FIELDS:
        raise OuterRefitError("outer-refit split bindings differ from exact schema")
    if not isinstance(slices, Mapping) or set(slices) != _DATA_SLICE_FIELDS:
        raise OuterRefitError("outer-refit data slices differ from exact schema")
    if capability.get("worker_policy") != _WORKER_POLICY:
        raise OuterRefitError("outer-refit worker policy differs")
    for field, value in bindings.items():
        if field not in {
            "client_partition_policy",
            "preprocessing_profile_name",
            "preprocessing_profile",
        }:
            _require_hash(value, field)
    if not isinstance(bindings.get("client_partition_policy"), str) or not bindings.get(
        "client_partition_policy"
    ):
        raise OuterRefitError("client partition policy identity is invalid")
    if not isinstance(bindings.get("preprocessing_profile"), Mapping):
        raise OuterRefitError("preprocessing profile specification is missing")
    for field in ("outer_repeat", "outer_fold", "eval_seed"):
        _require_int(scope.get(field), field)
    if not isinstance(scope.get("method_identity"), str) or not isinstance(
        scope.get("parent_method"), str
    ):
        raise OuterRefitError("outer-refit method scope is invalid")
    if not isinstance(candidate.get("candidate_id"), str) or not isinstance(
        candidate.get("parameters"), Mapping
    ):
        raise OuterRefitError("outer-refit candidate identity or parameters are invalid")
    _require_hash(candidate.get("candidate_sha256"), "candidate_sha256")
    _require_hash(candidate.get("parameters_sha256"), "parameters_sha256")
    if candidate.get("parameters_sha256") != canonical_sha256(candidate.get("parameters")):
        raise OuterRefitError("outer-refit parameter hash differs")
    for field, value in splits.items():
        if field != "client_membership_sha256":
            _require_hash(value, field)
    client_membership = splits.get("client_membership_sha256")
    if not isinstance(client_membership, Mapping) or set(client_membership) != set(CLIENT_NAMES):
        raise OuterRefitError("outer-refit client membership bindings differ")
    for client in CLIENT_NAMES:
        _require_hash(client_membership[client], f"client_membership.{client}")
    private_clients = slices.get("private_clients")
    if not isinstance(private_clients, Mapping) or set(private_clients) != set(CLIENT_NAMES):
        raise OuterRefitError("outer-refit private client slices differ")
    all_tokens: list[str] = []
    for client in CLIENT_NAMES:
        _validate_slice_shape(private_clients[client], client)
        all_tokens.extend(private_clients[client]["row_ids"])
    for role in ("v_ctrl", "v_sel"):
        _validate_slice_shape(slices[role], role)
        all_tokens.extend(slices[role]["row_ids"])
    if len(all_tokens) != len(set(all_tokens)):
        raise OuterRefitError("outer-refit data slices are not row-exclusive")
    inheritance = capability.get("matched_ablation_inheritance")
    if inheritance is None:
        if scope.get("method_identity") != scope.get("parent_method"):
            raise OuterRefitError("main method capability has mismatched identity")
        if candidate.get("mechanism_switch") is not None:
            raise OuterRefitError("main method capability has an ablation switch")
    elif not isinstance(inheritance, Mapping):
        raise OuterRefitError("matched ablation inheritance is malformed")
    else:
        if scope.get("method_identity") != inheritance.get("ablation_id"):
            raise OuterRefitError("matched ablation identity differs from inheritance")
        if candidate.get("mechanism_switch") != inheritance.get("mechanism_switch"):
            raise OuterRefitError("matched ablation switch differs from inheritance")
    _require_hash(capability.get("capability_sha256"), "capability_sha256")
    if not hmac.compare_digest(
        str(capability["capability_sha256"]), _artifact_hash(capability, "capability_sha256")
    ):
        raise OuterRefitError("outer-refit capability hash differs")


def validate_outer_refit_capability(
    capability: Mapping[str, object],
    hpo_plan: Mapping[str, object],
    closure: Mapping[str, object],
    ledger_entries: Sequence[Mapping[str, object]],
    *,
    attempt_receipts: Sequence[Mapping[str, object]],
    candidate_space: Mapping[str, object],
    nested_plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
    selection_receipt: Mapping[str, object],
    matched_ablation_inheritance: Mapping[str, object] | None = None,
) -> None:
    """Rebuild a refit capability from every exact frozen upstream."""
    _validate_capability_shape(capability)
    expected = build_outer_refit_capability(
        hpo_plan,
        closure,
        ledger_entries,
        attempt_receipts=attempt_receipts,
        candidate_space=candidate_space,
        nested_plan=nested_plan,
        group_manifest=group_manifest,
        selection_receipt=selection_receipt,
        eval_seed=int(capability["scope"]["eval_seed"]),
        matched_ablation_inheritance=matched_ablation_inheritance,
    )
    if dict(capability) != expected:
        raise OuterRefitError("outer-refit capability differs from exact upstreams")


def _require_expected_capability_hash(
    capability: Mapping[str, object], expected_capability_sha256: str
) -> None:
    _validate_capability_shape(capability)
    expected = _require_hash(expected_capability_sha256, "expected_capability_sha256")
    if not hmac.compare_digest(str(capability["capability_sha256"]), expected):
        raise OuterRefitError("worker received an unexpected refit capability")


def materialize_outer_refit_row_ids(
    capability: Mapping[str, object], *, role: str, expected_capability_sha256: str
) -> tuple[int, ...]:
    """Materialize one allowed refit role; manual and outer-test IDs are absent."""
    if not isinstance(role, str):
        raise OuterRefitError("role must be a string")
    if role in {"outer_test", "outer-test", "test"}:
        raise ForbiddenOuterTestRefitAccess(
            "outer-refit capabilities cannot materialize outer-test rows"
        )
    if role not in {*CLIENT_NAMES, "v_ctrl", "v_sel"}:
        raise OuterRefitError("role is not present in an outer-refit capability")
    _require_expected_capability_hash(capability, expected_capability_sha256)
    slices = capability["data_slices"]
    assert isinstance(slices, Mapping)
    if role in CLIENT_NAMES:
        clients = slices["private_clients"]
        assert isinstance(clients, Mapping)
        value = clients[role]
    else:
        value = slices[role]
    assert isinstance(value, Mapping)
    return tuple((_decode_row_token(token) for token in value["row_ids"]))


def outer_refit_candidate_configuration(
    capability: Mapping[str, object], *, expected_capability_sha256: str
) -> dict[str, object]:
    """Return the selected parent configuration plus any registered switch."""
    _require_expected_capability_hash(capability, expected_capability_sha256)
    return {
        "method_identity": capability["scope"]["method_identity"],
        "parent_method": capability["scope"]["parent_method"],
        **copy.deepcopy(dict(capability["candidate"])),
    }


def _role_bindings(capability: Mapping[str, object]) -> dict[str, object]:
    splits = capability["split_bindings"]
    assert isinstance(splits, Mapping)
    return {
        "private_client_membership_sha256": copy.deepcopy(splits["client_membership_sha256"]),
        "v_ctrl_membership_sha256": splits["v_ctrl_membership_sha256"],
        "v_sel_membership_sha256": splits["v_sel_membership_sha256"],
    }


def _build_stage_manifest(
    capability: Mapping[str, object],
    *,
    stage: str,
    predecessor: Mapping[str, object] | None,
    output_artifact_manifest_sha256: str,
) -> dict[str, object]:
    if stage not in _STAGE_ORDER:
        raise OuterRefitError("unknown outer-refit stage")
    index = _STAGE_ORDER.index(stage)
    if index == 0:
        if predecessor is not None:
            raise OuterRefitError("model stage cannot have a predecessor")
        predecessor_binding = None
    else:
        if not isinstance(predecessor, Mapping):
            raise OuterRefitError("outer-refit stage predecessor is missing")
        _validate_stage_shape(predecessor)
        expected_stage = _STAGE_ORDER[index - 1]
        if predecessor.get("stage") != expected_stage:
            raise OuterRefitError("outer-refit stage predecessor is out of order")
        predecessor_binding = {
            "stage": expected_stage,
            "stage_manifest_sha256": predecessor["stage_manifest_sha256"],
        }
    output_hash = _require_hash(output_artifact_manifest_sha256, "output_artifact_manifest_sha256")
    bindings = capability["bindings"]
    candidate = capability["candidate"]
    assert isinstance(bindings, Mapping)
    assert isinstance(candidate, Mapping)
    manifest: dict[str, object] = {
        "schema": REFIT_STAGE_SCHEMA,
        "status": "CLOSED_IMMUTABLE_OUTPUT",
        "stage": stage,
        "bindings": {
            "refit_capability_sha256": capability["capability_sha256"],
            "hpo_plan_sha256": bindings["hpo_plan_sha256"],
            "nested_plan_sha256": bindings["nested_plan_sha256"],
            "source_sha256": bindings["source_sha256"],
            "group_manifest_sha256": bindings["group_manifest_sha256"],
            "preprocessing_profile_sha256": bindings["preprocessing_profile_sha256"],
            "candidate_sha256": candidate["candidate_sha256"],
            "selection_receipt_sha256": bindings["selection_receipt_sha256"],
        },
        "scope": copy.deepcopy(capability["scope"]),
        "role_bindings": _role_bindings(capability),
        "predecessor": predecessor_binding,
        "output_artifact_manifest_sha256": output_hash,
    }
    manifest["stage_manifest_sha256"] = _artifact_hash(manifest, "stage_manifest_sha256")
    _validate_stage_shape(manifest)
    return manifest


def _validate_stage_shape(manifest: Mapping[str, object]) -> None:
    if not isinstance(manifest, Mapping) or set(manifest) != _STAGE_FIELDS:
        raise OuterRefitError("outer-refit stage fields differ from exact schema")
    if (
        manifest.get("schema") != REFIT_STAGE_SCHEMA
        or manifest.get("status") != "CLOSED_IMMUTABLE_OUTPUT"
        or manifest.get("stage") not in _STAGE_ORDER
    ):
        raise OuterRefitError("outer-refit stage schema, status, or kind differs")
    bindings = manifest.get("bindings")
    scope = manifest.get("scope")
    roles = manifest.get("role_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != _STAGE_BINDING_FIELDS:
        raise OuterRefitError("outer-refit stage bindings differ")
    if not isinstance(scope, Mapping) or set(scope) != _SCOPE_FIELDS:
        raise OuterRefitError("outer-refit stage scope differs")
    if not isinstance(roles, Mapping) or set(roles) != _ROLE_BINDING_FIELDS:
        raise OuterRefitError("outer-refit stage role bindings differ")
    for field, value in bindings.items():
        _require_hash(value, field)
    clients = roles.get("private_client_membership_sha256")
    if not isinstance(clients, Mapping) or set(clients) != set(CLIENT_NAMES):
        raise OuterRefitError("outer-refit stage private client bindings differ")
    for client in CLIENT_NAMES:
        _require_hash(clients[client], f"role_bindings.{client}")
    for field in ("v_ctrl_membership_sha256", "v_sel_membership_sha256"):
        _require_hash(roles.get(field), field)
    predecessor = manifest.get("predecessor")
    stage_index = _STAGE_ORDER.index(str(manifest["stage"]))
    if stage_index == 0:
        if predecessor is not None:
            raise OuterRefitError("model stage predecessor must be null")
    elif (
        not isinstance(predecessor, Mapping)
        or set(predecessor) != {"stage", "stage_manifest_sha256"}
        or predecessor.get("stage") != _STAGE_ORDER[stage_index - 1]
    ):
        raise OuterRefitError("outer-refit stage predecessor binding differs")
    else:
        _require_hash(predecessor.get("stage_manifest_sha256"), "predecessor.stage_manifest_sha256")
    _require_hash(
        manifest.get("output_artifact_manifest_sha256"), "output_artifact_manifest_sha256"
    )
    _require_hash(manifest.get("stage_manifest_sha256"), "stage_manifest_sha256")
    if not hmac.compare_digest(
        str(manifest["stage_manifest_sha256"]), _artifact_hash(manifest, "stage_manifest_sha256")
    ):
        raise OuterRefitError("outer-refit stage manifest hash differs")


def seal_outer_refit_model_manifest(
    capability: Mapping[str, object],
    *,
    expected_capability_sha256: str,
    model_artifact_manifest_sha256: str,
) -> dict[str, object]:
    _require_expected_capability_hash(capability, expected_capability_sha256)
    return _build_stage_manifest(
        capability,
        stage="refit_model",
        predecessor=None,
        output_artifact_manifest_sha256=model_artifact_manifest_sha256,
    )


def seal_outer_refit_prediction_manifest(
    capability: Mapping[str, object],
    model_manifest: Mapping[str, object],
    *,
    expected_capability_sha256: str,
    prediction_artifact_manifest_sha256: str,
) -> dict[str, object]:
    _require_expected_capability_hash(capability, expected_capability_sha256)
    return _build_stage_manifest(
        capability,
        stage="refit_validation_predictions",
        predecessor=model_manifest,
        output_artifact_manifest_sha256=prediction_artifact_manifest_sha256,
    )


def seal_outer_refit_threshold_manifest(
    capability: Mapping[str, object],
    prediction_manifest: Mapping[str, object],
    *,
    expected_capability_sha256: str,
    threshold_artifact_manifest_sha256: str,
) -> dict[str, object]:
    _require_expected_capability_hash(capability, expected_capability_sha256)
    return _build_stage_manifest(
        capability,
        stage="decision_threshold",
        predecessor=prediction_manifest,
        output_artifact_manifest_sha256=threshold_artifact_manifest_sha256,
    )


def validate_outer_refit_artifact_chain(
    capability: Mapping[str, object],
    model_manifest: Mapping[str, object],
    prediction_manifest: Mapping[str, object],
    threshold_manifest: Mapping[str, object],
    *,
    expected_capability_sha256: str,
) -> None:
    """Rebuild the ordered model/prediction/threshold closure chain."""
    _require_expected_capability_hash(capability, expected_capability_sha256)
    expected_model = seal_outer_refit_model_manifest(
        capability,
        expected_capability_sha256=expected_capability_sha256,
        model_artifact_manifest_sha256=str(model_manifest.get("output_artifact_manifest_sha256")),
    )
    if dict(model_manifest) != expected_model:
        raise OuterRefitError("refit model manifest differs from exact capability")
    expected_prediction = seal_outer_refit_prediction_manifest(
        capability,
        expected_model,
        expected_capability_sha256=expected_capability_sha256,
        prediction_artifact_manifest_sha256=str(
            prediction_manifest.get("output_artifact_manifest_sha256")
        ),
    )
    if dict(prediction_manifest) != expected_prediction:
        raise OuterRefitError("refit prediction manifest differs from exact chain")
    expected_threshold = seal_outer_refit_threshold_manifest(
        capability,
        expected_prediction,
        expected_capability_sha256=expected_capability_sha256,
        threshold_artifact_manifest_sha256=str(
            threshold_manifest.get("output_artifact_manifest_sha256")
        ),
    )
    if dict(threshold_manifest) != expected_threshold:
        raise OuterRefitError("refit threshold manifest differs from exact chain")


def _build_gate_manifest(
    capability: Mapping[str, object],
    model_manifest: Mapping[str, object],
    prediction_manifest: Mapping[str, object],
    threshold_manifest: Mapping[str, object],
    nested_plan: Mapping[str, Any],
    *,
    one_time_access_authorization_sha256: str,
) -> dict[str, object]:
    scope = capability["scope"]
    assert isinstance(scope, Mapping)
    _, fold, _ = _resolved_outer_fold(
        nested_plan, int(scope["outer_repeat"]), int(scope["outer_fold"])
    )
    outer_test = fold.get("outer_test")
    if not isinstance(outer_test, Mapping):
        raise OuterRefitError("nested plan lacks outer-test membership")
    membership_hash = _require_hash(
        outer_test.get("membership_sha256"), "outer_test.membership_sha256"
    )
    bindings = capability["bindings"]
    candidate = capability["candidate"]
    assert isinstance(bindings, Mapping)
    assert isinstance(candidate, Mapping)
    manifest: dict[str, object] = {
        "schema": OUTER_TEST_GATE_SCHEMA,
        "status": "ISSUED_UNOPENED",
        "bindings": {
            "refit_capability_sha256": capability["capability_sha256"],
            "model_stage_manifest_sha256": model_manifest["stage_manifest_sha256"],
            "prediction_stage_manifest_sha256": prediction_manifest["stage_manifest_sha256"],
            "threshold_stage_manifest_sha256": threshold_manifest["stage_manifest_sha256"],
            "hpo_plan_sha256": bindings["hpo_plan_sha256"],
            "nested_plan_sha256": bindings["nested_plan_sha256"],
            "source_sha256": bindings["source_sha256"],
            "group_manifest_sha256": bindings["group_manifest_sha256"],
            "preprocessing_profile_sha256": bindings["preprocessing_profile_sha256"],
            "candidate_sha256": candidate["candidate_sha256"],
            "selection_receipt_sha256": bindings["selection_receipt_sha256"],
            "one_time_access_authorization_sha256": _require_hash(
                one_time_access_authorization_sha256, "one_time_access_authorization_sha256"
            ),
        },
        "scope": copy.deepcopy(dict(scope)),
        "outer_test_seal": {"membership_sha256": membership_hash},
        "access_policy": copy.deepcopy(_ACCESS_POLICY),
    }
    manifest["gate_sha256"] = _artifact_hash(manifest, "gate_sha256")
    _validate_gate_manifest_shape(manifest)
    return manifest


def _validate_gate_manifest_shape(manifest: Mapping[str, object]) -> None:
    if not isinstance(manifest, Mapping) or set(manifest) != _GATE_FIELDS:
        raise OuterRefitError("outer-test gate fields differ from exact schema")
    if (
        manifest.get("schema") != OUTER_TEST_GATE_SCHEMA
        or manifest.get("status") != "ISSUED_UNOPENED"
    ):
        raise OuterRefitError("outer-test gate schema or status differs")
    bindings = manifest.get("bindings")
    scope = manifest.get("scope")
    seal = manifest.get("outer_test_seal")
    if not isinstance(bindings, Mapping) or set(bindings) != _GATE_BINDING_FIELDS:
        raise OuterRefitError("outer-test gate bindings differ")
    if not isinstance(scope, Mapping) or set(scope) != _SCOPE_FIELDS:
        raise OuterRefitError("outer-test gate scope differs")
    if not isinstance(seal, Mapping) or set(seal) != _OUTER_TEST_SEAL_FIELDS:
        raise OuterRefitError("outer-test gate seal differs")
    if manifest.get("access_policy") != _ACCESS_POLICY:
        raise OuterRefitError("outer-test access policy differs")
    for field, value in bindings.items():
        _require_hash(value, field)
    _require_hash(seal.get("membership_sha256"), "outer_test.membership_sha256")
    _require_hash(manifest.get("gate_sha256"), "gate_sha256")
    if not hmac.compare_digest(
        str(manifest["gate_sha256"]), _artifact_hash(manifest, "gate_sha256")
    ):
        raise OuterRefitError("outer-test gate hash differs")


_GATE_ISSUER_TOKEN = object()
_ISSUED_GATE_HASHES: set[str] = set()
_ISSUED_GATE_LOCK = threading.Lock()


class OuterTestGate:
    """Opaque in-process state for one issued outer-test gate."""

    __slots__ = ("_manifest", "_consumed", "_attempt_ids", "_receipts", "_lock")

    def __init__(self, manifest: Mapping[str, object], *, _issuer_token: object) -> None:
        if _issuer_token is not _GATE_ISSUER_TOKEN:
            raise OuterTestGateError("outer-test gates must be created by the issuer")
        _validate_gate_manifest_shape(manifest)
        self._manifest = copy.deepcopy(dict(manifest))
        self._consumed = False
        self._attempt_ids: set[str] = set()
        self._receipts: list[dict[str, object]] = []
        self._lock = threading.Lock()

    @property
    def consumed(self) -> bool:
        return self._consumed

    def public_manifest(self) -> dict[str, object]:
        return copy.deepcopy(self._manifest)

    def access_attempt_receipts(self) -> tuple[dict[str, object], ...]:
        return tuple((copy.deepcopy(receipt) for receipt in self._receipts))


@dataclass(frozen=True)
class OuterTestOpenResult:
    """Rows released by one gate opening plus its immutable receipt."""

    row_ids: tuple[int, ...]
    access_receipt: Mapping[str, object]


def validate_outer_test_gate(
    gate_manifest: Mapping[str, object],
    capability: Mapping[str, object],
    model_manifest: Mapping[str, object],
    prediction_manifest: Mapping[str, object],
    threshold_manifest: Mapping[str, object],
    hpo_plan: Mapping[str, object],
    closure: Mapping[str, object],
    ledger_entries: Sequence[Mapping[str, object]],
    *,
    attempt_receipts: Sequence[Mapping[str, object]],
    candidate_space: Mapping[str, object],
    nested_plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
    selection_receipt: Mapping[str, object],
    one_time_access_authorization_sha256: str,
    matched_ablation_inheritance: Mapping[str, object] | None = None,
) -> None:
    """Rebuild a gate from exact upstreams without opening outer test."""
    validate_outer_refit_capability(
        capability,
        hpo_plan,
        closure,
        ledger_entries,
        attempt_receipts=attempt_receipts,
        candidate_space=candidate_space,
        nested_plan=nested_plan,
        group_manifest=group_manifest,
        selection_receipt=selection_receipt,
        matched_ablation_inheritance=matched_ablation_inheritance,
    )
    validate_outer_refit_artifact_chain(
        capability,
        model_manifest,
        prediction_manifest,
        threshold_manifest,
        expected_capability_sha256=str(capability["capability_sha256"]),
    )
    expected = _build_gate_manifest(
        capability,
        model_manifest,
        prediction_manifest,
        threshold_manifest,
        nested_plan,
        one_time_access_authorization_sha256=one_time_access_authorization_sha256,
    )
    if dict(gate_manifest) != expected:
        raise OuterRefitError("outer-test gate differs from exact closed refit")


def issue_outer_test_gate(
    capability: Mapping[str, object],
    model_manifest: Mapping[str, object],
    prediction_manifest: Mapping[str, object],
    threshold_manifest: Mapping[str, object],
    hpo_plan: Mapping[str, object],
    closure: Mapping[str, object],
    ledger_entries: Sequence[Mapping[str, object]],
    *,
    attempt_receipts: Sequence[Mapping[str, object]],
    candidate_space: Mapping[str, object],
    nested_plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
    selection_receipt: Mapping[str, object],
    one_time_access_authorization_sha256: str,
    matched_ablation_inheritance: Mapping[str, object] | None = None,
) -> OuterTestGate:
    """Issue an opaque gate only after the complete refit chain is closed."""
    manifest = _build_gate_manifest(
        capability,
        model_manifest,
        prediction_manifest,
        threshold_manifest,
        nested_plan,
        one_time_access_authorization_sha256=one_time_access_authorization_sha256,
    )
    validate_outer_test_gate(
        manifest,
        capability,
        model_manifest,
        prediction_manifest,
        threshold_manifest,
        hpo_plan,
        closure,
        ledger_entries,
        attempt_receipts=attempt_receipts,
        candidate_space=candidate_space,
        nested_plan=nested_plan,
        group_manifest=group_manifest,
        selection_receipt=selection_receipt,
        one_time_access_authorization_sha256=one_time_access_authorization_sha256,
        matched_ablation_inheritance=matched_ablation_inheritance,
    )
    gate_hash = str(manifest["gate_sha256"])
    with _ISSUED_GATE_LOCK:
        if gate_hash in _ISSUED_GATE_HASHES:
            raise OuterTestGateError("identical one-time outer-test gate was already issued")
        _ISSUED_GATE_HASHES.add(gate_hash)
    return OuterTestGate(manifest, _issuer_token=_GATE_ISSUER_TOKEN)


def _access_receipt(
    gate: OuterTestGate,
    *,
    access_attempt_id: str,
    outcome: str,
    failure_code: str | None,
    failure_incident_sha256: str | None,
    outer_test_row_ids_sha256: str | None,
    rows_released: bool,
) -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema": OUTER_TEST_ACCESS_SCHEMA,
        "status": "RECORDED",
        "gate_sha256": gate._manifest["gate_sha256"],
        "access_attempt_id": access_attempt_id,
        "scope": copy.deepcopy(gate._manifest["scope"]),
        "outcome": outcome,
        "failure_code": failure_code,
        "failure_incident_sha256": failure_incident_sha256,
        "outer_test_row_ids_sha256": outer_test_row_ids_sha256,
        "rows_released": rows_released,
    }
    receipt["access_receipt_sha256"] = _artifact_hash(receipt, "access_receipt_sha256")
    validate_outer_test_access_receipt(receipt, gate._manifest)
    gate._receipts.append(copy.deepcopy(receipt))
    gate._attempt_ids.add(access_attempt_id)
    return receipt


def validate_outer_test_access_receipt(
    receipt: Mapping[str, object], gate_manifest: Mapping[str, object]
) -> None:
    """Validate a successful, failed, or rejected gate-access receipt."""
    _validate_gate_manifest_shape(gate_manifest)
    if not isinstance(receipt, Mapping) or set(receipt) != _ACCESS_FIELDS:
        raise OuterRefitError("outer-test access receipt fields differ")
    if (
        receipt.get("schema") != OUTER_TEST_ACCESS_SCHEMA
        or receipt.get("status") != "RECORDED"
        or receipt.get("gate_sha256") != gate_manifest.get("gate_sha256")
        or (receipt.get("scope") != gate_manifest.get("scope"))
    ):
        raise OuterRefitError("outer-test access receipt binding differs")
    attempt_id = receipt.get("access_attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id.strip():
        raise OuterRefitError("outer-test access attempt id is invalid")
    outcome = receipt.get("outcome")
    failure_code = receipt.get("failure_code")
    incident = receipt.get("failure_incident_sha256")
    row_hash = receipt.get("outer_test_row_ids_sha256")
    if outcome == "opened":
        if (
            failure_code is not None
            or incident is not None
            or receipt.get("rows_released") is not True
        ):
            raise OuterRefitError("successful outer-test receipt branch differs")
        _require_hash(row_hash, "outer_test_row_ids_sha256")
    elif outcome == "failed":
        if (
            failure_code not in _FAILED_OPEN_CODES
            or receipt.get("rows_released") is not False
            or row_hash is not None
        ):
            raise OuterRefitError("failed outer-test receipt branch differs")
        _require_hash(incident, "failure_incident_sha256")
    elif outcome == "rejected":
        if (
            failure_code not in _REJECT_FAILURE_CODES
            or receipt.get("rows_released") is not False
            or row_hash is not None
            or (incident is not None)
        ):
            raise OuterRefitError("rejected outer-test receipt branch differs")
    else:
        raise OuterRefitError("outer-test access receipt outcome is invalid")
    _require_hash(receipt.get("access_receipt_sha256"), "access_receipt_sha256")
    if not hmac.compare_digest(
        str(receipt["access_receipt_sha256"]), _artifact_hash(receipt, "access_receipt_sha256")
    ):
        raise OuterRefitError("outer-test access receipt hash differs")


def _reject_access(
    gate: OuterTestGate, access_attempt_id: str, failure_code: str, message: str
) -> None:
    receipt = _access_receipt(
        gate,
        access_attempt_id=access_attempt_id,
        outcome="rejected",
        failure_code=failure_code,
        failure_incident_sha256=None,
        outer_test_row_ids_sha256=None,
        rows_released=False,
    )
    raise OuterTestGateError(message, attempt_receipt=receipt)


def _open_outer_test_once_locked(
    gate: OuterTestGate,
    hpo_plan: Mapping[str, object],
    candidate_space: Mapping[str, object],
    nested_plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
    *,
    expected_gate_sha256: str,
    outer_repeat: int,
    outer_fold: int,
    eval_seed: int,
    access_attempt_id: str,
    loader_failure_incident_sha256: str | None = None,
) -> OuterTestOpenResult:
    """Open one exact outer-test fold once, recording success or failure.

    No row IDs are accepted from the caller.  An explicit loader incident
    consumes the gate and records a failed open without releasing rows.
    """
    if not isinstance(gate, OuterTestGate):
        raise OuterTestGateError("outer-test access requires an issued gate object")
    _validate_gate_manifest_shape(gate._manifest)
    expected_gate = _require_hash(expected_gate_sha256, "expected_gate_sha256")
    if not hmac.compare_digest(str(gate._manifest["gate_sha256"]), expected_gate):
        raise OuterTestGateError("caller expected a different outer-test gate")
    if not isinstance(access_attempt_id, str) or not access_attempt_id.strip():
        raise OuterTestGateError("access_attempt_id must be a non-empty string")
    if access_attempt_id in gate._attempt_ids:
        _reject_access(
            gate,
            access_attempt_id,
            "duplicate_access_attempt_id",
            "outer-test access attempt id was replayed",
        )
    if gate._consumed:
        _reject_access(
            gate,
            access_attempt_id,
            "replay_or_second_open",
            "outer-test gate has already been consumed",
        )
    requested_repeat = _require_int(outer_repeat, "outer_repeat")
    requested_fold = _require_int(outer_fold, "outer_fold")
    requested_seed = _require_int(eval_seed, "eval_seed")
    scope = gate._manifest["scope"]
    assert isinstance(scope, Mapping)
    if requested_repeat != scope.get("outer_repeat") or requested_fold != scope.get("outer_fold"):
        _reject_access(
            gate,
            access_attempt_id,
            "cross_outer_scope",
            "outer-test gate cannot cross repeat or fold",
        )
    if requested_seed != scope.get("eval_seed"):
        _reject_access(
            gate,
            access_attempt_id,
            "cross_eval_seed",
            "outer-test gate cannot cross evaluation seed",
        )
    bindings = gate._manifest["bindings"]
    assert isinstance(bindings, Mapping)
    if hpo_plan.get("hpo_plan_sha256") != bindings.get("hpo_plan_sha256"):
        _reject_access(
            gate, access_attempt_id, "cross_hpo_plan", "outer-test gate cannot cross HPO plans"
        )
    if nested_plan.get("nested_plan_sha256") != bindings.get("nested_plan_sha256"):
        _reject_access(
            gate,
            access_attempt_id,
            "cross_nested_plan",
            "outer-test gate cannot cross nested plans",
        )
    if group_manifest.get("group_manifest_sha256") != bindings.get("group_manifest_sha256"):
        _reject_access(
            gate,
            access_attempt_id,
            "cross_group_manifest",
            "outer-test gate cannot cross group manifests",
        )
    try:
        validate_hpo_plan(hpo_plan, candidate_space, nested_plan, group_manifest)
    except HpoPlanError:
        _reject_access(
            gate,
            access_attempt_id,
            "upstream_validation_failure",
            "outer-test gate upstream validation failed",
        )
    _, fold, _ = _resolved_outer_fold(nested_plan, requested_repeat, requested_fold)
    outer_test = fold.get("outer_test")
    if (
        not isinstance(outer_test, Mapping)
        or outer_test.get("membership_sha256")
        != gate._manifest["outer_test_seal"]["membership_sha256"]
    ):
        _reject_access(
            gate,
            access_attempt_id,
            "upstream_validation_failure",
            "outer-test membership seal differs",
        )
    root_groups = _root_group_map(group_manifest)
    if loader_failure_incident_sha256 is not None:
        incident_hash = _require_hash(
            loader_failure_incident_sha256, "loader_failure_incident_sha256"
        )
        gate._consumed = True
        receipt = _access_receipt(
            gate,
            access_attempt_id=access_attempt_id,
            outcome="failed",
            failure_code="outer_test_loader_failure",
            failure_incident_sha256=incident_hash,
            outer_test_row_ids_sha256=None,
            rows_released=False,
        )
        return OuterTestOpenResult(row_ids=(), access_receipt=receipt)
    gate._consumed = True
    try:
        rows = tuple(_membership_rows(outer_test, root_groups))
    except Exception as exc:
        incident_hash = canonical_sha256(
            {
                "failure_class": "internal_materialization_failure",
                "exception_type": type(exc).__name__,
                "gate_sha256": gate._manifest["gate_sha256"],
            }
        )
        receipt = _access_receipt(
            gate,
            access_attempt_id=access_attempt_id,
            outcome="failed",
            failure_code="internal_materialization_failure",
            failure_incident_sha256=incident_hash,
            outer_test_row_ids_sha256=None,
            rows_released=False,
        )
        raise OuterTestGateError(
            "outer-test materialization failed after gate consumption", attempt_receipt=receipt
        ) from exc
    row_tokens = [_row_token(row_id) for row_id in rows]
    receipt = _access_receipt(
        gate,
        access_attempt_id=access_attempt_id,
        outcome="opened",
        failure_code=None,
        failure_incident_sha256=None,
        outer_test_row_ids_sha256=canonical_sha256(row_tokens),
        rows_released=True,
    )
    return OuterTestOpenResult(row_ids=rows, access_receipt=receipt)


def open_outer_test_once(
    gate: OuterTestGate,
    hpo_plan: Mapping[str, object],
    candidate_space: Mapping[str, object],
    nested_plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
    *,
    expected_gate_sha256: str,
    outer_repeat: int,
    outer_fold: int,
    eval_seed: int,
    access_attempt_id: str,
    loader_failure_incident_sha256: str | None = None,
) -> OuterTestOpenResult:
    """Atomically open one exact outer-test fold at most once.

    Cross-scope and replay attempts are receipt-bearing rejections.  No manual
    row-ID input exists.  A recorded loader failure consumes the same one-time
    authorization as a successful open.
    """
    if not isinstance(gate, OuterTestGate):
        raise OuterTestGateError("outer-test access requires an issued gate object")
    with gate._lock:
        return _open_outer_test_once_locked(
            gate,
            hpo_plan,
            candidate_space,
            nested_plan,
            group_manifest,
            expected_gate_sha256=expected_gate_sha256,
            outer_repeat=outer_repeat,
            outer_fold=outer_fold,
            eval_seed=eval_seed,
            access_attempt_id=access_attempt_id,
            loader_failure_incident_sha256=loader_failure_incident_sha256,
        )
