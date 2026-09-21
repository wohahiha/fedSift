"""Result-blind nested split planning with fail-closed HPO data access.

This module plans the fixed FedSift structure:

* three repeats of five outer folds;
* three inner folds inside every outer-training set;
* group-safe 80/10/10 private, ``V_ctrl`` and ``V_sel`` roles; and
* five group-safe private clients under a fixed label-skew protocol.

Every partition is produced by :func:`generate_balanced_partition`.  A derived
group manifest is used for a subset because the partitioner intentionally
requires its input rows to be contiguous and exhaustive.  Original group IDs
never change; the derived manifest is fingerprinted together with the root
manifest, parent split and selected-group membership hashes.

The client policy is registered independently of performance: positive-label
shares are 10/15/20/25/30 percent and negative-label shares are reversed after
a domain-separated seed permutation of client identities.  The target
label-distribution total variation is therefore 0.30.  Changing that policy
must change the study and nested-plan fingerprints.
"""

from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import copy
import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from .balanced_group_split import BalanceConstraints, generate_balanced_partition
from .client_partition import (
    CLIENT_NAMES,
    MIN_NEGATIVE_PER_CLIENT,
    MIN_POSITIVE_PER_CLIENT,
    MIN_ROWS_PER_CLIENT,
    NEGATIVE_SHARE_PERCENT,
    POLICY_NAME as CLIENT_PARTITION_POLICY,
    POSITIVE_SHARE_PERCENT,
    ClientPartitionError,
    client_partition_policy_fingerprint,
    generate_client_partition,
    validate_client_partition,
)
from .candidate_space import CandidateSpaceError, assert_no_performance_fields
from .group_manifest import GroupRecord, group_manifest_fingerprint, group_records


class NestedPlanError(RuntimeError):
    """Raised when a nested plan is malformed, inconsistent or infeasible."""


class ForbiddenOuterTestAccess(NestedPlanError):
    """Raised before an HPO path can materialize an outer-test row ID."""


@dataclass(frozen=True)
class HPODataSlice:
    """Auditable row-ID capability for one allowed HPO role.

    The object deliberately contains no feature or label arrays.  A downstream
    loader may use ``row_ids`` only after checking the attached source, group,
    split and full-plan fingerprints.
    """

    study_id: str
    source_sha256: str
    group_manifest_sha256: str
    nested_plan_sha256: str
    outer_repeat: int
    outer_fold: int
    inner_fold: int
    role: str
    parent_split_sha256: str
    membership_sha256: str
    group_ids: tuple[str, ...]
    row_ids: tuple[int, ...]


OUTER_REPEATS = 3
OUTER_FOLDS = 5
INNER_FOLDS = 3
ROLE_NAMES = ("private", "v_ctrl", "v_sel")
ROLE_WEIGHTS = (0.8, 0.1, 0.1)
CLIENT_COUNT = 5
_VALIDATED_PLAN_CACHE: set[tuple[str, str]] = set()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_lines(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any((character not in "0123456789abcdef" for character in value))
    ):
        raise NestedPlanError(f"{name} must be a lowercase SHA-256")
    return value


def nested_plan_fingerprint(plan: Mapping[str, Any]) -> str:
    """Fingerprint every plan field except the fingerprint itself."""
    payload = {str(key): value for (key, value) in plan.items() if key != "nested_plan_sha256"}
    return _sha256_json(payload)


def _root_group_map(group_manifest: Mapping[str, Any]) -> dict[str, GroupRecord]:
    return {record.group_id: record for record in group_records(group_manifest)}


def _membership_sha256(group_ids: Sequence[str]) -> str:
    ordered = sorted(group_ids)
    if len(ordered) != len(set(ordered)):
        raise NestedPlanError("membership contains a duplicate group")
    return _sha256_json(ordered)


def _membership(group_ids: Sequence[str], root_groups: Mapping[str, GroupRecord]) -> dict[str, Any]:
    ordered = sorted(group_ids)
    if len(ordered) != len(set(ordered)):
        raise NestedPlanError("membership contains a duplicate group")
    unknown = sorted(set(ordered) - set(root_groups))
    if unknown:
        raise NestedPlanError(f"membership contains unknown groups: {unknown[:3]}")
    records = [root_groups[group_id] for group_id in ordered]
    row_count = sum((record.n for record in records))
    positive = sum((record.positive for record in records))
    negative = sum((record.negative for record in records))
    return {
        "group_ids": ordered,
        "group_count": len(ordered),
        "row_count": row_count,
        "positive": positive,
        "negative": negative,
        "prevalence": positive / row_count if row_count else None,
        "membership_sha256": _membership_sha256(ordered),
    }


def _root_binding(group_manifest: Mapping[str, Any]) -> dict[str, Any]:
    records = group_records(group_manifest)
    binding = {
        "dataset": group_manifest.get("dataset"),
        "source_sha256": _require_sha256(group_manifest.get("source_sha256"), "source hash"),
        "group_manifest_sha256": _require_sha256(
            group_manifest.get("group_manifest_sha256"), "group-manifest hash"
        ),
        "row_to_group_sha256": _require_sha256(
            group_manifest.get("row_to_group_sha256"), "row-to-group hash"
        ),
        "row_count": sum((record.n for record in records)),
        "group_count": len(records),
    }
    binding["binding_sha256"] = _sha256_json(binding)
    return binding


def _derived_group_manifest(
    root_manifest: Mapping[str, Any],
    group_ids: Sequence[str],
    *,
    scope: str,
    parent_split_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a valid subset manifest while preserving original group IDs."""
    if not isinstance(scope, str) or not scope:
        raise NestedPlanError("derived-manifest scope must be non-empty")
    _require_sha256(parent_split_sha256, "parent split hash")
    root_groups = _root_group_map(root_manifest)
    ordered_ids = sorted(group_ids)
    if not ordered_ids or len(ordered_ids) != len(set(ordered_ids)):
        raise NestedPlanError("derived manifest requires unique non-empty groups")
    if not set(ordered_ids) <= set(root_groups):
        raise NestedPlanError("derived manifest contains an unknown root group")
    local_cursor = 0
    serialized_groups: list[dict[str, Any]] = []
    for group_id in ordered_ids:
        record = root_groups[group_id]
        local_rows = list(range(local_cursor, local_cursor + record.n))
        local_cursor += record.n
        serialized_groups.append(
            {
                "group_id": record.group_id,
                "row_ids": local_rows,
                "n": record.n,
                "positive": record.positive,
                "negative": record.negative,
            }
        )
    assignment_payload = "\n".join(
        (f"{row_id}:{item['group_id']}" for item in serialized_groups for row_id in item["row_ids"])
    ).encode("utf-8")
    membership_sha256 = _membership_sha256(ordered_ids)
    derivation = {
        "schema": _identity("derived_group_manifest_binding"),
        "scope": scope,
        "root_group_manifest_sha256": _require_sha256(
            root_manifest.get("group_manifest_sha256"), "root group-manifest hash"
        ),
        "root_row_to_group_sha256": _require_sha256(
            root_manifest.get("row_to_group_sha256"), "root row-to-group hash"
        ),
        "parent_split_sha256": parent_split_sha256,
        "selected_group_membership_sha256": membership_sha256,
    }
    manifest: dict[str, Any] = {
        "schema": _identity("derived_group_manifest"),
        "dataset": root_manifest.get("dataset"),
        "source_path": root_manifest.get("source_path"),
        "source_sha256": root_manifest.get("source_sha256"),
        "row_count": local_cursor,
        "feature_count": root_manifest.get("feature_count"),
        "feature_names": root_manifest.get("feature_names"),
        "declared_target_name": root_manifest.get("declared_target_name"),
        "predictor_allowlist_explicit": root_manifest.get("predictor_allowlist_explicit"),
        "group_rule": root_manifest.get("group_rule"),
        "group_id_hash_inputs": root_manifest.get("group_id_hash_inputs"),
        "declared_target_excluded_by_loader": root_manifest.get(
            "declared_target_excluded_by_loader"
        ),
        "subject_id_provenance": root_manifest.get("subject_id_provenance"),
        "proxy_or_provenance_audit_required": root_manifest.get(
            "proxy_or_provenance_audit_required"
        ),
        "group_count": len(serialized_groups),
        "max_group_size": max((item["n"] for item in serialized_groups)),
        "mixed_label_group_count": sum(
            (1 for item in serialized_groups if item["positive"] and item["negative"])
        ),
        "positive_rows": sum((item["positive"] for item in serialized_groups)),
        "negative_rows": sum((item["negative"] for item in serialized_groups)),
        "row_to_group_sha256": hashlib.sha256(assignment_payload).hexdigest(),
        "groups": serialized_groups,
        "derivation": derivation,
    }
    manifest["group_manifest_sha256"] = group_manifest_fingerprint(manifest)
    group_records(manifest)
    binding = dict(derivation)
    binding["kind"] = "derived_subset"
    binding["derived_group_manifest_sha256"] = manifest["group_manifest_sha256"]
    return (manifest, binding)


def _split_context(
    *,
    study_id: str,
    input_manifest: Mapping[str, Any],
    scope: str,
    outer_repeat: int,
    outer_fold: int,
    inner_fold: int,
) -> dict[str, Any]:
    return {
        "study_id": study_id,
        "dataset_sha256": input_manifest["source_sha256"],
        "group_manifest_sha256": input_manifest["group_manifest_sha256"],
        "scope": scope,
        "outer_repeat": outer_repeat,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
    }


def _compact_partition(
    full: Mapping[str, Any], *, input_binding: Mapping[str, Any]
) -> dict[str, Any]:
    candidate_audit = full.get("candidate_audit")
    if not isinstance(candidate_audit, list) or len(candidate_audit) != 64:
        raise NestedPlanError("partition candidate audit is missing or incomplete")
    compact: dict[str, Any] = {
        "schema": _identity("nested_partition"),
        "result_blind_selection": full.get("result_blind_selection"),
        "performance_fields_used": full.get("performance_fields_used"),
        "label_stratified": full.get("label_stratified"),
        "label_fields_used": full.get("label_fields_used"),
        "context": full.get("context"),
        "context_sha256": full.get("context_sha256"),
        "input_binding": dict(input_binding),
        "candidate_attempts": full.get("candidate_attempts"),
        "accepted_candidates": full.get("accepted_candidates"),
        "selected_attempt": full.get("selected_attempt"),
        "selected_assignment_sha256": full.get("selected_assignment_sha256"),
        "part_names": full.get("part_names"),
        "weights": full.get("weights"),
        "constraints": full.get("constraints"),
        "parts": full.get("parts"),
        "assignment": full.get("assignment"),
        "candidate_audit_sha256": _sha256_json(candidate_audit),
        "full_partition_payload_sha256": _sha256_json(full),
    }
    compact["split_sha256"] = _sha256_json(compact)
    return compact


def _build_partition(
    root_manifest: Mapping[str, Any],
    group_ids: Sequence[str],
    *,
    study_id: str,
    scope: str,
    outer_repeat: int,
    outer_fold: int,
    inner_fold: int,
    parent_split_sha256: str,
    part_names: Sequence[str],
    weights: Sequence[float],
    constraints: BalanceConstraints,
    use_root_manifest: bool = False,
) -> dict[str, Any]:
    root_groups = _root_group_map(root_manifest)
    ordered_ids = sorted(group_ids)
    if use_root_manifest:
        if ordered_ids != sorted(root_groups):
            raise NestedPlanError("root partition must cover every root group")
        input_manifest = dict(root_manifest)
        input_binding = {
            "kind": "root",
            "schema": _identity("root_group_manifest_binding"),
            "root_group_manifest_sha256": root_manifest["group_manifest_sha256"],
            "root_row_to_group_sha256": root_manifest["row_to_group_sha256"],
            "parent_split_sha256": parent_split_sha256,
            "selected_group_membership_sha256": _membership_sha256(ordered_ids),
            "derived_group_manifest_sha256": root_manifest["group_manifest_sha256"],
        }
    else:
        input_manifest, input_binding = _derived_group_manifest(
            root_manifest, ordered_ids, scope=scope, parent_split_sha256=parent_split_sha256
        )
    full = generate_balanced_partition(
        input_manifest,
        part_names=part_names,
        weights=weights,
        context=_split_context(
            study_id=study_id,
            input_manifest=input_manifest,
            scope=scope,
            outer_repeat=outer_repeat,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
        ),
        constraints=constraints,
        candidate_attempts=64,
    )
    return _compact_partition(full, input_binding=input_binding)


def _assigned_groups(partition: Mapping[str, Any], part: str) -> list[str]:
    assignment = partition.get("assignment")
    if not isinstance(assignment, Mapping):
        raise NestedPlanError("partition assignment is missing")
    return sorted(
        (str(group_id) for (group_id, assigned_part) in assignment.items() if assigned_part == part)
    )


def _role_and_client_bundle(
    root_manifest: Mapping[str, Any],
    train_group_ids: Sequence[str],
    *,
    study_id: str,
    role_scope: str,
    client_scope: str,
    outer_repeat: int,
    outer_fold: int,
    inner_fold: int,
    parent_split_sha256: str,
    root_groups: Mapping[str, GroupRecord],
) -> dict[str, Any]:
    role_partition = _build_partition(
        root_manifest,
        train_group_ids,
        study_id=study_id,
        scope=role_scope,
        outer_repeat=outer_repeat,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        parent_split_sha256=parent_split_sha256,
        part_names=ROLE_NAMES,
        weights=ROLE_WEIGHTS,
        constraints=BalanceConstraints(min_positive_per_part=10, min_negative_per_part=10),
    )
    roles = {
        role: _membership(_assigned_groups(role_partition, role), root_groups)
        for role in ROLE_NAMES
    }
    private_ids = roles["private"]["group_ids"]
    try:
        client_partition = generate_client_partition(
            root_manifest,
            private_ids,
            study_id=study_id,
            scope=client_scope,
            outer_repeat=outer_repeat,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            role="private",
            parent_split_sha256=role_partition["split_sha256"],
        )
    except ClientPartitionError as error:
        raise NestedPlanError(f"registered client partition failed closed: {error}") from error
    clients = {
        client: _membership(_assigned_groups(client_partition, client), root_groups)
        for client in CLIENT_NAMES
    }
    for client, membership in clients.items():
        if membership["row_count"] < MIN_ROWS_PER_CLIENT:
            raise NestedPlanError(f"{client} has fewer than {MIN_ROWS_PER_CLIENT} private rows")
        if (
            membership["positive"] < MIN_POSITIVE_PER_CLIENT
            or membership["negative"] < MIN_NEGATIVE_PER_CLIENT
        ):
            raise NestedPlanError(f"{client} lacks registered class support")
    return {
        "role_partition": role_partition,
        "roles": roles,
        "client_partition": client_partition,
        "clients": clients,
    }


def generate_nested_plan(group_manifest: Mapping[str, Any], *, study_id: str) -> dict[str, Any]:
    """Generate the fixed 3x5 outer, 3-inner result-blind plan."""
    if not isinstance(study_id, str) or not study_id.strip():
        raise NestedPlanError("study_id must be a non-empty string")
    root_groups = _root_group_map(group_manifest)
    root_ids = sorted(root_groups)
    source_binding = _root_binding(group_manifest)
    root_parent_sha256 = source_binding["binding_sha256"]
    repetitions: list[dict[str, Any]] = []
    for outer_repeat in range(OUTER_REPEATS):
        outer_partition = _build_partition(
            group_manifest,
            root_ids,
            study_id=study_id,
            scope="outer",
            outer_repeat=outer_repeat,
            outer_fold=-1,
            inner_fold=-1,
            parent_split_sha256=root_parent_sha256,
            part_names=[f"fold_{index}" for index in range(OUTER_FOLDS)],
            weights=[1.0] * OUTER_FOLDS,
            constraints=BalanceConstraints(min_positive_per_part=30, min_negative_per_part=30),
            use_root_manifest=True,
        )
        outer_folds: list[dict[str, Any]] = []
        for outer_fold in range(OUTER_FOLDS):
            test_ids = _assigned_groups(outer_partition, f"fold_{outer_fold}")
            train_ids = sorted(set(root_ids) - set(test_ids))
            inner_partition = _build_partition(
                group_manifest,
                train_ids,
                study_id=study_id,
                scope="inner",
                outer_repeat=outer_repeat,
                outer_fold=outer_fold,
                inner_fold=-1,
                parent_split_sha256=outer_partition["split_sha256"],
                part_names=[f"inner_fold_{index}" for index in range(INNER_FOLDS)],
                weights=[1.0] * INNER_FOLDS,
                constraints=BalanceConstraints(min_positive_per_part=30, min_negative_per_part=30),
            )
            inner_folds: list[dict[str, Any]] = []
            for inner_fold in range(INNER_FOLDS):
                validation_ids = _assigned_groups(inner_partition, f"inner_fold_{inner_fold}")
                inner_train_ids = sorted(set(train_ids) - set(validation_ids))
                bundle = _role_and_client_bundle(
                    group_manifest,
                    inner_train_ids,
                    study_id=study_id,
                    role_scope="inner_roles",
                    client_scope="inner_private_clients_fixed_label_skew",
                    outer_repeat=outer_repeat,
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                    parent_split_sha256=inner_partition["split_sha256"],
                    root_groups=root_groups,
                )
                inner_folds.append(
                    {
                        "inner_fold": inner_fold,
                        "inner_train": _membership(inner_train_ids, root_groups),
                        "inner_validation": _membership(validation_ids, root_groups),
                        **bundle,
                    }
                )
            refit = _role_and_client_bundle(
                group_manifest,
                train_ids,
                study_id=study_id,
                role_scope="outer_refit_roles",
                client_scope="outer_refit_private_clients_fixed_label_skew",
                outer_repeat=outer_repeat,
                outer_fold=outer_fold,
                inner_fold=-1,
                parent_split_sha256=outer_partition["split_sha256"],
                root_groups=root_groups,
            )
            outer_folds.append(
                {
                    "outer_fold": outer_fold,
                    "outer_train": _membership(train_ids, root_groups),
                    "outer_test": _membership(test_ids, root_groups),
                    "inner_partition": inner_partition,
                    "inner_folds": inner_folds,
                    "outer_refit": refit,
                }
            )
        repetitions.append(
            {
                "outer_repeat": outer_repeat,
                "outer_partition": outer_partition,
                "outer_folds": outer_folds,
            }
        )
    plan: dict[str, Any] = {
        "schema": _identity("nested_plan"),
        "study_id": study_id,
        "status": "DRAFT_NOT_FROZEN",
        "result_blind_selection": True,
        "performance_fields_used": [],
        "label_stratified": True,
        "label_fields_used": ["positive", "negative", "prevalence"],
        "source_binding": source_binding,
        "design": {
            "outer_repeats": OUTER_REPEATS,
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "role_names": list(ROLE_NAMES),
            "role_display_names": {"private": "private", "v_ctrl": "V_ctrl", "v_sel": "V_sel"},
            "role_weights": list(ROLE_WEIGHTS),
            "client_count": CLIENT_COUNT,
            "client_names": list(CLIENT_NAMES),
            "candidate_attempts_per_partition": 64,
            "client_partition_policy": CLIENT_PARTITION_POLICY,
            "client_partition_policy_sha256": client_partition_policy_fingerprint(),
            "client_positive_share_percent_by_rank": list(POSITIVE_SHARE_PERCENT),
            "client_negative_share_percent_by_rank": list(NEGATIVE_SHARE_PERCENT),
            "client_target_label_distribution_total_variation": 0.3,
            "client_target_max_min_class_share_ratio": 3.0,
            "client_minimums": {
                "rows": MIN_ROWS_PER_CLIENT,
                "positive": MIN_POSITIVE_PER_CLIENT,
                "negative": MIN_NEGATIVE_PER_CLIENT,
            },
            "client_partition_scope": "registered controlled label-skew simulation; not observed multicenter heterogeneity",
            "outer_test_hpo_access": "forbidden_fail_closed",
        },
        "repetitions": repetitions,
    }
    plan["nested_plan_sha256"] = nested_plan_fingerprint(plan)
    validate_nested_plan(plan, group_manifest)
    return plan


def _verify_freeze_binding(plan: Mapping[str, Any], source_binding: Mapping[str, Any]) -> None:
    value = plan.get("freeze_binding")
    if not isinstance(value, Mapping):
        raise NestedPlanError("frozen nested plan has no freeze binding")
    required = {
        "schema",
        "study_id",
        "draft_nested_plan_sha256",
        "source_sha256",
        "group_manifest_sha256",
        "row_to_group_sha256",
        "protocol_sha256",
        "implementation_sha256",
        "client_partition_policy_sha256",
        "result_blind_confirmation",
        "performance_fields_used",
        "freeze_binding_sha256",
    }
    if set(value) != required:
        raise NestedPlanError("nested-plan freeze-binding schema differs")
    if value.get("schema") != _identity("nested_plan_freeze_binding"):
        raise NestedPlanError("nested-plan freeze-binding version differs")
    if value.get("study_id") != plan.get("study_id"):
        raise NestedPlanError("freeze binding study_id differs")
    for field in (
        "draft_nested_plan_sha256",
        "source_sha256",
        "group_manifest_sha256",
        "row_to_group_sha256",
        "protocol_sha256",
        "implementation_sha256",
        "client_partition_policy_sha256",
        "freeze_binding_sha256",
    ):
        _require_sha256(value.get(field), f"freeze binding {field}")
    if value.get("source_sha256") != source_binding.get("source_sha256"):
        raise NestedPlanError("freeze binding source hash differs")
    if value.get("group_manifest_sha256") != source_binding.get("group_manifest_sha256"):
        raise NestedPlanError("freeze binding group-manifest hash differs")
    if value.get("row_to_group_sha256") != source_binding.get("row_to_group_sha256"):
        raise NestedPlanError("freeze binding row-to-group hash differs")
    if value.get("client_partition_policy_sha256") != client_partition_policy_fingerprint():
        raise NestedPlanError("freeze binding client-policy hash differs")
    if (
        value.get("result_blind_confirmation") is not True
        or value.get("performance_fields_used") != []
    ):
        raise NestedPlanError("freeze binding is not result blind")
    stored_binding_hash = str(value["freeze_binding_sha256"])
    binding_payload = {
        str(key): item for (key, item) in value.items() if key != "freeze_binding_sha256"
    }
    if not hmac.compare_digest(stored_binding_hash, _sha256_json(binding_payload)):
        raise NestedPlanError("freeze binding hash mismatch")
    reconstructed_draft = copy.deepcopy(dict(plan))
    reconstructed_draft.pop("freeze_binding", None)
    reconstructed_draft["status"] = "DRAFT_NOT_FROZEN"
    reconstructed_draft["nested_plan_sha256"] = nested_plan_fingerprint(reconstructed_draft)
    if value.get("draft_nested_plan_sha256") != reconstructed_draft.get("nested_plan_sha256"):
        raise NestedPlanError("freeze binding draft-plan hash differs")


def freeze_nested_plan(
    plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
    *,
    protocol_sha256: str,
    implementation_sha256: str,
    client_policy_sha256: str,
) -> dict[str, Any]:
    """Freeze a validated draft and bind protocol, code and client policy.

    The caller supplies the precommit protocol and implementation hashes.  The
    client-policy hash is independently recomputed here, so a caller cannot
    freeze a draft against a different non-IID strength definition.
    """
    validate_nested_plan(plan, group_manifest, require_frozen=False)
    if plan.get("status") != "DRAFT_NOT_FROZEN":
        raise NestedPlanError("only a DRAFT_NOT_FROZEN plan may be frozen")
    if "freeze_binding" in plan:
        raise NestedPlanError("draft nested plan unexpectedly has a freeze binding")
    protocol_sha256 = _require_sha256(protocol_sha256, "protocol hash")
    implementation_sha256 = _require_sha256(implementation_sha256, "implementation hash")
    client_policy_sha256 = _require_sha256(client_policy_sha256, "client-policy hash")
    expected_client_policy = client_partition_policy_fingerprint()
    if not hmac.compare_digest(client_policy_sha256, expected_client_policy):
        raise NestedPlanError("requested client-policy hash differs from implementation")
    frozen = copy.deepcopy(dict(plan))
    source_binding = _root_binding(group_manifest)
    binding: dict[str, Any] = {
        "schema": _identity("nested_plan_freeze_binding"),
        "study_id": frozen["study_id"],
        "draft_nested_plan_sha256": frozen["nested_plan_sha256"],
        "source_sha256": source_binding["source_sha256"],
        "group_manifest_sha256": source_binding["group_manifest_sha256"],
        "row_to_group_sha256": source_binding["row_to_group_sha256"],
        "protocol_sha256": protocol_sha256,
        "implementation_sha256": implementation_sha256,
        "client_partition_policy_sha256": client_policy_sha256,
        "result_blind_confirmation": True,
        "performance_fields_used": [],
    }
    binding["freeze_binding_sha256"] = _sha256_json(binding)
    frozen["status"] = "FROZEN"
    frozen["freeze_binding"] = binding
    frozen["nested_plan_sha256"] = nested_plan_fingerprint(frozen)
    validate_nested_plan(frozen, group_manifest, require_frozen=True)
    return frozen


def _verify_membership(
    value: Any, expected_ids: Sequence[str], root_groups: Mapping[str, GroupRecord], name: str
) -> None:
    if not isinstance(value, Mapping):
        raise NestedPlanError(f"{name} membership is missing")
    expected = _membership(expected_ids, root_groups)
    if dict(value) != expected:
        raise NestedPlanError(f"{name} membership or count binding differs")


def _verify_partition(
    partition: Any,
    *,
    expected_group_ids: Sequence[str],
    expected_part_names: Sequence[str],
    expected_weights: Sequence[float],
    expected_constraints: BalanceConstraints,
    expected_study_id: str,
    expected_scope: str,
    expected_outer_repeat: int,
    expected_outer_fold: int,
    expected_inner_fold: int,
    expected_parent_split_sha256: str,
    root_manifest: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    root_groups: Mapping[str, GroupRecord],
    use_root_manifest: bool = False,
) -> None:
    if not isinstance(partition, Mapping):
        raise NestedPlanError("nested partition is missing")
    if partition.get("schema") != _identity("nested_partition"):
        raise NestedPlanError("nested partition schema differs")
    if partition.get("result_blind_selection") is not True:
        raise NestedPlanError("partition is not marked result blind")
    if partition.get("performance_fields_used") != []:
        raise NestedPlanError("partition records performance-dependent fields")
    if partition.get("label_stratified") is not True or partition.get("label_fields_used") != [
        "positive",
        "negative",
        "prevalence",
    ]:
        raise NestedPlanError("partition label stratification is not explicit")
    if partition.get("candidate_attempts") != 64:
        raise NestedPlanError("partition does not bind exactly 64 candidates")
    if partition.get("part_names") != list(expected_part_names):
        raise NestedPlanError("partition part names differ from registered design")
    if partition.get("weights") != [float(value) for value in expected_weights]:
        raise NestedPlanError("partition weights differ from registered design")
    if partition.get("constraints") != expected_constraints.__dict__:
        raise NestedPlanError("partition constraints differ from registered design")
    selected_attempt = partition.get("selected_attempt")
    accepted_candidates = partition.get("accepted_candidates")
    if (
        isinstance(selected_attempt, bool)
        or not isinstance(selected_attempt, int)
        or (not 0 <= selected_attempt < 64)
    ):
        raise NestedPlanError("partition selected attempt is invalid")
    if (
        isinstance(accepted_candidates, bool)
        or not isinstance(accepted_candidates, int)
        or (not 1 <= accepted_candidates <= 64)
    ):
        raise NestedPlanError("partition accepted-candidate count is invalid")
    expected_ids = sorted(expected_group_ids)
    assignment = partition.get("assignment")
    if not isinstance(assignment, Mapping) or sorted(assignment) != expected_ids:
        raise NestedPlanError("partition assignment is not exhaustive")
    if any((value not in expected_part_names for value in assignment.values())):
        raise NestedPlanError("partition assignment contains an unknown part")
    assignment_hash = _sha256_json(
        sorted(((group_id, assignment[group_id]) for group_id in assignment))
    )
    if not hmac.compare_digest(
        _require_sha256(partition.get("selected_assignment_sha256"), "assignment hash"),
        assignment_hash,
    ):
        raise NestedPlanError("partition assignment hash mismatch")
    context = partition.get("context")
    input_binding = partition.get("input_binding")
    if not isinstance(context, Mapping) or not isinstance(input_binding, Mapping):
        raise NestedPlanError("partition context or input binding is missing")
    if context.get("dataset_sha256") != source_binding.get("source_sha256"):
        raise NestedPlanError("partition source hash differs from root source")
    if context.get("group_manifest_sha256") != input_binding.get("derived_group_manifest_sha256"):
        raise NestedPlanError("partition derived-manifest hash is not bound")
    if input_binding.get("root_group_manifest_sha256") != source_binding.get(
        "group_manifest_sha256"
    ):
        raise NestedPlanError("partition root group-manifest hash differs")
    if input_binding.get("root_row_to_group_sha256") != source_binding.get("row_to_group_sha256"):
        raise NestedPlanError("partition root row-to-group hash differs")
    if input_binding.get("parent_split_sha256") != expected_parent_split_sha256:
        raise NestedPlanError("partition parent split hash differs")
    if input_binding.get("selected_group_membership_sha256") != _membership_sha256(expected_ids):
        raise NestedPlanError("partition input membership hash differs")
    if partition.get("context_sha256") != _sha256_json(dict(context)):
        raise NestedPlanError("partition context hash mismatch")
    expected_parts: dict[str, dict[str, Any]] = {}
    for part_name in expected_part_names:
        ids = sorted(
            (group_id for (group_id, assigned) in assignment.items() if assigned == part_name)
        )
        membership = _membership(ids, root_groups)
        expected_parts[part_name] = {
            "n": membership["row_count"],
            "positive": membership["positive"],
            "negative": membership["negative"],
            "groups": membership["group_count"],
            "prevalence": membership["prevalence"],
            "membership_sha256": _sha256_lines(ids),
        }
    if partition.get("parts") != expected_parts:
        raise NestedPlanError("partition part statistics or memberships differ")
    stored_split = _require_sha256(partition.get("split_sha256"), "split hash")
    payload = {str(key): value for (key, value) in partition.items() if key != "split_sha256"}
    if not hmac.compare_digest(stored_split, _sha256_json(payload)):
        raise NestedPlanError("partition split hash mismatch")
    try:
        expected_partition = _build_partition(
            root_manifest,
            expected_ids,
            study_id=expected_study_id,
            scope=expected_scope,
            outer_repeat=expected_outer_repeat,
            outer_fold=expected_outer_fold,
            inner_fold=expected_inner_fold,
            parent_split_sha256=expected_parent_split_sha256,
            part_names=expected_part_names,
            weights=expected_weights,
            constraints=expected_constraints,
            use_root_manifest=use_root_manifest,
        )
    except Exception as error:
        raise NestedPlanError(f"partition deterministic reconstruction failed: {error}") from error
    if dict(partition) != expected_partition:
        raise NestedPlanError("partition differs from the registered 64-candidate reconstruction")


def _verify_role_bundle(
    bundle: Mapping[str, Any],
    *,
    train_ids: Sequence[str],
    study_id: str,
    role_scope: str,
    client_scope: str,
    outer_repeat: int,
    outer_fold: int,
    inner_fold: int,
    expected_parent_split_sha256: str,
    group_manifest: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    root_groups: Mapping[str, GroupRecord],
) -> None:
    role_partition = bundle.get("role_partition")
    _verify_partition(
        role_partition,
        expected_group_ids=train_ids,
        expected_part_names=ROLE_NAMES,
        expected_weights=ROLE_WEIGHTS,
        expected_constraints=BalanceConstraints(min_positive_per_part=10, min_negative_per_part=10),
        expected_study_id=study_id,
        expected_scope=role_scope,
        expected_outer_repeat=outer_repeat,
        expected_outer_fold=outer_fold,
        expected_inner_fold=inner_fold,
        expected_parent_split_sha256=expected_parent_split_sha256,
        root_manifest=group_manifest,
        source_binding=source_binding,
        root_groups=root_groups,
    )
    if not isinstance(role_partition, Mapping):
        raise NestedPlanError("role partition is missing")
    roles = bundle.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != set(ROLE_NAMES):
        raise NestedPlanError("role membership map differs")
    role_sets: list[set[str]] = []
    for role in ROLE_NAMES:
        ids = _assigned_groups(role_partition, role)
        _verify_membership(roles[role], ids, root_groups, role)
        role_sets.append(set(ids))
    if set.union(*role_sets) != set(train_ids) or sum((len(value) for value in role_sets)) != len(
        set(train_ids)
    ):
        raise NestedPlanError("private/V_ctrl/V_sel are not exclusive and exhaustive")
    private_ids = _assigned_groups(role_partition, "private")
    client_partition = bundle.get("client_partition")
    if not isinstance(client_partition, Mapping):
        raise NestedPlanError("client partition is missing")
    try:
        validate_client_partition(
            client_partition,
            group_manifest,
            private_ids,
            study_id=study_id,
            scope=client_scope,
            outer_repeat=outer_repeat,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            role="private",
            parent_split_sha256=role_partition["split_sha256"],
        )
    except ClientPartitionError as error:
        raise NestedPlanError(f"registered client partition validation failed: {error}") from error
    clients = bundle.get("clients")
    if not isinstance(clients, Mapping) or set(clients) != set(CLIENT_NAMES):
        raise NestedPlanError("client membership map differs")
    client_sets: list[set[str]] = []
    for client in CLIENT_NAMES:
        ids = _assigned_groups(client_partition, client)
        _verify_membership(clients[client], ids, root_groups, client)
        membership = clients[client]
        if (
            membership["row_count"] < MIN_ROWS_PER_CLIENT
            or membership["positive"] < MIN_POSITIVE_PER_CLIENT
            or membership["negative"] < MIN_NEGATIVE_PER_CLIENT
        ):
            raise NestedPlanError("client class/size constraint differs")
        client_sets.append(set(ids))
    if set.union(*client_sets) != set(private_ids) or sum(
        (len(value) for value in client_sets)
    ) != len(set(private_ids)):
        raise NestedPlanError("clients are not group-exclusive and exhaustive")


def validate_nested_plan(
    plan: Mapping[str, Any], group_manifest: Mapping[str, Any], *, require_frozen: bool = False
) -> None:
    """Validate all nested identities and group-exclusive membership paths."""
    if not isinstance(require_frozen, bool):
        raise NestedPlanError("require_frozen must be boolean")
    if not isinstance(plan, Mapping):
        raise NestedPlanError("nested plan must be a mapping")
    if plan.get("schema") != _identity("nested_plan"):
        raise NestedPlanError("nested plan schema differs")
    status = plan.get("status")
    if status not in {"DRAFT_NOT_FROZEN", "FROZEN"}:
        raise NestedPlanError("nested plan status differs")
    expected_top_fields = {
        "schema",
        "study_id",
        "status",
        "result_blind_selection",
        "performance_fields_used",
        "label_stratified",
        "label_fields_used",
        "source_binding",
        "design",
        "repetitions",
        "nested_plan_sha256",
    }
    if status == "FROZEN":
        expected_top_fields.add("freeze_binding")
    if set(plan) != expected_top_fields:
        raise NestedPlanError("nested plan top-level fields differ from the exact schema")
    try:
        assert_no_performance_fields(plan)
    except CandidateSpaceError as exc:
        raise NestedPlanError(str(exc)) from exc
    stored_fingerprint = _require_sha256(plan.get("nested_plan_sha256"), "nested-plan hash")
    if not hmac.compare_digest(stored_fingerprint, nested_plan_fingerprint(plan)):
        raise NestedPlanError("nested plan fingerprint mismatch")
    if require_frozen and status != "FROZEN":
        raise NestedPlanError("formal consumer requires a FROZEN nested plan")
    if status == "DRAFT_NOT_FROZEN" and "freeze_binding" in plan:
        raise NestedPlanError("draft nested plan must not contain a freeze binding")
    if plan.get("result_blind_selection") is not True or plan.get("performance_fields_used") != []:
        raise NestedPlanError("nested plan is not result blind")
    study_id = plan.get("study_id")
    if not isinstance(study_id, str) or not study_id.strip():
        raise NestedPlanError("nested plan study_id is invalid")
    if plan.get("label_stratified") is not True or plan.get("label_fields_used") != [
        "positive",
        "negative",
        "prevalence",
    ]:
        raise NestedPlanError("nested plan does not declare label stratification")
    expected_design = {
        "outer_repeats": OUTER_REPEATS,
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "role_names": list(ROLE_NAMES),
        "role_display_names": {"private": "private", "v_ctrl": "V_ctrl", "v_sel": "V_sel"},
        "role_weights": list(ROLE_WEIGHTS),
        "client_count": CLIENT_COUNT,
        "client_names": list(CLIENT_NAMES),
        "candidate_attempts_per_partition": 64,
        "client_partition_policy": CLIENT_PARTITION_POLICY,
        "client_partition_policy_sha256": client_partition_policy_fingerprint(),
        "client_positive_share_percent_by_rank": list(POSITIVE_SHARE_PERCENT),
        "client_negative_share_percent_by_rank": list(NEGATIVE_SHARE_PERCENT),
        "client_target_label_distribution_total_variation": 0.3,
        "client_target_max_min_class_share_ratio": 3.0,
        "client_minimums": {
            "rows": MIN_ROWS_PER_CLIENT,
            "positive": MIN_POSITIVE_PER_CLIENT,
            "negative": MIN_NEGATIVE_PER_CLIENT,
        },
        "client_partition_scope": "registered controlled label-skew simulation; not observed multicenter heterogeneity",
        "outer_test_hpo_access": "forbidden_fail_closed",
    }
    if plan.get("design") != expected_design:
        raise NestedPlanError("nested plan design differs from the fixed design")
    source_binding = _root_binding(group_manifest)
    if plan.get("source_binding") != source_binding:
        raise NestedPlanError("source or group-manifest binding differs")
    if status == "FROZEN":
        _verify_freeze_binding(plan, source_binding)
    validation_cache_key = (stored_fingerprint, str(source_binding["group_manifest_sha256"]))
    if validation_cache_key in _VALIDATED_PLAN_CACHE:
        return
    if status == "FROZEN":
        freeze_binding = plan["freeze_binding"]
        draft_cache_key = (
            str(freeze_binding["draft_nested_plan_sha256"]),
            str(source_binding["group_manifest_sha256"]),
        )
        if draft_cache_key in _VALIDATED_PLAN_CACHE:
            if len(_VALIDATED_PLAN_CACHE) >= 64:
                _VALIDATED_PLAN_CACHE.clear()
            _VALIDATED_PLAN_CACHE.add(validation_cache_key)
            return
    root_groups = _root_group_map(group_manifest)
    root_ids = sorted(root_groups)
    repetitions = plan.get("repetitions")
    if not isinstance(repetitions, list) or len(repetitions) != OUTER_REPEATS:
        raise NestedPlanError("outer repeat count differs")
    root_parent_sha256 = source_binding["binding_sha256"]
    for outer_repeat, repeat_plan in enumerate(repetitions):
        if not isinstance(repeat_plan, Mapping) or repeat_plan.get("outer_repeat") != outer_repeat:
            raise NestedPlanError("outer repeat index differs")
        outer_partition = repeat_plan.get("outer_partition")
        _verify_partition(
            outer_partition,
            expected_group_ids=root_ids,
            expected_part_names=[f"fold_{index}" for index in range(OUTER_FOLDS)],
            expected_weights=[1.0] * OUTER_FOLDS,
            expected_constraints=BalanceConstraints(
                min_positive_per_part=30, min_negative_per_part=30
            ),
            expected_study_id=study_id,
            expected_scope="outer",
            expected_outer_repeat=outer_repeat,
            expected_outer_fold=-1,
            expected_inner_fold=-1,
            expected_parent_split_sha256=root_parent_sha256,
            root_manifest=group_manifest,
            source_binding=source_binding,
            root_groups=root_groups,
            use_root_manifest=True,
        )
        if not isinstance(outer_partition, Mapping):
            raise NestedPlanError("outer partition is missing")
        outer_folds = repeat_plan.get("outer_folds")
        if not isinstance(outer_folds, list) or len(outer_folds) != OUTER_FOLDS:
            raise NestedPlanError("outer fold count differs")
        seen_test: set[str] = set()
        for outer_fold, fold_plan in enumerate(outer_folds):
            if not isinstance(fold_plan, Mapping) or fold_plan.get("outer_fold") != outer_fold:
                raise NestedPlanError("outer fold index differs")
            test_ids = _assigned_groups(outer_partition, f"fold_{outer_fold}")
            train_ids = sorted(set(root_ids) - set(test_ids))
            if seen_test & set(test_ids):
                raise NestedPlanError("a group enters outer test twice in one repeat")
            seen_test.update(test_ids)
            _verify_membership(fold_plan.get("outer_train"), train_ids, root_groups, "outer_train")
            _verify_membership(fold_plan.get("outer_test"), test_ids, root_groups, "outer_test")
            inner_partition = fold_plan.get("inner_partition")
            _verify_partition(
                inner_partition,
                expected_group_ids=train_ids,
                expected_part_names=[f"inner_fold_{index}" for index in range(INNER_FOLDS)],
                expected_weights=[1.0] * INNER_FOLDS,
                expected_constraints=BalanceConstraints(
                    min_positive_per_part=30, min_negative_per_part=30
                ),
                expected_study_id=study_id,
                expected_scope="inner",
                expected_outer_repeat=outer_repeat,
                expected_outer_fold=outer_fold,
                expected_inner_fold=-1,
                expected_parent_split_sha256=outer_partition["split_sha256"],
                root_manifest=group_manifest,
                source_binding=source_binding,
                root_groups=root_groups,
            )
            if not isinstance(inner_partition, Mapping):
                raise NestedPlanError("inner partition is missing")
            inner_folds = fold_plan.get("inner_folds")
            if not isinstance(inner_folds, list) or len(inner_folds) != INNER_FOLDS:
                raise NestedPlanError("inner fold count differs")
            seen_validation: set[str] = set()
            for inner_fold, inner_plan in enumerate(inner_folds):
                if (
                    not isinstance(inner_plan, Mapping)
                    or inner_plan.get("inner_fold") != inner_fold
                ):
                    raise NestedPlanError("inner fold index differs")
                validation_ids = _assigned_groups(inner_partition, f"inner_fold_{inner_fold}")
                inner_train_ids = sorted(set(train_ids) - set(validation_ids))
                if seen_validation & set(validation_ids):
                    raise NestedPlanError("an inner group validates twice")
                seen_validation.update(validation_ids)
                _verify_membership(
                    inner_plan.get("inner_train"), inner_train_ids, root_groups, "inner_train"
                )
                _verify_membership(
                    inner_plan.get("inner_validation"),
                    validation_ids,
                    root_groups,
                    "inner_validation",
                )
                _verify_role_bundle(
                    inner_plan,
                    train_ids=inner_train_ids,
                    study_id=study_id,
                    role_scope="inner_roles",
                    client_scope="inner_private_clients_fixed_label_skew",
                    outer_repeat=outer_repeat,
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                    expected_parent_split_sha256=inner_partition["split_sha256"],
                    group_manifest=group_manifest,
                    source_binding=source_binding,
                    root_groups=root_groups,
                )
            if seen_validation != set(train_ids):
                raise NestedPlanError("inner validation is not OOF exhaustive")
            refit = fold_plan.get("outer_refit")
            if not isinstance(refit, Mapping):
                raise NestedPlanError("outer-refit plan is missing")
            _verify_role_bundle(
                refit,
                train_ids=train_ids,
                study_id=study_id,
                role_scope="outer_refit_roles",
                client_scope="outer_refit_private_clients_fixed_label_skew",
                outer_repeat=outer_repeat,
                outer_fold=outer_fold,
                inner_fold=-1,
                expected_parent_split_sha256=outer_partition["split_sha256"],
                group_manifest=group_manifest,
                source_binding=source_binding,
                root_groups=root_groups,
            )
        if seen_test != set(root_ids):
            raise NestedPlanError("outer test is not exhaustive within a repeat")
    if len(_VALIDATED_PLAN_CACHE) >= 64:
        _VALIDATED_PLAN_CACHE.clear()
    _VALIDATED_PLAN_CACHE.add(validation_cache_key)


def materialize_hpo_row_ids(
    plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
    *,
    outer_repeat: int,
    outer_fold: int,
    inner_fold: int,
    role: str,
) -> HPODataSlice:
    """Return row IDs for one HPO role; fail before any outer-test lookup.

    Allowed roles are ``private``, ``v_ctrl``, ``v_sel``,
    ``inner_validation`` and ``client_0`` through ``client_4``.  Outer-test and
    outer-refit access require separate post-HPO APIs and are intentionally not
    implemented here.
    """
    if role in {"outer_test", "test", "outer-test"}:
        raise ForbiddenOuterTestAccess("HPO loaders cannot materialize outer-test row IDs")
    allowed = {"private", "v_ctrl", "v_sel", "inner_validation", *CLIENT_NAMES}
    if role not in allowed:
        raise NestedPlanError(f"role is not an allowed HPO data slice: {role}")
    for value, upper, name in (
        (outer_repeat, OUTER_REPEATS, "outer_repeat"),
        (outer_fold, OUTER_FOLDS, "outer_fold"),
        (inner_fold, INNER_FOLDS, "inner_fold"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or (not 0 <= value < upper):
            raise NestedPlanError(f"{name} is outside the registered range")
    validate_nested_plan(plan, group_manifest, require_frozen=True)
    inner_plan = plan["repetitions"][outer_repeat]["outer_folds"][outer_fold]["inner_folds"][
        inner_fold
    ]
    if role == "inner_validation":
        membership = inner_plan["inner_validation"]
        parent_split = plan["repetitions"][outer_repeat]["outer_folds"][outer_fold][
            "inner_partition"
        ]["split_sha256"]
    elif role in CLIENT_NAMES:
        membership = inner_plan["clients"][role]
        parent_split = inner_plan["client_partition"]["split_sha256"]
    else:
        membership = inner_plan["roles"][role]
        parent_split = inner_plan["role_partition"]["split_sha256"]
    root_groups = _root_group_map(group_manifest)
    group_ids = tuple(membership["group_ids"])
    row_ids = tuple(
        sorted((row_id for group_id in group_ids for row_id in root_groups[group_id].row_ids))
    )
    return HPODataSlice(
        study_id=str(plan["study_id"]),
        source_sha256=str(plan["source_binding"]["source_sha256"]),
        group_manifest_sha256=str(plan["source_binding"]["group_manifest_sha256"]),
        nested_plan_sha256=str(plan["nested_plan_sha256"]),
        outer_repeat=outer_repeat,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        role=role,
        parent_split_sha256=str(parent_split),
        membership_sha256=str(membership["membership_sha256"]),
        group_ids=group_ids,
        row_ids=row_ids,
    )
