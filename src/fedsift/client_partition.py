"""Registered result-blind five-client label-skew partitioning.

The FedSift client simulation uses a fixed, interpretable non-IID policy instead of
choosing a Dirichlet concentration after seeing model performance.  Across the
five (seed-permuted) client identities, the target shares of all positive rows
are ``10/15/20/25/30%`` and the target shares of all negative rows are the
reverse sequence.  Consequently the registered target label-distribution
total variation is 0.30 and the maximum/minimum target class-share ratio is
3.0.  These quantities are protocol inputs, never tuned outcomes.

Exact-feature or subject groups are indivisible.  Sixty-four domain-separated
deterministic assignment attempts are generated from dataset, repetition,
outer-fold, inner-fold, role, source, group-manifest, parent-split and selected
membership identities.  The feasible assignment with the smallest exact
integer label-share error is selected; no model, loss or predictive metric is
available to this module.  If no attempt satisfies the registered per-client
minimums, generation fails closed.

This is a controlled label-skew simulation, not evidence of real hospital
heterogeneity.  It also does not independently parameterize covariate or
quantity shift (row-count differences may arise mechanically from class-share
targets), availability, system heterogeneity or additional institutions.
"""

from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import hashlib
import hmac
import json
import math
from typing import Any, Mapping, Sequence
from .group_manifest import GroupRecord, group_records


class ClientPartitionError(RuntimeError):
    """Raised when a client partition is malformed or internally inconsistent."""


class ClientPartitionInfeasibleError(ClientPartitionError):
    """Raised when the registered constraints have no accepted assignment."""


CLIENT_NAMES = tuple((f"client_{index}" for index in range(5)))
POSITIVE_SHARE_PERCENT = (10, 15, 20, 25, 30)
NEGATIVE_SHARE_PERCENT = tuple(reversed(POSITIVE_SHARE_PERCENT))
MIN_ROWS_PER_CLIENT = 24
MIN_POSITIVE_PER_CLIENT = 2
MIN_NEGATIVE_PER_CLIENT = 2
CANDIDATE_ATTEMPTS = 64
POLICY_NAME = "fixed_label_share_skew_tv030_group_safe_v1"
SCHEMA = _identity("registered_client_partition")
SEED_SCHEMA = _identity("client_partition_seed_chain")
SEED_DOMAIN = _identity("registered_client_label_skew")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any((character not in "0123456789abcdef" for character in value))
    ):
        raise ClientPartitionError(f"{name} must be a lowercase SHA-256")
    return value


def _require_index(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ClientPartitionError(f"{name} must be an integer >= {minimum}")
    return value


def _membership_sha256(group_ids: Sequence[str]) -> str:
    ordered = sorted(group_ids)
    if not ordered or len(ordered) != len(set(ordered)):
        raise ClientPartitionError("client input membership must be non-empty and unique")
    return _sha256_json(ordered)


def _root_group_map(group_manifest: Mapping[str, Any]) -> dict[str, GroupRecord]:
    return {record.group_id: record for record in group_records(group_manifest)}


def _policy() -> dict[str, Any]:
    return {
        "name": POLICY_NAME,
        "family": "fixed_label_distribution_skew",
        "client_count": len(CLIENT_NAMES),
        "client_names": list(CLIENT_NAMES),
        "positive_share_percent_by_rank": list(POSITIVE_SHARE_PERCENT),
        "negative_share_percent_by_rank": list(NEGATIVE_SHARE_PERCENT),
        "target_label_distribution_total_variation": 0.3,
        "target_max_min_class_share_ratio": 3.0,
        "strength_selection": "pre_registered_result_blind_not_tuned",
        "group_atomicity": "group_id_indivisible",
        "candidate_attempts": CANDIDATE_ATTEMPTS,
        "selection_objective": "minimum_exact_equal_weight_positive_negative_share_squared_error",
        "selection_tie_break": "lexicographic_assignment_sha256",
        "minimum_rows_per_client": MIN_ROWS_PER_CLIENT,
        "minimum_positive_per_client": MIN_POSITIVE_PER_CLIENT,
        "minimum_negative_per_client": MIN_NEGATIVE_PER_CLIENT,
        "performance_fields_used": [],
        "limitations": [
            "controlled_label_skew_not_observed_hospital_heterogeneity",
            "does_not_independently_parameterize_covariate_or_quantity_shift",
            "does_not_model_client_availability_or_system_heterogeneity",
            "five_pseudo_clients_do_not_establish_multicenter_external_validity",
        ],
    }


def client_partition_policy() -> dict[str, Any]:
    """Return a fresh JSON-safe copy of the exact registered client policy."""
    return json.loads(_canonical_json(_policy()))


def client_partition_policy_fingerprint() -> str:
    """Hash every fixed client-policy field used by generation and validation."""
    return _sha256_json(_policy())


def _seed_chain(
    *,
    study_id: str,
    dataset: Any,
    source_sha256: str,
    group_manifest_sha256: str,
    row_to_group_sha256: str,
    selected_group_membership_sha256: str,
    parent_split_sha256: str,
    outer_repeat: int,
    outer_fold: int,
    inner_fold: int,
    role: str,
    scope: str,
) -> dict[str, Any]:
    if not isinstance(study_id, str) or not study_id.strip():
        raise ClientPartitionError("study_id must be a non-empty string")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ClientPartitionError("dataset must be a non-empty string")
    if not isinstance(role, str) or not role.strip():
        raise ClientPartitionError("role must be a non-empty string")
    if not isinstance(scope, str) or not scope.strip():
        raise ClientPartitionError("scope must be a non-empty string")
    source_sha256 = _require_sha256(source_sha256, "source hash")
    group_manifest_sha256 = _require_sha256(group_manifest_sha256, "group-manifest hash")
    row_to_group_sha256 = _require_sha256(row_to_group_sha256, "row-to-group hash")
    selected_group_membership_sha256 = _require_sha256(
        selected_group_membership_sha256, "selected-membership hash"
    )
    parent_split_sha256 = _require_sha256(parent_split_sha256, "parent split hash")
    outer_repeat = _require_index(outer_repeat, "outer_repeat", minimum=0)
    outer_fold = _require_index(outer_fold, "outer_fold", minimum=0)
    inner_fold = _require_index(inner_fold, "inner_fold", minimum=-1)
    dataset_material = {
        "domain": f"{SEED_DOMAIN}/dataset",
        "study_id": study_id,
        "dataset": dataset,
        "source_sha256": source_sha256,
        "group_manifest_sha256": group_manifest_sha256,
        "row_to_group_sha256": row_to_group_sha256,
    }
    dataset_seed = _sha256_json(dataset_material)
    repetition_seed = _sha256_json(
        {
            "domain": f"{SEED_DOMAIN}/repetition",
            "dataset_seed_sha256": dataset_seed,
            "outer_repeat": outer_repeat,
        }
    )
    outer_seed = _sha256_json(
        {
            "domain": f"{SEED_DOMAIN}/outer",
            "repetition_seed_sha256": repetition_seed,
            "outer_fold": outer_fold,
            "parent_split_sha256": parent_split_sha256,
        }
    )
    inner_seed = _sha256_json(
        {
            "domain": f"{SEED_DOMAIN}/inner",
            "outer_seed_sha256": outer_seed,
            "inner_fold": inner_fold,
        }
    )
    role_seed = _sha256_json(
        {
            "domain": f"{SEED_DOMAIN}/role",
            "inner_seed_sha256": inner_seed,
            "role": role,
            "scope": scope,
            "selected_group_membership_sha256": selected_group_membership_sha256,
        }
    )
    chain: dict[str, Any] = {
        "schema": SEED_SCHEMA,
        "domain": SEED_DOMAIN,
        "study_id": study_id,
        "dataset": dataset,
        "source_sha256": source_sha256,
        "group_manifest_sha256": group_manifest_sha256,
        "row_to_group_sha256": row_to_group_sha256,
        "selected_group_membership_sha256": selected_group_membership_sha256,
        "parent_split_sha256": parent_split_sha256,
        "outer_repeat": outer_repeat,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "role": role,
        "scope": scope,
        "dataset_seed_sha256": dataset_seed,
        "repetition_seed_sha256": repetition_seed,
        "outer_seed_sha256": outer_seed,
        "inner_seed_sha256": inner_seed,
        "role_seed_sha256": role_seed,
    }
    chain["seed_chain_sha256"] = _sha256_json(chain)
    return chain


def _client_rank_order(role_seed_sha256: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            CLIENT_NAMES,
            key=lambda client: _sha256_text(f"{role_seed_sha256}|client-rank|{client}"),
        )
    )


def _ranked_share_maps(client_rank_order: Sequence[str]) -> tuple[dict[str, int], dict[str, int]]:
    if tuple(sorted(client_rank_order)) != tuple(sorted(CLIENT_NAMES)):
        raise ClientPartitionError("client rank order is not a permutation")
    positive = {
        client: POSITIVE_SHARE_PERCENT[rank] for (rank, client) in enumerate(client_rank_order)
    }
    negative = {
        client: NEGATIVE_SHARE_PERCENT[rank] for (rank, client) in enumerate(client_rank_order)
    }
    return (positive, negative)


def _objective_numerator(
    counts: Mapping[str, Mapping[str, int]],
    *,
    total_positive: int,
    total_negative: int,
    positive_share: Mapping[str, int],
    negative_share: Mapping[str, int],
) -> int:
    positive_error = sum(
        (
            (100 * counts[client]["positive"] - positive_share[client] * total_positive) ** 2
            for client in CLIENT_NAMES
        )
    )
    negative_error = sum(
        (
            (100 * counts[client]["negative"] - negative_share[client] * total_negative) ** 2
            for client in CLIENT_NAMES
        )
    )
    return (
        positive_error * total_negative * total_negative
        + negative_error * total_positive * total_positive
    )


def _constraint_violations(counts: Mapping[str, Mapping[str, int]]) -> list[str]:
    violations: list[str] = []
    for client in CLIENT_NAMES:
        values = counts[client]
        if values["rows"] < MIN_ROWS_PER_CLIENT:
            violations.append(f"{client}:rows<{MIN_ROWS_PER_CLIENT}")
        if values["positive"] < MIN_POSITIVE_PER_CLIENT:
            violations.append(f"{client}:positive<{MIN_POSITIVE_PER_CLIENT}")
        if values["negative"] < MIN_NEGATIVE_PER_CLIENT:
            violations.append(f"{client}:negative<{MIN_NEGATIVE_PER_CLIENT}")
    return violations


def _build_candidate(
    groups: Sequence[GroupRecord],
    *,
    attempt_seed_sha256: str,
    total_positive: int,
    total_negative: int,
    positive_share: Mapping[str, int],
    negative_share: Mapping[str, int],
) -> tuple[dict[str, str], dict[str, dict[str, int]], int, list[str]]:
    counts = {client: {"rows": 0, "positive": 0, "negative": 0} for client in CLIENT_NAMES}
    assignment: dict[str, str] = {}
    ordered_groups = sorted(
        groups,
        key=lambda record: (
            -record.n,
            -abs(record.positive - record.negative),
            _sha256_text(f"{attempt_seed_sha256}|group-order|{record.group_id}"),
        ),
    )
    for record in ordered_groups:
        before = _objective_numerator(
            counts,
            total_positive=total_positive,
            total_negative=total_negative,
            positive_share=positive_share,
            negative_share=negative_share,
        )
        choices: list[tuple[int, str, str]] = []
        for client in CLIENT_NAMES:
            values = counts[client]
            values["rows"] += record.n
            values["positive"] += record.positive
            values["negative"] += record.negative
            after = _objective_numerator(
                counts,
                total_positive=total_positive,
                total_negative=total_negative,
                positive_share=positive_share,
                negative_share=negative_share,
            )
            values["rows"] -= record.n
            values["positive"] -= record.positive
            values["negative"] -= record.negative
            tie = _sha256_text(f"{attempt_seed_sha256}|client-choice|{record.group_id}|{client}")
            choices.append((after - before, tie, client))
        chosen = min(choices)[2]
        assignment[record.group_id] = chosen
        counts[chosen]["rows"] += record.n
        counts[chosen]["positive"] += record.positive
        counts[chosen]["negative"] += record.negative
    objective = _objective_numerator(
        counts,
        total_positive=total_positive,
        total_negative=total_negative,
        positive_share=positive_share,
        negative_share=negative_share,
    )
    return (assignment, counts, objective, _constraint_violations(counts))


def _client_entry(
    client: str,
    assignment: Mapping[str, str],
    root_groups: Mapping[str, GroupRecord],
    *,
    total_positive: int,
    total_negative: int,
    positive_share: Mapping[str, int],
    negative_share: Mapping[str, int],
) -> dict[str, Any]:
    ids = sorted((group_id for (group_id, value) in assignment.items() if value == client))
    records = [root_groups[group_id] for group_id in ids]
    rows = sum((record.n for record in records))
    positive = sum((record.positive for record in records))
    negative = sum((record.negative for record in records))
    return {
        "group_ids": ids,
        "group_count": len(ids),
        "row_count": rows,
        "positive": positive,
        "negative": negative,
        "prevalence": positive / rows,
        "membership_sha256": _membership_sha256(ids),
        "target_positive_share": positive_share[client] / 100.0,
        "target_negative_share": negative_share[client] / 100.0,
        "actual_positive_share": positive / total_positive,
        "actual_negative_share": negative / total_negative,
        "positive_share_error": positive / total_positive - positive_share[client] / 100.0,
        "negative_share_error": negative / total_negative - negative_share[client] / 100.0,
    }


def _aggregate_diagnostics(
    clients: Mapping[str, Mapping[str, Any]], *, total_rows: int, total_positive: int
) -> dict[str, Any]:
    prevalences = [float(clients[client]["prevalence"]) for client in CLIENT_NAMES]
    positive_shares = [float(clients[client]["actual_positive_share"]) for client in CLIENT_NAMES]
    negative_shares = [float(clients[client]["actual_negative_share"]) for client in CLIENT_NAMES]
    target_positive = [float(clients[client]["target_positive_share"]) for client in CLIENT_NAMES]
    target_negative = [float(clients[client]["target_negative_share"]) for client in CLIENT_NAMES]
    mean_prevalence = sum(prevalences) / len(prevalences)
    return {
        "diagnostic_only_not_used_for_model_or_protocol_selection": True,
        "global_prevalence": total_positive / total_rows,
        "client_prevalence_min": min(prevalences),
        "client_prevalence_max": max(prevalences),
        "client_prevalence_range": max(prevalences) - min(prevalences),
        "client_prevalence_population_std": math.sqrt(
            sum(((value - mean_prevalence) ** 2 for value in prevalences)) / len(prevalences)
        ),
        "target_label_distribution_total_variation": 0.5
        * sum(
            (
                abs(positive - negative)
                for (positive, negative) in zip(target_positive, target_negative)
            )
        ),
        "attained_label_distribution_total_variation": 0.5
        * sum(
            (
                abs(positive - negative)
                for (positive, negative) in zip(positive_shares, negative_shares)
            )
        ),
        "positive_share_l1_error": sum(
            (abs(float(clients[client]["positive_share_error"])) for client in CLIENT_NAMES)
        ),
        "negative_share_l1_error": sum(
            (abs(float(clients[client]["negative_share_error"])) for client in CLIENT_NAMES)
        ),
        "label_shift_diagnostic_fields": [
            "client_prevalence_range",
            "client_prevalence_population_std",
            "attained_label_distribution_total_variation",
            "positive_share_l1_error",
            "negative_share_l1_error",
        ],
    }


def _construct_client_partition(
    group_manifest: Mapping[str, Any],
    group_ids: Sequence[str],
    *,
    study_id: str,
    scope: str,
    outer_repeat: int,
    outer_fold: int,
    inner_fold: int,
    role: str,
    parent_split_sha256: str,
) -> dict[str, Any]:
    if not isinstance(group_manifest, Mapping):
        raise ClientPartitionError("an auditable serialized group manifest is required")
    root_groups = _root_group_map(group_manifest)
    ordered_ids = sorted(group_ids)
    membership_sha256 = _membership_sha256(ordered_ids)
    if not set(ordered_ids) <= set(root_groups):
        raise ClientPartitionError("client input membership contains an unknown group")
    groups = tuple((root_groups[group_id] for group_id in ordered_ids))
    total_rows = sum((group.n for group in groups))
    total_positive = sum((group.positive for group in groups))
    total_negative = sum((group.negative for group in groups))
    preliminary = []
    if len(groups) < len(CLIENT_NAMES):
        preliminary.append("fewer groups than clients")
    if total_rows < len(CLIENT_NAMES) * MIN_ROWS_PER_CLIENT:
        preliminary.append("insufficient total rows for per-client minimum")
    if total_positive < len(CLIENT_NAMES) * MIN_POSITIVE_PER_CLIENT:
        preliminary.append("insufficient positive rows for per-client minimum")
    if total_negative < len(CLIENT_NAMES) * MIN_NEGATIVE_PER_CLIENT:
        preliminary.append("insufficient negative rows for per-client minimum")
    if preliminary:
        raise ClientPartitionInfeasibleError("; ".join(preliminary))
    source_sha256 = _require_sha256(group_manifest.get("source_sha256"), "source hash")
    group_manifest_sha256 = _require_sha256(
        group_manifest.get("group_manifest_sha256"), "group-manifest hash"
    )
    row_to_group_sha256 = _require_sha256(
        group_manifest.get("row_to_group_sha256"), "row-to-group hash"
    )
    seed_chain = _seed_chain(
        study_id=study_id,
        dataset=group_manifest.get("dataset"),
        source_sha256=source_sha256,
        group_manifest_sha256=group_manifest_sha256,
        row_to_group_sha256=row_to_group_sha256,
        selected_group_membership_sha256=membership_sha256,
        parent_split_sha256=parent_split_sha256,
        outer_repeat=outer_repeat,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        role=role,
        scope=scope,
    )
    client_rank_order = _client_rank_order(seed_chain["role_seed_sha256"])
    positive_share, negative_share = _ranked_share_maps(client_rank_order)
    objective_denominator = (
        10000 * total_positive * total_positive * total_negative * total_negative
    )
    candidate_audit: list[dict[str, Any]] = []
    accepted: list[tuple[int, str, int, dict[str, str]]] = []
    for attempt in range(CANDIDATE_ATTEMPTS):
        attempt_seed = _sha256_text(f"{seed_chain['role_seed_sha256']}|attempt|{attempt}")
        assignment, _, objective, violations = _build_candidate(
            groups,
            attempt_seed_sha256=attempt_seed,
            total_positive=total_positive,
            total_negative=total_negative,
            positive_share=positive_share,
            negative_share=negative_share,
        )
        assignment_sha256 = _sha256_json(
            sorted(((group_id, assignment[group_id]) for group_id in assignment))
        )
        entry = {
            "attempt": attempt,
            "attempt_seed_sha256": attempt_seed,
            "status": "ACCEPTED" if not violations else "REJECTED_CONSTRAINT",
            "assignment_sha256": assignment_sha256,
            "objective_numerator": objective,
            "objective_denominator": objective_denominator,
            "constraint_violations": violations,
        }
        candidate_audit.append(entry)
        if not violations:
            accepted.append((objective, assignment_sha256, attempt, assignment))
    if not accepted:
        raise ClientPartitionInfeasibleError(
            "no one of 64 registered deterministic attempts satisfies all constraints"
        )
    selected_objective, selected_assignment_sha256, selected_attempt, assignment = min(
        accepted, key=lambda value: (value[0], value[1])
    )
    clients = {
        client: _client_entry(
            client,
            assignment,
            root_groups,
            total_positive=total_positive,
            total_negative=total_negative,
            positive_share=positive_share,
            negative_share=negative_share,
        )
        for client in CLIENT_NAMES
    }
    diagnostics = _aggregate_diagnostics(
        clients, total_rows=total_rows, total_positive=total_positive
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "result_blind_selection": True,
        "performance_fields_used": [],
        "label_stratified": True,
        "label_fields_used": ["positive", "negative", "prevalence"],
        "policy": _policy(),
        "input_binding": {
            "dataset": group_manifest.get("dataset"),
            "source_sha256": source_sha256,
            "root_group_manifest_sha256": group_manifest_sha256,
            "root_row_to_group_sha256": row_to_group_sha256,
            "parent_split_sha256": _require_sha256(parent_split_sha256, "parent split hash"),
            "selected_group_membership_sha256": membership_sha256,
            "selected_group_count": len(groups),
            "selected_row_count": total_rows,
            "selected_positive": total_positive,
            "selected_negative": total_negative,
        },
        "seed_chain": seed_chain,
        "client_rank_order": list(client_rank_order),
        "target_positive_share_percent": positive_share,
        "target_negative_share_percent": negative_share,
        "candidate_attempts": CANDIDATE_ATTEMPTS,
        "accepted_candidates": len(accepted),
        "selected_attempt": selected_attempt,
        "selected_objective_numerator": selected_objective,
        "objective_denominator": objective_denominator,
        "selected_assignment_sha256": selected_assignment_sha256,
        "assignment": dict(sorted(assignment.items())),
        "clients": clients,
        "aggregate_diagnostics": diagnostics,
        "candidate_audit": candidate_audit,
        "candidate_audit_sha256": _sha256_json(candidate_audit),
    }
    payload["split_sha256"] = _sha256_json(payload)
    return payload


def generate_client_partition(
    group_manifest: Mapping[str, Any],
    group_ids: Sequence[str],
    *,
    study_id: str,
    scope: str,
    outer_repeat: int,
    outer_fold: int,
    inner_fold: int,
    role: str,
    parent_split_sha256: str,
) -> dict[str, Any]:
    """Generate the registered five-client group-safe label-skew partition."""
    return _construct_client_partition(
        group_manifest,
        group_ids,
        study_id=study_id,
        scope=scope,
        outer_repeat=outer_repeat,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        role=role,
        parent_split_sha256=parent_split_sha256,
    )


def validate_client_partition(
    partition: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
    group_ids: Sequence[str],
    *,
    study_id: str,
    scope: str,
    outer_repeat: int,
    outer_fold: int,
    inner_fold: int,
    role: str,
    parent_split_sha256: str,
) -> None:
    """Rebuild and compare every assignment, membership, diagnostic and hash."""
    if not isinstance(partition, Mapping):
        raise ClientPartitionError("client partition must be a mapping")
    if partition.get("schema") != SCHEMA:
        raise ClientPartitionError("client partition schema differs")
    stored = _require_sha256(partition.get("split_sha256"), "client split hash")
    payload = {str(key): value for (key, value) in partition.items() if key != "split_sha256"}
    if not hmac.compare_digest(stored, _sha256_json(payload)):
        raise ClientPartitionError("client partition hash mismatch")
    expected = _construct_client_partition(
        group_manifest,
        group_ids,
        study_id=study_id,
        scope=scope,
        outer_repeat=outer_repeat,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        role=role,
        parent_split_sha256=parent_split_sha256,
    )
    if dict(partition) != expected:
        raise ClientPartitionError(
            "client partition differs from the registered deterministic reconstruction"
        )


__all__ = [
    "CANDIDATE_ATTEMPTS",
    "CLIENT_NAMES",
    "ClientPartitionError",
    "ClientPartitionInfeasibleError",
    "MIN_NEGATIVE_PER_CLIENT",
    "MIN_POSITIVE_PER_CLIENT",
    "MIN_ROWS_PER_CLIENT",
    "NEGATIVE_SHARE_PERCENT",
    "POLICY_NAME",
    "POSITIVE_SHARE_PERCENT",
    "client_partition_policy",
    "client_partition_policy_fingerprint",
    "generate_client_partition",
    "validate_client_partition",
]
