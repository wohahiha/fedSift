"""Deterministic result-blind balanced partitioning of indivisible groups."""

from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Mapping, Sequence
from .group_manifest import GroupRecord, group_records


class SplitPlanningError(RuntimeError):
    """Raised when a split request is malformed."""


class InfeasiblePartitionError(SplitPlanningError):
    """Raised with a result-blind certificate when all 64 candidates fail."""

    def __init__(self, certificate: Mapping[str, Any]) -> None:
        self.certificate = dict(certificate)
        super().__init__(
            "no fixed result-blind partition candidate satisfies the registered constraints"
        )


@dataclass(frozen=True)
class BalanceConstraints:
    max_n_fraction: float = 0.02
    max_class_fraction: float = 0.03
    max_prevalence_deviation: float = 0.02
    min_positive_per_part: int = 1
    min_negative_per_part: int = 1
    required_candidate_attempts: int = 64

    def validate(self) -> None:
        for name, value in (
            ("max_n_fraction", self.max_n_fraction),
            ("max_class_fraction", self.max_class_fraction),
            ("max_prevalence_deviation", self.max_prevalence_deviation),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or (not math.isfinite(float(value)))
                or (float(value) < 0.0)
            ):
                raise SplitPlanningError(f"{name} must be finite and nonnegative")
        if (
            isinstance(self.min_positive_per_part, bool)
            or not isinstance(self.min_positive_per_part, Integral)
            or isinstance(self.min_negative_per_part, bool)
            or (not isinstance(self.min_negative_per_part, Integral))
            or (self.min_positive_per_part < 1)
            or (self.min_negative_per_part < 1)
        ):
            raise SplitPlanningError("each part must require both classes")
        if (
            isinstance(self.required_candidate_attempts, bool)
            or not isinstance(self.required_candidate_attempts, Integral)
            or self.required_candidate_attempts != 64
        ):
            raise SplitPlanningError("FedSift requires exactly 64 split candidates")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _seed_material(context: Mapping[str, Any], attempt: int) -> str:
    required = (
        "study_id",
        "dataset_sha256",
        "group_manifest_sha256",
        "scope",
        "outer_repeat",
        "outer_fold",
        "inner_fold",
    )
    if set(context) != set(required):
        raise SplitPlanningError("split context fields differ from the registered schema")
    for name in ("study_id", "scope"):
        if not isinstance(context[name], str) or not context[name]:
            raise SplitPlanningError(f"split context {name} must be a non-empty string")
    for name in ("dataset_sha256", "group_manifest_sha256"):
        value = context[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any((character not in "0123456789abcdef" for character in value))
        ):
            raise SplitPlanningError(f"split context {name} must be lowercase SHA-256")
    for name in ("outer_repeat", "outer_fold", "inner_fold"):
        if isinstance(context[name], bool) or not isinstance(context[name], Integral):
            raise SplitPlanningError(f"split context {name} must be an exact integer")
    return _canonical_json(
        {
            "domain": _identity("result_blind_balanced_partition"),
            "context": dict(context),
            "attempt": attempt,
        }
    )


def _normalized_load_score(
    loads: Sequence[dict[str, float]], targets: Sequence[dict[str, float]]
) -> tuple[float, float, float, float]:
    deviations: list[float] = []
    overfills: list[float] = []
    squared = 0.0
    for load, target in zip(loads, targets):
        for key in ("n", "positive", "negative", "groups"):
            denominator = max(float(target[key]), 1.0)
            deviation = abs(float(load[key]) - float(target[key])) / denominator
            overfill = max(0.0, float(load[key]) - float(target[key])) / denominator
            deviations.append(deviation)
            overfills.append(overfill)
            squared += deviation * deviation
    return (max(overfills), sum(overfills), max(deviations), squared)


def _candidate_assignment(
    groups: Sequence[GroupRecord],
    part_names: Sequence[str],
    weights: Sequence[float],
    context: Mapping[str, Any],
    attempt: int,
) -> dict[str, str]:
    seed_hex = _sha256_text(_seed_material(context, attempt))
    total_n = sum((group.n for group in groups))
    total_positive = sum((group.positive for group in groups))
    total_negative = sum((group.negative for group in groups))
    total_groups = len(groups)
    weight_sum = sum(weights)
    targets = [
        {
            "n": total_n * weight / weight_sum,
            "positive": total_positive * weight / weight_sum,
            "negative": total_negative * weight / weight_sum,
            "groups": total_groups * weight / weight_sum,
        }
        for weight in weights
    ]
    loads = [{"n": 0.0, "positive": 0.0, "negative": 0.0, "groups": 0.0} for _ in part_names]

    def group_key(group: GroupRecord) -> tuple[Any, ...]:
        tie = _sha256_text(f"{seed_hex}|group|{group.group_id}")
        return (-group.n, -max(group.positive, group.negative), tie, group.group_id)

    assignment: dict[str, str] = {}
    for group in sorted(groups, key=group_key):
        choices: list[tuple[Any, ...]] = []
        for part_index, part_name in enumerate(part_names):
            trial = [dict(value) for value in loads]
            trial[part_index]["n"] += group.n
            trial[part_index]["positive"] += group.positive
            trial[part_index]["negative"] += group.negative
            trial[part_index]["groups"] += 1
            score = _normalized_load_score(trial, targets)
            part_tie = _sha256_text(f"{seed_hex}|part|{group.group_id}|{part_name}")
            choices.append((*score, part_tie, part_index))
        selected = min(choices)[-1]
        loads[selected]["n"] += group.n
        loads[selected]["positive"] += group.positive
        loads[selected]["negative"] += group.negative
        loads[selected]["groups"] += 1
        assignment[group.group_id] = part_names[selected]
    return assignment


def _canonicalize_equal_parts(
    assignment: Mapping[str, str], part_names: Sequence[str], weights: Sequence[float]
) -> dict[str, str]:
    if max(weights) - min(weights) > 1e-15:
        return dict(assignment)
    signatures: list[tuple[str, str]] = []
    for part_name in part_names:
        members = sorted((group_id for (group_id, part) in assignment.items() if part == part_name))
        signatures.append((_sha256_text("\n".join(members)), part_name))
    ordered_old = [part for (_, part) in sorted(signatures)]
    remap = {old: new for (old, new) in zip(ordered_old, part_names)}
    return {group_id: remap[part] for (group_id, part) in assignment.items()}


def _partition_stats(
    groups: Sequence[GroupRecord], assignment: Mapping[str, str], part_names: Sequence[str]
) -> dict[str, dict[str, Any]]:
    stats = {
        name: {"n": 0, "positive": 0, "negative": 0, "groups": 0, "group_ids": []}
        for name in part_names
    }
    for group in groups:
        part = assignment[group.group_id]
        row = stats[part]
        row["n"] += group.n
        row["positive"] += group.positive
        row["negative"] += group.negative
        row["groups"] += 1
        row["group_ids"].append(group.group_id)
    for row in stats.values():
        row["group_ids"].sort()
        row["prevalence"] = row["positive"] / row["n"] if row["n"] else None
        row["membership_sha256"] = _sha256_text("\n".join(row["group_ids"]))
    return stats


def _evaluate_candidate(
    groups: Sequence[GroupRecord],
    assignment: Mapping[str, str],
    part_names: Sequence[str],
    weights: Sequence[float],
    constraints: BalanceConstraints,
    *,
    attempt: int,
) -> dict[str, Any]:
    stats = _partition_stats(groups, assignment, part_names)
    total_n = sum((group.n for group in groups))
    total_positive = sum((group.positive for group in groups))
    total_negative = sum((group.negative for group in groups))
    total_groups = len(groups)
    weight_sum = sum(weights)
    overall_prevalence = total_positive / total_n
    max_group_n = max((group.n for group in groups))
    max_group_positive = max((group.positive for group in groups))
    max_group_negative = max((group.negative for group in groups))
    deviations = {"n": [], "positive": [], "negative": [], "groups": []}
    normalized_sse = 0.0
    failures: list[dict[str, Any]] = []
    for name, weight in zip(part_names, weights):
        row = stats[name]
        targets = {
            "n": total_n * weight / weight_sum,
            "positive": total_positive * weight / weight_sum,
            "negative": total_negative * weight / weight_sum,
            "groups": total_groups * weight / weight_sum,
        }
        for key in deviations:
            difference = abs(float(row[key]) - targets[key])
            deviations[key].append(difference)
            normalized_sse += (difference / max(targets[key], 1.0)) ** 2
        n_allowance = max(max_group_n, math.ceil(constraints.max_n_fraction * targets["n"]))
        p_allowance = max(
            max_group_positive, math.ceil(constraints.max_class_fraction * targets["positive"])
        )
        q_allowance = max(
            max_group_negative, math.ceil(constraints.max_class_fraction * targets["negative"])
        )
        prevalence_ok = (
            row["prevalence"] is not None
            and abs(float(row["prevalence"]) - overall_prevalence)
            <= constraints.max_prevalence_deviation
        )
        checks = {
            "n_balance": deviations["n"][-1] <= n_allowance,
            "positive_balance": deviations["positive"][-1] <= p_allowance,
            "negative_balance": deviations["negative"][-1] <= q_allowance,
            "minimum_positive": row["positive"] >= constraints.min_positive_per_part,
            "minimum_negative": row["negative"] >= constraints.min_negative_per_part,
            "nonempty": row["n"] > 0,
            "prevalence": prevalence_ok,
        }
        for check, passed in checks.items():
            if not passed:
                failures.append({"part": name, "check": check})
    assignment_hash = _sha256_text(
        _canonical_json(sorted(((group_id, part) for (group_id, part) in assignment.items())))
    )
    largest_count_deviations = sorted(
        (max(deviations["n"]), max(deviations["positive"]), max(deviations["negative"])),
        reverse=True,
    )
    prevalence_deviations = [
        abs(float(row["prevalence"]) - overall_prevalence)
        for row in stats.values()
        if row["prevalence"] is not None
    ]
    max_prevalence_deviation = (
        max(prevalence_deviations) if len(prevalence_deviations) == len(stats) else 1.0
    )
    ranking = (
        *largest_count_deviations,
        max_prevalence_deviation,
        max(deviations["groups"]),
        normalized_sse,
        assignment_hash,
    )
    public_stats = {
        name: {key: value for (key, value) in row.items() if key != "group_ids"}
        for (name, row) in stats.items()
    }
    return {
        "attempt": attempt,
        "accepted": not failures,
        "failures": failures,
        "assignment_sha256": assignment_hash,
        "ranking": list(ranking),
        "max_prevalence_deviation": max_prevalence_deviation,
        "normalized_sse": normalized_sse,
        "parts": public_stats,
    }


def generate_balanced_partition(
    group_manifest: Mapping[str, Any],
    *,
    part_names: Sequence[str],
    weights: Sequence[float],
    context: Mapping[str, Any],
    constraints: BalanceConstraints | None = None,
    candidate_attempts: int = 64,
) -> dict[str, Any]:
    """Generate and select exactly 64 result-blind partition candidates."""
    registered = constraints or BalanceConstraints()
    registered.validate()
    if (
        isinstance(candidate_attempts, bool)
        or not isinstance(candidate_attempts, Integral)
        or candidate_attempts != registered.required_candidate_attempts
    ):
        raise SplitPlanningError("candidate attempt count differs from the registered 64")
    if len(part_names) < 2 or len(part_names) != len(weights):
        raise SplitPlanningError("part names and weights must have the same length >= 2")
    if any((not isinstance(name, str) or not name for name in part_names)) or len(
        set(part_names)
    ) != len(part_names):
        raise SplitPlanningError("part names must be unique non-empty strings")
    if any(
        (
            isinstance(weight, bool)
            or not isinstance(weight, Real)
            or (not math.isfinite(float(weight)))
            or (float(weight) <= 0.0)
            for weight in weights
        )
    ):
        raise SplitPlanningError("partition weights must be finite and positive")
    normalized_weights = tuple((float(weight) for weight in weights))
    if not math.isfinite(sum(normalized_weights)):
        raise SplitPlanningError("partition weight sum must be finite")
    if not isinstance(group_manifest, Mapping):
        raise SplitPlanningError("an auditable serialized group manifest is required")
    _seed_material(context, 0)
    groups = group_records(group_manifest)
    source_sha256 = group_manifest.get("source_sha256")
    group_manifest_sha256 = group_manifest.get("group_manifest_sha256")
    if context.get("dataset_sha256") != source_sha256:
        raise SplitPlanningError("split context dataset hash differs from manifest")
    if context.get("group_manifest_sha256") != group_manifest_sha256:
        raise SplitPlanningError("split context group-manifest hash differs from manifest")
    if len(groups) < len(part_names):
        raise SplitPlanningError("fewer groups than requested parts")
    candidates: list[dict[str, Any]] = []
    assignments: dict[int, dict[str, str]] = {}
    for attempt in range(candidate_attempts):
        assignment = _candidate_assignment(groups, part_names, normalized_weights, context, attempt)
        assignment = _canonicalize_equal_parts(assignment, part_names, normalized_weights)
        candidate = _evaluate_candidate(
            groups, assignment, part_names, normalized_weights, registered, attempt=attempt
        )
        candidates.append(candidate)
        assignments[attempt] = assignment
    accepted = [candidate for candidate in candidates if candidate["accepted"]]
    context_hash = _sha256_text(_canonical_json(dict(context)))
    if not accepted:
        raise InfeasiblePartitionError(
            {
                "schema": _identity("partition_infeasibility"),
                "context_sha256": context_hash,
                "candidate_attempts": candidate_attempts,
                "constraint": registered.__dict__,
                "result_blind_selection": True,
                "performance_fields_used": [],
                "label_stratified": True,
                "label_fields_used": ["positive", "negative", "prevalence"],
                "candidate_summaries": candidates,
                "automatic_relaxation": False,
            }
        )
    selected = min(accepted, key=lambda value: tuple(value["ranking"]))
    selected_assignment = assignments[int(selected["attempt"])]
    return {
        "schema": _identity("balanced_group_partition"),
        "result_blind_selection": True,
        "performance_fields_used": [],
        "label_stratified": True,
        "label_fields_used": ["positive", "negative", "prevalence"],
        "context": dict(context),
        "context_sha256": context_hash,
        "candidate_attempts": candidate_attempts,
        "accepted_candidates": len(accepted),
        "selected_attempt": int(selected["attempt"]),
        "selected_assignment_sha256": selected["assignment_sha256"],
        "part_names": list(part_names),
        "weights": list(normalized_weights),
        "constraints": registered.__dict__,
        "parts": selected["parts"],
        "assignment": dict(sorted(selected_assignment.items())),
        "candidate_audit": candidates,
    }
