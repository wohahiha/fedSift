"""Result-blind HPO planning with exact upstream and outer-fold bindings.

The unit inventory is a deterministic Cartesian product of an exact frozen
candidate space and an exact frozen nested plan. Candidate eligibility is
closed independently inside every ``(outer_repeat, outer_fold, method)`` scope;
there is no cross-outer authorization path. A selection authorization is valid
only when it is recomputed from the original terminal ledger.
"""

from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import copy
import hashlib
import math
from typing import Any, Mapping, Sequence
from .candidate_space import (
    MAIN_METHODS,
    CandidateSpaceError,
    assert_no_performance_fields,
    canonical_sha256,
    require_exact_int,
    require_sha256,
    validate_candidate_space,
)
from .nested_plan import NestedPlanError, validate_nested_plan
from .preprocessing import PreprocessingError, validate_preprocessing_profile_spec


class HpoPlanError(ValueError):
    """Raised when an HPO plan, ledger, or selection gate is invalid."""


DEFAULT_HPO_SEEDS: tuple[int, ...] = (1103, 1907, 2311)
NONRETRYABLE_FAILURE_CODES = frozenset(
    {
        "nonfinite_update",
        "numerical_instability",
        "privacy_accounting_failure",
        "privacy_target_mismatch",
        "scientific_contract_failure",
    }
)
INFRASTRUCTURE_FAILURE_CODES = frozenset(
    {"infrastructure_failure", "worker_interruption", "transient_io_failure"}
)
MATCHED_ABLATIONS: dict[str, dict[str, object]] = {
    "fedsift_uniform_schedule": {
        "parent_method": "fedsift",
        "mechanism_switch": {"time_schedule_enabled": False},
        "inherited_parent_fields": [
            "candidate_id",
            "candidate_sha256",
            "parameters.local_optimizer",
            "parameters.backend",
            "parameters.record_dp",
            "parameters.control_rule",
            "parameters.sift",
        ],
    },
    "fedsift_without_sift": {
        "parent_method": "fedsift",
        "mechanism_switch": {
            "changed_component": "public_control_sift_rule",
            "parent_value": "fedsift_supported_override",
            "replacement_value": "no_public_control_fixed_full_step",
            "sift_enabled": False,
            "public_control_queries_enabled": False,
            "fixed_alpha": 1.0,
            "non_query_round_behavior": "fixed_full_step",
        },
        "inherited_parent_fields": [
            "candidate_id",
            "candidate_sha256",
            "parameters.local_optimizer",
            "parameters.backend",
            "parameters.record_dp",
            "parameters.privacy_schedule",
            "parameters.sift",
        ],
    },
    "fedsift_public_argmin_rule": {
        "parent_method": "fedsift",
        "mechanism_switch": {
            "changed_component": "control_rule.selection_rule",
            "parent_value": "fedsift_supported_override",
            "replacement_value": "public_argmin_mean_logloss",
            "replacement_contract": {
                "tie_break": "larger_step",
                "safety_margin": "none",
                "uses_sift_safety_terms": False,
            },
        },
        "inherited_parent_fields": [
            "candidate_id",
            "candidate_sha256",
            "candidate_grid",
            "parameters.local_optimizer",
            "parameters.backend",
            "parameters.record_dp",
            "parameters.privacy_schedule",
            "parameters.control_rule.records",
            "parameters.control_rule.labels_visible",
            "parameters.control_rule.query_every_rounds",
            "parameters.control_rule.step_candidates",
        ],
    },
}
_PLAN_TOP_LEVEL_FIELDS = {
    "schema",
    "status",
    "study_id",
    "dataset_id",
    "bindings",
    "method_order",
    "nested_design_binding",
    "selection_contract",
    "equal_budget_contract",
    "method_contracts",
    "expected_unit_count",
    "units",
    "hpo_plan_sha256",
}


def _unit_id(identity: Mapping[str, object]) -> str:
    digest = hashlib.sha256(
        (_identity("hpo_unit_domain") + canonical_sha256(identity)).encode("ascii")
    ).hexdigest()
    return f"hpo_{digest[:24]}"


def _artifact_hash(value: Mapping[str, object], field: str) -> str:
    payload = copy.deepcopy(dict(value))
    payload.pop(field, None)
    return canonical_sha256(payload)


def _strict_seeds(values: Sequence[int]) -> list[int]:
    if isinstance(values, (str, bytes)):
        raise HpoPlanError("HPO seeds must be a sequence of exact integers")
    seeds: list[int] = []
    for index, raw_seed in enumerate(values):
        try:
            seed = require_exact_int(raw_seed, f"hpo_seeds[{index}]", minimum=0)
        except CandidateSpaceError as exc:
            raise HpoPlanError(str(exc)) from exc
        if seed > 2**63 - 1:
            raise HpoPlanError("HPO seeds must fit a signed 64-bit integer")
        seeds.append(seed)
    if not seeds or len(set(seeds)) != len(seeds):
        raise HpoPlanError("HPO seeds must be non-empty and unique")
    return seeds


def _validate_upstreams(
    candidate_space: Mapping[str, object],
    nested_plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
    preprocessing_profile_spec: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        validate_candidate_space(candidate_space, require_frozen=True)
    except CandidateSpaceError as exc:
        raise HpoPlanError(str(exc)) from exc
    try:
        validate_nested_plan(nested_plan, group_manifest, require_frozen=True)
    except NestedPlanError as exc:
        raise HpoPlanError(str(exc)) from exc
    candidate_study = candidate_space.get("study_id")
    nested_study = nested_plan.get("study_id")
    if (
        not isinstance(candidate_study, str)
        or not candidate_study.strip()
        or nested_study != candidate_study
    ):
        raise HpoPlanError("candidate space and nested plan must share one study_id")
    source = nested_plan.get("source_binding")
    design = nested_plan.get("design")
    if not isinstance(source, Mapping) or not isinstance(design, Mapping):
        raise HpoPlanError("nested plan lacks source or design bindings")
    dataset_id = source.get("dataset")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise HpoPlanError("nested plan dataset identity is missing")
    feature_names = group_manifest.get("feature_names")
    if not isinstance(feature_names, list):
        raise HpoPlanError("group manifest exact feature names are missing")
    try:
        validate_preprocessing_profile_spec(
            preprocessing_profile_spec, dataset=dataset_id, feature_names=feature_names
        )
    except PreprocessingError as exc:
        raise HpoPlanError(str(exc)) from exc
    try:
        dataset_sha256 = require_sha256(source.get("source_sha256"), "source_sha256")
        group_sha256 = require_sha256(source.get("group_manifest_sha256"), "group_manifest_sha256")
        row_group_sha256 = require_sha256(source.get("row_to_group_sha256"), "row_to_group_sha256")
        nested_sha256 = require_sha256(nested_plan.get("nested_plan_sha256"), "nested_plan_sha256")
        outer_repeats = require_exact_int(design.get("outer_repeats"), "outer_repeats", minimum=1)
        outer_folds = require_exact_int(design.get("outer_folds"), "outer_folds", minimum=2)
        inner_folds = require_exact_int(design.get("inner_folds"), "inner_folds", minimum=2)
    except CandidateSpaceError as exc:
        raise HpoPlanError(str(exc)) from exc
    freeze_bindings = candidate_space.get("freeze_bindings")
    if not isinstance(freeze_bindings, Mapping):
        raise HpoPlanError("candidate space lacks exact freeze bindings")
    try:
        protocol_sha256 = require_sha256(freeze_bindings.get("protocol_sha256"), "protocol_sha256")
        implementation_sha256 = require_sha256(
            freeze_bindings.get("implementation_contract_sha256"), "implementation_contract_sha256"
        )
    except CandidateSpaceError as exc:
        raise HpoPlanError(str(exc)) from exc
    nested_freeze = nested_plan.get("freeze_binding")
    if not isinstance(nested_freeze, Mapping):
        raise HpoPlanError("frozen nested plan lacks its exact freeze binding")
    try:
        nested_freeze_sha256 = require_sha256(
            nested_freeze.get("freeze_binding_sha256"), "nested_freeze_binding_sha256"
        )
        nested_protocol_sha256 = require_sha256(
            nested_freeze.get("protocol_sha256"), "nested_protocol_sha256"
        )
        nested_implementation_sha256 = require_sha256(
            nested_freeze.get("implementation_sha256"), "nested_implementation_sha256"
        )
        client_policy_sha256 = require_sha256(
            nested_freeze.get("client_partition_policy_sha256"), "client_partition_policy_sha256"
        )
    except CandidateSpaceError as exc:
        raise HpoPlanError(str(exc)) from exc
    if nested_protocol_sha256 != protocol_sha256:
        raise HpoPlanError("candidate and nested plans bind different protocols")
    if nested_implementation_sha256 != implementation_sha256:
        raise HpoPlanError("candidate and nested plans bind different implementations")
    return {
        "study_id": candidate_study,
        "dataset_id": dataset_id,
        "dataset_sha256": dataset_sha256,
        "group_manifest_sha256": group_sha256,
        "row_to_group_sha256": row_group_sha256,
        "nested_plan_sha256": nested_sha256,
        "nested_plan_status": nested_plan.get("status"),
        "nested_freeze_binding_sha256": nested_freeze_sha256,
        "client_partition_policy_sha256": client_policy_sha256,
        "client_partition_policy": design.get("client_partition_policy"),
        "outer_repeats": outer_repeats,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "protocol_sha256": protocol_sha256,
        "implementation_contract_sha256": implementation_sha256,
        "preprocessing_profile_name": preprocessing_profile_spec["name"],
        "preprocessing_profile_sha256": preprocessing_profile_spec["profile_sha256"],
        "preprocessing_profile": copy.deepcopy(dict(preprocessing_profile_spec)),
    }


def _split_bindings(
    nested_plan: Mapping[str, Any], outer_repeat: int, outer_fold: int, inner_fold: int
) -> dict[str, object]:
    repeat_plan = nested_plan["repetitions"][outer_repeat]
    fold_plan = repeat_plan["outer_folds"][outer_fold]
    inner_plan = fold_plan["inner_folds"][inner_fold]
    clients = inner_plan["clients"]
    return {
        "outer_partition_sha256": repeat_plan["outer_partition"]["split_sha256"],
        "inner_partition_sha256": fold_plan["inner_partition"]["split_sha256"],
        "inner_train_membership_sha256": inner_plan["inner_train"]["membership_sha256"],
        "inner_validation_membership_sha256": inner_plan["inner_validation"]["membership_sha256"],
        "role_partition_sha256": inner_plan["role_partition"]["split_sha256"],
        "private_membership_sha256": inner_plan["roles"]["private"]["membership_sha256"],
        "v_ctrl_membership_sha256": inner_plan["roles"]["v_ctrl"]["membership_sha256"],
        "v_sel_membership_sha256": inner_plan["roles"]["v_sel"]["membership_sha256"],
        "client_partition_sha256": inner_plan["client_partition"]["split_sha256"],
        "client_membership_sha256": {
            name: membership["membership_sha256"] for (name, membership) in sorted(clients.items())
        },
    }


def _assemble_hpo_plan(
    candidate_space: Mapping[str, object],
    nested_plan: Mapping[str, Any],
    upstream: Mapping[str, Any],
    *,
    hpo_seeds: Sequence[int],
    max_steps: int,
) -> dict[str, object]:
    seeds = list(hpo_seeds)
    candidate_count = int(candidate_space["candidate_count_per_method"])
    outer_repeats = int(upstream["outer_repeats"])
    outer_folds = int(upstream["outer_folds"])
    inner_folds = int(upstream["inner_folds"])
    minimum_closed_candidates = int(math.ceil(0.75 * candidate_count))
    max_failed_candidates = candidate_count - minimum_closed_candidates
    candidate_space_sha256 = str(candidate_space["manifest_sha256"])
    nested_plan_sha256 = str(upstream["nested_plan_sha256"])
    units: list[dict[str, object]] = []
    methods = candidate_space["methods"]
    assert isinstance(methods, Mapping)
    for method in MAIN_METHODS:
        roster = methods[method]
        assert isinstance(roster, Mapping)
        candidates = roster["candidates"]
        assert isinstance(candidates, list)
        for outer_repeat in range(outer_repeats):
            for outer_fold in range(outer_folds):
                for inner_fold in range(inner_folds):
                    split_bindings = _split_bindings(
                        nested_plan, outer_repeat, outer_fold, inner_fold
                    )
                    for candidate in candidates:
                        for hpo_seed in seeds:
                            identity: dict[str, object] = {
                                "study_id": upstream["study_id"],
                                "dataset_id": upstream["dataset_id"],
                                "dataset_sha256": upstream["dataset_sha256"],
                                "nested_plan_sha256": nested_plan_sha256,
                                "candidate_space_sha256": candidate_space_sha256,
                                "preprocessing_profile_name": upstream[
                                    "preprocessing_profile_name"
                                ],
                                "preprocessing_profile_sha256": upstream[
                                    "preprocessing_profile_sha256"
                                ],
                                "preprocessing_profile": copy.deepcopy(
                                    upstream["preprocessing_profile"]
                                ),
                                "method": method,
                                "outer_repeat": outer_repeat,
                                "outer_fold": outer_fold,
                                "inner_fold": inner_fold,
                                "candidate_id": candidate["candidate_id"],
                                "candidate_sha256": candidate["candidate_sha256"],
                                "hpo_seed": hpo_seed,
                                "max_steps": max_steps,
                                "split_bindings": copy.deepcopy(split_bindings),
                            }
                            units.append({"unit_id": _unit_id(identity), **identity})
    per_method_units = outer_repeats * outer_folds * inner_folds * candidate_count * len(seeds)
    per_outer_method_units = inner_folds * candidate_count * len(seeds)
    method_contract = {
        "candidate_count": candidate_count,
        "outer_repeats": outer_repeats,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "hpo_seeds": seeds,
        "max_steps": max_steps,
        "expected_units": per_method_units,
        "expected_units_per_outer_scope": per_outer_method_units,
        "minimum_fully_closed_candidates_per_outer_scope": minimum_closed_candidates,
        "max_failed_candidates_per_outer_scope": max_failed_candidates,
        "replacement_candidates": False,
        "successive_halving": False,
        "max_infrastructure_retries_per_unit": 1,
        "nonretryable_scientific_failures": sorted(NONRETRYABLE_FAILURE_CODES),
    }
    method_contracts = {method: copy.deepcopy(method_contract) for method in MAIN_METHODS}
    plan: dict[str, object] = {
        "schema": _identity("hpo_plan"),
        "status": "BOUND_RESULT_BLIND_UNIT_INVENTORY",
        "study_id": upstream["study_id"],
        "dataset_id": upstream["dataset_id"],
        "bindings": {
            "dataset_sha256": upstream["dataset_sha256"],
            "group_manifest_sha256": upstream["group_manifest_sha256"],
            "row_to_group_sha256": upstream["row_to_group_sha256"],
            "nested_plan_sha256": nested_plan_sha256,
            "nested_freeze_binding_sha256": upstream["nested_freeze_binding_sha256"],
            "candidate_space_sha256": candidate_space_sha256,
            "protocol_sha256": upstream["protocol_sha256"],
            "implementation_contract_sha256": upstream["implementation_contract_sha256"],
            "preprocessing_profile_name": upstream["preprocessing_profile_name"],
            "preprocessing_profile_sha256": upstream["preprocessing_profile_sha256"],
            "preprocessing_profile": copy.deepcopy(upstream["preprocessing_profile"]),
        },
        "method_order": list(MAIN_METHODS),
        "nested_design_binding": {
            "nested_plan_status": upstream["nested_plan_status"],
            "client_partition_policy": upstream["client_partition_policy"],
            "client_partition_policy_sha256": upstream["client_partition_policy_sha256"],
            "outer_repeats": outer_repeats,
            "outer_folds": outer_folds,
            "inner_folds": inner_folds,
        },
        "selection_contract": {
            "scope": "within_outer_repeat_outer_fold_only",
            "inner_evidence": "complete_inner_oof_and_all_registered_hpo_seeds",
            "outer_test_access": "forbidden",
            "lexicographic_objectives": [
                {"criterion": "log_loss", "direction": "minimize"},
                {"criterion": "average_precision", "direction": "maximize"},
                {"criterion": "auroc", "direction": "maximize"},
                {"criterion": "brier_score", "direction": "minimize"},
                {"criterion": "communication_bytes", "direction": "minimize"},
                {"criterion": "candidate_id", "direction": "lexicographic_min"},
            ],
            "development_context": _identity("selection_development_context"),
        },
        "equal_budget_contract": {
            "candidate_count": candidate_count,
            "outer_repeats": outer_repeats,
            "outer_folds": outer_folds,
            "inner_folds": inner_folds,
            "hpo_seeds": seeds,
            "max_steps": max_steps,
            "expected_units_per_method": per_method_units,
            "expected_units_per_outer_method": per_outer_method_units,
            "failed_candidates_consume_budget": True,
            "replacement_candidates": False,
            "selection_requires_original_terminal_ledger": True,
            "selection_scope": "within_outer_repeat_outer_fold_only",
        },
        "method_contracts": method_contracts,
        "expected_unit_count": len(units),
        "units": units,
    }
    plan["hpo_plan_sha256"] = _artifact_hash(plan, "hpo_plan_sha256")
    return plan


def build_hpo_plan(
    candidate_space: Mapping[str, object],
    nested_plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
    *,
    preprocessing_profile_spec: Mapping[str, Any],
    hpo_seeds: Sequence[int] = DEFAULT_HPO_SEEDS,
    max_steps: int,
) -> dict[str, object]:
    """Construct the exact symmetric HPO inventory from validated objects."""
    upstream = _validate_upstreams(
        candidate_space, nested_plan, group_manifest, preprocessing_profile_spec
    )
    seeds = _strict_seeds(hpo_seeds)
    try:
        step_count = require_exact_int(max_steps, "max_steps", minimum=1)
    except CandidateSpaceError as exc:
        raise HpoPlanError(str(exc)) from exc
    plan = _assemble_hpo_plan(
        candidate_space, nested_plan, upstream, hpo_seeds=seeds, max_steps=step_count
    )
    validate_hpo_plan(plan, candidate_space, nested_plan, group_manifest)
    return plan


_VALIDATED_PLAN_CONTENTS: set[str] = set()
_VALIDATED_AUTHORIZATION_CONTENTS: set[str] = set()


def _validation_content_key(*values: object) -> str | None:
    # Cache full JSON contents, never a caller-supplied hash or object identity.
    # Non-JSON container types take the full validation path so a tuple cannot
    # borrow a successful list validation through JSON normalization.
    def eligible(value):
        if type(value) is dict:
            return all(type(key) is str and eligible(child) for key, child in value.items())
        if type(value) is list:
            return all(eligible(child) for child in value)
        return value is None or type(value) in (str, int, float, bool)

    if not all(eligible(value) for value in values):
        return None
    try:
        return canonical_sha256(list(values))
    except CandidateSpaceError:
        return None


def _remember_validation(cache: set[str], key: str | None) -> None:
    if key is not None:
        if len(cache) >= 16:
            cache.clear()
        cache.add(key)


def validate_hpo_plan(
    plan: Mapping[str, object],
    candidate_space: Mapping[str, object],
    nested_plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
) -> None:
    """Rebuild and compare the complete Cartesian unit inventory."""
    content_key = _validation_content_key(plan, candidate_space, nested_plan, group_manifest)
    if content_key is not None and content_key in _VALIDATED_PLAN_CONTENTS:
        return
    if not isinstance(plan, Mapping):
        raise HpoPlanError("HPO plan must be a mapping")
    try:
        assert_no_performance_fields(plan)
    except CandidateSpaceError as exc:
        raise HpoPlanError(str(exc)) from exc
    if set(plan) != _PLAN_TOP_LEVEL_FIELDS:
        raise HpoPlanError("HPO plan top-level fields differ from the exact schema")
    if plan.get("schema") != _identity("hpo_plan"):
        raise HpoPlanError("HPO plan schema mismatch")
    if plan.get("status") != "BOUND_RESULT_BLIND_UNIT_INVENTORY":
        raise HpoPlanError("HPO plan is not a bound result-blind inventory")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping):
        raise HpoPlanError("HPO frozen bindings are missing")
    profile_spec = bindings.get("preprocessing_profile")
    if not isinstance(profile_spec, Mapping):
        raise HpoPlanError("HPO preprocessing profile binding is missing")
    upstream = _validate_upstreams(candidate_space, nested_plan, group_manifest, profile_spec)
    equal = plan.get("equal_budget_contract")
    if not isinstance(equal, Mapping):
        raise HpoPlanError("HPO equal-budget contract is missing")
    try:
        seeds = _strict_seeds(equal.get("hpo_seeds", ()))
        step_count = require_exact_int(equal.get("max_steps"), "max_steps", minimum=1)
    except CandidateSpaceError as exc:
        raise HpoPlanError(str(exc)) from exc
    expected = _assemble_hpo_plan(
        candidate_space, nested_plan, upstream, hpo_seeds=seeds, max_steps=step_count
    )
    if dict(plan) != expected:
        raise HpoPlanError("HPO plan differs from the exact upstream-derived Cartesian inventory")
    _remember_validation(_VALIDATED_PLAN_CONTENTS, content_key)


def _validate_attempts(entry: Mapping[str, object]) -> bool:
    if set(entry) != {"unit_id", "attempts"}:
        raise HpoPlanError("ledger entries accept only unit_id and attempts")
    attempts = entry.get("attempts")
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= 2:
        raise HpoPlanError("each HPO unit must have one or two recorded attempts")
    completed = False
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, Mapping):
            raise HpoPlanError("HPO attempt is not a mapping")
        outcome = attempt.get("outcome")
        expected_fields = {"attempt_index", "outcome", "attempt_receipt_sha256"}
        if outcome == "failed":
            expected_fields.add("failure_code")
        if set(attempt) != expected_fields:
            raise HpoPlanError("HPO attempt fields differ from the exact schema")
        if attempt.get("attempt_index") != index:
            raise HpoPlanError("HPO attempt indices must be contiguous from zero")
        try:
            require_sha256(attempt.get("attempt_receipt_sha256"), "attempt_receipt_sha256")
        except CandidateSpaceError as exc:
            raise HpoPlanError(str(exc)) from exc
        failure_code = attempt.get("failure_code")
        if outcome == "complete":
            if index != len(attempts) - 1:
                raise HpoPlanError("a complete attempt must be the final attempt")
            completed = True
        elif outcome == "failed":
            valid_codes = NONRETRYABLE_FAILURE_CODES | INFRASTRUCTURE_FAILURE_CODES
            if failure_code not in valid_codes:
                raise HpoPlanError("HPO failure code is not preregistered")
            if failure_code in NONRETRYABLE_FAILURE_CODES:
                if index != len(attempts) - 1:
                    raise HpoPlanError("scientific or numerical failures cannot be retried")
                if any(
                    (
                        previous.get("failure_code") not in INFRASTRUCTURE_FAILURE_CODES
                        for previous in attempts[:index]
                    )
                ):
                    raise HpoPlanError(
                        "a scientific failure may only follow an infrastructure retry"
                    )
            if index < len(attempts) - 1 and failure_code not in INFRASTRUCTURE_FAILURE_CODES:
                raise HpoPlanError("only infrastructure failures may precede a retry")
        else:
            raise HpoPlanError("HPO attempt outcome must be complete or failed")
    return completed


def _terminal_ledger_payload(
    plan: Mapping[str, object], ledger_entries: Sequence[Mapping[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if isinstance(ledger_entries, (str, bytes)):
        raise HpoPlanError("HPO ledger must be a sequence of mappings")
    try:
        assert_no_performance_fields(ledger_entries)
    except CandidateSpaceError as exc:
        raise HpoPlanError(str(exc)) from exc
    units = plan["units"]
    assert isinstance(units, list)
    expected = {str(unit["unit_id"]): unit for unit in units}
    observed: dict[str, Mapping[str, object]] = {}
    completed_units: set[str] = set()
    for entry in ledger_entries:
        if not isinstance(entry, Mapping):
            raise HpoPlanError("ledger entry is not a mapping")
        unit_id = entry.get("unit_id")
        if not isinstance(unit_id, str) or unit_id not in expected:
            raise HpoPlanError("ledger contains an unknown unit id")
        if unit_id in observed:
            raise HpoPlanError("ledger contains a duplicate unit id")
        if _validate_attempts(entry):
            completed_units.add(unit_id)
        observed[unit_id] = entry
    missing = set(expected) - set(observed)
    if missing:
        raise HpoPlanError(f"HPO ledger is open: {len(missing)} planned units are missing")
    candidate_units: dict[tuple[int, int, str, str], list[str]] = {}
    for unit_id, unit in expected.items():
        key = (
            int(unit["outer_repeat"]),
            int(unit["outer_fold"]),
            str(unit["method"]),
            str(unit["candidate_id"]),
        )
        candidate_units.setdefault(key, []).append(unit_id)
    contracts = plan["method_contracts"]
    assert isinstance(contracts, Mapping)
    design = plan["nested_design_binding"]
    assert isinstance(design, Mapping)
    authorizations: list[dict[str, object]] = []
    for outer_repeat in range(int(design["outer_repeats"])):
        for outer_fold in range(int(design["outer_folds"])):
            methods: dict[str, object] = {}
            for method in MAIN_METHODS:
                eligible: list[str] = []
                failed: list[str] = []
                candidate_ids = sorted(
                    (
                        key[3]
                        for key in candidate_units
                        if key[:3] == (outer_repeat, outer_fold, method)
                    )
                )
                for candidate_id in candidate_ids:
                    unit_ids = candidate_units[outer_repeat, outer_fold, method, candidate_id]
                    target = (
                        eligible
                        if all((unit_id in completed_units for unit_id in unit_ids))
                        else failed
                    )
                    target.append(candidate_id)
                contract = contracts[method]
                assert isinstance(contract, Mapping)
                minimum = int(contract["minimum_fully_closed_candidates_per_outer_scope"])
                maximum_failed = int(contract["max_failed_candidates_per_outer_scope"])
                if len(eligible) < minimum or len(failed) > maximum_failed:
                    raise HpoPlanError(
                        f"{method} exceeds the failure budget in outer ({outer_repeat}, {outer_fold})"
                    )
                methods[method] = {
                    "eligible_candidate_ids": eligible,
                    "failed_candidate_ids": failed,
                }
            scope: dict[str, object] = {
                "outer_repeat": outer_repeat,
                "outer_fold": outer_fold,
                "methods": methods,
            }
            scope["outer_authorization_sha256"] = _artifact_hash(
                scope, "outer_authorization_sha256"
            )
            authorizations.append(scope)
    ledger_payload = sorted(
        (copy.deepcopy(dict(entry)) for entry in ledger_entries),
        key=lambda entry: str(entry["unit_id"]),
    )
    return (ledger_payload, authorizations)


def close_hpo_ledger(
    plan: Mapping[str, object],
    ledger_entries: Sequence[Mapping[str, object]],
    *,
    attempt_receipts: Sequence[Mapping[str, object]],
    candidate_space: Mapping[str, object],
    nested_plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
) -> dict[str, object]:
    """Close planned units only after exact attempt receipts are verified."""
    from .hpo_attempt_receipt import HpoAttemptReceiptError, validate_hpo_attempt_receipt_catalog

    try:
        receipt_validation = validate_hpo_attempt_receipt_catalog(
            plan,
            ledger_entries,
            attempt_receipts,
            candidate_space=candidate_space,
            nested_plan=nested_plan,
            group_manifest=group_manifest,
        )
    except HpoAttemptReceiptError as exc:
        raise HpoPlanError(str(exc)) from exc
    ledger_payload, authorizations = _terminal_ledger_payload(plan, ledger_entries)
    units = plan["units"]
    assert isinstance(units, list)
    closure: dict[str, object] = {
        "schema": _identity("hpo_ledger_closure"),
        "status": "CLOSED_OUTER_LOCAL_SELECTION_AUTHORIZED",
        "hpo_plan_sha256": plan["hpo_plan_sha256"],
        "ledger_sha256": canonical_sha256(ledger_payload),
        "attempt_receipt_catalog_sha256": receipt_validation["attempt_receipt_catalog_sha256"],
        "receipt_catalog_validation_sha256": receipt_validation["catalog_validation_sha256"],
        "verified_attempt_receipt_count": receipt_validation["receipt_count"],
        "planned_unit_count": len(units),
        "terminal_unit_count": len(ledger_payload),
        "selection_scope": "within_outer_repeat_outer_fold_only",
        "outer_authorizations": authorizations,
        "replacement_candidates_added": 0,
    }
    closure["closure_sha256"] = _artifact_hash(closure, "closure_sha256")
    return closure


def validate_selection_authorization(
    plan: Mapping[str, object],
    closure: Mapping[str, object],
    ledger_entries: Sequence[Mapping[str, object]],
    *,
    attempt_receipts: Sequence[Mapping[str, object]],
    candidate_space: Mapping[str, object],
    nested_plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
) -> None:
    """Recompute authorization from the original terminal ledger."""
    content_key = _validation_content_key(
        plan, closure, ledger_entries, attempt_receipts, candidate_space, nested_plan, group_manifest
    )
    if content_key is not None and content_key in _VALIDATED_AUTHORIZATION_CONTENTS:
        return
    if not isinstance(closure, Mapping):
        raise HpoPlanError("selection authorization must be a mapping")
    expected = close_hpo_ledger(
        plan,
        ledger_entries,
        attempt_receipts=attempt_receipts,
        candidate_space=candidate_space,
        nested_plan=nested_plan,
        group_manifest=group_manifest,
    )
    if dict(closure) != expected:
        raise HpoPlanError("selection authorization differs from the original terminal ledger")
    _remember_validation(_VALIDATED_AUTHORIZATION_CONTENTS, content_key)


def _outer_authorization(
    closure: Mapping[str, object], outer_repeat: int, outer_fold: int
) -> Mapping[str, object]:
    rows = closure.get("outer_authorizations")
    if not isinstance(rows, list):
        raise HpoPlanError("selection authorization has no outer scopes")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("outer_repeat") == outer_repeat
        and (row.get("outer_fold") == outer_fold)
    ]
    if len(matches) != 1:
        raise HpoPlanError("selection authorization outer scope is not unique")
    return matches[0]


def inherit_matched_ablation(
    plan: Mapping[str, object],
    closure: Mapping[str, object],
    ledger_entries: Sequence[Mapping[str, object]],
    *,
    attempt_receipts: Sequence[Mapping[str, object]],
    candidate_space: Mapping[str, object],
    nested_plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
    ablation_id: str,
    outer_repeat: int,
    outer_fold: int,
    parent_candidate_id: str,
    parent_candidate_sha256: str,
    parent_selection_receipt_sha256: str,
) -> dict[str, object]:
    """Bind a matched ablation to one selected parent in one outer scope."""
    validate_selection_authorization(
        plan,
        closure,
        ledger_entries,
        attempt_receipts=attempt_receipts,
        candidate_space=candidate_space,
        nested_plan=nested_plan,
        group_manifest=group_manifest,
    )
    spec = MATCHED_ABLATIONS.get(ablation_id)
    if spec is None:
        raise HpoPlanError("matched ablation is not preregistered")
    if isinstance(outer_repeat, bool) or not isinstance(outer_repeat, int):
        raise HpoPlanError("outer_repeat must be an exact integer")
    if isinstance(outer_fold, bool) or not isinstance(outer_fold, int):
        raise HpoPlanError("outer_fold must be an exact integer")
    try:
        parent_hash = require_sha256(parent_candidate_sha256, "parent_candidate_sha256")
        selection_hash = require_sha256(
            parent_selection_receipt_sha256, "parent_selection_receipt_sha256"
        )
    except CandidateSpaceError as exc:
        raise HpoPlanError(str(exc)) from exc
    parent_method = str(spec["parent_method"])
    outer = _outer_authorization(closure, outer_repeat, outer_fold)
    methods = outer.get("methods")
    if not isinstance(methods, Mapping):
        raise HpoPlanError("outer authorization lacks method inventories")
    parent_inventory = methods.get(parent_method)
    if not isinstance(parent_inventory, Mapping) or parent_candidate_id not in parent_inventory.get(
        "eligible_candidate_ids", []
    ):
        raise HpoPlanError("matched ablation parent is not eligible in this outer scope")
    units = plan.get("units")
    if not isinstance(units, list):
        raise HpoPlanError("HPO plan units are missing")
    planned_parent_hashes = {
        str(unit["candidate_sha256"])
        for unit in units
        if unit["method"] == parent_method
        and unit["outer_repeat"] == outer_repeat
        and (unit["outer_fold"] == outer_fold)
        and (unit["candidate_id"] == parent_candidate_id)
    }
    if planned_parent_hashes != {parent_hash}:
        raise HpoPlanError("matched ablation parent hash differs from this outer plan")
    candidate_methods = candidate_space.get("methods")
    if not isinstance(candidate_methods, Mapping):
        raise HpoPlanError("candidate space method rosters are missing")
    parent_roster = candidate_methods.get(parent_method)
    if not isinstance(parent_roster, Mapping):
        raise HpoPlanError("matched ablation parent roster is missing")
    try:
        parent_space_hash = require_sha256(
            candidate_space.get("manifest_sha256"), "parent_candidate_space_sha256"
        )
        parent_roster_hash = require_sha256(
            parent_roster.get("roster_sha256"), "parent_roster_sha256"
        )
    except CandidateSpaceError as exc:
        raise HpoPlanError(str(exc)) from exc
    record: dict[str, object] = {
        "schema": _identity("matched_ablation_inheritance"),
        "method_identity": ablation_id,
        "ablation_id": ablation_id,
        "parent_method": parent_method,
        "outer_repeat": outer_repeat,
        "outer_fold": outer_fold,
        "parent_candidate_id": parent_candidate_id,
        "parent_candidate_sha256": parent_hash,
        "parent_candidate_space_sha256": parent_space_hash,
        "parent_roster_sha256": parent_roster_hash,
        "parent_selection_receipt_sha256": selection_hash,
        "selection_authorization_sha256": closure["closure_sha256"],
        "outer_authorization_sha256": outer["outer_authorization_sha256"],
        "configuration_source": "inherit_outer_selected_parent_candidate_exactly",
        "inherited_parent_fields": copy.deepcopy(spec["inherited_parent_fields"]),
        "mechanism_switch": copy.deepcopy(spec["mechanism_switch"]),
        "independent_candidate_roster": False,
        "additional_hpo_units": 0,
    }
    record["inheritance_sha256"] = _artifact_hash(record, "inheritance_sha256")
    return record
