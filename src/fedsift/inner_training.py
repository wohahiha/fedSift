"""Execute explicitly selected inner-development units on registered data.

This runner has no selection or outer-test interface. It creates the existing
sealed HPO output and attempt receipts, then computes threshold-free metrics
for development inspection. A complete formal study still requires a closed
ledger, candidate selection, refit, and one-time outer authorization.
"""

from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
from dataclasses import asdict
import copy
import hashlib
from pathlib import Path
from typing import Mapping, Sequence
from .candidate_space import canonical_sha256, require_sha256
from .evaluation import compute_hpo_probability_metrics
from .hpo_attempt_receipt import build_hpo_attempt_receipt
from .hpo_capability import (
    HpoUnitIndex,
    build_hpo_unit_capability,
    build_prevalidated_hpo_unit_capability,
    capability_candidate_parameters,
)
from .hpo_output import build_hpo_output_manifest, NONPRIVATE_PRIVACY_NOT_APPLICABLE
from .method_dispatch import build_method_execution_spec
from .preprocessing import preprocess_hpo_capability
from .runtime_environment import RuntimeEnvironmentGuard
from .study_factory import StudyConstruction
from .train_unit import execute_training_unit, seal_preprocessed_unit_data
from .training_budget_factory import (
    SharedTrainingBudgetPolicy,
    build_training_budget,
    validate_training_budget_factory_result,
)


class InnerTrainingError(RuntimeError):
    pass


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prediction(result, role: str) -> Mapping[str, object]:
    rows = [row for row in result.artifact["raw_native_predictions"] if row.get("role") == role]
    if len(rows) != 1:
        raise InnerTrainingError(f"exactly one {role} prediction payload required")
    return rows[0]


def _privacy_bindings(result, method_id: str) -> tuple[str, str]:
    if method_id == "fedavg_nonprivate":
        return (NONPRIVATE_PRIVACY_NOT_APPLICABLE, NONPRIVATE_PRIVACY_NOT_APPLICABLE)
    if result.parallel_privacy_report is None:
        raise InnerTrainingError("private unit lacks composed privacy report")
    report_hash = canonical_sha256(asdict(result.parallel_privacy_report))
    schedule_hash = canonical_sha256(
        [
            {
                "client_id": report.client_id,
                "registered_schedule_sha256": report.composed_report.registered_schedule_sha256,
                "execution_history_sha256": report.composed_report.execution_history_sha256,
            }
            for report in result.sequential_client_reports
        ]
    )
    return (report_hash, schedule_hash)


def execute_selected_hpo_unit(
    construction: StudyConstruction,
    workspace_root: Path | str,
    unit_id: str,
    policy: SharedTrainingBudgetPolicy,
    *,
    code_sha256: str,
    environment_sha256: str,
    dependency_lock_sha256: str,
    environment_guard: RuntimeEnvironmentGuard,
    unit_index: HpoUnitIndex | None = None,
) -> dict[str, object]:
    """Run one inner unit and return receipts plus development metrics."""
    code_hash = require_sha256(code_sha256, "code_sha256")
    environment_hash = require_sha256(environment_sha256, "environment_sha256")
    dependency_hash = require_sha256(dependency_lock_sha256, "dependency_lock_sha256")
    if (
        not isinstance(environment_guard, RuntimeEnvironmentGuard)
        or environment_guard.expected_sha256 != environment_hash
    ):
        raise InnerTrainingError("runtime guard differs from environment binding")
    environment_guard.check("before_development_hpo_unit")
    if unit_index is None:
        capability = build_hpo_unit_capability(
            construction.complete_hpo_plan,
            construction.frozen_candidate_space,
            construction.frozen_nested_plan,
            construction.group_manifest,
            unit_id=unit_id,
        )
    else:
        capability = build_prevalidated_hpo_unit_capability(
            construction.complete_hpo_plan,
            construction.frozen_candidate_space,
            construction.frozen_nested_plan,
            construction.group_manifest,
            unit_id=unit_id,
            unit_index=unit_index,
            expected_hpo_plan_sha256=str(construction.complete_hpo_plan["hpo_plan_sha256"]),
        )
    cap_hash = str(capability["capability_sha256"])
    identity, candidate = (capability["unit_identity"], capability["candidate"])
    if identity["max_steps"] != policy.server_rounds:
        raise InnerTrainingError("policy rounds differ from sealed unit")
    preprocessed = preprocess_hpo_capability(
        capability,
        construction.dataset_rows,
        construction.group_manifest,
        expected_capability_sha256=cap_hash,
    )
    labels = dict(zip(construction.dataset_rows.row_ids, construction.dataset_rows.labels))
    sealed = seal_preprocessed_unit_data(
        capability, preprocessed, labels, expected_capability_sha256=cap_hash
    )
    parameters = capability_candidate_parameters(capability, expected_capability_sha256=cap_hash)
    spec = build_method_execution_spec(
        method_id=str(identity["method"]), candidate_parameters=parameters, mechanism_switch=None
    )
    private_rows = {f"client_{i}": sealed.role(f"client_{i}").row_ids for i in range(5)}
    budget_result = build_training_budget(
        spec,
        parameters,
        private_rows,
        policy,
        expected_dispatch_sha256=spec.dispatch_sha256,
        expected_candidate_parameters_sha256=str(candidate["parameters_sha256"]),
        expected_policy_sha256=policy.policy_sha256,
    )
    validate_training_budget_factory_result(
        budget_result,
        spec,
        parameters,
        private_rows,
        policy,
        expected_dispatch_sha256=spec.dispatch_sha256,
        expected_candidate_parameters_sha256=str(candidate["parameters_sha256"]),
        expected_policy_sha256=policy.policy_sha256,
    )
    result = execute_training_unit(
        capability,
        sealed,
        budget_result.budget,
        expected_capability_sha256=cap_hash,
        expected_budget_sha256=budget_result.budget.budget_sha256,
    )
    prediction = _prediction(result, "inner_validation")
    report_hash, schedule_hash = _privacy_bindings(result, str(identity["method"]))
    output = build_hpo_output_manifest(
        capability,
        expected_capability_sha256=cap_hash,
        attempt_index=0,
        authoritative_rows=construction.dataset_rows,
        probabilities=prediction["probabilities"],
        preprocessing_manifest_sha256=sealed.preprocessing_artifact_sha256,
        model_manifest_sha256=result.artifact["model_manifest_sha256"],
        training_trace_sha256=result.artifact["artifact_sha256"],
        code_sha256=code_hash,
        environment_sha256=environment_hash,
        dependency_lock_sha256=dependency_hash,
        privacy_accounting_report_sha256=report_hash,
        privacy_schedule_sha256=schedule_hash,
        round_resource_receipts=result.round_resource_receipts,
        expected_rounds=policy.server_rounds,
    )
    attempt = build_hpo_attempt_receipt(
        capability,
        expected_capability_sha256=cap_hash,
        attempt_index=0,
        outcome="complete",
        code_sha256=code_hash,
        environment_sha256=environment_hash,
        dependency_lock_sha256=dependency_hash,
        privacy_accounting_report_sha256=report_hash,
        privacy_schedule_sha256=schedule_hash,
        exclusive_output_artifact_manifest_sha256=output["manifest_sha256"],
    )
    metrics = compute_hpo_probability_metrics(
        prediction["row_ids"], prediction["labels"], prediction["probabilities"]
    )
    artifact = {
        "schema": _identity("selected_development_hpo_unit"),
        "status": "complete_development_only_inner_validation",
        "unit_id": unit_id,
        "method_id": identity["method"],
        "candidate_id": candidate["candidate_id"],
        "hpo_output_manifest": output,
        "attempt_receipt": attempt,
        "development_metrics": metrics,
        "outer_test_accessed": False,
        "candidate_selection_performed": False,
    }
    environment_guard.check("after_development_hpo_unit")
    return artifact


def selected_unit_ids(
    construction: StudyConstruction,
    *,
    methods: Sequence[str],
    candidate_id: str,
    outer_repeat: int,
    outer_fold: int,
    inner_fold: int,
    hpo_seed: int,
) -> tuple[str, ...]:
    wanted = set(methods)
    matches = [
        row
        for row in construction.complete_hpo_plan["units"]
        if row["method"] in wanted
        and row["candidate_id"] == candidate_id
        and (row["outer_repeat"] == outer_repeat)
        and (row["outer_fold"] == outer_fold)
        and (row["inner_fold"] == inner_fold)
        and (row["hpo_seed"] == hpo_seed)
    ]
    if len(matches) != len(wanted) or {row["method"] for row in matches} != wanted:
        raise InnerTrainingError("requested unit selection is incomplete or ambiguous")
    return tuple((row["unit_id"] for row in sorted(matches, key=lambda row: row["method"])))
