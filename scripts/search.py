"""Result-blind, resumable orchestration for the FedSift formal HPO.

The script intentionally keeps outer-test data closed.  Workers persist only
sealed HPO output manifests and attempt receipts; development metrics returned
by the reusable unit runner are discarded before any artifact is written.
"""

from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
from fedsift.experiment_io import (
    registered_study,
    results_root,
    resolve_record_path,
    verify_release_sources,
)
import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
from fedsift.candidate_space import MAIN_METHODS, canonical_sha256
from fedsift.inner_training import execute_selected_hpo_unit, sha256_file
from fedsift.hpo_attempt_receipt import (
    NONPRIVATE_PRIVACY_NOT_APPLICABLE,
    build_hpo_attempt_receipt,
    validate_hpo_attempt_receipt,
)
from fedsift.hpo_capability import build_hpo_unit_index, build_prevalidated_hpo_unit_capability
from fedsift.hpo_output import validate_hpo_output_catalog, validate_hpo_output_manifest
from fedsift.hpo_plan import close_hpo_ledger
from fedsift.hpo_select import (
    build_hpo_selection_decision_from_stage_commitment,
    build_hpo_selection_stage_commitment,
)
from fedsift.runtime_environment import RuntimeEnvironmentError, RuntimeEnvironmentGuard
from fedsift.study_factory import build_study_construction, propose_study
from fedsift.training_budget_factory import SharedTrainingBudgetPolicy

TOP = results_root()
RUN_ROOT = TOP / "selection"
ENVIRONMENT_MANIFEST = ROOT / "environment/runtime_contract.json"
CLIENT_POLICY_SHA256 = "356fac82397469a91b4da89e27d7557590236d4fdf0df20259881c6bbff423c7"
CANDIDATE_COUNT = 24
HPO_SEEDS = (101, 211, 307)
MAX_STEPS = 50
SHARD_COUNT = 4
DATASETS = {
    "pima": {
        "study_id": _identity("pima_hpo_24c_3seed_50r"),
        "preprocessing_profile_name": "pima_primary_zero_as_missing_v1",
    },
    "retinopathy": {
        "study_id": _identity("debrecen_hpo_24c_3seed_50r"),
        "preprocessing_profile_name": "debrecen_numeric_median_standardize_v1",
    },
}


class SearchError(RuntimeError):
    pass


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _artifact(payload: Mapping[str, Any], hash_field: str) -> dict[str, Any]:
    value = copy.deepcopy(dict(payload))
    value[hash_field] = canonical_sha256(value)
    return value


def _verify_artifact(value: Mapping[str, Any], hash_field: str) -> None:
    if not isinstance(value, Mapping) or hash_field not in value:
        raise SearchError(f"artifact lacks {hash_field}")
    payload = copy.deepcopy(dict(value))
    stored = payload.pop(hash_field)
    if stored != canonical_sha256(payload):
        raise SearchError(f"artifact canonical hash differs: {hash_field}")


def _registered_implementation():
    verify_release_sources()
    return {"implementation_sha256": registered_study()["implementation_sha256"]}


def _environment() -> tuple[dict[str, Any], str]:
    manifest = _read_json(ENVIRONMENT_MANIFEST)
    digest = canonical_sha256(manifest)
    RuntimeEnvironmentGuard(manifest, expected_sha256=digest).check("formal_orchestrator")
    return (manifest, digest)


def _registered_runner_identity():
    return registered_study()["runner_sha256"]["search"]


def _protocol_payload(source: Mapping[str, Any], environment_sha256: str) -> dict[str, Any]:
    file_rows = copy.deepcopy(registered_study()["protocol_sources"])
    return {
        "schema": _identity("hpo_core_protocol"),
        "status": "RESULT_BLIND_HPO_AUTHORIZED_OUTER_TEST_CLOSED",
        "protocol_sources": file_rows,
        "implementation_sha256": source["implementation_sha256"],
        "orchestration_runner_sha256": _registered_runner_identity(),
        "environment_sha256": environment_sha256,
        "dependency_lock_sha256": registered_study()["dependency_lock_sha256"],
        "client_policy_sha256": CLIENT_POLICY_SHA256,
        "datasets": copy.deepcopy(DATASETS),
        "methods": list(MAIN_METHODS),
        "candidate_count": CANDIDATE_COUNT,
        "hpo_seeds": list(HPO_SEEDS),
        "inner_folds": [0, 1, 2],
        "outer_repeats": 3,
        "outer_folds_per_repeat": 5,
        "max_steps": MAX_STEPS,
        "worker_shards": SHARD_COUNT,
        "sharding_rule": "frozen_hpo_plan_unit_index_modulo_worker_shards",
        "retry_policy": "no_automatic_retry_stop_shard_after_first_new_failure",
        "result_visibility": "sealed_outputs_only_until_global_ledger_closure",
        "outer_test_accessed": False,
        "paper_claim_authorized": False,
    }


def _study_kwargs(dataset: str, protocol_sha256: str, implementation_sha256: str) -> dict[str, Any]:
    spec = DATASETS[dataset]
    return {
        "study_id": spec["study_id"],
        "protocol_sha256": protocol_sha256,
        "implementation_sha256": implementation_sha256,
        "client_policy_sha256": CLIENT_POLICY_SHA256,
        "candidate_count": CANDIDATE_COUNT,
        "hpo_seeds": HPO_SEEDS,
        "max_steps": MAX_STEPS,
        "preprocessing_profile_name": spec["preprocessing_profile_name"],
    }


def _policy(dataset: str) -> SharedTrainingBudgetPolicy:
    return SharedTrainingBudgetPolicy(
        server_rounds=MAX_STEPS,
        local_epochs=1,
        poisson_sample_rate=1.0,
        clip_norm=1.0,
        target_epsilon=8.0,
        target_delta=1e-05,
        model_family="logistic_screening",
        participating_client_ids=tuple((f"client_{index}" for index in range(5))),
        paired_initialization_seed=20260901,
        nonprivate_order_seed=f"{_identity('artifact_identity_10d49495')}{dataset}/formal-hpo-v1/nonprivate-order",
        partition_mode="fixed_label_driven_auxiliary_condition",
        fixed_auxiliary_partition_condition_sha256=CLIENT_POLICY_SHA256,
    )


def _authorization() -> dict[str, Any]:
    path = RUN_ROOT / "launch_authorization.json"
    authorization = _read_json(path)
    _verify_artifact(authorization, "launch_authorization_sha256")
    source = _registered_implementation()
    _, environment_sha256 = _environment()
    expected = {
        "implementation_sha256": source["implementation_sha256"],
        "orchestration_runner_sha256": _registered_runner_identity(),
        "environment_sha256": environment_sha256,
        "dependency_lock_sha256": registered_study()["dependency_lock_sha256"],
        "worker_shards": SHARD_COUNT,
    }
    for field, value in expected.items():
        if authorization.get(field) != value:
            raise SearchError(f"launch authorization differs: {field}")
    if authorization.get("status") != "FORMAL_HPO_LAUNCH_AUTHORIZED_OUTER_TEST_CLOSED":
        raise SearchError("formal HPO launch is not authorized")
    return authorization


def prepare(_: argparse.Namespace) -> None:
    if (RUN_ROOT / "launch_authorization.json").exists():
        existing = _authorization()
        print(
            json.dumps(
                {
                    "status": "reused_exact_launch_authorization",
                    "launch_authorization_sha256": existing["launch_authorization_sha256"],
                }
            )
        )
        return
    source = _registered_implementation()
    _, environment_sha256 = _environment()
    protocol = _artifact(_protocol_payload(source, environment_sha256), "protocol_sha256")
    _atomic_json(RUN_ROOT / "selection_protocol.json", protocol)
    proposals: dict[str, Any] = {}
    for dataset in DATASETS:
        proposal = propose_study(
            ROOT,
            dataset,
            **_study_kwargs(dataset, protocol["protocol_sha256"], source["implementation_sha256"]),
        )
        proposal_artifact = _artifact(
            {
                "schema": _identity("hpo_proposal_archive"),
                "status": proposal.status,
                "dataset_id": dataset,
                "bundle": proposal.bundle,
                "proposed_bundle_sha256": proposal.proposed_bundle_sha256,
                "outer_test_accessed": False,
            },
            "proposal_archive_sha256",
        )
        _atomic_json(RUN_ROOT / dataset / "proposal.json", proposal_artifact)
        proposals[dataset] = {
            "proposed_bundle_sha256": proposal.proposed_bundle_sha256,
            "proposal_archive_sha256": proposal_artifact["proposal_archive_sha256"],
            "planned_unit_count": proposal.bundle["hpo_plan_binding"]["expected_unit_count"],
        }
    authorization = _artifact(
        {
            "schema": _identity("hpo_launch_authorization"),
            "status": "FORMAL_HPO_LAUNCH_AUTHORIZED_OUTER_TEST_CLOSED",
            "protocol_sha256": protocol["protocol_sha256"],
            "implementation_sha256": source["implementation_sha256"],
            "orchestration_runner_sha256": _registered_runner_identity(),
            "environment_sha256": environment_sha256,
            "dependency_lock_sha256": registered_study()["dependency_lock_sha256"],
            "client_policy_sha256": CLIENT_POLICY_SHA256,
            "worker_shards": SHARD_COUNT,
            "retry_policy": "no_automatic_retry_stop_shard_after_first_new_failure",
            "proposals": proposals,
            "outer_test_accessed": False,
        },
        "launch_authorization_sha256",
    )
    _atomic_json(RUN_ROOT / "launch_authorization.json", authorization)
    print(
        json.dumps(
            {
                "status": "prepared",
                "launch_authorization_sha256": authorization["launch_authorization_sha256"],
                "proposals": proposals,
            }
        )
    )


def _construction(dataset: str, authorization: Mapping[str, Any]):
    protocol_sha256 = str(authorization["protocol_sha256"])
    implementation_sha256 = str(authorization["implementation_sha256"])
    proposal_archive = _read_json(RUN_ROOT / dataset / "proposal.json")
    _verify_artifact(proposal_archive, "proposal_archive_sha256")
    binding = authorization["proposals"][dataset]
    if proposal_archive["proposal_archive_sha256"] != binding["proposal_archive_sha256"]:
        raise SearchError("proposal archive differs from launch authorization")
    return build_study_construction(
        ROOT,
        dataset,
        **_study_kwargs(dataset, protocol_sha256, implementation_sha256),
        expected_bundle_sha256=binding["proposed_bundle_sha256"],
    )


def _planned_unit(raw: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema": _identity("selected_development_hpo_unit"),
        "status": "complete_development_only_inner_validation",
        "outer_test_accessed": False,
        "candidate_selection_performed": False,
    }
    if any((raw.get(key) != value for (key, value) in required.items())):
        raise SearchError("unit runner returned an unexpected top-level contract")
    return _artifact(
        {
            "schema": _identity("hpo_unit_artifact"),
            "status": "SEALED_COMPLETE_INNER_VALIDATION_OUTPUT",
            "unit_id": raw["unit_id"],
            "method_id": raw["method_id"],
            "candidate_id": raw["candidate_id"],
            "hpo_output_manifest": raw["hpo_output_manifest"],
            "attempt_receipt": raw["attempt_receipt"],
            "outer_test_accessed": False,
            "candidate_selection_performed": False,
            "development_metrics_persisted": False,
        },
        "formal_unit_artifact_sha256",
    )


def _capability(construction, unit_index, unit_id: str) -> dict[str, Any]:
    return build_prevalidated_hpo_unit_capability(
        construction.complete_hpo_plan,
        construction.frozen_candidate_space,
        construction.frozen_nested_plan,
        construction.group_manifest,
        unit_id=unit_id,
        unit_index=unit_index,
        expected_hpo_plan_sha256=construction.complete_hpo_plan["hpo_plan_sha256"],
    )


def _validate_complete_artifact(
    artifact: Mapping[str, Any], construction, unit_index, unit: Mapping[str, Any]
) -> None:
    _verify_artifact(artifact, "formal_unit_artifact_sha256")
    if (
        artifact.get("schema") != _identity("hpo_unit_artifact")
        or artifact.get("status") != "SEALED_COMPLETE_INNER_VALIDATION_OUTPUT"
        or artifact.get("unit_id") != unit["unit_id"]
        or (artifact.get("method_id") != unit["method"])
        or (artifact.get("candidate_id") != unit["candidate_id"])
        or (artifact.get("outer_test_accessed") is not False)
        or (artifact.get("candidate_selection_performed") is not False)
        or (artifact.get("development_metrics_persisted") is not False)
        or ("development_metrics" in artifact)
    ):
        raise SearchError("existing formal unit identity differs")
    receipt = artifact.get("attempt_receipt")
    manifest = artifact.get("hpo_output_manifest")
    if not isinstance(receipt, Mapping) or not isinstance(manifest, Mapping):
        raise SearchError("existing formal unit is incomplete")
    validate_hpo_attempt_receipt(receipt)
    capability = _capability(construction, unit_index, str(unit["unit_id"]))
    validate_hpo_output_manifest(
        manifest,
        capability,
        expected_capability_sha256=capability["capability_sha256"],
        authoritative_rows=construction.dataset_rows,
        authoritative_dataset_sha256=construction.dataset_rows.source_sha256,
        attempt_receipt=receipt,
    )


def _failure_code(error: BaseException) -> str:
    if isinstance(error, RuntimeEnvironmentError):
        return "infrastructure_failure"
    if isinstance(error, (OSError, IOError)):
        return "transient_io_failure"
    if isinstance(error, FloatingPointError):
        return "numerical_instability"
    return "scientific_contract_failure"


def _failure_artifact(
    error: BaseException, capability: Mapping[str, Any], authorization: Mapping[str, Any]
) -> dict[str, Any]:
    incident = _artifact(
        {
            "schema": _identity("hpo_failure_incident"),
            "failure_code": _failure_code(error),
            "exception_class_sha256": hashlib.sha256(
                type(error).__qualname__.encode("utf-8")
            ).hexdigest(),
            "unit_id": capability["unit_identity"]["unit_id"],
            "attempt_index": 0,
            "outer_test_accessed": False,
        },
        "failure_incident_sha256",
    )
    method = str(capability["unit_identity"]["method"])
    privacy = (
        NONPRIVATE_PRIVACY_NOT_APPLICABLE
        if method == "fedavg_nonprivate"
        else incident["failure_incident_sha256"]
    )
    receipt = build_hpo_attempt_receipt(
        capability,
        expected_capability_sha256=capability["capability_sha256"],
        attempt_index=0,
        outcome="failed",
        failure_code=incident["failure_code"],
        code_sha256=authorization["implementation_sha256"],
        environment_sha256=authorization["environment_sha256"],
        dependency_lock_sha256=authorization["dependency_lock_sha256"],
        privacy_accounting_report_sha256=privacy,
        privacy_schedule_sha256=privacy,
        failure_incident_sha256=incident["failure_incident_sha256"],
    )
    return _artifact(
        {
            "schema": _identity("hpo_failed_unit_artifact"),
            "status": "SEALED_FAILED_ATTEMPT_SHARD_STOPPED",
            "unit_id": capability["unit_identity"]["unit_id"],
            "failure_incident": incident,
            "attempt_receipt": receipt,
            "outer_test_accessed": False,
        },
        "failed_unit_artifact_sha256",
    )


def worker(arguments: argparse.Namespace) -> None:
    authorization = _authorization()
    if (
        arguments.shards != authorization["worker_shards"]
        or not 0 <= arguments.shard < arguments.shards
    ):
        raise SearchError("worker shard identity differs from authorization")
    construction = _construction(arguments.dataset, authorization)
    unit_index = build_hpo_unit_index(construction.complete_hpo_plan)
    all_units = list(construction.complete_hpo_plan["units"])
    assigned = [
        unit
        for (index, unit) in enumerate(all_units)
        if index % arguments.shards == arguments.shard
    ]
    unit_dir = RUN_ROOT / arguments.dataset / "units"
    failure_dir = RUN_ROOT / arguments.dataset / "failures"
    progress_path = RUN_ROOT / arguments.dataset / f"worker_{arguments.shard}_progress.json"
    executed = reused = 0
    for position, unit in enumerate(assigned, start=1):
        unit_id = str(unit["unit_id"])
        complete_path = unit_dir / f"{unit_id}.json"
        failed_path = failure_dir / f"{unit_id}.json"
        if failed_path.exists():
            raise SearchError(f"recorded failed unit requires review before resume: {unit_id}")
        if complete_path.exists():
            _validate_complete_artifact(_read_json(complete_path), construction, unit_index, unit)
            reused += 1
            state = "reused_exact_complete_unit"
        else:
            capability = _capability(construction, unit_index, unit_id)
            try:
                guard = RuntimeEnvironmentGuard(
                    _read_json(ENVIRONMENT_MANIFEST),
                    expected_sha256=authorization["environment_sha256"],
                )
                raw = execute_selected_hpo_unit(
                    construction,
                    ROOT,
                    unit_id,
                    _policy(arguments.dataset),
                    code_sha256=authorization["implementation_sha256"],
                    environment_sha256=authorization["environment_sha256"],
                    dependency_lock_sha256=authorization["dependency_lock_sha256"],
                    environment_guard=guard,
                    unit_index=unit_index,
                )
                formal = _planned_unit(raw)
                _validate_complete_artifact(formal, construction, unit_index, unit)
                _atomic_json(complete_path, formal)
                executed += 1
                state = "executed_and_sealed"
            except BaseException as error:
                failed = _failure_artifact(error, capability, authorization)
                _atomic_json(failed_path, failed)
                _atomic_json(
                    progress_path,
                    {
                        "schema": _identity("hpo_worker_progress"),
                        "dataset_id": arguments.dataset,
                        "shard": arguments.shard,
                        "shards": arguments.shards,
                        "assigned_unit_count": len(assigned),
                        "last_position": position,
                        "completed_unit_count": executed + reused,
                        "newly_executed_count": executed,
                        "reused_count": reused,
                        "status": "STOPPED_AFTER_NEW_FAILURE",
                        "failed_unit_id": unit_id,
                        "failure_code": failed["failure_incident"]["failure_code"],
                        "outer_test_accessed": False,
                    },
                )
                raise
        progress = {
            "schema": _identity("hpo_worker_progress"),
            "dataset_id": arguments.dataset,
            "shard": arguments.shard,
            "shards": arguments.shards,
            "assigned_unit_count": len(assigned),
            "last_position": position,
            "completed_unit_count": executed + reused,
            "newly_executed_count": executed,
            "reused_count": reused,
            "last_unit_id": unit_id,
            "last_unit_state": state,
            "status": "COMPLETE" if position == len(assigned) else "RUNNING",
            "outer_test_accessed": False,
        }
        _atomic_json(progress_path, progress)
        print(json.dumps(progress, separators=(",", ":")), flush=True)


def _complete_artifacts(dataset: str, construction, unit_index) -> Iterable[dict[str, Any]]:
    unit_dir = RUN_ROOT / dataset / "units"
    for unit in construction.complete_hpo_plan["units"]:
        path = unit_dir / f"{unit['unit_id']}.json"
        if not path.is_file():
            raise SearchError(f"formal HPO remains open; missing unit: {unit['unit_id']}")
        artifact = _read_json(path)
        _validate_complete_artifact(artifact, construction, unit_index, unit)
        yield artifact


def finalize(arguments: argparse.Namespace) -> None:
    authorization = _authorization()
    failure_dir = RUN_ROOT / arguments.dataset / "failures"
    failures = list(failure_dir.glob("*.json")) if failure_dir.exists() else []
    if failures:
        raise SearchError(f"formal HPO has {len(failures)} recorded failures; ledger remains open")
    construction = _construction(arguments.dataset, authorization)
    plan = construction.complete_hpo_plan
    unit_index = build_hpo_unit_index(plan)
    receipts: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    artifact_paths: list[Path] = []
    for unit, artifact in zip(
        plan["units"], _complete_artifacts(arguments.dataset, construction, unit_index), strict=True
    ):
        receipt = artifact["attempt_receipt"]
        receipts.append(receipt)
        ledger.append(
            {
                "unit_id": unit["unit_id"],
                "attempts": [
                    {
                        "attempt_index": 0,
                        "outcome": "complete",
                        "attempt_receipt_sha256": receipt["attempt_receipt_sha256"],
                    }
                ],
            }
        )
        artifact_paths.append(RUN_ROOT / arguments.dataset / "units" / f"{unit['unit_id']}.json")
    closure = close_hpo_ledger(
        plan,
        ledger,
        attempt_receipts=receipts,
        candidate_space=construction.frozen_candidate_space,
        nested_plan=construction.frozen_nested_plan,
        group_manifest=construction.group_manifest,
    )

    def manifests():
        for path in artifact_paths:
            yield _read_json(path)["hpo_output_manifest"]

    catalog = validate_hpo_output_catalog(
        plan,
        ledger,
        receipts,
        manifests(),
        candidate_space=construction.frozen_candidate_space,
        nested_plan=construction.frozen_nested_plan,
        group_manifest=construction.group_manifest,
        authoritative_rows=construction.dataset_rows,
        authoritative_dataset_sha256=construction.dataset_rows.source_sha256,
    )
    stage = build_hpo_selection_stage_commitment(
        plan,
        closure,
        ledger,
        receipts,
        catalog,
        candidate_space=construction.frozen_candidate_space,
        nested_plan=construction.frozen_nested_plan,
        group_manifest=construction.group_manifest,
        expected_catalog_validation_sha256=catalog["catalog_validation_sha256"],
    )
    evidence_dir = RUN_ROOT / arguments.dataset / "closed_evidence"
    _atomic_json(evidence_dir / "ledger.json", ledger)
    _atomic_json(evidence_dir / "attempt_receipts.json", receipts)
    _atomic_json(evidence_dir / "ledger_closure.json", closure)
    _atomic_json(evidence_dir / "output_catalog_validation.json", catalog)
    _atomic_json(evidence_dir / "selection_stage_commitment.json", stage)
    decision_hashes = []
    receipt_by_unit = {row["unit_id"]: row for row in receipts}
    unit_by_id = {row["unit_id"]: row for row in plan["units"]}
    for outer_repeat in range(3):
        for outer_fold in range(5):
            for method in MAIN_METHODS:
                unit_ids = [
                    unit_id
                    for (unit_id, row) in unit_by_id.items()
                    if row["outer_repeat"] == outer_repeat
                    and row["outer_fold"] == outer_fold
                    and (row["method"] == method)
                ]
                scope_receipts = [receipt_by_unit[unit_id] for unit_id in unit_ids]
                scope_outputs = [
                    _read_json(RUN_ROOT / arguments.dataset / "units" / f"{unit_id}.json")[
                        "hpo_output_manifest"
                    ]
                    for unit_id in unit_ids
                ]
                decision = build_hpo_selection_decision_from_stage_commitment(
                    plan,
                    closure,
                    stage,
                    scope_receipts,
                    scope_outputs,
                    candidate_space=construction.frozen_candidate_space,
                    nested_plan=construction.frozen_nested_plan,
                    group_manifest=construction.group_manifest,
                    authoritative_rows=construction.dataset_rows,
                    authoritative_dataset_sha256=construction.dataset_rows.source_sha256,
                    expected_stage_commitment_sha256=stage["stage_commitment_sha256"],
                    method=method,
                    outer_repeat=outer_repeat,
                    outer_fold=outer_fold,
                    unit_index=unit_index,
                )
                path = (
                    evidence_dir
                    / "selection_decisions"
                    / f"r{outer_repeat}_f{outer_fold}_{method}.json"
                )
                _atomic_json(path, decision)
                decision_hashes.append(
                    {
                        "outer_repeat": outer_repeat,
                        "outer_fold": outer_fold,
                        "method": method,
                        "selection_decision_manifest_sha256": decision[
                            "selection_decision_manifest_sha256"
                        ],
                    }
                )
    final = _artifact(
        {
            "schema": _identity("hpo_dataset_closure"),
            "status": "FORMAL_HPO_CLOSED_SCOPE_LOCAL_SELECTION_COMPLETE_OUTER_STILL_CLOSED",
            "dataset_id": arguments.dataset,
            "planned_unit_count": len(plan["units"]),
            "ledger_closure_sha256": closure["closure_sha256"],
            "output_catalog_validation_sha256": catalog["catalog_validation_sha256"],
            "selection_stage_commitment_sha256": stage["stage_commitment_sha256"],
            "selection_decision_catalog_sha256": canonical_sha256(decision_hashes),
            "selection_decision_count": len(decision_hashes),
            "outer_test_accessed": False,
        },
        "formal_hpo_dataset_closure_sha256",
    )
    _atomic_json(evidence_dir / "selection_closure.json", final)
    print(json.dumps(final))


def status(arguments: argparse.Namespace) -> None:
    datasets = [arguments.dataset] if arguments.dataset else list(DATASETS)
    rows = []
    for dataset in datasets:
        unit_dir = RUN_ROOT / dataset / "units"
        failure_dir = RUN_ROOT / dataset / "failures"
        rows.append(
            {
                "dataset_id": dataset,
                "complete_unit_artifact_count": (
                    len(list(unit_dir.glob("*.json"))) if unit_dir.exists() else 0
                ),
                "failed_unit_artifact_count": (
                    len(list(failure_dir.glob("*.json"))) if failure_dir.exists() else 0
                ),
                "planned_unit_count": 32400,
                "outer_test_accessed": False,
            }
        )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def self_test(_: argparse.Namespace) -> None:
    target = RUN_ROOT / ".self_test.json"
    payload = _artifact({"schema": "formal_hpo_self_test_v1", "value": [1, 2, 3]}, "sha256")
    _atomic_json(target, payload)
    _verify_artifact(_read_json(target), "sha256")
    target.unlink()
    indices = [
        [index for index in range(101) if index % SHARD_COUNT == shard]
        for shard in range(SHARD_COUNT)
    ]
    flattened = sorted((value for shard in indices for value in shard))
    if flattened != list(range(101)) or sum(map(len, indices)) != 101:
        raise SearchError("sharding self-test failed")
    protocol_fields = _protocol_payload(_registered_implementation(), _environment()[1])
    if (
        protocol_fields["outer_test_accessed"] is not False
        or protocol_fields["worker_shards"] != SHARD_COUNT
    ):
        raise SearchError("protocol self-test failed")
    print(json.dumps({"status": "SELF_TEST_OK", "runner_sha256": _registered_runner_identity()}))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.set_defaults(function=prepare)
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    worker_parser.add_argument("--shard", type=int, required=True)
    worker_parser.add_argument("--shards", type=int, default=SHARD_COUNT)
    worker_parser.set_defaults(function=worker)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    finalize_parser.set_defaults(function=finalize)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--dataset", choices=tuple(DATASETS))
    status_parser.set_defaults(function=status)
    self_test_parser = subparsers.add_parser("self-test")
    self_test_parser.set_defaults(function=self_test)
    arguments = parser.parse_args()
    started = time.monotonic()
    try:
        arguments.function(arguments)
    except BaseException as error:
        print(
            json.dumps(
                {
                    "status": "FAILED",
                    "command": arguments.command,
                    "error_class": type(error).__qualname__,
                    "elapsed_seconds": round(time.monotonic() - started, 6),
                }
            ),
            file=sys.stderr,
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
