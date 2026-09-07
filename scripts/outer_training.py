"""Train and evaluate one registered outer-fold unit with a committed selection and one-time test access."""

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
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

TOP = results_root()
HPO_ROOT = TOP / "selection"
THRESHOLD_GRID = tuple((index / 100.0 for index in range(10, 91, 5)))


class OuterTrainingError(RuntimeError):
    pass


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise OuterTrainingError(f"cannot read JSON: {path}") from exc


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _verify_self_hash(value: Mapping[str, Any], field: str) -> None:
    if not isinstance(value, Mapping) or not isinstance(value.get(field), str):
        raise OuterTrainingError(f"artifact lacks {field}")
    payload = copy.deepcopy(dict(value))
    stored = payload.pop(field)
    if stored != _canonical_sha256(payload):
        raise OuterTrainingError(f"artifact hash differs: {field}")


def _artifact(payload: Mapping[str, Any], hash_field: str) -> dict[str, Any]:
    value = copy.deepcopy(dict(payload))
    value[hash_field] = _canonical_sha256(value)
    return value


def _v_sel_prediction(result: Any) -> Mapping[str, Any]:
    artifact = getattr(result, "artifact", None)
    if not isinstance(artifact, Mapping):
        raise OuterTrainingError("training result artifact is missing")
    predictions = artifact.get("raw_native_predictions")
    if not isinstance(predictions, list):
        raise OuterTrainingError("training result prediction roster is missing")
    matches = [row for row in predictions if row.get("role") == "v_sel"]
    if len(matches) != 1:
        raise OuterTrainingError("exactly one V_sel prediction payload is required")
    return matches[0]


def _serialized_model_state(result: Any) -> list[dict[str, Any]]:
    state = getattr(result, "model_state", None)
    if not isinstance(state, Mapping) or not state:
        raise OuterTrainingError("training result model state is missing")
    rows: list[dict[str, Any]] = []
    for name, tensor in state.items():
        try:
            array = tensor.detach().cpu().numpy()
        except Exception as exc:
            raise OuterTrainingError("model state contains a non-tensor value") from exc
        rows.append(
            {
                "name": str(name),
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "values": array.tolist(),
            }
        )
    return rows


def build_preopen_artifact_chain(
    capability: Mapping[str, Any],
    result: Any,
    *,
    endpoint_contract_sha256: str,
    refit_preprocessing_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal model, V_sel prediction, and threshold artifacts before outer open."""
    from fedsift.evaluation import (
        EvaluationScope,
        ThresholdSelectionRule,
        VSelBinding,
        select_threshold_from_v_sel,
        validate_threshold_receipt,
    )
    from fedsift.outer_refit import (
        seal_outer_refit_model_manifest,
        seal_outer_refit_prediction_manifest,
        seal_outer_refit_threshold_manifest,
        validate_outer_refit_artifact_chain,
    )

    if not isinstance(endpoint_contract_sha256, str) or len(endpoint_contract_sha256) != 64:
        raise OuterTrainingError("endpoint contract hash is invalid")
    cap_hash = capability.get("capability_sha256")
    if not isinstance(cap_hash, str):
        raise OuterTrainingError("refit capability hash is missing")
    training = getattr(result, "artifact", None)
    if not isinstance(training, Mapping):
        raise OuterTrainingError("training result artifact is missing")
    required_training = {
        "status": "complete_metric_free",
        "capability_kind": "outer_refit",
        "capability_sha256": cap_hash,
        "outer_test_accessed": False,
        "evaluation_metrics_computed": False,
    }
    for field, value in required_training.items():
        if training.get(field) != value:
            raise OuterTrainingError(f"training result differs: {field}")
    model_artifact = _artifact(
        {
            "schema": _identity("outer_model_artifact"),
            "status": "SEALED_BEFORE_OUTER_OPEN",
            "capability_sha256": cap_hash,
            "training_result_artifact_sha256": training["artifact_sha256"],
            "model_manifest": training["model_manifest"],
            "model_manifest_sha256": training["model_manifest_sha256"],
            "final_model_state_sha256": training["final_model_state_sha256"],
            "serialized_model_state": _serialized_model_state(result),
            "refit_preprocessing_artifact": (
                None
                if refit_preprocessing_artifact is None
                else copy.deepcopy(dict(refit_preprocessing_artifact))
            ),
            "outer_test_accessed": False,
        },
        "model_artifact_sha256",
    )
    model_stage = seal_outer_refit_model_manifest(
        capability,
        expected_capability_sha256=cap_hash,
        model_artifact_manifest_sha256=model_artifact["model_artifact_sha256"],
    )
    raw = _v_sel_prediction(result)
    prediction_artifact = _artifact(
        {
            "schema": _identity("outer_v_sel_prediction_artifact"),
            "status": "SEALED_BEFORE_OUTER_OPEN",
            "capability_sha256": cap_hash,
            "model_stage_manifest_sha256": model_stage["stage_manifest_sha256"],
            "v_sel_membership_sha256": capability["split_bindings"]["v_sel_membership_sha256"],
            "row_ids": list(raw["row_ids"]),
            "labels": list(raw["labels"]),
            "probabilities": list(raw["probabilities"]),
            "prediction_payload_sha256": raw["payload_sha256"],
            "outer_test_accessed": False,
        },
        "prediction_artifact_sha256",
    )
    prediction_stage = seal_outer_refit_prediction_manifest(
        capability,
        model_stage,
        expected_capability_sha256=cap_hash,
        prediction_artifact_manifest_sha256=prediction_artifact["prediction_artifact_sha256"],
    )
    scope = capability["scope"]
    candidate = capability["candidate"]
    evaluation_scope = EvaluationScope(
        study_id=str(capability["study_id"]),
        outer_repeat=int(scope["outer_repeat"]),
        outer_fold=int(scope["outer_fold"]),
        method_id=str(scope["method_identity"]),
        candidate_id=str(candidate["candidate_id"]),
    )
    binding = VSelBinding(
        scope=evaluation_scope,
        membership_sha256=str(capability["split_bindings"]["v_sel_membership_sha256"]),
        prediction_artifact_sha256=prediction_artifact["prediction_artifact_sha256"],
    )
    rule = ThresholdSelectionRule(target_sensitivity=0.85, candidate_thresholds=THRESHOLD_GRID)
    receipt = select_threshold_from_v_sel(
        v_sel_binding=binding,
        row_ids=raw["row_ids"],
        labels=raw["labels"],
        probabilities=raw["probabilities"],
        rule=rule,
    )
    validate_threshold_receipt(
        receipt,
        expected_receipt_sha256=receipt["receipt_sha256"],
        expected_v_sel_binding=binding,
        expected_rule=rule,
        row_ids=raw["row_ids"],
        labels=raw["labels"],
        probabilities=raw["probabilities"],
    )
    threshold_artifact = _artifact(
        {
            "schema": _identity("outer_threshold_artifact"),
            "status": "SEALED_V_SEL_ONLY_BEFORE_OUTER_OPEN",
            "capability_sha256": cap_hash,
            "prediction_stage_manifest_sha256": prediction_stage["stage_manifest_sha256"],
            "endpoint_contract_sha256": endpoint_contract_sha256,
            "threshold_rule": rule.manifest(),
            "threshold_receipt": receipt,
            "outer_test_accessed": False,
        },
        "threshold_artifact_sha256",
    )
    threshold_stage = seal_outer_refit_threshold_manifest(
        capability,
        prediction_stage,
        expected_capability_sha256=cap_hash,
        threshold_artifact_manifest_sha256=threshold_artifact["threshold_artifact_sha256"],
    )
    validate_outer_refit_artifact_chain(
        capability,
        model_stage,
        prediction_stage,
        threshold_stage,
        expected_capability_sha256=cap_hash,
    )
    bundle = {
        "schema": _identity("outer_preopen_artifact_chain"),
        "status": "SEALED_CHAIN_READY_FOR_ONE_TIME_OUTER_GATE",
        "capability_sha256": cap_hash,
        "model_artifact": model_artifact,
        "model_stage": model_stage,
        "prediction_artifact": prediction_artifact,
        "prediction_stage": prediction_stage,
        "threshold_artifact": threshold_artifact,
        "threshold_stage": threshold_stage,
        "outer_test_accessed": False,
    }
    bundle["chain_sha256"] = _canonical_sha256(bundle)
    return bundle


def issue_preopen_outer_gate(
    capability: Mapping[str, Any],
    chain: Mapping[str, Any],
    hpo_plan: Mapping[str, Any],
    closure: Mapping[str, Any],
    ledger_entries: Any,
    *,
    attempt_receipts: Any,
    candidate_space: Mapping[str, Any],
    nested_plan: Mapping[str, Any],
    group_manifest: Mapping[str, Any],
    selection_receipt: Mapping[str, Any],
    one_time_access_authorization_sha256: str,
    matched_ablation_inheritance: Mapping[str, Any] | None = None,
):
    """Issue an opaque, unopened gate from an exact sealed pre-open chain."""
    from fedsift.outer_refit import issue_outer_test_gate

    if not isinstance(chain, Mapping):
        raise OuterTrainingError("pre-open chain is missing")
    payload = copy.deepcopy(dict(chain))
    stored = payload.pop("chain_sha256", None)
    if stored != _canonical_sha256(payload):
        raise OuterTrainingError("pre-open chain hash differs")
    if (
        chain.get("status") != "SEALED_CHAIN_READY_FOR_ONE_TIME_OUTER_GATE"
        or chain.get("outer_test_accessed") is not False
        or chain.get("capability_sha256") != capability.get("capability_sha256")
    ):
        raise OuterTrainingError("pre-open chain is not eligible for gate issuance")
    gate = issue_outer_test_gate(
        capability,
        chain["model_stage"],
        chain["prediction_stage"],
        chain["threshold_stage"],
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
    manifest = gate.public_manifest()
    return (gate, manifest)


def build_outer_unit_artifact(
    unit: Mapping[str, Any],
    chain: Mapping[str, Any],
    gate_manifest: Mapping[str, Any],
    access_receipt: Mapping[str, Any],
    outer_preprocessing_artifact: Mapping[str, Any],
    outer_prediction_artifact: Mapping[str, Any],
    evaluation_report: Mapping[str, Any],
    *,
    training_evidence: Mapping[str, Any],
    recovery_linkage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one immutable completed-unit artifact after a successful open."""
    if not isinstance(training_evidence, Mapping):
        raise OuterTrainingError("training evidence is missing")
    for name in ("training_artifact_sha256", "resource_report_sha256", "privacy_evidence_sha256"):
        value = training_evidence.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise OuterTrainingError(f"{name} is invalid")
    if access_receipt.get("outcome") != "opened" or access_receipt.get("rows_released") is not True:
        raise OuterTrainingError("completed outer unit requires a successful access receipt")
    if chain.get("capability_sha256") != gate_manifest.get("bindings", {}).get(
        "refit_capability_sha256"
    ):
        raise OuterTrainingError("outer unit chain and gate capability differ")
    if gate_manifest.get("gate_sha256") != access_receipt.get("gate_sha256"):
        raise OuterTrainingError("outer unit gate and access receipt differ")
    if outer_preprocessing_artifact.get("access_receipt_sha256") != access_receipt.get(
        "access_receipt_sha256"
    ):
        raise OuterTrainingError("outer preprocessing and access receipt differ")
    if evaluation_report.get("report_sha256") is None:
        raise OuterTrainingError("outer evaluation report is not sealed")
    payload = {
        "schema": _identity("outer_unit_artifact"),
        "status": "SEALED_COMPLETE_OUTER_EVALUATION",
        "unit_identity": {
            key: copy.deepcopy(unit[key])
            for key in (
                "unit_id",
                "dataset_id",
                "outer_repeat",
                "outer_fold",
                "eval_seed",
                "arm_type",
                "method_id",
            )
        },
        "preopen_chain": copy.deepcopy(dict(chain)),
        "gate_manifest": copy.deepcopy(dict(gate_manifest)),
        "access_receipt": copy.deepcopy(dict(access_receipt)),
        "outer_preprocessing_artifact": copy.deepcopy(dict(outer_preprocessing_artifact)),
        "outer_prediction_artifact": copy.deepcopy(dict(outer_prediction_artifact)),
        "evaluation_report": copy.deepcopy(dict(evaluation_report)),
        "training_evidence": copy.deepcopy(dict(training_evidence)),
        "outer_test_accessed": True,
        "candidate_selection_performed_after_outer_open": False,
    }
    if recovery_linkage is not None:
        payload["recovery_linkage"] = copy.deepcopy(dict(recovery_linkage))
    artifact = _artifact(payload, "formal_outer_unit_artifact_sha256")
    validate_outer_unit_artifact(artifact, expected_unit=unit)
    return artifact


def validate_outer_unit_artifact(
    artifact: Mapping[str, Any], *, expected_unit: Mapping[str, Any]
) -> None:
    _verify_self_hash(artifact, "formal_outer_unit_artifact_sha256")
    if (
        artifact.get("schema") != _identity("outer_unit_artifact")
        or artifact.get("status") != "SEALED_COMPLETE_OUTER_EVALUATION"
        or artifact.get("outer_test_accessed") is not True
        or (artifact.get("candidate_selection_performed_after_outer_open") is not False)
    ):
        raise OuterTrainingError("formal outer unit top-level contract differs")
    identity = artifact.get("unit_identity")
    if not isinstance(identity, Mapping):
        raise OuterTrainingError("formal outer unit identity is missing")
    for field in (
        "unit_id",
        "dataset_id",
        "outer_repeat",
        "outer_fold",
        "eval_seed",
        "arm_type",
        "method_id",
    ):
        if identity.get(field) != expected_unit.get(field):
            raise OuterTrainingError(f"formal outer unit identity differs: {field}")
    gate = artifact.get("gate_manifest")
    receipt = artifact.get("access_receipt")
    if not isinstance(gate, Mapping) or not isinstance(receipt, Mapping):
        raise OuterTrainingError("formal outer gate evidence is missing")
    if (
        receipt.get("gate_sha256") != gate.get("gate_sha256")
        or receipt.get("outcome") != "opened"
        or receipt.get("rows_released") is not True
    ):
        raise OuterTrainingError("formal outer access evidence differs")
    report = artifact.get("evaluation_report")
    if not isinstance(report, Mapping):
        raise OuterTrainingError("formal outer evaluation report is missing")
    _verify_self_hash(report, "report_sha256")
    chain = artifact.get("preopen_chain")
    if not isinstance(chain, Mapping):
        raise OuterTrainingError("formal outer pre-open chain is missing")
    _verify_self_hash(chain, "chain_sha256")
    if not isinstance(artifact.get("training_evidence"), Mapping):
        raise OuterTrainingError("formal outer training evidence is missing")
    recovery = artifact.get("recovery_linkage")
    if recovery is not None:
        if not isinstance(recovery, Mapping):
            raise OuterTrainingError("formal outer recovery linkage is malformed")
        required = {
            "schema": _identity("outer_linked_recovery"),
            "attempt_index": 1,
            "second_outer_gate_issued": False,
            "second_outer_open_performed": False,
        }
        for field, expected in required.items():
            if recovery.get(field) != expected:
                raise OuterTrainingError(f"formal outer recovery linkage differs: {field}")
        for field in (
            "original_failed_outer_unit_artifact_sha256",
            "original_access_journal_sha256",
            "original_gate_sha256",
            "original_access_receipt_sha256",
            "recovery_authorization_sha256",
        ):
            value = recovery.get(field)
            if not isinstance(value, str) or len(value) != 64:
                raise OuterTrainingError(f"formal outer recovery hash is invalid: {field}")
        if recovery["original_gate_sha256"] != gate.get("gate_sha256") or recovery[
            "original_access_receipt_sha256"
        ] != receipt.get("access_receipt_sha256"):
            raise OuterTrainingError("formal outer recovery linkage and access evidence differ")


def build_outer_failure_artifact(
    unit: Mapping[str, Any],
    error: BaseException,
    *,
    outer_test_accessed: bool,
    gate_sha256: str | None = None,
    access_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Persist a non-retryable failure without serializing exception text."""
    failure_code = "post_outer_open_failure" if outer_test_accessed else "pre_outer_open_failure"
    return _artifact(
        {
            "schema": _identity("outer_failed_unit_artifact"),
            "status": "SEALED_FAILED_ATTEMPT_SHARD_STOPPED",
            "unit_id": unit["unit_id"],
            "dataset_id": unit["dataset_id"],
            "failure_code": failure_code,
            "exception_class_sha256": hashlib.sha256(type(error).__qualname__.encode()).hexdigest(),
            "gate_sha256": gate_sha256,
            "access_receipt_sha256": access_receipt_sha256,
            "outer_test_accessed": bool(outer_test_accessed),
            "automatic_retry_authorized": False,
        },
        "failed_outer_unit_artifact_sha256",
    )


def _load_hpo_upstreams(dataset: str):
    import search as hpo_runner

    authorization = hpo_runner._authorization()
    construction = hpo_runner._construction(dataset, authorization)
    closed = HPO_ROOT / dataset / "closed_evidence"
    closure = _read_json(closed / "ledger_closure.json")
    ledger = _read_json(closed / "ledger.json")
    receipts = _read_json(closed / "attempt_receipts.json")
    if not isinstance(ledger, list) or not isinstance(receipts, list):
        raise OuterTrainingError("closed HPO ledger or receipt catalog is malformed")
    return (authorization, construction, closure, ledger, receipts, hpo_runner._policy(dataset))


def _build_refit_context(
    unit: Mapping[str, Any],
    construction: Any,
    closure: Mapping[str, Any],
    ledger: Any,
    receipts: Any,
):
    from fedsift.hpo_plan import inherit_matched_ablation
    from fedsift.outer_refit import build_outer_refit_capability, build_outer_selection_receipt

    binding = unit.get("selection_binding")
    if not isinstance(binding, Mapping):
        raise OuterTrainingError("outer unit selection binding is missing")
    selection_method = str(binding["selection_method"])
    decision_path = resolve_record_path(str(binding["selection_decision_path"]))
    decision = _read_json(decision_path)
    if decision.get("selection_decision_manifest_sha256") != binding.get(
        "selection_decision_manifest_sha256"
    ):
        raise OuterTrainingError("outer unit selection decision binding differs")
    plan = construction.complete_hpo_plan
    space = construction.frozen_candidate_space
    nested = construction.frozen_nested_plan
    groups = construction.group_manifest
    selection = build_outer_selection_receipt(
        plan,
        closure,
        ledger,
        attempt_receipts=receipts,
        candidate_space=space,
        nested_plan=nested,
        group_manifest=groups,
        method=selection_method,
        outer_repeat=int(unit["outer_repeat"]),
        outer_fold=int(unit["outer_fold"]),
        candidate_id=str(binding["candidate_id"]),
        candidate_sha256=str(binding["candidate_sha256"]),
        selection_decision_manifest_sha256=str(binding["selection_decision_manifest_sha256"]),
    )
    inheritance = None
    if unit["arm_type"] == "matched_fedsift_ablation":
        inheritance = inherit_matched_ablation(
            plan,
            closure,
            ledger,
            attempt_receipts=receipts,
            candidate_space=space,
            nested_plan=nested,
            group_manifest=groups,
            ablation_id=str(unit["method_id"]),
            outer_repeat=int(unit["outer_repeat"]),
            outer_fold=int(unit["outer_fold"]),
            parent_candidate_id=str(binding["candidate_id"]),
            parent_candidate_sha256=str(binding["candidate_sha256"]),
            parent_selection_receipt_sha256=str(selection["selection_receipt_sha256"]),
        )
    capability = build_outer_refit_capability(
        plan,
        closure,
        ledger,
        attempt_receipts=receipts,
        candidate_space=space,
        nested_plan=nested,
        group_manifest=groups,
        selection_receipt=selection,
        eval_seed=int(unit["eval_seed"]),
        matched_ablation_inheritance=inheritance,
    )
    return (selection, inheritance, capability)


def _public_serializable(value: Any) -> Any:
    """Serialize public dataclass fields while excluding factory seals."""
    if is_dataclass(value) and (not isinstance(value, type)):
        return {
            field.name: _public_serializable(getattr(value, field.name))
            for field in fields(value)
            if not field.name.startswith("_")
        }
    if isinstance(value, Mapping):
        return {str(key): _public_serializable(item) for (key, item) in value.items()}
    if isinstance(value, (list, tuple)):
        return [_public_serializable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise OuterTrainingError(f"unsupported public evidence value type: {type(value).__qualname__}")


def _training_evidence(result: Any) -> dict[str, Any]:
    privacy = copy.deepcopy(result.artifact["privacy_evidence"])
    resources = copy.deepcopy(result.resource_summary)
    evidence = {
        "training_artifact_sha256": result.artifact["artifact_sha256"],
        "resource_report_sha256": resources["report_sha256"],
        "privacy_evidence_sha256": _canonical_sha256(privacy),
        "resource_summary": resources,
        "round_resource_receipts": [
            copy.deepcopy(value) for value in result.round_resource_receipts
        ],
        "privacy_evidence": privacy,
        "sequential_client_reports": [
            _public_serializable(value) for value in result.sequential_client_reports
        ],
        "parallel_privacy_report": (
            None
            if result.parallel_privacy_report is None
            else _public_serializable(result.parallel_privacy_report)
        ),
    }
    evidence["training_evidence_sha256"] = _canonical_sha256(evidence)
    return evidence


def execute_outer_unit(
    unit: Mapping[str, Any], *, open_outer: bool = True, access_journal_path: Path | None = None
) -> dict[str, Any]:
    """Execute one committed unit; caller must persist success or failure once."""
    import torch
    from fedsift.evaluation import (
        EvaluationScope,
        OuterTestBinding,
        ThresholdSelectionRule,
        VSelBinding,
        evaluate_outer_test,
        validate_threshold_receipt,
    )
    from fedsift.method_dispatch import build_method_execution_spec
    from fedsift.modeling import predict_probabilities
    from fedsift.outer_refit import open_outer_test_once, outer_refit_candidate_configuration
    from fedsift.train_unit import (
        execute_training_unit,
        seal_preprocessed_unit_data,
        validate_training_unit_result,
    )
    from fedsift.training_budget_factory import (
        build_training_budget,
        validate_training_budget_factory_result,
    )
    from outer_preprocessing import (
        apply_frozen_preprocessing_to_opened_outer_test,
        preprocess_outer_refit_capability,
        validate_outer_refit_preprocessing,
    )

    authorization, construction, closure, ledger, receipts, policy = _load_hpo_upstreams(
        str(unit["dataset_id"])
    )
    selection, inheritance, capability = _build_refit_context(
        unit, construction, closure, ledger, receipts
    )
    cap_hash = str(capability["capability_sha256"])
    preprocessed = preprocess_outer_refit_capability(
        capability,
        construction.dataset_rows,
        construction.group_manifest,
        expected_capability_sha256=cap_hash,
    )
    validate_outer_refit_preprocessing(
        preprocessed,
        capability,
        construction.dataset_rows,
        construction.group_manifest,
        expected_capability_sha256=cap_hash,
    )
    labels_by_row = dict(zip(construction.dataset_rows.row_ids, construction.dataset_rows.labels))
    sealed = seal_preprocessed_unit_data(
        capability, preprocessed, labels_by_row, expected_capability_sha256=cap_hash
    )
    configuration = outer_refit_candidate_configuration(
        capability, expected_capability_sha256=cap_hash
    )
    parameters = configuration["parameters"]
    spec = build_method_execution_spec(
        method_id=str(unit["method_id"]),
        candidate_parameters=parameters,
        mechanism_switch=configuration["mechanism_switch"],
    )
    private_rows = {f"client_{index}": sealed.role(f"client_{index}").row_ids for index in range(5)}
    budget_result = build_training_budget(
        spec,
        parameters,
        private_rows,
        policy,
        expected_dispatch_sha256=spec.dispatch_sha256,
        expected_candidate_parameters_sha256=str(capability["candidate"]["parameters_sha256"]),
        expected_policy_sha256=policy.policy_sha256,
    )
    validate_training_budget_factory_result(
        budget_result,
        spec,
        parameters,
        private_rows,
        policy,
        expected_dispatch_sha256=spec.dispatch_sha256,
        expected_candidate_parameters_sha256=str(capability["candidate"]["parameters_sha256"]),
        expected_policy_sha256=policy.policy_sha256,
    )
    result = execute_training_unit(
        capability,
        sealed,
        budget_result.budget,
        expected_capability_sha256=cap_hash,
        expected_budget_sha256=budget_result.budget.budget_sha256,
    )
    validate_training_unit_result(
        result,
        capability,
        sealed,
        budget_result.budget,
        expected_capability_sha256=cap_hash,
        expected_budget_sha256=budget_result.budget.budget_sha256,
    )
    endpoint_hash = registered_study()["endpoint_contract_sha256"]
    chain = build_preopen_artifact_chain(
        capability,
        result,
        endpoint_contract_sha256=endpoint_hash,
        refit_preprocessing_artifact=preprocessed.artifact,
    )
    if not open_outer:
        evidence = _training_evidence(result)
        return _artifact(
            {
                "schema": _identity("outer_real_data_preopen_smoke"),
                "status": "PREOPEN_TRAINING_AND_CHAIN_COMPLETE_OUTER_UNREAD",
                "unit_id": unit["unit_id"],
                "dataset_id": unit["dataset_id"],
                "method_id": unit["method_id"],
                "capability_sha256": cap_hash,
                "preprocessing_artifact_sha256": preprocessed.artifact["artifact_sha256"],
                "training_evidence_sha256": evidence["training_evidence_sha256"],
                "preopen_chain_sha256": chain["chain_sha256"],
                "threshold_receipt_sha256": chain["threshold_artifact"]["threshold_receipt"][
                    "receipt_sha256"
                ],
                "quality_metrics_persisted": False,
                "outer_test_accessed": False,
            },
            "smoke_artifact_sha256",
        )
    recovery_linkage = None
    gate, gate_manifest = issue_preopen_outer_gate(
        capability,
        chain,
        construction.complete_hpo_plan,
        closure,
        ledger,
        attempt_receipts=receipts,
        candidate_space=construction.frozen_candidate_space,
        nested_plan=construction.frozen_nested_plan,
        group_manifest=construction.group_manifest,
        selection_receipt=selection,
        one_time_access_authorization_sha256=str(unit["one_time_access_authorization_sha256"]),
        matched_ablation_inheritance=inheritance,
    )
    if access_journal_path is not None:
        _atomic_json(
            access_journal_path,
            _artifact(
                {
                    "schema": _identity("outer_access_journal"),
                    "status": "GATE_ISSUED_OUTER_ACCESS_UNCERTAIN_IF_INTERRUPTED",
                    "unit_id": unit["unit_id"],
                    "gate_manifest": gate_manifest,
                    "access_receipt": None,
                },
                "access_journal_sha256",
            ),
        )
    opened = open_outer_test_once(
        gate,
        construction.complete_hpo_plan,
        construction.frozen_candidate_space,
        construction.frozen_nested_plan,
        construction.group_manifest,
        expected_gate_sha256=str(gate_manifest["gate_sha256"]),
        outer_repeat=int(unit["outer_repeat"]),
        outer_fold=int(unit["outer_fold"]),
        eval_seed=int(unit["eval_seed"]),
        access_attempt_id=f"formal-outer-{unit['unit_id']}",
    )
    if access_journal_path is not None:
        _atomic_json(
            access_journal_path,
            _artifact(
                {
                    "schema": _identity("outer_access_journal"),
                    "status": "OUTER_OPEN_RECORDED_PENDING_UNIT_SEAL",
                    "unit_id": unit["unit_id"],
                    "gate_manifest": gate_manifest,
                    "access_receipt": opened.access_receipt,
                },
                "access_journal_sha256",
            ),
        )
    opened_matrix = apply_frozen_preprocessing_to_opened_outer_test(
        preprocessed,
        construction.dataset_rows,
        opened.row_ids,
        opened.access_receipt,
        gate_manifest,
        expected_refit_artifact_sha256=str(preprocessed.artifact["artifact_sha256"]),
    )
    probabilities = (
        predict_probabilities(
            result.model,
            torch.tensor(opened_matrix.matrix, dtype=torch.float64),
            state=result.model_state,
        )
        .detach()
        .cpu()
        .tolist()
    )
    labels = [labels_by_row[row_id] for row_id in opened.row_ids]
    outer_prediction = _artifact(
        {
            "schema": _identity("outer_prediction_artifact"),
            "status": "SEALED_AFTER_ONE_TIME_OUTER_OPEN",
            "unit_id": unit["unit_id"],
            "gate_sha256": gate_manifest["gate_sha256"],
            "access_receipt_sha256": opened.access_receipt["access_receipt_sha256"],
            "outer_preprocessing_artifact_sha256": opened_matrix.artifact["artifact_sha256"],
            "row_ids": list(opened.row_ids),
            "labels": labels,
            "probabilities": probabilities,
        },
        "outer_prediction_artifact_sha256",
    )
    scope = EvaluationScope(
        study_id=str(capability["study_id"]),
        outer_repeat=int(unit["outer_repeat"]),
        outer_fold=int(unit["outer_fold"]),
        method_id=str(unit["method_id"]),
        candidate_id=str(capability["candidate"]["candidate_id"]),
    )
    vsel_artifact = chain["prediction_artifact"]
    vsel_binding = VSelBinding(
        scope=scope,
        membership_sha256=str(capability["split_bindings"]["v_sel_membership_sha256"]),
        prediction_artifact_sha256=str(vsel_artifact["prediction_artifact_sha256"]),
    )
    rule = ThresholdSelectionRule(target_sensitivity=0.85, candidate_thresholds=THRESHOLD_GRID)
    threshold_receipt = chain["threshold_artifact"]["threshold_receipt"]
    verified_threshold = validate_threshold_receipt(
        threshold_receipt,
        expected_receipt_sha256=str(threshold_receipt["receipt_sha256"]),
        expected_v_sel_binding=vsel_binding,
        expected_rule=rule,
        row_ids=vsel_artifact["row_ids"],
        labels=vsel_artifact["labels"],
        probabilities=vsel_artifact["probabilities"],
    )
    outer_binding = OuterTestBinding(
        scope=scope,
        membership_sha256=str(gate_manifest["outer_test_seal"]["membership_sha256"]),
        prediction_artifact_sha256=str(outer_prediction["outer_prediction_artifact_sha256"]),
    )
    report = evaluate_outer_test(
        outer_test_binding=outer_binding,
        row_ids=opened.row_ids,
        labels=labels,
        probabilities=probabilities,
        verified_threshold_receipt=verified_threshold,
    )
    evidence = _training_evidence(result)
    artifact = build_outer_unit_artifact(
        unit,
        chain,
        gate_manifest,
        opened.access_receipt,
        opened_matrix.artifact,
        outer_prediction,
        report,
        training_evidence=evidence,
        recovery_linkage=recovery_linkage,
    )
    if authorization.get("outer_test_accessed") is not False:
        raise OuterTrainingError("HPO authorization boundary changed during outer unit")
    return artifact
