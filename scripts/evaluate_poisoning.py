"""Run the frozen 12-unit rapid malicious-client stress matrix."""

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
import math
import os
import tempfile
import traceback
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
import evaluate_attacks as common
import outer_training as outer
from fedsift.method_dispatch import build_method_execution_spec
from fedsift.outer_refit import outer_refit_candidate_configuration
import fedsift.train_unit as train_unit
from fedsift.training_budget_factory import (
    build_training_budget,
    validate_training_budget_factory_result,
)
from outer_preprocessing import (
    preprocess_outer_refit_capability,
    validate_outer_refit_preprocessing,
)

ROOT = Path(__file__).resolve().parents[1]
TOP = results_root()
OUTER_PLAN_PATH = ROOT / "config" / "experiment_plan.json"
RESULT_ROOT = TOP / "followup" / "poisoning"


class PoisoningError(RuntimeError):
    pass


def poisoned_data(clean: Any, malicious: str) -> Any:
    tables = []
    for table in clean.roles:
        labels = (
            tuple((1 - int(y) for y in table.labels)) if table.role == malicious else table.labels
        )
        tables.append(
            train_unit._validated_role_table(table.role, table.row_ids, table.features, labels)
        )
    data_hash = train_unit._sha256(
        {
            "schema": _identity("sealed_training_unit_data"),
            "capability_sha256": clean.capability_sha256,
            "preprocessing_artifact_sha256": clean.preprocessing_artifact_sha256,
            "role_table_sha256": [table.table_sha256 for table in tables],
        }
    )
    return train_unit.SealedUnitData(
        capability_sha256=clean.capability_sha256,
        preprocessing_artifact_sha256=clean.preprocessing_artifact_sha256,
        roles=tuple(tables),
        data_sha256=data_hash,
        _factory_seal=train_unit._DATA_FACTORY_SEAL,
    )


def state_scaled(
    local: Mapping[str, torch.Tensor], base: Mapping[str, torch.Tensor], factor: float
) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        ((name, base[name] + factor * (value - base[name])) for (name, value) in local.items())
    )


def state_norm(local: Mapping[str, torch.Tensor], base: Mapping[str, torch.Tensor]) -> float:
    return math.sqrt(
        sum((float(torch.sum((local[name] - base[name]).double() ** 2).item()) for name in local))
    )


class LocalUpdateHook:

    def __init__(self, budget: Any, malicious: str, factor: float | None):
        self.order = [
            client
            for round_clients in budget.participating_clients_by_round
            for client in round_clients
        ]
        self.malicious = malicious
        self.factor = factor
        self.index = 0
        self.raw = []
        self.released = []
        self.original_non = train_unit._execute_nonprivate_client
        self.original_private = train_unit._execute_private_client

    def _apply(self, result: tuple[Any, ...], start: Mapping[str, torch.Tensor]) -> tuple[Any, ...]:
        if self.index >= len(self.order):
            raise PoisoningError("client-call schedule exceeded budget")
        client = self.order[self.index]
        self.index += 1
        local = result[0]
        if client == self.malicious:
            self.raw.append(state_norm(local, start))
            if self.factor is not None:
                local = state_scaled(local, start, self.factor)
            self.released.append(state_norm(local, start))
        return (local, *result[1:])

    def install(self) -> None:
        hook = self

        def non(*args: Any, **kwargs: Any):
            return hook._apply(hook.original_non(*args, **kwargs), kwargs["start_state"])

        def private(*args: Any, **kwargs: Any):
            return hook._apply(hook.original_private(*args, **kwargs), kwargs["start_state"])

        train_unit._execute_nonprivate_client = non
        train_unit._execute_private_client = private

    def restore(self) -> None:
        train_unit._execute_nonprivate_client = self.original_non
        train_unit._execute_private_client = self.original_private


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    p = np.clip(probabilities, 1e-12, 1 - 1e-12)
    pred = (p >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (labels == 1)))
    tn = int(np.sum((pred == 0) & (labels == 0)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "log_loss": float(log_loss(labels, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, p)),
        "auroc": float(roc_auc_score(labels, p)),
        "average_precision": float(average_precision_score(labels, p)),
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": float((sens + spec) / 2),
    }


def prepare_target(
    unit: Mapping[str, Any], upstream: tuple[Any, ...]
) -> tuple[Any, Any, Any, dict[str, Any]]:
    authorization, construction, closure, ledger, receipts, policy = upstream
    _, _, capability = outer._build_refit_context(unit, construction, closure, ledger, receipts)
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
    sealed = train_unit.seal_preprocessed_unit_data(
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
    private_rows = {f"client_{i}": sealed.role(f"client_{i}").row_ids for i in range(5)}
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
    return (capability, sealed, budget_result.budget, preprocessed.artifact)


def execute(
    spec: Mapping[str, Any], unit: Mapping[str, Any], prepared: tuple[Any, Any, Any, dict[str, Any]]
) -> dict[str, Any]:
    capability, clean, budget, _ = prepared
    cap_hash = str(capability["capability_sha256"])
    malicious = f"client_{int(spec['attack_seed']) % 5}"
    data = (
        poisoned_data(clean, malicious)
        if spec["attack"] == "malicious_client_label_flip"
        else clean
    )
    factor = (
        float(spec["sign_scale_factor"])
        if spec["attack"] == "malicious_client_sign_scale"
        else None
    )
    hook = LocalUpdateHook(budget, malicious, factor)
    hook.install()
    try:
        result = train_unit.execute_training_unit(
            capability,
            data,
            budget,
            expected_capability_sha256=cap_hash,
            expected_budget_sha256=budget.budget_sha256,
        )
    finally:
        hook.restore()
    if hook.index != len(hook.order):
        raise PoisoningError("client-call schedule incomplete")
    train_unit.validate_training_unit_result(
        result,
        capability,
        data,
        budget,
        expected_capability_sha256=cap_hash,
        expected_budget_sha256=budget.budget_sha256,
    )
    clean_artifact = common.unit_artifact(spec)
    raw, raw_labels = common.load_raw(str(spec["dataset_id"]))
    transformed, _, _, _ = common.model_and_transform(clean_artifact, raw)
    weight = result.model_state["output.weight"].detach().cpu().numpy().reshape(-1)
    bias = float(result.model_state["output.bias"].detach().cpu().numpy().reshape(-1)[0])
    rows = np.asarray(clean_artifact["outer_prediction_artifact"]["row_ids"], dtype=int)
    labels = np.asarray(clean_artifact["outer_prediction_artifact"]["labels"], dtype=int)
    if not np.array_equal(labels, raw_labels[rows]):
        raise PoisoningError("outer labels differ from registered source")
    probabilities = 1 / (1 + np.exp(-np.clip(transformed[rows] @ weight + bias, -40, 40)))
    threshold = float.fromhex(
        clean_artifact["preopen_chain"]["threshold_artifact"]["threshold_receipt"][
            "selected_threshold_hex"
        ]
    )
    attacked = metrics(labels, probabilities, threshold)
    clean_metrics = {
        **clean_artifact["evaluation_report"]["probability_metrics"],
        **clean_artifact["evaluation_report"]["threshold_metrics"],
    }
    deltas = {name: attacked[name] - float(clean_metrics[name]) for name in attacked}
    rounds = result.artifact.get("round_evidence", [])
    fallback = sum((float.fromhex(item["selected_alpha_hex"]) < 1.0 for item in rounds))
    output = {
        "schema": _identity("poisoning_unit"),
        "status": "SEALED_COMPLETE",
        "spec": dict(spec),
        "malicious_client_id": malicious,
        "training_budget_sha256": budget.budget_sha256,
        "training_result_artifact_sha256": result.artifact["artifact_sha256"],
        "outer_unit_id": unit["unit_id"],
        "attacked_metrics": attacked,
        "paired_delta_attacked_minus_clean": deltas,
        "nonfinite_model_or_predictions": bool(
            any((not torch.isfinite(v).all() for v in result.model_state.values()))
            or not np.isfinite(probabilities).all()
        ),
        "malicious_raw_update_norm_mean": float(np.mean(hook.raw)),
        "malicious_released_update_norm_mean": float(np.mean(hook.released)),
        "malicious_released_update_norm_max": float(np.max(hook.released)),
        "control_fallback_round_count": int(fallback),
        "server_rounds": budget.server_rounds,
        "participating_client_count": 5,
        "failed_attempt_consumed": False,
    }
    output["poisoning_unit_sha256"] = common.sha(output)
    return output


def worker(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    plan = common.load_plan()
    specs = [s for s in plan["poisoning_robustness"]["units"] if s["dataset_id"] == args.dataset]
    outer_plan = common.read_json(OUTER_PLAN_PATH)
    common.verify(outer_plan, "rapid_outer_plan_sha256")
    unit_index = {u["unit_id"]: u for u in outer_plan["units"]}
    upstream = outer._load_hpo_upstreams(args.dataset)
    cache: dict[str, tuple[Any, Any, Any, dict[str, Any]]] = {}
    for spec in specs:
        attack_id = common.sha({"domain": "rapid_poisoning_v1", "spec": spec})
        path = RESULT_ROOT / "units" / f"{attack_id}.json"
        if path.is_file():
            result = common.read_json(path)
            common.verify(result, "poisoning_unit_sha256")
        else:
            target = str(spec["outer_unit_id"])
            unit = unit_index[target]
            if target not in cache:
                cache[target] = prepare_target(unit, upstream)
            try:
                result = execute(spec, unit, cache[target])
                common.atomic_json(path, result)
            except Exception as exc:
                failure = {
                    "schema": _identity("poisoning_failure"),
                    "status": "FAILED_ATTEMPT_CONSUMED",
                    "spec": dict(spec),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                failure["failure_sha256"] = common.sha(failure)
                common.atomic_json(RESULT_ROOT / "failures" / f"{attack_id}.json", failure)
                raise
        print(
            json.dumps(
                {
                    "status": "UNIT_COMPLETE",
                    "dataset": args.dataset,
                    "attack": spec["attack"],
                    "method": spec["method_id"],
                }
            ),
            flush=True,
        )


def status(_: argparse.Namespace) -> None:
    plan = common.load_plan()
    complete = len(list((RESULT_ROOT / "units").glob("*.json")))
    failures = len(list((RESULT_ROOT / "failures").glob("*.json")))
    print(
        json.dumps(
            {
                "status": "RUNNING" if complete < 12 else "READY_TO_CLOSE",
                "complete": complete,
                "planned": 12,
                "failures": failures,
            }
        )
    )


def finalize(_: argparse.Namespace) -> None:
    plan = common.load_plan()
    specs = plan["poisoning_robustness"]["units"]
    failures = list((RESULT_ROOT / "failures").glob("*.json"))
    if failures:
        raise PoisoningError(f"poisoning failures present: {len(failures)}")
    rows = []
    for spec in specs:
        attack_id = common.sha({"domain": "rapid_poisoning_v1", "spec": spec})
        result = common.read_json(RESULT_ROOT / "units" / f"{attack_id}.json")
        common.verify(result, "poisoning_unit_sha256")
        rows.append(
            {
                **spec,
                **result["attacked_metrics"],
                **{
                    f"delta_{k}": v
                    for (k, v) in result["paired_delta_attacked_minus_clean"].items()
                },
                "malicious_client_id": result["malicious_client_id"],
                "malicious_released_update_norm_max": result["malicious_released_update_norm_max"],
                "control_fallback_round_count": result["control_fallback_round_count"],
                "nonfinite_model_or_predictions": result["nonfinite_model_or_predictions"],
            }
        )
    common.atomic_csv(RESULT_ROOT / "poisoning_summary.csv", rows)
    closure = {
        "schema": _identity("poisoning_closure"),
        "status": "POISONING_COMPLETE_12_OF_12",
        "closed_at_utc": datetime.now(timezone.utc).isoformat(),
        "rapid_followup_plan_sha256": plan[common.PLAN_HASH_FIELD],
        "unit_count": len(rows),
        "summary_sha256": hashlib.sha256(
            (RESULT_ROOT / "poisoning_summary.csv").read_bytes()
        ).hexdigest(),
    }
    closure["closure_sha256"] = common.sha(closure)
    common.atomic_json(RESULT_ROOT / "poisoning_closure.json", closure)
    print(json.dumps({"status": closure["status"], "closure_sha256": closure["closure_sha256"]}))


def self_test(_: argparse.Namespace) -> None:
    base = OrderedDict(w=torch.tensor([1.0]))
    local = OrderedDict(w=torch.tensor([3.0]))
    scaled = state_scaled(local, base, -3.0)
    if float(scaled["w"][0]) != -5.0:
        raise PoisoningError("sign-scale self-test failed")
    print(json.dumps({"status": "SELF_TEST_OK"}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status").set_defaults(func=status)
    sub.add_parser("finalize").set_defaults(func=finalize)
    sub.add_parser("self-test").set_defaults(func=self_test)
    w = sub.add_parser("worker")
    w.add_argument("--dataset", choices=("pima", "retinopathy"), required=True)
    w.set_defaults(func=worker)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
