"""Execute the registered membership and final-model reconstruction audits."""

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
import csv
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
TOP = results_root()
PLAN_PATH = ROOT / "config" / "evaluation_plan.json"
OUTER_ROOT = TOP / "main"
OUTER_CLOSURE = TOP / "main_summary" / "experiment_closure.json"
RESULT_ROOT = TOP / "followup"
PLAN_HASH_FIELD = "rapid_followup_plan_sha256"


class AttackError(RuntimeError):
    pass


def sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify(value: Mapping[str, Any], field: str) -> None:
    payload = copy.deepcopy(dict(value))
    stored = payload.pop(field, None)
    if stored != sha(payload):
        raise AttackError(f"self-hash mismatch: {field}")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def load_plan() -> dict[str, Any]:
    if not OUTER_CLOSURE.is_file():
        raise AttackError("experiment closure is missing")
    plan = read_json(PLAN_PATH)
    verify(plan, PLAN_HASH_FIELD)
    return plan


def load_raw(dataset: str) -> tuple[np.ndarray, np.ndarray]:
    if dataset == "pima":
        frame = pd.read_csv(ROOT / "data" / "diabetes.csv")
        return (frame.iloc[:, :-1].to_numpy(float), frame.iloc[:, -1].to_numpy(int))
    records, _ = arff.loadarff(
        ROOT / "data" / "raw" / "diabetic_retinopathy_debrecen" / "messidor_features.arff"
    )
    matrix = np.asarray(records.tolist(), dtype=float)
    return (matrix[:, :-1], matrix[:, -1].astype(int))


def unit_artifact(ref: Mapping[str, Any]) -> dict[str, Any]:
    path = OUTER_ROOT / str(ref["dataset_id"]) / "units" / f"{ref['outer_unit_id']}.json"
    artifact = read_json(path)
    if artifact["unit_identity"]["unit_id"] != ref["outer_unit_id"]:
        raise AttackError("outer identity mismatch")
    verify(artifact, "formal_outer_unit_artifact_sha256")
    return artifact


def model_and_transform(
    artifact: Mapping[str, Any], raw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model = artifact["preopen_chain"]["model_artifact"]
    entries = {entry["name"]: entry for entry in model["serialized_model_state"]}
    weight = np.asarray(entries["output.weight"]["values"], dtype=float).reshape(-1)
    bias = float(np.asarray(entries["output.bias"]["values"], dtype=float).reshape(-1)[0])
    transformed = raw.copy().astype(float)
    parameters = model["refit_preprocessing_artifact"]["parameters"]
    for item in parameters:
        j = int(item["column_index"])
        missing = ~np.isfinite(transformed[:, j])
        if item["zero_as_missing"]:
            missing |= transformed[:, j] == 0
        transformed[missing, j] = float(item["median"])
        transformed[:, j] = (transformed[:, j] - float(item["post_imputation_mean"])) / float(
            item["standardization_scale"]
        )
    logits = transformed @ weight + bias
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    return (transformed, probabilities, weight, np.asarray([bias]))


def balanced_membership_sample(
    member_rows: np.ndarray, outer_rows: np.ndarray, labels: np.ndarray, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in (0, 1):
        a = member_rows[labels[member_rows] == label]
        b = outer_rows[labels[outer_rows] == label]
        n = min(len(a), len(b))
        if n < 4:
            raise AttackError("insufficient label-balanced membership sample")
        selected.extend(rng.choice(a, n, replace=False).tolist())
        selected.extend(rng.choice(b, n, replace=False).tolist())
    return np.asarray(selected, dtype=int)


def roc_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    auc = float(roc_auc_score(y, score))
    fpr, tpr, _ = roc_curve(y, score)
    return {
        "roc_auc": auc,
        "maximum_advantage": float(np.max(tpr - fpr)),
        "tpr_at_fpr_le_0_01": float(np.max(tpr[fpr <= 0.01], initial=0.0)),
        "tpr_at_fpr_le_0_05": float(np.max(tpr[fpr <= 0.05], initial=0.0)),
    }


def bootstrap_auc(
    y: np.ndarray, score: np.ndarray, replicates: int, seed: int
) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    n = len(y)
    for _ in range(replicates):
        index = rng.integers(0, n, n)
        if len(np.unique(y[index])) == 2:
            values.append(float(roc_auc_score(y[index], score[index])))
    if not values:
        raise AttackError("bootstrap produced no valid replicate")
    return (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)), len(values))


def membership_unit(spec: Mapping[str, Any]) -> dict[str, Any]:
    artifact = unit_artifact(spec)
    raw, labels = load_raw(str(spec["dataset_id"]))
    _, probabilities, _, _ = model_and_transform(artifact, raw)
    roles = artifact["preopen_chain"]["model_artifact"]["refit_preprocessing_artifact"]["roles"]
    member_rows = np.asarray(
        sorted(
            {
                int(row)
                for (name, role) in roles.items()
                if name.startswith("client_")
                for row in role["row_ids"]
            }
        ),
        dtype=int,
    )
    outer = artifact["outer_prediction_artifact"]
    outer_rows = np.asarray(outer["row_ids"], dtype=int)
    chosen = balanced_membership_sample(member_rows, outer_rows, labels, int(spec["attack_seed"]))
    membership = np.isin(chosen, member_rows).astype(int)
    y = labels[chosen]
    p = np.clip(probabilities[chosen], 1e-12, 1 - 1e-12)
    strata = membership * 2 + y
    cal, eva = train_test_split(
        np.arange(len(chosen)),
        test_size=1 - float(spec["calibration_fraction"]),
        random_state=int(spec["attack_seed"]),
        stratify=strata,
    )
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    confidence = np.where(y == 1, p, 1 - p)
    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    if spec["attack"] == "loss_threshold":
        all_score = -loss
    else:
        features = np.column_stack([confidence, entropy, loss])
        model = LogisticRegression(random_state=int(spec["attack_seed"]), solver="liblinear")
        model.fit(features[cal], membership[cal])
        all_score = model.predict_proba(features)[:, 1]
    fpr, tpr, thresholds = roc_curve(membership[cal], all_score[cal])
    threshold = float(thresholds[int(np.argmax(tpr - fpr))])
    metrics = roc_metrics(membership[eva], all_score[eva])
    low, high, valid = bootstrap_auc(
        membership[eva],
        all_score[eva],
        int(spec["bootstrap_replicates"]),
        int(spec["attack_seed"]) ^ 20903,
    )
    result = {
        "schema": _identity("membership_unit"),
        "status": "SEALED_COMPLETE",
        "spec": dict(spec),
        "attacker_visibility": "final_model_probability_and_true_label",
        "sampling": {
            "label_balanced": True,
            "calibration_evaluation_row_disjoint": True,
            "member_count": int(np.sum(membership[eva] == 1)),
            "nonmember_count": int(np.sum(membership[eva] == 0)),
        },
        "calibrated_threshold": threshold,
        "evaluation_accuracy": float(np.mean((all_score[eva] >= threshold) == membership[eva])),
        **metrics,
        "roc_auc_ci95_low": low,
        "roc_auc_ci95_high": high,
        "bootstrap_valid_replicates": valid,
    }
    result["membership_unit_sha256"] = sha(result)
    return result


def run_membership(_: argparse.Namespace) -> None:
    plan = load_plan()
    specs = plan["membership_inference"]["units"]
    root = RESULT_ROOT / "membership"
    rows: list[dict[str, Any]] = []
    for spec in specs:
        uid = sha({"domain": "rapid_membership_v1", "spec": spec})
        path = root / "units" / f"{uid}.json"
        if path.is_file():
            result = read_json(path)
            verify(result, "membership_unit_sha256")
        else:
            result = membership_unit(spec)
            atomic_json(path, result)
        rows.append(
            {
                **spec,
                **{
                    k: result[k]
                    for k in (
                        "roc_auc",
                        "maximum_advantage",
                        "tpr_at_fpr_le_0_01",
                        "tpr_at_fpr_le_0_05",
                        "roc_auc_ci95_low",
                        "roc_auc_ci95_high",
                        "evaluation_accuracy",
                    )
                },
            }
        )
    atomic_csv(root / "membership_summary.csv", rows)
    closure = {
        "schema": _identity("membership_closure"),
        "status": "MEMBERSHIP_COMPLETE_12_OF_12",
        "closed_at_utc": datetime.now(timezone.utc).isoformat(),
        "rapid_followup_plan_sha256": plan[PLAN_HASH_FIELD],
        "unit_count": len(rows),
        "summary_sha256": hashlib.sha256(
            (root / "membership_summary.csv").read_bytes()
        ).hexdigest(),
    }
    closure["closure_sha256"] = sha(closure)
    atomic_json(root / "membership_closure.json", closure)
    print(json.dumps({"status": closure["status"], "closure_sha256": closure["closure_sha256"]}))


def reconstruction_unit(spec: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    artifact = unit_artifact(spec)
    raw, labels = load_raw(str(spec["dataset_id"]))
    transformed, probabilities, weight, bias = model_and_transform(artifact, raw)
    outer_rows = np.asarray(artifact["outer_prediction_artifact"]["row_ids"], dtype=int)
    rng = np.random.default_rng(int(spec["attack_seed"]))
    n = min(int(spec["sample_count"]), len(outer_rows))
    target_rows = rng.choice(outer_rows, n, replace=False)
    target_p = probabilities[target_rows]
    low = np.nanmin(transformed, axis=0)
    high = np.nanmax(transformed, axis=0)
    span = np.maximum(high - low, 1e-09)
    best_x = np.zeros((n, transformed.shape[1]))
    best_error = np.full(n, np.inf)
    torch.manual_seed(int(spec["attack_seed"]))
    tw = torch.tensor(weight, dtype=torch.float64)
    tb = torch.tensor(float(bias[0]), dtype=torch.float64)
    tlow = torch.tensor(low)
    tspan = torch.tensor(span)
    tp = torch.tensor(target_p)
    for _ in range(int(spec["restart_count"])):
        z = torch.randn((n, transformed.shape[1]), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([z], lr=0.08)
        for _ in range(int(spec["optimizer_steps"])):
            candidate = tlow + tspan * torch.sigmoid(z)
            predicted = torch.sigmoid(candidate @ tw + tb)
            loss = (
                torch.mean((predicted - tp) ** 2)
                + 1e-05 * torch.mean((candidate - (tlow + tspan / 2)) / tspan) ** 2
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        candidate = (tlow + tspan * torch.sigmoid(z)).detach().numpy()
        error = np.abs(
            1 / (1 + np.exp(-np.clip(candidate @ weight + float(bias[0]), -40, 40))) - target_p
        )
        improve = error < best_error
        best_error[improve] = error[improve]
        best_x[improve] = candidate[improve]
    truth = transformed[target_rows]
    normalized_mae = float(np.mean(np.abs(best_x - truth) / span))
    correlation = float(np.corrcoef(best_x.ravel(), truth.ravel())[0, 1])
    parameters = artifact["preopen_chain"]["model_artifact"]["refit_preprocessing_artifact"][
        "parameters"
    ]
    recon_raw = best_x.copy()
    true_raw = raw[target_rows].copy()
    for item in parameters:
        j = int(item["column_index"])
        recon_raw[:, j] = best_x[:, j] * float(item["standardization_scale"]) + float(
            item["post_imputation_mean"]
        )
    discrete = [j for j in range(raw.shape[1]) if len(np.unique(raw[:, j])) <= 20]
    discrete_rate = (
        float(np.mean(np.rint(recon_raw[:, discrete]) == np.rint(true_raw[:, discrete])))
        if discrete
        else float("nan")
    )
    result = {
        "schema": _identity("reconstruction_unit"),
        "status": "SEALED_COMPLETE",
        "spec": dict(spec),
        "normalized_mae": normalized_mae,
        "pearson_correlation": correlation,
        "discrete_feature_recovery_rate": discrete_rate,
        "label_recovery_rate": float(np.mean((target_p >= 0.5).astype(int) == labels[target_rows])),
        "attack_failure_rate": float(np.mean(best_error > 0.001)),
        "mean_probability_match_error": float(np.mean(best_error)),
        "evaluated_sample_count": n,
        "discrete_feature_count": len(discrete),
        "visibility": spec["visibility"],
        "objective_frozen_before_attack_results": True,
    }
    result["reconstruction_unit_sha256"] = sha(result)
    return result


def run_reconstruction(_: argparse.Namespace) -> None:
    plan = load_plan()
    specs = plan["model_output_attribute_reconstruction"]["units"]
    root = RESULT_ROOT / "reconstruction"
    rows = []
    for spec in specs:
        uid = sha({"domain": "rapid_reconstruction_v1", "spec": spec})
        path = root / "units" / f"{uid}.json"
        if path.is_file():
            result = read_json(path)
            verify(result, "reconstruction_unit_sha256")
        else:
            result = reconstruction_unit(spec)
            atomic_json(path, result)
        rows.append(
            {
                **spec,
                **{
                    k: result[k]
                    for k in (
                        "normalized_mae",
                        "pearson_correlation",
                        "discrete_feature_recovery_rate",
                        "label_recovery_rate",
                        "attack_failure_rate",
                        "mean_probability_match_error",
                    )
                },
            }
        )
    atomic_csv(root / "reconstruction_summary.csv", rows)
    closure = {
        "schema": _identity("reconstruction_closure"),
        "status": "RECONSTRUCTION_COMPLETE_6_OF_6",
        "closed_at_utc": datetime.now(timezone.utc).isoformat(),
        "rapid_followup_plan_sha256": plan[PLAN_HASH_FIELD],
        "unit_count": len(rows),
        "summary_sha256": hashlib.sha256(
            (root / "reconstruction_summary.csv").read_bytes()
        ).hexdigest(),
    }
    closure["closure_sha256"] = sha(closure)
    atomic_json(root / "reconstruction_closure.json", closure)
    print(json.dumps({"status": closure["status"], "closure_sha256": closure["closure_sha256"]}))


def status(_: argparse.Namespace) -> None:
    plan = load_plan()
    output = {}
    for name, section, folder in (
        ("membership", "membership_inference", "membership"),
        ("reconstruction", "model_output_attribute_reconstruction", "reconstruction"),
    ):
        planned = plan[section]["planned_unit_count"]
        complete = len(list((RESULT_ROOT / folder / "units").glob("*.json")))
        output[name] = {
            "complete": complete,
            "planned": planned,
            "closed": (RESULT_ROOT / folder / f"{folder}_closure.json").is_file(),
        }
    print(json.dumps(output))


def self_test(_: argparse.Namespace) -> None:
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = roc_metrics(y, score)
    if metrics["roc_auc"] != 1.0:
        raise AttackError("ROC self-test failed")
    print(json.dumps({"status": "SELF_TEST_OK"}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status").set_defaults(func=status)
    sub.add_parser("self-test").set_defaults(func=self_test)
    sub.add_parser("run-membership").set_defaults(func=run_membership)
    sub.add_parser("run-reconstruction").set_defaults(func=run_reconstruction)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
