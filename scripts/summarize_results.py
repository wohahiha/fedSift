"""Validate and aggregate the 70-model FedSift outer experiment."""

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
import statistics
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
TOP = results_root()
PLAN_PATH = ROOT / "config" / "experiment_plan.json"
RUN_ROOT = TOP / "main"
RESULT_ROOT = TOP / "main_summary"
PLAN_HASH_FIELD = _identity("outer_plan_hash_field")
METHODS = (
    "fedavg_nonprivate",
    "dp_fedavg",
    "dp_fedadam",
    "fedsift",
    "fedsift_public_argmin_rule",
    "fedsift_uniform_schedule",
    "fedsift_without_sift",
)
DATASETS = ("pima", "retinopathy")
HIGHER_IS_BETTER = {
    "average_precision",
    "auroc",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1",
    "balanced_accuracy",
    "mcc",
}
LOWER_IS_BETTER = {"brier_score", "log_loss"}
METRICS = tuple(sorted(HIGHER_IS_BETTER | LOWER_IS_BETTER))


class ResultSummaryError(RuntimeError):
    pass


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise ResultSummaryError(f"cannot read JSON: {path}") from exc


def _verify_hash(value: Mapping[str, Any], field: str) -> None:
    payload = copy.deepcopy(dict(value))
    stored = payload.pop(field, None)
    if stored != _canonical_sha256(payload):
        raise ResultSummaryError(f"artifact hash differs: {field}")


def _artifact(payload: dict[str, Any], field: str) -> dict[str, Any]:
    value = copy.deepcopy(payload)
    value[field] = _canonical_sha256(value)
    return value


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _plan() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan = _read_json(PLAN_PATH)
    _verify_hash(plan, PLAN_HASH_FIELD)
    units = plan.get("units")
    if not isinstance(units, list) or len(units) != 70:
        raise ResultSummaryError("experiment plan must contain exactly 70 units")
    return (plan, units)


def _failure_paths() -> list[Path]:
    paths: list[Path] = []
    for dataset in DATASETS:
        directory = RUN_ROOT / dataset / "failures"
        if directory.is_dir():
            paths.extend(sorted(directory.glob("*.json")))
    return paths


def _validate_unit(artifact: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import outer_training as formal_outer

    formal_outer.validate_outer_unit_artifact(artifact, expected_unit=expected)


def _mean_sd_ci(
    values: Iterable[float],
) -> tuple[int, float, float | None, float | None, float | None]:
    numbers = [float(value) for value in values]
    if not numbers:
        raise ResultSummaryError("cannot summarize an empty numeric series")
    mean = statistics.fmean(numbers)
    if len(numbers) == 1:
        return (1, mean, None, None, None)
    sd = statistics.stdev(numbers)
    t_critical = 2.7764451051977987 if len(numbers) == 5 else 1.959963984540054
    half_width = t_critical * sd / math.sqrt(len(numbers))
    return (len(numbers), mean, sd, mean - half_width, mean + half_width)


def _unit_rows(
    units: list[dict[str, Any]], artifacts: dict[str, Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    privacy_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    for unit in units:
        artifact = artifacts[str(unit["unit_id"])]
        report = artifact["evaluation_report"]
        metrics = dict(report["probability_metrics"])
        metrics.update(report["threshold_metrics"])
        metric_rows.append(
            {
                "dataset": unit["dataset_id"],
                "outer_repeat": unit["outer_repeat"],
                "outer_fold": unit["outer_fold"],
                "eval_seed": unit["eval_seed"],
                "method": unit["method_id"],
                "unit_id": unit["unit_id"],
                **{name: metrics.get(name) for name in METRICS},
            }
        )
        prediction = artifact["outer_prediction_artifact"]
        row_ids = prediction["row_ids"]
        labels = prediction["labels"]
        probabilities = prediction["probabilities"]
        if not len(row_ids) == len(labels) == len(probabilities):
            raise ResultSummaryError("outer prediction columns have unequal lengths")
        for row_id, label, probability in zip(row_ids, labels, probabilities):
            prediction_rows.append(
                {
                    "dataset": unit["dataset_id"],
                    "outer_fold": unit["outer_fold"],
                    "eval_seed": unit["eval_seed"],
                    "method": unit["method_id"],
                    "unit_id": unit["unit_id"],
                    "row_id": row_id,
                    "label": label,
                    "probability": probability,
                }
            )
        privacy = artifact["training_evidence"]["privacy_evidence"]
        privacy_rows.append(
            {
                "dataset": unit["dataset_id"],
                "outer_fold": unit["outer_fold"],
                "eval_seed": unit["eval_seed"],
                "method": unit["method_id"],
                "unit_id": unit["unit_id"],
                "privacy_status": privacy.get("status"),
                "target_epsilon": privacy.get("target_epsilon"),
                "target_delta": privacy.get("target_delta"),
                "static_schedule_epsilon_prv_upper": privacy.get(
                    "static_schedule_epsilon_prv_upper"
                ),
                "runtime_parallel_epsilon": privacy.get("runtime_parallel_epsilon"),
                "runtime_parallel_delta": privacy.get("runtime_parallel_delta"),
                "composition_rule": privacy.get("parallel_composition_rule"),
                "secure_aggregation_claim": privacy.get("secure_aggregation_claim"),
                "client_level_dp_claim": privacy.get("client_level_dp_claim"),
            }
        )
        totals = artifact["training_evidence"]["resource_summary"]["totals"]
        resource_rows.append(
            {
                "dataset": unit["dataset_id"],
                "outer_fold": unit["outer_fold"],
                "eval_seed": unit["eval_seed"],
                "method": unit["method_id"],
                "unit_id": unit["unit_id"],
                **{str(key): value for (key, value) in totals.items()},
            }
        )
    return (metric_rows, prediction_rows, privacy_rows, resource_rows)


def _summary_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped[str(row["dataset"]), str(row["method"])].append(row)
    output: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for method in METHODS:
            rows = sorted(grouped[dataset, method], key=lambda row: int(row["outer_fold"]))
            if len(rows) != 5:
                raise ResultSummaryError(f"expected five folds for {dataset}/{method}")
            for metric in METRICS:
                count, mean, sd, low, high = _mean_sd_ci((float(row[metric]) for row in rows))
                output.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "metric": metric,
                        "direction": "higher" if metric in HIGHER_IS_BETTER else "lower",
                        "fold_count": count,
                        "mean": mean,
                        "sd": sd,
                        "ci95_low_t": low,
                        "ci95_high_t": high,
                    }
                )
    return output


def _paired_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (str(row["dataset"]), int(row["outer_fold"]), str(row["method"])): row
        for row in metric_rows
    }
    output: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for comparator in METHODS:
            if comparator == "fedsift":
                continue
            for metric in METRICS:
                raw: list[float] = []
                benefit: list[float] = []
                for fold in range(5):
                    fedsift = float(index[dataset, fold, "fedsift"][metric])
                    other = float(index[dataset, fold, comparator][metric])
                    difference = fedsift - other
                    raw.append(difference)
                    benefit.append(difference if metric in HIGHER_IS_BETTER else -difference)
                count, mean, sd, low, high = _mean_sd_ci(raw)
                _, bmean, bsd, blow, bhigh = _mean_sd_ci(benefit)
                output.append(
                    {
                        "dataset": dataset,
                        "comparator": comparator,
                        "metric": metric,
                        "fold_count": count,
                        "fedsift_minus_comparator_mean": mean,
                        "fedsift_minus_comparator_sd": sd,
                        "fedsift_minus_comparator_ci95_low_t": low,
                        "fedsift_minus_comparator_ci95_high_t": high,
                        "benefit_oriented_mean": bmean,
                        "benefit_oriented_sd": bsd,
                        "benefit_oriented_ci95_low_t": blow,
                        "benefit_oriented_ci95_high_t": bhigh,
                    }
                )
    return output


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    return names


def status(_: argparse.Namespace) -> None:
    _, units = _plan()
    complete = sum(
        (
            (RUN_ROOT / str(unit["dataset_id"]) / "units" / f"{unit['unit_id']}.json").is_file()
            for unit in units
        )
    )
    failures = _failure_paths()
    print(
        json.dumps(
            {
                "status": "READY_TO_FINALIZE" if complete == 70 and (not failures) else "RUNNING",
                "complete_units": complete,
                "planned_units": 70,
                "failure_artifacts": len(failures),
                "closure_exists": (RESULT_ROOT / "experiment_closure.json").is_file(),
            },
            ensure_ascii=False,
        )
    )


def self_test(_: argparse.Namespace) -> None:
    sample = [1.0, 2.0, 3.0, 4.0, 5.0]
    count, mean, sd, low, high = _mean_sd_ci(sample)
    if count != 5 or mean != 3.0 or sd is None or (low is None) or (high is None):
        raise ResultSummaryError("summary self-test failed")
    if not low < mean < high:
        raise ResultSummaryError("summary interval ordering failed")
    plan, units = _plan()
    if plan.get("outer_prediction_or_quality_metrics_read_for_selection") is not False:
        raise ResultSummaryError("experiment plan result-blind marker differs")
    if len({str(unit["unit_id"]) for unit in units}) != 70:
        raise ResultSummaryError("experiment unit identity uniqueness failed")
    print(json.dumps({"status": "SELF_TEST_OK", "planned_units": 70}))


def finalize(_: argparse.Namespace) -> None:
    plan, units = _plan()
    failures = _failure_paths()
    if failures:
        raise ResultSummaryError(f"closure refused because {len(failures)} failure artifacts exist")
    artifacts: dict[str, Mapping[str, Any]] = {}
    for unit in units:
        path = RUN_ROOT / str(unit["dataset_id"]) / "units" / f"{unit['unit_id']}.json"
        if not path.is_file():
            raise ResultSummaryError(f"closure refused; missing unit {unit['unit_id']}")
        artifact = _read_json(path)
        _validate_unit(artifact, unit)
        artifacts[str(unit["unit_id"])] = artifact
    metric_rows, prediction_rows, privacy_rows, resource_rows = _unit_rows(units, artifacts)
    summary_rows = _summary_rows(metric_rows)
    paired_rows = _paired_rows(metric_rows)
    files = {
        "unit_metrics.csv": metric_rows,
        "oof_predictions.csv": prediction_rows,
        "method_summary.csv": summary_rows,
        "paired_fedsift_differences.csv": paired_rows,
        "privacy_summary.csv": privacy_rows,
        "resource_summary.csv": resource_rows,
    }
    for name, rows in files.items():
        _write_csv(RESULT_ROOT / name, rows, _fieldnames(rows))
    catalog = []
    for path in sorted(RESULT_ROOT.glob("*.csv")):
        catalog.append(
            {"path": path.name, "bytes": path.stat().st_size, "sha256": _file_sha256(path)}
        )
    closure = _artifact(
        {
            "schema": _identity("outer_closure"),
            "status": "OUTER_COMPLETE_70_OF_70",
            "closed_at_utc": datetime.now(timezone.utc).isoformat(),
            _identity("outer_plan_hash_field"): plan[PLAN_HASH_FIELD],
            "finalizer_sha256": _file_sha256(Path(__file__)),
            "planned_unit_count": 70,
            "completed_unit_count": 70,
            "failed_unit_count": 0,
            "datasets": {dataset: 35 for dataset in DATASETS},
            "methods": list(METHODS),
            "outer_folds": list(range(5)),
            "unit_artifact_catalog_sha256": _canonical_sha256(
                [
                    {
                        "unit_id": unit["unit_id"],
                        "artifact_sha256": artifacts[str(unit["unit_id"])][
                            "formal_outer_unit_artifact_sha256"
                        ],
                    }
                    for unit in units
                ]
            ),
            "result_file_catalog": catalog,
            "result_file_catalog_sha256": _canonical_sha256(catalog),
        },
        _identity("outer_closure_hash_field"),
    )
    _atomic_json(RESULT_ROOT / "experiment_closure.json", closure)
    print(json.dumps(closure, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status").set_defaults(func=status)
    subparsers.add_parser("self-test").set_defaults(func=self_test)
    subparsers.add_parser("finalize").set_defaults(func=finalize)
    arguments = parser.parse_args()
    arguments.func(arguments)


if __name__ == "__main__":
    main()
