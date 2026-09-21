"""Validate and summarize the declared privacy and resource matrices."""

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

ROOT = Path(__file__).resolve().parents[1]
TOP = results_root()
PLAN_PATH = ROOT / "config" / "evaluation_plan.json"
RUN_ROOT = TOP / "main"
RESULT_ROOT = TOP / "followup" / "privacy_resource"
HASH_FIELD = _identity("followup_plan_hash_field")


class ClosureError(RuntimeError):
    pass


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify_hash(value: Mapping[str, Any], field: str) -> None:
    payload = copy.deepcopy(dict(value))
    stored = payload.pop(field, None)
    if stored != canonical_sha256(payload):
        raise ClosureError(f"self-hash mismatch: {field}")


def atomic_json(path: Path, value: Any) -> None:
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


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ClosureError("refusing empty CSV")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_plan() -> dict[str, Any]:
    plan = read_json(PLAN_PATH)
    verify_hash(plan, HASH_FIELD)
    if plan["formal_privacy_accounting"]["planned_unit_count"] != 60:
        raise ClosureError("privacy denominator differs from 60")
    if plan["resource_and_communication"]["planned_unit_count"] != 70:
        raise ClosureError("resource denominator differs from 70")
    return plan


def unit_path(ref: Mapping[str, Any]) -> Path:
    return RUN_ROOT / str(ref["dataset_id"]) / "units" / f"{ref['outer_unit_id']}.json"


def validate_unit(path: Path, ref: Mapping[str, Any]) -> dict[str, Any]:
    artifact = read_json(path)
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import outer_training as formal_outer

    expected = {
        "dataset_id": ref["dataset_id"],
        "outer_fold": ref["outer_fold"],
        "eval_seed": ref["eval_seed"],
        "method_id": ref["method_id"],
        "unit_id": ref["outer_unit_id"],
        "outer_repeat": 0,
        "arm_type": (
            "matched_fedsift_ablation"
            if ref["method_id"]
            in {"fedsift_public_argmin_rule", "fedsift_uniform_schedule", "fedsift_without_sift"}
            else "main_method"
        ),
    }
    formal_outer.validate_outer_unit_artifact(artifact, expected_unit=expected)
    return artifact


def coverage(refs: list[dict[str, Any]]) -> int:
    return sum((unit_path(ref).is_file() for ref in refs))


def status(_: argparse.Namespace) -> None:
    plan = load_plan()
    privacy = plan["formal_privacy_accounting"]["units"]
    resources = plan["resource_and_communication"]["units"]
    print(
        json.dumps(
            {
                "status": "RUNNING" if coverage(resources) < 70 else "READY_TO_CLOSE",
                "privacy_complete": coverage(privacy),
                "privacy_planned": 60,
                "resource_complete": coverage(resources),
                "resource_planned": 70,
            },
            ensure_ascii=False,
        )
    )


def finalize(_: argparse.Namespace) -> None:
    plan = load_plan()
    privacy_refs = plan["formal_privacy_accounting"]["units"]
    resource_refs = plan["resource_and_communication"]["units"]
    if coverage(privacy_refs) != 60 or coverage(resource_refs) != 70:
        raise ClosureError("privacy/resource matrices are incomplete")
    failures = [
        p
        for dataset in ("pima", "retinopathy")
        for p in (RUN_ROOT / dataset / "failures").glob("*.json")
    ]
    if failures:
        raise ClosureError(f"outer failures present: {len(failures)}")
    cache: dict[str, dict[str, Any]] = {}
    for ref in resource_refs:
        cache[ref["outer_unit_id"]] = validate_unit(unit_path(ref), ref)
    privacy_rows: list[dict[str, Any]] = []
    for ref in privacy_refs:
        evidence = cache[ref["outer_unit_id"]]["training_evidence"]["privacy_evidence"]
        epsilon = float(evidence["runtime_parallel_epsilon"])
        target = float(evidence["target_epsilon"])
        delta = float(evidence["runtime_parallel_delta"])
        if (
            not all((math.isfinite(x) for x in (epsilon, target, delta)))
            or epsilon > target + 1e-09
        ):
            raise ClosureError(f"invalid privacy accounting: {ref['outer_unit_id']}")
        privacy_rows.append(
            {
                **ref,
                "privacy_status": evidence["status"],
                "target_epsilon": target,
                "target_delta": delta,
                "runtime_parallel_epsilon": epsilon,
                "static_schedule_epsilon_prv_upper": evidence["static_schedule_epsilon_prv_upper"],
                "composition_rule": evidence["parallel_composition_rule"],
                "secure_aggregation_claim": evidence["secure_aggregation_claim"],
                "client_level_dp_claim": evidence["client_level_dp_claim"],
            }
        )
    resource_rows: list[dict[str, Any]] = []
    for ref in resource_refs:
        totals = cache[ref["outer_unit_id"]]["training_evidence"]["resource_summary"]["totals"]
        if any((float(value) < 0 or not math.isfinite(float(value)) for value in totals.values())):
            raise ClosureError(f"invalid resource total: {ref['outer_unit_id']}")
        resource_rows.append({**ref, **totals})
    atomic_csv(RESULT_ROOT / "privacy_accounting.csv", privacy_rows)
    atomic_csv(RESULT_ROOT / "resource_and_communication.csv", resource_rows)
    outputs = {
        p.name: {"sha256": file_sha256(p), "bytes": p.stat().st_size}
        for p in sorted(RESULT_ROOT.glob("*.csv"))
    }
    closure = {
        "schema": _identity("privacy_resource_closure"),
        "status": "PRIVACY_60_OF_60_AND_RESOURCE_70_OF_70_COMPLETE",
        "closed_at_utc": datetime.now(timezone.utc).isoformat(),
        _identity("followup_plan_hash_field"): plan[HASH_FIELD],
        "privacy_unit_count": len(privacy_rows),
        "resource_unit_count": len(resource_rows),
        "outer_quality_metrics_consumed": False,
        "outputs": outputs,
    }
    closure["closure_sha256"] = canonical_sha256(closure)
    atomic_json(RESULT_ROOT / "privacy_resource_closure.json", closure)
    print(json.dumps({"status": closure["status"], "closure_sha256": closure["closure_sha256"]}))


def self_test(_: argparse.Namespace) -> None:
    payload = {"a": 1, "b": [2, 3]}
    wrapped = {**payload, "sha": canonical_sha256(payload)}
    verify_hash(wrapped, "sha")
    print(json.dumps({"status": "SELF_TEST_OK_RESULT_BLIND"}))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status").set_defaults(func=status)
    sub.add_parser("finalize").set_defaults(func=finalize)
    sub.add_parser("self-test").set_defaults(func=self_test)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
