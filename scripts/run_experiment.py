"""Execute the registered 70-unit FedSift experiment in a selected output directory."""

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
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
import outer_training as outer

TOP = results_root()
PREP_ROOT = ROOT / "config"
PLAN_PATH = PREP_ROOT / "experiment_plan.json"
AUTH_PATH = PREP_ROOT / "experiment_authorization.json"
RUN_ROOT = TOP / "main"
DATASETS = ("pima", "retinopathy")
SHARDS = 2


class ExperimentError(RuntimeError):
    pass


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise ExperimentError(f"cannot read JSON: {path}") from exc


def _verify_hash(value: Mapping[str, Any], field: str) -> None:
    if not isinstance(value, Mapping):
        raise ExperimentError("artifact is not an object")
    payload = copy.deepcopy(dict(value))
    stored = payload.pop(field, None)
    if stored != _canonical_sha256(payload):
        raise ExperimentError(f"artifact hash differs: {field}")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def runner_contract() -> dict[str, Any]:
    return {
        "schema": _identity("outer_runner_contract"),
        "implementation_status": "REGISTERED_EXPERIMENT_EXECUTABLE",
        "outer_worker_executable": AUTH_PATH.is_file(),
        "outer_test_open_possible": AUTH_PATH.is_file(),
        "planned_unit_count": 70,
        "worker_shards": SHARDS,
    }


def _plan() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan = _read_json(PLAN_PATH)
    _verify_hash(plan, _identity("outer_plan_hash_field"))
    units = plan.get("units")
    if not isinstance(units, list) or len(units) != 70:
        raise ExperimentError("registered experiment unit catalog differs")
    return (plan, units)


def _authorization() -> dict[str, Any]:
    value = _read_json(AUTH_PATH)
    _verify_hash(value, _identity("outer_authorization_hash_field"))
    plan, _ = _plan()
    expected = {
        "schema": _identity("outer_launch_authorization"),
        "status": _identity("outer_authorization_status"),
        "runner_sha256": registered_study()["runner_sha256"]["run_experiment"],
        _identity("outer_plan_hash_field"): plan[_identity("outer_plan_hash_field")],
        "worker_shards": SHARDS,
        "planned_unit_count": 70,
        "user_authorized_scope_reduction": True,
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise ExperimentError(f"experiment authorization differs: {field}")
    return value


def worker(arguments: argparse.Namespace) -> None:
    authorization = _authorization()
    _, units = _plan()
    assigned = [unit for (index, unit) in enumerate(units) if index % SHARDS == arguments.shard]
    progress_path = RUN_ROOT / f"worker_{arguments.shard}_progress.json"
    executed = reused = 0
    for position, unit in enumerate(assigned, start=1):
        dataset = str(unit["dataset_id"])
        unit_id = str(unit["unit_id"])
        root = RUN_ROOT / dataset
        complete_path = root / "units" / f"{unit_id}.json"
        failure_path = root / "failures" / f"{unit_id}.json"
        journal_path = root / "access_journal" / f"{unit_id}.json"
        if failure_path.exists():
            raise ExperimentError(f"recorded experiment failure requires review: {unit_id}")
        if complete_path.exists():
            outer.validate_outer_unit_artifact(_read_json(complete_path), expected_unit=unit)
            reused += 1
            state = "reused_complete_unit"
        elif journal_path.exists():
            raise ExperimentError(f"interrupted experiment unit has an outer access journal: {unit_id}")
        else:
            try:
                artifact = outer.execute_outer_unit(
                    unit, open_outer=True, access_journal_path=journal_path
                )
                outer.validate_outer_unit_artifact(artifact, expected_unit=unit)
                _atomic_json(complete_path, artifact)
                if journal_path.exists():
                    journal_path.unlink()
                executed += 1
                state = "executed_outer_once_and_sealed"
            except BaseException as error:
                journal = _read_json(journal_path) if journal_path.exists() else None
                receipt = journal.get("access_receipt") if isinstance(journal, Mapping) else None
                gate = journal.get("gate_manifest") if isinstance(journal, Mapping) else None
                failure = outer.build_outer_failure_artifact(
                    unit,
                    error,
                    outer_test_accessed=journal is not None,
                    gate_sha256=gate.get("gate_sha256") if isinstance(gate, Mapping) else None,
                    access_receipt_sha256=(
                        receipt.get("access_receipt_sha256")
                        if isinstance(receipt, Mapping)
                        else None
                    ),
                )
                _atomic_json(failure_path, failure)
                raise
        progress = {
            "schema": _identity("outer_worker_progress"),
            "status": "COMPLETE" if position == len(assigned) else "RUNNING",
            "shard": arguments.shard,
            "assigned_unit_count": len(assigned),
            "last_position": position,
            "completed_unit_count": executed + reused,
            "newly_executed_count": executed,
            "reused_count": reused,
            "last_unit_id": unit_id,
            "last_unit_state": state,
            _identity("outer_authorization_hash_field"): authorization[
                _identity("outer_authorization_hash_field")
            ],
        }
        _atomic_json(progress_path, progress)
        print(json.dumps(progress, separators=(",", ":")), flush=True)


def status(_: argparse.Namespace) -> None:
    value = runner_contract()
    value["plan_exists"] = PLAN_PATH.exists()
    value["authorization_exists"] = AUTH_PATH.exists()
    value["datasets"] = {}
    for dataset in DATASETS:
        root = RUN_ROOT / dataset
        value["datasets"][dataset] = {
            "completed": len(list((root / "units").glob("*.json"))),
            "planned": 35,
            "failures": len(list((root / "failures").glob("*.json"))),
            "access_journals": len(list((root / "access_journal").glob("*.json"))),
        }
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--shard", type=int, choices=range(SHARDS), required=True)
    worker_parser.set_defaults(func=worker)
    sub.add_parser("status").set_defaults(func=status)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
