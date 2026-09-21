"""Execute and verify the final FedSift experiment without overwriting its reference results."""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "output/reference"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT)]
from fedsift.experiment_io import registered_study, verify_release_sources
from tools.validation_math import assess_training


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def report(name, result):
    directory = ROOT / "output/validation"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (time.strftime("%Y%m%d_%H%M%S") + "_" + name + ".json")
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {key: value for key, value in result.items() if key not in ("captured", "differences")}
    summary["report_path"] = path.relative_to(ROOT).as_posix()
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return result


def runtime_check():
    from fedsift.runtime_environment import RuntimeEnvironmentGuard, canonical_hash

    expected = read(ROOT / "environment/runtime_contract.json")
    guard = RuntimeEnvironmentGuard(expected, expected_sha256=canonical_hash(expected))
    result = guard.check("final_distribution_verification")
    prefix = os.environ.get("FEDSIFT_RUNTIME_PREFIX")
    if not prefix or not Path(sys.executable).resolve().is_relative_to(Path(prefix).resolve()):
        raise RuntimeError("Use run.sh or run.ps1 to select the packaged runtime")
    return {"guard": result, "python": sys.executable, "packaged_runtime_prefix": prefix}


def source_check():
    import search

    result = verify_release_sources()
    authorization = search._authorization()
    return {
        **result,
        "registered_implementation_sha256": registered_study()["implementation_sha256"],
        "search_authorization_sha256": authorization["launch_authorization_sha256"],
    }


def models_check():
    import numpy as np
    import pandas as pd
    from scipy.io import arff
    import outer_training

    units = read(ROOT / "config/experiment_plan.json")["units"]
    frames = {"pima": pd.read_csv(ROOT / "data/diabetes.csv").iloc[:, :-1].to_numpy(dtype=float)}
    values, _ = arff.loadarff(next((ROOT / "data").rglob("*.arff")))
    frames["retinopathy"] = pd.DataFrame(values).iloc[:, :-1].to_numpy(dtype=float)
    maximum = 0.0
    rows = 0
    for unit in units:
        artifact = read(
            REFERENCE / "main" / unit["dataset_id"] / "units" / (unit["unit_id"] + ".json")
        )
        outer_training.validate_outer_unit_artifact(artifact, expected_unit=unit)
        prediction = artifact["outer_prediction_artifact"]
        model = artifact["preopen_chain"]["model_artifact"]
        matrix = frames[unit["dataset_id"]][np.asarray(prediction["row_ids"], dtype=int)].copy()
        for parameter in model["refit_preprocessing_artifact"]["parameters"]:
            column = int(parameter["column_index"])
            missing = ~np.isfinite(matrix[:, column])
            if parameter["zero_as_missing"]:
                missing |= matrix[:, column] == 0
            matrix[missing, column] = float(parameter["median"])
            matrix[:, column] = (
                matrix[:, column] - float(parameter["post_imputation_mean"])
            ) / float(parameter["standardization_scale"])
        states = {state["name"]: state for state in model["serialized_model_state"]}
        if set(states) != {"output.weight", "output.bias"}:
            raise RuntimeError("Unexpected registered model family")
        weights = np.asarray(states["output.weight"]["values"][0])
        bias = float(states["output.bias"]["values"][0])
        probabilities = 1 / (1 + np.exp(-np.clip(matrix @ weights + bias, -40, 40)))
        difference = float(np.max(np.abs(probabilities - np.asarray(prediction["probabilities"]))))
        if difference > 1e-12:
            raise RuntimeError("Model predictions changed: " + unit["unit_id"])
        maximum = max(maximum, difference)
        rows += len(matrix)
    if len(units) != 70:
        raise RuntimeError("Expected 70 registered main experiment units")
    return {
        "models": len(units),
        "prediction_rows": rows,
        "maximum_absolute_difference": maximum,
        "tolerance": 1e-12,
    }


def verify(args):
    runtime = runtime_check()
    source = source_check()
    checked = 0
    if not args.models_only:
        manifest = read(ROOT / "MANIFEST.json")
        for entry in manifest["files"]:
            path = ROOT / entry["path"]
            if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
                raise RuntimeError("Distribution file changed: " + entry["path"])
            checked += 1
    report(
        "verify",
        {
            "status": "PASS",
            "distribution_files_verified": checked,
            "runtime": runtime,
            "source": source,
            "models": models_check(),
        },
    )


def train_smoke(_):
    runtime = runtime_check()
    source = source_check()
    import outer_training

    units = read(ROOT / "config/experiment_plan.json")["units"]
    checks = []
    for dataset in ("pima", "retinopathy"):
        unit = next(
            u
            for u in units
            if u["dataset_id"] == dataset and u["method_id"] == "fedsift" and u["outer_fold"] == 0
        )
        expected = read(REFERENCE / "main" / dataset / "units" / (unit["unit_id"] + ".json"))
        captured = {}
        original = outer_training.build_preopen_artifact_chain

        def capture(*args, **kwargs):
            chain = original(*args, **kwargs)
            captured.update(chain=chain, training=outer_training._training_evidence(args[1]))
            return chain

        started = time.time()
        outer_training.build_preopen_artifact_chain = capture
        try:
            actual = outer_training.execute_outer_unit(unit, open_outer=False)
        finally:
            outer_training.build_preopen_artifact_chain = original
        assessment = assess_training(expected, captured)
        result = {
            "dataset": dataset,
            "unit_id": unit["unit_id"],
            "seconds": time.time() - started,
            "outer_test_accessed": actual["outer_test_accessed"],
            **assessment,
        }
        report("training_" + dataset, {**result, "captured": captured})
        checks.append(result)
    passed = all(c["numerically_equivalent"] and not c["outer_test_accessed"] for c in checks)
    report(
        "training",
        {
            "status": "PASS" if passed else "FAIL",
            "runtime": runtime,
            "source": source,
            "checks": checks,
        },
    )
    if not passed:
        raise RuntimeError("Training comparison contains substantive differences")


def new_run(kind, copy_models=False):
    directory = ROOT / "output/runs" / (time.strftime("%Y%m%d_%H%M%S") + "_" + kind)
    directory.mkdir(parents=True, exist_ok=False)
    if kind not in ("search", "resource"):
        shutil.copytree(REFERENCE / "selection", directory / "selection")
    if copy_models:
        for name in ("main", "followup"):
            shutil.copytree(REFERENCE / name, directory / name)
    return directory


def execute(directory, script, *arguments):
    environment = os.environ.copy()
    environment["FEDSIFT_OUTPUT_ROOT"] = str(directory)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = (
        str(ROOT / "src") + os.pathsep + str(ROOT / "scripts") + os.pathsep + str(ROOT)
    )
    subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts" / script), *arguments],
        cwd=ROOT,
        env=environment,
        check=True,
    )


def compare_tables(directory):
    comparisons = []
    for group in ("main_summary", "followup"):
        for original in sorted((REFERENCE / group).rglob("*.csv")):
            relative = original.relative_to(REFERENCE)
            rebuilt = directory / relative
            if not rebuilt.is_file():
                raise RuntimeError("Missing rebuilt table: " + str(relative))
            identical = sha(original) == sha(rebuilt)
            maximum = 0.0
            with (
                original.open(encoding="utf-8-sig", newline="") as left,
                rebuilt.open(encoding="utf-8-sig", newline="") as right,
            ):
                reader_left, reader_right = csv.DictReader(left), csv.DictReader(right)
                if reader_left.fieldnames != reader_right.fieldnames:
                    raise RuntimeError("Table columns changed: " + str(relative))
                rows_left, rows_right = list(reader_left), list(reader_right)
            if len(rows_left) != len(rows_right):
                raise RuntimeError("Table row count changed: " + str(relative))
            for a, b in zip(rows_left, rows_right):
                for key, value in a.items():
                    if value == b[key]:
                        continue
                    try:
                        x, y = float(value), float(b[key])
                    except ValueError:
                        raise RuntimeError("Table text changed: " + str(relative))
                    if not math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12):
                        raise RuntimeError("Table value changed: " + str(relative))
                    maximum = max(maximum, abs(x - y))
            comparisons.append(
                {
                    "path": str(relative),
                    "byte_identical": identical,
                    "maximum_absolute_difference": maximum,
                }
            )
    return comparisons


def summarize(directory):
    execute(directory, "summarize_results.py", "finalize")
    execute(directory, "analyze_results.py")
    execute(directory, "summarize_privacy_resources.py", "finalize")
    execute(directory, "evaluate_attacks.py", "run-membership")
    execute(directory, "evaluate_attacks.py", "run-reconstruction")
    execute(directory, "evaluate_poisoning.py", "finalize")
    execute(directory, "plot_poisoning.py")


def rebuild(args):
    runtime_check()
    source_check()
    directory = new_run(args.command, copy_models=args.command == "rebuild")
    if args.command == "replay":
        for shard in ("0", "1"):
            execute(directory, "run_experiment.py", "worker", "--shard", shard)
        execute(directory, "summarize_results.py", "finalize")
        for dataset in ("pima", "retinopathy"):
            execute(directory, "evaluate_poisoning.py", "worker", "--dataset", dataset)
    summarize(directory)
    report(
        args.command,
        {
            "status": "PASS",
            "output_directory": str(directory),
            "table_comparison": compare_tables(directory),
            "retrained_main_units": 70 if args.command == "replay" else 0,
            "retrained_poisoning_units": 12 if args.command == "replay" else 0,
        },
    )


def analyze_results(_):
    runtime_check()
    source_check()
    directory = ROOT / "output/runs" / (time.strftime("%Y%m%d_%H%M%S") + "_analysis")
    directory.mkdir(parents=True, exist_ok=False)
    execute(directory, "analyze_results.py", "--predictions", str(REFERENCE / "main_summary/oof_predictions.csv"))
    execute(directory, "plot_poisoning.py", "--summary", str(REFERENCE / "followup/poisoning/poisoning_summary.csv"))
    report("analysis", {"status": "PASS", "output_directory": str(directory)})


def benchmark(_):
    runtime_check()
    source_check()
    directory = new_run("resource")
    execute(directory, "benchmark_resources.py")
    result = read(directory / "resource_benchmark/benchmark.json")
    if any(
        result[k]
        for k in ("predictions_accessed", "performance_metrics_accessed", "outer_test_accessed")
    ):
        raise RuntimeError("Resource measurement crossed its data boundary")
    report(
        "resource",
        {
            "status": "PASS",
            "output_directory": str(directory),
            "result_status": result["status"],
            "artifact_sha256": result["artifact"]["artifact_sha256"],
            "absolute_timing_and_memory_are_machine_dependent": True,
        },
    )


def search_all(_):
    runtime_check()
    verify_release_sources()
    directory = new_run("search")
    execute(directory, "search.py", "prepare")
    for dataset in ("pima", "retinopathy"):
        for shard in ("0", "1", "2", "3"):
            execute(directory, "search.py", "worker", "--dataset", dataset, "--shard", shard)
        execute(directory, "search.py", "finalize", "--dataset", dataset)
    report(
        "search",
        {
            "status": "PASS",
            "output_directory": str(directory),
            "reference_outputs_overwritten": False,
        },
    )


def audit_search(_):
    runtime_check()
    verify_release_sources()
    manifest = read(REFERENCE / "selection_evidence.manifest.json")
    path = ROOT / manifest["archive"]
    if sha(path) != manifest["sha256"]:
        raise RuntimeError("Selection evidence archive changed")
    with zipfile.ZipFile(path) as archive:
        if len(archive.infolist()) != len(manifest["files"]):
            raise RuntimeError("Selection archive inventory changed")
        for row in manifest["files"]:
            if hashlib.sha256(archive.read(row["path"])).hexdigest() != row["sha256"]:
                raise RuntimeError("Selection evidence member changed: " + row["path"])
    from verify_selection import verify_selection
    result = verify_selection(REFERENCE)
    report("selection_archive", {**result, "evidence_files_verified": len(manifest["files"])})


def tests(args):
    runtime_check()
    subprocess.run(
        [sys.executable, "-B", str(ROOT / "tools/run_tests.py"), "--jobs", str(args.jobs)],
        cwd=ROOT,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description="FedSift experiment and reproduction entry point")
    commands = parser.add_subparsers(dest="command", required=True)
    verification = commands.add_parser("verify")
    verification.add_argument("--models-only", action="store_true")
    verification.set_defaults(func=verify)
    commands.add_parser("train-smoke").set_defaults(func=train_smoke)
    for name in ("rebuild", "replay"):
        commands.add_parser(name).set_defaults(func=rebuild)
    commands.add_parser("benchmark").set_defaults(func=benchmark)
    commands.add_parser("analyze").set_defaults(func=analyze_results)
    commands.add_parser("search").set_defaults(func=search_all)
    commands.add_parser("audit-search").set_defaults(func=audit_search)
    regression = commands.add_parser("tests")
    regression.add_argument("--jobs", type=int, default=1)
    regression.set_defaults(func=tests)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
