"""Run each regression module in a fresh process to bound retained test state."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import os
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CountedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed = 0

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed += 1


def run_module(module, result_file):
    suite = unittest.defaultTestLoader.loadTestsFromName(module)
    result = unittest.TextTestRunner(verbosity=2, resultclass=CountedResult).run(suite)
    counts = {
        "total": result.testsRun,
        "passed": result.passed,
        "skipped": len(result.skipped),
        "failed": len(result.failures),
        "errors": len(result.errors),
        "expected_failures": len(result.expectedFailures),
        "unexpected_successes": len(result.unexpectedSuccesses),
    }
    Path(result_file).write_text(json.dumps(counts, indent=2), encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--module", action="append", dest="modules")
    parser.add_argument("--run-module", help=argparse.SUPPRESS)
    parser.add_argument("--result-file", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.run_module:
        if not args.result_file:
            parser.error("--run-module requires --result-file")
        return run_module(args.run_module, args.result_file)
    if not 1 <= args.jobs <= 4:
        parser.error("--jobs must be between 1 and 4")
    available = sorted("tests." + p.stem for p in (ROOT / "tests").glob("test_*.py"))
    modules = args.modules or available
    if not set(modules).issubset(available):
        parser.error("Unknown test module")
    destination = ROOT / "output" / "validation" / ("tests_" + time.strftime("%Y%m%d_%H%M%S"))
    destination.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT)

    def run(module):
        started = time.time()
        path = destination / (module + ".log")
        count_path = destination / (module + ".json")
        with path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                [sys.executable, "-B", str(Path(__file__).resolve()),
                 "--run-module", module, "--result-file", str(count_path)],
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        return {
            "module": module,
            "status": "PASS" if process.returncode == 0 else "FAIL",
            "exit_code": process.returncode,
            "seconds": time.time() - started,
            "log": str(path.relative_to(ROOT)),
            "counts": json.loads(count_path.read_text(encoding="utf-8")) if count_path.is_file() else None,
        }

    records = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run, module) for module in modules]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps(record), flush=True)
            (destination / "progress.json").write_text(
                json.dumps(records, indent=2), encoding="utf-8"
            )
    result = {
        "status": "PASS" if all(r["status"] == "PASS" for r in records) else "FAIL",
        "modules": len(records),
        "records": sorted(records, key=lambda r: r["module"]),
        "test_counts": {
            name: sum(r["counts"][name] for r in records if r["counts"] is not None)
            for name in ("total", "passed", "skipped", "failed", "errors", "expected_failures", "unexpected_successes")
        },
        "modules_without_test_counts": [r["module"] for r in records if r["counts"] is None],
    }
    (destination / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "modules": len(records),
                "test_counts": result["test_counts"],
                "result": str(destination / "result.json"),
            }
        ),
        flush=True,
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
