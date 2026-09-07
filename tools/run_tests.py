"""Run each regression module in a fresh process to bound retained test state."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--module", action="append", dest="modules")
    args = parser.parse_args()
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
        with path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                [sys.executable, "-B", "-m", "unittest", "-v", module],
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
    }
    (destination / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "modules": len(records),
                "result": str(destination / "result.json"),
            }
        ),
        flush=True,
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
