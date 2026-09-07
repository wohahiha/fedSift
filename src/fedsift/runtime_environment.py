"""Check a separately committed runtime inside the process doing the work.

Snapshots are proposals, not evidence of scientific protocol approval. These
checks establish reproducibility of the measured process, not host isolation
or protection against a malicious operator.
"""

from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import hashlib
import hmac
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from collections.abc import Mapping
import torch

ENVIRONMENT_KEYS = (
    "BLIS_NUM_THREADS",
    "CUDA_VISIBLE_DEVICES",
    "MKL_DYNAMIC",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_DYNAMIC",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONHASHSEED",
    "VECLIB_MAXIMUM_THREADS",
)
PACKAGES = ("numpy", "scipy", "scikit-learn", "opacus", "torch", "pandas")
SCHEMA = _identity("in_process_environment")


class RuntimeEnvironmentError(RuntimeError):
    pass


def canonical_hash(value: object) -> str:
    data = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def observe_environment() -> dict[str, object]:
    """Read actual settings without changing threads, affinity, or packages."""
    if sys.platform != "linux" or not hasattr(os, "sched_getaffinity"):
        raise RuntimeEnvironmentError("registered resource execution requires Linux")
    cpu_lines = Path("/proc/cpuinfo").read_text().splitlines()
    cpu_model = next(
        (line.split(":", 1)[1].strip() for line in cpu_lines if line.startswith("model name")), None
    )
    if cpu_model is None or not Path("/proc/self/status").is_file():
        raise RuntimeEnvironmentError("Linux CPU and process observations unavailable")
    return {
        "schema": SCHEMA,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "process_nice": os.getpriority(os.PRIO_PROCESS, 0),
        "bytecode_write_disabled": sys.dont_write_bytecode,
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        "torch_build": str(torch.__version__),
        "torch_config_sha256": hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "torch_cuda_available": torch.cuda.is_available(),
        "environment": {name: os.environ.get(name) for name in ENVIRONMENT_KEYS},
    }


class RuntimeEnvironmentGuard:

    def __init__(self, expected: Mapping[str, object], *, expected_sha256: str):
        self.expected = json.loads(json.dumps(dict(expected)))
        if self.expected.get("schema") != SCHEMA or not hmac.compare_digest(
            canonical_hash(self.expected), expected_sha256
        ):
            raise RuntimeEnvironmentError("environment manifest commitment differs")
        self.expected_sha256 = expected_sha256
        self.check_count = 0

    def check(self, phase: str) -> dict[str, object]:
        observed = observe_environment()
        if set(observed) != set(self.expected):
            raise RuntimeEnvironmentError(f"runtime environment fields differ at {phase}")
        differing = sorted(
            (
                key
                for key in set(observed) | set(self.expected)
                if observed.get(key) != self.expected.get(key)
            )
        )
        if differing:
            raise RuntimeEnvironmentError(
                f"runtime environment drift at {phase}: {', '.join(differing)}"
            )
        self.check_count += 1
        return {
            "phase": phase,
            "environment_sha256": self.expected_sha256,
            "check_index": self.check_count,
            "status": "match",
        }

    def measurement_backend(self, backend):
        """Check outside the timing interval, before and after every unit."""

        def guarded(operation, context):
            self.check("before_measured_unit")
            result = backend(operation, context)
            self.check("after_measured_unit")
            return result

        return guarded
