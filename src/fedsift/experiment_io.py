"""Locations and integrity checks for the final experiment distribution."""

from pathlib import Path, PurePosixPath
from functools import lru_cache
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def results_root() -> Path:
    base = (ROOT / "output").resolve()
    target = Path(os.environ.get("FEDSIFT_OUTPUT_ROOT", str(base / "reference"))).resolve()
    if not target.is_relative_to(base):
        raise ValueError("Experiment outputs must stay under the code/output directory")
    return target


def resolve_record_path(relative: str) -> Path:
    """Resolve an immutable record's logical path to the final file layout."""
    path = PurePosixPath(relative.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Recorded path is outside the experiment")
    mapping = _read(ROOT / "provenance/path_map.json")
    text = path.as_posix()
    for prefix, specification in mapping["prefixes"].items():
        if text.startswith(prefix):
            suffix = PurePosixPath(text[len(prefix) :])
            parts = [mapping["components"].get(part, part) for part in suffix.parts]
            base = results_root() if specification == "results" else ROOT
            result = base.joinpath(*parts)
            if not result.resolve().is_relative_to(base.resolve()):
                raise ValueError("Resolved record path escaped its root")
            return result
    result = ROOT.joinpath(*path.parts)
    if not result.resolve().is_relative_to(ROOT):
        raise ValueError("Recorded path escaped the project")
    return result


@lru_cache(maxsize=1)
def verify_release_sources() -> dict:
    """Check final source bytes independently of the original study identity."""
    manifest = _read(ROOT / "provenance/release_sources.json")
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if not path.resolve().is_relative_to(ROOT) or not path.is_file():
            raise RuntimeError("Release file is missing: " + entry["path"])
        with path.open("rb") as source:
            digest = hashlib.sha256()
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if path.stat().st_size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
            raise RuntimeError("Release file changed: " + entry["path"])
    return {"status": "PASS", "verified_files": len(manifest["files"])}


@lru_cache(maxsize=1)
def registered_study() -> dict:
    verify_release_sources()
    return _read(ROOT / "provenance/study_identity.json")
