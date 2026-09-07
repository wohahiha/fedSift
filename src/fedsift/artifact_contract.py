"""Registered wire-format identities and random seed namespaces.

These values belong to the recorded experiment. Keeping them separate from
Python names permits API cleanup without changing hashes or random streams.
"""

from pathlib import Path
import json

_PATH = Path(__file__).resolve().parents[2] / "provenance" / "artifact_identities.json"
_IDENTITIES = json.loads(_PATH.read_text(encoding="utf-8"))


def identity(name: str) -> str:
    return _IDENTITIES[name]
