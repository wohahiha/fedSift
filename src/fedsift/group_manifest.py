"""Label-independent group construction for result-blind split planning.

The module deliberately has no dependency on pandas or scikit-learn.  It reads
the two local tabular sources with the Python standard library and hashes only
raw predictors (or an explicitly supplied subject identifier).  Labels are
counted for stratification audits but never enter a group identifier.
"""

from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import csv
import hashlib
import hmac
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class GroupManifestError(RuntimeError):
    """Raised when source rows cannot produce an auditable group manifest."""


_CANONICAL_NAN = bytes.fromhex("7ff8000000000000")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def group_manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    """Hash every manifest field except the fingerprint itself."""
    payload = {
        str(key): value for (key, value) in manifest.items() if key != "group_manifest_sha256"
    }
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _parse_numeric(value: str) -> float:
    token = value.strip()
    if token in {"", "?", "NA", "NaN", "nan", "NULL", "null"}:
        return float("nan")
    number = float(token)
    if math.isinf(number):
        raise GroupManifestError("infinite source value is not supported")
    return number


def canonical_float64_bytes(value: float) -> bytes:
    """Return a platform-independent predictor encoding.

    All NaNs share one quiet-NaN payload and signed zero is normalized.  Finite
    values are encoded as big-endian IEEE-754 binary64.
    """
    number = float(value)
    if math.isnan(number):
        return _CANONICAL_NAN
    if math.isinf(number):
        raise GroupManifestError("infinite source value is not supported")
    if number == 0.0:
        number = 0.0
    return struct.pack(">d", number)


def _feature_group_id(features: Sequence[float]) -> str:
    digest = hashlib.sha256(_identity("exact_feature_group_domain").encode("utf-8"))
    digest.update(struct.pack(">I", len(features)))
    for value in features:
        digest.update(canonical_float64_bytes(float(value)))
    return digest.hexdigest()


def _subject_group_id(subject_id: str, salt: str) -> str:
    if not salt:
        raise GroupManifestError("subject-id grouping requires a non-empty salt")
    digest = hashlib.sha256(_identity("subject_group_domain").encode("utf-8"))
    digest.update(salt.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(str(subject_id).encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class DatasetRows:
    dataset: str
    source_path: str
    source_sha256: str
    feature_names: tuple[str, ...]
    features: tuple[tuple[float, ...], ...]
    labels: tuple[int, ...]
    row_ids: tuple[int, ...]
    target_name: str = ""
    predictor_allowlist_explicit: bool = False

    def __post_init__(self) -> None:
        n = len(self.row_ids)
        if len(self.features) != n or len(self.labels) != n:
            raise GroupManifestError("dataset row arrays have inconsistent lengths")
        if self.row_ids != tuple(range(n)):
            raise GroupManifestError("row ids must be contiguous and zero based")
        if not self.feature_names:
            raise GroupManifestError("dataset has no predictors")
        if any((len(row) != len(self.feature_names) for row in self.features)):
            raise GroupManifestError("feature row width mismatch")
        if any((label not in (0, 1) for label in self.labels)):
            raise GroupManifestError("target must be binary")
        if self.target_name and self.target_name in self.feature_names:
            raise GroupManifestError("declared target appears in predictor names")


@dataclass(frozen=True)
class GroupRecord:
    group_id: str
    row_ids: tuple[int, ...]
    n: int
    positive: int
    negative: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "row_ids": list(self.row_ids),
            "n": self.n,
            "positive": self.positive,
            "negative": self.negative,
        }


def load_csv_rows(
    path: Path | str, *, dataset: str, target: str, predictors: Sequence[str] | None = None
) -> DatasetRows:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or target not in reader.fieldnames:
            raise GroupManifestError("CSV target column is missing")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise GroupManifestError("CSV column names must be unique")
        feature_names = (
            tuple((str(name) for name in predictors))
            if predictors is not None
            else tuple((name for name in reader.fieldnames if name != target))
        )
        if (
            not feature_names
            or len(set(feature_names)) != len(feature_names)
            or target in feature_names
            or any((name not in reader.fieldnames for name in feature_names))
        ):
            raise GroupManifestError("CSV predictor allowlist is invalid")
        features: list[tuple[float, ...]] = []
        labels: list[int] = []
        for row in reader:
            features.append(tuple((_parse_numeric(str(row[name])) for name in feature_names)))
            label_value = _parse_numeric(str(row[target]))
            if math.isnan(label_value) or label_value not in (0.0, 1.0):
                raise GroupManifestError("CSV target must be 0 or 1")
            labels.append(int(label_value))
    return DatasetRows(
        dataset=dataset,
        source_path=str(source),
        source_sha256=sha256_file(source),
        feature_names=feature_names,
        features=tuple(features),
        labels=tuple(labels),
        row_ids=tuple(range(len(labels))),
        target_name=target,
        predictor_allowlist_explicit=predictors is not None,
    )


def _arff_attribute_name(line: str) -> str:
    remainder = line.strip()[len("@attribute") :].strip()
    if not remainder:
        raise GroupManifestError("ARFF attribute declaration is empty")
    if remainder[0] in {'"', "'"}:
        quote = remainder[0]
        end = remainder.find(quote, 1)
        if end <= 0:
            raise GroupManifestError("ARFF quoted attribute is malformed")
        return remainder[1:end]
    return remainder.split(None, 1)[0]


def load_arff_rows(
    path: Path | str, *, dataset: str, target: str, predictors: Sequence[str] | None = None
) -> DatasetRows:
    source = Path(path).resolve()
    attributes: list[str] = []
    data_rows: list[list[str]] = []
    in_data = False
    with source.open("r", encoding="utf-8", newline="") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            if not in_data:
                lowered = line.lower()
                if lowered.startswith("@attribute"):
                    attributes.append(_arff_attribute_name(line))
                elif lowered == "@data":
                    in_data = True
                continue
            data_rows.append(next(csv.reader([line])))
    if not in_data or not attributes or target not in attributes:
        raise GroupManifestError("ARFF schema or target is missing")
    if len(set(attributes)) != len(attributes):
        raise GroupManifestError("ARFF attribute names must be unique")
    target_index = attributes.index(target)
    feature_names = (
        tuple((str(name) for name in predictors))
        if predictors is not None
        else tuple((name for (index, name) in enumerate(attributes) if index != target_index))
    )
    if (
        not feature_names
        or len(set(feature_names)) != len(feature_names)
        or target in feature_names
        or any((name not in attributes for name in feature_names))
    ):
        raise GroupManifestError("ARFF predictor allowlist is invalid")
    feature_indices = tuple((attributes.index(name) for name in feature_names))
    features: list[tuple[float, ...]] = []
    labels: list[int] = []
    for row in data_rows:
        if len(row) != len(attributes):
            raise GroupManifestError("ARFF data width differs from schema")
        features.append(tuple((_parse_numeric(row[index]) for index in feature_indices)))
        label_value = _parse_numeric(row[target_index])
        if math.isnan(label_value) or label_value not in (0.0, 1.0):
            raise GroupManifestError("ARFF target must be 0 or 1")
        labels.append(int(label_value))
    return DatasetRows(
        dataset=dataset,
        source_path=str(source),
        source_sha256=sha256_file(source),
        feature_names=feature_names,
        features=tuple(features),
        labels=tuple(labels),
        row_ids=tuple(range(len(labels))),
        target_name=target,
        predictor_allowlist_explicit=predictors is not None,
    )


def build_group_manifest(
    rows: DatasetRows,
    *,
    subject_ids: Sequence[str] | None = None,
    subject_salt: str | None = None,
    subject_id_provenance: str | None = None,
) -> dict[str, Any]:
    if subject_ids is not None and len(subject_ids) != len(rows.row_ids):
        raise GroupManifestError("subject id count differs from dataset rows")
    if subject_ids is not None and (not subject_salt):
        raise GroupManifestError("subject-id grouping requires a non-empty salt")
    if subject_ids is not None and (not str(subject_id_provenance or "").strip()):
        raise GroupManifestError("subject-id grouping requires a provenance statement")
    if subject_ids is not None and any((not str(value).strip() for value in subject_ids)):
        raise GroupManifestError("subject ids must be non-empty")
    if not rows.row_ids:
        raise GroupManifestError("dataset has no rows")
    members: dict[str, list[int]] = {}
    for index, features in enumerate(rows.features):
        group_id = (
            _subject_group_id(str(subject_ids[index]), str(subject_salt))
            if subject_ids is not None
            else _feature_group_id(features)
        )
        members.setdefault(group_id, []).append(index)
    groups: list[GroupRecord] = []
    for group_id, member_ids in members.items():
        ordered = tuple(sorted(member_ids))
        positive = sum((rows.labels[row_id] for row_id in ordered))
        groups.append(
            GroupRecord(
                group_id=group_id,
                row_ids=ordered,
                n=len(ordered),
                positive=positive,
                negative=len(ordered) - positive,
            )
        )
    groups.sort(key=lambda value: value.group_id)
    mixed = sum((1 for group in groups if group.positive and group.negative))
    assignment_payload = "\n".join(
        (f"{row_id}:{group.group_id}" for group in groups for row_id in group.row_ids)
    ).encode("utf-8")
    manifest: dict[str, Any] = {
        "schema": _identity("group_manifest"),
        "dataset": rows.dataset,
        "source_path": rows.source_path,
        "source_sha256": rows.source_sha256,
        "row_count": len(rows.row_ids),
        "feature_count": len(rows.feature_names),
        "feature_names": list(rows.feature_names),
        "declared_target_name": rows.target_name,
        "predictor_allowlist_explicit": rows.predictor_allowlist_explicit,
        "group_rule": (
            "salted_subject_id_sha256"
            if subject_ids is not None
            else "label_excluded_canonical_float64_exact_feature_sha256"
        ),
        "group_id_hash_inputs": (
            "caller_supplied_subject_ids"
            if subject_ids is not None
            else "loaded_predictor_matrix_excluding_declared_target"
        ),
        "declared_target_excluded_by_loader": subject_ids is None,
        "subject_id_provenance": str(subject_id_provenance) if subject_ids is not None else None,
        "proxy_or_provenance_audit_required": True,
        "group_count": len(groups),
        "max_group_size": max((group.n for group in groups)),
        "mixed_label_group_count": mixed,
        "positive_rows": sum(rows.labels),
        "negative_rows": len(rows.labels) - sum(rows.labels),
        "row_to_group_sha256": hashlib.sha256(assignment_payload).hexdigest(),
        "groups": [group.as_dict() for group in groups],
    }
    manifest["group_manifest_sha256"] = group_manifest_fingerprint(manifest)
    return manifest


def group_records(manifest: Mapping[str, Any]) -> tuple[GroupRecord, ...]:

    def exact_int(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise GroupManifestError(f"{name} must be an exact integer")
        return value

    required_fields = {
        "schema",
        "dataset",
        "source_path",
        "source_sha256",
        "row_count",
        "feature_count",
        "feature_names",
        "declared_target_name",
        "predictor_allowlist_explicit",
        "group_rule",
        "group_id_hash_inputs",
        "declared_target_excluded_by_loader",
        "subject_id_provenance",
        "proxy_or_provenance_audit_required",
        "group_count",
        "max_group_size",
        "mixed_label_group_count",
        "positive_rows",
        "negative_rows",
        "row_to_group_sha256",
        "groups",
        "group_manifest_sha256",
    }
    schema = manifest.get("schema")
    if schema == _identity("derived_group_manifest"):
        required_fields.add("derivation")
    elif schema != _identity("group_manifest"):
        raise GroupManifestError("group manifest schema is invalid")
    actual_fields = set(manifest)
    missing = sorted(required_fields - actual_fields)
    extra = sorted(actual_fields - required_fields)
    if missing or extra:
        raise GroupManifestError(
            f"group manifest fields differ from the exact schema: missing={missing}, extra={extra}"
        )
    if schema == _identity("derived_group_manifest"):
        derivation = manifest.get("derivation")
        expected_derivation_fields = {
            "schema",
            "scope",
            "root_group_manifest_sha256",
            "root_row_to_group_sha256",
            "parent_split_sha256",
            "selected_group_membership_sha256",
        }
        if (
            not isinstance(derivation, Mapping)
            or set(derivation) != expected_derivation_fields
            or derivation.get("schema") != _identity("derived_group_manifest_binding")
            or (not isinstance(derivation.get("scope"), str))
            or (not derivation.get("scope"))
        ):
            raise GroupManifestError("derived group-manifest binding is malformed")
        for field in expected_derivation_fields - {"schema", "scope"}:
            value = derivation.get(field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any((character not in "0123456789abcdef" for character in value))
            ):
                raise GroupManifestError(f"derived group-manifest binding hash is invalid: {field}")
    stored_fingerprint = manifest.get("group_manifest_sha256")
    if (
        not isinstance(stored_fingerprint, str)
        or len(stored_fingerprint) != 64
        or any((character not in "0123456789abcdef" for character in stored_fingerprint))
        or (not hmac.compare_digest(stored_fingerprint, group_manifest_fingerprint(manifest)))
    ):
        raise GroupManifestError("group manifest fingerprint mismatch")
    source_sha256 = manifest.get("source_sha256")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any((character not in "0123456789abcdef" for character in source_sha256))
    ):
        raise GroupManifestError("group manifest source hash is invalid")
    records: list[GroupRecord] = []
    for item in manifest.get("groups", []):
        if not isinstance(item, Mapping):
            raise GroupManifestError("group manifest entry must be an object")
        expected_group_fields = {"group_id", "row_ids", "n", "positive", "negative"}
        if set(item) != expected_group_fields:
            raise GroupManifestError("group manifest entry fields differ from the exact schema")
        group_id = item.get("group_id")
        raw_row_ids = item.get("row_ids")
        if not isinstance(group_id, str) or not group_id:
            raise GroupManifestError("group id must be a non-empty string")
        if not isinstance(raw_row_ids, list):
            raise GroupManifestError("group row ids must be a JSON list")
        row_ids = tuple((exact_int(value, "row id") for value in raw_row_ids))
        record = GroupRecord(
            group_id=group_id,
            row_ids=row_ids,
            n=exact_int(item.get("n"), "group n"),
            positive=exact_int(item.get("positive"), "group positive"),
            negative=exact_int(item.get("negative"), "group negative"),
        )
        if (
            record.n <= 0
            or record.positive < 0
            or record.negative < 0
            or (record.n != len(record.row_ids))
            or (record.positive + record.negative != record.n)
            or (tuple(sorted(record.row_ids)) != record.row_ids)
            or (len(set(record.row_ids)) != len(record.row_ids))
            or any((value < 0 for value in record.row_ids))
        ):
            raise GroupManifestError("group manifest counts are inconsistent")
        records.append(record)
    if not records:
        raise GroupManifestError("group manifest has no groups")
    if len({record.group_id for record in records}) != len(records):
        raise GroupManifestError("group ids are not unique")
    all_rows = [row_id for record in records for row_id in record.row_ids]
    if len(all_rows) != len(set(all_rows)):
        raise GroupManifestError("a row appears in more than one group")
    row_count = exact_int(manifest["row_count"], "row_count")
    if row_count < 1 or set(all_rows) != set(range(row_count)):
        raise GroupManifestError("group rows do not exactly cover the dataset")
    derived_assignment = "\n".join(
        (
            f"{row_id}:{record.group_id}"
            for record in sorted(records, key=lambda value: value.group_id)
            for row_id in record.row_ids
        )
    ).encode("utf-8")
    derived = {
        "group_count": len(records),
        "max_group_size": max((record.n for record in records)),
        "mixed_label_group_count": sum(
            (1 for record in records if record.positive and record.negative)
        ),
        "positive_rows": sum((record.positive for record in records)),
        "negative_rows": sum((record.negative for record in records)),
        "row_to_group_sha256": hashlib.sha256(derived_assignment).hexdigest(),
    }
    for name, expected in derived.items():
        observed = manifest.get(name)
        if observed != expected:
            raise GroupManifestError(f"group manifest derived field mismatch: {name}")
    return tuple(records)
