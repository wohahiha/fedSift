"""Result-blind real-data pre-execution factory for FedSift.

The factory has exactly one data-loading entry point:
``load_registered_dataset(workspace_root, dataset_id)``.  The proposal path
reconstructs and validates every full planning artifact in memory, then
returns only a compact hash/count commitment.  After that hash is recorded
externally, the sealed construction path rebuilds the full runner objects.
The compact bundle never embeds row data, outer-fold memberships, the full
nested plan, the full HPO inventory, observed outputs, or post-HPO decisions.

This is an integrity and accidental-misuse boundary.  It is not encryption and
does not isolate the source data from a malicious process on the same host.
"""

from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import copy
import hmac
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
from .candidate_space import (
    MAIN_METHODS,
    MIN_CANDIDATE_COUNT,
    CandidateSpaceError,
    assert_no_performance_fields,
    build_candidate_space,
    canonical_sha256,
    freeze_candidate_space,
    require_exact_int,
    require_sha256,
    validate_candidate_space,
)
from .dataset_registry import DatasetRegistryError, load_registered_dataset
from .group_manifest import (
    DatasetRows,
    GroupManifestError,
    build_group_manifest,
    canonical_float64_bytes,
    group_records,
)
from .hpo_plan import HpoPlanError, build_hpo_plan, validate_hpo_plan
from .nested_plan import (
    INNER_FOLDS,
    OUTER_FOLDS,
    OUTER_REPEATS,
    NestedPlanError,
    freeze_nested_plan,
    generate_nested_plan,
    validate_nested_plan,
)
from .preprocessing import (
    PreprocessingError,
    preprocessing_profile,
    validate_preprocessing_profile_spec,
)


class StudyFactoryError(RuntimeError):
    """Raised when a real-data pre-execution bundle cannot be reconstructed."""


@dataclass(frozen=True, slots=True)
class StudyProposal:
    """Compact result-blind commitment; never an execution authorization."""

    status: str
    bundle: dict[str, object]
    proposed_bundle_sha256: str


_FACTORY_SEAL = object()


@dataclass(frozen=True, slots=True, init=False)
class StudyConstruction:
    """Validated full runner inputs plus a separate compact binding bundle.

    The mappings remain ordinary JSON-like objects so existing FedSift APIs can
    consume them directly.  ``frozen=True`` prevents field replacement; the
    rebuild validator detects any mutation inside a mapping or DatasetRows.
    """

    dataset_rows: DatasetRows
    group_manifest: dict[str, Any]
    frozen_nested_plan: dict[str, Any]
    frozen_candidate_space: dict[str, object]
    preprocessing_profile_spec: dict[str, Any]
    complete_hpo_plan: dict[str, object]
    bundle: dict[str, object]
    _factory_seal: object = field(repr=False, compare=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise StudyFactoryError(
            "RealN1StudyConstruction instances can only be created by the factory"
        )

    @property
    def rows(self) -> DatasetRows:
        return self.dataset_rows

    @property
    def nested_plan(self) -> dict[str, Any]:
        return self.frozen_nested_plan

    @property
    def candidate_space(self) -> dict[str, object]:
        return self.frozen_candidate_space

    @property
    def preprocessing_profile(self) -> dict[str, Any]:
        return self.preprocessing_profile_spec

    @property
    def hpo_plan(self) -> dict[str, object]:
        return self.complete_hpo_plan


@dataclass(frozen=True, slots=True)
class _StudyParts:
    """Private validated parts; carrying these does not authorize execution."""

    dataset_rows: DatasetRows
    group_manifest: dict[str, Any]
    frozen_nested_plan: dict[str, Any]
    frozen_candidate_space: dict[str, object]
    preprocessing_profile_spec: dict[str, Any]
    complete_hpo_plan: dict[str, object]
    bundle: dict[str, object]


SCHEMA = _identity("real_data_preexecution_bundle")
STATUS = "RESULT_BLIND_PREEXECUTION_COMMITMENT_NOT_EXECUTABLE"
PROPOSAL_STATUS = "PROPOSAL_REQUIRES_EXTERNAL_HASH_RECORD_BEFORE_REBUILD"
SECURITY_BOUNDARY = (
    "local_integrity_and_accidental_misuse_boundary_not_encryption_or_malicious_host_isolation"
)
_TOP_LEVEL_FIELDS = {
    "schema",
    "status",
    "security_boundary",
    "result_blind",
    "source_binding",
    "group_binding",
    "nested_plan_binding",
    "candidate_space_binding",
    "preprocessing_binding",
    "hpo_plan_binding",
    "preexecution_boundary",
    "bundle_sha256",
}
_SOURCE_FIELDS = {
    "study_id",
    "dataset_id",
    "registered_source_uri",
    "source_sha256",
    "row_count",
    "positive_count",
    "negative_count",
    "feature_count",
    "feature_names_sha256",
    "declared_target_name_sha256",
}
_GROUP_FIELDS = {
    "group_manifest_sha256",
    "row_to_group_sha256",
    "group_count",
    "max_group_size",
    "mixed_label_group_count",
}
_NESTED_FIELDS = {
    "nested_plan_sha256",
    "freeze_binding_sha256",
    "protocol_sha256",
    "implementation_sha256",
    "client_policy_sha256",
    "outer_repeats",
    "outer_folds_per_repeat",
    "inner_folds_per_outer_fold",
    "outer_scope_count",
    "inner_scope_count",
    "split_plan_count",
    "unique_split_plan_count",
}
_CANDIDATE_FIELDS = {
    "candidate_space_sha256",
    "method_count",
    "candidate_count_per_method",
    "total_candidate_count",
}
_PREPROCESSING_FIELDS = {
    "profile_name",
    "profile_sha256",
    "dataset_id",
    "predictor_count",
    "predictor_names_sha256",
}
_HPO_FIELDS = {
    "hpo_plan_sha256",
    "hpo_seeds",
    "hpo_seed_count",
    "max_steps",
    "expected_unit_count",
    "expected_units_per_method",
    "expected_units_per_outer_method",
}
_BOUNDARY_FIELDS = {
    "source_loader",
    "returned_payload",
    "full_group_manifest_embedded",
    "frozen_nested_plan_embedded",
    "frozen_candidate_space_embedded",
    "preprocessing_profile_spec_embedded",
    "complete_hpo_plan_embedded",
    "outer_payloads_embedded",
    "observed_outputs_embedded",
    "post_hpo_decisions_embedded",
    "training_or_evaluation_executed",
}
_FORBIDDEN_PREEXECUTION_KEYS = {
    "attempt_receipt",
    "attempt_receipts",
    "authorization",
    "authorizations",
    "gate",
    "gates",
    "hpo_closure",
    "ledger",
    "ledgers",
    "outer_data",
    "outer_features",
    "outer_labels",
    "outer_rows",
    "outer_test",
    "outer_test_data",
    "outer_test_rows",
    "selection",
    "selection_authorization",
    "selection_decision",
    "selection_gate",
}


def _required_hash(value: object, field: str) -> str:
    try:
        return require_sha256(value, field)
    except CandidateSpaceError as exc:
        raise StudyFactoryError(str(exc)) from exc


def _exact_positive_int(value: object, field: str) -> int:
    try:
        return require_exact_int(value, field, minimum=1)
    except CandidateSpaceError as exc:
        raise StudyFactoryError(str(exc)) from exc


def _study_identifier(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise StudyFactoryError("study_id must be a non-empty trimmed string")
    return value


def _dataset_identifier(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise StudyFactoryError("dataset_id must be a non-empty trimmed string")
    return value


def _profile_identifier(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise StudyFactoryError("preprocessing_profile_name must be a non-empty trimmed string")
    return value


def _strict_hpo_seeds(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise StudyFactoryError("hpo_seeds must be a sequence of exact integers")
    seeds: list[int] = []
    for index, value in enumerate(values):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            or (value > 2**63 - 1)
        ):
            raise StudyFactoryError(
                f"hpo_seeds[{index}] must be a nonnegative signed-64-bit integer"
            )
        seeds.append(value)
    if not seeds or len(seeds) != len(set(seeds)):
        raise StudyFactoryError("hpo_seeds must be non-empty and unique")
    return tuple(seeds)


def _assert_preexecution_only(value: object, path: str = "bundle") -> None:
    try:
        assert_no_performance_fields(value, path)
    except CandidateSpaceError as exc:
        raise StudyFactoryError(str(exc)) from exc
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise StudyFactoryError(f"{path} contains a non-string key")
            key = raw_key.strip().lower().replace("-", "_")
            if key in _FORBIDDEN_PREEXECUTION_KEYS:
                raise StudyFactoryError(
                    f"pre-execution bundle contains forbidden field {path}.{raw_key}"
                )
            _assert_preexecution_only(child, f"{path}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_preexecution_only(child, f"{path}[{index}]")


def study_bundle_fingerprint(bundle: Mapping[str, object]) -> str:
    """Hash every bundle field except ``bundle_sha256`` itself."""
    if not isinstance(bundle, Mapping):
        raise StudyFactoryError("study bundle must be a mapping")
    payload = {str(key): value for (key, value) in bundle.items() if key != "bundle_sha256"}
    try:
        return canonical_sha256(payload)
    except CandidateSpaceError as exc:
        raise StudyFactoryError(str(exc)) from exc


def _seal_construction(
    *,
    dataset_rows: DatasetRows,
    group_manifest: dict[str, Any],
    frozen_nested_plan: dict[str, Any],
    frozen_candidate_space: dict[str, object],
    preprocessing_profile_spec: dict[str, Any],
    complete_hpo_plan: dict[str, object],
    bundle: dict[str, object],
) -> StudyConstruction:
    construction = object.__new__(StudyConstruction)
    values = {
        "dataset_rows": dataset_rows,
        "group_manifest": group_manifest,
        "frozen_nested_plan": frozen_nested_plan,
        "frozen_candidate_space": frozen_candidate_space,
        "preprocessing_profile_spec": preprocessing_profile_spec,
        "complete_hpo_plan": complete_hpo_plan,
        "bundle": bundle,
        "_factory_seal": _FACTORY_SEAL,
    }
    for name, value in values.items():
        object.__setattr__(construction, name, value)
    return construction


def _nested_counts(plan: Mapping[str, Any]) -> dict[str, int]:
    repetitions = plan.get("repetitions")
    if not isinstance(repetitions, list):
        raise StudyFactoryError("nested plan repetitions are missing")
    outer_scope_count = 0
    inner_scope_count = 0
    split_hashes: list[str] = []
    for repeat in repetitions:
        if not isinstance(repeat, Mapping):
            raise StudyFactoryError("nested plan repetition is malformed")
        outer_partition = repeat.get("outer_partition")
        outer_folds = repeat.get("outer_folds")
        if not isinstance(outer_partition, Mapping) or not isinstance(outer_folds, list):
            raise StudyFactoryError("nested plan outer structure is malformed")
        split_hashes.append(str(outer_partition.get("split_sha256")))
        outer_scope_count += len(outer_folds)
        for outer in outer_folds:
            if not isinstance(outer, Mapping):
                raise StudyFactoryError("nested plan outer fold is malformed")
            inner_partition = outer.get("inner_partition")
            inner_folds = outer.get("inner_folds")
            outer_refit = outer.get("outer_refit")
            if not all(
                (isinstance(item, Mapping) for item in (inner_partition, outer_refit))
            ) or not isinstance(inner_folds, list):
                raise StudyFactoryError("nested plan inner structure is malformed")
            split_hashes.append(str(inner_partition.get("split_sha256")))
            for key in ("role_partition", "client_partition"):
                split = outer_refit.get(key)
                if not isinstance(split, Mapping):
                    raise StudyFactoryError("outer-refit split is malformed")
                split_hashes.append(str(split.get("split_sha256")))
            inner_scope_count += len(inner_folds)
            for inner in inner_folds:
                if not isinstance(inner, Mapping):
                    raise StudyFactoryError("nested plan inner fold is malformed")
                for key in ("role_partition", "client_partition"):
                    split = inner.get(key)
                    if not isinstance(split, Mapping):
                        raise StudyFactoryError("inner split is malformed")
                    split_hashes.append(str(split.get("split_sha256")))
    expected = {
        "outer_scope_count": OUTER_REPEATS * OUTER_FOLDS,
        "inner_scope_count": OUTER_REPEATS * OUTER_FOLDS * INNER_FOLDS,
        "split_plan_count": OUTER_REPEATS
        + OUTER_REPEATS * OUTER_FOLDS
        + 2 * OUTER_REPEATS * OUTER_FOLDS * INNER_FOLDS
        + 2 * OUTER_REPEATS * OUTER_FOLDS,
    }
    observed = {
        "outer_scope_count": outer_scope_count,
        "inner_scope_count": inner_scope_count,
        "split_plan_count": len(split_hashes),
    }
    if len(repetitions) != OUTER_REPEATS or observed != expected:
        raise StudyFactoryError("nested plan counts differ from fixed 3x5x3 design")
    for value in split_hashes:
        _required_hash(value, "nested split hash")
    if len(set(split_hashes)) != len(split_hashes):
        raise StudyFactoryError("nested split hashes are not unique")
    return {**observed, "unique_split_plan_count": len(set(split_hashes))}


def _construct_study_parts(
    workspace_root: Path | str,
    dataset_id: str,
    *,
    study_id: str,
    protocol_sha256: str,
    implementation_sha256: str,
    client_policy_sha256: str,
    candidate_count: int,
    hpo_seeds: Sequence[int],
    max_steps: int,
    preprocessing_profile_name: str,
) -> _StudyParts:
    """Construct full artifacts, validate them, and retain compact bindings."""
    study = _study_identifier(study_id)
    dataset = _dataset_identifier(dataset_id)
    profile_name = _profile_identifier(preprocessing_profile_name)
    protocol_hash = _required_hash(protocol_sha256, "protocol_sha256")
    implementation_hash = _required_hash(implementation_sha256, "implementation_sha256")
    client_policy_hash = _required_hash(client_policy_sha256, "client_policy_sha256")
    try:
        candidates_per_method = require_exact_int(
            candidate_count, "candidate_count", minimum=MIN_CANDIDATE_COUNT
        )
    except CandidateSpaceError as exc:
        raise StudyFactoryError(str(exc)) from exc
    seeds = _strict_hpo_seeds(hpo_seeds)
    steps = _exact_positive_int(max_steps, "max_steps")
    try:
        rows = load_registered_dataset(workspace_root, dataset)
        manifest = build_group_manifest(rows)
        records = group_records(manifest)
        profile_spec = preprocessing_profile(profile_name)
        validate_preprocessing_profile_spec(
            profile_spec, dataset=dataset, feature_names=rows.feature_names
        )
        draft_nested = generate_nested_plan(manifest, study_id=study)
        nested = freeze_nested_plan(
            draft_nested,
            manifest,
            protocol_sha256=protocol_hash,
            implementation_sha256=implementation_hash,
            client_policy_sha256=client_policy_hash,
        )
        validate_nested_plan(nested, manifest, require_frozen=True)
        draft_candidates = build_candidate_space(study, candidate_count=candidates_per_method)
        candidate_space = freeze_candidate_space(
            draft_candidates,
            protocol_sha256=protocol_hash,
            implementation_contract_sha256=implementation_hash,
        )
        validate_candidate_space(candidate_space, require_frozen=True)
        hpo_plan = build_hpo_plan(
            candidate_space,
            nested,
            manifest,
            preprocessing_profile_spec=profile_spec,
            hpo_seeds=seeds,
            max_steps=steps,
        )
        validate_hpo_plan(hpo_plan, candidate_space, nested, manifest)
    except (
        CandidateSpaceError,
        DatasetRegistryError,
        GroupManifestError,
        HpoPlanError,
        NestedPlanError,
        PreprocessingError,
        TypeError,
        ValueError,
    ) as exc:
        raise StudyFactoryError(str(exc)) from exc
    nested_counts = _nested_counts(nested)
    expected_total_units = (
        len(MAIN_METHODS)
        * OUTER_REPEATS
        * OUTER_FOLDS
        * INNER_FOLDS
        * candidates_per_method
        * len(seeds)
    )
    expected_per_method = (
        OUTER_REPEATS * OUTER_FOLDS * INNER_FOLDS * candidates_per_method * len(seeds)
    )
    expected_per_outer_method = INNER_FOLDS * candidates_per_method * len(seeds)
    equal_budget = hpo_plan.get("equal_budget_contract")
    freeze_binding = nested.get("freeze_binding")
    if not isinstance(equal_budget, Mapping) or not isinstance(freeze_binding, Mapping):
        raise StudyFactoryError("validated plans lack required compact bindings")
    if (
        hpo_plan.get("expected_unit_count") != expected_total_units
        or len(hpo_plan.get("units", ())) != expected_total_units
        or equal_budget.get("expected_units_per_method") != expected_per_method
        or (equal_budget.get("expected_units_per_outer_method") != expected_per_outer_method)
    ):
        raise StudyFactoryError("HPO plan is not the complete equal-budget inventory")
    positive_count = sum(rows.labels)
    feature_names = list(rows.feature_names)
    predictor_names = list(profile_spec["predictor_allowlist"])
    bundle: dict[str, object] = {
        "schema": SCHEMA,
        "status": STATUS,
        "security_boundary": SECURITY_BOUNDARY,
        "result_blind": True,
        "source_binding": {
            "study_id": study,
            "dataset_id": dataset,
            "registered_source_uri": rows.source_path,
            "source_sha256": rows.source_sha256,
            "row_count": len(rows.row_ids),
            "positive_count": positive_count,
            "negative_count": len(rows.row_ids) - positive_count,
            "feature_count": len(rows.feature_names),
            "feature_names_sha256": canonical_sha256(feature_names),
            "declared_target_name_sha256": canonical_sha256(rows.target_name),
        },
        "group_binding": {
            "group_manifest_sha256": manifest["group_manifest_sha256"],
            "row_to_group_sha256": manifest["row_to_group_sha256"],
            "group_count": len(records),
            "max_group_size": manifest["max_group_size"],
            "mixed_label_group_count": manifest["mixed_label_group_count"],
        },
        "nested_plan_binding": {
            "nested_plan_sha256": nested["nested_plan_sha256"],
            "freeze_binding_sha256": freeze_binding["freeze_binding_sha256"],
            "protocol_sha256": protocol_hash,
            "implementation_sha256": implementation_hash,
            "client_policy_sha256": client_policy_hash,
            "outer_repeats": OUTER_REPEATS,
            "outer_folds_per_repeat": OUTER_FOLDS,
            "inner_folds_per_outer_fold": INNER_FOLDS,
            **nested_counts,
        },
        "candidate_space_binding": {
            "candidate_space_sha256": candidate_space["manifest_sha256"],
            "method_count": len(MAIN_METHODS),
            "candidate_count_per_method": candidates_per_method,
            "total_candidate_count": len(MAIN_METHODS) * candidates_per_method,
        },
        "preprocessing_binding": {
            "profile_name": profile_spec["name"],
            "profile_sha256": profile_spec["profile_sha256"],
            "dataset_id": profile_spec["dataset"],
            "predictor_count": len(predictor_names),
            "predictor_names_sha256": canonical_sha256(predictor_names),
        },
        "hpo_plan_binding": {
            "hpo_plan_sha256": hpo_plan["hpo_plan_sha256"],
            "hpo_seeds": list(seeds),
            "hpo_seed_count": len(seeds),
            "max_steps": steps,
            "expected_unit_count": expected_total_units,
            "expected_units_per_method": expected_per_method,
            "expected_units_per_outer_method": expected_per_outer_method,
        },
        "preexecution_boundary": {
            "source_loader": _identity("dataset_registry_load_registered_dataset"),
            "returned_payload": "compact_hashes_and_counts_only",
            "full_group_manifest_embedded": False,
            "frozen_nested_plan_embedded": False,
            "frozen_candidate_space_embedded": False,
            "preprocessing_profile_spec_embedded": False,
            "complete_hpo_plan_embedded": False,
            "outer_payloads_embedded": False,
            "observed_outputs_embedded": False,
            "post_hpo_decisions_embedded": False,
            "training_or_evaluation_executed": False,
        },
    }
    _assert_preexecution_only(bundle)
    bundle["bundle_sha256"] = study_bundle_fingerprint(bundle)
    return _StudyParts(
        dataset_rows=rows,
        group_manifest=manifest,
        frozen_nested_plan=nested,
        frozen_candidate_space=candidate_space,
        preprocessing_profile_spec=profile_spec,
        complete_hpo_plan=hpo_plan,
        bundle=bundle,
    )


def validate_study_bundle_integrity(
    bundle: Mapping[str, object], *, expected_bundle_sha256: str
) -> None:
    """Validate exact shape, self-hash, external hash, and blind boundary."""
    expected_hash = _required_hash(expected_bundle_sha256, "expected_bundle_sha256")
    _assert_preexecution_only(bundle)
    if not isinstance(bundle, Mapping) or set(bundle) != _TOP_LEVEL_FIELDS:
        raise StudyFactoryError("study bundle fields differ from the exact schema")
    if bundle.get("schema") != SCHEMA or bundle.get("status") != STATUS:
        raise StudyFactoryError("study bundle schema or status differs")
    if bundle.get("security_boundary") != SECURITY_BOUNDARY:
        raise StudyFactoryError("study bundle security boundary differs")
    if bundle.get("result_blind") is not True:
        raise StudyFactoryError("study bundle is not explicitly result blind")
    section_fields = (
        ("source_binding", _SOURCE_FIELDS),
        ("group_binding", _GROUP_FIELDS),
        ("nested_plan_binding", _NESTED_FIELDS),
        ("candidate_space_binding", _CANDIDATE_FIELDS),
        ("preprocessing_binding", _PREPROCESSING_FIELDS),
        ("hpo_plan_binding", _HPO_FIELDS),
        ("preexecution_boundary", _BOUNDARY_FIELDS),
    )
    for section_name, expected_fields in section_fields:
        section = bundle.get(section_name)
        if not isinstance(section, Mapping) or set(section) != expected_fields:
            raise StudyFactoryError(
                f"study bundle {section_name} fields differ from the exact schema"
            )
    source = dict(bundle["source_binding"])
    study = _study_identifier(source.get("study_id"))
    del study
    dataset = _dataset_identifier(source.get("dataset_id"))
    source_hash = _required_hash(source.get("source_sha256"), "source_sha256")
    if source.get("registered_source_uri") != f"registered://{dataset}/{source_hash}":
        raise StudyFactoryError("registered source URI differs from its binding")
    for field in ("row_count", "feature_count", "positive_count", "negative_count"):
        value = source.get(field)
        minimum = 1 if field in {"row_count", "feature_count"} else 0
        try:
            require_exact_int(value, f"source_binding.{field}", minimum=minimum)
        except CandidateSpaceError as exc:
            raise StudyFactoryError(str(exc)) from exc
    if source["positive_count"] + source["negative_count"] != source["row_count"]:
        raise StudyFactoryError("source class counts do not equal row_count")
    for field in ("feature_names_sha256", "declared_target_name_sha256"):
        _required_hash(source.get(field), f"source_binding.{field}")
    group = dict(bundle["group_binding"])
    for field in ("group_manifest_sha256", "row_to_group_sha256"):
        _required_hash(group.get(field), f"group_binding.{field}")
    for field in ("group_count", "max_group_size", "mixed_label_group_count"):
        minimum = 1 if field != "mixed_label_group_count" else 0
        try:
            require_exact_int(group.get(field), field, minimum=minimum)
        except CandidateSpaceError as exc:
            raise StudyFactoryError(str(exc)) from exc
    if group["group_count"] > source["row_count"]:
        raise StudyFactoryError("group_count exceeds source row_count")
    nested = dict(bundle["nested_plan_binding"])
    for field in (
        "nested_plan_sha256",
        "freeze_binding_sha256",
        "protocol_sha256",
        "implementation_sha256",
        "client_policy_sha256",
    ):
        _required_hash(nested.get(field), f"nested_plan_binding.{field}")
    exact_nested_counts = {
        "outer_repeats": OUTER_REPEATS,
        "outer_folds_per_repeat": OUTER_FOLDS,
        "inner_folds_per_outer_fold": INNER_FOLDS,
        "outer_scope_count": OUTER_REPEATS * OUTER_FOLDS,
        "inner_scope_count": OUTER_REPEATS * OUTER_FOLDS * INNER_FOLDS,
        "split_plan_count": 138,
        "unique_split_plan_count": 138,
    }
    if any((nested.get(field) != value for (field, value) in exact_nested_counts.items())):
        raise StudyFactoryError("nested-plan counts differ from exact 3x5x3 design")
    candidate = dict(bundle["candidate_space_binding"])
    candidate_count = _exact_positive_int(
        candidate.get("candidate_count_per_method"), "candidate_count_per_method"
    )
    if candidate_count < MIN_CANDIDATE_COUNT or candidate_count > 256:
        raise StudyFactoryError("candidate count is outside the frozen safety range")
    if (
        candidate.get("method_count") != len(MAIN_METHODS)
        or candidate.get("total_candidate_count") != len(MAIN_METHODS) * candidate_count
    ):
        raise StudyFactoryError("candidate-space counts violate equal budgeting")
    _required_hash(candidate.get("candidate_space_sha256"), "candidate_space_sha256")
    profile_binding = dict(bundle["preprocessing_binding"])
    profile_name = _profile_identifier(profile_binding.get("profile_name"))
    try:
        profile_spec = preprocessing_profile(profile_name)
        validate_preprocessing_profile_spec(profile_spec, dataset=dataset)
    except PreprocessingError as exc:
        raise StudyFactoryError(str(exc)) from exc
    predictor_names = list(profile_spec["predictor_allowlist"])
    if (
        profile_binding.get("profile_sha256") != profile_spec["profile_sha256"]
        or profile_binding.get("dataset_id") != dataset
        or profile_binding.get("predictor_count") != len(predictor_names)
        or (profile_binding.get("predictor_names_sha256") != canonical_sha256(predictor_names))
        or (source.get("feature_names_sha256") != canonical_sha256(predictor_names))
    ):
        raise StudyFactoryError("preprocessing binding differs from its full profile")
    hpo = dict(bundle["hpo_plan_binding"])
    _required_hash(hpo.get("hpo_plan_sha256"), "hpo_plan_sha256")
    seeds = _strict_hpo_seeds(hpo.get("hpo_seeds", ()))
    steps = _exact_positive_int(hpo.get("max_steps"), "max_steps")
    expected_per_outer_method = INNER_FOLDS * candidate_count * len(seeds)
    expected_per_method = OUTER_REPEATS * OUTER_FOLDS * expected_per_outer_method
    expected_units = len(MAIN_METHODS) * expected_per_method
    if (
        hpo.get("hpo_seed_count") != len(seeds)
        or hpo.get("expected_units_per_outer_method") != expected_per_outer_method
        or hpo.get("expected_units_per_method") != expected_per_method
        or (hpo.get("expected_unit_count") != expected_units)
    ):
        raise StudyFactoryError("HPO counts violate the complete equal-budget product")
    del steps
    boundary = dict(bundle["preexecution_boundary"])
    expected_boundary = {
        "source_loader": _identity("dataset_registry_load_registered_dataset"),
        "returned_payload": "compact_hashes_and_counts_only",
        "full_group_manifest_embedded": False,
        "frozen_nested_plan_embedded": False,
        "frozen_candidate_space_embedded": False,
        "preprocessing_profile_spec_embedded": False,
        "complete_hpo_plan_embedded": False,
        "outer_payloads_embedded": False,
        "observed_outputs_embedded": False,
        "post_hpo_decisions_embedded": False,
        "training_or_evaluation_executed": False,
    }
    if boundary != expected_boundary:
        raise StudyFactoryError("pre-execution payload boundary differs")
    stored_hash = _required_hash(bundle.get("bundle_sha256"), "bundle_sha256")
    actual_hash = study_bundle_fingerprint(bundle)
    if not hmac.compare_digest(stored_hash, actual_hash):
        raise StudyFactoryError("study bundle self-hash differs")
    if not hmac.compare_digest(stored_hash, expected_hash):
        raise StudyFactoryError("study bundle differs from external expected hash")


def _validate_explicit_bundle_inputs(
    bundle: Mapping[str, object],
    dataset_id: str,
    *,
    study_id: str,
    protocol_sha256: str,
    implementation_sha256: str,
    client_policy_sha256: str,
    candidate_count: int,
    hpo_seeds: Sequence[int],
    max_steps: int,
    preprocessing_profile_name: str,
) -> None:
    source = dict(bundle["source_binding"])
    nested = dict(bundle["nested_plan_binding"])
    candidate = dict(bundle["candidate_space_binding"])
    profile = dict(bundle["preprocessing_binding"])
    hpo = dict(bundle["hpo_plan_binding"])
    explicit = {
        "study_id": _study_identifier(study_id),
        "dataset_id": _dataset_identifier(dataset_id),
        "protocol_sha256": _required_hash(protocol_sha256, "protocol_sha256"),
        "implementation_sha256": _required_hash(implementation_sha256, "implementation_sha256"),
        "client_policy_sha256": _required_hash(client_policy_sha256, "client_policy_sha256"),
        "candidate_count": _exact_positive_int(candidate_count, "candidate_count"),
        "hpo_seeds": list(_strict_hpo_seeds(hpo_seeds)),
        "max_steps": _exact_positive_int(max_steps, "max_steps"),
        "profile_name": _profile_identifier(preprocessing_profile_name),
    }
    observed = {
        "study_id": source["study_id"],
        "dataset_id": source["dataset_id"],
        "protocol_sha256": nested["protocol_sha256"],
        "implementation_sha256": nested["implementation_sha256"],
        "client_policy_sha256": nested["client_policy_sha256"],
        "candidate_count": candidate["candidate_count_per_method"],
        "hpo_seeds": hpo["hpo_seeds"],
        "max_steps": hpo["max_steps"],
        "profile_name": profile["profile_name"],
    }
    if explicit != observed:
        differing = sorted((field for field in explicit if explicit[field] != observed.get(field)))
        raise StudyFactoryError(
            "explicit reconstruction inputs differ from bundle: " + ", ".join(differing)
        )


def _dataset_rows_equal(left: DatasetRows, right: DatasetRows) -> bool:
    scalar_fields = (
        "dataset",
        "source_path",
        "source_sha256",
        "feature_names",
        "labels",
        "row_ids",
        "target_name",
        "predictor_allowlist_explicit",
    )
    if any((getattr(left, field) != getattr(right, field) for field in scalar_fields)):
        return False
    if len(left.features) != len(right.features):
        return False
    for left_row, right_row in zip(left.features, right.features):
        if len(left_row) != len(right_row):
            return False
        if any(
            (
                canonical_float64_bytes(left_value) != canonical_float64_bytes(right_value)
                for (left_value, right_value) in zip(left_row, right_row)
            )
        ):
            return False
    return True


def propose_study(
    workspace_root: Path | str,
    dataset_id: str,
    *,
    study_id: str,
    protocol_sha256: str,
    implementation_sha256: str,
    client_policy_sha256: str,
    candidate_count: int,
    hpo_seeds: Sequence[int],
    max_steps: int,
    preprocessing_profile_name: str,
) -> StudyProposal:
    """Produce only a compact commitment and hash for external recording.

    The returned proposal is intentionally not factory-sealed and cannot be
    consumed where a ``StudyConstruction`` is required.  After recording
    ``proposed_bundle_sha256`` outside this process, the caller must invoke
    ``build_real_n1_study_construction`` with that value as the external
    expected hash.
    """
    parts = _construct_study_parts(
        workspace_root,
        dataset_id,
        study_id=study_id,
        protocol_sha256=protocol_sha256,
        implementation_sha256=implementation_sha256,
        client_policy_sha256=client_policy_sha256,
        candidate_count=candidate_count,
        hpo_seeds=hpo_seeds,
        max_steps=max_steps,
        preprocessing_profile_name=preprocessing_profile_name,
    )
    compact = copy.deepcopy(parts.bundle)
    proposal = StudyProposal(
        status=PROPOSAL_STATUS, bundle=compact, proposed_bundle_sha256=str(compact["bundle_sha256"])
    )
    validate_study_proposal(proposal)
    return proposal


def validate_study_proposal(proposal: StudyProposal) -> None:
    """Validate a compact proposal without granting execution authority."""
    if not isinstance(proposal, StudyProposal):
        raise StudyFactoryError("proposal has the wrong result type")
    if proposal.status != PROPOSAL_STATUS:
        raise StudyFactoryError("proposal status differs")
    proposed_hash = _required_hash(proposal.proposed_bundle_sha256, "proposed_bundle_sha256")
    validate_study_bundle_integrity(proposal.bundle, expected_bundle_sha256=proposed_hash)


def build_study_construction(
    workspace_root: Path | str,
    dataset_id: str,
    *,
    study_id: str,
    protocol_sha256: str,
    implementation_sha256: str,
    client_policy_sha256: str,
    candidate_count: int,
    hpo_seeds: Sequence[int],
    max_steps: int,
    preprocessing_profile_name: str,
    expected_bundle_sha256: str,
) -> StudyConstruction:
    """Build all validated runner objects plus a separate compact bundle."""
    expected_hash = _required_hash(expected_bundle_sha256, "expected_bundle_sha256")
    parts = _construct_study_parts(
        workspace_root,
        dataset_id,
        study_id=study_id,
        protocol_sha256=protocol_sha256,
        implementation_sha256=implementation_sha256,
        client_policy_sha256=client_policy_sha256,
        candidate_count=candidate_count,
        hpo_seeds=hpo_seeds,
        max_steps=max_steps,
        preprocessing_profile_name=preprocessing_profile_name,
    )
    validate_study_bundle_integrity(parts.bundle, expected_bundle_sha256=expected_hash)
    construction = _seal_construction(
        dataset_rows=parts.dataset_rows,
        group_manifest=parts.group_manifest,
        frozen_nested_plan=parts.frozen_nested_plan,
        frozen_candidate_space=parts.frozen_candidate_space,
        preprocessing_profile_spec=parts.preprocessing_profile_spec,
        complete_hpo_plan=parts.complete_hpo_plan,
        bundle=parts.bundle,
    )
    if getattr(construction, "_factory_seal", None) is not _FACTORY_SEAL:
        raise StudyFactoryError("factory failed to seal the construction")
    return construction


def build_study_bundle(
    workspace_root: Path | str,
    dataset_id: str,
    *,
    study_id: str,
    protocol_sha256: str,
    implementation_sha256: str,
    client_policy_sha256: str,
    candidate_count: int,
    hpo_seeds: Sequence[int],
    max_steps: int,
    preprocessing_profile_name: str,
    expected_bundle_sha256: str,
) -> dict[str, object]:
    """Build and return only the externally bound compact bundle."""
    expected_hash = _required_hash(expected_bundle_sha256, "expected_bundle_sha256")
    parts = _construct_study_parts(
        workspace_root,
        dataset_id,
        study_id=study_id,
        protocol_sha256=protocol_sha256,
        implementation_sha256=implementation_sha256,
        client_policy_sha256=client_policy_sha256,
        candidate_count=candidate_count,
        hpo_seeds=hpo_seeds,
        max_steps=max_steps,
        preprocessing_profile_name=preprocessing_profile_name,
    )
    validate_study_bundle_integrity(parts.bundle, expected_bundle_sha256=expected_hash)
    return copy.deepcopy(parts.bundle)


def validate_study_bundle(
    bundle: Mapping[str, object],
    workspace_root: Path | str,
    dataset_id: str,
    *,
    study_id: str,
    protocol_sha256: str,
    implementation_sha256: str,
    client_policy_sha256: str,
    candidate_count: int,
    hpo_seeds: Sequence[int],
    max_steps: int,
    preprocessing_profile_name: str,
    expected_bundle_sha256: str,
) -> None:
    """Rebuild all full artifacts and exact-compare the compact bundle."""
    validate_study_bundle_integrity(bundle, expected_bundle_sha256=expected_bundle_sha256)
    _validate_explicit_bundle_inputs(
        bundle,
        dataset_id,
        study_id=study_id,
        protocol_sha256=protocol_sha256,
        implementation_sha256=implementation_sha256,
        client_policy_sha256=client_policy_sha256,
        candidate_count=candidate_count,
        hpo_seeds=hpo_seeds,
        max_steps=max_steps,
        preprocessing_profile_name=preprocessing_profile_name,
    )
    rebuilt = _construct_study_parts(
        workspace_root,
        dataset_id,
        study_id=study_id,
        protocol_sha256=protocol_sha256,
        implementation_sha256=implementation_sha256,
        client_policy_sha256=client_policy_sha256,
        candidate_count=candidate_count,
        hpo_seeds=hpo_seeds,
        max_steps=max_steps,
        preprocessing_profile_name=preprocessing_profile_name,
    )
    if dict(bundle) != rebuilt.bundle:
        raise StudyFactoryError("study bundle differs from full real-data reconstruction")


def validate_study_construction(
    construction: StudyConstruction,
    workspace_root: Path | str,
    dataset_id: str,
    *,
    study_id: str,
    protocol_sha256: str,
    implementation_sha256: str,
    client_policy_sha256: str,
    candidate_count: int,
    hpo_seeds: Sequence[int],
    max_steps: int,
    preprocessing_profile_name: str,
    expected_bundle_sha256: str,
) -> None:
    """Rebuild and exact-compare every full runner object and the bundle."""
    if (
        not isinstance(construction, StudyConstruction)
        or getattr(construction, "_factory_seal", None) is not _FACTORY_SEAL
    ):
        raise StudyFactoryError("construction is not an authentic live factory instance")
    validate_study_bundle_integrity(
        construction.bundle, expected_bundle_sha256=expected_bundle_sha256
    )
    _validate_explicit_bundle_inputs(
        construction.bundle,
        dataset_id,
        study_id=study_id,
        protocol_sha256=protocol_sha256,
        implementation_sha256=implementation_sha256,
        client_policy_sha256=client_policy_sha256,
        candidate_count=candidate_count,
        hpo_seeds=hpo_seeds,
        max_steps=max_steps,
        preprocessing_profile_name=preprocessing_profile_name,
    )
    try:
        exact_manifest = build_group_manifest(construction.dataset_rows)
        group_records(construction.group_manifest)
        if exact_manifest != construction.group_manifest:
            raise StudyFactoryError(
                "construction DatasetRows do not reconstruct its group manifest"
            )
        validate_nested_plan(
            construction.frozen_nested_plan, construction.group_manifest, require_frozen=True
        )
        validate_candidate_space(construction.frozen_candidate_space, require_frozen=True)
        validate_preprocessing_profile_spec(
            construction.preprocessing_profile_spec,
            dataset=construction.dataset_rows.dataset,
            feature_names=construction.dataset_rows.feature_names,
        )
        validate_hpo_plan(
            construction.complete_hpo_plan,
            construction.frozen_candidate_space,
            construction.frozen_nested_plan,
            construction.group_manifest,
        )
    except StudyFactoryError:
        raise
    except (
        CandidateSpaceError,
        GroupManifestError,
        HpoPlanError,
        NestedPlanError,
        PreprocessingError,
        TypeError,
        ValueError,
    ) as exc:
        raise StudyFactoryError(str(exc)) from exc
    rebuilt = _construct_study_parts(
        workspace_root,
        dataset_id,
        study_id=study_id,
        protocol_sha256=protocol_sha256,
        implementation_sha256=implementation_sha256,
        client_policy_sha256=client_policy_sha256,
        candidate_count=candidate_count,
        hpo_seeds=hpo_seeds,
        max_steps=max_steps,
        preprocessing_profile_name=preprocessing_profile_name,
    )
    if (
        not _dataset_rows_equal(construction.dataset_rows, rebuilt.dataset_rows)
        or construction.group_manifest != rebuilt.group_manifest
        or construction.frozen_nested_plan != rebuilt.frozen_nested_plan
        or (construction.frozen_candidate_space != rebuilt.frozen_candidate_space)
        or (construction.preprocessing_profile_spec != rebuilt.preprocessing_profile_spec)
        or (construction.complete_hpo_plan != rebuilt.complete_hpo_plan)
        or (construction.bundle != rebuilt.bundle)
    ):
        raise StudyFactoryError("construction differs from full real-data reconstruction")


build_study_bundle = build_study_bundle
propose_study_bundle = propose_study
validate_study_bundle = validate_study_bundle
__all__ = [
    "SCHEMA",
    "SECURITY_BOUNDARY",
    "STATUS",
    "PROPOSAL_STATUS",
    "StudyConstruction",
    "StudyProposal",
    "StudyFactoryError",
    "build_real_n1_study_construction",
    "build_real_n1_study_bundle",
    "build_study_bundle",
    "propose_real_n1_study",
    "propose_real_n1_study_bundle",
    "study_bundle_fingerprint",
    "validate_real_n1_study_bundle",
    "validate_real_n1_study_bundle_integrity",
    "validate_real_n1_study_construction",
    "validate_real_n1_study_proposal",
    "validate_study_bundle",
]
