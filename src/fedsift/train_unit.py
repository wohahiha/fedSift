"""Fail-closed execution of one sealed FedSift training capability.

This module is deliberately a *unit* runner.  It executes one already frozen
HPO or outer-refit capability and never chooses a candidate, a budget, a
threshold, or an endpoint.  The caller must independently commit the
capability hash and the :class:`TrainingBudget` hash before execution.

The runner uses the registered local-training gate and server kernels.  It
does not provide a generic optimizer fallback.  Private runs are reported only
as conditional/unconditional record-level DP for this one fixed run; no hash
or receipt in this module turns a multi-candidate HPO study into the same
single-run privacy guarantee.
"""

from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import copy
import hashlib
import hmac
import json
import math
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from numbers import Integral, Real
from typing import Any, Mapping, Sequence
import numpy as np
import torch
from . import baseline_math
from .candidate_space import MAIN_METHODS, canonical_sha256
from .control_rule import (
    ControlDecisionScope,
    ControlEvidenceBinding,
    decide_fedsift,
    decide_public_argmin,
    validate_control_decision_receipt,
)
from .hpo_capability import (
    SCHEMA as HPO_CAPABILITY_SCHEMA,
    capability_candidate_parameters,
    materialize_capability_row_ids,
)
from .hpo_plan import MATCHED_ABLATIONS
from .local_training import (
    CorrectionPolicy,
    FedProxCorrection,
    LocalTrainingContext,
    NoCorrection,
    PoissonLocalTrainingGate,
    PoissonStepReceipt,
    ScaffoldCorrection,
    build_correction_source_binding,
    build_fedprox_correction,
    build_scaffold_correction,
    canonical_nonprivate_explicit_minibatches,
    poisson_step_receipt_fingerprint,
)
from .method_dispatch import (
    ALL_METHODS,
    MethodExecutionSpec,
    build_method_execution_spec,
    resolve_server_kernel,
    validate_method_execution_spec,
)
from .modeling import (
    InitializationDomain,
    MODEL_FAMILIES,
    SelectedBCEGradientBridge,
    build_fixed_model,
    extract_model_state,
    fixed_model_manifest,
    fixed_model_manifest_sha256,
    model_state_sha256,
    predict_probabilities,
    validate_model_state,
)
from .outer_refit import (
    REFIT_CAPABILITY_SCHEMA,
    materialize_outer_refit_row_ids,
    outer_refit_candidate_configuration,
)
from .preprocessing import PreprocessedRoles, _matrix_sha256, preprocessing_artifact_fingerprint
from .privacy_accounting import (
    ParallelCompositionConditions,
    ParallelRecordDPReport,
    NoiseCalibrationResult,
    PoissonDPStage,
    RecordDPReport,
    SequentialClientRunDPReport,
    account_poisson_dpsgd,
    calibrate_two_phase_noise,
    client_run_manifest_fingerprint,
    conditionally_compose_label_driven_record_partitions,
    parallel_compose_disjoint_record_partitions,
    sequential_compose_runtime_reports,
    validate_sequential_client_run_report,
)
from .resource_accounting import aggregate_resource_receipts, build_round_resource_receipt
from .sofim_math import (
    SofimState,
    sofim_aggregate_normalized_client_proxy,
    sofim_init_like,
    validate_sofim_normalized_proxy_receipt,
    validate_sofim_step_receipt,
)


class TrainingUnitError(RuntimeError):
    """Raised when one frozen training unit cannot be proved complete."""


class ForbiddenTrainingRoleAccess(TrainingUnitError):
    """Raised before an outer-test role can enter this runner."""


class TrainingUnitFailure(TrainingUnitError):
    """Atomic failure: no partial model or prediction artifact is returned."""

    def __init__(self, message: str, *, failure_receipt: Mapping[str, object]):
        super().__init__(message)
        self.failure_receipt = copy.deepcopy(dict(failure_receipt))


_DATA_FACTORY_SEAL = object()
_RESULT_FACTORY_SEAL = object()
_RESOURCE_RESULT_FACTORY_SEAL = object()
_CLIENT_NAMES = tuple((f"client_{index}" for index in range(5)))
_OUTER_ALIASES = frozenset({"outer_test", "outer-test", "test"})
_PARTITION_MODES = frozenset({"fixed_nonprivate", "fixed_label_driven_auxiliary_condition"})
_SINGLE_RUN_PRIVACY_CLAIM = "conditional_or_unconditional_record_level_dp_for_one_externally_fixed_training_run_only_not_the_full_hpo_or_research_release"


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise TrainingUnitError("value is not canonical JSON") from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value.lower() != value
        or any((character not in "0123456789abcdef" for character in value))
    ):
        raise TrainingUnitError(f"{name} must be a lowercase SHA-256")
    return value


def _identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any((character in value for character in "\r\n\t"))
    ):
        raise TrainingUnitError(f"{name} must be a plain non-empty identifier")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise TrainingUnitError(f"{name} must be a positive exact integer")
    return int(value)


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise TrainingUnitError(f"{name} must be a nonnegative exact integer")
    return int(value)


def _positive_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TrainingUnitError(f"{name} must be a finite positive real")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise TrainingUnitError(f"{name} must be a finite positive real")
    return result


def _seed(value: object, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result >= 2**63:
        raise TrainingUnitError(f"{name} must be smaller than 2**63")
    return result


def _state_copy(state: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    if not isinstance(state, Mapping) or not state:
        raise TrainingUnitError("tensor state must be a non-empty mapping")
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise TrainingUnitError("tensor state entry is malformed")
        if not value.dtype.is_floating_point or not bool(torch.isfinite(value).all()):
            raise TrainingUnitError("tensor state must be finite and floating point")
        result[name] = value.detach().clone()
    return result


def _tensor_state_sha256(state: Mapping[str, torch.Tensor], *, role: str) -> str:
    copied = _state_copy(state)
    return _sha256(
        {
            "schema": _identity("training_tensor_state"),
            "role": role,
            "parameters": [
                {
                    "name": name,
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "values_hex": [
                        float(item).hex() for item in value.detach().cpu().reshape(-1).tolist()
                    ],
                }
                for (name, value) in copied.items()
            ],
        }
    )


def _state_delta(
    newer: Mapping[str, torch.Tensor], older: Mapping[str, torch.Tensor]
) -> OrderedDict[str, torch.Tensor]:
    if tuple(newer) != tuple(older):
        raise TrainingUnitError("state parameter orders differ")
    return OrderedDict(
        ((name, newer[name].detach().clone() - older[name].detach().clone()) for name in older)
    )


def _state_add_scaled(
    base: Mapping[str, torch.Tensor], direction: Mapping[str, torch.Tensor], alpha: float
) -> OrderedDict[str, torch.Tensor]:
    if tuple(base) != tuple(direction):
        raise TrainingUnitError("state and direction parameter orders differ")
    result = OrderedDict(
        (
            (name, base[name].detach().clone() + alpha * direction[name].detach().clone())
            for name in base
        )
    )
    if any((not bool(torch.isfinite(value).all()) for value in result.values())):
        raise TrainingUnitError("scaled server update is non-finite")
    return result


def _zeros_like(state: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(((name, torch.zeros_like(value)) for (name, value) in state.items()))


def _weighted_delta(
    local_states: Sequence[Mapping[str, torch.Tensor]],
    base_state: Mapping[str, torch.Tensor],
    counts: Sequence[int],
) -> OrderedDict[str, torch.Tensor]:
    if not local_states or len(local_states) != len(counts):
        raise TrainingUnitError("local states and client counts do not align")
    total = float(sum((_positive_int(value, "client example count") for value in counts)))
    result = OrderedDict(((name, torch.zeros_like(value)) for (name, value) in base_state.items()))
    for local, count in zip(local_states, counts):
        delta = _state_delta(local, base_state)
        weight = float(count) / total
        for name in result:
            result[name] = result[name] + weight * delta[name]
    return result


@dataclass(frozen=True, slots=True)
class ExplicitClientBatchPlan:
    """One externally frozen non-private local-SGD batch plan."""

    client_id: str
    epochs: tuple[tuple[tuple[int, ...], ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "client_id", _identifier(self.client_id, "client_id"))
        epochs = tuple(self.epochs)
        if not epochs:
            raise TrainingUnitError("explicit non-private plan must contain an epoch")
        normalized: list[tuple[tuple[int, ...], ...]] = []
        for epoch in epochs:
            batches = tuple(epoch)
            if not batches:
                raise TrainingUnitError("each explicit epoch must contain a batch")
            normalized_batches: list[tuple[int, ...]] = []
            for batch in batches:
                values = tuple(batch)
                if not values or any((type(value) is not int or value < 0 for value in values)):
                    raise TrainingUnitError("explicit batch row IDs are invalid")
                if len(values) != len(set(values)):
                    raise TrainingUnitError("one explicit batch repeats a row ID")
                normalized_batches.append(values)
            normalized.append(tuple(normalized_batches))
        object.__setattr__(self, "epochs", tuple(normalized))

    @property
    def optimizer_steps(self) -> int:
        return sum((len(epoch) for epoch in self.epochs))


@dataclass(frozen=True, slots=True)
class TrainingBudget:
    """Performance-free, externally committed execution budget."""

    server_rounds: int
    dp_optimizer_steps_by_round: tuple[int, ...]
    nonprivate_batch_plans_by_round: tuple[tuple[ExplicitClientBatchPlan, ...], ...]
    participating_clients_by_round: tuple[tuple[str, ...], ...]
    poisson_sample_rate: float
    base_noise_multiplier: float
    clip_norm: float
    target_epsilon: float
    target_delta: float
    model_family: str
    initialization_seed: int
    partition_mode: str
    fixed_auxiliary_partition_condition_sha256: str | None = None
    budget_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        rounds = _positive_int(self.server_rounds, "server_rounds")
        object.__setattr__(self, "server_rounds", rounds)
        steps = tuple(
            (
                _positive_int(value, f"dp_optimizer_steps_by_round[{index}]")
                for (index, value) in enumerate(self.dp_optimizer_steps_by_round)
            )
        )
        if len(steps) != rounds:
            raise TrainingUnitError("DP step roster length differs from server rounds")
        object.__setattr__(self, "dp_optimizer_steps_by_round", steps)
        participants: list[tuple[str, ...]] = []
        for index, raw in enumerate(self.participating_clients_by_round):
            values = tuple((_identifier(value, "participating client") for value in raw))
            if not values or len(values) != len(set(values)):
                raise TrainingUnitError("participating clients must be non-empty and unique")
            participants.append(values)
        if len(participants) != rounds:
            raise TrainingUnitError("participation roster length differs from server rounds")
        if any((values != participants[0] for values in participants[1:])):
            raise TrainingUnitError(
                _identity(
                    "n1_unit_runner_requires_one_fixed_client_participation_roster_across_rounds"
                )
            )
        object.__setattr__(self, "participating_clients_by_round", tuple(participants))
        plans = tuple((tuple(value) for value in self.nonprivate_batch_plans_by_round))
        if len(plans) != rounds:
            raise TrainingUnitError("non-private plan roster length differs from rounds")
        for index, round_plans in enumerate(plans):
            if not all((type(value) is ExplicitClientBatchPlan for value in round_plans)):
                raise TrainingUnitError("non-private plan has an unrecognized type")
            ids = tuple((value.client_id for value in round_plans))
            if ids != participants[index]:
                raise TrainingUnitError(
                    "explicit non-private plans must follow the frozen participant order"
                )
        object.__setattr__(self, "nonprivate_batch_plans_by_round", plans)
        rate = _positive_real(self.poisson_sample_rate, "poisson_sample_rate")
        if rate > 1.0:
            raise TrainingUnitError("Poisson sample rate must not exceed one")
        object.__setattr__(self, "poisson_sample_rate", rate)
        object.__setattr__(
            self,
            "base_noise_multiplier",
            _positive_real(self.base_noise_multiplier, "base_noise_multiplier"),
        )
        object.__setattr__(self, "clip_norm", _positive_real(self.clip_norm, "clip_norm"))
        object.__setattr__(
            self, "target_epsilon", _positive_real(self.target_epsilon, "target_epsilon")
        )
        delta = _positive_real(self.target_delta, "target_delta")
        if delta >= 1.0:
            raise TrainingUnitError("target delta must be smaller than one")
        object.__setattr__(self, "target_delta", delta)
        if self.model_family not in MODEL_FAMILIES:
            raise TrainingUnitError(_identity("model_family_is_outside_the_frozen_n1_roster"))
        object.__setattr__(
            self, "initialization_seed", _seed(self.initialization_seed, "initialization_seed")
        )
        if self.partition_mode not in _PARTITION_MODES:
            raise TrainingUnitError("partition mode is not registered")
        condition_hash = self.fixed_auxiliary_partition_condition_sha256
        if self.partition_mode == "fixed_label_driven_auxiliary_condition":
            condition_hash = _require_sha256(
                condition_hash, "fixed auxiliary partition condition hash"
            )
        elif condition_hash is not None:
            raise TrainingUnitError(
                "a fixed non-private partition cannot carry a label condition hash"
            )
        object.__setattr__(self, "fixed_auxiliary_partition_condition_sha256", condition_hash)
        payload = {
            "schema": _identity("training_budget"),
            "server_rounds": self.server_rounds,
            "dp_optimizer_steps_by_round": self.dp_optimizer_steps_by_round,
            "nonprivate_batch_plans_by_round": tuple(
                (
                    tuple((asdict(plan) for plan in round_plans))
                    for round_plans in self.nonprivate_batch_plans_by_round
                )
            ),
            "participating_clients_by_round": self.participating_clients_by_round,
            "poisson_sample_rate": self.poisson_sample_rate,
            "base_noise_multiplier": self.base_noise_multiplier,
            "clip_norm": self.clip_norm,
            "target_epsilon": self.target_epsilon,
            "target_delta": self.target_delta,
            "model_family": self.model_family,
            "initialization_seed": self.initialization_seed,
            "partition_mode": self.partition_mode,
            "fixed_auxiliary_partition_condition_sha256": self.fixed_auxiliary_partition_condition_sha256,
        }
        object.__setattr__(self, "budget_sha256", _sha256(payload))


@dataclass(frozen=True, slots=True)
class FrozenRoleTable:
    role: str
    row_ids: tuple[int, ...]
    features: tuple[tuple[float, ...], ...]
    labels: tuple[int, ...]
    matrix_sha256: str
    labels_sha256: str
    table_sha256: str
    _factory_seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._factory_seal is not _DATA_FACTORY_SEAL:
            raise TrainingUnitError("role tables must be produced by the sealing factory")


@dataclass(frozen=True, slots=True)
class SealedUnitData:
    capability_sha256: str
    preprocessing_artifact_sha256: str
    roles: tuple[FrozenRoleTable, ...]
    data_sha256: str
    _factory_seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._factory_seal is not _DATA_FACTORY_SEAL:
            raise TrainingUnitError("unit data must be produced by the sealing factory")

    def role(self, name: str) -> FrozenRoleTable:
        matches = tuple((value for value in self.roles if value.role == name))
        if len(matches) != 1:
            raise TrainingUnitError("sealed unit data role is missing or duplicated")
        return matches[0]


@dataclass(frozen=True, slots=True)
class _CapabilityScope:
    kind: str
    capability_sha256: str
    study_id: str
    dataset_id: str
    method_id: str
    parent_method: str
    candidate_id: str
    candidate_sha256: str
    candidate_parameters_sha256: str
    candidate_parameters: dict[str, object]
    mechanism_switch: dict[str, object] | None
    outer_repeat: int
    outer_fold: int
    inner_fold: int
    seed_repeat: int
    max_server_rounds: int | None
    role_row_ids: tuple[tuple[str, tuple[int, ...]], ...]
    v_ctrl_membership_sha256: str


@dataclass(frozen=True, slots=True)
class ControlExecutionEvidence:
    scope: ControlDecisionScope
    binding: ControlEvidenceBinding
    row_ids: tuple[int, ...]
    labels: tuple[int, ...]
    expected_alphas: tuple[float, ...]
    candidate_table: tuple[dict[str, object], ...]
    rule_name: str
    safety_margin_z: float | None
    minimum_control_improvement: float | None
    receipt: dict[str, object]


@dataclass(frozen=True, slots=True)
class TrainingUnitResult:
    """In-memory model plus sealed, metric-free evidence artifacts."""

    model: torch.nn.Module
    model_state: OrderedDict[str, torch.Tensor]
    artifact: dict[str, object]
    dispatch_spec: MethodExecutionSpec
    round_resource_receipts: tuple[dict[str, object], ...]
    resource_summary: dict[str, object]
    local_step_receipts: tuple[PoissonStepReceipt, ...]
    sequential_client_reports: tuple[SequentialClientRunDPReport, ...]
    parallel_privacy_report: ParallelRecordDPReport | None
    control_evidence: tuple[ControlExecutionEvidence, ...]
    sofim_receipt_pairs: tuple[tuple[dict[str, object], dict[str, object]], ...]
    _factory_seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._factory_seal is not _RESULT_FACTORY_SEAL:
            raise TrainingUnitError("training results must be produced by the runner")


@dataclass(frozen=True, slots=True)
class ResourceOnlyTrainingResult:
    """Sanitized resource ledger from the same sealed training execution.

    This result deliberately cannot carry a trained state, validation output,
    local objective value, or public-query decision table.  The opaque audit
    commitment proves which sealed execution produced the resource ledger
    without releasing those excluded payloads.
    """

    round_resource_receipts: tuple[dict[str, object], ...]
    resource_summary: dict[str, object]
    audit_commitment: dict[str, object]
    _factory_seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._factory_seal is not _RESOURCE_RESULT_FACTORY_SEAL:
            raise TrainingUnitError("resource-only results must be produced by the runner")


def _rehash_without(value: Mapping[str, object], field_name: str) -> str:
    payload = copy.deepcopy(dict(value))
    payload.pop(field_name, None)
    return canonical_sha256(payload)


def _resolve_capability_scope(
    capability: Mapping[str, object], *, expected_capability_sha256: str
) -> _CapabilityScope:
    expected = _require_sha256(expected_capability_sha256, "expected capability hash")
    if not isinstance(capability, Mapping):
        raise TrainingUnitError("capability must be a mapping")
    schema = capability.get("schema")
    role_ids: list[tuple[str, tuple[int, ...]]] = []
    if schema == HPO_CAPABILITY_SCHEMA:
        for role in (*_CLIENT_NAMES, "v_ctrl", "v_sel", "inner_validation"):
            role_ids.append(
                (
                    role,
                    materialize_capability_row_ids(
                        capability, role=role, expected_capability_sha256=expected
                    ),
                )
            )
        identity = capability["unit_identity"]
        candidate = capability["candidate"]
        splits = capability["split_bindings"]
        assert isinstance(identity, Mapping)
        assert isinstance(candidate, Mapping)
        assert isinstance(splits, Mapping)
        method_id = str(identity["method"])
        parameters = capability_candidate_parameters(
            capability, expected_capability_sha256=expected
        )
        mechanism_switch = (
            copy.deepcopy(MATCHED_ABLATIONS[method_id]["mechanism_switch"])
            if method_id in MATCHED_ABLATIONS
            else None
        )
        parent = (
            str(MATCHED_ABLATIONS[method_id]["parent_method"])
            if method_id in MATCHED_ABLATIONS
            else method_id
        )
        scope = _CapabilityScope(
            kind="hpo",
            capability_sha256=expected,
            study_id=str(capability["study_id"]),
            dataset_id=str(capability["dataset_id"]),
            method_id=method_id,
            parent_method=parent,
            candidate_id=str(candidate["candidate_id"]),
            candidate_sha256=str(candidate["candidate_sha256"]),
            candidate_parameters_sha256=str(candidate["parameters_sha256"]),
            candidate_parameters=parameters,
            mechanism_switch=mechanism_switch,
            outer_repeat=int(identity["outer_repeat"]),
            outer_fold=int(identity["outer_fold"]),
            inner_fold=int(identity["inner_fold"]),
            seed_repeat=int(identity["hpo_seed"]),
            max_server_rounds=int(identity["max_steps"]),
            role_row_ids=tuple(role_ids),
            v_ctrl_membership_sha256=str(splits["v_ctrl_membership_sha256"]),
        )
    elif schema == REFIT_CAPABILITY_SCHEMA:
        for role in (*_CLIENT_NAMES, "v_ctrl", "v_sel"):
            role_ids.append(
                (
                    role,
                    materialize_outer_refit_row_ids(
                        capability, role=role, expected_capability_sha256=expected
                    ),
                )
            )
        configuration = outer_refit_candidate_configuration(
            capability, expected_capability_sha256=expected
        )
        candidate = capability["candidate"]
        raw_scope = capability["scope"]
        splits = capability["split_bindings"]
        assert isinstance(candidate, Mapping)
        assert isinstance(raw_scope, Mapping)
        assert isinstance(splits, Mapping)
        parameters = configuration.get("parameters")
        if not isinstance(parameters, Mapping):
            raise TrainingUnitError("outer-refit parameters are missing")
        switch = configuration.get("mechanism_switch")
        if switch is not None and (not isinstance(switch, Mapping)):
            raise TrainingUnitError("outer-refit mechanism switch is malformed")
        scope = _CapabilityScope(
            kind="outer_refit",
            capability_sha256=expected,
            study_id=str(capability["study_id"]),
            dataset_id=str(capability["dataset_id"]),
            method_id=str(configuration["method_identity"]),
            parent_method=str(configuration["parent_method"]),
            candidate_id=str(candidate["candidate_id"]),
            candidate_sha256=str(candidate["candidate_sha256"]),
            candidate_parameters_sha256=str(candidate["parameters_sha256"]),
            candidate_parameters=copy.deepcopy(dict(parameters)),
            mechanism_switch=None if switch is None else copy.deepcopy(dict(switch)),
            outer_repeat=int(raw_scope["outer_repeat"]),
            outer_fold=int(raw_scope["outer_fold"]),
            inner_fold=0,
            seed_repeat=int(raw_scope["eval_seed"]),
            max_server_rounds=None,
            role_row_ids=tuple(role_ids),
            v_ctrl_membership_sha256=str(splits["v_ctrl_membership_sha256"]),
        )
    else:
        raise TrainingUnitError("only sealed HPO or outer-refit capabilities are allowed")
    if capability.get("capability_sha256") != expected:
        raise TrainingUnitError("capability differs from its independent commitment")
    _identifier(scope.study_id, "study_id")
    _identifier(scope.dataset_id, "dataset_id")
    _identifier(scope.candidate_id, "candidate_id")
    _require_sha256(scope.candidate_sha256, "candidate_sha256")
    _require_sha256(scope.candidate_parameters_sha256, "parameters_sha256")
    if scope.candidate_parameters_sha256 != canonical_sha256(scope.candidate_parameters):
        raise TrainingUnitError("candidate parameter fingerprint differs")
    return scope


def materialize_training_role(
    capability: Mapping[str, object], *, role: str, expected_capability_sha256: str
) -> tuple[int, ...]:
    """Return an allowed role and reject outer-test aliases before lookup."""
    if not isinstance(role, str):
        raise TrainingUnitError("role must be a string")
    if role in _OUTER_ALIASES:
        raise ForbiddenTrainingRoleAccess(
            "the training-unit capability never authorizes outer-test access"
        )
    scope = _resolve_capability_scope(
        capability, expected_capability_sha256=expected_capability_sha256
    )
    roles = dict(scope.role_row_ids)
    if role not in roles:
        raise TrainingUnitError("role is absent from this capability")
    return roles[role]


def _validated_role_table(
    role: str,
    row_ids: Sequence[int],
    features: Sequence[Sequence[object]],
    labels: Sequence[object],
) -> FrozenRoleTable:
    ids = tuple(row_ids)
    raw_features = tuple((tuple(row) for row in features))
    raw_labels = tuple(labels)
    if not ids or len(raw_features) != len(ids) or len(raw_labels) != len(ids):
        raise TrainingUnitError(f"{role} role arrays are empty or misaligned")
    width = len(raw_features[0])
    if width <= 0 or any((len(row) != width for row in raw_features)):
        raise TrainingUnitError(f"{role} feature matrix width differs")
    matrix: list[tuple[float, ...]] = []
    for row in raw_features:
        converted = tuple((float(value) for value in row))
        if any((not math.isfinite(value) for value in converted)):
            raise TrainingUnitError(f"{role} feature matrix is non-finite")
        matrix.append(converted)
    label_values: list[int] = []
    for value in raw_labels:
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) not in (0, 1):
            raise TrainingUnitError(f"{role} labels must be exact binary integers")
        label_values.append(int(value))
    matrix_values = tuple(matrix)
    labels_values = tuple(label_values)
    matrix_hash = _matrix_sha256(role, ids, matrix_values)
    labels_hash = _sha256(
        {"schema": _identity("role_labels"), "role": role, "row_ids": ids, "labels": labels_values}
    )
    table_hash = _sha256(
        {
            "schema": _identity("frozen_role_table"),
            "role": role,
            "row_ids": ids,
            "matrix_sha256": matrix_hash,
            "labels_sha256": labels_hash,
        }
    )
    return FrozenRoleTable(
        role=role,
        row_ids=ids,
        features=matrix_values,
        labels=labels_values,
        matrix_sha256=matrix_hash,
        labels_sha256=labels_hash,
        table_sha256=table_hash,
        _factory_seal=_DATA_FACTORY_SEAL,
    )


def seal_preprocessed_unit_data(
    capability: Mapping[str, object],
    preprocessed: PreprocessedRoles,
    labels_by_row_id: Mapping[int, int],
    *,
    expected_capability_sha256: str,
) -> SealedUnitData:
    """Bind exact capability roles to one already produced preprocessing artifact.

    The artifact must self-hash and explicitly bind the capability hash.  The
    role matrices and their row order are checked against the artifact.  This
    function never accepts or materializes an outer-test role.
    """
    scope = _resolve_capability_scope(
        capability, expected_capability_sha256=expected_capability_sha256
    )
    if type(preprocessed) is not PreprocessedRoles:
        raise TrainingUnitError("preprocessed roles have an unrecognized type")
    artifact = preprocessed.artifact
    if not isinstance(artifact, Mapping):
        raise TrainingUnitError("preprocessing artifact is missing")
    stored = _require_sha256(artifact.get("artifact_sha256"), "preprocessing artifact hash")
    if not hmac.compare_digest(stored, preprocessing_artifact_fingerprint(artifact)):
        raise TrainingUnitError("preprocessing artifact fingerprint differs")
    input_binding = artifact.get("input_binding")
    if (
        not isinstance(input_binding, Mapping)
        or input_binding.get("capability_sha256") != scope.capability_sha256
    ):
        raise TrainingUnitError("preprocessing artifact is not bound to this capability")
    exact_roles = dict(scope.role_row_ids)
    if set(preprocessed.matrices) != set(exact_roles):
        raise TrainingUnitError("preprocessed role roster differs from capability")
    artifact_roles = artifact.get("roles")
    if not isinstance(artifact_roles, Mapping) or set(artifact_roles) != set(exact_roles):
        raise TrainingUnitError("preprocessing artifact role roster differs")
    if not isinstance(labels_by_row_id, Mapping):
        raise TrainingUnitError("label lookup must be a mapping")
    tables: list[FrozenRoleTable] = []
    for role, row_ids in scope.role_row_ids:
        try:
            labels = tuple((labels_by_row_id[row_id] for row_id in row_ids))
        except KeyError as exc:
            raise TrainingUnitError("one capability row is absent from the label lookup") from exc
        table = _validated_role_table(role, row_ids, preprocessed.matrices[role], labels)
        binding = artifact_roles[role]
        if not isinstance(binding, Mapping):
            raise TrainingUnitError("preprocessing role binding is malformed")
        if (
            binding.get("row_count") != len(row_ids)
            or binding.get("row_ids") != list(row_ids)
            or binding.get("matrix_sha256") != table.matrix_sha256
        ):
            raise TrainingUnitError("preprocessing role content differs from its artifact")
        tables.append(table)
    widths = {len(table.features[0]) for table in tables}
    if len(widths) != 1:
        raise TrainingUnitError("preprocessed role feature widths differ")
    data_hash = _sha256(
        {
            "schema": _identity("sealed_training_unit_data"),
            "capability_sha256": scope.capability_sha256,
            "preprocessing_artifact_sha256": stored,
            "role_table_sha256": [table.table_sha256 for table in tables],
        }
    )
    return SealedUnitData(
        capability_sha256=scope.capability_sha256,
        preprocessing_artifact_sha256=stored,
        roles=tuple(tables),
        data_sha256=data_hash,
        _factory_seal=_DATA_FACTORY_SEAL,
    )


def _validate_sealed_data(scope: _CapabilityScope, value: SealedUnitData) -> None:
    if type(value) is not SealedUnitData or value._factory_seal is not _DATA_FACTORY_SEAL:
        raise TrainingUnitError("training data are not factory sealed")
    if value.capability_sha256 != scope.capability_sha256:
        raise TrainingUnitError("training data are bound to another capability")
    expected_roles = scope.role_row_ids
    if tuple((table.role for table in value.roles)) != tuple(
        (role for (role, _) in expected_roles)
    ):
        raise TrainingUnitError("sealed data role order differs from capability")
    rebuilt: list[FrozenRoleTable] = []
    for table, (role, row_ids) in zip(value.roles, expected_roles):
        if table._factory_seal is not _DATA_FACTORY_SEAL or table.row_ids != row_ids:
            raise TrainingUnitError("sealed role membership differs from capability")
        rebuilt_table = _validated_role_table(role, row_ids, table.features, table.labels)
        if rebuilt_table != table:
            raise TrainingUnitError("sealed role table was modified after sealing")
        rebuilt.append(rebuilt_table)
    expected_hash = _sha256(
        {
            "schema": _identity("sealed_training_unit_data"),
            "capability_sha256": scope.capability_sha256,
            "preprocessing_artifact_sha256": _require_sha256(
                value.preprocessing_artifact_sha256, "preprocessing artifact hash"
            ),
            "role_table_sha256": [table.table_sha256 for table in rebuilt],
        }
    )
    if not hmac.compare_digest(value.data_sha256, expected_hash):
        raise TrainingUnitError("sealed unit-data hash differs")


def _budget_payload(value: TrainingBudget) -> dict[str, object]:
    return {
        "schema": _identity("training_budget"),
        "server_rounds": value.server_rounds,
        "dp_optimizer_steps_by_round": value.dp_optimizer_steps_by_round,
        "nonprivate_batch_plans_by_round": tuple(
            (
                tuple((asdict(plan) for plan in round_plans))
                for round_plans in value.nonprivate_batch_plans_by_round
            )
        ),
        "participating_clients_by_round": value.participating_clients_by_round,
        "poisson_sample_rate": value.poisson_sample_rate,
        "base_noise_multiplier": value.base_noise_multiplier,
        "clip_norm": value.clip_norm,
        "target_epsilon": value.target_epsilon,
        "target_delta": value.target_delta,
        "model_family": value.model_family,
        "initialization_seed": value.initialization_seed,
        "partition_mode": value.partition_mode,
        "fixed_auxiliary_partition_condition_sha256": value.fixed_auxiliary_partition_condition_sha256,
    }


def _validate_budget(
    budget: TrainingBudget,
    *,
    expected_budget_sha256: str,
    scope: _CapabilityScope,
    data: SealedUnitData,
) -> None:
    if type(budget) is not TrainingBudget:
        raise TrainingUnitError("budget has an unrecognized type")
    expected = _require_sha256(expected_budget_sha256, "expected budget hash")
    actual = _sha256(_budget_payload(budget))
    if budget.budget_sha256 != actual or not hmac.compare_digest(actual, expected):
        raise TrainingUnitError("training budget differs from its independent commitment")
    if scope.max_server_rounds is not None and budget.server_rounds != scope.max_server_rounds:
        raise TrainingUnitError("HPO server-round budget differs from sealed max_steps")
    available_clients = {table.role for table in data.roles if table.role in _CLIENT_NAMES}
    participants = set(budget.participating_clients_by_round[0])
    if not participants.issubset(available_clients):
        raise TrainingUnitError("budget references a client outside the capability")
    for round_plans in budget.nonprivate_batch_plans_by_round:
        for plan in round_plans:
            allowed = set(data.role(plan.client_id).row_ids)
            for epoch in plan.epochs:
                for batch in epoch:
                    if not set(batch).issubset(allowed):
                        raise TrainingUnitError(
                            "explicit non-private batch contains an unauthorized row"
                        )


def _full_private_schedule(
    spec: MethodExecutionSpec, parameters: Mapping[str, object], budget: TrainingBudget
) -> tuple[PoissonDPStage, ...]:
    total_steps = sum(budget.dp_optimizer_steps_by_round)
    if spec.privacy_schedule_handler == "uniform_candidate_specific_calibration":
        return (
            PoissonDPStage(
                "uniform", budget.poisson_sample_rate, budget.base_noise_multiplier, total_steps
            ),
        )
    if spec.privacy_schedule_handler != "two_phase_time_candidate_specific_calibration":
        raise TrainingUnitError("private dispatch has no registered privacy schedule")
    if total_steps < 2:
        raise TrainingUnitError("two-phase time schedule requires at least two total steps")
    raw = parameters.get("privacy_schedule")
    if not isinstance(raw, Mapping) or raw.get("kind") != "two_phase_time":
        raise TrainingUnitError("time schedule parameters are missing")
    fraction = _positive_real(raw.get("saving_round_fraction"), "saving fraction")
    if fraction >= 1.0:
        raise TrainingUnitError("saving fraction must be smaller than one")
    saving_rounds = int(math.ceil(budget.server_rounds * fraction))
    if saving_rounds <= 0 or saving_rounds >= budget.server_rounds:
        raise TrainingUnitError(
            "saving_round_fraction leaves no non-empty saving or spending round phase"
        )
    first_steps = sum(budget.dp_optimizer_steps_by_round[:saving_rounds])
    saving = _positive_real(raw.get("saving_sigma_factor"), "saving sigma factor")
    spending = _positive_real(raw.get("spending_sigma_factor"), "spending sigma factor")
    return (
        PoissonDPStage(
            "time_saving",
            budget.poisson_sample_rate,
            budget.base_noise_multiplier * saving,
            first_steps,
        ),
        PoissonDPStage(
            "time_spending",
            budget.poisson_sample_rate,
            budget.base_noise_multiplier * spending,
            total_steps - first_steps,
        ),
    )


def _calibration_geometry(
    spec: MethodExecutionSpec, parameters: Mapping[str, object], budget: TrainingBudget
) -> tuple[int, float, float]:
    """Return phase-one steps and factors without reading any result data."""
    total_steps = sum(budget.dp_optimizer_steps_by_round)
    if total_steps < 2:
        raise TrainingUnitError("minimal-noise calibration requires at least two optimizer steps")
    if spec.privacy_schedule_handler == "uniform_candidate_specific_calibration":
        first_steps = sum(budget.dp_optimizer_steps_by_round[:1]) if budget.server_rounds > 1 else 1
        if first_steps >= total_steps:
            first_steps = total_steps - 1
        return (first_steps, 1.0, 1.0)
    if spec.privacy_schedule_handler != "two_phase_time_candidate_specific_calibration":
        raise TrainingUnitError("private dispatch has no calibration geometry")
    schedule = parameters.get("privacy_schedule")
    if not isinstance(schedule, Mapping) or schedule.get("kind") != "two_phase_time":
        raise TrainingUnitError("time schedule parameters are missing")
    fraction = _positive_real(schedule.get("saving_round_fraction"), "saving fraction")
    if fraction >= 1.0:
        raise TrainingUnitError("saving fraction must be smaller than one")
    saving_rounds = int(math.ceil(budget.server_rounds * fraction))
    if saving_rounds <= 0 or saving_rounds >= budget.server_rounds:
        raise TrainingUnitError(
            "saving_round_fraction leaves no non-empty saving or spending round phase"
        )
    first_steps = sum(budget.dp_optimizer_steps_by_round[:saving_rounds])
    return (
        first_steps,
        _positive_real(schedule.get("saving_sigma_factor"), "saving sigma factor"),
        _positive_real(schedule.get("spending_sigma_factor"), "spending sigma factor"),
    )


@lru_cache(maxsize=256)
def _cached_minimal_noise_calibration(
    sample_rate: float,
    total_steps: int,
    phase_one_steps: int,
    phase_one_factor: float,
    phase_two_factor: float,
    target_epsilon: float,
    target_delta: float,
) -> NoiseCalibrationResult:
    return calibrate_two_phase_noise(
        sample_rate=sample_rate,
        total_steps=total_steps,
        phase_one_steps=phase_one_steps,
        phase_one_factor=phase_one_factor,
        phase_two_factor=phase_two_factor,
        target_epsilon=target_epsilon,
        delta=target_delta,
        eps_error=0.01,
        delta_error=target_delta / 1000.0,
        initial_noise_upper=1.0,
        maximum_noise_multiplier=1000000.0,
        noise_relative_tolerance=0.0001,
        max_bisection_iterations=80,
    )


def _require_minimal_noise_calibration(
    spec: MethodExecutionSpec, parameters: Mapping[str, object], budget: TrainingBudget
) -> tuple[NoiseCalibrationResult, dict[str, object]]:
    first_steps, first_factor, second_factor = _calibration_geometry(spec, parameters, budget)
    calibration = _cached_minimal_noise_calibration(
        budget.poisson_sample_rate,
        sum(budget.dp_optimizer_steps_by_round),
        first_steps,
        first_factor,
        second_factor,
        budget.target_epsilon,
        budget.target_delta,
    )
    calibrated_base = float(calibration.base_noise_multiplier)
    if budget.base_noise_multiplier.hex() != calibrated_base.hex():
        raise TrainingUnitError(
            "budget base noise differs from the result-blind minimal feasible calibration"
        )
    evidence: dict[str, object] = {
        "schema": _identity("training_unit_noise_calibration_evidence"),
        "candidate_parameters_sha256": spec.candidate_parameters_sha256,
        "privacy_schedule_handler": spec.privacy_schedule_handler,
        "sample_rate_hex": budget.poisson_sample_rate.hex(),
        "total_steps": sum(budget.dp_optimizer_steps_by_round),
        "phase_one_steps": first_steps,
        "phase_one_factor_hex": first_factor.hex(),
        "phase_two_factor_hex": second_factor.hex(),
        "target_epsilon_hex": budget.target_epsilon.hex(),
        "target_delta_hex": budget.target_delta.hex(),
        "lower_infeasible_bound_hex": float(calibration.lower_infeasible_bound).hex(),
        "upper_minimal_feasible_bound_hex": calibrated_base.hex(),
        "bisection_iterations": int(calibration.bisection_iterations),
        "bracket_expansions": int(calibration.bracket_expansions),
        "noise_relative_tolerance_hex": float(calibration.noise_relative_tolerance).hex(),
        "calibrated_epsilon_prv_upper_hex": float(calibration.report.epsilon_prv_upper).hex(),
        "performance_metrics_consumed": False,
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    return (calibration, evidence)


def _round_private_schedules(
    full: Sequence[PoissonDPStage], steps_by_round: Sequence[int]
) -> tuple[tuple[PoissonDPStage, ...], ...]:
    expanded = [
        (stage.phase, stage.sample_rate, stage.noise_multiplier)
        for stage in full
        for _ in range(stage.steps)
    ]
    schedules: list[tuple[PoissonDPStage, ...]] = []
    offset = 0
    for count in steps_by_round:
        segment = expanded[offset : offset + count]
        if len(segment) != count:
            raise TrainingUnitError("round schedule exceeds full private schedule")
        compressed: list[PoissonDPStage] = []
        for phase, rate, sigma in segment:
            if compressed and (
                compressed[-1].phase,
                compressed[-1].sample_rate,
                compressed[-1].noise_multiplier,
            ) == (phase, rate, sigma):
                compressed[-1] = replace(compressed[-1], steps=compressed[-1].steps + 1)
            else:
                compressed.append(PoissonDPStage(phase, rate, sigma, 1))
        schedules.append(tuple(compressed))
        offset += count
    if offset != len(expanded):
        raise TrainingUnitError("round schedules do not exhaust the private schedule")
    return tuple(schedules)


def _derive_seed(scope: _CapabilityScope, *, purpose: str, round_index: int, client_id: str) -> int:
    digest = _sha256(
        {
            "schema": _identity("training_unit_seed_domain"),
            "capability_sha256": scope.capability_sha256,
            "method_id": scope.method_id,
            "candidate_sha256": scope.candidate_sha256,
            "seed_repeat": scope.seed_repeat,
            "server_round": round_index,
            "client_id": client_id,
            "purpose": purpose,
        }
    )
    return int(digest[:16], 16) % 2**63


def _context(
    scope: _CapabilityScope, *, client_id: str, federated_round: int
) -> LocalTrainingContext:
    return LocalTrainingContext(
        study_id=scope.study_id,
        dataset_id=scope.dataset_id,
        outer_repeat=scope.outer_repeat,
        outer_fold=scope.outer_fold,
        inner_fold=scope.inner_fold,
        seed_repeat=scope.seed_repeat,
        client_id=client_id,
        federated_round=federated_round,
        method_id=scope.method_id,
        candidate_sha256=scope.candidate_sha256,
    )


def _correction_for_gate(
    *,
    spec: MethodExecutionSpec,
    scope: _CapabilityScope,
    bridge: SelectedBCEGradientBridge,
    round_global_state: Mapping[str, torch.Tensor],
    context: LocalTrainingContext,
    server_control: Mapping[str, torch.Tensor] | None,
    client_control: Mapping[str, torch.Tensor] | None,
    source_round: int,
) -> tuple[CorrectionPolicy, NoCorrection | FedProxCorrection | ScaffoldCorrection]:
    if spec.correction_handler == "none":
        return (CorrectionPolicy("none", scope.candidate_sha256, None, ()), NoCorrection())
    if spec.correction_handler == "fedprox_post_privacy_data_independent_correction":
        objective = scope.candidate_parameters.get("local_objective")
        if not isinstance(objective, Mapping):
            raise TrainingUnitError("FedProx objective is absent")
        mu = _positive_real(objective.get("prox_mu"), "FedProx mu")
        evidence = _sha256(
            {
                "schema": _identity("round_global_anchor_evidence"),
                "context_sha256": context.context_sha256,
                "round_global_state_sha256": model_state_sha256(bridge.model, round_global_state),
            }
        )
        source_context = _context(
            scope, client_id="server", federated_round=context.federated_round
        )
        binding = build_correction_source_binding(
            bridge,
            round_global_state,
            secondary_state=None,
            source_kind="round_global_anchor",
            source_evidence_sha256=evidence,
            source_context=source_context,
        )
        policy = CorrectionPolicy("fedprox", scope.candidate_sha256, mu, (binding,))
        correction = build_fedprox_correction(
            bridge, round_global_state, mu=mu, round_global_anchor_receipt_sha256=evidence
        )
        return (policy, correction)
    if spec.correction_handler != "scaffold_post_privacy_prior_state_correction":
        raise TrainingUnitError("dispatch correction handler is unknown")
    if server_control is None or client_control is None:
        raise TrainingUnitError("SCAFFOLD controls are missing")
    first = context.federated_round == 1
    provenance = "fixed_before_private_training" if first else "prior_dp_control_release"
    source_kind = "fixed_scaffold_controls" if first else "prior_dp_scaffold_controls"
    evidence = _sha256(
        {
            "schema": _identity("scaffold_control_source_evidence"),
            "method_id": scope.method_id,
            "candidate_sha256": scope.candidate_sha256,
            "client_id": context.client_id,
            "source_round": source_round,
            "server_control_sha256": _tensor_state_sha256(
                server_control, role="scaffold_server_control"
            ),
            "client_control_sha256": _tensor_state_sha256(
                client_control, role=f"scaffold_client_control:{context.client_id}"
            ),
        }
    )
    source_context = _context(
        scope,
        client_id=context.client_id,
        federated_round=context.federated_round if first else source_round,
    )
    binding = build_correction_source_binding(
        bridge,
        server_control,
        secondary_state=client_control,
        source_kind=source_kind,
        source_evidence_sha256=evidence,
        source_context=source_context,
    )
    policy = CorrectionPolicy("scaffold", scope.candidate_sha256, None, (binding,))
    correction = build_scaffold_correction(
        bridge,
        server_control,
        client_control,
        provenance_kind=provenance,
        source_evidence_sha256=evidence,
    )
    return (policy, correction)


def _torch_role(table: FrozenRoleTable) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(table.features, dtype=torch.float64),
        torch.tensor(table.labels, dtype=torch.float64),
        torch.tensor(table.row_ids, dtype=torch.int64),
    )


def _client_gradient_bridges(
    model: torch.nn.Module, data: SealedUnitData, client_ids: Sequence[str]
) -> OrderedDict[str, SelectedBCEGradientBridge]:
    """Bind each fixed participating client table exactly once per unit."""
    clients = tuple(client_ids)
    if not clients or len(clients) != len(set(clients)):
        raise TrainingUnitError("gradient-bridge client roster must be non-empty and unique")
    bridges: OrderedDict[str, SelectedBCEGradientBridge] = OrderedDict()
    for client_id in clients:
        table = data.role(client_id)
        if table.role != client_id or client_id not in _CLIENT_NAMES:
            raise TrainingUnitError("gradient-bridge role is not a canonical client")
        features, labels, row_ids = _torch_role(table)
        bridges[client_id] = SelectedBCEGradientBridge(model, features, labels, row_ids)
    return bridges


def _execute_nonprivate_client(
    *,
    bridge: SelectedBCEGradientBridge,
    start_state: Mapping[str, torch.Tensor],
    plan: ExplicitClientBatchPlan,
    learning_rate: float,
) -> tuple[OrderedDict[str, torch.Tensor], int, int]:

    def gradient_fn(
        state: Mapping[str, torch.Tensor], batch: Sequence[int]
    ) -> OrderedDict[str, torch.Tensor]:
        result = bridge.selected_gradient_batch(state, tuple(batch))
        if not result.selected_row_ids:
            raise TrainingUnitError("non-private explicit batch cannot be empty")
        return OrderedDict(
            ((name, value.mean(dim=0)) for (name, value) in result.gradients.items())
        )

    output = canonical_nonprivate_explicit_minibatches(
        start_state, plan.epochs, gradient_fn, learning_rate
    )
    evaluations = sum((len(batch) for epoch in plan.epochs for batch in epoch))
    return (output, plan.optimizer_steps, evaluations)


def _execute_private_client(
    *,
    model: torch.nn.Module,
    bridge: SelectedBCEGradientBridge,
    start_state: Mapping[str, torch.Tensor],
    table: FrozenRoleTable,
    schedule: Sequence[PoissonDPStage],
    budget: TrainingBudget,
    scope: _CapabilityScope,
    spec: MethodExecutionSpec,
    round_index: int,
    server_control: Mapping[str, torch.Tensor] | None,
    client_control: Mapping[str, torch.Tensor] | None,
) -> tuple[OrderedDict[str, torch.Tensor], RecordDPReport, tuple[PoissonStepReceipt, ...]]:
    if bridge.model is not model:
        raise TrainingUnitError("client gradient bridge is bound to a different model")
    context = _context(scope, client_id=table.role, federated_round=round_index)
    policy, correction = _correction_for_gate(
        spec=spec,
        scope=scope,
        bridge=bridge,
        round_global_state=start_state,
        context=context,
        server_control=server_control,
        client_control=client_control,
        source_round=max(1, round_index - 1),
    )
    state_hash = model_state_sha256(model, start_state)
    anchor_source = _sha256(
        {
            "schema": _identity("local_start_state_source"),
            "context_sha256": context.context_sha256,
            "state_sha256": state_hash,
        }
    )
    sampling_seed = _derive_seed(
        scope, purpose="poisson_sampling", round_index=round_index, client_id=table.role
    )
    noise_seed = _derive_seed(
        scope, purpose="gaussian_noise", round_index=round_index, client_id=table.role
    )
    if sampling_seed == noise_seed:
        raise TrainingUnitError("derived sampling and Gaussian seeds collide")
    optimizer = scope.candidate_parameters["local_optimizer"]
    assert isinstance(optimizer, Mapping)
    local_lr = _positive_real(optimizer.get("learning_rate"), "local learning rate")
    gate = PoissonLocalTrainingGate(
        tuple(schedule),
        gradient_bridge=bridge,
        population_row_ids=bridge.canonical_population_row_ids,
        clip_norm=budget.clip_norm,
        learning_rate=local_lr,
        delta=budget.target_delta,
        sampling_seed=sampling_seed,
        noise_seed=noise_seed,
        context=context,
        correction_policy=policy,
        initial_model_state_sha256=state_hash,
        initial_model_state_source_sha256=anchor_source,
        eps_error=0.01,
        delta_error=budget.target_delta / 1000.0,
    )
    state = _state_copy(start_state)
    receipts: list[PoissonStepReceipt] = []
    for step_index in range(1, gate.registered_steps + 1):
        result = gate.execute_poisson_optimizer_step(
            state, optimizer_step_index=step_index, correction=correction
        )
        gate.validate_step_result(result, state)
        receipts.append(result.receipt)
        state = result.updated_state
    report = gate.close(executed_optimizer_steps=len(receipts))
    return (state, report, tuple(receipts))


def _model_payload_bytes(state: Mapping[str, torch.Tensor]) -> int:
    result = sum((value.numel() * value.element_size() for value in state.values()))
    if result <= 0:
        raise TrainingUnitError("model payload is empty")
    return result


def _control_query_due(
    spec: MethodExecutionSpec, parameters: Mapping[str, object], round_index: int
) -> bool:
    if not spec.public_control_queries_enabled:
        return False
    control = parameters.get("control_rule")
    if not isinstance(control, Mapping):
        raise TrainingUnitError("public-control parameters are missing")
    cadence = _positive_int(control.get("query_every_rounds"), "control query cadence")
    return round_index % cadence == 0


def _execute_control_query(
    *,
    scope: _CapabilityScope,
    spec: MethodExecutionSpec,
    data: SealedUnitData,
    model: torch.nn.Module,
    base_state: Mapping[str, torch.Tensor],
    direction: Mapping[str, torch.Tensor],
    round_index: int,
    query_index: int,
) -> tuple[float, ControlExecutionEvidence]:
    control_parameters = scope.candidate_parameters.get("control_rule")
    if not isinstance(control_parameters, Mapping):
        raise TrainingUnitError("control-rule parameters are missing")
    alphas = tuple((float(value) for value in control_parameters["step_candidates"]))
    table = data.role("v_ctrl")
    features = np.asarray(table.features, dtype=np.float64)
    labels = np.asarray(table.labels, dtype=np.float64)

    def predict_fn(state: Mapping[str, torch.Tensor], values: np.ndarray) -> np.ndarray:
        tensor = torch.tensor(values, dtype=torch.float64)
        return predict_probabilities(model, tensor, state=state).detach().cpu().numpy()

    candidate_table = baseline_math.score_public_direction_grid(
        base_state, direction, features, labels, alphas=alphas, predict_fn=predict_fn
    )
    decision_scope = ControlDecisionScope(
        study_id=scope.study_id,
        dataset_id=scope.dataset_id,
        outer_repeat=scope.outer_repeat,
        outer_fold=scope.outer_fold,
        eval_seed=scope.seed_repeat,
        candidate_id=scope.candidate_id,
        server_round=round_index,
        query_index=query_index,
    )
    binding = ControlEvidenceBinding(
        control_membership_sha256=scope.v_ctrl_membership_sha256,
        preprocessing_artifact_sha256=data.preprocessing_artifact_sha256,
        model_manifest_sha256=fixed_model_manifest_sha256(model),
        candidate_parameters_sha256=scope.candidate_parameters_sha256,
        global_state_sha256=model_state_sha256(model, base_state),
        aggregate_direction_sha256=_tensor_state_sha256(
            direction, role="public_control_aggregate_direction"
        ),
    )
    if spec.public_control_handler == "decide_public_argmin":
        rule_name = "public_argmin_mean_logloss"
        safety = None
        minimum = None
        receipt = decide_public_argmin(
            scope=decision_scope,
            binding=binding,
            row_ids=table.row_ids,
            labels=table.labels,
            candidate_table=candidate_table,
            expected_alphas=alphas,
        )
    elif spec.public_control_handler == "decide_fedsift":
        sift = scope.candidate_parameters.get("sift")
        if not isinstance(sift, Mapping):
            raise TrainingUnitError("FedSift safety parameters are missing")
        rule_name = "fedsift_supported_override"
        safety = float(sift["safety_margin_z"])
        minimum = float(sift["minimum_control_improvement"])
        receipt = decide_fedsift(
            scope=decision_scope,
            binding=binding,
            row_ids=table.row_ids,
            labels=table.labels,
            candidate_table=candidate_table,
            expected_alphas=alphas,
            safety_margin_z=safety,
            minimum_control_improvement=minimum,
        )
    else:
        raise TrainingUnitError("query round has no registered control handler")
    validate_control_decision_receipt(
        receipt,
        expected_receipt_sha256=str(receipt["receipt_sha256"]),
        rule_name=rule_name,
        scope=decision_scope,
        binding=binding,
        row_ids=table.row_ids,
        labels=table.labels,
        candidate_table=candidate_table,
        expected_alphas=alphas,
        safety_margin_z=safety,
        minimum_control_improvement=minimum,
    )
    selected = float.fromhex(str(receipt["selected_alpha_hex"]))
    return (
        selected,
        ControlExecutionEvidence(
            scope=decision_scope,
            binding=binding,
            row_ids=table.row_ids,
            labels=table.labels,
            expected_alphas=alphas,
            candidate_table=tuple(copy.deepcopy(candidate_table)),
            rule_name=rule_name,
            safety_margin_z=safety,
            minimum_control_improvement=minimum,
            receipt=copy.deepcopy(receipt),
        ),
    )


def _prediction_payload(
    *, model: torch.nn.Module, state: Mapping[str, torch.Tensor], table: FrozenRoleTable
) -> dict[str, object]:
    probabilities = (
        predict_probabilities(model, torch.tensor(table.features, dtype=torch.float64), state=state)
        .detach()
        .cpu()
        .tolist()
    )
    payload: dict[str, object] = {
        "schema": _identity("raw_native_prediction_payload"),
        "role": table.role,
        "row_ids": list(table.row_ids),
        "labels": list(table.labels),
        "probabilities": [float(value) for value in probabilities],
        "raw_native_uncalibrated": True,
        "decision_threshold_applied": False,
        "model_state_sha256": model_state_sha256(model, state),
        "role_table_sha256": table.table_sha256,
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    return payload


def _parallel_conditions(mode: str) -> ParallelCompositionConditions:
    label_driven = mode == "fixed_label_driven_auxiliary_condition"
    return ParallelCompositionConditions(
        partitions_fixed_before_private_training=True,
        assignment_uses_no_unprotected_private_values=not label_driven,
        assignment_uses_private_labels_or_outcomes=label_driven,
        mechanisms_read_only_their_assigned_partition=True,
        records_never_migrate_or_recur_across_partitions=True,
        no_raw_record_cross_partition_state=True,
        client_rng_streams_domain_separated_and_nonreused=True,
        client_randomness_independent_or_joint_dp_proved=True,
    )


def _failure_receipt(
    *, stage: str, capability_sha256: object, budget_sha256: object, error: BaseException
) -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema": _identity("training_unit_failure"),
        "status": "failed_closed",
        "stage": stage,
        "capability_sha256": capability_sha256 if isinstance(capability_sha256, str) else None,
        "budget_sha256": budget_sha256 if isinstance(budget_sha256, str) else None,
        "error_type": type(error).__name__,
        "partial_model_released": False,
        "partial_prediction_released": False,
        "performance_metric_consumed": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def _execute_training_unit_core(
    capability: Mapping[str, object],
    sealed_data: SealedUnitData,
    budget: TrainingBudget,
    *,
    expected_capability_sha256: str,
    expected_budget_sha256: str,
    resource_only: bool,
) -> TrainingUnitResult | ResourceOnlyTrainingResult:
    """Execute one committed capability, with a single prediction-free exit."""
    stage = "preflight"
    try:
        scope = _resolve_capability_scope(
            capability, expected_capability_sha256=expected_capability_sha256
        )
        _validate_sealed_data(scope, sealed_data)
        _validate_budget(
            budget, expected_budget_sha256=expected_budget_sha256, scope=scope, data=sealed_data
        )
        if scope.method_id not in ALL_METHODS:
            raise TrainingUnitError("capability method is outside the registered roster")
        spec = build_method_execution_spec(
            method_id=scope.method_id,
            candidate_parameters=scope.candidate_parameters,
            mechanism_switch=scope.mechanism_switch,
        )
        validate_method_execution_spec(
            spec,
            method_id=scope.method_id,
            candidate_parameters=scope.candidate_parameters,
            mechanism_switch=scope.mechanism_switch,
            expected_dispatch_sha256=spec.dispatch_sha256,
        )
        kernel = resolve_server_kernel(spec)
        widths = {len(table.features[0]) for table in sealed_data.roles}
        if len(widths) != 1:
            raise TrainingUnitError("training roles have different feature widths")
        input_dim = next(iter(widths))
        model = build_fixed_model(
            budget.model_family,
            input_dim,
            initialization_seed=budget.initialization_seed,
            initialization_domain=InitializationDomain(
                study_id=scope.study_id,
                outer_repeat=scope.outer_repeat,
                outer_fold=scope.outer_fold,
                client_id="server",
                model_role="global_screening_model",
            ),
        )
        global_state = extract_model_state(model)
        model_manifest = fixed_model_manifest(model)
        model_manifest_hash = fixed_model_manifest_sha256(model)
        model_bytes = _model_payload_bytes(global_state)
        optimizer = scope.candidate_parameters.get("local_optimizer")
        backend = scope.candidate_parameters.get("backend")
        if not isinstance(optimizer, Mapping) or not isinstance(backend, Mapping):
            raise TrainingUnitError("candidate optimizer or backend is missing")
        local_lr = _positive_real(optimizer.get("learning_rate"), "local learning rate")
        full_schedule: tuple[PoissonDPStage, ...] | None = None
        round_schedules: tuple[tuple[PoissonDPStage, ...], ...] | None = None
        static_privacy_report: RecordDPReport | None = None
        noise_calibration_evidence: dict[str, object] | None = None
        if spec.privacy_mode != "nonprivate_explicit_minibatch":
            calibration, noise_calibration_evidence = _require_minimal_noise_calibration(
                spec, scope.candidate_parameters, budget
            )
            full_schedule = _full_private_schedule(spec, scope.candidate_parameters, budget)
            static_privacy_report = account_poisson_dpsgd(
                full_schedule,
                delta=budget.target_delta,
                eps_error=0.01,
                delta_error=budget.target_delta / 1000.0,
            )
            if static_privacy_report.epsilon_prv_upper > budget.target_epsilon:
                raise TrainingUnitError(
                    "frozen noise schedule exceeds the committed target epsilon"
                )
            if not math.isclose(
                static_privacy_report.epsilon_prv_upper,
                calibration.report.epsilon_prv_upper,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise TrainingUnitError(
                    "executed schedule differs from the minimal-noise calibration history"
                )
            round_schedules = _round_private_schedules(
                full_schedule, budget.dp_optimizer_steps_by_round
            )
        adam_state: baseline_math.FedAdamState | None = None
        yogi_state: baseline_math.FedYogiState | None = None
        sofim_state: SofimState | None = None
        scaffold_server_control: OrderedDict[str, torch.Tensor] | None = None
        scaffold_client_controls: dict[str, OrderedDict[str, torch.Tensor]] = {}
        if spec.server_backend_handler == "fedadam_server_step":
            tau = _positive_real(backend.get("tau"), "FedAdam tau")
            adam_state = baseline_math.fedadam_init_like(global_state, tau)
        elif spec.server_backend_handler == "fedyogi_server_step":
            tau = _positive_real(backend.get("tau"), "FedYogi tau")
            yogi_state = baseline_math.fedyogi_init_like(global_state, tau)
        elif spec.server_backend_handler == "sofim_server_step_with_receipt":
            sofim_state = sofim_init_like(global_state)
        elif spec.server_backend_handler == "scaffold_server_batched_weighted":
            scaffold_server_control = _zeros_like(global_state)
            scaffold_client_controls = {
                client: _zeros_like(global_state) for client in _CLIENT_NAMES
            }
        stage = "training_rounds"
        client_bridges = _client_gradient_bridges(
            model, sealed_data, budget.participating_clients_by_round[0]
        )
        runtime_reports: dict[str, list[RecordDPReport]] = {
            client: [] for client in budget.participating_clients_by_round[0]
        }
        expected_contexts: dict[str, dict[int, str]] = {
            client: {} for client in budget.participating_clients_by_round[0]
        }
        all_step_receipts: list[PoissonStepReceipt] = []
        resource_receipts: list[dict[str, object]] = []
        control_events: list[ControlExecutionEvidence] = []
        sofim_pairs: list[tuple[dict[str, object], dict[str, object]]] = []
        round_evidence: list[dict[str, object]] = []
        query_index = 0
        for round_index in range(1, budget.server_rounds + 1):
            base_state = _state_copy(global_state)
            participants = budget.participating_clients_by_round[round_index - 1]
            local_states: list[OrderedDict[str, torch.Tensor]] = []
            counts: list[int] = []
            round_step_receipts: list[PoissonStepReceipt] = []
            round_local_steps = 0
            round_gradient_evaluations = 0
            round_actual_steps: OrderedDict[str, int] = OrderedDict()
            new_scaffold_controls: list[OrderedDict[str, torch.Tensor]] = []
            old_scaffold_controls: list[OrderedDict[str, torch.Tensor]] = []
            for client_position, client_id in enumerate(participants):
                table = sealed_data.role(client_id)
                counts.append(len(table.row_ids))
                if spec.privacy_mode == "nonprivate_explicit_minibatch":
                    plan = budget.nonprivate_batch_plans_by_round[round_index - 1][client_position]
                    local, steps, evaluations = _execute_nonprivate_client(
                        bridge=client_bridges[client_id],
                        start_state=base_state,
                        plan=plan,
                        learning_rate=local_lr,
                    )
                else:
                    assert round_schedules is not None
                    local, report, receipts = _execute_private_client(
                        model=model,
                        bridge=client_bridges[client_id],
                        start_state=base_state,
                        table=table,
                        schedule=round_schedules[round_index - 1],
                        budget=budget,
                        scope=scope,
                        spec=spec,
                        round_index=round_index,
                        server_control=scaffold_server_control,
                        client_control=scaffold_client_controls.get(client_id),
                    )
                    runtime_reports[client_id].append(report)
                    expected_contexts[client_id][round_index] = _context(
                        scope, client_id=client_id, federated_round=round_index
                    ).context_sha256
                    round_step_receipts.extend(receipts)
                    steps = len(receipts)
                    evaluations = sum((value.sampled_record_count for value in receipts))
                local_states.append(local)
                round_local_steps += steps
                round_gradient_evaluations += evaluations
                round_actual_steps[client_id] = steps
                if spec.server_backend_handler == "scaffold_server_batched_weighted":
                    assert scaffold_server_control is not None
                    old_control = scaffold_client_controls[client_id]
                    new_control = baseline_math.scaffold_option2_control(
                        old_control,
                        scaffold_server_control,
                        base_state,
                        local,
                        actual_local_steps=steps,
                        local_learning_rate=local_lr,
                    )
                    old_scaffold_controls.append(old_control)
                    new_scaffold_controls.append(new_control)
            all_step_receipts.extend(round_step_receipts)
            input_state_hash = model_state_sha256(model, base_state)
            alpha = 1.0
            query_executed = False
            server_receipts: dict[str, object] = {}
            if spec.server_backend_handler == "fedavg_server_average":
                global_state = kernel(local_states, counts)
            elif spec.server_backend_handler == "scaffold_server_batched_weighted":
                assert scaffold_server_control is not None
                model_deltas = [_state_delta(value, base_state) for value in local_states]
                control_deltas = [
                    _state_delta(new, old)
                    for (new, old) in zip(new_scaffold_controls, old_scaffold_controls)
                ]
                global_state, scaffold_server_control = kernel(
                    base_state,
                    scaffold_server_control,
                    model_deltas,
                    control_deltas,
                    counts,
                    selected_client_count=len(participants),
                    total_client_count=len(_CLIENT_NAMES),
                    server_learning_rate=_positive_real(
                        backend.get("server_learning_rate"), "SCAFFOLD server learning rate"
                    ),
                )
                for client_id, control in zip(participants, new_scaffold_controls):
                    scaffold_client_controls[client_id] = control
            elif spec.server_backend_handler in {"fedadam_server_step", "fedyogi_server_step"}:
                aggregate_delta = _weighted_delta(local_states, base_state, counts)
                if spec.server_backend_handler == "fedadam_server_step":
                    assert adam_state is not None
                    proposed, adam_state = kernel(
                        base_state,
                        aggregate_delta,
                        adam_state,
                        beta1=float(backend["beta1"]),
                        beta2=float(backend["beta2"]),
                        server_learning_rate=float(backend["server_learning_rate"]),
                        tau=float(backend["tau"]),
                    )
                else:
                    assert yogi_state is not None
                    proposed, yogi_state = kernel(
                        base_state,
                        aggregate_delta,
                        yogi_state,
                        beta1=float(backend["beta1"]),
                        beta2=float(backend["beta2"]),
                        server_learning_rate=float(backend["server_learning_rate"]),
                        tau=float(backend["tau"]),
                    )
                direction = _state_delta(proposed, base_state)
                if _control_query_due(spec, scope.candidate_parameters, round_index):
                    query_index += 1
                    alpha, event = _execute_control_query(
                        scope=scope,
                        spec=spec,
                        data=sealed_data,
                        model=model,
                        base_state=base_state,
                        direction=direction,
                        round_index=round_index,
                        query_index=query_index,
                    )
                    control_events.append(event)
                    query_executed = True
                    server_receipts["control_decision_receipt_sha256"] = event.receipt[
                        "receipt_sha256"
                    ]
                    server_receipts["control_raw_candidate_table_sha256"] = canonical_sha256(
                        list(event.candidate_table)
                    )
                global_state = _state_add_scaled(base_state, direction, alpha)
            elif spec.server_backend_handler == "sofim_server_step_with_receipt":
                assert sofim_state is not None
                deltas = OrderedDict(
                    (
                        (client, _state_delta(local, base_state))
                        for (client, local) in zip(participants, local_states)
                    )
                )
                total = float(sum(counts))
                weights = OrderedDict(
                    (
                        (client, float(count) / total)
                        for (client, count) in zip(participants, counts)
                    )
                )
                learning_rates = OrderedDict(((client, local_lr) for client in participants))
                aggregate_proxy, proxy_receipt = sofim_aggregate_normalized_client_proxy(
                    expected_client_order=participants,
                    client_deltas=deltas,
                    actual_local_steps=round_actual_steps,
                    local_learning_rates=learning_rates,
                    aggregation_weights=weights,
                )
                global_state, sofim_state, _, step_receipt = kernel(
                    base_state,
                    aggregate_proxy,
                    sofim_state,
                    beta=float(backend["beta"]),
                    rho=float(backend["rho"]),
                    server_learning_rate=float(backend["server_learning_rate"]),
                    normalized_proxy_receipt=proxy_receipt,
                    expected_normalized_proxy_receipt_sha256=str(proxy_receipt["receipt_sha256"]),
                )
                sofim_pairs.append((copy.deepcopy(proxy_receipt), copy.deepcopy(step_receipt)))
                server_receipts.update(
                    {
                        "sofim_normalized_proxy_receipt_sha256": proxy_receipt["receipt_sha256"],
                        "sofim_step_receipt_sha256": step_receipt["receipt_sha256"],
                    }
                )
            else:
                raise TrainingUnitError("resolved server kernel has no execution branch")
            output_state_hash = model_state_sha256(model, global_state)
            control_candidate_count = 0
            control_record_count = 0
            if query_executed:
                latest = control_events[-1]
                control_candidate_count = len(latest.expected_alphas)
                control_record_count = len(latest.row_ids)
            resource = build_round_resource_receipt(
                method_id=scope.method_id,
                server_round=round_index,
                participating_client_ids=participants,
                total_client_count=len(_CLIENT_NAMES),
                model_payload_bytes=model_bytes,
                model_manifest_sha256=model_manifest_hash,
                local_optimizer_steps=round_local_steps,
                sampled_record_gradient_evaluations=round_gradient_evaluations,
                control_payload_bytes=(
                    model_bytes
                    if spec.server_backend_handler == "scaffold_server_batched_weighted"
                    else 0
                ),
                public_control_query_executed=query_executed,
                public_control_record_count=control_record_count,
                public_control_candidate_count=control_candidate_count,
            )
            resource_receipts.append(resource)
            round_evidence.append(
                {
                    "server_round": round_index,
                    "dispatch_sha256": spec.dispatch_sha256,
                    "server_kernel_handler": spec.server_backend_handler,
                    "resolved_kernel_name": kernel.__name__,
                    "input_model_state_sha256": input_state_hash,
                    "output_model_state_sha256": output_state_hash,
                    "selected_alpha_hex": alpha.hex(),
                    "local_optimizer_steps": round_local_steps,
                    "sampled_record_gradient_evaluations": round_gradient_evaluations,
                    "poisson_step_receipt_sha256": [
                        value.receipt_sha256 for value in round_step_receipts
                    ],
                    "correction_kinds": sorted(
                        {value.correction_kind for value in round_step_receipts}
                    ),
                    "server_receipts": server_receipts,
                    "resource_receipt_sha256": resource["receipt_sha256"],
                }
            )
        stage = "privacy_closure"
        sequential_reports: list[SequentialClientRunDPReport] = []
        parallel_report: ParallelRecordDPReport | None = None
        if spec.privacy_mode != "nonprivate_explicit_minibatch":
            client_reports: OrderedDict[str, SequentialClientRunDPReport] = OrderedDict()
            for client_id in budget.participating_clients_by_round[0]:
                contexts = expected_contexts[client_id]
                manifest_hash = client_run_manifest_fingerprint(
                    client_id=client_id,
                    method_id=scope.method_id,
                    candidate_sha256=scope.candidate_sha256,
                    population_row_ids_sha256=runtime_reports[client_id][0]
                    .execution_history[0]
                    .population_row_ids_sha256,
                    expected_round_context_sha256=contexts,
                )
                composed = sequential_compose_runtime_reports(
                    runtime_reports[client_id],
                    expected_round_context_sha256=contexts,
                    client_run_manifest_sha256=manifest_hash,
                    delta=budget.target_delta,
                    eps_error=0.01,
                    delta_error=budget.target_delta / 1000.0,
                )
                client_reports[client_id] = composed
                sequential_reports.append(composed)
            memberships = OrderedDict(
                ((client, sealed_data.role(client).row_ids) for client in client_reports)
            )
            conditions = _parallel_conditions(budget.partition_mode)
            if budget.partition_mode == "fixed_label_driven_auxiliary_condition":
                assert budget.fixed_auxiliary_partition_condition_sha256 is not None
                parallel_report = conditionally_compose_label_driven_record_partitions(
                    client_reports,
                    memberships,
                    conditions=conditions,
                    fixed_auxiliary_partition_condition_sha256=budget.fixed_auxiliary_partition_condition_sha256,
                )
            else:
                parallel_report = parallel_compose_disjoint_record_partitions(
                    client_reports, memberships, conditions=conditions
                )
            assert static_privacy_report is not None
            if parallel_report.epsilon > budget.target_epsilon:
                raise TrainingUnitError("runtime composed epsilon exceeds target")
        stage = "artifact_closure"
        resource_summary = aggregate_resource_receipts(
            resource_receipts,
            expected_method_id=scope.method_id,
            expected_rounds=budget.server_rounds,
            expected_model_manifest_sha256=model_manifest_hash,
        )
        privacy_artifact: dict[str, object]
        if parallel_report is None:
            privacy_artifact = {
                "status": "NOT_APPLICABLE_NONPRIVATE",
                "single_run_claim": None,
                "hpo_or_research_release_same_epsilon_claim": False,
            }
        else:
            assert static_privacy_report is not None
            privacy_artifact = {
                "status": "complete_single_fixed_run_only",
                "single_run_claim": _SINGLE_RUN_PRIVACY_CLAIM,
                "target_epsilon": budget.target_epsilon,
                "target_delta": budget.target_delta,
                "static_schedule_epsilon_prv_upper": static_privacy_report.epsilon_prv_upper,
                "minimal_noise_calibration": noise_calibration_evidence,
                "runtime_parallel_epsilon": parallel_report.epsilon,
                "runtime_parallel_delta": parallel_report.delta,
                "parallel_composition_rule": parallel_report.composition_rule,
                "conditioning_status": parallel_report.conditioning_status,
                "sequential_client_run_artifact_sha256": [
                    value.artifact_sha256 for value in sequential_reports
                ],
                "hpo_or_research_release_same_epsilon_claim": False,
                "secure_aggregation_claim": False,
                "client_level_dp_claim": False,
            }
        if resource_only:
            execution_commitment = canonical_sha256(
                {
                    "dispatch": spec.manifest(),
                    "round_evidence": round_evidence,
                    "privacy_evidence": privacy_artifact,
                    "local_step_receipt_sha256": [
                        value.receipt_sha256 for value in all_step_receipts
                    ],
                    "sequential_client_run_artifact_sha256": [
                        value.artifact_sha256 for value in sequential_reports
                    ],
                    "parallel_privacy_report": (
                        asdict(parallel_report) if parallel_report is not None else None
                    ),
                    "sofim_receipt_pairs": sofim_pairs,
                }
            )
            audit: dict[str, object] = {
                "schema": _identity("resource_only_training_result"),
                "status": "complete_resource_only",
                "capability_sha256": scope.capability_sha256,
                "budget_sha256": budget.budget_sha256,
                "sealed_data_sha256": sealed_data.data_sha256,
                "method_id": scope.method_id,
                "dispatch_sha256": spec.dispatch_sha256,
                "resource_report_sha256": resource_summary["report_sha256"],
                "accounting_commitment_sha256": canonical_sha256(privacy_artifact),
                "execution_commitment_sha256": execution_commitment,
                "restricted_payloads_absent": True,
            }
            audit["audit_sha256"] = canonical_sha256(audit)
            resource_result = ResourceOnlyTrainingResult(
                round_resource_receipts=tuple(copy.deepcopy(resource_receipts)),
                resource_summary=copy.deepcopy(resource_summary),
                audit_commitment=audit,
                _factory_seal=_RESOURCE_RESULT_FACTORY_SEAL,
            )
            validate_resource_only_training_result(
                resource_result,
                capability,
                sealed_data,
                budget,
                expected_capability_sha256=expected_capability_sha256,
                expected_budget_sha256=expected_budget_sha256,
            )
            return resource_result
        prediction_roles = ["v_sel"]
        if scope.kind == "hpo":
            prediction_roles.append("inner_validation")
        predictions = [
            _prediction_payload(model=model, state=global_state, table=sealed_data.role(role))
            for role in prediction_roles
        ]
        artifact: dict[str, object] = {
            "schema": _identity("training_unit_result"),
            "status": "complete_metric_free",
            "capability_kind": scope.kind,
            "capability_sha256": scope.capability_sha256,
            "budget_sha256": budget.budget_sha256,
            "sealed_data_sha256": sealed_data.data_sha256,
            "method_id": scope.method_id,
            "parent_method": scope.parent_method,
            "candidate_id": scope.candidate_id,
            "candidate_sha256": scope.candidate_sha256,
            "candidate_parameters_sha256": scope.candidate_parameters_sha256,
            "dispatch": spec.manifest(),
            "preprocessing_artifact_sha256": sealed_data.preprocessing_artifact_sha256,
            "model_manifest": model_manifest,
            "model_manifest_sha256": model_manifest_hash,
            "final_model_state_sha256": model_state_sha256(model, global_state),
            "round_evidence": round_evidence,
            "privacy_evidence": privacy_artifact,
            "resource_summary": copy.deepcopy(resource_summary),
            "raw_native_predictions": predictions,
            "evaluation_metrics_computed": False,
            "candidate_selection_performed": False,
            "outer_test_accessed": False,
        }
        artifact["artifact_sha256"] = canonical_sha256(artifact)
        result = TrainingUnitResult(
            model=model,
            model_state=validate_model_state(model, global_state),
            artifact=artifact,
            dispatch_spec=spec,
            round_resource_receipts=tuple(copy.deepcopy(resource_receipts)),
            resource_summary=copy.deepcopy(resource_summary),
            local_step_receipts=tuple(all_step_receipts),
            sequential_client_reports=tuple(sequential_reports),
            parallel_privacy_report=parallel_report,
            control_evidence=tuple(control_events),
            sofim_receipt_pairs=tuple(sofim_pairs),
            _factory_seal=_RESULT_FACTORY_SEAL,
        )
        validate_training_unit_result(
            result,
            capability,
            sealed_data,
            budget,
            expected_capability_sha256=expected_capability_sha256,
            expected_budget_sha256=expected_budget_sha256,
        )
        return result
    except TrainingUnitFailure:
        raise
    except Exception as exc:
        receipt = _failure_receipt(
            stage=stage,
            capability_sha256=(
                capability.get("capability_sha256") if isinstance(capability, Mapping) else None
            ),
            budget_sha256=budget.budget_sha256 if isinstance(budget, TrainingBudget) else None,
            error=exc,
        )
        raise TrainingUnitFailure(
            f"training unit failed closed during {stage}", failure_receipt=receipt
        ) from exc


def execute_training_unit(
    capability: Mapping[str, object],
    sealed_data: SealedUnitData,
    budget: TrainingBudget,
    *,
    expected_capability_sha256: str,
    expected_budget_sha256: str,
) -> TrainingUnitResult:
    """Execute one sealed unit and return its raw native validation outputs."""
    result = _execute_training_unit_core(
        capability,
        sealed_data,
        budget,
        expected_capability_sha256=expected_capability_sha256,
        expected_budget_sha256=expected_budget_sha256,
        resource_only=False,
    )
    if type(result) is not TrainingUnitResult:
        raise TrainingUnitError("full training path returned a resource-only result")
    return result


def execute_training_unit_resource_only(
    capability: Mapping[str, object],
    sealed_data: SealedUnitData,
    budget: TrainingBudget,
    *,
    expected_capability_sha256: str,
    expected_budget_sha256: str,
) -> ResourceOnlyTrainingResult:
    """Run the identical core but stop before validation-output inference."""
    result = _execute_training_unit_core(
        capability,
        sealed_data,
        budget,
        expected_capability_sha256=expected_capability_sha256,
        expected_budget_sha256=expected_budget_sha256,
        resource_only=True,
    )
    if type(result) is not ResourceOnlyTrainingResult:
        raise TrainingUnitError("resource-only path returned a full training result")
    return result


def validate_training_unit_result(
    result: TrainingUnitResult,
    capability: Mapping[str, object],
    sealed_data: SealedUnitData,
    budget: TrainingBudget,
    *,
    expected_capability_sha256: str,
    expected_budget_sha256: str,
) -> None:
    """Validate hashes, predictions, dispatch, receipts, and privacy closure."""
    if type(result) is not TrainingUnitResult or result._factory_seal is not _RESULT_FACTORY_SEAL:
        raise TrainingUnitError("training result has an unrecognized type")
    scope = _resolve_capability_scope(
        capability, expected_capability_sha256=expected_capability_sha256
    )
    _validate_sealed_data(scope, sealed_data)
    _validate_budget(
        budget, expected_budget_sha256=expected_budget_sha256, scope=scope, data=sealed_data
    )
    validate_method_execution_spec(
        result.dispatch_spec,
        method_id=scope.method_id,
        candidate_parameters=scope.candidate_parameters,
        mechanism_switch=scope.mechanism_switch,
        expected_dispatch_sha256=result.dispatch_spec.dispatch_sha256,
    )
    expected_noise_evidence: dict[str, object] | None = None
    if result.dispatch_spec.privacy_mode != "nonprivate_explicit_minibatch":
        _, expected_noise_evidence = _require_minimal_noise_calibration(
            result.dispatch_spec, scope.candidate_parameters, budget
        )
    state = validate_model_state(result.model, result.model_state)
    state_hash = model_state_sha256(result.model, state)
    artifact = result.artifact
    if not isinstance(artifact, Mapping):
        raise TrainingUnitError("training result artifact is missing")
    if artifact.get("artifact_sha256") != _rehash_without(artifact, "artifact_sha256"):
        raise TrainingUnitError("training result artifact hash differs")
    if (
        artifact.get("capability_sha256") != scope.capability_sha256
        or artifact.get("budget_sha256") != budget.budget_sha256
        or artifact.get("sealed_data_sha256") != sealed_data.data_sha256
        or (artifact.get("method_id") != scope.method_id)
        or (artifact.get("final_model_state_sha256") != state_hash)
        or (artifact.get("evaluation_metrics_computed") is not False)
        or (artifact.get("candidate_selection_performed") is not False)
        or (artifact.get("outer_test_accessed") is not False)
        or (artifact.get("dispatch") != result.dispatch_spec.manifest())
    ):
        raise TrainingUnitError("training result artifact identity differs")
    privacy_evidence = artifact.get("privacy_evidence")
    if not isinstance(privacy_evidence, Mapping):
        raise TrainingUnitError("training result privacy evidence is missing")
    if expected_noise_evidence is None:
        if privacy_evidence.get("status") != "NOT_APPLICABLE_NONPRIVATE":
            raise TrainingUnitError("non-private run has private schedule evidence")
    elif privacy_evidence.get("minimal_noise_calibration") != expected_noise_evidence:
        raise TrainingUnitError("minimal-noise calibration evidence differs")
    rebuilt_resource = aggregate_resource_receipts(
        result.round_resource_receipts,
        expected_method_id=scope.method_id,
        expected_rounds=budget.server_rounds,
        expected_model_manifest_sha256=fixed_model_manifest_sha256(result.model),
    )
    if (
        rebuilt_resource != result.resource_summary
        or artifact.get("resource_summary") != rebuilt_resource
    ):
        raise TrainingUnitError("resource evidence differs from round traces")
    for receipt in result.local_step_receipts:
        if poisson_step_receipt_fingerprint(receipt) != receipt.receipt_sha256:
            raise TrainingUnitError("local-step receipt fingerprint differs")
    for report in result.sequential_client_reports:
        validate_sequential_client_run_report(report)
    if scope.method_id == "fedavg_nonprivate":
        if result.sequential_client_reports or result.parallel_privacy_report is not None:
            raise TrainingUnitError("non-private run contains private accounting evidence")
    elif result.parallel_privacy_report is None:
        raise TrainingUnitError("private run lacks cross-client composition evidence")
    else:
        participant_order = budget.participating_clients_by_round[0]
        if (
            tuple((report.client_id for report in result.sequential_client_reports))
            != participant_order
        ):
            raise TrainingUnitError("sequential client report order differs")
        report_map = OrderedDict(
            ((report.client_id, report) for report in result.sequential_client_reports)
        )
        memberships = OrderedDict(
            ((client, sealed_data.role(client).row_ids) for client in participant_order)
        )
        conditions = _parallel_conditions(budget.partition_mode)
        if budget.partition_mode == "fixed_label_driven_auxiliary_condition":
            assert budget.fixed_auxiliary_partition_condition_sha256 is not None
            rebuilt_parallel = conditionally_compose_label_driven_record_partitions(
                report_map,
                memberships,
                conditions=conditions,
                fixed_auxiliary_partition_condition_sha256=budget.fixed_auxiliary_partition_condition_sha256,
            )
        else:
            rebuilt_parallel = parallel_compose_disjoint_record_partitions(
                report_map, memberships, conditions=conditions
            )
        if rebuilt_parallel != result.parallel_privacy_report:
            raise TrainingUnitError("parallel privacy composition evidence differs")
    rounds = artifact.get("round_evidence")
    if not isinstance(rounds, list) or len(rounds) != budget.server_rounds:
        raise TrainingUnitError("round evidence roster differs")
    recorded_step_hashes = [
        digest for row in rounds for digest in row.get("poisson_step_receipt_sha256", [])
    ]
    if recorded_step_hashes != [receipt.receipt_sha256 for receipt in result.local_step_receipts]:
        raise TrainingUnitError("round evidence omits or reorders local-step receipts")
    if [row.get("resource_receipt_sha256") for row in rounds] != [
        receipt["receipt_sha256"] for receipt in result.round_resource_receipts
    ]:
        raise TrainingUnitError("round evidence differs from resource receipts")
    recorded_control_hashes = [
        row["server_receipts"]["control_decision_receipt_sha256"]
        for row in rounds
        if "control_decision_receipt_sha256" in row.get("server_receipts", {})
    ]
    if recorded_control_hashes != [
        event.receipt["receipt_sha256"] for event in result.control_evidence
    ]:
        raise TrainingUnitError("round evidence differs from control receipts")
    for event in result.control_evidence:
        validate_control_decision_receipt(
            event.receipt,
            expected_receipt_sha256=str(event.receipt["receipt_sha256"]),
            rule_name=event.rule_name,
            scope=event.scope,
            binding=event.binding,
            row_ids=event.row_ids,
            labels=event.labels,
            candidate_table=event.candidate_table,
            expected_alphas=event.expected_alphas,
            safety_margin_z=event.safety_margin_z,
            minimum_control_improvement=event.minimum_control_improvement,
        )
    for proxy, step in result.sofim_receipt_pairs:
        proxy_hash = str(proxy["receipt_sha256"])
        validate_sofim_normalized_proxy_receipt(proxy, expected_receipt_sha256=proxy_hash)
        validate_sofim_step_receipt(
            step,
            expected_receipt_sha256=str(step["receipt_sha256"]),
            normalized_proxy_receipt=proxy,
            expected_normalized_proxy_receipt_sha256=proxy_hash,
        )
    prediction_roles = ["v_sel"] + (["inner_validation"] if scope.kind == "hpo" else [])
    expected_predictions = [
        _prediction_payload(model=result.model, state=state, table=sealed_data.role(role))
        for role in prediction_roles
    ]
    if artifact.get("raw_native_predictions") != expected_predictions:
        raise TrainingUnitError("raw-native predictions differ from final model state")


_RESOURCE_AUDIT_FIELDS = frozenset(
    {
        "schema",
        "status",
        "capability_sha256",
        "budget_sha256",
        "sealed_data_sha256",
        "method_id",
        "dispatch_sha256",
        "resource_report_sha256",
        "accounting_commitment_sha256",
        "execution_commitment_sha256",
        "restricted_payloads_absent",
        "audit_sha256",
    }
)
_RESOURCE_AUDIT_FORBIDDEN_TOKENS = (
    "alpha",
    "candidate_table",
    "control",
    "model",
    "prediction",
    "loss",
    "metric",
    "outer",
)


def _validate_resource_audit_payload(value: Mapping[str, object]) -> None:
    if set(value) != _RESOURCE_AUDIT_FIELDS:
        raise TrainingUnitError("resource-only audit commitment schema differs")
    serialized = _canonical_json(value).lower()
    if any((token in serialized for token in _RESOURCE_AUDIT_FORBIDDEN_TOKENS)):
        raise TrainingUnitError("resource-only audit commitment exposes a restricted payload")
    if value.get("audit_sha256") != _rehash_without(value, "audit_sha256"):
        raise TrainingUnitError("resource-only audit commitment hash differs")
    for field_name in (
        "capability_sha256",
        "budget_sha256",
        "sealed_data_sha256",
        "dispatch_sha256",
        "resource_report_sha256",
        "accounting_commitment_sha256",
        "execution_commitment_sha256",
        "audit_sha256",
    ):
        _require_sha256(value.get(field_name), field_name)


def validate_resource_only_training_result(
    result: ResourceOnlyTrainingResult,
    capability: Mapping[str, object],
    sealed_data: SealedUnitData,
    budget: TrainingBudget,
    *,
    expected_capability_sha256: str,
    expected_budget_sha256: str,
) -> None:
    """Rebuild the sanitized resource ledger without a trained-state input."""
    if (
        type(result) is not ResourceOnlyTrainingResult
        or result._factory_seal is not _RESOURCE_RESULT_FACTORY_SEAL
    ):
        raise TrainingUnitError("resource-only result has an unrecognized type")
    scope = _resolve_capability_scope(
        capability, expected_capability_sha256=expected_capability_sha256
    )
    _validate_sealed_data(scope, sealed_data)
    _validate_budget(
        budget, expected_budget_sha256=expected_budget_sha256, scope=scope, data=sealed_data
    )
    spec = build_method_execution_spec(
        method_id=scope.method_id,
        candidate_parameters=scope.candidate_parameters,
        mechanism_switch=scope.mechanism_switch,
    )
    validate_method_execution_spec(
        spec,
        method_id=scope.method_id,
        candidate_parameters=scope.candidate_parameters,
        mechanism_switch=scope.mechanism_switch,
        expected_dispatch_sha256=spec.dispatch_sha256,
    )
    if spec.privacy_mode != "nonprivate_explicit_minibatch":
        _require_minimal_noise_calibration(spec, scope.candidate_parameters, budget)
    summary = result.resource_summary
    if not isinstance(summary, Mapping):
        raise TrainingUnitError("resource-only summary is missing")
    manifest_hash = _require_sha256(summary.get("model_manifest_sha256"), "resource manifest hash")
    rebuilt = aggregate_resource_receipts(
        result.round_resource_receipts,
        expected_method_id=scope.method_id,
        expected_rounds=budget.server_rounds,
        expected_model_manifest_sha256=manifest_hash,
    )
    if rebuilt != summary:
        raise TrainingUnitError("resource-only summary differs from its round ledger")
    audit = result.audit_commitment
    if not isinstance(audit, Mapping):
        raise TrainingUnitError("resource-only audit commitment is missing")
    _validate_resource_audit_payload(audit)
    if (
        audit.get("schema") != _identity("resource_only_training_result")
        or audit.get("status") != "complete_resource_only"
        or audit.get("capability_sha256") != scope.capability_sha256
        or (audit.get("budget_sha256") != budget.budget_sha256)
        or (audit.get("sealed_data_sha256") != sealed_data.data_sha256)
        or (audit.get("method_id") != scope.method_id)
        or (audit.get("dispatch_sha256") != spec.dispatch_sha256)
        or (audit.get("resource_report_sha256") != summary.get("report_sha256"))
        or (audit.get("restricted_payloads_absent") is not True)
    ):
        raise TrainingUnitError("resource-only audit commitment identity differs")


__all__ = [
    "ControlExecutionEvidence",
    "ExplicitClientBatchPlan",
    "ForbiddenTrainingRoleAccess",
    "FrozenRoleTable",
    "SealedUnitData",
    "TrainingBudget",
    "TrainingUnitError",
    "TrainingUnitFailure",
    "TrainingUnitResult",
    "ResourceOnlyTrainingResult",
    "execute_training_unit",
    "execute_training_unit_resource_only",
    "materialize_training_role",
    "seal_preprocessed_unit_data",
    "validate_resource_only_training_result",
    "validate_training_unit_result",
]
