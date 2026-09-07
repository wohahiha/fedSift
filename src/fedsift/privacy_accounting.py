"""Strict record-level privacy accounting for FedSift.

This module accounts only the Poisson-subsampled Gaussian mechanisms used by
record-level DP-SGD.  Opacus 1.5.4's PRV accountant is the primary accountant.
The RDP accountant is advanced over the identical mechanism history as a
secondary finite bound; this does not by itself prove numerical agreement
between two accountants.  The API deliberately has no model-quality or test-set
inputs, so target-epsilon calibration is result blind by construction.

The guarantees returned here are *not* client-level DP or an end-to-end system
claim.  Parallel composition across clients is available only for a sealed
complete client-run report, concrete record-partition membership, and explicit
fixed/disjoint/randomness conditions.  Label-driven partitions have a separate
API whose bound is explicitly conditional on one frozen auxiliary partition;
it is never reported as end-to-end DP from raw benchmark records.
"""

from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import hashlib
import hmac
import json
import math
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import metadata
from numbers import Integral, Real
from typing import Any, Mapping, Sequence
import opacus
from opacus.accountants import PRVAccountant, RDPAccountant
from scipy.integrate import IntegrationWarning

EXPECTED_OPACUS_VERSION = "1.5.4"
PRIMARY_ACCOUNTANT = "opacus_prv"
SECONDARY_ACCOUNTANT = "opacus_rdp"
MECHANISM = "poisson_subsampled_gaussian_dp_sgd"
ADJACENCY = "add_or_remove_one_record"
RECORD_LEVEL_SCOPE = "record_level_dp_for_accounted_poisson_dpsgd_steps_only"
STATIC_REPORT_SCHEMA = _identity("record_dp_plan_report")
RUNTIME_REPORT_SCHEMA = _identity("record_dp_runtime_report")
STATIC_REPORT_KIND = "STATIC_PLAN_ACCOUNTING_NOT_EXECUTION_EVIDENCE"
RUNTIME_REPORT_KIND = "RUNTIME_HISTORY_WITH_STEP_EVIDENCE_REACCOUNTABLE"
CLIENT_RUN_REPORT_SCHEMA = _identity("sequential_client_run_dp_report")
CLIENT_RUN_REPORT_KIND = "COMPLETE_PREREGISTERED_CLIENT_RUN_SEQUENTIAL_COMPOSITION"
PARALLEL_REPORT_SCHEMA = _identity("parallel_record_dp_report")
RDP_EVIDENCE_STATUS = "same_history_secondary_finite_bound_only"
STATIC_EVIDENCE_CONFIDENTIALITY = "public_plan_accounting_no_execution_history"
RUNTIME_EVIDENCE_CONFIDENTIALITY = (
    "internal_audit_history_contains_private_commitments_do_not_publish"
)
EXCLUDED_CLAIMS = (
    "client_level_dp",
    "user_level_dp",
    "end_to_end_system_dp",
    "secure_aggregation",
    "privacy_of_unaccounted_outputs",
    "cryptographic_rng_security",
    "authenticated_execution_or_source_receipts",
    "frozen_candidate_catalog_fidelity",
)
_CLIENT_RUN_FACTORY_SEAL = object()


class PrivacyAccountingError(RuntimeError):
    """Raised when an FedSift privacy-accounting contract cannot be proved."""


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PrivacyAccountingError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise PrivacyAccountingError(f"{name} must be finite")
    return number


def _positive_real(value: object, name: str) -> float:
    number = _finite_real(value, name)
    if number <= 0.0:
        raise PrivacyAccountingError(f"{name} must be positive")
    return number


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise PrivacyAccountingError(f"{name} must be an integer")
    number = int(value)
    if number <= 0:
        raise PrivacyAccountingError(f"{name} must be positive")
    return number


def _lower_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any((character not in "0123456789abcdef" for character in value))
    ):
        raise PrivacyAccountingError(f"{name} must be a lowercase SHA-256")
    return value


def _plain_identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any((character in value for character in "\r\n\t"))
    ):
        raise PrivacyAccountingError(f"{name} must be a non-empty plain identifier")
    return value


def _nonnegative_exact_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise PrivacyAccountingError(f"{name} must be a nonnegative exact Python integer")
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise PrivacyAccountingError(f"{name} must be an integer")
    number = int(value)
    if number < 0:
        raise PrivacyAccountingError(f"{name} must be nonnegative")
    return number


def _sample_rate(value: object) -> float:
    rate = _positive_real(value, "sample rate")
    if rate > 1.0:
        raise PrivacyAccountingError("sample rate must be in (0, 1]")
    return rate


def _privacy_parameters(
    *, delta: object, eps_error: object, delta_error: object | None
) -> tuple[float, float, float]:
    delta_value = _positive_real(delta, "delta")
    if delta_value >= 1.0:
        raise PrivacyAccountingError("delta must be in (0, 1)")
    eps_error_value = _positive_real(eps_error, "epsilon error")
    if eps_error_value >= 1.0:
        raise PrivacyAccountingError("epsilon error must be in (0, 1)")
    if delta_error is None:
        delta_error_value = delta_value / 1000.0
        if delta_error_value == 0.0:
            raise PrivacyAccountingError(
                "default delta error underflowed; provide an explicit positive value"
            )
    else:
        delta_error_value = _positive_real(delta_error, "delta error")
    if delta_error_value >= delta_value:
        raise PrivacyAccountingError("delta error must be smaller than delta")
    return (delta_value, eps_error_value, delta_error_value)


def _require_fixed_opacus() -> None:
    try:
        distribution_version = metadata.version("opacus")
    except metadata.PackageNotFoundError as exc:
        raise PrivacyAccountingError("Opacus is not installed") from exc
    module_version = str(getattr(opacus, "__version__", ""))
    if distribution_version != EXPECTED_OPACUS_VERSION or module_version != EXPECTED_OPACUS_VERSION:
        raise PrivacyAccountingError(
            f"privacy accounting requires exactly Opacus {EXPECTED_OPACUS_VERSION}; observed distribution={distribution_version!r}, module={module_version!r}"
        )


@dataclass(frozen=True, slots=True)
class PoissonDPStage:
    """One contiguous segment of Poisson-subsampled Gaussian DP-SGD steps."""

    phase: str
    sample_rate: float
    noise_multiplier: float
    steps: int

    def __post_init__(self) -> None:
        if not isinstance(self.phase, str) or not self.phase or self.phase.strip() != self.phase:
            raise PrivacyAccountingError("phase must be a non-empty, whitespace-trimmed string")
        object.__setattr__(self, "sample_rate", _sample_rate(self.sample_rate))
        object.__setattr__(
            self, "noise_multiplier", _positive_real(self.noise_multiplier, "noise multiplier")
        )
        object.__setattr__(self, "steps", _positive_int(self.steps, "steps"))


@dataclass(frozen=True, slots=True)
class RuntimeMechanismStep:
    """Reconstructable evidence reference for one runtime accountant step.

    The evidence hash binds the completed mechanism/update event supplied by
    the execution gate.  The accountant validates and preserves the hash; it
    does not independently prove that the referenced computation occurred.
    """

    optimizer_step_index: int
    phase: str
    sample_rate: float
    noise_multiplier: float
    sampled_record_count: int
    gaussian_mechanism_executed: bool
    mechanism_evidence_sha256: str
    population_row_ids_sha256: str
    sampling_rng_stream_sha256: str
    noise_rng_stream_sha256: str
    local_training_context_sha256: str
    client_id: str
    federated_round: int
    method_id: str
    candidate_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "optimizer_step_index",
            _positive_int(self.optimizer_step_index, "optimizer step index"),
        )
        if not isinstance(self.phase, str) or not self.phase or self.phase.strip() != self.phase:
            raise PrivacyAccountingError(
                "runtime phase must be a non-empty, whitespace-trimmed string"
            )
        object.__setattr__(self, "sample_rate", _sample_rate(self.sample_rate))
        object.__setattr__(
            self, "noise_multiplier", _positive_real(self.noise_multiplier, "noise multiplier")
        )
        object.__setattr__(
            self,
            "sampled_record_count",
            _nonnegative_int(self.sampled_record_count, "sampled record count"),
        )
        if (
            type(self.gaussian_mechanism_executed) is not bool
            or not self.gaussian_mechanism_executed
        ):
            raise PrivacyAccountingError("runtime history requires an executed Gaussian mechanism")
        object.__setattr__(
            self,
            "mechanism_evidence_sha256",
            _lower_sha256(self.mechanism_evidence_sha256, "mechanism evidence hash"),
        )
        sampling_stream = _lower_sha256(self.sampling_rng_stream_sha256, "sampling RNG stream hash")
        noise_stream = _lower_sha256(self.noise_rng_stream_sha256, "noise RNG stream hash")
        if hmac.compare_digest(sampling_stream, noise_stream):
            raise PrivacyAccountingError("sampling and noise RNG streams collide")
        object.__setattr__(self, "sampling_rng_stream_sha256", sampling_stream)
        object.__setattr__(self, "noise_rng_stream_sha256", noise_stream)
        object.__setattr__(
            self,
            "population_row_ids_sha256",
            _lower_sha256(self.population_row_ids_sha256, "population row-ID hash"),
        )
        object.__setattr__(
            self,
            "local_training_context_sha256",
            _lower_sha256(self.local_training_context_sha256, "local-training context hash"),
        )
        object.__setattr__(self, "client_id", _plain_identifier(self.client_id, "client ID"))
        object.__setattr__(
            self, "federated_round", _nonnegative_exact_int(self.federated_round, "federated round")
        )
        object.__setattr__(self, "method_id", _plain_identifier(self.method_id, "method ID"))
        object.__setattr__(
            self, "candidate_sha256", _lower_sha256(self.candidate_sha256, "candidate hash")
        )


@dataclass(frozen=True, slots=True)
class RecordDPReport:
    """Plan-only or runtime record-level report with reconstructable history."""

    schema: str
    report_kind: str
    primary_accountant: str
    secondary_accountant: str
    opacus_version: str
    mechanism: str
    adjacency: str
    guarantee_scope: str
    epsilon_prv_upper: float
    epsilon_rdp: float
    delta: float
    eps_error: float
    delta_error: float
    accounted_steps: int
    registered_steps: int
    empty_sample_steps: int
    complete: bool
    accounted_schedule: tuple[PoissonDPStage, ...]
    registered_schedule: tuple[PoissonDPStage, ...]
    execution_history: tuple[RuntimeMechanismStep, ...]
    execution_history_sha256: str
    accounted_schedule_sha256: str
    registered_schedule_sha256: str
    rdp_evidence_status: str
    evidence_confidentiality: str
    performance_metrics_consumed: bool
    claims_not_made: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SequentialClientRunDPReport:
    """Factory-sealed composition of every preregistered round for one client.

    A complete ``RecordDPReport`` closes only one local gate.  This distinct
    type closes the higher-level client-run boundary and is the only runtime
    report accepted by cross-client parallel composition.
    """

    schema: str
    report_kind: str
    client_id: str
    method_id: str
    candidate_sha256: str
    population_row_ids_sha256: str
    expected_federated_rounds: tuple[int, ...]
    expected_round_context_sha256: tuple[tuple[int, str], ...]
    client_run_manifest_sha256: str
    component_report_sha256: tuple[str, ...]
    component_execution_history_sha256: tuple[str, ...]
    component_step_counts: tuple[int, ...]
    composed_report: RecordDPReport
    artifact_sha256: str
    _factory_seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._factory_seal is not _CLIENT_RUN_FACTORY_SEAL:
            raise PrivacyAccountingError(
                "sequential client-run report was not produced by the composition gate"
            )


@dataclass(frozen=True, slots=True)
class NoiseCalibrationResult:
    """Result-blind base-noise calibration for a fixed two-phase schedule."""

    schema: str
    target_epsilon: float
    base_noise_multiplier: float
    phase_one_factor: float
    phase_two_factor: float
    lower_infeasible_bound: float
    upper_feasible_bound: float
    bisection_iterations: int
    bracket_expansions: int
    noise_relative_tolerance: float
    report: RecordDPReport
    result_blind: bool
    performance_metrics_consumed: bool


@dataclass(frozen=True, slots=True)
class ParallelCompositionConditions:
    """External facts required before record-level parallel composition."""

    partitions_fixed_before_private_training: bool
    assignment_uses_no_unprotected_private_values: bool
    assignment_uses_private_labels_or_outcomes: bool
    mechanisms_read_only_their_assigned_partition: bool
    records_never_migrate_or_recur_across_partitions: bool
    no_raw_record_cross_partition_state: bool
    client_rng_streams_domain_separated_and_nonreused: bool
    client_randomness_independent_or_joint_dp_proved: bool


@dataclass(frozen=True, slots=True)
class ParallelRecordDPReport:
    """Max composition after caller attestations and membership checks."""

    schema: str
    epsilon: float
    delta: float
    client_count: int
    client_names: tuple[str, ...]
    partition_membership_sha256: str
    composition_rule: str
    guarantee_scope: str
    adjacency: str
    conditions_attested_and_membership_checked: bool
    conditioning_status: str
    fixed_auxiliary_partition_condition_sha256: str | None
    claims_not_made: tuple[str, ...]


def _validated_schedule(schedule: Sequence[PoissonDPStage]) -> tuple[PoissonDPStage, ...]:
    if isinstance(schedule, (str, bytes)) or not isinstance(schedule, Sequence):
        raise PrivacyAccountingError("schedule must be a sequence of PoissonDPStage")
    stages = tuple(schedule)
    if not stages or not all((isinstance(stage, PoissonDPStage) for stage in stages)):
        raise PrivacyAccountingError("schedule must contain one or more PoissonDPStage values")
    return stages


def _schedule_payload(schedule: Sequence[PoissonDPStage]) -> list[dict[str, Any]]:
    return [
        {
            "phase": stage.phase,
            "sample_rate": stage.sample_rate,
            "noise_multiplier": stage.noise_multiplier,
            "steps": stage.steps,
        }
        for stage in schedule
    ]


def schedule_fingerprint(schedule: Sequence[PoissonDPStage]) -> str:
    """Hash all schedule fields using canonical finite JSON."""
    stages = _validated_schedule(schedule)
    payload = json.dumps(
        _schedule_payload(stages),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _runtime_history_payload(history: Sequence[RuntimeMechanismStep]) -> list[dict[str, Any]]:
    return [
        {
            "optimizer_step_index": step.optimizer_step_index,
            "phase": step.phase,
            "sample_rate": step.sample_rate,
            "noise_multiplier": step.noise_multiplier,
            "sampled_record_count": step.sampled_record_count,
            "gaussian_mechanism_executed": step.gaussian_mechanism_executed,
            "mechanism_evidence_sha256": step.mechanism_evidence_sha256,
            "population_row_ids_sha256": step.population_row_ids_sha256,
            "sampling_rng_stream_sha256": step.sampling_rng_stream_sha256,
            "noise_rng_stream_sha256": step.noise_rng_stream_sha256,
            "local_training_context_sha256": step.local_training_context_sha256,
            "client_id": step.client_id,
            "federated_round": step.federated_round,
            "method_id": step.method_id,
            "candidate_sha256": step.candidate_sha256,
        }
        for step in history
    ]


def _runtime_history_fingerprint(history: Sequence[RuntimeMechanismStep]) -> str:
    payload = json.dumps(
        _runtime_history_payload(history),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validated_expected_round_contexts(
    expected_round_context_sha256: Mapping[int, str],
) -> tuple[tuple[int, str], ...]:
    if not isinstance(expected_round_context_sha256, Mapping) or not expected_round_context_sha256:
        raise PrivacyAccountingError("expected round contexts must be a non-empty mapping")
    values = tuple(
        sorted(
            (
                (
                    _nonnegative_exact_int(round_index, "expected federated round"),
                    _lower_sha256(context_hash, "expected round context hash"),
                )
                for (round_index, context_hash) in expected_round_context_sha256.items()
            )
        )
    )
    rounds = tuple((round_index for (round_index, _) in values))
    if len(rounds) != len(set(rounds)):
        raise PrivacyAccountingError("expected federated rounds are duplicated")
    if rounds != tuple(range(rounds[0], rounds[0] + len(rounds))):
        raise PrivacyAccountingError("expected federated rounds must be contiguous with no gap")
    return values


def client_run_manifest_fingerprint(
    *,
    client_id: str,
    method_id: str,
    candidate_sha256: str,
    population_row_ids_sha256: str,
    expected_round_context_sha256: Mapping[int, str],
) -> str:
    """Fingerprint the preregistered identity and complete round roster."""
    contexts = _validated_expected_round_contexts(expected_round_context_sha256)
    return _canonical_sha256(
        {
            "schema": _identity("client_run_manifest"),
            "client_id": _plain_identifier(client_id, "client ID"),
            "method_id": _plain_identifier(method_id, "method ID"),
            "candidate_sha256": _lower_sha256(candidate_sha256, "candidate hash"),
            "population_row_ids_sha256": _lower_sha256(
                population_row_ids_sha256, "population row-ID hash"
            ),
            "expected_round_context_sha256": contexts,
        }
    )


def _runtime_report_fingerprint(report: RecordDPReport) -> str:
    if type(report) is not RecordDPReport:
        raise PrivacyAccountingError("runtime report has the wrong exact type")
    return _canonical_sha256(
        {
            "schema": _identity("runtime_report_fingerprint"),
            "report_schema": report.schema,
            "report_kind": report.report_kind,
            "primary_accountant": report.primary_accountant,
            "secondary_accountant": report.secondary_accountant,
            "opacus_version": report.opacus_version,
            "mechanism": report.mechanism,
            "adjacency": report.adjacency,
            "guarantee_scope": report.guarantee_scope,
            "epsilon_prv_upper_hex": float(report.epsilon_prv_upper).hex(),
            "epsilon_rdp_hex": float(report.epsilon_rdp).hex(),
            "delta_hex": float(report.delta).hex(),
            "eps_error_hex": float(report.eps_error).hex(),
            "delta_error_hex": float(report.delta_error).hex(),
            "accounted_steps": report.accounted_steps,
            "registered_steps": report.registered_steps,
            "empty_sample_steps": report.empty_sample_steps,
            "complete": report.complete,
            "execution_history_sha256": report.execution_history_sha256,
            "accounted_schedule_sha256": report.accounted_schedule_sha256,
            "registered_schedule_sha256": report.registered_schedule_sha256,
            "rdp_evidence_status": report.rdp_evidence_status,
            "evidence_confidentiality": report.evidence_confidentiality,
            "performance_metrics_consumed": report.performance_metrics_consumed,
            "claims_not_made": report.claims_not_made,
        }
    )


def _validated_runtime_history(
    history: Sequence[RuntimeMechanismStep], accounted_schedule: Sequence[PoissonDPStage]
) -> tuple[RuntimeMechanismStep, ...]:
    if isinstance(history, (str, bytes)) or not isinstance(history, Sequence):
        raise PrivacyAccountingError("runtime execution history must be a sequence")
    steps = tuple(history)
    if not all((isinstance(step, RuntimeMechanismStep) for step in steps)):
        raise PrivacyAccountingError(
            "runtime execution history must contain RuntimeMechanismStep values"
        )
    expected = tuple(
        ((phase, rate, noise) for (_, phase, rate, noise) in _expanded_schedule(accounted_schedule))
    )
    observed = tuple(((step.phase, step.sample_rate, step.noise_multiplier) for step in steps))
    if len(steps) != len(expected) or observed != expected:
        raise PrivacyAccountingError(
            "runtime execution history differs from the accounted schedule"
        )
    if tuple((step.optimizer_step_index for step in steps)) != tuple(range(1, len(steps) + 1)):
        raise PrivacyAccountingError(
            "runtime execution history indices are not contiguous from one"
        )
    return steps


def _compressed_history(schedule: Sequence[PoissonDPStage]) -> list[tuple[float, float, int]]:
    history: list[tuple[float, float, int]] = []
    for stage in schedule:
        item = (stage.noise_multiplier, stage.sample_rate, stage.steps)
        if history and history[-1][:2] == item[:2]:
            previous = history[-1]
            history[-1] = (previous[0], previous[1], previous[2] + item[2])
        else:
            history.append(item)
    return history


def _advance_accountants(
    prv: PRVAccountant, rdp: RDPAccountant, schedule: Sequence[PoissonDPStage]
) -> None:
    for stage in schedule:
        for _ in range(stage.steps):
            prv.step(noise_multiplier=stage.noise_multiplier, sample_rate=stage.sample_rate)
            rdp.step(noise_multiplier=stage.noise_multiplier, sample_rate=stage.sample_rate)


AccountingScheduleCacheKey = tuple[tuple[str, float, float, int], ...]
AccountingValueCacheEntry = tuple[float, float, str]


def _accounting_schedule_cache_key(
    schedule: Sequence[PoissonDPStage],
) -> AccountingScheduleCacheKey:
    """Bind the complete phase-aware compressed accounting geometry.

    Opacus itself compresses adjacent equal ``(sigma, q)`` mechanisms and does
    not retain a phase label.  The cache is deliberately stricter: adjacent
    stages are merged only when phase, q, and sigma all agree.  Consequently a
    phase-semantic change cannot reuse an otherwise numerically equal entry.
    """
    stages = _validated_schedule(schedule)
    result: list[tuple[str, float, float, int]] = []
    for stage in stages:
        item = (stage.phase, stage.sample_rate, stage.noise_multiplier, stage.steps)
        if result and result[-1][:3] == item[:3]:
            previous = result[-1]
            result[-1] = (previous[0], previous[1], previous[2], previous[3] + item[3])
        else:
            result.append(item)
    return tuple(result)


def _solve_accountant_values(
    prv: PRVAccountant,
    rdp: RDPAccountant,
    schedule: Sequence[PoissonDPStage],
    *,
    delta: float,
    eps_error: float,
    delta_error: float,
) -> AccountingValueCacheEntry:
    """Perform one uncached Opacus numerical solve for validated accountants."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Optimal order is the .* alpha.*")
            warnings.filterwarnings("error", category=IntegrationWarning)
            epsilon_prv = float(
                prv.get_epsilon(delta=delta, eps_error=eps_error, delta_error=delta_error)
            )
        maximum_order = 256
        while True:
            alphas = (
                [1.0 + value / 10.0 for value in range(1, 100)]
                + list(range(12, 65))
                + list(range(72, maximum_order + 1, 8))
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Optimal order is the .* alpha.*")
                epsilon_rdp_value, best_alpha = rdp.get_privacy_spent(delta=delta, alphas=alphas)
            epsilon_rdp = float(epsilon_rdp_value)
            if float(best_alpha) < float(maximum_order):
                break
            if maximum_order >= 8192:
                raise PrivacyAccountingError("RDP optimum remained on the largest registered order")
            maximum_order *= 2
    except IntegrationWarning as exc:
        raise PrivacyAccountingError(
            "PRV numerical integration emitted IntegrationWarning"
        ) from exc
    except PrivacyAccountingError:
        raise
    except Exception as exc:
        raise PrivacyAccountingError("Opacus privacy accounting failed") from exc
    if (
        not math.isfinite(epsilon_prv)
        or epsilon_prv < 0.0
        or (not math.isfinite(epsilon_rdp))
        or (epsilon_rdp < 0.0)
    ):
        raise PrivacyAccountingError("Opacus returned a non-finite privacy bound")
    return (epsilon_prv, epsilon_rdp, RDP_EVIDENCE_STATUS)


@lru_cache(maxsize=2048)
def _cached_accountant_values(
    schedule_key: AccountingScheduleCacheKey,
    delta: float,
    eps_error: float,
    delta_error: float,
    opacus_version: str,
) -> AccountingValueCacheEntry:
    """Solve one public accounting geometry and cache numerical scalars only."""
    if opacus_version != EXPECTED_OPACUS_VERSION:
        raise PrivacyAccountingError("accounting cache key has an unexpected Opacus version")
    _require_fixed_opacus()
    if not schedule_key:
        raise PrivacyAccountingError("accounting cache key has an empty schedule")
    schedule = tuple(
        (
            PoissonDPStage(
                phase=phase, sample_rate=sample_rate, noise_multiplier=noise_multiplier, steps=steps
            )
            for (phase, sample_rate, noise_multiplier, steps) in schedule_key
        )
    )
    privacy_delta, privacy_eps_error, privacy_delta_error = _privacy_parameters(
        delta=delta, eps_error=eps_error, delta_error=delta_error
    )
    prv = PRVAccountant()
    rdp = RDPAccountant()
    _advance_accountants(prv, rdp, schedule)
    return _solve_accountant_values(
        prv,
        rdp,
        schedule,
        delta=privacy_delta,
        eps_error=privacy_eps_error,
        delta_error=privacy_delta_error,
    )


def clear_accountant_value_cache() -> None:
    """Clear only the process-local public-geometry numerical cache."""
    _cached_accountant_values.cache_clear()


def accountant_value_cache_info() -> object:
    """Return hit/miss/current-size counters without exposing cached values."""
    return _cached_accountant_values.cache_info()


def _accountant_values(
    prv: PRVAccountant,
    rdp: RDPAccountant,
    schedule: Sequence[PoissonDPStage],
    *,
    delta: float,
    eps_error: float,
    delta_error: float,
) -> AccountingValueCacheEntry:
    expected_history = _compressed_history(schedule)
    if list(prv.history) != expected_history or list(rdp.history) != expected_history:
        raise PrivacyAccountingError(
            "PRV and RDP accountants do not contain the registered identical history"
        )
    if not expected_history:
        return (0.0, 0.0, "zero_step_identity_no_secondary_comparison")
    values = _cached_accountant_values(
        _accounting_schedule_cache_key(schedule),
        delta,
        eps_error,
        delta_error,
        EXPECTED_OPACUS_VERSION,
    )
    if (
        type(values) is not tuple
        or len(values) != 3
        or (not math.isfinite(values[0]))
        or (values[0] < 0.0)
        or (not math.isfinite(values[1]))
        or (values[1] < 0.0)
        or (values[2] != RDP_EVIDENCE_STATUS)
    ):
        raise PrivacyAccountingError("accounting numerical cache entry is invalid")
    return values


def _report_from_accountants(
    prv: PRVAccountant,
    rdp: RDPAccountant,
    accounted_schedule: Sequence[PoissonDPStage],
    registered_schedule: Sequence[PoissonDPStage],
    *,
    delta: float,
    eps_error: float,
    delta_error: float,
    empty_sample_steps: int,
    complete: bool,
    report_kind: str,
    execution_history: Sequence[RuntimeMechanismStep],
) -> RecordDPReport:
    accounted = tuple(accounted_schedule)
    if not all((isinstance(stage, PoissonDPStage) for stage in accounted)):
        raise PrivacyAccountingError("accounted schedule must contain only PoissonDPStage values")
    registered = _validated_schedule(registered_schedule)
    accounted_steps = sum((stage.steps for stage in accounted))
    registered_steps = sum((stage.steps for stage in registered))
    empty_steps = _nonnegative_int(empty_sample_steps, "empty sample steps")
    if empty_steps > accounted_steps:
        raise PrivacyAccountingError("empty sample steps exceed accounted steps")
    if complete != (accounted_steps == registered_steps):
        raise PrivacyAccountingError("report completion flag differs from schedule prefix")
    history = tuple(execution_history)
    if report_kind == STATIC_REPORT_KIND:
        schema = STATIC_REPORT_SCHEMA
        evidence_confidentiality = STATIC_EVIDENCE_CONFIDENTIALITY
        if accounted != registered or not complete or empty_steps != 0 or history:
            raise PrivacyAccountingError(
                "static plan accounting cannot contain runtime execution evidence"
            )
    elif report_kind == RUNTIME_REPORT_KIND:
        schema = RUNTIME_REPORT_SCHEMA
        evidence_confidentiality = RUNTIME_EVIDENCE_CONFIDENTIALITY
        history = _validated_runtime_history(history, accounted)
        if empty_steps != sum((step.sampled_record_count == 0 for step in history)):
            raise PrivacyAccountingError(
                "runtime empty-sample count differs from execution history"
            )
    else:
        raise PrivacyAccountingError("record-DP report kind is not registered")
    epsilon_prv, epsilon_rdp, rdp_evidence = _accountant_values(
        prv, rdp, accounted, delta=delta, eps_error=eps_error, delta_error=delta_error
    )
    empty_fingerprint = hashlib.sha256(b"[]").hexdigest()
    return RecordDPReport(
        schema=schema,
        report_kind=report_kind,
        primary_accountant=PRIMARY_ACCOUNTANT,
        secondary_accountant=SECONDARY_ACCOUNTANT,
        opacus_version=EXPECTED_OPACUS_VERSION,
        mechanism=MECHANISM,
        adjacency=ADJACENCY,
        guarantee_scope=RECORD_LEVEL_SCOPE,
        epsilon_prv_upper=epsilon_prv,
        epsilon_rdp=epsilon_rdp,
        delta=delta,
        eps_error=eps_error,
        delta_error=delta_error,
        accounted_steps=accounted_steps,
        registered_steps=registered_steps,
        empty_sample_steps=empty_steps,
        complete=complete,
        accounted_schedule=accounted,
        registered_schedule=registered,
        execution_history=history,
        execution_history_sha256=_runtime_history_fingerprint(history),
        accounted_schedule_sha256=(
            schedule_fingerprint(accounted) if accounted else empty_fingerprint
        ),
        registered_schedule_sha256=schedule_fingerprint(registered),
        rdp_evidence_status=rdp_evidence,
        evidence_confidentiality=evidence_confidentiality,
        performance_metrics_consumed=False,
        claims_not_made=EXCLUDED_CLAIMS,
    )


def account_poisson_dpsgd(
    schedule: Sequence[PoissonDPStage],
    *,
    delta: float,
    eps_error: float = 0.01,
    delta_error: float | None = None,
) -> RecordDPReport:
    """Account a complete heterogeneous Poisson DP-SGD schedule.

    Every scheduled step is composed, including a step whose realized Poisson
    sample is empty.  Realized batch sizes do not alter the registered sample
    rate and therefore are intentionally not accepted by this static API.
    """
    _require_fixed_opacus()
    stages = _validated_schedule(schedule)
    delta_value, eps_error_value, delta_error_value = _privacy_parameters(
        delta=delta, eps_error=eps_error, delta_error=delta_error
    )
    prv = PRVAccountant()
    rdp = RDPAccountant()
    _advance_accountants(prv, rdp, stages)
    return _report_from_accountants(
        prv,
        rdp,
        stages,
        stages,
        delta=delta_value,
        eps_error=eps_error_value,
        delta_error=delta_error_value,
        empty_sample_steps=0,
        complete=True,
        report_kind=STATIC_REPORT_KIND,
        execution_history=(),
    )


def make_two_phase_schedule(
    *,
    sample_rate: float,
    base_noise_multiplier: float,
    total_steps: int,
    phase_one_steps: int,
    phase_one_factor: float,
    phase_two_factor: float,
) -> tuple[PoissonDPStage, PoissonDPStage]:
    """Create two non-empty phases whose factors multiply one base sigma."""
    rate = _sample_rate(sample_rate)
    base = _positive_real(base_noise_multiplier, "base noise multiplier")
    total = _positive_int(total_steps, "total steps")
    first = _positive_int(phase_one_steps, "phase one steps")
    if first >= total:
        raise PrivacyAccountingError("phase one steps must be smaller than total steps")
    first_factor = _positive_real(phase_one_factor, "phase one factor")
    second_factor = _positive_real(phase_two_factor, "phase two factor")
    return (
        PoissonDPStage(
            phase="phase_one", sample_rate=rate, noise_multiplier=base * first_factor, steps=first
        ),
        PoissonDPStage(
            phase="phase_two",
            sample_rate=rate,
            noise_multiplier=base * second_factor,
            steps=total - first,
        ),
    )


def calibrate_two_phase_noise(
    *,
    sample_rate: float,
    total_steps: int,
    phase_one_steps: int,
    phase_one_factor: float,
    phase_two_factor: float,
    target_epsilon: float,
    delta: float,
    eps_error: float = 0.01,
    delta_error: float | None = None,
    initial_noise_upper: float = 1.0,
    maximum_noise_multiplier: float = 1000000.0,
    noise_relative_tolerance: float = 0.0001,
    max_bisection_iterations: int = 80,
) -> NoiseCalibrationResult:
    """Calibrate the smallest bracketed base sigma meeting the PRV target.

    Search inputs are only mechanism, schedule and privacy-target parameters.
    No loss, accuracy, validation or test metric can enter this function.
    The returned upper bracket is feasible under the PRV upper bound.
    """
    _require_fixed_opacus()
    rate = _sample_rate(sample_rate)
    total = _positive_int(total_steps, "total steps")
    first = _positive_int(phase_one_steps, "phase one steps")
    if first >= total:
        raise PrivacyAccountingError("phase one steps must be smaller than total steps")
    first_factor = _positive_real(phase_one_factor, "phase one factor")
    second_factor = _positive_real(phase_two_factor, "phase two factor")
    target = _positive_real(target_epsilon, "target epsilon")
    delta_value, eps_error_value, delta_error_value = _privacy_parameters(
        delta=delta, eps_error=eps_error, delta_error=delta_error
    )
    upper = _positive_real(initial_noise_upper, "initial noise upper")
    maximum = _positive_real(maximum_noise_multiplier, "maximum noise multiplier")
    if upper > maximum:
        raise PrivacyAccountingError("initial noise upper exceeds maximum noise multiplier")
    relative_tolerance = _positive_real(noise_relative_tolerance, "noise relative tolerance")
    if relative_tolerance >= 1.0:
        raise PrivacyAccountingError("noise relative tolerance must be in (0, 1)")
    max_iterations = _positive_int(max_bisection_iterations, "maximum bisection iterations")

    def report_for(base: float) -> RecordDPReport:
        return account_poisson_dpsgd(
            make_two_phase_schedule(
                sample_rate=rate,
                base_noise_multiplier=base,
                total_steps=total,
                phase_one_steps=first,
                phase_one_factor=first_factor,
                phase_two_factor=second_factor,
            ),
            delta=delta_value,
            eps_error=eps_error_value,
            delta_error=delta_error_value,
        )

    expansions = 0
    upper_report = report_for(upper)
    while upper_report.epsilon_prv_upper > target:
        if upper >= maximum:
            raise PrivacyAccountingError(
                "target epsilon is not feasible inside the registered noise bound"
            )
        upper = min(maximum, upper * 2.0)
        expansions += 1
        upper_report = report_for(upper)
    lower = 0.0
    iterations = 0
    while iterations < max_iterations:
        width = upper - lower
        if width <= relative_tolerance * upper:
            break
        candidate = (lower + upper) / 2.0
        if candidate <= 0.0:
            raise PrivacyAccountingError("noise calibration produced a nonpositive candidate")
        candidate_report = report_for(candidate)
        iterations += 1
        if candidate_report.epsilon_prv_upper <= target:
            upper = candidate
            upper_report = candidate_report
        else:
            lower = candidate
    if upper - lower > relative_tolerance * upper:
        raise PrivacyAccountingError(
            "noise calibration did not converge within the registered iteration limit"
        )
    if upper_report.epsilon_prv_upper > target:
        raise PrivacyAccountingError("calibrated PRV upper bound exceeds target epsilon")
    if lower <= 0.0 or report_for(lower).epsilon_prv_upper <= target:
        raise PrivacyAccountingError(
            "noise calibration did not establish a positive infeasible lower bound"
        )
    return NoiseCalibrationResult(
        schema=_identity("two_phase_noise_calibration"),
        target_epsilon=target,
        base_noise_multiplier=upper,
        phase_one_factor=first_factor,
        phase_two_factor=second_factor,
        lower_infeasible_bound=lower,
        upper_feasible_bound=upper,
        bisection_iterations=iterations,
        bracket_expansions=expansions,
        noise_relative_tolerance=relative_tolerance,
        report=upper_report,
        result_blind=True,
        performance_metrics_consumed=False,
    )


def _expanded_schedule(
    schedule: Sequence[PoissonDPStage],
) -> tuple[tuple[int, str, float, float], ...]:
    return tuple(
        (
            (stage_index, stage.phase, stage.sample_rate, stage.noise_multiplier)
            for (stage_index, stage) in enumerate(schedule)
            for _ in range(stage.steps)
        )
    )


def _prefix_schedule(
    expanded: Sequence[tuple[int, str, float, float]], steps: int
) -> tuple[PoissonDPStage, ...]:
    prefix = expanded[:steps]
    stages: list[PoissonDPStage] = []
    previous_stage_index: int | None = None
    for stage_index, phase, rate, noise in prefix:
        if (
            stages
            and previous_stage_index == stage_index
            and (stages[-1].phase == phase)
            and (stages[-1].sample_rate == rate)
            and (stages[-1].noise_multiplier == noise)
        ):
            previous = stages[-1]
            stages[-1] = PoissonDPStage(
                phase=previous.phase,
                sample_rate=previous.sample_rate,
                noise_multiplier=previous.noise_multiplier,
                steps=previous.steps + 1,
            )
        else:
            stages.append(
                PoissonDPStage(phase=phase, sample_rate=rate, noise_multiplier=noise, steps=1)
            )
        previous_stage_index = stage_index
    return tuple(stages)


class RuntimePrefixAccountant:
    """Advance accounting exactly once for every executed optimizer step."""

    def __init__(
        self,
        schedule: Sequence[PoissonDPStage],
        *,
        delta: float,
        eps_error: float = 0.01,
        delta_error: float | None = None,
    ) -> None:
        _require_fixed_opacus()
        self._schedule = _validated_schedule(schedule)
        self._expanded = _expanded_schedule(self._schedule)
        self._delta, self._eps_error, self._delta_error = _privacy_parameters(
            delta=delta, eps_error=eps_error, delta_error=delta_error
        )
        self._prv = PRVAccountant()
        self._rdp = RDPAccountant()
        self._recorded_steps = 0
        self._empty_sample_steps = 0
        self._execution_history: list[RuntimeMechanismStep] = []
        self._closed = False
        self._poisoned = False

    @property
    def recorded_steps(self) -> int:
        return self._recorded_steps

    @property
    def registered_steps(self) -> int:
        return len(self._expanded)

    @property
    def registered_schedule_sha256(self) -> str:
        return schedule_fingerprint(self._schedule)

    @property
    def accounted_schedule_sha256(self) -> str:
        if self._recorded_steps <= 0:
            return hashlib.sha256(b"[]").hexdigest()
        return schedule_fingerprint(_prefix_schedule(self._expanded, self._recorded_steps))

    @property
    def execution_history_sha256(self) -> str:
        return _runtime_history_fingerprint(tuple(self._execution_history))

    def record_optimizer_step(
        self,
        *,
        optimizer_step_index: int,
        sample_rate: float,
        noise_multiplier: float,
        sampled_record_count: int,
        gaussian_mechanism_executed: bool,
        mechanism_evidence_sha256: str,
        population_row_ids_sha256: str,
        sampling_rng_stream_sha256: str,
        noise_rng_stream_sha256: str,
        local_training_context_sha256: str,
        client_id: str,
        federated_round: int,
        method_id: str,
        candidate_sha256: str,
    ) -> None:
        """Record one completed optimizer step, including an empty Poisson draw."""
        if self._closed or self._poisoned:
            raise PrivacyAccountingError("runtime accountant is closed or poisoned")
        index = _positive_int(optimizer_step_index, "optimizer step index")
        if index != self._recorded_steps + 1:
            raise PrivacyAccountingError("optimizer step index is not the next registered step")
        if self._recorded_steps >= len(self._expanded):
            raise PrivacyAccountingError("optimizer executed beyond the registered schedule")
        rate = _sample_rate(sample_rate)
        noise = _positive_real(noise_multiplier, "noise multiplier")
        sampled = _nonnegative_int(sampled_record_count, "sampled record count")
        if type(gaussian_mechanism_executed) is not bool or not gaussian_mechanism_executed:
            raise PrivacyAccountingError(
                "each accounted slot requires an executed Gaussian mechanism"
            )
        evidence_hash = _lower_sha256(mechanism_evidence_sha256, "mechanism evidence hash")
        _, expected_phase, expected_rate, expected_noise = self._expanded[self._recorded_steps]
        if rate != expected_rate or noise != expected_noise:
            raise PrivacyAccountingError("executed mechanism differs from the registered schedule")
        step_evidence = RuntimeMechanismStep(
            optimizer_step_index=index,
            phase=expected_phase,
            sample_rate=rate,
            noise_multiplier=noise,
            sampled_record_count=sampled,
            gaussian_mechanism_executed=gaussian_mechanism_executed,
            mechanism_evidence_sha256=evidence_hash,
            population_row_ids_sha256=population_row_ids_sha256,
            sampling_rng_stream_sha256=sampling_rng_stream_sha256,
            noise_rng_stream_sha256=noise_rng_stream_sha256,
            local_training_context_sha256=local_training_context_sha256,
            client_id=client_id,
            federated_round=federated_round,
            method_id=method_id,
            candidate_sha256=candidate_sha256,
        )
        try:
            self._prv.step(noise_multiplier=noise, sample_rate=rate)
            self._rdp.step(noise_multiplier=noise, sample_rate=rate)
        except Exception as exc:
            self._poisoned = True
            raise PrivacyAccountingError("runtime accountant step failed") from exc
        self._execution_history.append(step_evidence)
        self._recorded_steps += 1
        if sampled == 0:
            self._empty_sample_steps += 1

    def prefix_report(self) -> RecordDPReport:
        if self._poisoned:
            raise PrivacyAccountingError("runtime accountant is poisoned")
        prefix = _prefix_schedule(self._expanded, self._recorded_steps)
        return _report_from_accountants(
            self._prv,
            self._rdp,
            prefix,
            self._schedule,
            delta=self._delta,
            eps_error=self._eps_error,
            delta_error=self._delta_error,
            empty_sample_steps=self._empty_sample_steps,
            complete=self._recorded_steps == len(self._expanded),
            report_kind=RUNTIME_REPORT_KIND,
            execution_history=tuple(self._execution_history),
        )

    def close(self, *, executed_optimizer_steps: int) -> RecordDPReport:
        """Close only when execution, accounting and the schedule all agree."""
        if self._closed or self._poisoned:
            raise PrivacyAccountingError("runtime accountant is closed or poisoned")
        executed = _nonnegative_int(executed_optimizer_steps, "executed optimizer steps")
        if executed != self._recorded_steps or self._recorded_steps != len(self._expanded):
            raise PrivacyAccountingError(
                "execution steps, accounted steps and registered steps do not match"
            )
        report = self.prefix_report()
        if not report.complete:
            raise PrivacyAccountingError("runtime accounting report is not complete")
        self._closed = True
        return report


def _reconstruct_runtime_report(report: RecordDPReport) -> RecordDPReport:
    if report.schema != RUNTIME_REPORT_SCHEMA or report.report_kind != RUNTIME_REPORT_KIND:
        raise PrivacyAccountingError(
            "parallel composition rejects static plan accounting and requires a runtime report"
        )
    if type(report.complete) is not bool or not report.complete:
        raise PrivacyAccountingError("client runtime report is not complete")
    registered = _validated_schedule(report.registered_schedule)
    accounted = tuple(report.accounted_schedule)
    if not accounted or not all((isinstance(stage, PoissonDPStage) for stage in accounted)):
        raise PrivacyAccountingError("runtime report has no valid accounted schedule")
    if accounted != registered:
        raise PrivacyAccountingError(
            "complete runtime report schedule differs from its registered schedule"
        )
    history = _validated_runtime_history(report.execution_history, accounted)
    if not hmac.compare_digest(
        _lower_sha256(report.execution_history_sha256, "execution history hash"),
        _runtime_history_fingerprint(history),
    ):
        raise PrivacyAccountingError("runtime execution-history hash differs")
    if not hmac.compare_digest(
        _lower_sha256(report.accounted_schedule_sha256, "accounted schedule hash"),
        schedule_fingerprint(accounted),
    ) or not hmac.compare_digest(
        _lower_sha256(report.registered_schedule_sha256, "registered schedule hash"),
        schedule_fingerprint(registered),
    ):
        raise PrivacyAccountingError("runtime schedule hash differs")
    delta, eps_error, delta_error = _privacy_parameters(
        delta=report.delta, eps_error=report.eps_error, delta_error=report.delta_error
    )
    _require_fixed_opacus()
    prv = PRVAccountant()
    rdp = RDPAccountant()
    _advance_accountants(prv, rdp, accounted)
    expected = _report_from_accountants(
        prv,
        rdp,
        accounted,
        registered,
        delta=delta,
        eps_error=eps_error,
        delta_error=delta_error,
        empty_sample_steps=sum((step.sampled_record_count == 0 for step in history)),
        complete=True,
        report_kind=RUNTIME_REPORT_KIND,
        execution_history=history,
    )
    if report != expected:
        raise PrivacyAccountingError(
            "runtime report differs from exact schedule/history re-accounting"
        )
    return expected


def _single_gate_runtime_identity(report: RecordDPReport) -> tuple[str, str, str, str, int, str]:
    history = report.execution_history
    if not history:
        raise PrivacyAccountingError("runtime gate report has no execution history")
    identities = {
        (
            step.client_id,
            step.method_id,
            step.candidate_sha256,
            step.population_row_ids_sha256,
            step.federated_round,
            step.local_training_context_sha256,
        )
        for step in history
    }
    if len(identities) != 1:
        raise PrivacyAccountingError(
            "one local-gate report mixes client, method, candidate, population, round, or context identity"
        )
    return next(iter(identities))


def _client_run_artifact_payload(report: SequentialClientRunDPReport) -> dict[str, Any]:
    return {
        "schema": report.schema,
        "report_kind": report.report_kind,
        "client_id": report.client_id,
        "method_id": report.method_id,
        "candidate_sha256": report.candidate_sha256,
        "population_row_ids_sha256": report.population_row_ids_sha256,
        "expected_federated_rounds": report.expected_federated_rounds,
        "expected_round_context_sha256": report.expected_round_context_sha256,
        "client_run_manifest_sha256": report.client_run_manifest_sha256,
        "component_report_sha256": report.component_report_sha256,
        "component_execution_history_sha256": report.component_execution_history_sha256,
        "component_step_counts": report.component_step_counts,
        "composed_report_sha256": _runtime_report_fingerprint(report.composed_report),
    }


def validate_sequential_client_run_report(
    report: SequentialClientRunDPReport,
) -> SequentialClientRunDPReport:
    """Rebuild a factory-sealed complete-client-run composition artifact."""
    if type(report) is not SequentialClientRunDPReport:
        raise PrivacyAccountingError(
            "parallel composition requires an exact sequential client-run report"
        )
    if report._factory_seal is not _CLIENT_RUN_FACTORY_SEAL:
        raise PrivacyAccountingError("sequential client-run report seal differs")
    if report.schema != CLIENT_RUN_REPORT_SCHEMA or report.report_kind != CLIENT_RUN_REPORT_KIND:
        raise PrivacyAccountingError("sequential client-run report schema differs")
    client_id = _plain_identifier(report.client_id, "client ID")
    method_id = _plain_identifier(report.method_id, "method ID")
    candidate_hash = _lower_sha256(report.candidate_sha256, "candidate hash")
    population_hash = _lower_sha256(report.population_row_ids_sha256, "population row-ID hash")
    contexts = tuple(report.expected_round_context_sha256)
    if (
        not contexts
        or any((not isinstance(value, tuple) or len(value) != 2 for value in contexts))
        or len({value[0] for value in contexts}) != len(contexts)
    ):
        raise PrivacyAccountingError("client-run expected context roster is invalid")
    validated_contexts = _validated_expected_round_contexts(dict(contexts))
    expected_rounds = tuple((round_index for (round_index, _) in validated_contexts))
    if tuple(report.expected_federated_rounds) != expected_rounds:
        raise PrivacyAccountingError("client-run expected round roster differs")
    manifest_hash = client_run_manifest_fingerprint(
        client_id=client_id,
        method_id=method_id,
        candidate_sha256=candidate_hash,
        population_row_ids_sha256=population_hash,
        expected_round_context_sha256=dict(validated_contexts),
    )
    if not hmac.compare_digest(
        _lower_sha256(report.client_run_manifest_sha256, "client-run manifest hash"), manifest_hash
    ):
        raise PrivacyAccountingError("client-run manifest hash differs")
    composed = _reconstruct_runtime_report(report.composed_report)
    observed_identities = tuple(
        (
            (step.client_id, step.method_id, step.candidate_sha256, step.population_row_ids_sha256)
            for step in composed.execution_history
        )
    )
    if not observed_identities or any(
        (
            value != (client_id, method_id, candidate_hash, population_hash)
            for value in observed_identities
        )
    ):
        raise PrivacyAccountingError("client-run composed history identity differs")
    observed_context_by_round: dict[int, str] = {}
    for step in composed.execution_history:
        prior = observed_context_by_round.setdefault(
            step.federated_round, step.local_training_context_sha256
        )
        if prior != step.local_training_context_sha256:
            raise PrivacyAccountingError("one round contains multiple context hashes")
    if tuple(sorted(observed_context_by_round.items())) != validated_contexts:
        raise PrivacyAccountingError(
            "client-run composed history omits or adds a preregistered round"
        )
    counts = tuple(report.component_step_counts)
    report_hashes = tuple(report.component_report_sha256)
    history_hashes = tuple(report.component_execution_history_sha256)
    if (
        len(counts) != len(expected_rounds)
        or len(report_hashes) != len(expected_rounds)
        or len(history_hashes) != len(expected_rounds)
        or any((type(value) is not int or value <= 0 for value in counts))
        or (sum(counts) != composed.accounted_steps)
        or any((_lower_sha256(value, "component report hash") != value for value in report_hashes))
        or any(
            (
                _lower_sha256(value, "component execution-history hash") != value
                for value in history_hashes
            )
        )
    ):
        raise PrivacyAccountingError("client-run component roster is invalid")
    offset = 0
    for component_index, count in enumerate(counts):
        segment = composed.execution_history[offset : offset + count]
        expected_round = expected_rounds[component_index]
        if not segment or any((step.federated_round != expected_round for step in segment)):
            raise PrivacyAccountingError(
                "client-run component boundaries differ from expected rounds"
            )
        reindexed = tuple(
            (
                RuntimeMechanismStep(
                    optimizer_step_index=index,
                    phase=step.phase,
                    sample_rate=step.sample_rate,
                    noise_multiplier=step.noise_multiplier,
                    sampled_record_count=step.sampled_record_count,
                    gaussian_mechanism_executed=step.gaussian_mechanism_executed,
                    mechanism_evidence_sha256=step.mechanism_evidence_sha256,
                    population_row_ids_sha256=step.population_row_ids_sha256,
                    sampling_rng_stream_sha256=step.sampling_rng_stream_sha256,
                    noise_rng_stream_sha256=step.noise_rng_stream_sha256,
                    local_training_context_sha256=step.local_training_context_sha256,
                    client_id=step.client_id,
                    federated_round=step.federated_round,
                    method_id=step.method_id,
                    candidate_sha256=step.candidate_sha256,
                )
                for (index, step) in enumerate(segment, start=1)
            )
        )
        if not hmac.compare_digest(
            _runtime_history_fingerprint(reindexed), history_hashes[component_index]
        ):
            raise PrivacyAccountingError("client-run component execution-history hash differs")
        offset += count
    artifact_hash = _canonical_sha256(_client_run_artifact_payload(report))
    if not hmac.compare_digest(
        _lower_sha256(report.artifact_sha256, "client-run artifact hash"), artifact_hash
    ):
        raise PrivacyAccountingError("client-run artifact self-hash differs")
    return report


def sequential_compose_runtime_reports(
    reports: Sequence[RecordDPReport],
    *,
    expected_round_context_sha256: Mapping[int, str],
    client_run_manifest_sha256: str,
    delta: float,
    eps_error: float = 0.01,
    delta_error: float | None = None,
) -> SequentialClientRunDPReport:
    """Close and re-account every preregistered round for one client.

    The caller must supply the frozen round-to-context roster and its manifest
    hash. Missing, duplicate, unexpected, or context-drifted rounds fail closed.
    """
    if isinstance(reports, (str, bytes)) or not isinstance(reports, Sequence):
        raise PrivacyAccountingError("runtime reports must be a sequence")
    values = tuple(reports)
    if not values or not all((type(report) is RecordDPReport for report in values)):
        raise PrivacyAccountingError(
            "sequential composition requires one or more exact runtime reports"
        )
    reconstructed = tuple((_reconstruct_runtime_report(report) for report in values))
    identities = tuple((_single_gate_runtime_identity(report) for report in reconstructed))
    client_id, method_id, candidate_hash, population_hash, _, _ = identities[0]
    if any((identity[:4] != identities[0][:4] for identity in identities)):
        raise PrivacyAccountingError(
            "sequential composition mixes client, method, candidate, or population"
        )
    expected_contexts = _validated_expected_round_contexts(expected_round_context_sha256)
    expected_rounds = tuple((round_index for (round_index, _) in expected_contexts))
    observed_rounds = tuple((identity[4] for identity in identities))
    if observed_rounds != expected_rounds:
        raise PrivacyAccountingError(
            "runtime reports omit, duplicate, reorder, or add a preregistered round"
        )
    if tuple(((identity[4], identity[5]) for identity in identities)) != expected_contexts:
        raise PrivacyAccountingError(
            "runtime report context differs from the preregistered round roster"
        )
    expected_manifest = client_run_manifest_fingerprint(
        client_id=client_id,
        method_id=method_id,
        candidate_sha256=candidate_hash,
        population_row_ids_sha256=population_hash,
        expected_round_context_sha256=dict(expected_contexts),
    )
    if not hmac.compare_digest(
        _lower_sha256(client_run_manifest_sha256, "client-run manifest hash"), expected_manifest
    ):
        raise PrivacyAccountingError("client-run manifest hash differs")
    combined_schedule = tuple(
        (stage for report in reconstructed for stage in report.registered_schedule)
    )
    original_history = tuple(
        (step for report in reconstructed for step in report.execution_history)
    )
    evidence_hashes = tuple((step.mechanism_evidence_sha256 for step in original_history))
    if len(evidence_hashes) != len(set(evidence_hashes)):
        raise PrivacyAccountingError(
            "sequential composition rejects reused mechanism-step evidence"
        )
    rng_streams = tuple(
        (
            stream
            for step in original_history
            for stream in (step.sampling_rng_stream_sha256, step.noise_rng_stream_sha256)
        )
    )
    if len(rng_streams) != len(set(rng_streams)):
        raise PrivacyAccountingError(
            "sequential composition rejects reused sampling or noise RNG streams"
        )
    combined_history = tuple(
        (
            RuntimeMechanismStep(
                optimizer_step_index=index,
                phase=step.phase,
                sample_rate=step.sample_rate,
                noise_multiplier=step.noise_multiplier,
                sampled_record_count=step.sampled_record_count,
                gaussian_mechanism_executed=step.gaussian_mechanism_executed,
                mechanism_evidence_sha256=step.mechanism_evidence_sha256,
                population_row_ids_sha256=step.population_row_ids_sha256,
                sampling_rng_stream_sha256=step.sampling_rng_stream_sha256,
                noise_rng_stream_sha256=step.noise_rng_stream_sha256,
                local_training_context_sha256=step.local_training_context_sha256,
                client_id=step.client_id,
                federated_round=step.federated_round,
                method_id=step.method_id,
                candidate_sha256=step.candidate_sha256,
            )
            for (index, step) in enumerate(original_history, start=1)
        )
    )
    privacy_delta, privacy_eps_error, privacy_delta_error = _privacy_parameters(
        delta=delta, eps_error=eps_error, delta_error=delta_error
    )
    _require_fixed_opacus()
    prv = PRVAccountant()
    rdp = RDPAccountant()
    _advance_accountants(prv, rdp, combined_schedule)
    composed = _report_from_accountants(
        prv,
        rdp,
        combined_schedule,
        combined_schedule,
        delta=privacy_delta,
        eps_error=privacy_eps_error,
        delta_error=privacy_delta_error,
        empty_sample_steps=sum((step.sampled_record_count == 0 for step in combined_history)),
        complete=True,
        report_kind=RUNTIME_REPORT_KIND,
        execution_history=combined_history,
    )
    unsealed = SequentialClientRunDPReport(
        schema=CLIENT_RUN_REPORT_SCHEMA,
        report_kind=CLIENT_RUN_REPORT_KIND,
        client_id=client_id,
        method_id=method_id,
        candidate_sha256=candidate_hash,
        population_row_ids_sha256=population_hash,
        expected_federated_rounds=expected_rounds,
        expected_round_context_sha256=expected_contexts,
        client_run_manifest_sha256=expected_manifest,
        component_report_sha256=tuple(
            (_runtime_report_fingerprint(report) for report in reconstructed)
        ),
        component_execution_history_sha256=tuple(
            (report.execution_history_sha256 for report in reconstructed)
        ),
        component_step_counts=tuple((report.accounted_steps for report in reconstructed)),
        composed_report=composed,
        artifact_sha256="0" * 64,
        _factory_seal=_CLIENT_RUN_FACTORY_SEAL,
    )
    sealed = SequentialClientRunDPReport(
        schema=unsealed.schema,
        report_kind=unsealed.report_kind,
        client_id=unsealed.client_id,
        method_id=unsealed.method_id,
        candidate_sha256=unsealed.candidate_sha256,
        population_row_ids_sha256=unsealed.population_row_ids_sha256,
        expected_federated_rounds=unsealed.expected_federated_rounds,
        expected_round_context_sha256=unsealed.expected_round_context_sha256,
        client_run_manifest_sha256=unsealed.client_run_manifest_sha256,
        component_report_sha256=unsealed.component_report_sha256,
        component_execution_history_sha256=unsealed.component_execution_history_sha256,
        component_step_counts=unsealed.component_step_counts,
        composed_report=composed,
        artifact_sha256=_canonical_sha256(_client_run_artifact_payload(unsealed)),
        _factory_seal=_CLIENT_RUN_FACTORY_SEAL,
    )
    return validate_sequential_client_run_report(sealed)


def _validated_record_id(value: object) -> int:
    if type(value) is not int or value < 0:
        raise PrivacyAccountingError("record IDs must be nonnegative exact Python integers")
    return value


def _record_id_sequence_fingerprint(values: Sequence[int]) -> str:
    payload = json.dumps(
        {"schema": _identity("row_id_sequence"), "row_ids": tuple(values)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parallel_compose_disjoint_record_partitions(
    client_reports: Mapping[str, SequentialClientRunDPReport],
    client_record_ids: Mapping[str, Sequence[int]],
    *,
    conditions: ParallelCompositionConditions,
    allow_fixed_auxiliary_label_condition: bool,
    fixed_auxiliary_partition_condition_sha256: str | None,
) -> ParallelRecordDPReport:
    if not isinstance(client_reports, Mapping) or not client_reports:
        raise PrivacyAccountingError("client reports must be a non-empty mapping")
    if not isinstance(client_record_ids, Mapping):
        raise PrivacyAccountingError("client record partitions must be a mapping")
    if not isinstance(conditions, ParallelCompositionConditions):
        raise PrivacyAccountingError("parallel composition conditions are missing")
    if type(conditions.assignment_uses_private_labels_or_outcomes) is not bool:
        raise PrivacyAccountingError(
            "label/outcome-driven assignment condition must be an explicit boolean"
        )
    if allow_fixed_auxiliary_label_condition:
        if not conditions.assignment_uses_private_labels_or_outcomes:
            raise PrivacyAccountingError(
                "conditional composition requires explicit label/outcome-driven assignment"
            )
        if conditions.assignment_uses_no_unprotected_private_values:
            raise PrivacyAccountingError(
                "conditional label-driven composition cannot attest that no protected private value was used"
            )
        auxiliary_hash = _lower_sha256(
            fixed_auxiliary_partition_condition_sha256, "fixed auxiliary partition condition hash"
        )
        conditioning_status = "conditional_on_explicit_fixed_auxiliary_label_driven_partition_not_end_to_end_from_raw_records"
        composition_rule = "conditional_parallel_max_given_fixed_auxiliary_partition"
        guarantee_scope = "conditional_record_level_dp_given_the_exact_fixed_auxiliary_label_driven_partition;not_end_to_end_from_raw_benchmark_records"
    else:
        if conditions.assignment_uses_private_labels_or_outcomes:
            raise PrivacyAccountingError(
                "label- or outcome-driven client assignment cannot use unconditional parallel max composition"
            )
        if not conditions.assignment_uses_no_unprotected_private_values:
            raise PrivacyAccountingError(
                "unconditional parallel composition requires assignment without unprotected private values"
            )
        if fixed_auxiliary_partition_condition_sha256 is not None:
            raise PrivacyAccountingError(
                "unconditional composition cannot carry an auxiliary condition hash"
            )
        auxiliary_hash = None
        conditioning_status = "unconditional_fixed_nonprivate_partition"
        composition_rule = "parallel_max_over_fixed_pairwise_disjoint_record_partitions"
        guarantee_scope = "record_level_parallel_composition_over_caller_attested_fixed_and_mechanically_disjoint_partitions"
    condition_values = (
        conditions.partitions_fixed_before_private_training,
        conditions.mechanisms_read_only_their_assigned_partition,
        conditions.records_never_migrate_or_recur_across_partitions,
        conditions.no_raw_record_cross_partition_state,
        conditions.client_rng_streams_domain_separated_and_nonreused,
        conditions.client_randomness_independent_or_joint_dp_proved,
    )
    if any((type(value) is not bool or not value for value in condition_values)):
        raise PrivacyAccountingError(
            "every fixed-disjoint-partition condition must be explicitly true"
        )
    raw_names = tuple(client_reports)
    if any((not isinstance(name, str) or not name for name in raw_names)):
        raise PrivacyAccountingError("client names must be non-empty strings")
    names = tuple(sorted(raw_names))
    if set(client_record_ids) != set(names):
        raise PrivacyAccountingError("client reports and record partitions differ")
    all_ids: set[int] = set()
    membership_payload: dict[str, list[int]] = {}
    epsilon_values: list[float] = []
    delta_values: list[float] = []
    all_rng_streams: set[str] = set()
    for name in names:
        client_run = client_reports[name]
        if type(client_run) is not SequentialClientRunDPReport:
            raise PrivacyAccountingError(
                "cross-client composition requires a complete sequential client-run report, not a single local-gate report"
            )
        validate_sequential_client_run_report(client_run)
        if client_run.client_id != name:
            raise PrivacyAccountingError("client report identity differs from its mapping key")
        reconstructed = _reconstruct_runtime_report(client_run.composed_report)
        client_streams = tuple(
            (
                stream
                for step in reconstructed.execution_history
                for stream in (step.sampling_rng_stream_sha256, step.noise_rng_stream_sha256)
            )
        )
        if len(client_streams) != len(set(client_streams)):
            raise PrivacyAccountingError("one client runtime report reuses an RNG stream")
        if all_rng_streams & set(client_streams):
            raise PrivacyAccountingError(
                "client runtime reports reuse an RNG stream across partitions"
            )
        all_rng_streams.update(client_streams)
        epsilon_values.append(reconstructed.epsilon_prv_upper)
        delta_values.append(reconstructed.delta)
        raw_ids = client_record_ids[name]
        if isinstance(raw_ids, (str, bytes)) or not isinstance(raw_ids, Sequence):
            raise PrivacyAccountingError("each client partition must be a sequence")
        ids = [_validated_record_id(value) for value in raw_ids]
        if not ids or len(ids) != len(set(ids)):
            raise PrivacyAccountingError(
                "each client partition must be non-empty and internally unique"
            )
        overlap = all_ids & set(ids)
        if overlap:
            raise PrivacyAccountingError("client record partitions overlap")
        all_ids.update(ids)
        sorted_ids = sorted(ids)
        report_population_hashes = {
            step.population_row_ids_sha256 for step in reconstructed.execution_history
        }
        if report_population_hashes != {_record_id_sequence_fingerprint(sorted_ids)}:
            raise PrivacyAccountingError(
                "client report population hash differs from supplied membership"
            )
        membership_payload[name] = sorted_ids
    payload = json.dumps(
        membership_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return ParallelRecordDPReport(
        schema=PARALLEL_REPORT_SCHEMA,
        epsilon=max(epsilon_values),
        delta=max(delta_values),
        client_count=len(names),
        client_names=names,
        partition_membership_sha256=hashlib.sha256(payload).hexdigest(),
        composition_rule=composition_rule,
        guarantee_scope=guarantee_scope,
        adjacency=ADJACENCY,
        conditions_attested_and_membership_checked=True,
        conditioning_status=conditioning_status,
        fixed_auxiliary_partition_condition_sha256=auxiliary_hash,
        claims_not_made=EXCLUDED_CLAIMS,
    )


def parallel_compose_disjoint_record_partitions(
    client_reports: Mapping[str, SequentialClientRunDPReport],
    client_record_ids: Mapping[str, Sequence[int]],
    *,
    conditions: ParallelCompositionConditions,
) -> ParallelRecordDPReport:
    """Apply unconditional max composition to fixed nonprivate partitions.

    A label/outcome-driven assignment is always rejected here. Use the
    explicitly conditional API below only when the exact partition is treated
    as a fixed auxiliary condition and no end-to-end raw-record claim is made.
    """
    return _parallel_compose_disjoint_record_partitions(
        client_reports,
        client_record_ids,
        conditions=conditions,
        allow_fixed_auxiliary_label_condition=False,
        fixed_auxiliary_partition_condition_sha256=None,
    )


def conditionally_compose_label_driven_record_partitions(
    client_reports: Mapping[str, SequentialClientRunDPReport],
    client_record_ids: Mapping[str, Sequence[int]],
    *,
    conditions: ParallelCompositionConditions,
    fixed_auxiliary_partition_condition_sha256: str,
) -> ParallelRecordDPReport:
    """Return a max bound conditional on one exact fixed label-driven split.

    This is not an end-to-end DP guarantee from raw benchmark records. The
    auxiliary-condition hash must be the frozen partition artifact used by the
    runner, and that external binding remains a trust root.
    """
    return _parallel_compose_disjoint_record_partitions(
        client_reports,
        client_record_ids,
        conditions=conditions,
        allow_fixed_auxiliary_label_condition=True,
        fixed_auxiliary_partition_condition_sha256=fixed_auxiliary_partition_condition_sha256,
    )
