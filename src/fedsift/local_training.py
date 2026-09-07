"""Fail-closed result-blind local training gates for FedSift.

The private gate owns iid Poisson sampling, selected-row gradient dispatch,
full-parameter Gaussian noise, the joint DP kernel, typed correction
derivation, the optimizer update, and runtime privacy accounting. Sampling and
noise use separate deterministic torch PRNG domains solely to reproduce
experiments. They are not deployment CSPRNGs and this module is not a secure
runtime or an end-to-end privacy proof.
"""

from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import hashlib
import hmac
import json
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Callable, Mapping, Sequence
import torch
from .baseline_math import local_sgd_explicit_batches
from .dp_kernel import add_post_privacy_correction, privatize_per_record_gradients
from .modeling import (
    ModelingError,
    SelectedBCEGradientBridge,
    model_state_sha256,
    row_id_sequence_sha256,
    validate_model_state,
)
from .privacy_accounting import (
    PoissonDPStage,
    PrivacyAccountingError,
    RecordDPReport,
    RuntimePrefixAccountant,
    schedule_fingerprint,
)

TensorState = OrderedDict[str, torch.Tensor]
POISSON_STEP_RECEIPT_SCHEMA = _identity("poisson_optimizer_step_receipt")
MECHANISM_CORE_SCHEMA = _identity("poisson_mechanism_core_evidence")
EXPERIMENTAL_RNG_CLAIM = "deterministic_torch_prng_for_experiment_reproduction_not_csprng_not_deployment_security_evidence"
RANDOMNESS_PAIRING_CLAIM = "paired_seed_repeat_index_only_actual_prng_streams_are_method_candidate_isolated_not_common_random_numbers"
RECEIPT_CONFIDENTIALITY = (
    "internal_audit_artifact_contains_private_gradient_commitment_do_not_publish"
)
MECHANISM_ORDER = (
    "internal_iid_poisson_mask_draw",
    "canonical_selected_row_gradient_bridge",
    "typed_correction_derivation_before_noise",
    "full_parameter_gaussian_noise_draw",
    "joint_clip_sum_noise_fixed_normalization_kernel",
    "typed_post_privacy_correction",
    "candidate_parameter_update",
    "mechanism_core_evidence_hash",
    "runtime_privacy_accounting",
    "receipt_commit",
)
SCAFFOLD_PROVENANCE_KINDS = frozenset({"fixed_before_private_training", "prior_dp_control_release"})
CORRECTION_SOURCE_KINDS = frozenset(
    {"round_global_anchor", "fixed_scaffold_controls", "prior_dp_scaffold_controls"}
)
METHOD_CORRECTION_KIND = {
    "dp_fedavg": "none",
    "dp_fedprox": "fedprox",
    "dp_fedprox_adapted": "fedprox",
    "dp_scaffold": "scaffold",
    "dp_scaffold_adapted": "scaffold",
    "dp_fedadam": "none",
    "dp_fedyogi": "none",
    "dp_fedsofim_delta_proxy_adapted": "none",
    "time_dpfedadam": "none",
    "public_argmin_time_dpfedadam": "none",
    "fedsift": "none",
    "fedsift_uniform_schedule": "none",
    "fedsift_without_sift": "none",
    "fedsift_public_argmin_rule": "none",
}
_CORRECTION_FACTORY_SEAL = object()


class LocalTrainingGateError(RuntimeError):
    """Raised when a local-training evidence contract cannot be proved."""


class LocalTrainingGatePoisoned(LocalTrainingGateError):
    """Raised after an ambiguous or inconsistent private-step attempt."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _lower_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any((character not in "0123456789abcdef" for character in value))
    ):
        raise LocalTrainingGateError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise LocalTrainingGateError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise LocalTrainingGateError(f"{name} must be finite and positive")
    return number


def _positive_exact_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise LocalTrainingGateError(f"{name} must be a positive exact Python int")
    return value


def _nonnegative_exact_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise LocalTrainingGateError(f"{name} must be a nonnegative exact Python int")
    return value


def _seed(value: object, name: str) -> int:
    if type(value) is not int or value < 0 or value >= 2**63:
        raise LocalTrainingGateError(f"{name} must be an exact Python int in [0, 2**63)")
    return value


def _plain_identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any((character in value for character in "\r\n\t"))
    ):
        raise LocalTrainingGateError(f"{name} must be a non-empty plain identifier")
    return value


def _tensor_mapping_sha256(values: Mapping[str, torch.Tensor], *, schema: str) -> str:
    if not isinstance(values, Mapping) or not values:
        raise LocalTrainingGateError("tensor mapping must be non-empty")
    payload: list[dict[str, Any]] = []
    for name, tensor in values.items():
        if (
            not isinstance(name, str)
            or not name
            or (not isinstance(tensor, torch.Tensor))
            or (not tensor.dtype.is_floating_point)
            or (not torch.isfinite(tensor).all())
        ):
            raise LocalTrainingGateError("tensor mapping is not finite floating point")
        value = tensor.detach().cpu().contiguous()
        payload.append(
            {
                "name": name,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "values": [float(item).hex() for item in value.reshape(-1).tolist()],
            }
        )
    return _sha256_json({"schema": schema, "parameters": payload})


def _validated_population(row_ids: Sequence[int]) -> tuple[int, ...]:
    if isinstance(row_ids, (str, bytes)) or not isinstance(row_ids, Sequence):
        raise LocalTrainingGateError("population row IDs must be a sequence")
    values = tuple(row_ids)
    if not values:
        raise LocalTrainingGateError("population row IDs must be non-empty")
    if any((type(value) is not int or value < 0 for value in values)):
        raise LocalTrainingGateError("population row IDs must be exact nonnegative Python integers")
    if len(values) != len(set(values)):
        raise LocalTrainingGateError("population row IDs must be unique")
    if tuple(sorted(values)) != values:
        raise LocalTrainingGateError("population row IDs must be in canonical order")
    return values


@dataclass(frozen=True, slots=True)
class PostPrivacyCorrectionAuthorization:
    """Disabled legacy self-attestation retained only for import compatibility."""

    source: str
    fixed_before_current_poisson_draw: bool
    independent_of_current_raw_records: bool
    contains_no_current_step_private_statistic: bool

    def __post_init__(self) -> None:
        raise LocalTrainingGateError(
            "legacy arbitrary correction authorization is disabled; use a typed NoCorrection, FedProxCorrection, or ScaffoldCorrection"
        )


@dataclass(frozen=True, slots=True)
class NoCorrection:
    schema: str = _identity("no_correction")

    def __post_init__(self) -> None:
        if self.schema != _identity("no_correction"):
            raise LocalTrainingGateError("no-correction schema differs")


@dataclass(frozen=True, slots=True)
class FedProxCorrection:
    """Factory-sealed anchor from which the gate derives mu(w-w0)."""

    schema: str
    mu: float
    round_global_anchor: TensorState
    model_manifest_sha256: str
    round_global_anchor_state_sha256: str
    round_global_anchor_receipt_sha256: str
    correction_spec_sha256: str
    _factory_seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._factory_seal is not _CORRECTION_FACTORY_SEAL:
            raise LocalTrainingGateError("FedProx correction was not factory sealed")


@dataclass(frozen=True, slots=True)
class ScaffoldCorrection:
    """Factory-sealed fixed or previously-DP control-variate pair."""

    schema: str
    server_control: TensorState
    client_control: TensorState
    provenance_kind: str
    source_evidence_sha256: str
    model_manifest_sha256: str
    server_control_sha256: str
    client_control_sha256: str
    correction_spec_sha256: str
    _factory_seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._factory_seal is not _CORRECTION_FACTORY_SEAL:
            raise LocalTrainingGateError("SCAFFOLD correction was not factory sealed")


CorrectionSpec = NoCorrection | FedProxCorrection | ScaffoldCorrection


@dataclass(frozen=True, slots=True)
class LocalTrainingContext:
    """Structured identity used to domain-separate every local PRNG stream."""

    study_id: str
    dataset_id: str
    outer_repeat: int
    outer_fold: int
    inner_fold: int
    seed_repeat: int
    client_id: str
    federated_round: int
    method_id: str
    candidate_sha256: str

    def __post_init__(self) -> None:
        for name in ("study_id", "dataset_id", "client_id", "method_id"):
            object.__setattr__(self, name, _plain_identifier(getattr(self, name), name))
        for name in ("outer_repeat", "outer_fold", "inner_fold", "seed_repeat", "federated_round"):
            object.__setattr__(self, name, _nonnegative_exact_int(getattr(self, name), name))
        object.__setattr__(
            self, "candidate_sha256", _lower_sha256(self.candidate_sha256, "candidate hash")
        )

    @property
    def context_sha256(self) -> str:
        return _sha256_json(
            {
                "schema": _identity("local_training_context"),
                "study_id": self.study_id,
                "dataset_id": self.dataset_id,
                "outer_repeat": self.outer_repeat,
                "outer_fold": self.outer_fold,
                "inner_fold": self.inner_fold,
                "seed_repeat": self.seed_repeat,
                "client_id": self.client_id,
                "federated_round": self.federated_round,
                "method_id": self.method_id,
                "candidate_sha256": self.candidate_sha256,
            }
        )


@dataclass(frozen=True, slots=True)
class CorrectionSourceBinding:
    """Pre-mask binding from trusted source evidence to exact source states."""

    source_kind: str
    source_evidence_sha256: str
    model_manifest_sha256: str
    primary_state_sha256: str
    secondary_state_sha256: str | None
    source_context: LocalTrainingContext

    def __post_init__(self) -> None:
        if self.source_kind not in CORRECTION_SOURCE_KINDS:
            raise LocalTrainingGateError("correction source kind is not allowed")
        for name in ("source_evidence_sha256", "model_manifest_sha256", "primary_state_sha256"):
            object.__setattr__(
                self, name, _lower_sha256(getattr(self, name), name.replace("_", " "))
            )
        if self.secondary_state_sha256 is not None:
            object.__setattr__(
                self,
                "secondary_state_sha256",
                _lower_sha256(self.secondary_state_sha256, "secondary state hash"),
            )
        if type(self.source_context) is not LocalTrainingContext:
            raise LocalTrainingGateError("correction source context type differs")
        if self.source_kind == "round_global_anchor":
            if self.secondary_state_sha256 is not None:
                raise LocalTrainingGateError("round-global anchor cannot contain a secondary state")
        elif self.secondary_state_sha256 is None:
            raise LocalTrainingGateError(
                "SCAFFOLD source requires server and client control states"
            )

    @property
    def binding_sha256(self) -> str:
        return _sha256_json(
            {
                "schema": _identity("correction_source_binding"),
                "source_kind": self.source_kind,
                "source_evidence_sha256": self.source_evidence_sha256,
                "model_manifest_sha256": self.model_manifest_sha256,
                "primary_state_sha256": self.primary_state_sha256,
                "secondary_state_sha256": self.secondary_state_sha256,
                "source_context_sha256": self.source_context.context_sha256,
            }
        )


@dataclass(frozen=True, slots=True)
class CorrectionPolicy:
    """Pre-mask correction policy and trusted source-evidence allowlist."""

    expected_kind: str
    candidate_sha256: str
    fedprox_mu: float | None
    trusted_sources: tuple[CorrectionSourceBinding, ...]

    def __post_init__(self) -> None:
        if self.expected_kind not in {"none", "fedprox", "scaffold"}:
            raise LocalTrainingGateError("correction policy kind is not allowed")
        object.__setattr__(
            self,
            "candidate_sha256",
            _lower_sha256(self.candidate_sha256, "correction-policy candidate hash"),
        )
        sources = tuple(self.trusted_sources)
        if not all((type(value) is CorrectionSourceBinding for value in sources)):
            raise LocalTrainingGateError("trusted correction sources must be typed bindings")
        source_hashes = tuple((value.source_evidence_sha256 for value in sources))
        if (
            len(source_hashes) != len(set(source_hashes))
            or tuple(sorted(source_hashes)) != source_hashes
        ):
            raise LocalTrainingGateError(
                "trusted correction sources must be evidence-sorted and unique"
            )
        object.__setattr__(self, "trusted_sources", sources)
        if self.expected_kind == "none":
            if self.fedprox_mu is not None or sources:
                raise LocalTrainingGateError(
                    "no-correction policy cannot register mu or source evidence"
                )
        elif self.expected_kind == "fedprox":
            if self.fedprox_mu is None or not sources:
                raise LocalTrainingGateError(
                    "FedProx policy requires fixed mu and trusted anchor evidence"
                )
            object.__setattr__(
                self, "fedprox_mu", _positive_real(self.fedprox_mu, "FedProx policy mu")
            )
        elif self.fedprox_mu is not None or not sources:
            raise LocalTrainingGateError(
                "SCAFFOLD policy requires trusted source evidence and no FedProx mu"
            )

    @property
    def policy_sha256(self) -> str:
        return _sha256_json(
            {
                "schema": _identity("correction_policy"),
                "expected_kind": self.expected_kind,
                "candidate_sha256": self.candidate_sha256,
                "fedprox_mu_hex": None if self.fedprox_mu is None else self.fedprox_mu.hex(),
                "trusted_source_binding_sha256": tuple(
                    (value.binding_sha256 for value in self.trusted_sources)
                ),
            }
        )


def build_correction_source_binding(
    gradient_bridge: SelectedBCEGradientBridge,
    primary_state: Mapping[str, torch.Tensor],
    *,
    secondary_state: Mapping[str, torch.Tensor] | None,
    source_kind: str,
    source_evidence_sha256: str,
    source_context: LocalTrainingContext,
) -> CorrectionSourceBinding:
    """Bind trusted evidence to the exact pre-mask anchor/control states."""
    if type(gradient_bridge) is not SelectedBCEGradientBridge:
        raise LocalTrainingGateError("correction source requires the fixed bridge")
    if type(source_context) is not LocalTrainingContext:
        raise LocalTrainingGateError("correction source context type differs")
    evidence_hash = _lower_sha256(source_evidence_sha256, "correction source evidence hash")
    try:
        primary = validate_model_state(gradient_bridge.model, primary_state)
        primary_hash = model_state_sha256(gradient_bridge.model, primary)
        secondary_hash = (
            None
            if secondary_state is None
            else model_state_sha256(
                gradient_bridge.model, validate_model_state(gradient_bridge.model, secondary_state)
            )
        )
    except ModelingError as exc:
        raise LocalTrainingGateError("correction source state is invalid") from exc
    return CorrectionSourceBinding(
        source_kind=source_kind,
        source_evidence_sha256=evidence_hash,
        model_manifest_sha256=gradient_bridge.model_manifest_sha256,
        primary_state_sha256=primary_hash,
        secondary_state_sha256=secondary_hash,
        source_context=source_context,
    )


def build_fedprox_correction(
    gradient_bridge: SelectedBCEGradientBridge,
    round_global_anchor: Mapping[str, torch.Tensor],
    *,
    mu: float,
    round_global_anchor_receipt_sha256: str,
) -> FedProxCorrection:
    """Seal a typed FedProx anchor; tensor correction is derived in-gate."""
    if type(gradient_bridge) is not SelectedBCEGradientBridge:
        raise LocalTrainingGateError("FedProx requires the fixed modeling bridge")
    mu_value = _positive_real(mu, "FedProx mu")
    receipt_hash = _lower_sha256(
        round_global_anchor_receipt_sha256, "round-global anchor receipt hash"
    )
    try:
        anchor = validate_model_state(gradient_bridge.model, round_global_anchor)
        anchor_hash = model_state_sha256(gradient_bridge.model, anchor)
    except ModelingError as exc:
        raise LocalTrainingGateError("FedProx round-global anchor is invalid") from exc
    spec_hash = _sha256_json(
        {
            "schema": _identity("fedprox_correction_spec"),
            "mu_hex": mu_value.hex(),
            "model_manifest_sha256": gradient_bridge.model_manifest_sha256,
            "round_global_anchor_state_sha256": anchor_hash,
            "round_global_anchor_receipt_sha256": receipt_hash,
        }
    )
    return FedProxCorrection(
        schema=_identity("fedprox_correction"),
        mu=mu_value,
        round_global_anchor=anchor,
        model_manifest_sha256=gradient_bridge.model_manifest_sha256,
        round_global_anchor_state_sha256=anchor_hash,
        round_global_anchor_receipt_sha256=receipt_hash,
        correction_spec_sha256=spec_hash,
        _factory_seal=_CORRECTION_FACTORY_SEAL,
    )


def build_scaffold_correction(
    gradient_bridge: SelectedBCEGradientBridge,
    server_control: Mapping[str, torch.Tensor],
    client_control: Mapping[str, torch.Tensor],
    *,
    provenance_kind: str,
    source_evidence_sha256: str,
) -> ScaffoldCorrection:
    """Seal fixed-before-training or prior-DP SCAFFOLD control variates."""
    if type(gradient_bridge) is not SelectedBCEGradientBridge:
        raise LocalTrainingGateError("SCAFFOLD requires the fixed modeling bridge")
    if provenance_kind not in SCAFFOLD_PROVENANCE_KINDS:
        raise LocalTrainingGateError("SCAFFOLD provenance kind is not allowed")
    evidence_hash = _lower_sha256(source_evidence_sha256, "control source evidence hash")
    try:
        server = validate_model_state(gradient_bridge.model, server_control)
        client = validate_model_state(gradient_bridge.model, client_control)
        server_hash = model_state_sha256(gradient_bridge.model, server)
        client_hash = model_state_sha256(gradient_bridge.model, client)
    except ModelingError as exc:
        raise LocalTrainingGateError("SCAFFOLD control variates are invalid") from exc
    spec_hash = _sha256_json(
        {
            "schema": _identity("scaffold_correction_spec"),
            "provenance_kind": provenance_kind,
            "source_evidence_sha256": evidence_hash,
            "model_manifest_sha256": gradient_bridge.model_manifest_sha256,
            "server_control_sha256": server_hash,
            "client_control_sha256": client_hash,
        }
    )
    return ScaffoldCorrection(
        schema=_identity("scaffold_correction"),
        server_control=server,
        client_control=client,
        provenance_kind=provenance_kind,
        source_evidence_sha256=evidence_hash,
        model_manifest_sha256=gradient_bridge.model_manifest_sha256,
        server_control_sha256=server_hash,
        client_control_sha256=client_hash,
        correction_spec_sha256=spec_hash,
        _factory_seal=_CORRECTION_FACTORY_SEAL,
    )


@dataclass(frozen=True, slots=True)
class PoissonStepReceipt:
    """Exact v2 internal receipt for one completed private optimizer step."""

    schema: str
    optimizer_step_index: int
    phase: str
    sample_rate: float
    noise_multiplier: float
    clip_norm: float
    learning_rate: float
    population_size: int
    sampled_record_count: int
    empty_poisson_draw: bool
    fixed_expected_batch_normalization: float
    gaussian_noise_std: float
    parameter_names: tuple[str, ...]
    local_training_context_sha256: str
    method_id: str
    candidate_sha256: str
    model_manifest_sha256: str
    initial_model_state_sha256: str
    initial_model_state_source_sha256: str
    input_model_state_sha256: str
    output_model_state_sha256: str
    source_table_sha256: str
    population_row_ids_sha256: str
    selected_row_ids: tuple[int, ...]
    selected_row_ids_sha256: str
    poisson_mask_sha256: str
    gradient_lineage_sha256: str
    correction_policy_sha256: str
    correction_kind: str
    correction_lineage_sha256: str
    registered_schedule_sha256: str
    accounted_schedule_sha256: str
    execution_history_sha256: str
    sampling_rng_domain: str
    sampling_rng_domain_sha256: str
    sampling_rng_stream_sha256: str
    sampling_seed_commitment_sha256: str
    noise_rng_domain: str
    noise_rng_domain_sha256: str
    noise_rng_stream_sha256: str
    noise_seed_commitment_sha256: str
    gaussian_noise_tensors_sha256: str
    mechanism_evidence_sha256: str
    previous_receipt_sha256: str
    accountant_steps_after: int
    mechanism_order: tuple[str, ...]
    rng_security_claim: str
    randomness_pairing_claim: str
    receipt_confidentiality: str
    performance_metrics_consumed: bool
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class PoissonStepResult:
    """Candidate state safe to commit only with its gate-validated receipt."""

    updated_state: TensorState
    receipt: PoissonStepReceipt


def _receipt_payload(receipt: PoissonStepReceipt) -> dict[str, Any]:
    return {
        "schema": receipt.schema,
        "optimizer_step_index": receipt.optimizer_step_index,
        "phase": receipt.phase,
        "sample_rate": receipt.sample_rate,
        "noise_multiplier": receipt.noise_multiplier,
        "clip_norm": receipt.clip_norm,
        "learning_rate": receipt.learning_rate,
        "population_size": receipt.population_size,
        "sampled_record_count": receipt.sampled_record_count,
        "empty_poisson_draw": receipt.empty_poisson_draw,
        "fixed_expected_batch_normalization": receipt.fixed_expected_batch_normalization,
        "gaussian_noise_std": receipt.gaussian_noise_std,
        "parameter_names": receipt.parameter_names,
        "local_training_context_sha256": receipt.local_training_context_sha256,
        "method_id": receipt.method_id,
        "candidate_sha256": receipt.candidate_sha256,
        "model_manifest_sha256": receipt.model_manifest_sha256,
        "initial_model_state_sha256": receipt.initial_model_state_sha256,
        "initial_model_state_source_sha256": receipt.initial_model_state_source_sha256,
        "input_model_state_sha256": receipt.input_model_state_sha256,
        "output_model_state_sha256": receipt.output_model_state_sha256,
        "source_table_sha256": receipt.source_table_sha256,
        "population_row_ids_sha256": receipt.population_row_ids_sha256,
        "selected_row_ids": receipt.selected_row_ids,
        "selected_row_ids_sha256": receipt.selected_row_ids_sha256,
        "poisson_mask_sha256": receipt.poisson_mask_sha256,
        "gradient_lineage_sha256": receipt.gradient_lineage_sha256,
        "correction_policy_sha256": receipt.correction_policy_sha256,
        "correction_kind": receipt.correction_kind,
        "correction_lineage_sha256": receipt.correction_lineage_sha256,
        "registered_schedule_sha256": receipt.registered_schedule_sha256,
        "accounted_schedule_sha256": receipt.accounted_schedule_sha256,
        "execution_history_sha256": receipt.execution_history_sha256,
        "sampling_rng_domain": receipt.sampling_rng_domain,
        "sampling_rng_domain_sha256": receipt.sampling_rng_domain_sha256,
        "sampling_rng_stream_sha256": receipt.sampling_rng_stream_sha256,
        "sampling_seed_commitment_sha256": receipt.sampling_seed_commitment_sha256,
        "noise_rng_domain": receipt.noise_rng_domain,
        "noise_rng_domain_sha256": receipt.noise_rng_domain_sha256,
        "noise_rng_stream_sha256": receipt.noise_rng_stream_sha256,
        "noise_seed_commitment_sha256": receipt.noise_seed_commitment_sha256,
        "gaussian_noise_tensors_sha256": receipt.gaussian_noise_tensors_sha256,
        "mechanism_evidence_sha256": receipt.mechanism_evidence_sha256,
        "previous_receipt_sha256": receipt.previous_receipt_sha256,
        "accountant_steps_after": receipt.accountant_steps_after,
        "mechanism_order": receipt.mechanism_order,
        "rng_security_claim": receipt.rng_security_claim,
        "randomness_pairing_claim": receipt.randomness_pairing_claim,
        "receipt_confidentiality": receipt.receipt_confidentiality,
        "performance_metrics_consumed": receipt.performance_metrics_consumed,
    }


def poisson_step_receipt_fingerprint(receipt: PoissonStepReceipt) -> str:
    """Validate exact receipt constants and rebuild its self-hash."""
    if type(receipt) is not PoissonStepReceipt:
        raise LocalTrainingGateError("step receipt has an unrecognized type")
    if (
        receipt.schema != POISSON_STEP_RECEIPT_SCHEMA
        or receipt.mechanism_order != MECHANISM_ORDER
        or receipt.rng_security_claim != EXPERIMENTAL_RNG_CLAIM
        or (receipt.randomness_pairing_claim != RANDOMNESS_PAIRING_CLAIM)
        or (receipt.receipt_confidentiality != RECEIPT_CONFIDENTIALITY)
        or (type(receipt.performance_metrics_consumed) is not bool)
        or receipt.performance_metrics_consumed
    ):
        raise LocalTrainingGateError("step receipt fixed claims differ")
    _positive_real(receipt.learning_rate, "receipt learning rate")
    _lower_sha256(receipt.gaussian_noise_tensors_sha256, "receipt Gaussian-noise tensor hash")
    expected = _sha256_json(_receipt_payload(receipt))
    if not hmac.compare_digest(_lower_sha256(receipt.receipt_sha256, "receipt hash"), expected):
        raise LocalTrainingGateError("step receipt self-hash differs")
    return expected


def _draw_internal_poisson_mask(
    *, population_size: int, sample_rate: float, device: torch.device, generator: torch.Generator
) -> torch.Tensor:
    """Draw N independent Bernoulli(q) indicators from the sampling stream."""
    uniforms = torch.rand(
        (population_size,), dtype=torch.float64, device=device, generator=generator
    )
    if (
        uniforms.shape != (population_size,)
        or uniforms.device != device
        or (not torch.isfinite(uniforms).all())
    ):
        raise LocalTrainingGateError("Poisson sampling source returned invalid uniforms")
    return uniforms < sample_rate


def _draw_full_parameter_gaussian_noise(
    state: Mapping[str, torch.Tensor], *, noise_std: float, generator: torch.Generator
) -> TensorState:
    """Draw all Gaussian tensors, including after an empty Poisson draw."""
    first = next(iter(state.values()))
    if not isinstance(generator, torch.Generator) or torch.device(generator.device) != first.device:
        raise LocalTrainingGateError("Gaussian generator device differs from the model")
    output: TensorState = OrderedDict()
    for name, parameter in state.items():
        standard_normal = torch.randn(
            tuple(parameter.shape),
            dtype=parameter.dtype,
            device=parameter.device,
            generator=generator,
        )
        if (
            tuple(standard_normal.shape) != tuple(parameter.shape)
            or standard_normal.dtype != parameter.dtype
            or standard_normal.device != parameter.device
            or (not torch.isfinite(standard_normal).all())
        ):
            raise LocalTrainingGateError("Gaussian source returned an invalid tensor")
        value = standard_normal * noise_std
        if not torch.isfinite(value).all():
            raise LocalTrainingGateError("scaled Gaussian noise is non-finite")
        output[name] = value
    if tuple(output) != tuple(state):
        raise LocalTrainingGateError("Gaussian noise does not cover every parameter")
    return output


def _apply_gradient_update(
    state: Mapping[str, torch.Tensor], gradient: Mapping[str, torch.Tensor], *, learning_rate: float
) -> TensorState:
    if tuple(gradient) != tuple(state):
        raise LocalTrainingGateError("update gradient parameter identity differs")
    updated: TensorState = OrderedDict()
    for name, parameter in state.items():
        value = parameter - learning_rate * gradient[name]
        if (
            tuple(value.shape) != tuple(parameter.shape)
            or value.dtype != parameter.dtype
            or value.device != parameter.device
            or (not torch.isfinite(value).all())
        ):
            raise LocalTrainingGateError("private optimizer update is invalid")
        updated[name] = value
    return updated


def _mask_sha256(mask: torch.Tensor, population_hash: str) -> str:
    if mask.dtype != torch.bool or mask.ndim != 1:
        raise LocalTrainingGateError("internal Poisson mask structure is invalid")
    return _sha256_json(
        {
            "schema": _identity("internal_poisson_mask"),
            "population_row_ids_sha256": population_hash,
            "mask": [bool(value) for value in mask.detach().cpu().tolist()],
        }
    )


def _derive_correction(
    spec: CorrectionSpec,
    state: TensorState,
    bridge: SelectedBCEGradientBridge,
    input_state_hash: str,
) -> tuple[TensorState, str, str]:
    names = tuple(state)
    if type(spec) is NoCorrection:
        if spec.schema != _identity("no_correction"):
            raise LocalTrainingGateError("no-correction schema differs")
        correction = OrderedDict(((name, torch.zeros_like(state[name])) for name in names))
        lineage = _sha256_json(
            {
                "schema": _identity("no_correction_lineage"),
                "input_model_state_sha256": input_state_hash,
                "model_manifest_sha256": bridge.model_manifest_sha256,
            }
        )
        return (correction, "none", lineage)
    if type(spec) is FedProxCorrection:
        if spec.schema != _identity("fedprox_correction"):
            raise LocalTrainingGateError("FedProx correction schema differs")
        if spec._factory_seal is not _CORRECTION_FACTORY_SEAL:
            raise LocalTrainingGateError("FedProx correction seal differs")
        mu = _positive_real(spec.mu, "FedProx mu")
        if spec.model_manifest_sha256 != bridge.model_manifest_sha256:
            raise LocalTrainingGateError("FedProx model lineage differs")
        anchor = validate_model_state(bridge.model, spec.round_global_anchor)
        anchor_hash = model_state_sha256(bridge.model, anchor)
        if not hmac.compare_digest(
            _lower_sha256(spec.round_global_anchor_state_sha256, "round-global anchor state hash"),
            anchor_hash,
        ):
            raise LocalTrainingGateError("FedProx anchor content hash differs")
        source_hash = _lower_sha256(
            spec.round_global_anchor_receipt_sha256, "round-global anchor receipt hash"
        )
        expected_spec_hash = _sha256_json(
            {
                "schema": _identity("fedprox_correction_spec"),
                "mu_hex": mu.hex(),
                "model_manifest_sha256": bridge.model_manifest_sha256,
                "round_global_anchor_state_sha256": anchor_hash,
                "round_global_anchor_receipt_sha256": source_hash,
            }
        )
        if not hmac.compare_digest(
            _lower_sha256(spec.correction_spec_sha256, "FedProx spec hash"), expected_spec_hash
        ):
            raise LocalTrainingGateError("FedProx correction spec hash differs")
        correction = OrderedDict(((name, mu * (state[name] - anchor[name])) for name in names))
        lineage = _sha256_json(
            {
                "schema": _identity("fedprox_correction_lineage"),
                "model_manifest_sha256": bridge.model_manifest_sha256,
                "input_model_state_sha256": input_state_hash,
                "round_global_anchor_state_sha256": anchor_hash,
                "round_global_anchor_receipt_sha256": source_hash,
                "mu_hex": mu.hex(),
            }
        )
        return (correction, "fedprox", lineage)
    if type(spec) is ScaffoldCorrection:
        if spec.schema != _identity("scaffold_correction"):
            raise LocalTrainingGateError("SCAFFOLD correction schema differs")
        if spec._factory_seal is not _CORRECTION_FACTORY_SEAL:
            raise LocalTrainingGateError("SCAFFOLD correction seal differs")
        if spec.provenance_kind not in SCAFFOLD_PROVENANCE_KINDS:
            raise LocalTrainingGateError("SCAFFOLD provenance kind is not allowed")
        if spec.model_manifest_sha256 != bridge.model_manifest_sha256:
            raise LocalTrainingGateError("SCAFFOLD model lineage differs")
        server = validate_model_state(bridge.model, spec.server_control)
        client = validate_model_state(bridge.model, spec.client_control)
        server_hash = model_state_sha256(bridge.model, server)
        client_hash = model_state_sha256(bridge.model, client)
        if not hmac.compare_digest(
            _lower_sha256(spec.server_control_sha256, "server control hash"), server_hash
        ) or not hmac.compare_digest(
            _lower_sha256(spec.client_control_sha256, "client control hash"), client_hash
        ):
            raise LocalTrainingGateError("SCAFFOLD control content hash differs")
        source_hash = _lower_sha256(spec.source_evidence_sha256, "control source evidence hash")
        expected_spec_hash = _sha256_json(
            {
                "schema": _identity("scaffold_correction_spec"),
                "provenance_kind": spec.provenance_kind,
                "source_evidence_sha256": source_hash,
                "model_manifest_sha256": bridge.model_manifest_sha256,
                "server_control_sha256": server_hash,
                "client_control_sha256": client_hash,
            }
        )
        if not hmac.compare_digest(
            _lower_sha256(spec.correction_spec_sha256, "SCAFFOLD spec hash"), expected_spec_hash
        ):
            raise LocalTrainingGateError("SCAFFOLD correction spec hash differs")
        correction = OrderedDict(((name, server[name] - client[name]) for name in names))
        lineage = _sha256_json(
            {
                "schema": _identity("scaffold_correction_lineage"),
                "model_manifest_sha256": bridge.model_manifest_sha256,
                "input_model_state_sha256": input_state_hash,
                "server_control_sha256": server_hash,
                "client_control_sha256": client_hash,
                "provenance_kind": spec.provenance_kind,
                "source_evidence_sha256": source_hash,
            }
        )
        return (correction, "scaffold", lineage)
    raise LocalTrainingGateError(
        "correction must be a typed NoCorrection, FedProxCorrection, or ScaffoldCorrection"
    )


def _enforce_correction_policy(
    spec: CorrectionSpec,
    policy: CorrectionPolicy,
    bridge: SelectedBCEGradientBridge,
    context: LocalTrainingContext,
) -> None:
    if type(spec) is NoCorrection:
        observed_kind = "none"
        source_hash: str | None = None
    elif type(spec) is FedProxCorrection:
        observed_kind = "fedprox"
        source_hash = spec.round_global_anchor_receipt_sha256
        if policy.fedprox_mu is None or float(spec.mu) != policy.fedprox_mu:
            raise LocalTrainingGateError(
                "FedProx correction mu differs from the preregistered policy"
            )
    elif type(spec) is ScaffoldCorrection:
        observed_kind = "scaffold"
        source_hash = spec.source_evidence_sha256
    else:
        raise LocalTrainingGateError("correction object type is not allowed")
    if observed_kind != policy.expected_kind:
        raise LocalTrainingGateError("correction kind differs from the preregistered method policy")
    if source_hash is None:
        return
    matches = tuple(
        (
            binding
            for binding in policy.trusted_sources
            if binding.source_evidence_sha256 == source_hash
        )
    )
    if len(matches) != 1:
        raise LocalTrainingGateError(
            "correction source is absent from the preregistered trusted allowlist"
        )
    binding = matches[0]
    if binding.model_manifest_sha256 != bridge.model_manifest_sha256:
        raise LocalTrainingGateError("correction source model lineage differs")
    source_context = binding.source_context
    common_fields = (
        "study_id",
        "dataset_id",
        "outer_repeat",
        "outer_fold",
        "inner_fold",
        "seed_repeat",
        "method_id",
        "candidate_sha256",
    )
    if any((getattr(source_context, name) != getattr(context, name) for name in common_fields)):
        raise LocalTrainingGateError("correction source experiment context differs")
    if type(spec) is FedProxCorrection:
        if (
            binding.source_kind != "round_global_anchor"
            or binding.primary_state_sha256 != spec.round_global_anchor_state_sha256
            or binding.secondary_state_sha256 is not None
            or (source_context.federated_round != context.federated_round)
            or (source_context.client_id != "server")
        ):
            raise LocalTrainingGateError(
                "FedProx anchor differs from its trusted round-global binding"
            )
        return
    assert type(spec) is ScaffoldCorrection
    expected_source_kind = {
        "fixed_before_private_training": "fixed_scaffold_controls",
        "prior_dp_control_release": "prior_dp_scaffold_controls",
    }[spec.provenance_kind]
    if (
        binding.source_kind != expected_source_kind
        or binding.primary_state_sha256 != spec.server_control_sha256
        or binding.secondary_state_sha256 != spec.client_control_sha256
        or (source_context.client_id not in {context.client_id, "server"})
    ):
        raise LocalTrainingGateError("SCAFFOLD controls differ from their trusted source binding")
    if spec.provenance_kind == "prior_dp_control_release":
        if source_context.federated_round >= context.federated_round:
            raise LocalTrainingGateError(
                "prior-DP SCAFFOLD controls must precede the current round"
            )
    elif source_context.federated_round > context.federated_round:
        raise LocalTrainingGateError("fixed SCAFFOLD controls cannot originate in a future round")


class PoissonLocalTrainingGate:
    """Own the complete Poisson mechanism-to-accounting execution boundary."""

    def __init__(
        self,
        schedule: Sequence[PoissonDPStage],
        *,
        gradient_bridge: SelectedBCEGradientBridge,
        population_row_ids: Sequence[int],
        clip_norm: float,
        learning_rate: float,
        delta: float,
        sampling_seed: int,
        noise_seed: int,
        context: LocalTrainingContext,
        correction_policy: CorrectionPolicy,
        initial_model_state_sha256: str,
        initial_model_state_source_sha256: str,
        eps_error: float = 0.01,
        delta_error: float | None = None,
    ) -> None:
        if isinstance(schedule, (str, bytes)) or not isinstance(schedule, Sequence):
            raise LocalTrainingGateError("private schedule must be a sequence")
        stages = tuple(schedule)
        if not stages or not all((isinstance(stage, PoissonDPStage) for stage in stages)):
            raise LocalTrainingGateError("private schedule contains an invalid stage")
        if type(gradient_bridge) is not SelectedBCEGradientBridge:
            raise LocalTrainingGateError("private gate requires the fixed BCE gradient bridge")
        population = _validated_population(population_row_ids)
        if population != gradient_bridge.canonical_population_row_ids:
            raise LocalTrainingGateError(
                "gate population differs from the gradient bridge population"
            )
        sampling_root = _seed(sampling_seed, "sampling seed")
        noise_root = _seed(noise_seed, "noise seed")
        if sampling_root == noise_root:
            raise LocalTrainingGateError("sampling and Gaussian-noise seeds must differ")
        if type(context) is not LocalTrainingContext:
            raise LocalTrainingGateError("local-training context type differs")
        if type(correction_policy) is not CorrectionPolicy:
            raise LocalTrainingGateError("correction policy type differs")
        if not hmac.compare_digest(context.candidate_sha256, correction_policy.candidate_sha256):
            raise LocalTrainingGateError(
                "correction policy candidate differs from local-training context"
            )
        expected_correction_kind = METHOD_CORRECTION_KIND.get(context.method_id)
        if expected_correction_kind is None:
            raise LocalTrainingGateError(
                "local-training method is absent from the fixed correction-kind registry"
            )
        if correction_policy.expected_kind != expected_correction_kind:
            raise LocalTrainingGateError("method identity and preregistered correction kind differ")
        initial_state_hash = _lower_sha256(initial_model_state_sha256, "initial model state hash")
        initial_source_hash = _lower_sha256(
            initial_model_state_source_sha256, "initial model state source hash"
        )
        self._schedule = stages
        self._expanded = tuple(
            (
                (stage.phase, stage.sample_rate, stage.noise_multiplier)
                for stage in stages
                for _ in range(stage.steps)
            )
        )
        self._schedule_sha256 = schedule_fingerprint(stages)
        self._bridge = gradient_bridge
        self._context = context
        self._correction_policy = correction_policy
        self._initial_model_state_sha256 = initial_state_hash
        self._initial_model_state_source_sha256 = initial_source_hash
        self._population_row_ids = population
        self._population_row_ids_sha256 = row_id_sequence_sha256(population)
        self._population_size = len(population)
        self._clip_norm = _positive_real(clip_norm, "clip norm")
        self._learning_rate = _positive_real(learning_rate, "learning rate")
        self._sampling_seed = sampling_root
        self._noise_seed = noise_root
        self._sampling_rng_domain = f"{context.context_sha256}/iid_poisson_sampling_v1"
        self._noise_rng_domain = f"{context.context_sha256}/gaussian_parameter_noise_v1"
        if self._sampling_rng_domain == self._noise_rng_domain:
            raise LocalTrainingGateError("sampling and noise RNG domains collide")
        try:
            self._accountant = RuntimePrefixAccountant(
                stages, delta=delta, eps_error=eps_error, delta_error=delta_error
            )
        except PrivacyAccountingError as exc:
            raise LocalTrainingGateError(
                "runtime privacy accountant initialization failed"
            ) from exc
        self._committed_steps = 0
        self._poisoned = False
        self._closed = False
        self._used_rng_stream_commitments: set[str] = set()
        self._receipt_ledger: dict[int, PoissonStepReceipt] = {}
        self._expected_next_input_state_sha256: str | None = initial_state_hash
        self._previous_receipt_sha256 = _sha256_json(
            {
                "schema": _identity("poisson_receipt_chain_genesis"),
                "registered_schedule_sha256": self._schedule_sha256,
                "local_training_context_sha256": self._context.context_sha256,
                "correction_policy_sha256": self._correction_policy.policy_sha256,
                "model_manifest_sha256": self._bridge.model_manifest_sha256,
                "initial_model_state_sha256": self._initial_model_state_sha256,
                "initial_model_state_source_sha256": self._initial_model_state_source_sha256,
                "source_table_sha256": self._bridge.source_table_sha256,
                "population_row_ids_sha256": self._population_row_ids_sha256,
            }
        )

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def committed_steps(self) -> int:
        return self._committed_steps

    @property
    def registered_steps(self) -> int:
        return len(self._expanded)

    @property
    def population_row_ids(self) -> tuple[int, ...]:
        return self._population_row_ids

    def _require_live(self) -> None:
        if self._poisoned:
            raise LocalTrainingGatePoisoned("private local-training gate is poisoned")
        if self._closed:
            raise LocalTrainingGateError("private local-training gate is closed")

    def _poison(self) -> None:
        self._poisoned = True

    def _rng_stream(
        self, *, purpose: str, root_seed: int, domain: str, step_index: int, device: torch.device
    ) -> tuple[torch.Generator, str, str, int]:
        material = {
            "schema": _identity("experimental_rng_stream"),
            "purpose": purpose,
            "domain": domain,
            "root_seed": root_seed,
            "optimizer_step_index": step_index,
            "registered_schedule_sha256": self._schedule_sha256,
            "population_row_ids_sha256": self._population_row_ids_sha256,
            "model_manifest_sha256": self._bridge.model_manifest_sha256,
            "local_training_context_sha256": self._context.context_sha256,
            "device_type": device.type,
        }
        stream_commitment = _sha256_json(material)
        if stream_commitment in self._used_rng_stream_commitments:
            raise LocalTrainingGateError("an experimental RNG stream would be reused")
        seed_value = int(stream_commitment[:16], 16) % 2**63
        seed_commitment = _sha256_json(
            {
                "schema": _identity("experimental_rng_seed_commitment"),
                "stream_commitment_sha256": stream_commitment,
                "derived_seed": seed_value,
            }
        )
        generator = torch.Generator(device=device).manual_seed(seed_value)
        return (generator, stream_commitment, seed_commitment, seed_value)

    def execute_poisson_optimizer_step(
        self,
        state: Mapping[str, torch.Tensor],
        *,
        optimizer_step_index: int,
        correction: CorrectionSpec | None = None,
    ) -> PoissonStepResult:
        """Execute one gate-owned Poisson DP-SGD step and commit its evidence.

        There is intentionally no mask, selected-gradient mapping, q, sigma, or
        generator parameter. The caller supplies only the current fixed-model
        state, next registered step index, and an optional typed correction.
        """
        self._require_live()
        try:
            step_index = _positive_exact_int(optimizer_step_index, "optimizer step index")
            if step_index != self._committed_steps + 1 or step_index > len(self._expanded):
                raise LocalTrainingGateError(
                    "optimizer step index differs from the registered next step"
                )
            phase, rate, noise_multiplier = self._expanded[step_index - 1]
            state_copy = validate_model_state(self._bridge.model, state)
            input_state_hash = model_state_sha256(self._bridge.model, state_copy)
            if self._expected_next_input_state_sha256 is not None and (
                not hmac.compare_digest(input_state_hash, self._expected_next_input_state_sha256)
            ):
                raise LocalTrainingGateError(
                    "input model state breaks the committed local-step lineage"
                )
            device = next(iter(state_copy.values())).device
            sampling_generator, sampling_stream, sampling_seed_commitment, sampling_derived = (
                self._rng_stream(
                    purpose="iid_poisson_sampling",
                    root_seed=self._sampling_seed,
                    domain=self._sampling_rng_domain,
                    step_index=step_index,
                    device=device,
                )
            )
            noise_generator, noise_stream, noise_seed_commitment, noise_derived = self._rng_stream(
                purpose="gaussian_parameter_noise",
                root_seed=self._noise_seed,
                domain=self._noise_rng_domain,
                step_index=step_index,
                device=device,
            )
            if (
                sampling_stream == noise_stream
                or sampling_derived == noise_derived
                or sampling_seed_commitment == noise_seed_commitment
            ):
                raise LocalTrainingGateError("sampling and noise RNG streams collide")
            self._used_rng_stream_commitments.update({sampling_stream, noise_stream})
            mask = _draw_internal_poisson_mask(
                population_size=self._population_size,
                sample_rate=rate,
                device=device,
                generator=sampling_generator,
            )
            mask_hash = _mask_sha256(mask, self._population_row_ids_sha256)
            mask_values = mask.detach().cpu().tolist()
            selected_row_ids = tuple(
                (
                    row_id
                    for (row_id, selected) in zip(self._population_row_ids, mask_values)
                    if bool(selected)
                )
            )
            sampled_count = len(selected_row_ids)
            selected_hash = row_id_sequence_sha256(selected_row_ids)
            batch = self._bridge.selected_gradient_batch(state_copy, selected_row_ids)
            gradients = self._bridge.validate_batch(batch, state_copy, selected_row_ids)
            correction_spec: CorrectionSpec = NoCorrection() if correction is None else correction
            _enforce_correction_policy(
                correction_spec, self._correction_policy, self._bridge, self._context
            )
            correction_values, correction_kind, correction_lineage = _derive_correction(
                correction_spec, state_copy, self._bridge, input_state_hash
            )
            fixed_normalization = rate * self._population_size
            noise_std = noise_multiplier * self._clip_norm
            gaussian_noise = _draw_full_parameter_gaussian_noise(
                state_copy, noise_std=noise_std, generator=noise_generator
            )
            private_gradient = privatize_per_record_gradients(
                gradients,
                gaussian_noise,
                clip_norm=self._clip_norm,
                fixed_normalization=fixed_normalization,
            )
            corrected_gradient = add_post_privacy_correction(private_gradient, correction_values)
            candidate_state = _apply_gradient_update(
                state_copy, corrected_gradient, learning_rate=self._learning_rate
            )
            output_state_hash = model_state_sha256(self._bridge.model, candidate_state)
            noise_hash = _tensor_mapping_sha256(
                gaussian_noise, schema=_identity("gaussian_noise_tensors")
            )
            mechanism_evidence = _sha256_json(
                {
                    "schema": MECHANISM_CORE_SCHEMA,
                    "optimizer_step_index": step_index,
                    "phase": phase,
                    "sample_rate": rate,
                    "noise_multiplier": noise_multiplier,
                    "clip_norm": self._clip_norm,
                    "fixed_expected_batch_normalization": fixed_normalization,
                    "local_training_context_sha256": self._context.context_sha256,
                    "method_id": self._context.method_id,
                    "candidate_sha256": self._context.candidate_sha256,
                    "model_manifest_sha256": self._bridge.model_manifest_sha256,
                    "initial_model_state_sha256": self._initial_model_state_sha256,
                    "initial_model_state_source_sha256": self._initial_model_state_source_sha256,
                    "input_model_state_sha256": input_state_hash,
                    "output_model_state_sha256": output_state_hash,
                    "source_table_sha256": self._bridge.source_table_sha256,
                    "population_row_ids_sha256": self._population_row_ids_sha256,
                    "selected_row_ids_sha256": selected_hash,
                    "poisson_mask_sha256": mask_hash,
                    "gradient_lineage_sha256": batch.gradient_lineage_sha256,
                    "correction_policy_sha256": self._correction_policy.policy_sha256,
                    "correction_kind": correction_kind,
                    "correction_lineage_sha256": correction_lineage,
                    "sampling_rng_stream_commitment_sha256": sampling_stream,
                    "noise_rng_stream_commitment_sha256": noise_stream,
                    "gaussian_noise_tensors_sha256": noise_hash,
                    "kernel": "joint_l2_clip_sum_noise_fixed_qN_normalization_v1",
                    "optimizer": "plain_sgd_parameter_update_v1",
                    "learning_rate_hex": self._learning_rate.hex(),
                }
            )
            self._accountant.record_optimizer_step(
                optimizer_step_index=step_index,
                sample_rate=rate,
                noise_multiplier=noise_multiplier,
                sampled_record_count=sampled_count,
                gaussian_mechanism_executed=True,
                mechanism_evidence_sha256=mechanism_evidence,
                population_row_ids_sha256=self._population_row_ids_sha256,
                sampling_rng_stream_sha256=sampling_stream,
                noise_rng_stream_sha256=noise_stream,
                local_training_context_sha256=self._context.context_sha256,
                client_id=self._context.client_id,
                federated_round=self._context.federated_round,
                method_id=self._context.method_id,
                candidate_sha256=self._context.candidate_sha256,
            )
            if self._accountant.recorded_steps != step_index:
                raise LocalTrainingGateError(
                    "optimizer update and privacy-accounting counts diverged"
                )
            receipt_without_hash = PoissonStepReceipt(
                schema=POISSON_STEP_RECEIPT_SCHEMA,
                optimizer_step_index=step_index,
                phase=phase,
                sample_rate=rate,
                noise_multiplier=noise_multiplier,
                clip_norm=self._clip_norm,
                learning_rate=self._learning_rate,
                population_size=self._population_size,
                sampled_record_count=sampled_count,
                empty_poisson_draw=sampled_count == 0,
                fixed_expected_batch_normalization=fixed_normalization,
                gaussian_noise_std=noise_std,
                parameter_names=tuple(state_copy),
                local_training_context_sha256=self._context.context_sha256,
                method_id=self._context.method_id,
                candidate_sha256=self._context.candidate_sha256,
                model_manifest_sha256=self._bridge.model_manifest_sha256,
                initial_model_state_sha256=self._initial_model_state_sha256,
                initial_model_state_source_sha256=self._initial_model_state_source_sha256,
                input_model_state_sha256=input_state_hash,
                output_model_state_sha256=output_state_hash,
                source_table_sha256=self._bridge.source_table_sha256,
                population_row_ids_sha256=self._population_row_ids_sha256,
                selected_row_ids=selected_row_ids,
                selected_row_ids_sha256=selected_hash,
                poisson_mask_sha256=mask_hash,
                gradient_lineage_sha256=batch.gradient_lineage_sha256,
                correction_policy_sha256=self._correction_policy.policy_sha256,
                correction_kind=correction_kind,
                correction_lineage_sha256=correction_lineage,
                registered_schedule_sha256=self._accountant.registered_schedule_sha256,
                accounted_schedule_sha256=self._accountant.accounted_schedule_sha256,
                execution_history_sha256=self._accountant.execution_history_sha256,
                sampling_rng_domain=self._sampling_rng_domain,
                sampling_rng_domain_sha256=_sha256_json({"domain": self._sampling_rng_domain}),
                sampling_rng_stream_sha256=sampling_stream,
                sampling_seed_commitment_sha256=sampling_seed_commitment,
                noise_rng_domain=self._noise_rng_domain,
                noise_rng_domain_sha256=_sha256_json({"domain": self._noise_rng_domain}),
                noise_rng_stream_sha256=noise_stream,
                noise_seed_commitment_sha256=noise_seed_commitment,
                gaussian_noise_tensors_sha256=noise_hash,
                mechanism_evidence_sha256=mechanism_evidence,
                previous_receipt_sha256=self._previous_receipt_sha256,
                accountant_steps_after=self._accountant.recorded_steps,
                mechanism_order=MECHANISM_ORDER,
                rng_security_claim=EXPERIMENTAL_RNG_CLAIM,
                randomness_pairing_claim=RANDOMNESS_PAIRING_CLAIM,
                receipt_confidentiality=RECEIPT_CONFIDENTIALITY,
                performance_metrics_consumed=False,
                receipt_sha256="0" * 64,
            )
            receipt = PoissonStepReceipt(
                **{
                    **_receipt_payload(receipt_without_hash),
                    "receipt_sha256": _sha256_json(_receipt_payload(receipt_without_hash)),
                }
            )
            poisson_step_receipt_fingerprint(receipt)
        except Exception as exc:
            self._poison()
            if isinstance(exc, LocalTrainingGatePoisoned):
                raise
            raise LocalTrainingGatePoisoned(
                "private sampling, mechanism, update, evidence, and accounting did not close atomically"
            ) from exc
        try:
            self._receipt_ledger[step_index] = receipt
            self._committed_steps = step_index
            self._expected_next_input_state_sha256 = output_state_hash
            self._previous_receipt_sha256 = receipt.receipt_sha256
        except Exception as exc:
            self._poison()
            raise LocalTrainingGatePoisoned(
                "accounted step evidence could not be committed to the gate ledger"
            ) from exc
        return PoissonStepResult(updated_state=candidate_state, receipt=receipt)

    def validate_step_result(
        self, result: PoissonStepResult, input_state: Mapping[str, torch.Tensor]
    ) -> None:
        """Validate receipt/state hashes against this gate's committed ledger."""
        if type(result) is not PoissonStepResult:
            raise LocalTrainingGateError("step result has an unrecognized type")
        receipt = result.receipt
        rebuilt = poisson_step_receipt_fingerprint(receipt)
        committed = self._receipt_ledger.get(receipt.optimizer_step_index)
        if committed is None or receipt != committed:
            raise LocalTrainingGateError(
                "step receipt differs from the gate's committed evidence ledger"
            )
        if not hmac.compare_digest(rebuilt, committed.receipt_sha256):
            raise LocalTrainingGateError("committed receipt fingerprint differs")
        try:
            input_hash = model_state_sha256(self._bridge.model, input_state)
            output_hash = model_state_sha256(self._bridge.model, result.updated_state)
        except ModelingError as exc:
            raise LocalTrainingGateError("step result model state is invalid") from exc
        if not hmac.compare_digest(
            input_hash, receipt.input_model_state_sha256
        ) or not hmac.compare_digest(output_hash, receipt.output_model_state_sha256):
            raise LocalTrainingGateError("step result state lineage differs")

    def prefix_privacy_report(self) -> RecordDPReport:
        self._require_live()
        try:
            return self._accountant.prefix_report()
        except PrivacyAccountingError as exc:
            self._poison()
            raise LocalTrainingGatePoisoned("runtime privacy prefix could not be proved") from exc

    def close(self, *, executed_optimizer_steps: int) -> RecordDPReport:
        self._require_live()
        executed = _positive_exact_int(executed_optimizer_steps, "executed optimizer steps")
        if executed != self._committed_steps or executed != len(self._expanded):
            self._poison()
            raise LocalTrainingGatePoisoned(
                "execution, committed update, and registered step counts differ"
            )
        try:
            report = self._accountant.close(executed_optimizer_steps=executed)
        except PrivacyAccountingError as exc:
            self._poison()
            raise LocalTrainingGatePoisoned("runtime privacy accountant did not close") from exc
        self._closed = True
        return report


def canonical_nonprivate_explicit_minibatches(
    start_state: Mapping[str, torch.Tensor],
    batch_plan_by_epoch: Sequence[Sequence[Any]],
    batch_gradient_fn: Callable[[Mapping[str, torch.Tensor], Any], Mapping[str, torch.Tensor]],
    learning_rate: float,
) -> TensorState:
    """Use the canonical non-private explicit-minibatch path only."""
    return local_sgd_explicit_batches(
        start_state, batch_plan_by_epoch, batch_gradient_fn, learning_rate
    )
