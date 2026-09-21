"""Fail-closed method dispatch for the FedSift comparison roster.

The candidate roster is only a design artifact.  This module is the explicit
bridge from a registered method identity to the local mechanism, correction,
server kernel, privacy schedule, and public-control rule that a runner must
execute.  Unknown names, incompatible candidate parameters, and incomplete
matched-ablation switches are rejected instead of falling back to FedAvg.

The dispatch contract contains no observed predictions or metrics.  It is
therefore safe to construct and freeze before any outer-test access.
"""

from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import copy
import hmac
import math
from dataclasses import asdict, dataclass
from typing import Callable, Mapping
from . import baseline_math
from .candidate_space import MAIN_METHODS, canonical_sha256
from .hpo_plan import MATCHED_ABLATIONS
from .sofim_math import sofim_server_step_with_receipt


class MethodDispatchError(RuntimeError):
    """Raised when a registered method cannot be dispatched exactly."""


ALL_METHODS: tuple[str, ...] = (*MAIN_METHODS, *tuple(MATCHED_ABLATIONS))


@dataclass(frozen=True, slots=True)
class MethodExecutionSpec:
    """Exact result-blind execution identity for one registered method."""

    schema: str
    method_id: str
    parent_method: str
    privacy_mode: str
    local_update_handler: str
    correction_handler: str
    server_backend_handler: str
    privacy_schedule_handler: str
    public_control_handler: str
    public_control_queries_enabled: bool
    non_query_round_handler: str
    resource_accounting_method_id: str
    scaffold_variant: str
    sofim_proxy_contract: str
    claim_boundary: str
    candidate_parameters_sha256: str
    mechanism_switch_sha256: str
    dispatch_sha256: str

    def manifest(self) -> dict[str, object]:
        return asdict(self)


_BACKEND_NAME = {
    "fedavg_nonprivate": "weighted_average",
    "dp_fedavg": "weighted_average",
    "dp_fedprox_adapted": "weighted_average",
    "dp_scaffold_adapted": "scaffold_option2_batched_weighted_adapted",
    "dp_fedadam": "fedadam_paper_aligned",
    "dp_fedyogi": "fedyogi_paper_aligned",
    "dp_fedsofim_delta_proxy_adapted": "dp_fedsofim_delta_proxy_adapted",
    "time_dpfedadam": "fedadam_paper_aligned",
    "public_argmin_time_dpfedadam": "fedadam_paper_aligned",
    "fedsift": "fedadam_paper_aligned",
}
_SERVER_HANDLER = {
    "weighted_average": "fedavg_server_average",
    "scaffold_option2_batched_weighted_adapted": "scaffold_server_batched_weighted",
    "fedadam_paper_aligned": "fedadam_server_step",
    "fedyogi_paper_aligned": "fedyogi_server_step",
    "dp_fedsofim_delta_proxy_adapted": "sofim_server_step_with_receipt",
}
_SERVER_KERNELS: dict[str, Callable[..., object]] = {
    "fedavg_server_average": baseline_math.fedavg_server_average,
    "scaffold_server_batched_weighted": baseline_math.scaffold_server_batched_weighted,
    "fedadam_server_step": baseline_math.fedadam_server_step,
    "fedyogi_server_step": baseline_math.fedyogi_server_step,
    "sofim_server_step_with_receipt": sofim_server_step_with_receipt,
}


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise MethodDispatchError(f"{field} must be a mapping")
    return value


def _positive_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MethodDispatchError(f"{field} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise MethodDispatchError(f"{field} must be a finite positive number")
    return result


def _nonnegative_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MethodDispatchError(f"{field} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise MethodDispatchError(f"{field} must be a finite nonnegative number")
    return result


def _open_unit_float(value: object, field: str) -> float:
    result = _positive_float(value, field)
    if not result < 1.0:
        raise MethodDispatchError(f"{field} must be strictly between zero and one")
    return result


def _closed_open_unit_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MethodDispatchError(f"{field} must be in [0, 1)")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise MethodDispatchError(f"{field} must be in [0, 1)")
    return result


def _exact_parent(method_id: str) -> str:
    if method_id in MAIN_METHODS:
        return method_id
    record = MATCHED_ABLATIONS.get(method_id)
    if not isinstance(record, Mapping):
        raise MethodDispatchError("unknown method identity; fallback is forbidden")
    parent = record.get("parent_method")
    if not isinstance(parent, str) or parent not in MAIN_METHODS:
        raise MethodDispatchError("matched ablation parent is invalid")
    return parent


def _validate_common_candidate(
    parent_method: str, parameters: Mapping[str, object]
) -> Mapping[str, object]:
    optimizer = _mapping(parameters.get("local_optimizer"), "local_optimizer")
    if set(optimizer) != {"name", "learning_rate"} or optimizer.get("name") != "sgd":
        raise MethodDispatchError("all candidates require plain local SGD")
    _positive_float(optimizer.get("learning_rate"), "local learning rate")
    backend = _mapping(parameters.get("backend"), "backend")
    expected_backend = _BACKEND_NAME[parent_method]
    if backend.get("name") != expected_backend:
        raise MethodDispatchError(
            f"{parent_method} backend differs from its registered implementation"
        )
    if expected_backend == "weighted_average":
        if set(backend) != {"name", "server_learning_rate"}:
            raise MethodDispatchError("weighted-average backend fields differ")
        if _positive_float(backend.get("server_learning_rate"), "server learning rate") != 1.0:
            raise MethodDispatchError("weighted-average server learning rate must be one")
    elif expected_backend == "scaffold_option2_batched_weighted_adapted":
        if set(backend) != {"name", "server_learning_rate"}:
            raise MethodDispatchError("SCAFFOLD backend fields differ")
        _positive_float(backend.get("server_learning_rate"), "server learning rate")
    elif expected_backend in {"fedadam_paper_aligned", "fedyogi_paper_aligned"}:
        if set(backend) != {
            "name",
            "beta1",
            "beta2",
            "server_learning_rate",
            "tau",
            "second_moment_initialization",
        }:
            raise MethodDispatchError("FedOpt backend fields differ")
        _closed_open_unit_float(backend.get("beta1"), "FedOpt beta1")
        _closed_open_unit_float(backend.get("beta2"), "FedOpt beta2")
        _positive_float(backend.get("server_learning_rate"), "FedOpt server learning rate")
        _positive_float(backend.get("tau"), "FedOpt tau")
        if backend.get("second_moment_initialization") != "tau_squared":
            raise MethodDispatchError("FedOpt second-moment initialization differs")
    elif expected_backend == "dp_fedsofim_delta_proxy_adapted":
        if set(backend) != {
            "name",
            "server_learning_rate",
            "beta",
            "rho",
            "moment_initialization",
            "bias_correction",
            "warmup_rounds",
            "preconditioned_vector",
            "client_proxy_definition",
            "aggregation",
            "claim_boundary",
        }:
            raise MethodDispatchError("SOFIM adapted backend fields differ")
        _positive_float(backend.get("server_learning_rate"), "SOFIM server learning rate")
        _open_unit_float(backend.get("beta"), "SOFIM beta")
        _positive_float(backend.get("rho"), "SOFIM rho")
        if (
            backend.get("moment_initialization") != "zeros"
            or backend.get("bias_correction") is not False
            or backend.get("warmup_rounds") != 0
            or (backend.get("preconditioned_vector") != "current_normalized_trajectory_proxy")
            or (
                backend.get("client_proxy_definition")
                != "negative_local_minus_global_divided_by_actual_steps_times_local_lr"
            )
            or (
                backend.get("aggregation")
                != "same_preregistered_server_weights_as_model_delta_methods"
            )
            or (
                backend.get("claim_boundary")
                != "adapted_proxy_not_paper_or_official_repository_reproduction"
            )
        ):
            raise MethodDispatchError("SOFIM adapted identity or claim boundary differs")
    allowed_top = {"local_optimizer", "backend"}
    if parent_method != "fedavg_nonprivate":
        record_dp = _mapping(parameters.get("record_dp"), "record_dp")
        if record_dp != {
            "mechanism": "common_client_record_level_dp_sgd",
            "accounting": "candidate_specific_same_target_epsilon_delta",
        }:
            raise MethodDispatchError("record-DP candidate contract differs")
        schedule = _mapping(parameters.get("privacy_schedule"), "privacy_schedule")
        expected_kind = (
            "two_phase_time"
            if parent_method in {"time_dpfedadam", "public_argmin_time_dpfedadam", "fedsift"}
            else "uniform"
        )
        if schedule.get("kind") != expected_kind:
            raise MethodDispatchError("candidate privacy schedule kind differs")
        if expected_kind == "uniform":
            if set(schedule) != {"kind"}:
                raise MethodDispatchError("uniform schedule fields differ")
        else:
            if set(schedule) != {
                "kind",
                "saving_round_fraction",
                "saving_sigma_factor",
                "spending_sigma_factor",
            }:
                raise MethodDispatchError("two-phase schedule fields differ")
            saving_fraction = _open_unit_float(
                schedule.get("saving_round_fraction"), "saving round fraction"
            )
            if not 0.0 < saving_fraction < 1.0:
                raise MethodDispatchError("saving round fraction differs")
            _positive_float(schedule.get("saving_sigma_factor"), "saving sigma factor")
            _positive_float(schedule.get("spending_sigma_factor"), "spending sigma factor")
        allowed_top.update({"record_dp", "privacy_schedule"})
    if parent_method == "fedavg_nonprivate" and (
        "record_dp" in parameters or "privacy_schedule" in parameters
    ):
        raise MethodDispatchError("non-private FedAvg cannot carry a DP schedule")
    if parent_method == "dp_fedprox_adapted":
        objective = _mapping(parameters.get("local_objective"), "local_objective")
        if set(objective) != {"name", "prox_mu"} or objective.get("name") != "fedprox":
            raise MethodDispatchError("FedProx local objective differs")
        _positive_float(objective.get("prox_mu"), "FedProx mu")
        allowed_top.add("local_objective")
    if parent_method in {"public_argmin_time_dpfedadam", "fedsift"}:
        control = _mapping(parameters.get("control_rule"), "control_rule")
        if (
            control.get("records") != "same_frozen_V_ctrl"
            or control.get("labels_visible") is not True
            or (not isinstance(control.get("step_candidates"), list))
            or (len(control["step_candidates"]) < 2)
        ):
            raise MethodDispatchError("public-control candidate contract differs")
        query_every = control.get("query_every_rounds")
        if isinstance(query_every, bool) or not isinstance(query_every, int) or query_every < 1:
            raise MethodDispatchError("public-control query cadence is invalid")
        allowed_top.add("control_rule")
    if parent_method == "public_argmin_time_dpfedadam":
        control = _mapping(parameters.get("control_rule"), "control_rule")
        if set(control) != {
            "records",
            "labels_visible",
            "step_candidates",
            "query_every_rounds",
            "name",
            "tie_break",
            "safety_margin",
        }:
            raise MethodDispatchError("public argmin control fields differ")
        if (
            control.get("name") != "public_argmin_mean_logloss"
            or control.get("tie_break") != "larger_step"
            or control.get("safety_margin") != "none"
        ):
            raise MethodDispatchError("public argmin rule differs")
    if parent_method == "fedsift":
        control = _mapping(parameters.get("control_rule"), "control_rule")
        sift = _mapping(parameters.get("sift"), "sift")
        if set(control) != {
            "records",
            "labels_visible",
            "step_candidates",
            "query_every_rounds",
            "name",
        } or set(sift) != {"safety_margin_z", "minimum_control_improvement", "fallback_step"}:
            raise MethodDispatchError("FedSift control or support fields differ")
        if control.get("name") != "fedsift_supported_override":
            raise MethodDispatchError("FedSift control rule differs")
        _nonnegative_float(sift.get("safety_margin_z"), "FedSift safety margin z")
        minimum = sift.get("minimum_control_improvement")
        if (
            isinstance(minimum, bool)
            or not isinstance(minimum, (int, float))
            or (not math.isfinite(float(minimum)))
            or (float(minimum) < 0.0)
            or (sift.get("fallback_step") != 1.0)
        ):
            raise MethodDispatchError("FedSift support/fallback parameters differ")
        allowed_top.add("sift")
    if set(parameters) != allowed_top:
        raise MethodDispatchError(
            "candidate top-level fields differ from the registered method schema"
        )
    return backend


def _validated_switch(
    method_id: str, mechanism_switch: Mapping[str, object] | None
) -> tuple[Mapping[str, object] | None, str]:
    if method_id in MAIN_METHODS:
        if mechanism_switch is not None:
            raise MethodDispatchError("main methods cannot carry an ablation switch")
        return (None, canonical_sha256({"applicability": "not_applicable_main_method"}))
    expected = MATCHED_ABLATIONS[method_id]["mechanism_switch"]
    if not isinstance(mechanism_switch, Mapping) or dict(mechanism_switch) != expected:
        raise MethodDispatchError("matched ablation switch differs from preregistration")
    payload = copy.deepcopy(dict(mechanism_switch))
    return (payload, canonical_sha256(payload))


def build_method_execution_spec(
    *,
    method_id: str,
    candidate_parameters: Mapping[str, object],
    mechanism_switch: Mapping[str, object] | None = None,
) -> MethodExecutionSpec:
    """Resolve one exact method configuration without reading any results."""
    if not isinstance(method_id, str) or method_id not in ALL_METHODS:
        raise MethodDispatchError("unknown method identity; fallback is forbidden")
    parameters = _mapping(candidate_parameters, "candidate_parameters")
    parent = _exact_parent(method_id)
    backend = _validate_common_candidate(parent, parameters)
    _, switch_hash = _validated_switch(method_id, mechanism_switch)
    privacy_mode = (
        "nonprivate_explicit_minibatch"
        if parent == "fedavg_nonprivate"
        else "conditional_record_dp_poisson_per_fixed_run"
    )
    correction = {
        "dp_fedprox_adapted": "fedprox_post_privacy_data_independent_correction",
        "dp_scaffold_adapted": "scaffold_post_privacy_prior_state_correction",
    }.get(parent, "none")
    local_handler = (
        "canonical_nonprivate_explicit_minibatches"
        if parent == "fedavg_nonprivate"
        else "PoissonLocalTrainingGate"
    )
    schedule_handler = (
        "not_applicable_nonprivate"
        if parent == "fedavg_nonprivate"
        else "uniform_candidate_specific_calibration"
    )
    if parent in {"time_dpfedadam", "public_argmin_time_dpfedadam", "fedsift"}:
        schedule_handler = "two_phase_time_candidate_specific_calibration"
    if method_id == "fedsift_uniform_schedule":
        schedule_handler = "uniform_candidate_specific_calibration"
    public_control = "none"
    queries_enabled = False
    non_query = "fixed_full_step_alpha_1"
    if parent == "public_argmin_time_dpfedadam":
        public_control = "decide_public_argmin"
        queries_enabled = True
    elif parent == "fedsift":
        public_control = "decide_fedsift"
        queries_enabled = True
    if method_id == "fedsift_without_sift":
        public_control = "none"
        queries_enabled = False
    elif method_id == "fedsift_public_argmin_rule":
        public_control = "decide_public_argmin"
        queries_enabled = True
    scaffold_variant = (
        "option_ii_batched_weighted_registered_client_weights"
        if parent == "dp_scaffold_adapted"
        else "not_applicable"
    )
    sofim_contract = (
        "negative_delta_div_actual_steps_times_local_lr_then_fixed_weight_average"
        if parent == "dp_fedsofim_delta_proxy_adapted"
        else "not_applicable"
    )
    claim_boundary = (
        "adapted_trajectory_average_proxy_not_paper_or_official_repo_reproduction"
        if parent == "dp_fedsofim_delta_proxy_adapted"
        else _identity("registered_method_implementation")
    )
    payload: dict[str, object] = {
        "schema": _identity("method_execution_spec"),
        "method_id": method_id,
        "parent_method": parent,
        "privacy_mode": privacy_mode,
        "local_update_handler": local_handler,
        "correction_handler": correction,
        "server_backend_handler": _SERVER_HANDLER[str(backend["name"])],
        "privacy_schedule_handler": schedule_handler,
        "public_control_handler": public_control,
        "public_control_queries_enabled": queries_enabled,
        "non_query_round_handler": non_query,
        "resource_accounting_method_id": method_id,
        "scaffold_variant": scaffold_variant,
        "sofim_proxy_contract": sofim_contract,
        "claim_boundary": claim_boundary,
        "candidate_parameters_sha256": canonical_sha256(parameters),
        "mechanism_switch_sha256": switch_hash,
    }
    payload["dispatch_sha256"] = canonical_sha256(payload)
    return MethodExecutionSpec(**payload)


def validate_method_execution_spec(
    spec: MethodExecutionSpec,
    *,
    method_id: str,
    candidate_parameters: Mapping[str, object],
    mechanism_switch: Mapping[str, object] | None = None,
    expected_dispatch_sha256: str,
) -> None:
    """Rebuild a dispatch spec from the registered inputs and commitment."""
    if type(spec) is not MethodExecutionSpec:
        raise MethodDispatchError("execution spec has an unrecognized type")
    expected = build_method_execution_spec(
        method_id=method_id,
        candidate_parameters=candidate_parameters,
        mechanism_switch=mechanism_switch,
    )
    if not isinstance(expected_dispatch_sha256, str) or not hmac.compare_digest(
        expected.dispatch_sha256, expected_dispatch_sha256
    ):
        raise MethodDispatchError("dispatch commitment differs")
    if spec != expected:
        raise MethodDispatchError("execution spec differs from exact registered inputs")


def resolve_server_kernel(spec: MethodExecutionSpec) -> Callable[..., object]:
    """Return the registered pure server kernel; never return a default."""
    if type(spec) is not MethodExecutionSpec:
        raise MethodDispatchError("execution spec has an unrecognized type")
    kernel = _SERVER_KERNELS.get(spec.server_backend_handler)
    if kernel is None:
        raise MethodDispatchError("registered server handler has no exact kernel")
    return kernel


__all__ = [
    "ALL_METHODS",
    "MethodDispatchError",
    "MethodExecutionSpec",
    "build_method_execution_spec",
    "resolve_server_kernel",
    "validate_method_execution_spec",
]
