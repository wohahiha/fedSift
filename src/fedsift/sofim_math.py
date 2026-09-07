"""Result-blind SOFIM transform for FedSift's delta-derived gradient proxy.

For each participating client, FedSift defines

    normalized_proxy_i = -delta_i / (actual_local_steps_i * local_lr_i)

where delta_i = local_i - global. The proxies are aggregated with a
precommitted client order and aggregation weights. This quantity is the
average of the gradients encountered along each client's local trajectory
when constant-step SGD is used. It is not the paper's current full-batch
gradient G_t.

The server applies the rank-one inverse from DP-FedSOFIM v3 to that proxy:

    M_t = beta * M_(t-1) + (1-beta) * proxy_t
    theta_(t+1) = theta_t - eta * (rho I + M_t M_t^T)^-1 * proxy_t

This is an explicit FedSift adaptation. It is neither a reproduction of the
paper's gradient-level algorithm nor a reproduction of the official
repository's bias-corrected Mhat direction and warmup behavior.
"""

from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import hashlib
import json
import math
import re
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any
import torch

State = OrderedDict[str, torch.Tensor]
SOFIM_METHOD_ID = "dp_fedsofim_delta_proxy_adapted"
SOFIM_PROXY_DEFINITION = "normalized_proxy_i=-delta_i/(actual_local_steps_i*local_lr_i)"
SOFIM_PROXY_INTERPRETATION = (
    "local_trajectory_average_gradient_proxy_not_current_full_batch_gradient"
)
SOFIM_VARIANT = "paper_v3_rank_one_on_normalized_local_trajectory_proxy"
_STATE_SCHEMA = _identity("sofim_state")
_TENSOR_MAPPING_SCHEMA = _identity("sofim_tensor_mapping")
_AGGREGATION_PLAN_SCHEMA = _identity("sofim_aggregation_plan")
_PROXY_RECEIPT_SCHEMA = _identity("sofim_normalized_proxy_receipt")
_STEP_RECEIPT_SCHEMA = _identity("sofim_server_step_receipt")
_SHA256_RE = re.compile("^[0-9a-f]{64}$")
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
_SUPPORTED_DEVICE_TYPES = {"cpu", "cuda"}
_PROXY_RECEIPT_FIELDS = {
    "schema",
    "status",
    "method_id",
    "proxy_definition",
    "interpretation_boundary",
    "client_order",
    "aggregation_plan_sha256",
    "aggregation_weight_sum_hex",
    "clients",
    "aggregate_proxy_sha256",
    "receipt_sha256",
}
_PROXY_CLIENT_FIELDS = {
    "client_id",
    "actual_local_steps",
    "local_learning_rate_hex",
    "aggregation_weight_hex",
    "client_delta_sha256",
    "normalized_proxy_sha256",
}
_STEP_RECEIPT_FIELDS = {
    "schema",
    "status",
    "method_id",
    "variant",
    "official_repository_reproduction",
    "interpretation_boundary",
    "normalized_proxy_receipt_sha256",
    "input_model_state_sha256",
    "input_optimizer_state_sha256",
    "aggregate_proxy_sha256",
    "beta_hex",
    "rho_hex",
    "server_learning_rate_hex",
    "direction_sha256",
    "output_model_state_sha256",
    "output_optimizer_state_sha256",
    "receipt_sha256",
}


class SofimMathError(RuntimeError):
    """Raised when the FedSift SOFIM delta-proxy contract is violated."""


def _canonical_sha256(value: object) -> str:
    try:
        rendered = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise SofimMathError("value is not strict canonical JSON") from exc
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SofimMathError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SofimMathError(f"{name} must be a non-empty stripped string")
    return value


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SofimMathError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise SofimMathError(f"{name} must be finite and positive")
    return number


def _finite_weight(value: Any, name: str) -> float:
    number = _finite_positive(value, name)
    if number > 1.0:
        raise SofimMathError(f"{name} must not exceed one")
    return number


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise SofimMathError(f"{name} must be an integer")
    number = int(value)
    if number <= 0:
        raise SofimMathError(f"{name} must be positive")
    return number


def _beta(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SofimMathError("beta must be a real number")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number < 1.0:
        raise SofimMathError("beta must be finite and in [0,1)")
    return number


def _tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise SofimMathError(f"{name} is not a tensor")
    if value.dtype not in _SUPPORTED_DTYPES:
        raise SofimMathError(f"{name} must use a supported real floating dtype")
    if value.layout != torch.strided or value.device.type not in _SUPPORTED_DEVICE_TYPES:
        raise SofimMathError(f"{name} must be a materialized dense tensor")
    try:
        finite = bool(torch.isfinite(value.detach()).all().item())
    except RuntimeError as exc:
        raise SofimMathError(f"{name} cannot be checked for finiteness") from exc
    if not finite:
        raise SofimMathError(f"{name} contains a non-finite value")
    return value


def _copy_tensor_mapping(state: Any, name: str) -> State:
    if not isinstance(state, Mapping) or len(state) == 0:
        raise SofimMathError(f"{name} must be a non-empty mapping")
    copied: State = OrderedDict()
    reference_dtype: torch.dtype | None = None
    reference_device: torch.device | None = None
    for raw_key, raw_value in state.items():
        key = _identifier(raw_key, f"{name} parameter name")
        value = _tensor(raw_value, f"{name}[{key}]")
        if reference_dtype is None:
            reference_dtype = value.dtype
            reference_device = value.device
        elif value.dtype != reference_dtype or value.device != reference_device:
            raise SofimMathError(f"{name} parameters must share one dtype and one device")
        copied[key] = value.detach().clone()
    return copied


def _same_structure(
    base_state: Mapping[str, torch.Tensor],
    proxy_state: Mapping[str, torch.Tensor],
    moment_state: Mapping[str, torch.Tensor],
) -> tuple[str, ...]:
    names = tuple(base_state)
    if tuple(proxy_state) != names or tuple(moment_state) != names:
        raise SofimMathError("state parameter identity or order differs")
    for name in names:
        base = base_state[name]
        proxy = proxy_state[name]
        moment = moment_state[name]
        if base.shape != proxy.shape or base.shape != moment.shape:
            raise SofimMathError("state tensor shape differs")
        if base.dtype != proxy.dtype or base.dtype != moment.dtype:
            raise SofimMathError("state tensor dtype differs")
        if base.device != proxy.device or base.device != moment.device:
            raise SofimMathError("state tensor device differs")
    return names


def _client_order(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SofimMathError(f"{name} must be a sequence")
    order = tuple((_identifier(item, f"{name} client id") for item in value))
    if not order or len(order) != len(set(order)):
        raise SofimMathError(f"{name} must contain unique client ids")
    return order


def _ordered_client_mapping(
    value: Any, expected_order: tuple[str, ...], name: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SofimMathError(f"{name} must be a mapping")
    if tuple(value) != expected_order:
        raise SofimMathError(f"{name} client identity or order differs")
    return value


def _tensor_fingerprint(tensor: torch.Tensor) -> str:
    value = _tensor(tensor, "tensor fingerprint input").detach().cpu().contiguous()
    flattened = [float(item).hex() for item in value.reshape(-1).tolist()]
    return _canonical_sha256(
        {"dtype": str(value.dtype), "shape": list(value.shape), "values": flattened}
    )


def sofim_tensor_mapping_sha256(state: Mapping[str, torch.Tensor], *, role: str) -> str:
    """Hash an ordered tensor mapping; device identity is intentionally omitted."""
    role = _identifier(role, "tensor mapping role")
    copied = _copy_tensor_mapping(state, role)
    return _canonical_sha256(
        {
            "schema": _TENSOR_MAPPING_SCHEMA,
            "role": role,
            "parameters": [
                {"name": name, "tensor_sha256": _tensor_fingerprint(value)}
                for (name, value) in copied.items()
            ],
        }
    )


@dataclass(frozen=True, slots=True, init=False, eq=False)
class SofimState:
    """Immutable-by-API server EMA state for the normalized gradient proxy."""

    _moment_items: tuple[tuple[str, torch.Tensor], ...]

    def __init__(self, moment: Mapping[str, torch.Tensor]) -> None:
        copied = _copy_tensor_mapping(moment, "moment")
        object.__setattr__(
            self,
            "_moment_items",
            tuple(((name, value.detach().clone()) for (name, value) in copied.items())),
        )

    def __getattribute__(self, name: str) -> object:
        if name == "_moment_items":
            raise AttributeError("SOFIM state storage is private")
        return object.__getattribute__(self, name)

    @property
    def moment(self) -> State:
        return self._materialize_for_kernel()

    @property
    def parameter_names(self) -> tuple[str, ...]:
        items = object.__getattribute__(self, "_moment_items")
        return tuple((name for (name, _) in items))

    def _materialize_for_kernel(self) -> State:
        items = object.__getattribute__(self, "_moment_items")
        return OrderedDict(((name, value.detach().clone()) for (name, value) in items))

    def __repr__(self) -> str:
        return f"SofimState(parameter_names={self.parameter_names!r})"

    def __reduce_ex__(self, protocol: int) -> object:
        raise TypeError("SofimState custom-object pickling is disabled; use sofim_serialize_state")


def sofim_state_sha256(state: SofimState) -> str:
    if not isinstance(state, SofimState):
        raise SofimMathError("state must be SofimState")
    return sofim_tensor_mapping_sha256(
        state._materialize_for_kernel(), role="sofim_optimizer_moment"
    )


def sofim_init_like(params: Mapping[str, torch.Tensor]) -> SofimState:
    """Initialize M_-1 = 0 with the exact parameter structure."""
    base = _copy_tensor_mapping(params, "params")
    return SofimState(
        OrderedDict(((name, torch.zeros_like(value)) for (name, value) in base.items()))
    )


def _mapping_metadata(state: Mapping[str, torch.Tensor]) -> dict[str, object]:
    return {
        "parameter_order": list(state),
        "parameters": [
            {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)}
            for (name, value) in state.items()
        ],
        "device_policy": "device_not_hashed_map_location_allowed",
    }


def sofim_serialize_state(state: SofimState) -> dict[str, object]:
    """Return a weights-only-safe plain mapping instead of pickling SofimState."""
    if not isinstance(state, SofimState):
        raise SofimMathError("state must be SofimState")
    moment = state._materialize_for_kernel()
    return {
        "schema": _STATE_SCHEMA,
        "metadata": _mapping_metadata(moment),
        "moment": {name: value.detach().clone() for (name, value) in moment.items()},
        "state_sha256": sofim_state_sha256(state),
    }


def sofim_deserialize_state(
    payload: Mapping[str, object], *, expected_state_sha256: str
) -> SofimState:
    """Restore a state against a separately committed state hash."""
    external_hash = _require_sha256(expected_state_sha256, "expected_state_sha256")
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema",
        "metadata",
        "moment",
        "state_sha256",
    }:
        raise SofimMathError("serialized state fields differ")
    if payload.get("schema") != _STATE_SCHEMA:
        raise SofimMathError("serialized state schema differs")
    moment = _copy_tensor_mapping(payload.get("moment"), "serialized moment")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping) or dict(metadata) != _mapping_metadata(moment):
        raise SofimMathError("serialized state metadata differs")
    expected_hash = _require_sha256(payload.get("state_sha256"), "state_sha256")
    if expected_hash != external_hash:
        raise SofimMathError("serialized state differs from external commitment")
    restored = SofimState(moment)
    if sofim_state_sha256(restored) != expected_hash:
        raise SofimMathError("serialized state hash differs")
    return restored


def _validated_plan_inputs(
    expected_client_order: Sequence[str], aggregation_weights: Mapping[str, Real]
) -> tuple[tuple[str, ...], tuple[float, ...], float]:
    order = _client_order(expected_client_order, "expected_client_order")
    weights_mapping = _ordered_client_mapping(aggregation_weights, order, "aggregation_weights")
    weights = tuple(
        (_finite_weight(weights_mapping[client_id], f"weight[{client_id}]") for client_id in order)
    )
    weight_sum = math.fsum(weights)
    if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=8.0 * math.ulp(1.0)):
        raise SofimMathError("aggregation weights must sum to one")
    return (order, weights, weight_sum)


def sofim_aggregation_plan_sha256(
    expected_client_order: Sequence[str], aggregation_weights: Mapping[str, Real]
) -> str:
    """Hash the client order and weights that must be fixed before deltas exist."""
    order, weights, _ = _validated_plan_inputs(expected_client_order, aggregation_weights)
    return _canonical_sha256(
        {
            "schema": _AGGREGATION_PLAN_SCHEMA,
            "client_order": list(order),
            "aggregation_weight_hex": [value.hex() for value in weights],
        }
    )


def _receipt_sha256(receipt: Mapping[str, object], hash_field: str) -> str:
    payload = dict(receipt)
    payload.pop(hash_field, None)
    return _canonical_sha256(payload)


def sofim_aggregate_normalized_client_proxy(
    *,
    expected_client_order: Sequence[str],
    client_deltas: Mapping[str, Mapping[str, torch.Tensor]],
    actual_local_steps: Mapping[str, int],
    local_learning_rates: Mapping[str, Real],
    aggregation_weights: Mapping[str, Real],
) -> tuple[State, dict[str, object]]:
    """Build the precommitted weighted trajectory-average gradient proxy."""
    order, weights, weight_sum = _validated_plan_inputs(expected_client_order, aggregation_weights)
    deltas = _ordered_client_mapping(client_deltas, order, "client_deltas")
    steps_mapping = _ordered_client_mapping(actual_local_steps, order, "actual_local_steps")
    lr_mapping = _ordered_client_mapping(local_learning_rates, order, "local_learning_rates")
    plan_hash = sofim_aggregation_plan_sha256(order, aggregation_weights)
    copied_deltas: OrderedDict[str, State] = OrderedDict()
    steps: list[int] = []
    learning_rates: list[float] = []
    parameter_names: tuple[str, ...] | None = None
    reference_state: State | None = None
    for client_id in order:
        delta = _copy_tensor_mapping(deltas[client_id], f"client_deltas[{client_id}]")
        if parameter_names is None:
            parameter_names = tuple(delta)
            reference_state = delta
        else:
            assert reference_state is not None
            _same_structure(reference_state, delta, reference_state)
        copied_deltas[client_id] = delta
        steps.append(_positive_int(steps_mapping[client_id], f"actual_local_steps[{client_id}]"))
        learning_rates.append(
            _finite_positive(lr_mapping[client_id], f"local_learning_rates[{client_id}]")
        )
    assert parameter_names is not None and reference_state is not None
    reference = reference_state[parameter_names[0]]
    accumulator = OrderedDict(
        (
            (
                name,
                torch.zeros(
                    reference_state[name].shape, dtype=torch.float64, device=reference.device
                ),
            )
            for name in parameter_names
        )
    )
    client_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for index, client_id in enumerate(order):
            scale = float(steps[index]) * learning_rates[index]
            if not math.isfinite(scale) or scale <= 0.0:
                raise SofimMathError(f"normalization scale[{client_id}] is invalid")
            normalized64: State = OrderedDict()
            for name in parameter_names:
                value64 = -copied_deltas[client_id][name].to(torch.float64) / scale
                if not bool(torch.isfinite(value64).all().item()):
                    raise SofimMathError(f"normalized proxy[{client_id}] is non-finite")
                normalized64[name] = value64.detach().clone()
                accumulator[name].add_(weights[index] * value64)
            client_rows.append(
                {
                    "client_id": client_id,
                    "actual_local_steps": steps[index],
                    "local_learning_rate_hex": learning_rates[index].hex(),
                    "aggregation_weight_hex": weights[index].hex(),
                    "client_delta_sha256": sofim_tensor_mapping_sha256(
                        copied_deltas[client_id], role=f"client_delta:{client_id}"
                    ),
                    "normalized_proxy_sha256": sofim_tensor_mapping_sha256(
                        normalized64, role=f"normalized_proxy:{client_id}"
                    ),
                }
            )
        aggregate: State = OrderedDict()
        for name in parameter_names:
            value = accumulator[name].to(reference.dtype).detach().clone()
            if not bool(torch.isfinite(value).all().item()):
                raise SofimMathError("aggregate normalized proxy is not representable")
            aggregate[name] = value
    receipt: dict[str, object] = {
        "schema": _PROXY_RECEIPT_SCHEMA,
        "status": "complete",
        "method_id": SOFIM_METHOD_ID,
        "proxy_definition": SOFIM_PROXY_DEFINITION,
        "interpretation_boundary": SOFIM_PROXY_INTERPRETATION,
        "client_order": list(order),
        "aggregation_plan_sha256": plan_hash,
        "aggregation_weight_sum_hex": weight_sum.hex(),
        "clients": client_rows,
        "aggregate_proxy_sha256": sofim_tensor_mapping_sha256(
            aggregate, role="aggregate_normalized_proxy"
        ),
    }
    receipt["receipt_sha256"] = _receipt_sha256(receipt, "receipt_sha256")
    validate_sofim_normalized_proxy_receipt(
        receipt, expected_receipt_sha256=str(receipt["receipt_sha256"])
    )
    return (
        OrderedDict(((name, value.detach().clone()) for (name, value) in aggregate.items())),
        receipt,
    )


def _float_from_canonical_hex(value: object, name: str) -> float:
    if not isinstance(value, str):
        raise SofimMathError(f"{name} must be a canonical float hex string")
    try:
        number = float.fromhex(value)
    except ValueError as exc:
        raise SofimMathError(f"{name} is not a float hex string") from exc
    if not math.isfinite(number) or number.hex() != value:
        raise SofimMathError(f"{name} is not canonical finite float hex")
    return number


def validate_sofim_normalized_proxy_receipt(
    receipt: Mapping[str, object], *, expected_receipt_sha256: str
) -> None:
    """Validate receipt invariants, self-hash, and external commitment."""
    expected = _require_sha256(expected_receipt_sha256, "expected_receipt_sha256")
    if not isinstance(receipt, Mapping) or set(receipt) != _PROXY_RECEIPT_FIELDS:
        raise SofimMathError("normalized proxy receipt fields differ")
    if (
        receipt.get("schema") != _PROXY_RECEIPT_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("method_id") != SOFIM_METHOD_ID
        or (receipt.get("proxy_definition") != SOFIM_PROXY_DEFINITION)
        or (receipt.get("interpretation_boundary") != SOFIM_PROXY_INTERPRETATION)
    ):
        raise SofimMathError("normalized proxy receipt identity differs")
    order = _client_order(receipt.get("client_order"), "receipt client_order")
    _require_sha256(receipt.get("aggregation_plan_sha256"), "aggregation_plan_sha256")
    _require_sha256(receipt.get("aggregate_proxy_sha256"), "aggregate_proxy_sha256")
    rows = receipt.get("clients")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise SofimMathError("normalized proxy receipt clients must be a sequence")
    if len(rows) != len(order):
        raise SofimMathError("normalized proxy receipt client count differs")
    parsed_weights: list[float] = []
    plan_weights: OrderedDict[str, float] = OrderedDict()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != _PROXY_CLIENT_FIELDS:
            raise SofimMathError("normalized proxy client receipt fields differ")
        client_id = _identifier(row.get("client_id"), "receipt client_id")
        if client_id != order[index]:
            raise SofimMathError("normalized proxy receipt client order differs")
        _positive_int(row.get("actual_local_steps"), "receipt actual_local_steps")
        learning_rate = _float_from_canonical_hex(
            row.get("local_learning_rate_hex"), "local_learning_rate_hex"
        )
        _finite_positive(learning_rate, "receipt local learning rate")
        weight = _float_from_canonical_hex(
            row.get("aggregation_weight_hex"), "aggregation_weight_hex"
        )
        _finite_weight(weight, "receipt aggregation weight")
        parsed_weights.append(weight)
        plan_weights[client_id] = weight
        _require_sha256(row.get("client_delta_sha256"), "client_delta_sha256")
        _require_sha256(row.get("normalized_proxy_sha256"), "normalized_proxy_sha256")
    weight_sum = math.fsum(parsed_weights)
    recorded_sum = _float_from_canonical_hex(
        receipt.get("aggregation_weight_sum_hex"), "aggregation_weight_sum_hex"
    )
    if recorded_sum != weight_sum or not math.isclose(
        weight_sum, 1.0, rel_tol=0.0, abs_tol=8.0 * math.ulp(1.0)
    ):
        raise SofimMathError("normalized proxy receipt weight sum differs")
    if receipt.get("aggregation_plan_sha256") != sofim_aggregation_plan_sha256(order, plan_weights):
        raise SofimMathError("normalized proxy receipt aggregation plan differs")
    stored = _require_sha256(receipt.get("receipt_sha256"), "receipt_sha256")
    actual = _receipt_sha256(receipt, "receipt_sha256")
    if stored != actual or stored != expected:
        raise SofimMathError("normalized proxy receipt hash differs")


def sofim_server_step_from_proxy(
    base_state: Mapping[str, torch.Tensor],
    aggregate_proxy: Mapping[str, torch.Tensor],
    optimizer_state: SofimState,
    *,
    beta: float,
    rho: float,
    server_learning_rate: float,
) -> tuple[State, SofimState, State]:
    """Apply H(M_t) to the normalized proxy and take a subtractive step."""
    beta_value = _beta(beta)
    rho_value = _finite_positive(rho, "rho")
    learning_rate = _finite_positive(server_learning_rate, "server_learning_rate")
    if not isinstance(optimizer_state, SofimState):
        raise SofimMathError("optimizer_state must be SofimState")
    base = _copy_tensor_mapping(base_state, "base_state")
    proxy = _copy_tensor_mapping(aggregate_proxy, "aggregate_proxy")
    previous_moment = optimizer_state._materialize_for_kernel()
    names = _same_structure(base, proxy, previous_moment)
    reference = base[names[0]]
    with torch.no_grad():
        next_moment: State = OrderedDict()
        moment64: State = OrderedDict()
        for name in names:
            value64 = beta_value * previous_moment[name].to(torch.float64) + (
                1.0 - beta_value
            ) * proxy[name].to(torch.float64)
            value = value64.to(reference.dtype).detach().clone()
            if not bool(torch.isfinite(value).all().item()):
                raise SofimMathError("SOFIM moment update is not representable")
            next_moment[name] = value
            moment64[name] = value.to(torch.float64)
        norm_sq = torch.zeros((), dtype=torch.float64, device=reference.device)
        moment_dot_proxy = torch.zeros((), dtype=torch.float64, device=reference.device)
        for name in names:
            proxy64 = proxy[name].to(torch.float64)
            norm_sq = norm_sq + torch.sum(moment64[name] * moment64[name])
            moment_dot_proxy = moment_dot_proxy + torch.sum(moment64[name] * proxy64)
        curvature_scale = norm_sq + rho_value
        if not all(
            (
                bool(torch.isfinite(value).item())
                for value in (norm_sq, moment_dot_proxy, curvature_scale)
            )
        ) or bool((curvature_scale <= 0.0).item()):
            raise SofimMathError("SOFIM rank-one curvature scalar is invalid")
        has_curvature_direction = bool((norm_sq > 0.0).item())
        parallel_coefficient = moment_dot_proxy / norm_sq if has_curvature_direction else None
        output: State = OrderedDict()
        direction: State = OrderedDict()
        for name in names:
            proxy64 = proxy[name].to(torch.float64)
            if parallel_coefficient is None:
                preconditioned64 = proxy64 / rho_value
            else:
                parallel64 = moment64[name] * parallel_coefficient
                orthogonal64 = proxy64 - parallel64
                preconditioned64 = orthogonal64 / rho_value + parallel64 / curvature_scale
            updated64 = base[name].to(torch.float64) - learning_rate * preconditioned64
            if not bool(torch.isfinite(preconditioned64).all().item()) or not bool(
                torch.isfinite(updated64).all().item()
            ):
                raise SofimMathError("SOFIM server update is non-finite")
            preconditioned = preconditioned64.to(reference.dtype).detach().clone()
            updated = updated64.to(reference.dtype).detach().clone()
            if not bool(torch.isfinite(preconditioned).all().item()) or not bool(
                torch.isfinite(updated).all().item()
            ):
                raise SofimMathError("SOFIM server update is not representable")
            direction[name] = preconditioned
            output[name] = updated
    state = SofimState(next_moment)
    return (
        OrderedDict(((name, value.detach().clone()) for (name, value) in output.items())),
        state,
        OrderedDict(((name, value.detach().clone()) for (name, value) in direction.items())),
    )


def sofim_server_step_with_receipt(
    base_state: Mapping[str, torch.Tensor],
    aggregate_proxy: Mapping[str, torch.Tensor],
    optimizer_state: SofimState,
    *,
    beta: float,
    rho: float,
    server_learning_rate: float,
    normalized_proxy_receipt: Mapping[str, object],
    expected_normalized_proxy_receipt_sha256: str,
) -> tuple[State, SofimState, State, dict[str, object]]:
    """Execute one server step and bind its exact inputs and outputs."""
    proxy_receipt_hash = _require_sha256(
        expected_normalized_proxy_receipt_sha256, "expected_normalized_proxy_receipt_sha256"
    )
    validate_sofim_normalized_proxy_receipt(
        normalized_proxy_receipt, expected_receipt_sha256=proxy_receipt_hash
    )
    beta_value = _beta(beta)
    rho_value = _finite_positive(rho, "rho")
    learning_rate = _finite_positive(server_learning_rate, "server_learning_rate")
    input_model_hash = sofim_tensor_mapping_sha256(base_state, role="sofim_input_model")
    input_optimizer_hash = sofim_state_sha256(optimizer_state)
    proxy_hash = sofim_tensor_mapping_sha256(aggregate_proxy, role="aggregate_normalized_proxy")
    if normalized_proxy_receipt.get("aggregate_proxy_sha256") != proxy_hash:
        raise SofimMathError("aggregate proxy differs from its normalized-proxy receipt")
    output, next_state, direction = sofim_server_step_from_proxy(
        base_state,
        aggregate_proxy,
        optimizer_state,
        beta=beta_value,
        rho=rho_value,
        server_learning_rate=learning_rate,
    )
    receipt: dict[str, object] = {
        "schema": _STEP_RECEIPT_SCHEMA,
        "status": "complete",
        "method_id": SOFIM_METHOD_ID,
        "variant": SOFIM_VARIANT,
        "official_repository_reproduction": False,
        "interpretation_boundary": SOFIM_PROXY_INTERPRETATION,
        "normalized_proxy_receipt_sha256": proxy_receipt_hash,
        "input_model_state_sha256": input_model_hash,
        "input_optimizer_state_sha256": input_optimizer_hash,
        "aggregate_proxy_sha256": proxy_hash,
        "beta_hex": beta_value.hex(),
        "rho_hex": rho_value.hex(),
        "server_learning_rate_hex": learning_rate.hex(),
        "direction_sha256": sofim_tensor_mapping_sha256(
            direction, role="sofim_preconditioned_direction"
        ),
        "output_model_state_sha256": sofim_tensor_mapping_sha256(output, role="sofim_output_model"),
        "output_optimizer_state_sha256": sofim_state_sha256(next_state),
    }
    receipt["receipt_sha256"] = _receipt_sha256(receipt, "receipt_sha256")
    validate_sofim_step_receipt(
        receipt,
        expected_receipt_sha256=str(receipt["receipt_sha256"]),
        normalized_proxy_receipt=normalized_proxy_receipt,
        expected_normalized_proxy_receipt_sha256=proxy_receipt_hash,
    )
    return (output, next_state, direction, receipt)


def validate_sofim_step_receipt(
    receipt: Mapping[str, object],
    *,
    expected_receipt_sha256: str,
    normalized_proxy_receipt: Mapping[str, object],
    expected_normalized_proxy_receipt_sha256: str,
) -> None:
    """Validate a server-step receipt and its committed proxy-receipt link."""
    expected = _require_sha256(expected_receipt_sha256, "expected_receipt_sha256")
    expected_proxy_receipt_hash = _require_sha256(
        expected_normalized_proxy_receipt_sha256, "expected_normalized_proxy_receipt_sha256"
    )
    validate_sofim_normalized_proxy_receipt(
        normalized_proxy_receipt, expected_receipt_sha256=expected_proxy_receipt_hash
    )
    if not isinstance(receipt, Mapping) or set(receipt) != _STEP_RECEIPT_FIELDS:
        raise SofimMathError("SOFIM step receipt fields differ")
    if (
        receipt.get("schema") != _STEP_RECEIPT_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("method_id") != SOFIM_METHOD_ID
        or (receipt.get("variant") != SOFIM_VARIANT)
        or (receipt.get("official_repository_reproduction") is not False)
        or (receipt.get("interpretation_boundary") != SOFIM_PROXY_INTERPRETATION)
    ):
        raise SofimMathError("SOFIM step receipt identity differs")
    for field in (
        "normalized_proxy_receipt_sha256",
        "input_model_state_sha256",
        "input_optimizer_state_sha256",
        "aggregate_proxy_sha256",
        "direction_sha256",
        "output_model_state_sha256",
        "output_optimizer_state_sha256",
    ):
        _require_sha256(receipt.get(field), field)
    if receipt.get("normalized_proxy_receipt_sha256") != expected_proxy_receipt_hash or receipt.get(
        "aggregate_proxy_sha256"
    ) != normalized_proxy_receipt.get("aggregate_proxy_sha256"):
        raise SofimMathError("SOFIM step receipt proxy linkage differs")
    beta_value = _float_from_canonical_hex(receipt.get("beta_hex"), "beta_hex")
    _beta(beta_value)
    for field in ("rho_hex", "server_learning_rate_hex"):
        _finite_positive(_float_from_canonical_hex(receipt.get(field), field), field)
    stored = _require_sha256(receipt.get("receipt_sha256"), "receipt_sha256")
    actual = _receipt_sha256(receipt, "receipt_sha256")
    if stored != actual or stored != expected:
        raise SofimMathError("SOFIM step receipt hash differs")


__all__ = [
    "SOFIM_METHOD_ID",
    "SOFIM_PROXY_DEFINITION",
    "SOFIM_PROXY_INTERPRETATION",
    "SOFIM_VARIANT",
    "SofimMathError",
    "SofimState",
    "sofim_aggregate_normalized_client_proxy",
    "sofim_aggregation_plan_sha256",
    "sofim_deserialize_state",
    "sofim_init_like",
    "sofim_serialize_state",
    "sofim_server_step_from_proxy",
    "sofim_server_step_with_receipt",
    "sofim_state_sha256",
    "sofim_tensor_mapping_sha256",
    "validate_sofim_normalized_proxy_receipt",
    "validate_sofim_step_receipt",
]
