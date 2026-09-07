"""Pure paper-aligned optimizer kernels and result-blind comparison helpers.

These functions contain no random-number generation, dataset access, file I/O,
or global configuration.  Training runners must provide explicit batches,
gradients, client weights, and optimizer state.
"""

from __future__ import annotations
import math
from collections import OrderedDict
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Callable, Mapping, Sequence
import numpy as np
import torch

State = OrderedDict[str, torch.Tensor]


class BaselineMathError(RuntimeError):
    """Raised when a paper-aligned pure-kernel contract is violated."""


def _positive_exact_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise BaselineMathError(f"{name} must be a positive exact integer")
    return int(value)


def _positive_finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise BaselineMathError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise BaselineMathError(f"{name} must be finite and positive")
    return number


def _validate_tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise BaselineMathError(f"{name} is not a tensor")
    if not value.dtype.is_floating_point:
        raise BaselineMathError(f"{name} must use a floating-point dtype")
    if not torch.isfinite(value).all():
        raise BaselineMathError(f"{name} contains a non-finite value")
    return value


def _state_copy(state: Mapping[str, torch.Tensor]) -> State:
    if not state:
        raise BaselineMathError("state must not be empty")
    result: State = OrderedDict()
    for name, value in state.items():
        tensor = _validate_tensor(value, f"state[{name}]").detach().clone()
        result[str(name)] = tensor
    return result


def _same_structure(*states: Mapping[str, torch.Tensor]) -> tuple[str, ...]:
    if not states:
        raise BaselineMathError("at least one state is required")
    names = tuple(states[0])
    if not names:
        raise BaselineMathError("state must not be empty")
    reference = {name: _validate_tensor(states[0][name], f"state[0][{name}]") for name in names}
    reference_shapes = {name: tuple(reference[name].shape) for name in names}
    reference_dtypes = {name: reference[name].dtype for name in names}
    reference_devices = {name: reference[name].device for name in names}
    for state_index, state in enumerate(states):
        if tuple(state) != names:
            raise BaselineMathError("state parameter order differs")
        for name in names:
            tensor = _validate_tensor(state[name], f"state[{state_index}][{name}]")
            if tuple(tensor.shape) != reference_shapes[name]:
                raise BaselineMathError("state tensor shape differs")
            if tensor.dtype != reference_dtypes[name] or tensor.device != reference_devices[name]:
                raise BaselineMathError("state tensor dtype or device differs")
    return names


def local_sgd_explicit_batches(
    start_state: Mapping[str, torch.Tensor],
    batch_plan_by_epoch: Sequence[Sequence[Any]],
    batch_gradient_fn: Callable[[Mapping[str, torch.Tensor], Any], Mapping[str, torch.Tensor]],
    learning_rate: float,
) -> State:
    """Canonical local SGD over an explicit, caller-frozen minibatch plan."""
    learning_rate = _positive_finite_float(learning_rate, "local learning rate")
    if not batch_plan_by_epoch or any((not epoch for epoch in batch_plan_by_epoch)):
        raise BaselineMathError("every local epoch must contain an explicit batch")
    state = _state_copy(start_state)
    for epoch in batch_plan_by_epoch:
        for batch in epoch:
            gradient = batch_gradient_fn(state, batch)
            names = _same_structure(state, gradient)
            updated: State = OrderedDict()
            for name in names:
                value = state[name] - learning_rate * gradient[name]
                if not torch.isfinite(value).all():
                    raise BaselineMathError("local SGD produced a non-finite state")
                updated[name] = value
            state = updated
    return state


def fedavg_server_average(
    local_states: Sequence[Mapping[str, torch.Tensor]], client_example_counts: Sequence[int]
) -> State:
    """McMahan-style example-count-weighted model averaging."""
    if not local_states or len(local_states) != len(client_example_counts):
        raise BaselineMathError("local states and example counts must align")
    counts = [_positive_exact_int(value, "client example count") for value in client_example_counts]
    names = _same_structure(*local_states)
    total = float(sum(counts))
    output: State = OrderedDict()
    for name in names:
        accumulator = torch.zeros_like(local_states[0][name])
        for state, count in zip(local_states, counts):
            accumulator = accumulator + state[name] * (float(count) / total)
        if not torch.isfinite(accumulator).all():
            raise BaselineMathError("FedAvg produced a non-finite state")
        output[name] = accumulator
    return output


@dataclass(frozen=True)
class FedAdamState:
    m: State
    v: State


@dataclass(frozen=True)
class FedYogiState:
    m: State
    v: State


def fedadam_init_like(params: Mapping[str, torch.Tensor], tau: float) -> FedAdamState:
    """Paper-aligned initialization with ``m_-1=0`` and ``v_-1=tau^2``."""
    tau = _positive_finite_float(tau, "FedAdam tau")
    base = _state_copy(params)
    state = FedAdamState(
        m=OrderedDict(((name, torch.zeros_like(value)) for (name, value) in base.items())),
        v=OrderedDict(((name, torch.full_like(value, tau**2)) for (name, value) in base.items())),
    )
    if any((not torch.all(value > 0.0) for value in state.v.values())):
        raise BaselineMathError("FedAdam tau squared underflows in the parameter dtype")
    return state


def fedadam_server_step(
    base_state: Mapping[str, torch.Tensor],
    aggregate_client_delta: Mapping[str, torch.Tensor],
    optimizer_state: FedAdamState,
    *,
    beta1: float,
    beta2: float,
    server_learning_rate: float,
    tau: float,
) -> tuple[State, FedAdamState]:
    """FedAdam step for deltas defined as ``local_model - global_model``."""
    if (
        isinstance(beta1, bool)
        or not isinstance(beta1, Real)
        or isinstance(beta2, bool)
        or (not isinstance(beta2, Real))
    ):
        raise BaselineMathError("FedAdam beta values must be real numbers")
    beta1 = float(beta1)
    beta2 = float(beta2)
    if not math.isfinite(beta1) or not math.isfinite(beta2):
        raise BaselineMathError("FedAdam beta values must be finite")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise BaselineMathError("FedAdam beta values must be in [0,1)")
    server_learning_rate = _positive_finite_float(
        server_learning_rate, "FedAdam server learning rate"
    )
    tau = _positive_finite_float(tau, "FedAdam tau")
    names = _same_structure(
        base_state, aggregate_client_delta, optimizer_state.m, optimizer_state.v
    )
    if any((torch.any(optimizer_state.v[name] < 0.0) for name in names)):
        raise BaselineMathError("FedAdam second moment must be nonnegative")
    output: State = OrderedDict()
    next_m: State = OrderedDict()
    next_v: State = OrderedDict()
    for name in names:
        m_value = beta1 * optimizer_state.m[name] + (1.0 - beta1) * aggregate_client_delta[name]
        v_value = beta2 * optimizer_state.v[name] + (1.0 - beta2) * aggregate_client_delta[
            name
        ].pow(2)
        direction = server_learning_rate * m_value / (torch.sqrt(v_value) + tau)
        value = base_state[name] + direction
        if not torch.isfinite(value).all() or not torch.isfinite(v_value).all():
            raise BaselineMathError("FedAdam produced a non-finite state")
        output[name] = value
        next_m[name] = m_value
        next_v[name] = v_value
    return (output, FedAdamState(m=next_m, v=next_v))


def fedyogi_init_like(params: Mapping[str, torch.Tensor], tau: float) -> FedYogiState:
    """FedOpt initialization with ``m_-1=0`` and ``v_-1=tau^2``."""
    adam_state = fedadam_init_like(params, tau)
    return FedYogiState(m=adam_state.m, v=adam_state.v)


def fedyogi_server_step(
    base_state: Mapping[str, torch.Tensor],
    aggregate_client_delta: Mapping[str, torch.Tensor],
    optimizer_state: FedYogiState,
    *,
    beta1: float,
    beta2: float,
    server_learning_rate: float,
    tau: float,
) -> tuple[State, FedYogiState]:
    """FedYogi step for deltas defined as ``local_model - global_model``.

    This is the Yogi second-moment rule in Reddi et al. FedOpt Algorithm 2:
    ``v_t = v_(t-1) - (1-beta2) * delta_t^2 * sign(v_(t-1)-delta_t^2)``.
    """
    if (
        isinstance(beta1, bool)
        or not isinstance(beta1, Real)
        or isinstance(beta2, bool)
        or (not isinstance(beta2, Real))
    ):
        raise BaselineMathError("FedYogi beta values must be real numbers")
    beta1 = float(beta1)
    beta2 = float(beta2)
    if not math.isfinite(beta1) or not math.isfinite(beta2):
        raise BaselineMathError("FedYogi beta values must be finite")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise BaselineMathError("FedYogi beta values must be in [0,1)")
    server_learning_rate = _positive_finite_float(
        server_learning_rate, "FedYogi server learning rate"
    )
    tau = _positive_finite_float(tau, "FedYogi tau")
    names = _same_structure(
        base_state, aggregate_client_delta, optimizer_state.m, optimizer_state.v
    )
    if any((torch.any(optimizer_state.v[name] < 0.0) for name in names)):
        raise BaselineMathError("FedYogi second moment must be nonnegative")
    output: State = OrderedDict()
    next_m: State = OrderedDict()
    next_v: State = OrderedDict()
    for name in names:
        delta = aggregate_client_delta[name]
        squared_delta = delta.pow(2)
        m_value = beta1 * optimizer_state.m[name] + (1.0 - beta1) * delta
        v_value = optimizer_state.v[name] - (1.0 - beta2) * squared_delta * torch.sign(
            optimizer_state.v[name] - squared_delta
        )
        if torch.any(v_value < 0.0):
            raise BaselineMathError("FedYogi produced a negative second moment")
        direction = server_learning_rate * m_value / (torch.sqrt(v_value) + tau)
        value = base_state[name] + direction
        if (
            not torch.isfinite(value).all()
            or not torch.isfinite(m_value).all()
            or (not torch.isfinite(v_value).all())
        ):
            raise BaselineMathError("FedYogi produced a non-finite state")
        output[name] = value
        next_m[name] = m_value
        next_v[name] = v_value
    return (output, FedYogiState(m=next_m, v=next_v))


def scaffold_local_step(
    local_state: Mapping[str, torch.Tensor],
    stochastic_gradient: Mapping[str, torch.Tensor],
    server_control: Mapping[str, torch.Tensor],
    client_control: Mapping[str, torch.Tensor],
    local_learning_rate: float,
) -> State:
    local_learning_rate = _positive_finite_float(
        local_learning_rate, "SCAFFOLD local learning rate"
    )
    names = _same_structure(local_state, stochastic_gradient, server_control, client_control)
    result: State = OrderedDict()
    for name in names:
        value = local_state[name] - local_learning_rate * (
            stochastic_gradient[name] + server_control[name] - client_control[name]
        )
        if not torch.isfinite(value).all():
            raise BaselineMathError("SCAFFOLD local step is non-finite")
        result[name] = value
    return result


def scaffold_option2_control(
    old_client_control: Mapping[str, torch.Tensor],
    server_control: Mapping[str, torch.Tensor],
    round_start_state: Mapping[str, torch.Tensor],
    final_local_state: Mapping[str, torch.Tensor],
    *,
    actual_local_steps: int,
    local_learning_rate: float,
) -> State:
    actual_local_steps = _positive_exact_int(actual_local_steps, "SCAFFOLD actual local steps")
    local_learning_rate = _positive_finite_float(
        local_learning_rate, "SCAFFOLD local learning rate"
    )
    names = _same_structure(
        old_client_control, server_control, round_start_state, final_local_state
    )
    scale = 1.0 / (float(actual_local_steps) * local_learning_rate)
    result: State = OrderedDict()
    for name in names:
        value = (
            old_client_control[name]
            - server_control[name]
            + scale * (round_start_state[name] - final_local_state[name])
        )
        if not torch.isfinite(value).all():
            raise BaselineMathError("SCAFFOLD Option II control is non-finite")
        result[name] = value
    return result


def _state_deltas(
    newer: Sequence[Mapping[str, torch.Tensor]], older: Sequence[Mapping[str, torch.Tensor]]
) -> list[State]:
    if len(newer) != len(older) or not newer:
        raise BaselineMathError("paired state collections must align")
    deltas: list[State] = []
    for new, old in zip(newer, older):
        names = _same_structure(new, old)
        deltas.append(OrderedDict(((name, new[name] - old[name]) for name in names)))
    return deltas


def scaffold_server_original(
    round_start_state: Mapping[str, torch.Tensor],
    server_control: Mapping[str, torch.Tensor],
    model_deltas: Sequence[Mapping[str, torch.Tensor]],
    control_deltas: Sequence[Mapping[str, torch.Tensor]],
    *,
    total_client_count: int,
    server_learning_rate: float,
) -> tuple[State, State]:
    """Original equal-client SCAFFOLD server update (Algorithm 1)."""
    selected = len(model_deltas)
    if selected == 0 or len(control_deltas) != selected:
        raise BaselineMathError("SCAFFOLD selected-client states must align")
    total_client_count = _positive_exact_int(total_client_count, "SCAFFOLD total client count")
    server_learning_rate = _positive_finite_float(
        server_learning_rate, "SCAFFOLD server learning rate"
    )
    if total_client_count < selected:
        raise BaselineMathError("invalid SCAFFOLD participation or server rate")
    names = _same_structure(round_start_state, server_control, *model_deltas, *control_deltas)
    model: State = OrderedDict()
    control: State = OrderedDict()
    for name in names:
        mean_model = sum(
            (value[name] for value in model_deltas), torch.zeros_like(round_start_state[name])
        ) / float(selected)
        mean_control = sum(
            (value[name] for value in control_deltas), torch.zeros_like(server_control[name])
        ) / float(selected)
        model[name] = round_start_state[name] + server_learning_rate * mean_model
        control[name] = (
            server_control[name] + float(selected) / float(total_client_count) * mean_control
        )
        if not torch.isfinite(model[name]).all() or not torch.isfinite(control[name]).all():
            raise BaselineMathError("SCAFFOLD server update is non-finite")
    return (model, control)


def scaffold_server_batched_weighted(
    round_start_state: Mapping[str, torch.Tensor],
    server_control: Mapping[str, torch.Tensor],
    model_deltas: Sequence[Mapping[str, torch.Tensor]],
    control_deltas: Sequence[Mapping[str, torch.Tensor]],
    example_counts: Sequence[int],
    *,
    selected_client_count: int,
    total_client_count: int,
    server_learning_rate: float,
) -> tuple[State, State]:
    """Explicit sample-weighted SCAFFOLD adaptation; not original Algorithm 1."""
    if (
        not model_deltas
        or len(model_deltas) != len(control_deltas)
        or len(model_deltas) != len(example_counts)
        or (selected_client_count != len(model_deltas))
    ):
        raise BaselineMathError("weighted SCAFFOLD collections must align")
    counts = [
        _positive_exact_int(value, "weighted SCAFFOLD example count") for value in example_counts
    ]
    selected_client_count = _positive_exact_int(
        selected_client_count, "weighted SCAFFOLD selected client count"
    )
    total_client_count = _positive_exact_int(
        total_client_count, "weighted SCAFFOLD total client count"
    )
    server_learning_rate = _positive_finite_float(
        server_learning_rate, "weighted SCAFFOLD server learning rate"
    )
    if total_client_count < selected_client_count:
        raise BaselineMathError("invalid weighted SCAFFOLD participation or rate")
    names = _same_structure(round_start_state, server_control, *model_deltas, *control_deltas)
    total_examples = float(sum(counts))
    weights = [float(value) / total_examples for value in counts]
    model: State = OrderedDict()
    control: State = OrderedDict()
    participation = float(selected_client_count) / float(total_client_count)
    for name in names:
        weighted_model = sum(
            (weight * value[name] for (weight, value) in zip(weights, model_deltas)),
            torch.zeros_like(round_start_state[name]),
        )
        weighted_control = sum(
            (weight * value[name] for (weight, value) in zip(weights, control_deltas)),
            torch.zeros_like(server_control[name]),
        )
        model[name] = round_start_state[name] + server_learning_rate * weighted_model
        control[name] = server_control[name] + participation * weighted_control
        if not torch.isfinite(model[name]).all() or not torch.isfinite(control[name]).all():
            raise BaselineMathError("weighted SCAFFOLD server update is non-finite")
    return (model, control)


def score_public_direction_grid(
    base_state: Mapping[str, torch.Tensor],
    single_fedadam_direction: Mapping[str, torch.Tensor],
    public_features: np.ndarray,
    public_labels: np.ndarray,
    *,
    alphas: Sequence[float],
    predict_fn: Callable[[Mapping[str, torch.Tensor], np.ndarray], np.ndarray],
) -> list[dict[str, Any]]:
    """Build one shared candidate table for FedSift and its information-matched arm."""
    names = _same_structure(base_state, single_fedadam_direction)
    if not alphas or len(set((float(value) for value in alphas))) != len(alphas):
        raise BaselineMathError("public direction alphas must be unique and non-empty")
    labels = np.asarray(public_labels, dtype=np.float64).reshape(-1)
    if len(labels) == 0 or not set(np.unique(labels).tolist()).issubset({0.0, 1.0}):
        raise BaselineMathError("public labels must be non-empty and binary")
    features = np.asarray(public_features, dtype=np.float64)
    if (
        features.ndim != 2
        or features.shape[0] != labels.shape[0]
        or (not np.all(np.isfinite(features)))
    ):
        raise BaselineMathError("public features must be a finite row-aligned matrix")
    rows: list[dict[str, Any]] = []
    for order, alpha_value in enumerate(alphas):
        alpha = float(alpha_value)
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise BaselineMathError("public direction alpha must be in [0,1]")
        state = OrderedDict(
            ((name, base_state[name] + alpha * single_fedadam_direction[name]) for name in names)
        )
        probability = np.asarray(predict_fn(state, features), dtype=np.float64).reshape(-1)
        if probability.shape != labels.shape or not np.all(np.isfinite(probability)):
            raise BaselineMathError("public candidate prediction is malformed")
        if np.any(probability < 0.0) or np.any(probability > 1.0):
            raise BaselineMathError("public candidate probability is outside [0,1]")
        clipped = np.clip(probability, 1e-07, 1.0 - 1e-07)
        losses = -(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
        rows.append(
            {
                "order": order,
                "alpha": alpha,
                "mean_log_loss": float(np.mean(losses)),
                "per_record_log_loss": losses.tolist(),
                "probability": probability.tolist(),
            }
        )
    return rows


def select_public_argmin(candidate_table: Sequence[Mapping[str, Any]]) -> float:
    """Select minimum mean log loss, breaking exact ties toward the larger step."""
    if not candidate_table:
        raise BaselineMathError("public candidate table is empty")
    rows: list[tuple[float, float]] = []
    for row in candidate_table:
        loss = float(row["mean_log_loss"])
        alpha = float(row["alpha"])
        if not math.isfinite(loss) or not math.isfinite(alpha):
            raise BaselineMathError("public candidate table contains non-finite values")
        rows.append((loss, -alpha))
    selected = min(range(len(rows)), key=lambda index: rows[index])
    return float(candidate_table[selected]["alpha"])
