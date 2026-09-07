"""Result-blind construction of fair, executable FedSift budgets.

The factory consumes only a registered method execution specification,
candidate design parameters, a shared training policy, and the private row-ID
roster assigned to each participating client.  It has no capability for
reading validation predictions, observed utility, or an outer-test role.

Private methods receive the same result-blind local traversal target:
``ceil(local_epochs / q)`` Poisson optimizer steps per server round.  A record
is therefore sampled approximately ``local_epochs`` times in expectation.
Non-private FedAvg instead receives explicit, complete epochs whose minibatch
size is the nearest positive integer to ``q * n``.  These are different
mechanisms, but both target the same amount of local data traversal without
using performance feedback.
"""

from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import copy
import hashlib
import hmac
import math
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from numbers import Integral, Real
from typing import Mapping, Sequence
from .candidate_space import assert_no_performance_fields, canonical_sha256, require_sha256
from .hpo_plan import MATCHED_ABLATIONS
from .method_dispatch import ALL_METHODS, MethodExecutionSpec, validate_method_execution_spec
from .modeling import MODEL_FAMILIES
from .privacy_accounting import NoiseCalibrationResult, calibrate_two_phase_noise
from .train_unit import ExplicitClientBatchPlan, TrainingBudget


class TrainingBudgetFactoryError(ValueError):
    """Raised when a budget cannot be constructed without hidden discretion."""


_RESULT_FACTORY_SEAL = object()
_OUTER_AUTHORITY_ALIASES = frozenset(
    {"outer", "outer_test", "outer-test", "test", "test_partition"}
)
_REGISTERED_CLIENT_IDS = frozenset((f"client_{index}" for index in range(5)))
_PARTITION_MODES = frozenset({"fixed_nonprivate", "fixed_label_driven_auxiliary_condition"})
_INITIALIZATION_POLICY = "paired_seed_by_training_unit_independent_of_method_and_candidate"
_NONPRIVATE_ORDER_POLICY = (
    "sha256_domain_separated_by_client_round_epoch_independent_of_method_candidate"
)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise TrainingBudgetFactoryError(f"{name} must be a positive exact integer")
    return int(value)


def _nonnegative_seed(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TrainingBudgetFactoryError(f"{name} must be a nonnegative exact integer")
    result = int(value)
    if result < 0 or result >= 2**63:
        raise TrainingBudgetFactoryError(f"{name} must be in [0, 2**63)")
    return result


def _positive_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TrainingBudgetFactoryError(f"{name} must be a finite positive real")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise TrainingBudgetFactoryError(f"{name} must be a finite positive real")
    return result


def _plain_identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any((character in value for character in "\r\n\t"))
    ):
        raise TrainingBudgetFactoryError(f"{name} must be a plain identifier")
    return value


def _policy_payload(policy: "SharedTrainingBudgetPolicy") -> dict[str, object]:
    return {
        "schema": _identity("shared_training_budget_policy"),
        "server_rounds": policy.server_rounds,
        "local_epochs": policy.local_epochs,
        "poisson_sample_rate": policy.poisson_sample_rate,
        "clip_norm": policy.clip_norm,
        "target_epsilon": policy.target_epsilon,
        "target_delta": policy.target_delta,
        "model_family": policy.model_family,
        "participating_client_ids": policy.participating_client_ids,
        "paired_initialization_seed": policy.paired_initialization_seed,
        "nonprivate_order_seed": policy.nonprivate_order_seed,
        "initialization_policy": policy.initialization_policy,
        "nonprivate_order_policy": policy.nonprivate_order_policy,
        "partition_mode": policy.partition_mode,
        "fixed_auxiliary_partition_condition_sha256": policy.fixed_auxiliary_partition_condition_sha256,
    }


@dataclass(frozen=True, slots=True)
class SharedTrainingBudgetPolicy:
    """One externally committed policy shared by every comparison arm."""

    server_rounds: int
    local_epochs: int
    poisson_sample_rate: float
    clip_norm: float
    target_epsilon: float
    target_delta: float
    model_family: str
    participating_client_ids: tuple[str, ...]
    paired_initialization_seed: int
    nonprivate_order_seed: str
    partition_mode: str = "fixed_nonprivate"
    fixed_auxiliary_partition_condition_sha256: str | None = None
    initialization_policy: str = _INITIALIZATION_POLICY
    nonprivate_order_policy: str = _NONPRIVATE_ORDER_POLICY
    policy_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "server_rounds", _positive_int(self.server_rounds, "server_rounds")
        )
        object.__setattr__(self, "local_epochs", _positive_int(self.local_epochs, "local_epochs"))
        q = _positive_real(self.poisson_sample_rate, "poisson_sample_rate")
        if q > 1.0:
            raise TrainingBudgetFactoryError("Poisson sample rate must not exceed one")
        object.__setattr__(self, "poisson_sample_rate", q)
        object.__setattr__(self, "clip_norm", _positive_real(self.clip_norm, "clip_norm"))
        object.__setattr__(
            self, "target_epsilon", _positive_real(self.target_epsilon, "target_epsilon")
        )
        delta = _positive_real(self.target_delta, "target_delta")
        if delta >= 1.0:
            raise TrainingBudgetFactoryError("target_delta must be smaller than one")
        object.__setattr__(self, "target_delta", delta)
        if self.model_family not in MODEL_FAMILIES:
            raise TrainingBudgetFactoryError("model family is outside the frozen roster")
        clients = tuple(
            (
                _plain_identifier(value, "participating client")
                for value in self.participating_client_ids
            )
        )
        if not clients or len(clients) != len(set(clients)):
            raise TrainingBudgetFactoryError("participating clients must be non-empty and unique")
        if any((value.lower() in _OUTER_AUTHORITY_ALIASES for value in clients)):
            raise TrainingBudgetFactoryError(
                "an outer-test authority cannot be a training participant"
            )
        if not set(clients).issubset(_REGISTERED_CLIENT_IDS):
            raise TrainingBudgetFactoryError(
                "a participant is outside the registered five-client training roster"
            )
        object.__setattr__(self, "participating_client_ids", clients)
        object.__setattr__(
            self,
            "paired_initialization_seed",
            _nonnegative_seed(self.paired_initialization_seed, "paired_initialization_seed"),
        )
        object.__setattr__(
            self,
            "nonprivate_order_seed",
            _plain_identifier(self.nonprivate_order_seed, "nonprivate_order_seed"),
        )
        if self.initialization_policy != _INITIALIZATION_POLICY:
            raise TrainingBudgetFactoryError("initialization pairing policy differs")
        if self.nonprivate_order_policy != _NONPRIVATE_ORDER_POLICY:
            raise TrainingBudgetFactoryError("non-private order policy differs")
        if self.partition_mode not in _PARTITION_MODES:
            raise TrainingBudgetFactoryError("partition mode is not registered")
        condition = self.fixed_auxiliary_partition_condition_sha256
        if self.partition_mode == "fixed_label_driven_auxiliary_condition":
            try:
                condition = require_sha256(condition, "fixed partition condition")
            except Exception as exc:
                raise TrainingBudgetFactoryError(
                    "fixed label-driven partition requires an external condition hash"
                ) from exc
        elif condition is not None:
            raise TrainingBudgetFactoryError(
                "a fixed non-private partition cannot carry a condition hash"
            )
        object.__setattr__(self, "fixed_auxiliary_partition_condition_sha256", condition)
        object.__setattr__(self, "policy_sha256", canonical_sha256(_policy_payload(self)))


@dataclass(frozen=True, slots=True)
class TrainingBudgetFactoryResult:
    """Factory-sealed executable budget and its result-blind construction proof."""

    budget: TrainingBudget
    artifact: dict[str, object]
    _factory_seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._factory_seal is not _RESULT_FACTORY_SEAL:
            raise TrainingBudgetFactoryError(
                "budget-factory results must be produced by this module"
            )


def _validate_policy(policy: SharedTrainingBudgetPolicy, *, expected_policy_sha256: str) -> None:
    if type(policy) is not SharedTrainingBudgetPolicy:
        raise TrainingBudgetFactoryError("shared policy has an unrecognized type")
    try:
        expected = require_sha256(expected_policy_sha256, "expected policy hash")
    except Exception as exc:
        raise TrainingBudgetFactoryError("expected policy hash is malformed") from exc
    rebuilt = canonical_sha256(_policy_payload(policy))
    if policy.policy_sha256 != rebuilt or not hmac.compare_digest(rebuilt, expected):
        raise TrainingBudgetFactoryError("shared training policy commitment differs")


def _validated_rows(
    values: Mapping[str, Sequence[int]], policy: SharedTrainingBudgetPolicy
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if not isinstance(values, Mapping):
        raise TrainingBudgetFactoryError("private row roster must be a mapping")
    if tuple(values) != policy.participating_client_ids:
        raise TrainingBudgetFactoryError(
            "private row roster must exactly follow the shared participant order"
        )
    result: list[tuple[str, tuple[int, ...]]] = []
    globally_seen: set[int] = set()
    for client_id in policy.participating_client_ids:
        raw = values[client_id]
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise TrainingBudgetFactoryError("client row IDs must be a sequence")
        rows = tuple(raw)
        if not rows or any(
            (
                isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0
                for value in rows
            )
        ):
            raise TrainingBudgetFactoryError(
                "client row IDs must be non-empty nonnegative exact integers"
            )
        normalized = tuple((int(value) for value in rows))
        if len(normalized) != len(set(normalized)):
            raise TrainingBudgetFactoryError("one client repeats a private row ID")
        overlap = globally_seen.intersection(normalized)
        if overlap:
            raise TrainingBudgetFactoryError(
                "private client row partitions must be globally disjoint"
            )
        globally_seen.update(normalized)
        result.append((client_id, normalized))
    return tuple(result)


def _mechanism_switch(method_id: str) -> Mapping[str, object] | None:
    if method_id not in MATCHED_ABLATIONS:
        return None
    value = MATCHED_ABLATIONS[method_id]["mechanism_switch"]
    if not isinstance(value, Mapping):
        raise TrainingBudgetFactoryError("matched-ablation switch is malformed")
    return copy.deepcopy(dict(value))


def _validate_dispatch_and_parameters(
    spec: MethodExecutionSpec,
    candidate_parameters: Mapping[str, object],
    *,
    expected_dispatch_sha256: str,
    expected_candidate_parameters_sha256: str,
) -> dict[str, object]:
    if type(spec) is not MethodExecutionSpec or spec.method_id not in ALL_METHODS:
        raise TrainingBudgetFactoryError("execution spec has an unrecognized type")
    if not isinstance(candidate_parameters, Mapping):
        raise TrainingBudgetFactoryError("candidate parameters must be a mapping")
    try:
        assert_no_performance_fields(candidate_parameters)
    except Exception as exc:
        raise TrainingBudgetFactoryError(
            "candidate design contains a forbidden observed-output field"
        ) from exc
    parameters = copy.deepcopy(dict(candidate_parameters))
    actual_parameter_hash = canonical_sha256(parameters)
    try:
        expected_parameter_hash = require_sha256(
            expected_candidate_parameters_sha256, "expected candidate-parameter hash"
        )
        expected_dispatch = require_sha256(expected_dispatch_sha256, "expected dispatch hash")
    except Exception as exc:
        raise TrainingBudgetFactoryError("external dispatch binding is malformed") from exc
    if not hmac.compare_digest(actual_parameter_hash, expected_parameter_hash):
        raise TrainingBudgetFactoryError("candidate parameter commitment differs")
    try:
        validate_method_execution_spec(
            spec,
            method_id=spec.method_id,
            candidate_parameters=parameters,
            mechanism_switch=_mechanism_switch(spec.method_id),
            expected_dispatch_sha256=expected_dispatch,
        )
    except Exception as exc:
        raise TrainingBudgetFactoryError(
            "method execution spec differs from the registered candidate"
        ) from exc
    return parameters


def _row_roster_sha256(
    rows: Sequence[tuple[str, tuple[int, ...]]],
) -> tuple[str, tuple[tuple[str, str], ...]]:
    per_client = tuple(
        (
            (
                client_id,
                canonical_sha256(
                    {
                        "schema": _identity("private_client_row_membership"),
                        "client_id": client_id,
                        "row_ids": row_ids,
                    }
                ),
            )
            for (client_id, row_ids) in rows
        )
    )
    return (
        canonical_sha256(
            {
                "schema": _identity("private_client_row_roster"),
                "client_membership_sha256": per_client,
            }
        ),
        per_client,
    )


def _shuffle_digest(
    *, seed: str, client_id: str, server_round: int, local_epoch: int, row_id: int
) -> str:
    return hashlib.sha256(
        f"{_identity('nonprivate_explicit_order_v1')}{seed}\x00{client_id}\x00{server_round}\x00{local_epoch}\x00{row_id}".encode(
            "utf-8"
        )
    ).hexdigest()


def _explicit_batch_plans(
    rows: Sequence[tuple[str, tuple[int, ...]]], policy: SharedTrainingBudgetPolicy
) -> tuple[tuple[ExplicitClientBatchPlan, ...], ...]:
    previous_orders: dict[tuple[str, int], tuple[int, ...]] = {}
    rounds: list[tuple[ExplicitClientBatchPlan, ...]] = []
    for server_round in range(1, policy.server_rounds + 1):
        client_plans: list[ExplicitClientBatchPlan] = []
        for client_id, row_ids in rows:
            n = len(row_ids)
            batch_size = max(1, min(n, int(math.floor(policy.poisson_sample_rate * n + 0.5))))
            epochs: list[tuple[tuple[int, ...], ...]] = []
            for local_epoch in range(1, policy.local_epochs + 1):
                ordered = tuple(
                    sorted(
                        row_ids,
                        key=lambda row_id: (
                            _shuffle_digest(
                                seed=policy.nonprivate_order_seed,
                                client_id=client_id,
                                server_round=server_round,
                                local_epoch=local_epoch,
                                row_id=row_id,
                            ),
                            row_id,
                        ),
                    )
                )
                previous = previous_orders.get((client_id, local_epoch))
                if previous == ordered and len(ordered) > 1:
                    shift_source = _shuffle_digest(
                        seed=policy.nonprivate_order_seed,
                        client_id=client_id,
                        server_round=server_round,
                        local_epoch=local_epoch,
                        row_id=ordered[0],
                    )
                    shift = 1 + int(shift_source[:16], 16) % (len(ordered) - 1)
                    ordered = ordered[shift:] + ordered[:shift]
                previous_orders[client_id, local_epoch] = ordered
                epochs.append(
                    tuple(
                        (
                            ordered[offset : offset + batch_size]
                            for offset in range(0, n, batch_size)
                        )
                    )
                )
            client_plans.append(ExplicitClientBatchPlan(client_id=client_id, epochs=tuple(epochs)))
        rounds.append(tuple(client_plans))
    return tuple(rounds)


def _calibration_geometry(
    spec: MethodExecutionSpec,
    parameters: Mapping[str, object],
    policy: SharedTrainingBudgetPolicy,
    *,
    steps_per_round: int,
) -> tuple[int, int, float, float, int | None]:
    total_steps = policy.server_rounds * steps_per_round
    if total_steps < 2:
        raise TrainingBudgetFactoryError(
            "private noise calibration requires at least two optimizer steps"
        )
    if spec.privacy_schedule_handler == "uniform_candidate_specific_calibration":
        first_steps = steps_per_round if policy.server_rounds > 1 else 1
        if first_steps >= total_steps:
            first_steps = total_steps - 1
        return (total_steps, first_steps, 1.0, 1.0, None)
    if spec.privacy_schedule_handler != "two_phase_time_candidate_specific_calibration":
        raise TrainingBudgetFactoryError("private dispatch has no calibration rule")
    schedule = parameters.get("privacy_schedule")
    if not isinstance(schedule, Mapping) or schedule.get("kind") != "two_phase_time":
        raise TrainingBudgetFactoryError("time-schedule candidate parameters are absent")
    fraction = _positive_real(schedule.get("saving_round_fraction"), "saving_round_fraction")
    if fraction >= 1.0:
        raise TrainingBudgetFactoryError("saving_round_fraction must be below one")
    saving_rounds = int(math.ceil(policy.server_rounds * fraction))
    if saving_rounds <= 0 or saving_rounds >= policy.server_rounds:
        raise TrainingBudgetFactoryError(
            "time schedule leaves no non-empty saving or spending round phase"
        )
    return (
        total_steps,
        saving_rounds * steps_per_round,
        _positive_real(schedule.get("saving_sigma_factor"), "saving_sigma_factor"),
        _positive_real(schedule.get("spending_sigma_factor"), "spending_sigma_factor"),
        saving_rounds,
    )


@lru_cache(maxsize=256)
def _cached_calibration(
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


def _calibration_evidence(
    calibration: NoiseCalibrationResult | None,
    *,
    spec: MethodExecutionSpec,
    steps_per_round: int,
    total_steps: int,
    phase_one_steps: int | None,
    phase_one_factor: float | None,
    phase_two_factor: float | None,
    saving_rounds: int | None,
) -> dict[str, object]:
    if calibration is None:
        return {
            "schema": _identity("budget_noise_calibration"),
            "status": "not_applicable_nonprivate",
            "base_noise_multiplier_hex": (1.0).hex(),
            "observed_outputs_consumed": False,
        }
    return {
        "schema": _identity("budget_noise_calibration"),
        "status": "complete_result_blind",
        "schedule_handler": spec.privacy_schedule_handler,
        "steps_per_round": steps_per_round,
        "total_steps": total_steps,
        "saving_round_count": saving_rounds,
        "phase_one_steps": phase_one_steps,
        "phase_one_factor_hex": float(phase_one_factor).hex(),
        "phase_two_factor_hex": float(phase_two_factor).hex(),
        "base_noise_multiplier_hex": float(calibration.base_noise_multiplier).hex(),
        "lower_infeasible_bound_hex": float(calibration.lower_infeasible_bound).hex(),
        "upper_feasible_bound_hex": float(calibration.upper_feasible_bound).hex(),
        "calibrated_epsilon_upper_hex": float(calibration.report.epsilon_prv_upper).hex(),
        "bisection_iterations": int(calibration.bisection_iterations),
        "bracket_expansions": int(calibration.bracket_expansions),
        "noise_relative_tolerance_hex": float(calibration.noise_relative_tolerance).hex(),
        "result_blind": bool(calibration.result_blind),
        "observed_outputs_consumed": False,
    }


def _construct_training_budget(
    spec: MethodExecutionSpec,
    candidate_parameters: Mapping[str, object],
    private_row_ids_by_client: Mapping[str, Sequence[int]],
    policy: SharedTrainingBudgetPolicy,
    *,
    expected_dispatch_sha256: str,
    expected_candidate_parameters_sha256: str,
    expected_policy_sha256: str,
) -> TrainingBudgetFactoryResult:
    _validate_policy(policy, expected_policy_sha256=expected_policy_sha256)
    parameters = _validate_dispatch_and_parameters(
        spec,
        candidate_parameters,
        expected_dispatch_sha256=expected_dispatch_sha256,
        expected_candidate_parameters_sha256=expected_candidate_parameters_sha256,
    )
    rows = _validated_rows(private_row_ids_by_client, policy)
    roster_hash, client_membership_hashes = _row_roster_sha256(rows)
    explicit_plans = _explicit_batch_plans(rows, policy)
    steps_per_round = int(math.ceil(policy.local_epochs / policy.poisson_sample_rate))
    dp_steps = tuple((steps_per_round for _ in range(policy.server_rounds)))
    calibration: NoiseCalibrationResult | None = None
    total_steps = policy.server_rounds * steps_per_round
    phase_one_steps: int | None = None
    first_factor: float | None = None
    second_factor: float | None = None
    saving_rounds: int | None = None
    base_noise_multiplier = 1.0
    if spec.privacy_mode != "nonprivate_explicit_minibatch":
        total_steps, phase_one_steps, first_factor, second_factor, saving_rounds = (
            _calibration_geometry(spec, parameters, policy, steps_per_round=steps_per_round)
        )
        calibration = _cached_calibration(
            policy.poisson_sample_rate,
            total_steps,
            phase_one_steps,
            first_factor,
            second_factor,
            policy.target_epsilon,
            policy.target_delta,
        )
        base_noise_multiplier = float(calibration.base_noise_multiplier)
    participant_roster = tuple(
        (policy.participating_client_ids for _ in range(policy.server_rounds))
    )
    budget = TrainingBudget(
        server_rounds=policy.server_rounds,
        dp_optimizer_steps_by_round=dp_steps,
        nonprivate_batch_plans_by_round=explicit_plans,
        participating_clients_by_round=participant_roster,
        poisson_sample_rate=policy.poisson_sample_rate,
        base_noise_multiplier=base_noise_multiplier,
        clip_norm=policy.clip_norm,
        target_epsilon=policy.target_epsilon,
        target_delta=policy.target_delta,
        model_family=policy.model_family,
        initialization_seed=policy.paired_initialization_seed,
        partition_mode=policy.partition_mode,
        fixed_auxiliary_partition_condition_sha256=policy.fixed_auxiliary_partition_condition_sha256,
    )
    client_traversal = []
    for client_id, row_ids in rows:
        n = len(row_ids)
        client_traversal.append(
            {
                "client_id": client_id,
                "private_record_count": n,
                "nearest_nonprivate_batch_size": max(
                    1, min(n, int(math.floor(policy.poisson_sample_rate * n + 0.5)))
                ),
                "nonprivate_exact_record_evaluations_per_round": policy.local_epochs * n,
                "private_expected_record_evaluations_per_round_hex": (
                    policy.poisson_sample_rate * steps_per_round * n
                ).hex(),
            }
        )
    calibration_artifact = _calibration_evidence(
        calibration,
        spec=spec,
        steps_per_round=steps_per_round,
        total_steps=total_steps,
        phase_one_steps=phase_one_steps,
        phase_one_factor=first_factor,
        phase_two_factor=second_factor,
        saving_rounds=saving_rounds,
    )
    plan_hash = canonical_sha256(
        tuple((tuple((asdict(plan) for plan in round_plans)) for round_plans in explicit_plans))
    )
    artifact: dict[str, object] = {
        "schema": _identity("training_budget_factory_result"),
        "status": "complete_result_blind",
        "method_id": spec.method_id,
        "parent_method": spec.parent_method,
        "matched_parent_candidate_rules": spec.method_id in MATCHED_ABLATIONS,
        "candidate_parameters_sha256": spec.candidate_parameters_sha256,
        "dispatch_sha256": spec.dispatch_sha256,
        "shared_policy_sha256": policy.policy_sha256,
        "private_row_roster_sha256": roster_hash,
        "client_membership_sha256": [
            {"client_id": client_id, "membership_sha256": digest}
            for (client_id, digest) in client_membership_hashes
        ],
        "budget_sha256": budget.budget_sha256,
        "dp_step_rule": "ceil_local_epochs_divided_by_poisson_q_per_round",
        "dp_optimizer_steps_per_round": steps_per_round,
        "dp_optimizer_steps_by_round": list(dp_steps),
        "traversal_budget_interpretation": "private_expected_evaluations_approximately_E_times_n_nonprivate_exactly_E_times_n_per_round",
        "client_traversal_contract": client_traversal,
        "nonprivate_explicit_plan_sha256": plan_hash,
        "nonprivate_epoch_contract": "every_private_training_row_exactly_once_per_explicit_epoch",
        "nonprivate_round_domains_independent": True,
        "nonprivate_order_excludes_method_and_candidate": True,
        "initialization_seed_paired_across_methods_and_candidates": True,
        "private_rng_domain_owner": "train_unit_method_candidate_client_round_purpose_domains",
        "noise_calibration": calibration_artifact,
        "test_partition_authority": "forbidden_and_not_accepted",
        "observed_outputs_consumed": False,
    }
    try:
        assert_no_performance_fields(artifact)
    except Exception as exc:
        raise TrainingBudgetFactoryError(
            "budget artifact contains a forbidden observed-output field"
        ) from exc
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return TrainingBudgetFactoryResult(
        budget=budget, artifact=artifact, _factory_seal=_RESULT_FACTORY_SEAL
    )


def build_training_budget(
    spec: MethodExecutionSpec,
    candidate_parameters: Mapping[str, object],
    private_row_ids_by_client: Mapping[str, Sequence[int]],
    policy: SharedTrainingBudgetPolicy,
    *,
    expected_dispatch_sha256: str,
    expected_candidate_parameters_sha256: str,
    expected_policy_sha256: str,
) -> TrainingBudgetFactoryResult:
    """Build one executable budget without consulting observed outcomes."""
    result = _construct_training_budget(
        spec,
        candidate_parameters,
        private_row_ids_by_client,
        policy,
        expected_dispatch_sha256=expected_dispatch_sha256,
        expected_candidate_parameters_sha256=expected_candidate_parameters_sha256,
        expected_policy_sha256=expected_policy_sha256,
    )
    validate_training_budget_factory_result(
        result,
        spec,
        candidate_parameters,
        private_row_ids_by_client,
        policy,
        expected_dispatch_sha256=expected_dispatch_sha256,
        expected_candidate_parameters_sha256=expected_candidate_parameters_sha256,
        expected_policy_sha256=expected_policy_sha256,
    )
    return result


def validate_training_budget_factory_result(
    result: TrainingBudgetFactoryResult,
    spec: MethodExecutionSpec,
    candidate_parameters: Mapping[str, object],
    private_row_ids_by_client: Mapping[str, Sequence[int]],
    policy: SharedTrainingBudgetPolicy,
    *,
    expected_dispatch_sha256: str,
    expected_candidate_parameters_sha256: str,
    expected_policy_sha256: str,
) -> None:
    """Rebuild the complete factory output and reject tampering or forgery."""
    if (
        type(result) is not TrainingBudgetFactoryResult
        or result._factory_seal is not _RESULT_FACTORY_SEAL
    ):
        raise TrainingBudgetFactoryError("budget-factory result is not factory sealed")
    artifact = result.artifact
    if not isinstance(artifact, Mapping):
        raise TrainingBudgetFactoryError("budget-factory artifact is missing")
    stored_hash = artifact.get("artifact_sha256")
    payload = copy.deepcopy(dict(artifact))
    payload.pop("artifact_sha256", None)
    if stored_hash != canonical_sha256(payload):
        raise TrainingBudgetFactoryError("budget-factory artifact hash differs")
    try:
        assert_no_performance_fields(artifact)
    except Exception as exc:
        raise TrainingBudgetFactoryError(
            "budget-factory artifact contains observed-output fields"
        ) from exc
    expected = _construct_training_budget(
        spec,
        candidate_parameters,
        private_row_ids_by_client,
        policy,
        expected_dispatch_sha256=expected_dispatch_sha256,
        expected_candidate_parameters_sha256=expected_candidate_parameters_sha256,
        expected_policy_sha256=expected_policy_sha256,
    )
    if result.budget != expected.budget or artifact != expected.artifact:
        raise TrainingBudgetFactoryError(
            "budget-factory result differs from its committed result-blind inputs"
        )


__all__ = [
    "SharedTrainingBudgetPolicy",
    "TrainingBudgetFactoryError",
    "TrainingBudgetFactoryResult",
    "build_training_budget",
    "validate_training_budget_factory_result",
]
