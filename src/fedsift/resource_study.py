"""Real registered-data adapter for the result-blind FedSift cost benchmark.

This module is deliberately narrower than an HPO worker.  It validates one
factory-sealed :class:`StudyConstruction`, selects the preregistered
``candidate_0000`` unit for every main method at one explicit HPO scope, and
performs preprocessing, dispatch construction, privacy calibration, budget
construction, and model-manifest construction before timing starts.

The timed executor reuses only those immutable preparations.  Every call runs
``execute_training_unit_resource_only`` afresh, validates the sanitized result,
checks the precomputed model-manifest commitment, and returns a deep copy of
the round resource receipts.  It never returns a trained model, predictions,
validation losses, metrics, control decisions, or selection evidence.
"""

from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import copy
import hmac
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence
from .candidate_space import MAIN_METHODS, canonical_sha256, require_sha256
from .cost_benchmark import MeasurementBackend, measure_linux_process_rss, run_cost_benchmark
from .resource_protocol import (
    ResourceProtocolBoundaryError,
    assert_resource_protocol_result_blind,
    build_resource_protocol_capability,
    build_resource_protocol_order_manifest,
    validate_resource_protocol_capability,
)
from .hpo_capability import build_hpo_unit_capability, capability_candidate_parameters
from .method_dispatch import MethodExecutionSpec, build_method_execution_spec
from .modeling import InitializationDomain, build_fixed_model, fixed_model_manifest_sha256
from .preprocessing import preprocess_hpo_capability
from .study_factory import StudyConstruction, StudyFactoryError, validate_study_construction
from .train_unit import (
    ResourceOnlyTrainingResult,
    SealedUnitData,
    execute_training_unit_resource_only,
    seal_preprocessed_unit_data,
    validate_resource_only_training_result,
)
from .training_budget_factory import (
    SharedTrainingBudgetPolicy,
    TrainingBudgetFactoryResult,
    build_training_budget,
    validate_training_budget_factory_result,
)


class ResourceStudyError(ValueError):
    """Raised when the registered-data resource preparation is inconsistent."""


class ResourceStudyBoundaryError(ResourceProtocolBoundaryError, ResourceStudyError):
    """Raised when result, selection, or outer-test authority crosses the gate."""


PREPARATION_SCHEMA = _identity("real_data_resource_smoke_preparation")
INPUT_BINDING_SCHEMA = _identity("real_data_resource_smoke_input_binding")
SCOPE_SCHEMA = _identity("real_data_resource_smoke_hpo_scope")
FORMAL_CANDIDATE_ID = "candidate_0000"
NUMERIC_CANDIDATE_ID = 0
MINIMUM_QUERY_COVERAGE_ROUNDS = 5
REGISTERED_CLIENTS: tuple[str, ...] = tuple((f"client_{index}" for index in range(5)))
_PREPARATION_FACTORY_SEAL = object()


def _exact_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ResourceStudyError(f"{name} must be an exact integer >= {minimum}")
    return value


def _plain_identifier(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or any((character in value for character in "\r\n\t"))
    ):
        raise ResourceStudyError(f"{name} must be a plain identifier")
    return value


def _sha256(value: object, name: str) -> str:
    try:
        return require_sha256(value, name)
    except Exception as exc:
        raise ResourceStudyError(f"{name} is not a SHA-256 digest") from exc


def _reject_outer_authority(
    *, outer_capability: object | None, outer_gate: object | None, outer_test_gate: object | None
) -> None:
    if any((value is not None for value in (outer_capability, outer_gate, outer_test_gate))):
        raise ResourceStudyBoundaryError(
            "registered-data resource smoke cannot receive outer authority"
        )


def _scope_payload(scope: "RealDataResourceSmokeScope") -> dict[str, object]:
    return {
        "schema": SCOPE_SCHEMA,
        "outer_repeat": scope.outer_repeat,
        "outer_fold": scope.outer_fold,
        "inner_fold": scope.inner_fold,
        "hpo_seed": scope.hpo_seed,
    }


@dataclass(frozen=True, slots=True)
class ResourceStudyScope:
    """One externally predeclared HPO scope for the cost-only probe."""

    outer_repeat: int
    outer_fold: int
    inner_fold: int
    hpo_seed: int
    scope_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("outer_repeat", "outer_fold", "inner_fold", "hpo_seed"):
            object.__setattr__(self, name, _exact_int(getattr(self, name), name, minimum=0))
        object.__setattr__(self, "scope_sha256", canonical_sha256(_scope_payload(self)))


@dataclass(frozen=True, slots=True)
class _PreparedMethod:
    method_id: str
    unit_id: str
    capability: dict[str, object] = field(repr=False)
    sealed_data: SealedUnitData = field(repr=False)
    dispatch_spec: MethodExecutionSpec = field(repr=False)
    budget_result: TrainingBudgetFactoryResult = field(repr=False)
    model_manifest_sha256: str
    binding: dict[str, object]


def _preparation_manifest(
    *,
    bundle_sha256: str,
    scope_sha256: str,
    policy_sha256: str,
    input_binding_sha256: str,
    resource_protocol_capability_sha256: str,
    model_manifest_sha256: str,
    expected_rounds: int,
    method_bindings: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "schema": PREPARATION_SCHEMA,
        "status": "ready_result_blind_registered_data_resource_only",
        "study_bundle_sha256": bundle_sha256,
        "hpo_scope_sha256": scope_sha256,
        "shared_policy_sha256": policy_sha256,
        "input_binding_sha256": input_binding_sha256,
        "cost_smoke_capability_sha256": resource_protocol_capability_sha256,
        "model_manifest_sha256": model_manifest_sha256,
        "candidate_id": NUMERIC_CANDIDATE_ID,
        "candidate_roster_id": FORMAL_CANDIDATE_ID,
        "expected_rounds": expected_rounds,
        "method_bindings": [copy.deepcopy(dict(value)) for value in method_bindings],
    }


@dataclass(frozen=True, slots=True, init=False)
class ResourceStudyPreparation:
    """Factory-sealed cached preparations plus a narrow fresh-run executor."""

    _input_binding: dict[str, object] = field(repr=False)
    _cost_capability: dict[str, object] = field(repr=False)
    _scope: ResourceStudyScope = field(repr=False)
    _policy: SharedTrainingBudgetPolicy = field(repr=False)
    _prepared_methods: tuple[_PreparedMethod, ...] = field(repr=False)
    _expected_requests: tuple[dict[str, object], ...] = field(repr=False)
    _manifest: dict[str, object] = field(repr=False)
    preparation_sha256: str
    _factory_seal: object = field(repr=False, compare=False)

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ResourceStudyError("resource-smoke preparations can only be created by the factory")

    @property
    def input_binding(self) -> dict[str, object]:
        return copy.deepcopy(self._input_binding)

    @property
    def cost_capability(self) -> dict[str, object]:
        return copy.deepcopy(self._cost_capability)

    @property
    def method_bindings(self) -> tuple[dict[str, object], ...]:
        return tuple((copy.deepcopy(value.binding) for value in self._prepared_methods))

    @property
    def scope(self) -> ResourceStudyScope:
        return self._scope

    def manifest(self) -> dict[str, object]:
        result = copy.deepcopy(self._manifest)
        result["preparation_sha256"] = self.preparation_sha256
        return result

    def resource_executor(
        self,
        *,
        method_id: str,
        candidate_id: int,
        phase: str,
        sequence_index: int,
        block_index: int,
        row_index: int,
        order_position: int,
        synthetic_input: object,
    ) -> list[dict[str, object]]:
        """Execute one exact scheduled method afresh and return receipts only."""
        _validate_preparation_internal(
            self,
            expected_preparation_sha256=self.preparation_sha256,
            validate_cached_preparations=False,
        )
        sequence = _exact_int(sequence_index, "sequence_index", minimum=1)
        if sequence > len(self._expected_requests):
            raise ResourceStudyBoundaryError("executor context is outside the frozen cost schedule")
        actual_context = {
            "sequence_index": sequence,
            "phase": _plain_identifier(phase, "phase"),
            "block_index": _exact_int(block_index, "block_index", minimum=1),
            "row_index": _exact_int(row_index, "row_index", minimum=0),
            "order_position": _exact_int(order_position, "order_position", minimum=0),
            "method_id": _plain_identifier(method_id, "method_id"),
            "candidate_id": _exact_int(candidate_id, "candidate_id", minimum=0),
        }
        actual_context["context_sha256"] = canonical_sha256(actual_context)
        expected_context = self._expected_requests[sequence - 1]
        if actual_context != expected_context:
            raise ResourceStudyBoundaryError(
                "executor method, candidate, or schedule context differs"
            )
        frozen_input = copy.deepcopy(synthetic_input)
        assert_resource_protocol_result_blind(frozen_input, "real_data_input_binding")
        if (
            frozen_input != self._input_binding
            or canonical_sha256(frozen_input) != self._cost_capability["input_sha256"]
        ):
            raise ResourceStudyBoundaryError(
                "executor input differs from the frozen result-blind input hash"
            )
        matches = tuple((value for value in self._prepared_methods if value.method_id == method_id))
        if len(matches) != 1:
            raise ResourceStudyBoundaryError(
                "executor method is absent or duplicated in the frozen preparation"
            )
        prepared = matches[0]
        _validate_prepared_method(
            prepared, self._policy, self._scope, validate_budget_factory=False
        )
        result = execute_training_unit_resource_only(
            prepared.capability,
            prepared.sealed_data,
            prepared.budget_result.budget,
            expected_capability_sha256=str(prepared.capability["capability_sha256"]),
            expected_budget_sha256=prepared.budget_result.budget.budget_sha256,
        )
        if type(result) is not ResourceOnlyTrainingResult:
            raise ResourceStudyError("resource-only execution returned an unrecognized result")
        validate_resource_only_training_result(
            result,
            prepared.capability,
            prepared.sealed_data,
            prepared.budget_result.budget,
            expected_capability_sha256=str(prepared.capability["capability_sha256"]),
            expected_budget_sha256=prepared.budget_result.budget.budget_sha256,
        )
        if result.resource_summary.get("model_manifest_sha256") != prepared.model_manifest_sha256:
            raise ResourceStudyError(
                "fresh resource summary differs from the precomputed model manifest"
            )
        receipts = [copy.deepcopy(value) for value in result.round_resource_receipts]
        assert_resource_protocol_result_blind(receipts, "real_data_round_receipts")
        return receipts


def _seal_preparation(
    *,
    input_binding: dict[str, object],
    cost_capability: dict[str, object],
    scope: ResourceStudyScope,
    policy: SharedTrainingBudgetPolicy,
    prepared_methods: tuple[_PreparedMethod, ...],
    expected_requests: tuple[dict[str, object], ...],
    manifest: dict[str, object],
) -> ResourceStudyPreparation:
    result = object.__new__(ResourceStudyPreparation)
    for name, value in {
        "_input_binding": input_binding,
        "_cost_capability": cost_capability,
        "_scope": scope,
        "_policy": policy,
        "_prepared_methods": prepared_methods,
        "_expected_requests": expected_requests,
        "_manifest": manifest,
        "preparation_sha256": canonical_sha256(manifest),
        "_factory_seal": _PREPARATION_FACTORY_SEAL,
    }.items():
        object.__setattr__(result, name, value)
    return result


def _bundle_mappings(
    construction: StudyConstruction,
) -> tuple[
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
]:
    if not isinstance(construction, StudyConstruction):
        raise ResourceStudyError("input must be a factory-sealed RealN1StudyConstruction")
    bundle = construction.bundle
    if not isinstance(bundle, Mapping):
        raise ResourceStudyError("study construction bundle is missing")
    names = (
        "source_binding",
        "nested_plan_binding",
        "candidate_space_binding",
        "preprocessing_binding",
        "hpo_plan_binding",
    )
    mappings: list[Mapping[str, object]] = []
    for name in names:
        value = bundle.get(name)
        if not isinstance(value, Mapping):
            raise ResourceStudyError(f"study bundle {name} is missing")
        mappings.append(value)
    return tuple(mappings)


def _validate_policy_against_bundle(
    policy: SharedTrainingBudgetPolicy,
    *,
    expected_policy_sha256: str,
    client_policy_sha256: str,
    max_steps: int,
) -> None:
    if type(policy) is not SharedTrainingBudgetPolicy:
        raise ResourceStudyError("shared policy has an unrecognized type")
    expected = _sha256(expected_policy_sha256, "expected_policy_sha256")
    if not hmac.compare_digest(policy.policy_sha256, expected):
        raise ResourceStudyError("shared training policy differs from its external commitment")
    if policy.server_rounds != max_steps:
        raise ResourceStudyError("shared server_rounds must equal the sealed HPO max_steps")
    if policy.server_rounds < MINIMUM_QUERY_COVERAGE_ROUNDS:
        raise ResourceStudyError(
            "resource smoke is too short to cover candidate-zero query cadence"
        )
    if policy.participating_client_ids != REGISTERED_CLIENTS:
        raise ResourceStudyError(
            "resource smoke must participate all five clients in canonical order"
        )
    if (
        policy.partition_mode != "fixed_label_driven_auxiliary_condition"
        or policy.fixed_auxiliary_partition_condition_sha256 != client_policy_sha256
    ):
        raise ResourceStudyError("label-driven partition condition must equal client_policy_sha256")


def _select_unit_id(
    construction: StudyConstruction, *, method_id: str, scope: ResourceStudyScope
) -> str:
    units = construction.complete_hpo_plan.get("units")
    if not isinstance(units, list):
        raise ResourceStudyError("complete HPO inventory is missing")
    matches = [
        value
        for value in units
        if isinstance(value, Mapping)
        and value.get("method") == method_id
        and (value.get("candidate_id") == FORMAL_CANDIDATE_ID)
        and (value.get("outer_repeat") == scope.outer_repeat)
        and (value.get("outer_fold") == scope.outer_fold)
        and (value.get("inner_fold") == scope.inner_fold)
        and (value.get("hpo_seed") == scope.hpo_seed)
    ]
    if len(matches) != 1 or type(matches[0].get("unit_id")) is not str:
        raise ResourceStudyError("predeclared scope does not resolve one candidate_0000 HPO unit")
    return str(matches[0]["unit_id"])


def _labels_by_row_id(construction: StudyConstruction) -> dict[int, int]:
    rows = construction.dataset_rows
    if len(rows.row_ids) != len(rows.labels):
        raise ResourceStudyError("registered row IDs and labels are misaligned")
    return dict(zip(rows.row_ids, rows.labels))


def _model_manifest_for(
    *,
    capability: Mapping[str, object],
    sealed_data: SealedUnitData,
    policy: SharedTrainingBudgetPolicy,
) -> str:
    identity = capability.get("unit_identity")
    if not isinstance(identity, Mapping):
        raise ResourceStudyError("HPO capability identity is missing")
    widths = {
        len(table.features[0])
        for table in sealed_data.roles
        if table.features and table.features[0]
    }
    if len(widths) != 1:
        raise ResourceStudyError("sealed training roles do not share one feature width")
    model = build_fixed_model(
        policy.model_family,
        next(iter(widths)),
        initialization_seed=policy.paired_initialization_seed,
        initialization_domain=InitializationDomain(
            study_id=str(capability["study_id"]),
            outer_repeat=int(identity["outer_repeat"]),
            outer_fold=int(identity["outer_fold"]),
            client_id="server",
            model_role="global_screening_model",
        ),
    )
    return fixed_model_manifest_sha256(model)


def _prepare_method(
    construction: StudyConstruction,
    *,
    method_id: str,
    scope: ResourceStudyScope,
    policy: SharedTrainingBudgetPolicy,
    labels_by_row_id: Mapping[int, int],
) -> _PreparedMethod:
    unit_id = _select_unit_id(construction, method_id=method_id, scope=scope)
    capability = build_hpo_unit_capability(
        construction.complete_hpo_plan,
        construction.frozen_candidate_space,
        construction.frozen_nested_plan,
        construction.group_manifest,
        unit_id=unit_id,
    )
    capability_hash = str(capability["capability_sha256"])
    candidate = capability.get("candidate")
    identity = capability.get("unit_identity")
    if not isinstance(candidate, Mapping) or not isinstance(identity, Mapping):
        raise ResourceStudyError("prepared capability identity is malformed")
    if (
        candidate.get("candidate_id") != FORMAL_CANDIDATE_ID
        or identity.get("method") != method_id
        or identity.get("max_steps") != policy.server_rounds
    ):
        raise ResourceStudyError(
            "prepared capability differs from method, candidate, or round binding"
        )
    parameters = capability_candidate_parameters(
        capability, expected_capability_sha256=capability_hash
    )
    control = parameters.get("control_rule")
    if isinstance(control, Mapping):
        cadence = _exact_int(
            control.get("query_every_rounds"), "candidate-zero query cadence", minimum=1
        )
        if policy.server_rounds < cadence:
            raise ResourceStudyError(
                "resource smoke does not reach the candidate-zero query cadence"
            )
    preprocessed = preprocess_hpo_capability(
        capability,
        construction.dataset_rows,
        construction.group_manifest,
        expected_capability_sha256=capability_hash,
    )
    sealed_data = seal_preprocessed_unit_data(
        capability, preprocessed, labels_by_row_id, expected_capability_sha256=capability_hash
    )
    spec = build_method_execution_spec(
        method_id=method_id, candidate_parameters=parameters, mechanism_switch=None
    )
    private_rows = {
        client_id: sealed_data.role(client_id).row_ids for client_id in REGISTERED_CLIENTS
    }
    budget_result = build_training_budget(
        spec,
        parameters,
        private_rows,
        policy,
        expected_dispatch_sha256=spec.dispatch_sha256,
        expected_candidate_parameters_sha256=str(candidate["parameters_sha256"]),
        expected_policy_sha256=policy.policy_sha256,
    )
    validate_training_budget_factory_result(
        budget_result,
        spec,
        parameters,
        private_rows,
        policy,
        expected_dispatch_sha256=spec.dispatch_sha256,
        expected_candidate_parameters_sha256=str(candidate["parameters_sha256"]),
        expected_policy_sha256=policy.policy_sha256,
    )
    model_hash = _model_manifest_for(capability=capability, sealed_data=sealed_data, policy=policy)
    binding = {
        "method_id": method_id,
        "unit_id": unit_id,
        "candidate_id": NUMERIC_CANDIDATE_ID,
        "candidate_roster_id": FORMAL_CANDIDATE_ID,
        "candidate_sha256": candidate["candidate_sha256"],
        "candidate_parameters_sha256": candidate["parameters_sha256"],
        "hpo_capability_sha256": capability_hash,
        "dispatch_sha256": spec.dispatch_sha256,
        "sealed_data_sha256": sealed_data.data_sha256,
        "budget_sha256": budget_result.budget.budget_sha256,
        "shared_policy_sha256": policy.policy_sha256,
        "model_manifest_sha256": model_hash,
        "scope_sha256": scope.scope_sha256,
    }
    assert_resource_protocol_result_blind(binding, "prepared_method_binding")
    return _PreparedMethod(
        method_id=method_id,
        unit_id=unit_id,
        capability=capability,
        sealed_data=sealed_data,
        dispatch_spec=spec,
        budget_result=budget_result,
        model_manifest_sha256=model_hash,
        binding=binding,
    )


def _input_binding(
    construction: StudyConstruction,
    *,
    scope: ResourceStudyScope,
    policy: SharedTrainingBudgetPolicy,
    prepared_methods: Sequence[_PreparedMethod],
) -> dict[str, object]:
    bundle = construction.bundle
    source = bundle["source_binding"]
    nested = bundle["nested_plan_binding"]
    candidates = bundle["candidate_space_binding"]
    preprocessing = bundle["preprocessing_binding"]
    hpo = bundle["hpo_plan_binding"]
    assert all(
        (isinstance(value, Mapping) for value in (source, nested, candidates, preprocessing, hpo))
    )
    method_bindings = [value.binding for value in prepared_methods]
    result: dict[str, object] = {
        "schema": INPUT_BINDING_SCHEMA,
        "status": "registered_data_hash_binding_only_no_observed_outputs",
        "study_bundle_sha256": bundle["bundle_sha256"],
        "study_id": source["study_id"],
        "dataset_id": source["dataset_id"],
        "source_sha256": source["source_sha256"],
        "group_manifest_sha256": bundle["group_binding"]["group_manifest_sha256"],
        "nested_plan_sha256": nested["nested_plan_sha256"],
        "candidate_space_sha256": candidates["candidate_space_sha256"],
        "preprocessing_profile_sha256": preprocessing["profile_sha256"],
        "hpo_plan_sha256": hpo["hpo_plan_sha256"],
        "hpo_scope": _scope_payload(scope),
        "hpo_scope_sha256": scope.scope_sha256,
        "candidate_id": NUMERIC_CANDIDATE_ID,
        "candidate_roster_id": FORMAL_CANDIDATE_ID,
        "method_ids": sorted(MAIN_METHODS),
        "method_bindings_sha256": canonical_sha256(method_bindings),
        "shared_policy_sha256": policy.policy_sha256,
        "expected_rounds": policy.server_rounds,
        "participating_client_count": len(REGISTERED_CLIENTS),
        "resource_only": True,
    }
    assert_resource_protocol_result_blind(result, "real_data_input_binding")
    return result


def _expected_requests(
    capability: Mapping[str, object], input_binding: Mapping[str, object]
) -> tuple[dict[str, object], ...]:
    order = build_resource_protocol_order_manifest(
        capability,
        synthetic_input=input_binding,
        expected_capability_sha256=str(capability["capability_sha256"]),
    )
    result: list[dict[str, object]] = []
    for phase, field in (("warmup", "warmup_orders"), ("measured", "measured_orders")):
        entries = order[field]
        if not isinstance(entries, list):
            raise ResourceStudyError("cost order manifest is malformed")
        for entry in entries:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("method_ids"), list):
                raise ResourceStudyError("cost order entry is malformed")
            for position, method_id in enumerate(entry["method_ids"]):
                context: dict[str, object] = {
                    "sequence_index": len(result) + 1,
                    "phase": phase,
                    "block_index": entry["block_index"],
                    "row_index": entry["row_index"],
                    "order_position": position,
                    "method_id": method_id,
                    "candidate_id": NUMERIC_CANDIDATE_ID,
                }
                context["context_sha256"] = canonical_sha256(context)
                result.append(context)
    return tuple(result)


def build_resource_study(
    construction: StudyConstruction,
    workspace_root: Path | str,
    *,
    expected_bundle_sha256: str,
    scope: ResourceStudyScope,
    expected_scope_sha256: str,
    shared_policy: SharedTrainingBudgetPolicy,
    expected_policy_sha256: str,
    warmup_passes: int = 1,
    latin_square_repetitions: int = 1,
    order_seed: str = _identity("registered_data_resource_cost_smoke"),
    outer_capability: object | None = None,
    outer_gate: object | None = None,
    outer_test_gate: object | None = None,
) -> ResourceStudyPreparation:
    """Prepare all ten candidate-zero main methods outside the timed region."""
    _reject_outer_authority(
        outer_capability=outer_capability, outer_gate=outer_gate, outer_test_gate=outer_test_gate
    )
    expected_bundle = _sha256(expected_bundle_sha256, "expected_bundle_sha256")
    if type(scope) is not ResourceStudyScope:
        raise ResourceStudyError("scope has an unrecognized type")
    expected_scope = _sha256(expected_scope_sha256, "expected_scope_sha256")
    if scope.scope_sha256 != canonical_sha256(_scope_payload(scope)) or not hmac.compare_digest(
        scope.scope_sha256, expected_scope
    ):
        raise ResourceStudyError("HPO scope differs from its external commitment")
    source, nested, candidates, preprocessing, hpo = _bundle_mappings(construction)
    bundle_hash = _sha256(construction.bundle.get("bundle_sha256"), "bundle_sha256")
    if not hmac.compare_digest(bundle_hash, expected_bundle):
        raise ResourceStudyError("study construction differs from the external bundle commitment")
    max_steps = _exact_int(hpo.get("max_steps"), "HPO max_steps", minimum=1)
    client_policy_hash = _sha256(nested.get("client_policy_sha256"), "client_policy_sha256")
    _validate_policy_against_bundle(
        shared_policy,
        expected_policy_sha256=expected_policy_sha256,
        client_policy_sha256=client_policy_hash,
        max_steps=max_steps,
    )
    try:
        validate_study_construction(
            construction,
            workspace_root,
            str(source["dataset_id"]),
            study_id=str(source["study_id"]),
            protocol_sha256=str(nested["protocol_sha256"]),
            implementation_sha256=str(nested["implementation_sha256"]),
            client_policy_sha256=client_policy_hash,
            candidate_count=int(candidates["candidate_count_per_method"]),
            hpo_seeds=tuple(hpo["hpo_seeds"]),
            max_steps=max_steps,
            preprocessing_profile_name=str(preprocessing["profile_name"]),
            expected_bundle_sha256=expected_bundle,
        )
    except (StudyFactoryError, KeyError, TypeError, ValueError) as exc:
        raise ResourceStudyError(
            "real-data construction failed exact factory reconstruction"
        ) from exc
    labels = _labels_by_row_id(construction)
    prepared = tuple(
        (
            _prepare_method(
                construction,
                method_id=method_id,
                scope=scope,
                policy=shared_policy,
                labels_by_row_id=labels,
            )
            for method_id in MAIN_METHODS
        )
    )
    if tuple((value.method_id for value in prepared)) != MAIN_METHODS:
        raise ResourceStudyError("prepared method roster differs")
    model_hashes = {value.model_manifest_sha256 for value in prepared}
    if len(model_hashes) != 1:
        raise ResourceStudyError("paired initialization did not produce one model manifest")
    model_hash = next(iter(model_hashes))
    input_binding = _input_binding(
        construction, scope=scope, policy=shared_policy, prepared_methods=prepared
    )
    cost_capability = build_resource_protocol_capability(
        method_ids=MAIN_METHODS,
        synthetic_input=input_binding,
        expected_rounds=max_steps,
        model_manifest_sha256=model_hash,
        candidate_id=NUMERIC_CANDIDATE_ID,
        warmup_passes=warmup_passes,
        latin_square_repetitions=latin_square_repetitions,
        order_seed=order_seed,
    )
    expected_requests = _expected_requests(cost_capability, input_binding)
    method_bindings = [value.binding for value in prepared]
    manifest = _preparation_manifest(
        bundle_sha256=bundle_hash,
        scope_sha256=scope.scope_sha256,
        policy_sha256=shared_policy.policy_sha256,
        input_binding_sha256=canonical_sha256(input_binding),
        resource_protocol_capability_sha256=str(cost_capability["capability_sha256"]),
        model_manifest_sha256=model_hash,
        expected_rounds=max_steps,
        method_bindings=method_bindings,
    )
    assert_resource_protocol_result_blind(manifest, "real_data_preparation_manifest")
    result = _seal_preparation(
        input_binding=input_binding,
        cost_capability=cost_capability,
        scope=scope,
        policy=shared_policy,
        prepared_methods=prepared,
        expected_requests=expected_requests,
        manifest=manifest,
    )
    validate_resource_study_preparation(
        result, expected_preparation_sha256=result.preparation_sha256
    )
    return result


def _validate_prepared_method(
    value: _PreparedMethod,
    policy: SharedTrainingBudgetPolicy,
    scope: ResourceStudyScope,
    *,
    validate_budget_factory: bool = True,
) -> None:
    if type(value) is not _PreparedMethod:
        raise ResourceStudyError("prepared method has an unrecognized type")
    capability = value.capability
    candidate = capability.get("candidate")
    identity = capability.get("unit_identity")
    if not isinstance(candidate, Mapping) or not isinstance(identity, Mapping):
        raise ResourceStudyError("prepared method capability is malformed")
    expected_binding = {
        "method_id": value.method_id,
        "unit_id": value.unit_id,
        "candidate_id": NUMERIC_CANDIDATE_ID,
        "candidate_roster_id": FORMAL_CANDIDATE_ID,
        "candidate_sha256": candidate.get("candidate_sha256"),
        "candidate_parameters_sha256": candidate.get("parameters_sha256"),
        "hpo_capability_sha256": capability.get("capability_sha256"),
        "dispatch_sha256": value.dispatch_spec.dispatch_sha256,
        "sealed_data_sha256": value.sealed_data.data_sha256,
        "budget_sha256": value.budget_result.budget.budget_sha256,
        "shared_policy_sha256": policy.policy_sha256,
        "model_manifest_sha256": value.model_manifest_sha256,
        "scope_sha256": scope.scope_sha256,
    }
    if value.binding != expected_binding:
        raise ResourceStudyError("prepared method hash binding differs")
    if (
        identity.get("method") != value.method_id
        or identity.get("outer_repeat") != scope.outer_repeat
        or identity.get("outer_fold") != scope.outer_fold
        or (identity.get("inner_fold") != scope.inner_fold)
        or (identity.get("hpo_seed") != scope.hpo_seed)
        or (identity.get("max_steps") != policy.server_rounds)
        or (candidate.get("candidate_id") != FORMAL_CANDIDATE_ID)
        or (value.sealed_data.capability_sha256 != capability.get("capability_sha256"))
        or (value.budget_result.artifact.get("shared_policy_sha256") != policy.policy_sha256)
    ):
        raise ResourceStudyError("prepared method identity, scope, candidate, or policy differs")
    parameters = capability_candidate_parameters(
        capability, expected_capability_sha256=str(capability["capability_sha256"])
    )
    if validate_budget_factory:
        private_rows = {
            client_id: value.sealed_data.role(client_id).row_ids for client_id in REGISTERED_CLIENTS
        }
        validate_training_budget_factory_result(
            value.budget_result,
            value.dispatch_spec,
            parameters,
            private_rows,
            policy,
            expected_dispatch_sha256=value.dispatch_spec.dispatch_sha256,
            expected_candidate_parameters_sha256=str(candidate["parameters_sha256"]),
            expected_policy_sha256=policy.policy_sha256,
        )


def _validate_preparation_internal(
    preparation: ResourceStudyPreparation,
    *,
    expected_preparation_sha256: str,
    validate_cached_preparations: bool = True,
) -> None:
    if (
        type(preparation) is not ResourceStudyPreparation
        or preparation._factory_seal is not _PREPARATION_FACTORY_SEAL
    ):
        raise ResourceStudyError("resource-smoke preparation is not factory sealed")
    expected = _sha256(expected_preparation_sha256, "expected_preparation_sha256")
    actual = canonical_sha256(preparation._manifest)
    if preparation.preparation_sha256 != actual or not hmac.compare_digest(actual, expected):
        raise ResourceStudyError("resource-smoke preparation commitment differs")
    assert_resource_protocol_result_blind(preparation._manifest, "real_data_preparation_manifest")
    if preparation._scope.scope_sha256 != canonical_sha256(_scope_payload(preparation._scope)):
        raise ResourceStudyError("prepared HPO scope hash differs")
    if preparation._policy.policy_sha256 != preparation._manifest.get("shared_policy_sha256"):
        raise ResourceStudyError("prepared policy hash differs")
    input_hash = canonical_sha256(preparation._input_binding)
    if input_hash != preparation._manifest.get("input_binding_sha256"):
        raise ResourceStudyError("prepared input binding hash differs")
    validate_resource_protocol_capability(
        preparation._cost_capability,
        synthetic_input=preparation._input_binding,
        expected_capability_sha256=str(preparation._manifest["cost_smoke_capability_sha256"]),
    )
    if (
        preparation._cost_capability.get("expected_rounds") != preparation._policy.server_rounds
        or preparation._cost_capability.get("candidate_id") != NUMERIC_CANDIDATE_ID
        or preparation._cost_capability.get("model_manifest_sha256")
        != preparation._manifest.get("model_manifest_sha256")
    ):
        raise ResourceStudyError("cost capability differs from round, candidate, or model binding")
    if tuple((value.method_id for value in preparation._prepared_methods)) != MAIN_METHODS:
        raise ResourceStudyError("prepared method roster differs")
    for value in preparation._prepared_methods:
        _validate_prepared_method(
            value,
            preparation._policy,
            preparation._scope,
            validate_budget_factory=validate_cached_preparations,
        )
    bindings = [value.binding for value in preparation._prepared_methods]
    if preparation._manifest.get("method_bindings") != bindings:
        raise ResourceStudyError("prepared method manifest differs")
    expected_requests = _expected_requests(preparation._cost_capability, preparation._input_binding)
    if preparation._expected_requests != expected_requests:
        raise ResourceStudyError("prepared cost schedule context differs")


def validate_resource_study_preparation(
    preparation: ResourceStudyPreparation, *, expected_preparation_sha256: str
) -> None:
    """Fail closed on every cached identity and hash before measurement."""
    _validate_preparation_internal(
        preparation, expected_preparation_sha256=expected_preparation_sha256
    )


def run_real_data_cost_benchmark(
    preparation: ResourceStudyPreparation,
    *,
    expected_preparation_sha256: str,
    expected_capability_sha256: str,
    measurement_backend: MeasurementBackend = measure_linux_process_rss,
    outer_capability: object | None = None,
    outer_gate: object | None = None,
    outer_test_gate: object | None = None,
) -> dict[str, object]:
    """Run the existing prewarmed cyclic benchmark on the real-data adapter."""
    _reject_outer_authority(
        outer_capability=outer_capability, outer_gate=outer_gate, outer_test_gate=outer_test_gate
    )
    validate_resource_study_preparation(
        preparation, expected_preparation_sha256=expected_preparation_sha256
    )
    expected_capability = _sha256(expected_capability_sha256, "expected_capability_sha256")
    if not hmac.compare_digest(
        str(preparation._cost_capability["capability_sha256"]), expected_capability
    ):
        raise ResourceStudyError("cost capability differs from its external commitment")
    return run_cost_benchmark(
        preparation._cost_capability,
        synthetic_input=preparation._input_binding,
        expected_capability_sha256=expected_capability,
        resource_executor=preparation.resource_executor,
        measurement_backend=measurement_backend,
    )


__all__ = [
    "FORMAL_CANDIDATE_ID",
    "INPUT_BINDING_SCHEMA",
    "MINIMUM_QUERY_COVERAGE_ROUNDS",
    "NUMERIC_CANDIDATE_ID",
    "PREPARATION_SCHEMA",
    "REGISTERED_CLIENTS",
    "RealDataResourceSmokeBoundaryError",
    "RealDataResourceSmokeError",
    "RealDataResourceSmokePreparation",
    "RealDataResourceSmokeScope",
    "build_real_data_resource_smoke",
    "run_real_data_cost_benchmark",
    "validate_real_data_resource_smoke_preparation",
]
