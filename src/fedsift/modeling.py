"""Fixed result-blind models and selected-row per-record gradients for FedSift.

The two admitted model families are deliberately small and frozen:

* ``ScreeningMLP``: input -> 16 -> 8 -> 1 with ReLU hidden activations;
* ``LogisticScreening``: one affine input -> 1 logit, convex under BCE loss.

All parameters are float64.  Initialization is deterministic and separated by
a structural domain that contains no performance field.  The selected-row
gradient bridge uses ``torch.func.functional_call`` + ``grad`` + ``vmap``.  Its
sealed batch binds the canonical population row IDs, exact selected-row order,
model state, source table, and gradient tensors; the private local-training
gate consumes that batch rather than a bare mapping.
"""

from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import hashlib
import hmac
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

MODEL_DTYPE = torch.float64
MODEL_DTYPE_NAME = "torch.float64"
MODEL_FAMILIES = ("screening_mlp", "logistic_screening")
MLP_WIDTHS = (16, 8)
INITIALIZATION_ALGORITHM = "per_parameter_sha256_domain_uniform_v1"
SELECTED_GRADIENT_BATCH_SCHEMA = _identity("selected_gradient_batch")
SELECTED_GRADIENT_ALGORITHM = "torch_func_vmap_bce_logits_selected_rows_v1"


class ModelingError(RuntimeError):
    """Raised when a fixed-model or selected-row gradient contract fails."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _tensor_fingerprint(tensor: torch.Tensor) -> str:
    if not isinstance(tensor, torch.Tensor):
        raise ModelingError("tensor fingerprint input is not a tensor")
    value = tensor.detach().cpu().contiguous()
    if value.dtype.is_floating_point:
        if not torch.isfinite(value).all():
            raise ModelingError("tensor fingerprint input is non-finite")
        flattened: list[str | int] = [float(item).hex() for item in value.reshape(-1).tolist()]
    elif value.dtype == torch.int64:
        flattened = [int(item) for item in value.reshape(-1).tolist()]
    else:
        raise ModelingError("tensor fingerprint dtype is not supported")
    return _sha256_json(
        {"dtype": str(value.dtype), "shape": list(value.shape), "values": flattened}
    )


def row_id_sequence_sha256(row_ids: Sequence[int]) -> str:
    """Hash one sequence of exact, nonnegative Python row IDs."""
    if isinstance(row_ids, (str, bytes)) or not isinstance(row_ids, Sequence):
        raise ModelingError("row IDs must be a sequence")
    values = tuple(row_ids)
    if any((type(value) is not int or value < 0 for value in values)):
        raise ModelingError("row IDs must be exact nonnegative Python integers")
    if len(values) != len(set(values)):
        raise ModelingError("row IDs must be unique")
    return _sha256_json({"schema": _identity("row_id_sequence"), "row_ids": values})


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ModelingError(f"{name} must be an integer")
    number = int(value)
    if number < 0:
        raise ModelingError(f"{name} must be nonnegative")
    return number


def _positive_int(value: object, name: str) -> int:
    number = _nonnegative_int(value, name)
    if number <= 0:
        raise ModelingError(f"{name} must be positive")
    return number


def _plain_identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any((character in value for character in "\r\n\t"))
    ):
        raise ModelingError(f"{name} must be a non-empty plain identifier")
    return value


def _model_device(value: object) -> torch.device:
    try:
        device = torch.device(value)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ModelingError("model device is invalid") from exc
    if device.type not in {"cpu", "cuda"}:
        raise ModelingError("fixed models support only CPU or CUDA devices")
    if device.type == "cuda" and (not torch.cuda.is_available()):
        raise ModelingError("requested CUDA model device is unavailable")
    return device


@dataclass(frozen=True, slots=True)
class InitializationDomain:
    """Structural, performance-free initialization namespace."""

    study_id: str
    outer_repeat: int
    outer_fold: int
    client_id: str
    model_role: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "study_id", _plain_identifier(self.study_id, "study id"))
        object.__setattr__(
            self, "outer_repeat", _nonnegative_int(self.outer_repeat, "outer repeat")
        )
        object.__setattr__(self, "outer_fold", _nonnegative_int(self.outer_fold, "outer fold"))
        object.__setattr__(self, "client_id", _plain_identifier(self.client_id, "client id"))
        object.__setattr__(self, "model_role", _plain_identifier(self.model_role, "model role"))

    def payload(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "outer_repeat": self.outer_repeat,
            "outer_fold": self.outer_fold,
            "client_id": self.client_id,
            "model_role": self.model_role,
            "performance_fields_used": [],
        }

    def fingerprint(self) -> str:
        return _sha256_json(self.payload())


class _FixedLinear(nn.Module):
    """Affine layer allocated without consuming any global RNG state."""

    def __init__(self, in_features: int, out_features: int, *, device: torch.device) -> None:
        super().__init__()
        self.in_features = _positive_int(in_features, "linear input width")
        self.out_features = _positive_int(out_features, "linear output width")
        self.weight = nn.Parameter(
            torch.zeros((self.out_features, self.in_features), dtype=MODEL_DTYPE, device=device)
        )
        self.bias = nn.Parameter(
            torch.zeros((self.out_features,), dtype=MODEL_DTYPE, device=device)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, self.weight, self.bias)


class _FixedScreeningBase(nn.Module):
    family: str

    def __init__(
        self,
        input_dim: int,
        *,
        initialization_seed: int,
        initialization_domain: InitializationDomain,
        device: object = "cpu",
    ) -> None:
        super().__init__()
        self.input_dim = _positive_int(input_dim, "input dimension")
        self.initialization_seed = _nonnegative_int(initialization_seed, "initialization seed")
        if self.initialization_seed >= 2**63:
            raise ModelingError("initialization seed must be smaller than 2**63")
        if not isinstance(initialization_domain, InitializationDomain):
            raise ModelingError("initialization domain is missing")
        self.initialization_domain = initialization_domain
        self.model_device = _model_device(device)
        self._manifest_json = ""
        self._manifest_sha256 = ""

    def _finish_initialization(self) -> None:
        manifest = _expected_manifest(self)
        self._manifest_json = _canonical_json(manifest)
        self._manifest_sha256 = hashlib.sha256(self._manifest_json.encode("utf-8")).hexdigest()
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if parameter.dtype != MODEL_DTYPE or parameter.device != self.model_device:
                    raise ModelingError("fixed model parameter dtype or device drifted")
                fan_in = _fan_in_for_parameter(self, name)
                bound = 1.0 / math.sqrt(float(fan_in))
                seed_payload = {
                    "schema": _identity("parameter_initialization_seed"),
                    "model_manifest_sha256": self._manifest_sha256,
                    "parameter_name": name,
                    "algorithm": INITIALIZATION_ALGORITHM,
                }
                derived_seed = (
                    int.from_bytes(
                        hashlib.sha256(_canonical_json(seed_payload).encode("utf-8")).digest()[:8],
                        "big",
                    )
                    & (1 << 63) - 1
                )
                generator = torch.Generator(device=self.model_device).manual_seed(derived_seed)
                parameter.uniform_(-bound, bound, generator=generator)
        validate_fixed_model(self)

    @property
    def model_manifest_sha256(self) -> str:
        validate_fixed_model(self)
        return self._manifest_sha256

    def model_manifest(self) -> dict[str, Any]:
        validate_fixed_model(self)
        return json.loads(self._manifest_json)


class ScreeningMLP(_FixedScreeningBase):
    """Frozen input -> 16 -> 8 -> 1 float64 screening MLP."""

    family = "screening_mlp"

    def __init__(
        self,
        input_dim: int,
        *,
        initialization_seed: int,
        initialization_domain: InitializationDomain,
        device: object = "cpu",
    ) -> None:
        super().__init__(
            input_dim,
            initialization_seed=initialization_seed,
            initialization_domain=initialization_domain,
            device=device,
        )
        self.hidden1 = _FixedLinear(self.input_dim, MLP_WIDTHS[0], device=self.model_device)
        self.hidden2 = _FixedLinear(MLP_WIDTHS[0], MLP_WIDTHS[1], device=self.model_device)
        self.output = _FixedLinear(MLP_WIDTHS[1], 1, device=self.model_device)
        self._finish_initialization()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.hidden1(features))
        hidden = F.relu(self.hidden2(hidden))
        return self.output(hidden).squeeze(-1)


class LogisticScreening(_FixedScreeningBase):
    """Frozen convex affine logistic-screening model."""

    family = "logistic_screening"

    def __init__(
        self,
        input_dim: int,
        *,
        initialization_seed: int,
        initialization_domain: InitializationDomain,
        device: object = "cpu",
    ) -> None:
        super().__init__(
            input_dim,
            initialization_seed=initialization_seed,
            initialization_domain=initialization_domain,
            device=device,
        )
        self.output = _FixedLinear(self.input_dim, 1, device=self.model_device)
        self._finish_initialization()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(features).squeeze(-1)


def _parameter_schema(model: _FixedScreeningBase) -> list[dict[str, Any]]:
    return [
        {"name": name, "shape": list(parameter.shape), "dtype": MODEL_DTYPE_NAME}
        for (name, parameter) in model.named_parameters()
    ]


def _expected_manifest(model: _FixedScreeningBase) -> dict[str, Any]:
    if type(model) is ScreeningMLP:
        architecture = {
            "input_width": model.input_dim,
            "hidden_widths": list(MLP_WIDTHS),
            "output_width": 1,
            "hidden_activation": "relu",
            "output": "single_logit",
        }
    elif type(model) is LogisticScreening:
        architecture = {
            "input_width": model.input_dim,
            "hidden_widths": [],
            "output_width": 1,
            "hidden_activation": None,
            "output": "single_logit",
            "convex_objective": "binary_cross_entropy_with_logits",
        }
    else:
        raise ModelingError("model class is outside the fixed families")
    return {
        "schema": _identity("fixed_model_manifest"),
        "family": model.family,
        "architecture": architecture,
        "parameters": _parameter_schema(model),
        "dtype": MODEL_DTYPE_NAME,
        "device": str(model.model_device),
        "initialization": {
            "algorithm": INITIALIZATION_ALGORITHM,
            "root_seed": model.initialization_seed,
            "domain": model.initialization_domain.payload(),
            "domain_sha256": model.initialization_domain.fingerprint(),
        },
        "training_loss": "binary_cross_entropy_with_logits_per_record",
        "result_blind_model_choice": True,
        "performance_fields_used": [],
    }


def _fan_in_for_parameter(model: _FixedScreeningBase, name: str) -> int:
    module_name, _, _ = name.rpartition(".")
    try:
        module = model.get_submodule(module_name)
    except AttributeError as exc:
        raise ModelingError("parameter is not attached to a fixed affine module") from exc
    if not isinstance(module, _FixedLinear):
        raise ModelingError("parameter is not attached to a fixed affine module")
    return module.in_features


def validate_fixed_model(model: nn.Module) -> None:
    """Validate class, manifest, parameter order, dtype, device and finiteness."""
    if type(model) not in {ScreeningMLP, LogisticScreening}:
        raise ModelingError("model class is outside the fixed families")
    assert isinstance(model, _FixedScreeningBase)
    expected_manifest = _expected_manifest(model)
    expected_json = _canonical_json(expected_manifest)
    expected_hash = hashlib.sha256(expected_json.encode("utf-8")).hexdigest()
    if model._manifest_json != expected_json or model._manifest_sha256 != expected_hash:
        raise ModelingError("fixed model manifest or hash differs")
    if tuple(model.named_buffers()):
        raise ModelingError("fixed models must not contain mutable buffers")
    expected_names = tuple((item["name"] for item in expected_manifest["parameters"]))
    parameters = tuple(model.named_parameters())
    if tuple((name for (name, _) in parameters)) != expected_names:
        raise ModelingError("fixed model parameter order differs")
    for item, (name, parameter) in zip(expected_manifest["parameters"], parameters):
        if (
            name != item["name"]
            or list(parameter.shape) != item["shape"]
            or parameter.dtype != MODEL_DTYPE
            or (parameter.device != model.model_device)
            or (not torch.isfinite(parameter).all())
        ):
            raise ModelingError("fixed model parameter structure is invalid")


def build_fixed_model(
    family: str,
    input_dim: int,
    *,
    initialization_seed: int,
    initialization_domain: InitializationDomain,
    device: object = "cpu",
) -> ScreeningMLP | LogisticScreening:
    """Build one admitted family without accepting any performance input."""
    if family == "screening_mlp":
        return ScreeningMLP(
            input_dim,
            initialization_seed=initialization_seed,
            initialization_domain=initialization_domain,
            device=device,
        )
    if family == "logistic_screening":
        return LogisticScreening(
            input_dim,
            initialization_seed=initialization_seed,
            initialization_domain=initialization_domain,
            device=device,
        )
    raise ModelingError(f"unknown fixed model family: {family!r}")


def extract_model_state(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    validate_fixed_model(model)
    return OrderedDict(
        ((name, parameter.detach().clone()) for (name, parameter) in model.named_parameters())
    )


def fixed_model_manifest(model: nn.Module) -> dict[str, Any]:
    validate_fixed_model(model)
    assert isinstance(model, _FixedScreeningBase)
    return model.model_manifest()


def fixed_model_manifest_sha256(model: nn.Module) -> str:
    validate_fixed_model(model)
    assert isinstance(model, _FixedScreeningBase)
    return model.model_manifest_sha256


def validate_model_state(
    model: nn.Module, state: Mapping[str, torch.Tensor]
) -> OrderedDict[str, torch.Tensor]:
    validate_fixed_model(model)
    if not isinstance(state, Mapping):
        raise ModelingError("model state must be a mapping")
    expected = tuple(model.named_parameters())
    expected_names = tuple((name for (name, _) in expected))
    if tuple(state) != expected_names:
        raise ModelingError("model state parameter identity or order differs")
    output: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, parameter in expected:
        value = state[name]
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != tuple(parameter.shape)
            or value.dtype != MODEL_DTYPE
            or (value.device != parameter.device)
            or (not torch.isfinite(value).all())
        ):
            raise ModelingError("model state tensor structure is invalid")
        output[name] = value.detach().clone()
    return output


def model_state_sha256(model: nn.Module, state: Mapping[str, torch.Tensor]) -> str:
    """Hash a validated fixed-model state without serializing device identity."""
    validated = validate_model_state(model, state)
    return _sha256_json(
        {
            "schema": _identity("fixed_model_state"),
            "model_manifest_sha256": fixed_model_manifest_sha256(model),
            "parameters": [
                {"name": name, "tensor_sha256": _tensor_fingerprint(value)}
                for (name, value) in validated.items()
            ],
        }
    )


def _validated_prediction_features(
    model: _FixedScreeningBase, features: torch.Tensor
) -> torch.Tensor:
    if (
        not isinstance(features, torch.Tensor)
        or features.ndim != 2
        or int(features.shape[1]) != model.input_dim
        or (features.dtype != MODEL_DTYPE)
        or (features.device != model.model_device)
        or (not torch.isfinite(features).all())
    ):
        raise ModelingError("prediction feature matrix is invalid")
    return features.detach()


def predict_logits(
    model: nn.Module, features: torch.Tensor, *, state: Mapping[str, torch.Tensor] | None = None
) -> torch.Tensor:
    validate_fixed_model(model)
    assert isinstance(model, _FixedScreeningBase)
    values = _validated_prediction_features(model, features)
    parameter_state = (
        extract_model_state(model) if state is None else validate_model_state(model, state)
    )
    with torch.no_grad():
        logits = functional_call(model, parameter_state, (values,), strict=True)
    if (
        logits.shape != (int(values.shape[0]),)
        or logits.dtype != MODEL_DTYPE
        or logits.device != model.model_device
        or (not torch.isfinite(logits).all())
    ):
        raise ModelingError("model logits are invalid")
    return logits


def predict_probabilities(
    model: nn.Module, features: torch.Tensor, *, state: Mapping[str, torch.Tensor] | None = None
) -> torch.Tensor:
    probabilities = torch.sigmoid(predict_logits(model, features, state=state))
    if not torch.isfinite(probabilities).all() or not bool(
        ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
    ):
        raise ModelingError("model probabilities are invalid")
    return probabilities


def _validate_row_table_structure(
    model: _FixedScreeningBase,
    features: torch.Tensor,
    labels: torch.Tensor,
    row_ids: torch.Tensor,
    selected_row_ids: torch.Tensor,
) -> tuple[int, list[int], list[int]]:
    if (
        not isinstance(features, torch.Tensor)
        or features.ndim != 2
        or int(features.shape[1]) != model.input_dim
        or (features.dtype != MODEL_DTYPE)
        or (features.device != model.model_device)
    ):
        raise ModelingError("feature table structure is invalid")
    rows = int(features.shape[0])
    if (
        not isinstance(labels, torch.Tensor)
        or labels.shape != (rows,)
        or labels.dtype != MODEL_DTYPE
        or (labels.device != model.model_device)
    ):
        raise ModelingError("label vector structure is invalid")
    if (
        not isinstance(row_ids, torch.Tensor)
        or row_ids.shape != (rows,)
        or row_ids.dtype != torch.int64
        or (row_ids.device != model.model_device)
    ):
        raise ModelingError("row-ID vector structure is invalid")
    if (
        not isinstance(selected_row_ids, torch.Tensor)
        or selected_row_ids.ndim != 1
        or selected_row_ids.dtype != torch.int64
        or (selected_row_ids.device != model.model_device)
    ):
        raise ModelingError("selected row-ID vector structure is invalid")
    all_ids = [int(value) for value in row_ids.detach().cpu().tolist()]
    selected_ids = [int(value) for value in selected_row_ids.detach().cpu().tolist()]
    if any((value < 0 for value in all_ids)) or any((value < 0 for value in selected_ids)):
        raise ModelingError("row IDs must be nonnegative")
    if len(all_ids) != len(set(all_ids)):
        raise ModelingError("row IDs must be unique")
    if len(selected_ids) != len(set(selected_ids)):
        raise ModelingError("selected row IDs must be unique")
    unknown = sorted(set(selected_ids) - set(all_ids))
    if unknown:
        raise ModelingError(f"selected row IDs are outside the bound table: {unknown[:3]}")
    return (rows, all_ids, selected_ids)


def selected_per_record_bce_gradients(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
    features: torch.Tensor,
    labels: torch.Tensor,
    row_ids: torch.Tensor,
    selected_row_ids: torch.Tensor,
) -> OrderedDict[str, torch.Tensor]:
    """Return BCE-with-logits gradients for explicitly selected row IDs only.

    The full table is inspected only for tensor shape/dtype/device and row-ID
    binding.  Feature and label values are indexed first and validated only for
    selected rows, so an empty selection never evaluates any row and an
    unselected sentinel cannot enter the gradient computation.
    """
    validate_fixed_model(model)
    assert isinstance(model, _FixedScreeningBase)
    parameter_state = validate_model_state(model, state)
    _, all_ids, selected_ids = _validate_row_table_structure(
        model, features, labels, row_ids, selected_row_ids
    )
    if not selected_ids:
        return OrderedDict(
            (
                (
                    name,
                    torch.empty(
                        (0, *parameter.shape), dtype=parameter.dtype, device=parameter.device
                    ),
                )
                for (name, parameter) in parameter_state.items()
            )
        )
    position_by_id = {row_id: index for (index, row_id) in enumerate(all_ids)}
    positions = torch.tensor(
        [position_by_id[row_id] for row_id in selected_ids],
        dtype=torch.int64,
        device=model.model_device,
    )
    selected_features = features.index_select(0, positions)
    selected_labels = labels.index_select(0, positions)
    if not torch.isfinite(selected_features).all():
        raise ModelingError("selected features contain a non-finite value")
    if not torch.isfinite(selected_labels).all() or not bool(
        ((selected_labels == 0.0) | (selected_labels == 1.0)).all()
    ):
        raise ModelingError("selected labels must be finite binary values")

    def one_record_loss(
        parameters: Mapping[str, torch.Tensor], feature: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        logit = functional_call(model, parameters, (feature,), strict=True)
        return F.binary_cross_entropy_with_logits(logit, label, reduction="sum")

    gradient_function = grad(one_record_loss)
    try:
        gradients = vmap(gradient_function, in_dims=(None, 0, 0), out_dims=0, randomness="error")(
            parameter_state, selected_features, selected_labels
        )
    except Exception as exc:
        raise ModelingError("selected-row torch.func gradient computation failed") from exc
    if not isinstance(gradients, Mapping) or tuple(gradients) != tuple(parameter_state):
        raise ModelingError("per-record gradient parameter identity differs")
    output: OrderedDict[str, torch.Tensor] = OrderedDict()
    selected_count = len(selected_ids)
    for name, parameter in parameter_state.items():
        value = gradients[name]
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != (selected_count, *parameter.shape)
            or value.dtype != parameter.dtype
            or (value.device != parameter.device)
            or (not torch.isfinite(value).all())
        ):
            raise ModelingError("per-record gradient tensor structure is invalid")
        output[name] = value.detach()
    return output


@dataclass(frozen=True, slots=True)
class SelectedGradientBatch:
    """Exact-schema gradient batch emitted by ``SelectedBCEGradientBridge``."""

    schema: str
    model_manifest_sha256: str
    input_state_sha256: str
    source_table_sha256: str
    population_row_ids_sha256: str
    selected_row_ids: tuple[int, ...]
    selected_row_ids_sha256: str
    parameter_names: tuple[str, ...]
    gradients: OrderedDict[str, torch.Tensor]
    gradient_tensor_sha256: str
    gradient_lineage_sha256: str


def _validated_gradient_mapping(
    gradients: Mapping[str, torch.Tensor],
    parameter_state: Mapping[str, torch.Tensor],
    *,
    selected_count: int,
) -> OrderedDict[str, torch.Tensor]:
    if not isinstance(gradients, Mapping) or tuple(gradients) != tuple(parameter_state):
        raise ModelingError("gradient batch parameter identity or order differs")
    output: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, parameter in parameter_state.items():
        value = gradients[name]
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != (selected_count, *parameter.shape)
            or value.dtype != parameter.dtype
            or (value.device != parameter.device)
            or (not torch.isfinite(value).all())
        ):
            raise ModelingError("gradient batch tensor structure is invalid")
        output[name] = value.detach().clone()
    return output


def _gradient_tensor_fingerprint(
    gradients: Mapping[str, torch.Tensor],
    parameter_state: Mapping[str, torch.Tensor],
    *,
    selected_row_ids: tuple[int, ...],
) -> str:
    validated = _validated_gradient_mapping(
        gradients, parameter_state, selected_count=len(selected_row_ids)
    )
    return _sha256_json(
        {
            "schema": _identity("selected_gradient_tensors"),
            "selected_row_ids_sha256": row_id_sequence_sha256(selected_row_ids),
            "parameters": [
                {"name": name, "tensor_sha256": _tensor_fingerprint(value)}
                for (name, value) in validated.items()
            ],
        }
    )


def _gradient_lineage_fingerprint(
    *,
    model_manifest_sha256: str,
    input_state_sha256: str,
    source_table_sha256: str,
    population_row_ids_sha256: str,
    selected_row_ids_sha256: str,
    gradient_tensor_sha256: str,
) -> str:
    return _sha256_json(
        {
            "schema": _identity("selected_gradient_lineage"),
            "algorithm": SELECTED_GRADIENT_ALGORITHM,
            "model_manifest_sha256": model_manifest_sha256,
            "input_state_sha256": input_state_sha256,
            "source_table_sha256": source_table_sha256,
            "population_row_ids_sha256": population_row_ids_sha256,
            "selected_row_ids_sha256": selected_row_ids_sha256,
            "gradient_tensor_sha256": gradient_tensor_sha256,
        }
    )


class SelectedBCEGradientBridge:
    """Bind one fixed predictor table and compute selected-row BCE gradients.

    The bridge clones the table once.  Each call receives the gate-selected row
    IDs explicitly and emits a batch whose hashes are revalidated by the gate.
    This is an integrity/evidence boundary, not a hostile-process sandbox.
    """

    def __init__(
        self, model: nn.Module, features: torch.Tensor, labels: torch.Tensor, row_ids: torch.Tensor
    ) -> None:
        validate_fixed_model(model)
        assert isinstance(model, _FixedScreeningBase)
        empty_selection = torch.empty((0,), dtype=torch.int64, device=model.model_device)
        rows, all_ids, _ = _validate_row_table_structure(
            model, features, labels, row_ids, empty_selection
        )
        if not torch.isfinite(features).all():
            raise ModelingError("gradient bridge features contain a non-finite value")
        if not torch.isfinite(labels).all() or not bool(((labels == 0.0) | (labels == 1.0)).all()):
            raise ModelingError("gradient bridge labels must be finite binary values")
        self._model = model
        self._features = features.detach().clone()
        self._labels = labels.detach().clone()
        self._row_ids = row_ids.detach().clone()
        self._canonical_population_row_ids = tuple(sorted(all_ids))
        self._model_manifest_sha256 = fixed_model_manifest_sha256(model)
        self._population_row_ids_sha256 = row_id_sequence_sha256(self._canonical_population_row_ids)
        self._source_table_sha256 = _sha256_json(
            {
                "schema": _identity("gradient_source_table"),
                "model_manifest_sha256": self._model_manifest_sha256,
                "features_sha256": _tensor_fingerprint(self._features),
                "labels_sha256": _tensor_fingerprint(self._labels),
                "row_id_table_order_sha256": row_id_sequence_sha256(tuple(all_ids)),
                "canonical_population_row_ids_sha256": self._population_row_ids_sha256,
            }
        )

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def model_manifest_sha256(self) -> str:
        return self._model_manifest_sha256

    @property
    def canonical_population_row_ids(self) -> tuple[int, ...]:
        return self._canonical_population_row_ids

    @property
    def population_row_ids_sha256(self) -> str:
        return self._population_row_ids_sha256

    @property
    def source_table_sha256(self) -> str:
        return self._source_table_sha256

    def _validated_selection(self, selected_row_ids: Sequence[int]) -> tuple[int, ...]:
        if isinstance(selected_row_ids, (str, bytes)) or not isinstance(selected_row_ids, Sequence):
            raise ModelingError("selected row IDs must be a sequence")
        selected = tuple(selected_row_ids)
        if any((type(value) is not int or value < 0 for value in selected)):
            raise ModelingError("selected row IDs must be exact nonnegative Python integers")
        if len(selected) != len(set(selected)):
            raise ModelingError("selected row IDs must be unique")
        population = frozenset(self._canonical_population_row_ids)
        if any((row_id not in population for row_id in selected)):
            raise ModelingError("selected row IDs are outside the bridge population")
        return selected

    def selected_gradient_batch(
        self, state: Mapping[str, torch.Tensor], selected_row_ids: Sequence[int]
    ) -> SelectedGradientBatch:
        selected = self._validated_selection(selected_row_ids)
        parameter_state = validate_model_state(self._model, state)
        selected_tensor = torch.tensor(selected, dtype=torch.int64, device=self._model.model_device)
        gradients = selected_per_record_bce_gradients(
            self._model,
            parameter_state,
            self._features,
            self._labels,
            self._row_ids,
            selected_tensor,
        )
        input_hash = model_state_sha256(self._model, parameter_state)
        selected_hash = row_id_sequence_sha256(selected)
        gradient_hash = _gradient_tensor_fingerprint(
            gradients, parameter_state, selected_row_ids=selected
        )
        lineage_hash = _gradient_lineage_fingerprint(
            model_manifest_sha256=self._model_manifest_sha256,
            input_state_sha256=input_hash,
            source_table_sha256=self._source_table_sha256,
            population_row_ids_sha256=self._population_row_ids_sha256,
            selected_row_ids_sha256=selected_hash,
            gradient_tensor_sha256=gradient_hash,
        )
        return SelectedGradientBatch(
            schema=SELECTED_GRADIENT_BATCH_SCHEMA,
            model_manifest_sha256=self._model_manifest_sha256,
            input_state_sha256=input_hash,
            source_table_sha256=self._source_table_sha256,
            population_row_ids_sha256=self._population_row_ids_sha256,
            selected_row_ids=selected,
            selected_row_ids_sha256=selected_hash,
            parameter_names=tuple(parameter_state),
            gradients=gradients,
            gradient_tensor_sha256=gradient_hash,
            gradient_lineage_sha256=lineage_hash,
        )

    def validate_batch(
        self,
        batch: SelectedGradientBatch,
        state: Mapping[str, torch.Tensor],
        selected_row_ids: Sequence[int],
    ) -> OrderedDict[str, torch.Tensor]:
        """Rebuild every batch hash and return detached gradient clones."""
        if type(batch) is not SelectedGradientBatch:
            raise ModelingError("gradient bridge returned an unrecognized batch type")
        selected = self._validated_selection(selected_row_ids)
        parameter_state = validate_model_state(self._model, state)
        input_hash = model_state_sha256(self._model, parameter_state)
        selected_hash = row_id_sequence_sha256(selected)
        gradient_hash = _gradient_tensor_fingerprint(
            batch.gradients, parameter_state, selected_row_ids=selected
        )
        lineage_hash = _gradient_lineage_fingerprint(
            model_manifest_sha256=self._model_manifest_sha256,
            input_state_sha256=input_hash,
            source_table_sha256=self._source_table_sha256,
            population_row_ids_sha256=self._population_row_ids_sha256,
            selected_row_ids_sha256=selected_hash,
            gradient_tensor_sha256=gradient_hash,
        )
        expected_scalars = (
            (batch.schema, SELECTED_GRADIENT_BATCH_SCHEMA),
            (batch.model_manifest_sha256, self._model_manifest_sha256),
            (batch.input_state_sha256, input_hash),
            (batch.source_table_sha256, self._source_table_sha256),
            (batch.population_row_ids_sha256, self._population_row_ids_sha256),
            (batch.selected_row_ids_sha256, selected_hash),
            (batch.gradient_tensor_sha256, gradient_hash),
            (batch.gradient_lineage_sha256, lineage_hash),
        )
        if batch.selected_row_ids != selected or batch.parameter_names != tuple(parameter_state):
            raise ModelingError("gradient batch row or parameter identity differs")
        if any(
            (
                not isinstance(observed, str) or not hmac.compare_digest(observed, expected)
                for (observed, expected) in expected_scalars
            )
        ):
            raise ModelingError("gradient batch lineage or content hash differs")
        return _validated_gradient_mapping(
            batch.gradients, parameter_state, selected_count=len(selected)
        )
