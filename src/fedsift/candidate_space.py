"""Result-blind, equal-budget candidate rosters for FedSift.

This module defines design inputs only. It deliberately has no API for reading
predictions or metrics. Candidate zero is a transparent reference.  It is
followed by predeclared, primary-source one-factor sentinels and then by a
deterministic normative Latin hypercube (LHS) over the remaining slots.

The two local-learning-rate sentinels use the same candidate ids for every
method.  The FedAdam-backed methods additionally share the same numeric
backend/local sentinels and the same LHS rows by candidate id.  FedYogi keeps
that layout but uses its source-specific low-beta2 sentinel explicitly.
"""

from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping, Sequence
from .probability_contract import PROBABILITY_CLIP_EPSILON


class CandidateSpaceError(ValueError):
    """Raised when a candidate-space contract is malformed or drifts."""


MAIN_METHODS: tuple[str, ...] = (
    "fedavg_nonprivate",
    "dp_fedavg",
    "dp_fedprox_adapted",
    "dp_scaffold_adapted",
    "dp_fedadam",
    "dp_fedyogi",
    "dp_fedsofim_delta_proxy_adapted",
    "time_dpfedadam",
    "public_argmin_time_dpfedadam",
    "fedsift",
)
DEFAULT_CANDIDATE_COUNT = 24
MIN_CANDIDATE_COUNT = 10
DEFAULT_DESIGN_SEED = _identity("source_sentinels_plus_normative_lhs_result_blind")
FORBIDDEN_PERFORMANCE_FIELDS = frozenset(
    {
        "accuracy",
        "ap",
        "auprc",
        "auroc",
        "average_precision",
        "brier",
        "brier_score",
        "calibration_error",
        "f1",
        "fnr",
        "log_loss",
        "loss",
        "mean_log_loss",
        "metric",
        "metrics",
        "performance",
        "precision",
        "prediction",
        "predictions",
        "probability",
        "recall",
        "roc_auc",
        "score",
        "scores",
        "utility",
    }
)


def _canonical_bytes(value: object) -> bytes:
    try:
        rendered = json.dumps(
            value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
    except (TypeError, ValueError) as exc:
        raise CandidateSpaceError("value is not strict canonical JSON") from exc
    return rendered.encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise CandidateSpaceError(f"{field} must be a lowercase SHA-256 hex string")
    if value != value.lower() or any((char not in "0123456789abcdef" for char in value)):
        raise CandidateSpaceError(f"{field} must be a lowercase SHA-256 hex string")
    return value


def require_exact_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CandidateSpaceError(f"{field} must be an integer >= {minimum}")
    return value


def assert_no_performance_fields(value: object, path: str = "root") -> None:
    """Reject observed-performance fields anywhere in a design artifact."""
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise CandidateSpaceError(f"{path} contains a non-string JSON key")
            key = raw_key.strip().lower().replace("-", "_")
            if key in FORBIDDEN_PERFORMANCE_FIELDS:
                raise CandidateSpaceError(
                    f"result-blind artifact contains forbidden field {path}.{raw_key}"
                )
            assert_no_performance_fields(child, f"{path}.{raw_key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_no_performance_fields(child, f"{path}[{index}]")
        return
    if isinstance(value, float) and (not math.isfinite(value)):
        raise CandidateSpaceError(f"{path} contains a non-finite float")


@dataclass(frozen=True)
class Dimension:
    path: tuple[str, ...]
    kind: str
    reference: float | int
    coupling_key: str
    low: float | None = None
    high: float | None = None
    choices: tuple[float | int, ...] = ()

    def __post_init__(self) -> None:
        if not self.path or any((not item for item in self.path)):
            raise CandidateSpaceError("dimension path must contain non-empty strings")
        if not self.coupling_key:
            raise CandidateSpaceError("dimension coupling_key cannot be empty")
        if self.kind in {"linear", "log"}:
            if self.low is None or self.high is None:
                raise CandidateSpaceError("continuous dimensions require low/high")
            if not all((math.isfinite(float(x)) for x in (self.low, self.high))):
                raise CandidateSpaceError("dimension bounds must be finite")
            if not float(self.low) < float(self.high):
                raise CandidateSpaceError("dimension low must be less than high")
            if self.kind == "log" and float(self.low) <= 0.0:
                raise CandidateSpaceError("log dimension low must be positive")
        elif self.kind == "choice":
            if len(self.choices) < 2 or len(set(self.choices)) != len(self.choices):
                raise CandidateSpaceError("choice dimensions require unique choices")
        else:
            raise CandidateSpaceError(f"unsupported dimension kind: {self.kind}")

    def map_coordinate(self, coordinate: float) -> float | int:
        if not math.isfinite(coordinate) or not 0.0 <= coordinate < 1.0:
            raise CandidateSpaceError("LHS coordinate must be in [0, 1)")
        if self.kind == "linear":
            assert self.low is not None and self.high is not None
            return float(self.low + coordinate * (self.high - self.low))
        if self.kind == "log":
            assert self.low is not None and self.high is not None
            low_log = math.log(float(self.low))
            high_log = math.log(float(self.high))
            return float(math.exp(low_log + coordinate * (high_log - low_log)))
        selected = min(int(coordinate * len(self.choices)), len(self.choices) - 1)
        return self.choices[selected]

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "path": list(self.path),
            "kind": self.kind,
            "reference": self.reference,
            "coupling_key": self.coupling_key,
        }
        if self.kind in {"linear", "log"}:
            payload.update({"low": self.low, "high": self.high})
        else:
            payload["choices"] = list(self.choices)
        return payload


@dataclass(frozen=True)
class SourceSentinel:
    """One primary-source value that changes exactly one candidate0 leaf."""

    sentinel_id: str
    path: tuple[str, ...]
    value: float | int
    coupling_key: str
    source_locator: str
    source_basis: str
    pairing_scope: str

    def __post_init__(self) -> None:
        if not self.sentinel_id or not self.path or any((not item for item in self.path)):
            raise CandidateSpaceError("source sentinel identity/path cannot be empty")
        if not self.coupling_key or not self.source_locator or (not self.source_basis):
            raise CandidateSpaceError("source sentinel provenance cannot be empty")
        if not self.pairing_scope:
            raise CandidateSpaceError("source sentinel pairing scope cannot be empty")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise CandidateSpaceError("source sentinel value must be numeric")
        if not math.isfinite(float(self.value)):
            raise CandidateSpaceError("source sentinel value must be finite")

    def generation(self, *, reference_value: float | int) -> dict[str, object]:
        return {
            "kind": "source_sentinel",
            "sentinel_id": self.sentinel_id,
            "changed_path": list(self.path),
            "coupling_key": self.coupling_key,
            "reference_value": reference_value,
            "sentinel_value": self.value,
            "source_locator": self.source_locator,
            "source_basis": self.source_basis,
            "pairing_scope": self.pairing_scope,
        }


LOCAL_LR = Dimension(
    ("local_optimizer", "learning_rate"),
    "log",
    0.03,
    "shared/local_learning_rate",
    low=0.001,
    high=3.1622776601683795,
)
FEDADAM_SERVER_LR = Dimension(
    ("backend", "server_learning_rate"),
    "choice",
    0.05,
    "shared/fedopt/server_learning_rate",
    choices=(0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 1.0, 3.1622776601683795, 10.0),
)
FEDADAM_BETA1 = Dimension(
    ("backend", "beta1"),
    "choice",
    0.9,
    "shared/fedopt/beta1",
    choices=(0.0, 0.1, 0.25, 0.5, 0.8, 0.9, 0.95),
)
FEDADAM_BETA2 = Dimension(
    ("backend", "beta2"),
    "choice",
    0.99,
    "shared/fedopt/beta2",
    choices=(0.5, 0.8, 0.9, 0.99, 0.999, 0.9999),
)
FEDADAM_TAU = Dimension(
    ("backend", "tau"),
    "choice",
    0.001,
    "shared/fedopt/tau",
    choices=(1e-05, 0.0001, 0.001, 0.01, 0.05, 0.1),
)
SOFIM_SERVER_LR = Dimension(
    ("backend", "server_learning_rate"),
    "choice",
    0.1,
    "dp_fedsofim/server_learning_rate",
    choices=(0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 3.0, 4.0, 5.0),
)
SOFIM_BETA = Dimension(
    ("backend", "beta"), "choice", 0.9, "dp_fedsofim/beta", choices=(0.8, 0.85, 0.9, 0.95, 0.99)
)
SOFIM_RHO = Dimension(
    ("backend", "rho"),
    "choice",
    0.5,
    "dp_fedsofim/rho",
    choices=(0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0),
)
TIME_SAVING_FRACTION = Dimension(
    ("privacy_schedule", "saving_round_fraction"),
    "choice",
    0.4,
    "shared/time_schedule/saving_round_fraction",
    choices=(0.2, 0.3, 0.4, 0.5, 0.6),
)
TIME_SAVING_FACTOR = Dimension(
    ("privacy_schedule", "saving_sigma_factor"),
    "linear",
    1.35,
    "shared/time_schedule/saving_sigma_factor",
    low=1.1,
    high=1.8,
)
TIME_SPENDING_FACTOR = Dimension(
    ("privacy_schedule", "spending_sigma_factor"),
    "linear",
    0.85,
    "shared/time_schedule/spending_sigma_factor",
    low=0.65,
    high=0.95,
)
CONTROL_QUERY_EVERY = Dimension(
    ("control_rule", "query_every_rounds"),
    "choice",
    5,
    "shared/public_control/query_every_rounds",
    choices=(1, 2, 3, 5, 6, 10),
)
SIFT_SAFETY_Z = Dimension(
    ("sift", "safety_margin_z"),
    "choice",
    1.6448536269514722,
    "fedsift/sift/safety_margin_z",
    choices=(0.0, 1.2815515655446004, 1.6448536269514722, 1.959963984540054, 2.3263478740408408),
)
SIFT_DISABLE_OVERRIDE_THRESHOLD = -math.log(PROBABILITY_CLIP_EPSILON)
SIFT_MIN_IMPROVEMENT = Dimension(
    ("sift", "minimum_control_improvement"),
    "choice",
    0.0,
    "fedsift/sift/minimum_control_improvement",
    choices=(
        0.0,
        0.001,
        0.0025,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.5,
        1.0,
        SIFT_DISABLE_OVERRIDE_THRESHOLD,
    ),
)
FEDADAM_DIMENSIONS = (LOCAL_LR, FEDADAM_SERVER_LR, FEDADAM_BETA1, FEDADAM_BETA2, FEDADAM_TAU)
TIME_DIMENSIONS = (
    *FEDADAM_DIMENSIONS,
    TIME_SAVING_FRACTION,
    TIME_SAVING_FACTOR,
    TIME_SPENDING_FACTOR,
)
METHOD_DIMENSIONS: dict[str, tuple[Dimension, ...]] = {
    "fedavg_nonprivate": (LOCAL_LR,),
    "dp_fedavg": (LOCAL_LR,),
    "dp_fedprox_adapted": (
        LOCAL_LR,
        Dimension(
            ("local_objective", "prox_mu"), "log", 0.01, "dp_fedprox/prox_mu", low=0.0001, high=1.0
        ),
    ),
    "dp_scaffold_adapted": (
        LOCAL_LR,
        Dimension(
            ("backend", "server_learning_rate"),
            "log",
            1.0,
            "dp_scaffold/server_learning_rate",
            low=0.01,
            high=5.0,
        ),
    ),
    "dp_fedadam": FEDADAM_DIMENSIONS,
    "dp_fedyogi": FEDADAM_DIMENSIONS,
    "dp_fedsofim_delta_proxy_adapted": (LOCAL_LR, SOFIM_SERVER_LR, SOFIM_BETA, SOFIM_RHO),
    "time_dpfedadam": TIME_DIMENSIONS,
    "public_argmin_time_dpfedadam": (*TIME_DIMENSIONS, CONTROL_QUERY_EVERY),
    "fedsift": (*TIME_DIMENSIONS, CONTROL_QUERY_EVERY, SIFT_SAFETY_Z, SIFT_MIN_IMPROVEMENT),
}
_FEDOPT_SOURCE = "https://arxiv.org/pdf/2003.00295"
_FEDPROX_SOURCE = "https://proceedings.mlsys.org/paper_files/paper/2020/file/1f5fe83998a09396ebe6477d9475ba0c-Paper.pdf"
_DP_FEDSOFIM_REFERENCE = "https://arxiv.org/pdf/2601.09166v3"


def _sentinel(
    sentinel_id: str,
    path: tuple[str, ...],
    value: float | int,
    coupling_key: str,
    source_locator: str,
    source_basis: str,
    pairing_scope: str,
) -> SourceSentinel:
    return SourceSentinel(
        sentinel_id=sentinel_id,
        path=path,
        value=value,
        coupling_key=coupling_key,
        source_locator=source_locator,
        source_basis=source_basis,
        pairing_scope=pairing_scope,
    )


_SHARED_LOCAL_SENTINELS = (
    _sentinel(
        "shared_local_lr_lower_0p001",
        LOCAL_LR.path,
        0.001,
        LOCAL_LR.coupling_key,
        _DP_FEDSOFIM_REFERENCE,
        "DP baseline client-learning-rate lower boundary in Table 2",
        "all_main_methods_candidate_0001",
    ),
    _sentinel(
        "shared_local_lr_dp_upper_0p5",
        LOCAL_LR.path,
        0.5,
        LOCAL_LR.coupling_key,
        _DP_FEDSOFIM_REFERENCE,
        "DP baseline client-learning-rate upper boundary in Table 2",
        "all_main_methods_candidate_0002",
    ),
)
_FEDADAM_SHARED_SENTINELS = (
    *_SHARED_LOCAL_SENTINELS,
    _sentinel(
        "fedopt_local_lr_upper_sqrt10",
        LOCAL_LR.path,
        3.1622776601683795,
        LOCAL_LR.coupling_key,
        _FEDOPT_SOURCE,
        "FedOpt Appendix D.2 common local-learning-rate upper boundary",
        "fedopt_backbone_methods_candidate_0003",
    ),
    _sentinel(
        "fedopt_server_lr_anchor_1",
        FEDADAM_SERVER_LR.path,
        1.0,
        FEDADAM_SERVER_LR.coupling_key,
        _FEDOPT_SOURCE,
        "FedOpt Appendix D.2 and DP baseline server-learning-rate anchor",
        "fedopt_backbone_methods_candidate_0004",
    ),
    _sentinel(
        "fedopt_tau_lower_1e_minus_5",
        FEDADAM_TAU.path,
        1e-05,
        FEDADAM_TAU.coupling_key,
        _FEDOPT_SOURCE,
        "FedOpt Appendix D.2 tau lower boundary",
        "fedopt_backbone_methods_candidate_0005",
    ),
    _sentinel(
        "fedopt_tau_upper_0p1",
        FEDADAM_TAU.path,
        0.1,
        FEDADAM_TAU.coupling_key,
        _FEDOPT_SOURCE,
        "FedOpt Appendix D.2 tau upper boundary",
        "fedopt_backbone_methods_candidate_0006",
    ),
    _sentinel(
        "fedopt_beta1_zero",
        FEDADAM_BETA1.path,
        0.0,
        FEDADAM_BETA1.coupling_key,
        _DP_FEDSOFIM_REFERENCE,
        "DP-FedAdam and DP-FedYogi beta1 source grid includes zero",
        "fedopt_backbone_methods_candidate_0007",
    ),
)
_FEDADAM_LOW_BETA2_SENTINEL = _sentinel(
    "fedadam_beta2_low_0p8",
    FEDADAM_BETA2.path,
    0.8,
    FEDADAM_BETA2.coupling_key,
    _DP_FEDSOFIM_REFERENCE,
    "DP-FedAdam Table 2 low-beta2 source sentinel",
    "fedadam_backbone_methods_candidate_0008",
)
_FEDYOGI_LOW_BETA2_SENTINEL = _sentinel(
    "fedyogi_beta2_low_0p5",
    FEDADAM_BETA2.path,
    0.5,
    FEDADAM_BETA2.coupling_key,
    _DP_FEDSOFIM_REFERENCE,
    "DP-FedYogi Table 2 method-specific low-beta2 source sentinel",
    "dp_fedyogi_method_specific_candidate_0008",
)
_FEDPROX_SENTINELS = (
    *_SHARED_LOCAL_SENTINELS,
    _sentinel(
        "fedprox_mu_0p001",
        ("local_objective", "prox_mu"),
        0.001,
        "dp_fedprox/prox_mu",
        _FEDPROX_SOURCE,
        "FedProx reported mu candidate set",
        "dp_fedprox_adapted_candidate_0003",
    ),
    _sentinel(
        "fedprox_mu_0p1",
        ("local_objective", "prox_mu"),
        0.1,
        "dp_fedprox/prox_mu",
        _FEDPROX_SOURCE,
        "FedProx reported mu candidate set",
        "dp_fedprox_adapted_candidate_0004",
    ),
    _sentinel(
        "fedprox_mu_1",
        ("local_objective", "prox_mu"),
        1.0,
        "dp_fedprox/prox_mu",
        _FEDPROX_SOURCE,
        "FedProx reported mu candidate set",
        "dp_fedprox_adapted_candidate_0005",
    ),
)
_SCAFFOLD_SENTINELS = (
    *_SHARED_LOCAL_SENTINELS,
    _sentinel(
        "scaffold_server_lr_lower_0p01",
        ("backend", "server_learning_rate"),
        0.01,
        "dp_scaffold/server_learning_rate",
        _DP_FEDSOFIM_REFERENCE,
        "DP-SCAFFOLD Table 2 server-learning-rate lower boundary",
        "dp_scaffold_adapted_candidate_0003",
    ),
    _sentinel(
        "scaffold_server_lr_upper_5",
        ("backend", "server_learning_rate"),
        5.0,
        "dp_scaffold/server_learning_rate",
        _DP_FEDSOFIM_REFERENCE,
        "DP-SCAFFOLD Table 2 server-learning-rate upper boundary",
        "dp_scaffold_adapted_candidate_0004",
    ),
)
_SOFIM_SENTINELS = (
    *_SHARED_LOCAL_SENTINELS,
    _sentinel(
        "sofim_server_lr_lower_0p001",
        SOFIM_SERVER_LR.path,
        0.001,
        SOFIM_SERVER_LR.coupling_key,
        _DP_FEDSOFIM_REFERENCE,
        "DP-FedSOFIM Table 2 eta lower boundary",
        "dp_fedsofim_candidate_0003",
    ),
    _sentinel(
        "sofim_server_lr_upper_5",
        SOFIM_SERVER_LR.path,
        5.0,
        SOFIM_SERVER_LR.coupling_key,
        _DP_FEDSOFIM_REFERENCE,
        "DP-FedSOFIM Table 2 eta upper boundary",
        "dp_fedsofim_candidate_0004",
    ),
    _sentinel(
        "sofim_rho_lower_0p01",
        SOFIM_RHO.path,
        0.01,
        SOFIM_RHO.coupling_key,
        _DP_FEDSOFIM_REFERENCE,
        "DP-FedSOFIM Table 2 rho lower boundary",
        "dp_fedsofim_candidate_0005",
    ),
    _sentinel(
        "sofim_rho_upper_20",
        SOFIM_RHO.path,
        20.0,
        SOFIM_RHO.coupling_key,
        _DP_FEDSOFIM_REFERENCE,
        "DP-FedSOFIM Table 2 rho upper boundary",
        "dp_fedsofim_candidate_0006",
    ),
    _sentinel(
        "sofim_beta_lower_0p8",
        SOFIM_BETA.path,
        0.8,
        SOFIM_BETA.coupling_key,
        _DP_FEDSOFIM_REFERENCE,
        "DP-FedSOFIM Table 2 beta lower boundary",
        "dp_fedsofim_candidate_0007",
    ),
    _sentinel(
        "sofim_beta_upper_0p99",
        SOFIM_BETA.path,
        0.99,
        SOFIM_BETA.coupling_key,
        _DP_FEDSOFIM_REFERENCE,
        "DP-FedSOFIM Table 2 beta upper boundary",
        "dp_fedsofim_candidate_0008",
    ),
)
METHOD_SOURCE_SENTINELS: dict[str, tuple[SourceSentinel, ...]] = {
    "fedavg_nonprivate": _SHARED_LOCAL_SENTINELS,
    "dp_fedavg": _SHARED_LOCAL_SENTINELS,
    "dp_fedprox_adapted": _FEDPROX_SENTINELS,
    "dp_scaffold_adapted": _SCAFFOLD_SENTINELS,
    "dp_fedadam": (*_FEDADAM_SHARED_SENTINELS, _FEDADAM_LOW_BETA2_SENTINEL),
    "dp_fedyogi": (*_FEDADAM_SHARED_SENTINELS, _FEDYOGI_LOW_BETA2_SENTINEL),
    "dp_fedsofim_delta_proxy_adapted": _SOFIM_SENTINELS,
    "time_dpfedadam": (*_FEDADAM_SHARED_SENTINELS, _FEDADAM_LOW_BETA2_SENTINEL),
    "public_argmin_time_dpfedadam": (*_FEDADAM_SHARED_SENTINELS, _FEDADAM_LOW_BETA2_SENTINEL),
    "fedsift": (*_FEDADAM_SHARED_SENTINELS, _FEDADAM_LOW_BETA2_SENTINEL),
}
if set(METHOD_SOURCE_SENTINELS) != set(MAIN_METHODS):
    raise CandidateSpaceError("source-sentinel registry must cover every main method")
if max((len(values) for values in METHOD_SOURCE_SENTINELS.values())) + 2 != MIN_CANDIDATE_COUNT:
    raise CandidateSpaceError("minimum candidate count must leave one LHS row")
REFERENCE_BASIS: dict[str, dict[str, str]] = {
    "fedavg_nonprivate": {
        "kind": "literature_reference",
        "basis": "McMahan et al. 2017 Algorithm 1; explicit epoch minibatches",
        "source_locator": "https://proceedings.mlr.press/v54/mcmahan17a.html",
        "parameter_provenance": _identity(
            "declared_n1_reference_point_not_claimed_as_paper_recommended_hyperparameters"
        ),
    },
    "dp_fedavg": {
        "kind": "transparent_record_dp_adaptation",
        "basis": "paper-aligned FedAvg backend plus common record-level DP-SGD contract",
        "source_locator": "https://proceedings.mlr.press/v54/mcmahan17a.html",
        "parameter_provenance": _identity(
            "declared_n1_reference_point_under_the_common_privacy_contract"
        ),
    },
    "dp_fedprox_adapted": {
        "kind": "transparent_record_dp_adaptation",
        "basis": "Li et al. 2020 FedProx objective; common record-level DP-SGD contract",
        "source_locator": "https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html",
        "parameter_provenance": _identity(
            "declared_n1_reference_point_record_dp_adaptation_is_explicit"
        ),
    },
    "dp_scaffold_adapted": {
        "kind": "transparent_record_dp_adaptation",
        "basis": _identity(
            "karimireddy_et_al_2020_scaffold_option_ii_local_control_update_example_weighted_batched_server_aggre"
        ),
        "source_locator": "https://proceedings.mlr.press/v119/karimireddy20a.html",
        "parameter_provenance": _identity(
            "declared_n1_reference_point_neither_the_record_dp_mechanism_nor_the_example_weighted_aggregation_is_"
        ),
    },
    "dp_fedadam": {
        "kind": "literature_backend_with_record_dp",
        "basis": "Reddi et al. 2021 FedAdam with v_minus_1 >= tau_squared; common record-level DP-SGD",
        "source_locator": "https://openreview.net/forum?id=LkFG3lB13U5",
        "parameter_provenance": _identity(
            "declared_n1_reference_point_backend_identity_follows_fedopt"
        ),
    },
    "dp_fedyogi": {
        "kind": "literature_backend_with_record_dp",
        "basis": "Reddi et al. 2021 FedYogi with v_minus_1 >= tau_squared; common record-level DP-SGD",
        "source_locator": "https://openreview.net/forum?id=LkFG3lB13U5",
        "parameter_provenance": _identity(
            "declared_n1_reference_point_backend_identity_follows_fedopt"
        ),
    },
    "dp_fedsofim_delta_proxy_adapted": {
        "kind": "transparent_record_dp_adaptation",
        "basis": "Nair et al. 2026 rank-one H(M)G server preconditioner applied to the preregistered normalized local-trajectory gradient proxy",
        "source_locator": "https://arxiv.org/abs/2601.09166v3",
        "parameter_provenance": _identity(
            "eta_beta_and_rho_cover_the_paper_search_union_candidate_zero_uses_the_official_repository_cli_defaul"
        ),
    },
    "time_dpfedadam": {
        "kind": "current_transparent_reference",
        "basis": "predeclared record-level time schedule on the paper-aligned FedAdam backend",
        "source_locator": _identity(
            "docs_analysis_result_blind_development_protocol_draft_20260831_md_6"
        ),
        "parameter_provenance": "result-blind carry-forward of explicit schedule defaults",
    },
    "public_argmin_time_dpfedadam": {
        "kind": "current_transparent_reference",
        "basis": "equal-information public-control argmin on the matched time-DP-FedAdam backend",
        "source_locator": _identity(
            "docs_analysis_result_blind_development_protocol_draft_20260831_md_5"
        ),
        "parameter_provenance": "predeclared equal-information control rule",
    },
    "fedsift": {
        "kind": "current_transparent_reference",
        "basis": "FedAdam backend, time schedule, and Sift rule jointly represented in one candidate",
        "source_locator": _identity(
            "docs_analysis_result_blind_development_protocol_draft_20260831_md_6"
        ),
        "parameter_provenance": "result-blind transparent reference; not claimed as already optimal",
    },
}


def _base_parameters(method: str) -> dict[str, object]:
    base: dict[str, object] = {"local_optimizer": {"name": "sgd", "learning_rate": 0.03}}
    if method in {"fedavg_nonprivate", "dp_fedavg"}:
        base["backend"] = {"name": "weighted_average", "server_learning_rate": 1.0}
    elif method == "dp_fedprox_adapted":
        base.update(
            {
                "backend": {"name": "weighted_average", "server_learning_rate": 1.0},
                "local_objective": {"name": "fedprox", "prox_mu": 0.01},
            }
        )
    elif method == "dp_scaffold_adapted":
        base["backend"] = {
            "name": "scaffold_option2_batched_weighted_adapted",
            "server_learning_rate": 1.0,
        }
    elif method == "dp_fedyogi":
        base["backend"] = {
            "name": "fedyogi_paper_aligned",
            "beta1": 0.9,
            "beta2": 0.99,
            "server_learning_rate": 0.05,
            "tau": 0.001,
            "second_moment_initialization": "tau_squared",
        }
    elif method == "dp_fedsofim_delta_proxy_adapted":
        base["backend"] = {
            "name": "dp_fedsofim_delta_proxy_adapted",
            "server_learning_rate": 0.1,
            "beta": 0.9,
            "rho": 0.5,
            "moment_initialization": "zeros",
            "bias_correction": False,
            "warmup_rounds": 0,
            "preconditioned_vector": "current_normalized_trajectory_proxy",
            "client_proxy_definition": "negative_local_minus_global_divided_by_actual_steps_times_local_lr",
            "aggregation": "same_preregistered_server_weights_as_model_delta_methods",
            "claim_boundary": "adapted_proxy_not_paper_or_official_repository_reproduction",
        }
    else:
        base["backend"] = {
            "name": "fedadam_paper_aligned",
            "beta1": 0.9,
            "beta2": 0.99,
            "server_learning_rate": 0.05,
            "tau": 0.001,
            "second_moment_initialization": "tau_squared",
        }
    if method != "fedavg_nonprivate":
        base["record_dp"] = {
            "mechanism": "common_client_record_level_dp_sgd",
            "accounting": "candidate_specific_same_target_epsilon_delta",
        }
        base["privacy_schedule"] = {"kind": "uniform"}
    if method in {"time_dpfedadam", "public_argmin_time_dpfedadam", "fedsift"}:
        base["privacy_schedule"] = {
            "kind": "two_phase_time",
            "saving_round_fraction": 0.4,
            "saving_sigma_factor": 1.35,
            "spending_sigma_factor": 0.85,
        }
    if method in {"public_argmin_time_dpfedadam", "fedsift"}:
        base["control_rule"] = {
            "records": "same_frozen_V_ctrl",
            "labels_visible": True,
            "step_candidates": [0.0, 0.25, 0.5, 0.75, 1.0],
            "query_every_rounds": 5,
        }
    if method == "public_argmin_time_dpfedadam":
        control = base["control_rule"]
        assert isinstance(control, dict)
        control.update(
            {
                "name": "public_argmin_mean_logloss",
                "tie_break": "larger_step",
                "safety_margin": "none",
            }
        )
    if method == "fedsift":
        control = base["control_rule"]
        assert isinstance(control, dict)
        control["name"] = "fedsift_supported_override"
        base["sift"] = {
            "safety_margin_z": 1.6448536269514722,
            "minimum_control_improvement": 0.0,
            "fallback_step": 1.0,
        }
    return base


def _set_path(target: dict[str, object], path: Sequence[str], value: object) -> None:
    current = target
    for key in path[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            raise CandidateSpaceError(f"candidate path is not a mapping: {'.'.join(path)}")
        current = child
    current[path[-1]] = value


def _get_path(target: Mapping[str, object], path: Sequence[str]) -> object:
    current: object = target
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise CandidateSpaceError(f"candidate path is absent: {'.'.join(path)}")
        current = current[key]
    return current


def _different_leaf_paths(
    left: object, right: object, path: tuple[str, ...] = ()
) -> tuple[tuple[str, ...], ...]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return (path,)
        differences: list[tuple[str, ...]] = []
        for key in sorted(left):
            if not isinstance(key, str):
                return (path,)
            differences.extend(_different_leaf_paths(left[key], right[key], (*path, key)))
        return tuple(differences)
    return () if left == right else (path,)


def _digest_int(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256(_canonical_bytes(list(parts))).digest()[:8], "big")


def _lhs_coordinate(design_seed: str, coupling_key: str, row: int, row_count: int) -> float:
    strata = sorted(
        range(row_count),
        key=lambda stratum: (
            _digest_int("permutation", design_seed, coupling_key, row_count, stratum),
            stratum,
        ),
    )
    selected_stratum = strata[row]
    jitter_bits = _digest_int("jitter", design_seed, coupling_key, row_count, row)
    jitter = (jitter_bits + 0.5) / float(2**64)
    return float((selected_stratum + jitter) / row_count)


def _candidate_id(index: int) -> str:
    return f"candidate_{index:04d}"


def _hash_candidate(candidate: Mapping[str, object]) -> dict[str, object]:
    payload = copy.deepcopy(dict(candidate))
    payload.pop("candidate_sha256", None)
    payload["candidate_sha256"] = canonical_sha256(payload)
    return payload


def _hash_roster(roster: Mapping[str, object]) -> dict[str, object]:
    payload = copy.deepcopy(dict(roster))
    payload.pop("roster_sha256", None)
    payload["roster_sha256"] = canonical_sha256(payload)
    return payload


def _hash_bundle(bundle: Mapping[str, object]) -> dict[str, object]:
    payload = copy.deepcopy(dict(bundle))
    payload.pop("manifest_sha256", None)
    payload["manifest_sha256"] = canonical_sha256(payload)
    return payload


def _build_method_roster(method: str, count: int, design_seed: str) -> dict[str, object]:
    dimensions = METHOD_DIMENSIONS[method]
    if len({dimension.coupling_key for dimension in dimensions}) != len(dimensions):
        raise CandidateSpaceError(f"{method} repeats an LHS coupling key")
    sentinels = METHOD_SOURCE_SENTINELS[method]
    if len({sentinel.sentinel_id for sentinel in sentinels}) != len(sentinels):
        raise CandidateSpaceError(f"{method} repeats a source sentinel id")
    dimension_by_key = {dimension.coupling_key: dimension for dimension in dimensions}
    for sentinel in sentinels:
        dimension = dimension_by_key.get(sentinel.coupling_key)
        if dimension is None or dimension.path != sentinel.path:
            raise CandidateSpaceError(f"{method} source sentinel is not one registered dimension")
    lhs_rows = count - 1 - len(sentinels)
    if lhs_rows < 1:
        raise CandidateSpaceError(f"{method} candidate budget leaves no normative LHS row")
    candidates: list[dict[str, object]] = []
    base_parameters = _base_parameters(method)
    candidates.append(
        _hash_candidate(
            {
                "candidate_id": _candidate_id(0),
                "generation": copy.deepcopy(REFERENCE_BASIS[method]),
                "parameters": base_parameters,
            }
        )
    )
    for sentinel_index, sentinel in enumerate(sentinels, start=1):
        parameters = _base_parameters(method)
        reference = _get_path(parameters, sentinel.path)
        if (
            isinstance(reference, bool)
            or not isinstance(reference, (int, float))
            or reference == sentinel.value
        ):
            raise CandidateSpaceError(
                f"{method} source sentinel must change its candidate0 coordinate"
            )
        _set_path(parameters, sentinel.path, sentinel.value)
        if _different_leaf_paths(base_parameters, parameters) != (sentinel.path,):
            raise CandidateSpaceError(f"{method} source sentinel is not a one-factor candidate")
        candidates.append(
            _hash_candidate(
                {
                    "candidate_id": _candidate_id(sentinel_index),
                    "generation": sentinel.generation(reference_value=reference),
                    "parameters": parameters,
                }
            )
        )
    first_lhs_index = len(sentinels) + 1
    for row in range(lhs_rows):
        parameters = _base_parameters(method)
        coordinates: dict[str, float] = {}
        strata: dict[str, int] = {}
        for dimension in dimensions:
            coordinate = _lhs_coordinate(design_seed, dimension.coupling_key, row, lhs_rows)
            coordinates[dimension.coupling_key] = coordinate
            strata[dimension.coupling_key] = min(int(coordinate * lhs_rows), lhs_rows - 1)
            _set_path(parameters, dimension.path, dimension.map_coordinate(coordinate))
        candidates.append(
            _hash_candidate(
                {
                    "candidate_id": _candidate_id(first_lhs_index + row),
                    "generation": {
                        "kind": "lhs",
                        "design": "normative_latin_hypercube_v1",
                        "lhs_row": row,
                        "lhs_row_count": lhs_rows,
                        "coordinates": coordinates,
                        "strata": strata,
                    },
                    "parameters": parameters,
                }
            )
        )
    return _hash_roster(
        {
            "method": method,
            "identity": REFERENCE_BASIS[method],
            "candidate_count": count,
            "source_sentinel_count": len(sentinels),
            "lhs_count": lhs_rows,
            "dimensions": [dimension.as_dict() for dimension in dimensions],
            "candidates": candidates,
        }
    )


def build_candidate_space(
    study_id: str,
    *,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    design_seed: str = DEFAULT_DESIGN_SEED,
) -> dict[str, object]:
    """Build all equal-sized, result-blind main-method rosters."""
    if not isinstance(study_id, str) or not study_id.strip():
        raise CandidateSpaceError("study_id must be a non-empty string")
    count = require_exact_int(candidate_count, "candidate_count", minimum=MIN_CANDIDATE_COUNT)
    if count > 256:
        raise CandidateSpaceError("candidate_count exceeds the preregistration safety cap")
    if not isinstance(design_seed, str) or not design_seed:
        raise CandidateSpaceError("design_seed must be a non-empty string")
    methods: dict[str, object] = {}
    for method in MAIN_METHODS:
        methods[method] = _build_method_roster(method, count, design_seed)
    result = _hash_bundle(
        {
            "schema": _identity("candidate_space"),
            "status": "DRAFT_NOT_FROZEN",
            "study_id": study_id,
            "method_order": list(MAIN_METHODS),
            "candidate_count_per_method": count,
            "design": {
                "kind": "candidate0_plus_source_sentinels_plus_normative_lhs_v1",
                "design_seed": design_seed,
                "candidate0_policy": "literature_or_current_transparent_reference",
                "source_sentinel_policy": "predeclared_primary_source_one_factor_from_candidate0",
                "shared_coordinate_policy": "universal_local_sentinel_ids_and_fedopt_family_numeric_sentinel_lhs_ids_are_coupled",
                "minimum_candidate_count": MIN_CANDIDATE_COUNT,
                "outcome_or_prediction_fields_allowed": False,
            },
            "methods": methods,
        }
    )
    validate_candidate_space(result)
    return result


def _expected_hash(value: Mapping[str, object], field: str) -> str:
    payload = copy.deepcopy(dict(value))
    payload.pop(field, None)
    return canonical_sha256(payload)


def validate_candidate_space(bundle: Mapping[str, object], *, require_frozen: bool = False) -> None:
    assert_no_performance_fields(bundle)
    if not isinstance(bundle, Mapping):
        raise CandidateSpaceError("candidate space must be a mapping")
    if bundle.get("schema") != _identity("candidate_space"):
        raise CandidateSpaceError("candidate-space schema mismatch")
    status = bundle.get("status")
    if status not in {"DRAFT_NOT_FROZEN", "FROZEN"}:
        raise CandidateSpaceError("candidate-space status is invalid")
    expected_top_level = {
        "schema",
        "status",
        "study_id",
        "method_order",
        "candidate_count_per_method",
        "design",
        "methods",
        "manifest_sha256",
    }
    if status == "FROZEN":
        expected_top_level.add("freeze_bindings")
    if set(bundle) != expected_top_level:
        raise CandidateSpaceError("candidate-space top-level fields differ from the schema")
    study_id = bundle.get("study_id")
    if not isinstance(study_id, str) or not study_id.strip():
        raise CandidateSpaceError("candidate-space study_id must be non-empty")
    if require_frozen and status != "FROZEN":
        raise CandidateSpaceError("HPO rejects a candidate space that is not frozen")
    if bundle.get("method_order") != list(MAIN_METHODS):
        raise CandidateSpaceError("the exact ten-method order is not frozen")
    count = require_exact_int(
        bundle.get("candidate_count_per_method"),
        "candidate_count_per_method",
        minimum=MIN_CANDIDATE_COUNT,
    )
    methods = bundle.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != set(MAIN_METHODS):
        raise CandidateSpaceError("candidate space must contain exactly ten main methods")
    design = bundle.get("design")
    if not isinstance(design, Mapping) or set(design) != {
        "kind",
        "design_seed",
        "candidate0_policy",
        "source_sentinel_policy",
        "shared_coordinate_policy",
        "minimum_candidate_count",
        "outcome_or_prediction_fields_allowed",
    }:
        raise CandidateSpaceError("candidate-space design contract is malformed")
    if design.get("kind") != "candidate0_plus_source_sentinels_plus_normative_lhs_v1":
        raise CandidateSpaceError("candidate-space design kind drift")
    design_seed = design.get("design_seed")
    if not isinstance(design_seed, str) or not design_seed:
        raise CandidateSpaceError("candidate-space design seed is invalid")
    if design.get("candidate0_policy") != "literature_or_current_transparent_reference":
        raise CandidateSpaceError("candidate0 policy drift")
    if (
        design.get("source_sentinel_policy")
        != "predeclared_primary_source_one_factor_from_candidate0"
    ):
        raise CandidateSpaceError("source-sentinel policy drift")
    if (
        design.get("shared_coordinate_policy")
        != "universal_local_sentinel_ids_and_fedopt_family_numeric_sentinel_lhs_ids_are_coupled"
    ):
        raise CandidateSpaceError("shared-coordinate policy drift")
    if design.get("minimum_candidate_count") != MIN_CANDIDATE_COUNT:
        raise CandidateSpaceError("minimum candidate count drift")
    if design.get("outcome_or_prediction_fields_allowed") is not False:
        raise CandidateSpaceError("result fields cannot be enabled in a candidate space")
    allowed_reference_kinds = {
        "literature_reference",
        "transparent_record_dp_adaptation",
        "literature_backend_with_record_dp",
        "current_transparent_reference",
    }
    for method in MAIN_METHODS:
        roster = methods.get(method)
        if not isinstance(roster, Mapping):
            raise CandidateSpaceError(f"missing roster for {method}")
        if set(roster) != {
            "method",
            "identity",
            "candidate_count",
            "source_sentinel_count",
            "lhs_count",
            "dimensions",
            "candidates",
            "roster_sha256",
        }:
            raise CandidateSpaceError(f"{method} roster fields differ from the schema")
        if dict(roster) != _build_method_roster(method, count, design_seed):
            raise CandidateSpaceError(f"{method} roster differs from normative generation")
        if roster.get("method") != method or roster.get("candidate_count") != count:
            raise CandidateSpaceError(f"{method} violates the equal candidate budget")
        candidates = roster.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != count:
            raise CandidateSpaceError(f"{method} candidate roster length mismatch")
        expected_sentinels = len(METHOD_SOURCE_SENTINELS[method])
        if roster.get("source_sentinel_count") != expected_sentinels:
            raise CandidateSpaceError(f"{method} source-sentinel count drift")
        if roster.get("lhs_count") != count - expected_sentinels - 1:
            raise CandidateSpaceError(f"{method} LHS count drift")
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise CandidateSpaceError(f"{method} candidate is not a mapping")
            if set(candidate) != {"candidate_id", "generation", "parameters", "candidate_sha256"}:
                raise CandidateSpaceError(f"{method} candidate fields differ from the schema")
            if candidate.get("candidate_id") != _candidate_id(index):
                raise CandidateSpaceError(f"{method} candidate ids are not canonical")
            if candidate.get("candidate_sha256") != _expected_hash(candidate, "candidate_sha256"):
                raise CandidateSpaceError(f"{method} candidate hash drift")
            generation = candidate.get("generation")
            if not isinstance(generation, Mapping):
                raise CandidateSpaceError(f"{method} candidate generation is missing")
            if index == 0:
                if generation.get("kind") not in allowed_reference_kinds:
                    raise CandidateSpaceError(f"{method} candidate0 is not transparent")
                if not isinstance(generation.get("basis"), str) or not generation.get("basis"):
                    raise CandidateSpaceError(f"{method} candidate0 lacks its basis")
            elif index <= expected_sentinels:
                if generation.get("kind") != "source_sentinel":
                    raise CandidateSpaceError(
                        f"{method} pre-LHS candidate is not a source sentinel"
                    )
            elif generation.get("kind") != "lhs":
                raise CandidateSpaceError(f"{method} post-sentinel candidate is not LHS")
        if roster.get("roster_sha256") != _expected_hash(roster, "roster_sha256"):
            raise CandidateSpaceError(f"{method} roster hash drift")
    if status == "FROZEN":
        bindings = bundle.get("freeze_bindings")
        if not isinstance(bindings, Mapping) or set(bindings) != {
            "protocol_sha256",
            "implementation_contract_sha256",
        }:
            raise CandidateSpaceError("frozen candidate space lacks exact freeze bindings")
        for key, value in bindings.items():
            require_sha256(value, str(key))
    if bundle.get("manifest_sha256") != _expected_hash(bundle, "manifest_sha256"):
        raise CandidateSpaceError("candidate-space manifest hash drift")


def freeze_candidate_space(
    bundle: Mapping[str, object], *, protocol_sha256: str, implementation_contract_sha256: str
) -> dict[str, object]:
    """Bind a reviewed draft to immutable upstream protocol/code identities."""
    validate_candidate_space(bundle)
    if bundle.get("status") != "DRAFT_NOT_FROZEN":
        raise CandidateSpaceError("only a draft candidate space can be frozen")
    frozen = copy.deepcopy(dict(bundle))
    frozen["status"] = "FROZEN"
    frozen["freeze_bindings"] = {
        "protocol_sha256": require_sha256(protocol_sha256, "protocol_sha256"),
        "implementation_contract_sha256": require_sha256(
            implementation_contract_sha256, "implementation_contract_sha256"
        ),
    }
    frozen = _hash_bundle(frozen)
    validate_candidate_space(frozen, require_frozen=True)
    return frozen


def candidate_by_id(
    bundle: Mapping[str, object], method: str, candidate_id: str
) -> dict[str, object]:
    validate_candidate_space(bundle)
    if method not in MAIN_METHODS:
        raise CandidateSpaceError(f"unknown main method: {method}")
    methods = bundle["methods"]
    assert isinstance(methods, Mapping)
    roster = methods[method]
    assert isinstance(roster, Mapping)
    candidates = roster["candidates"]
    assert isinstance(candidates, list)
    for candidate in candidates:
        if candidate["candidate_id"] == candidate_id:
            return copy.deepcopy(candidate)
    raise CandidateSpaceError(f"unknown candidate id for {method}: {candidate_id}")
