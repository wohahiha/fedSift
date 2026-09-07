"""Result-blind public-control rules for FedSift and its equal-information arm.

Both rules consume the same candidate probability table.  The public-argmin
arm selects the smallest mean control log loss.  FedSift accepts a partial-step
override only when its paired loss improvement over the full step passes the
predeclared descriptive margin; otherwise it falls back to the full step.

The margin is deliberately labelled a heuristic.  Reusing one control set in
adaptive rounds does not turn a normal-standard-error expression into a
finite-sample or clinical guarantee.
"""

from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import copy
import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Mapping, Sequence
from .probability_contract import PROBABILITY_CLIP_EPSILON


class ControlRuleError(ValueError):
    """Raised when public-control evidence or a decision receipt is invalid."""


_SHA256_LENGTH = 64
_LOG_CLIP = PROBABILITY_CLIP_EPSILON
_CANDIDATE_FIELDS = {"order", "alpha", "mean_log_loss", "per_record_log_loss", "probability"}


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True
        )
    except (TypeError, ValueError) as exc:
        raise ControlRuleError("value is not canonical-JSON serializable") from exc


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or value.lower() != value
        or any((character not in "0123456789abcdef" for character in value))
    ):
        raise ControlRuleError(f"{field} must be a lowercase SHA-256 hex string")
    return value


def _require_identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ControlRuleError(f"{field} must be a non-empty canonical string")
    return value


def _require_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ControlRuleError(f"{field} must be an exact integer")
    result = int(value)
    if result < minimum:
        raise ControlRuleError(f"{field} must be >= {minimum}")
    return result


def _require_float(
    value: object, field: str, *, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ControlRuleError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ControlRuleError(f"{field} must be finite")
    if minimum is not None and result < minimum:
        raise ControlRuleError(f"{field} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ControlRuleError(f"{field} must be <= {maximum}")
    if result == 0.0:
        result = 0.0
    return result


@dataclass(frozen=True)
class ControlDecisionScope:
    study_id: str
    dataset_id: str
    outer_repeat: int
    outer_fold: int
    eval_seed: int
    candidate_id: str
    server_round: int
    query_index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "study_id", _require_identifier(self.study_id, "study_id"))
        object.__setattr__(self, "dataset_id", _require_identifier(self.dataset_id, "dataset_id"))
        object.__setattr__(self, "outer_repeat", _require_int(self.outer_repeat, "outer_repeat"))
        object.__setattr__(self, "outer_fold", _require_int(self.outer_fold, "outer_fold"))
        object.__setattr__(self, "eval_seed", _require_int(self.eval_seed, "eval_seed"))
        object.__setattr__(
            self, "candidate_id", _require_identifier(self.candidate_id, "candidate_id")
        )
        object.__setattr__(
            self, "server_round", _require_int(self.server_round, "server_round", minimum=1)
        )
        object.__setattr__(
            self, "query_index", _require_int(self.query_index, "query_index", minimum=1)
        )

    def manifest(self) -> dict[str, object]:
        return {
            "study_id": self.study_id,
            "dataset_id": self.dataset_id,
            "outer_repeat": self.outer_repeat,
            "outer_fold": self.outer_fold,
            "eval_seed": self.eval_seed,
            "candidate_id": self.candidate_id,
            "server_round": self.server_round,
            "query_index": self.query_index,
        }


@dataclass(frozen=True)
class ControlEvidenceBinding:
    control_membership_sha256: str
    preprocessing_artifact_sha256: str
    model_manifest_sha256: str
    candidate_parameters_sha256: str
    global_state_sha256: str
    aggregate_direction_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "control_membership_sha256",
            "preprocessing_artifact_sha256",
            "model_manifest_sha256",
            "candidate_parameters_sha256",
            "global_state_sha256",
            "aggregate_direction_sha256",
        ):
            object.__setattr__(self, field, _require_sha256(getattr(self, field), field))

    def manifest(self) -> dict[str, str]:
        return {
            "control_membership_sha256": self.control_membership_sha256,
            "preprocessing_artifact_sha256": self.preprocessing_artifact_sha256,
            "model_manifest_sha256": self.model_manifest_sha256,
            "candidate_parameters_sha256": self.candidate_parameters_sha256,
            "global_state_sha256": self.global_state_sha256,
            "aggregate_direction_sha256": self.aggregate_direction_sha256,
        }


def _canonical_rows(row_ids: object, labels: object) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if (
        isinstance(row_ids, (str, bytes))
        or not isinstance(row_ids, Sequence)
        or isinstance(labels, (str, bytes))
        or (not isinstance(labels, Sequence))
    ):
        raise ControlRuleError("row_ids and labels must be one-dimensional sequences")
    raw_ids = list(row_ids)
    raw_labels = list(labels)
    if not raw_ids or len(raw_ids) != len(raw_labels):
        raise ControlRuleError("control row_ids and labels must be non-empty and aligned")
    canonical_ids = tuple(
        (_require_int(value, f"row_ids[{index}]") for (index, value) in enumerate(raw_ids))
    )
    if len(set(canonical_ids)) != len(canonical_ids):
        raise ControlRuleError("control row_ids must be unique")
    canonical_labels: list[int] = []
    for index, value in enumerate(raw_labels):
        label = _require_int(value, f"labels[{index}]")
        if label not in (0, 1):
            raise ControlRuleError("control labels must be exact binary values")
        canonical_labels.append(label)
    if set(canonical_labels) != {0, 1}:
        raise ControlRuleError("control evidence must contain both binary classes")
    return (canonical_ids, tuple(canonical_labels))


def _expected_alphas(values: object) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, tuple) or (not values):
        raise ControlRuleError("expected_alphas must be an explicit non-empty tuple")
    result = tuple(
        (
            _require_float(value, f"expected_alphas[{index}]", minimum=0.0, maximum=1.0)
            for (index, value) in enumerate(values)
        )
    )
    if result != tuple(sorted(result)) or len(set(result)) != len(result):
        raise ControlRuleError("expected_alphas must be strictly increasing and unique")
    if 0.0 not in result or 1.0 not in result:
        raise ControlRuleError("the shared candidate grid must contain zero and the full step")
    return result


def _validated_candidate_table(
    candidate_table: object, labels: tuple[int, ...], expected_alphas: tuple[float, ...]
) -> tuple[list[dict[str, object]], list[tuple[float, ...]]]:
    if isinstance(candidate_table, (str, bytes)) or not isinstance(candidate_table, Sequence):
        raise ControlRuleError("candidate_table must be a sequence")
    raw_rows = list(candidate_table)
    if len(raw_rows) != len(expected_alphas):
        raise ControlRuleError("candidate table length differs from the frozen alpha grid")
    summaries: list[dict[str, object]] = []
    losses_by_candidate: list[tuple[float, ...]] = []
    for order, (raw_row, expected_alpha) in enumerate(zip(raw_rows, expected_alphas)):
        if not isinstance(raw_row, Mapping) or set(raw_row) != _CANDIDATE_FIELDS:
            raise ControlRuleError("candidate row has an unexpected schema")
        if _require_int(raw_row.get("order"), "candidate order") != order:
            raise ControlRuleError("candidate order differs from the frozen grid")
        alpha = _require_float(raw_row.get("alpha"), "candidate alpha", minimum=0.0, maximum=1.0)
        if alpha != expected_alpha:
            raise ControlRuleError("candidate alpha differs from the frozen grid")
        probabilities = raw_row.get("probability")
        supplied_losses = raw_row.get("per_record_log_loss")
        if (
            isinstance(probabilities, (str, bytes))
            or not isinstance(probabilities, Sequence)
            or isinstance(supplied_losses, (str, bytes))
            or (not isinstance(supplied_losses, Sequence))
        ):
            raise ControlRuleError("candidate probabilities and losses must be sequences")
        probability_values = list(probabilities)
        loss_values = list(supplied_losses)
        if len(probability_values) != len(labels) or len(loss_values) != len(labels):
            raise ControlRuleError("candidate evidence is not row aligned")
        recomputed_losses: list[float] = []
        probability_hex: list[str] = []
        for index, (label, probability_value, supplied_loss_value) in enumerate(
            zip(labels, probability_values, loss_values)
        ):
            probability = _require_float(
                probability_value,
                f"candidate[{order}].probability[{index}]",
                minimum=0.0,
                maximum=1.0,
            )
            clipped = min(max(probability, _LOG_CLIP), 1.0 - _LOG_CLIP)
            expected_loss = -math.log(clipped) if label == 1 else -math.log1p(-clipped)
            supplied_loss = _require_float(
                supplied_loss_value, f"candidate[{order}].per_record_log_loss[{index}]", minimum=0.0
            )
            if not math.isclose(supplied_loss, expected_loss, rel_tol=1e-12, abs_tol=1e-12):
                raise ControlRuleError(
                    "candidate loss is inconsistent with labels and probabilities"
                )
            recomputed_losses.append(expected_loss)
            probability_hex.append(probability.hex())
        mean_loss = math.fsum(recomputed_losses) / len(recomputed_losses)
        supplied_mean = _require_float(
            raw_row.get("mean_log_loss"), "candidate mean_log_loss", minimum=0.0
        )
        if not math.isclose(supplied_mean, mean_loss, rel_tol=1e-12, abs_tol=1e-12):
            raise ControlRuleError("candidate mean loss is inconsistent with row losses")
        losses = tuple(recomputed_losses)
        losses_by_candidate.append(losses)
        summaries.append(
            {
                "order": order,
                "alpha_hex": alpha.hex(),
                "mean_log_loss_hex": mean_loss.hex(),
                "probabilities_sha256": canonical_sha256(probability_hex),
                "losses_sha256": canonical_sha256([value.hex() for value in losses]),
            }
        )
    return (summaries, losses_by_candidate)


def _decision_payload(
    *,
    rule_name: str,
    scope: ControlDecisionScope,
    binding: ControlEvidenceBinding,
    row_ids: object,
    labels: object,
    candidate_table: object,
    expected_alphas: object,
    safety_margin_z: object = None,
    minimum_control_improvement: object = None,
) -> dict[str, object]:
    if rule_name not in {"public_argmin_mean_logloss", "fedsift_supported_override"}:
        raise ControlRuleError("unknown public-control rule")
    if not isinstance(scope, ControlDecisionScope):
        raise ControlRuleError("scope must be a ControlDecisionScope")
    if not isinstance(binding, ControlEvidenceBinding):
        raise ControlRuleError("binding must be a ControlEvidenceBinding")
    canonical_ids, canonical_labels = _canonical_rows(row_ids, labels)
    alphas = _expected_alphas(expected_alphas)
    summaries, losses_by_candidate = _validated_candidate_table(
        candidate_table, canonical_labels, alphas
    )
    means = [math.fsum(losses) / len(losses) for losses in losses_by_candidate]
    rule: dict[str, object] = {
        "name": rule_name,
        "candidate_alphas_hex": [alpha.hex() for alpha in alphas],
        "classification_or_test_metric_consumed": False,
        "control_metric": "paired_binary_log_loss",
        "tie_break": "larger_alpha",
        "prediction_rule": "candidate_probabilities_on_same_frozen_V_ctrl_rows",
    }
    decision_rows: list[dict[str, object]] = []
    if rule_name == "public_argmin_mean_logloss":
        if safety_margin_z is not None or minimum_control_improvement is not None:
            raise ControlRuleError("public argmin cannot receive FedSift margin parameters")
        selected_index = min(range(len(alphas)), key=lambda index: (means[index], -alphas[index]))
        status = "selected_public_mean_logloss_argmin"
        for index, alpha in enumerate(alphas):
            decision_rows.append({**summaries[index], "selected": index == selected_index})
    else:
        margin_z = _require_float(safety_margin_z, "safety_margin_z", minimum=0.0)
        minimum_improvement = _require_float(
            minimum_control_improvement, "minimum_control_improvement", minimum=0.0
        )
        rule.update(
            {
                "full_step_alpha_hex": (1.0).hex(),
                "fallback_alpha_hex": (1.0).hex(),
                "safety_margin_z_hex": margin_z.hex(),
                "minimum_control_improvement_hex": minimum_improvement.hex(),
                "paired_standard_error": "sample_sd_ddof_1_div_sqrt_n",
                "override_acceptance_rule": "alpha_ne_full_and_mean_improvement_gt_minimum_and_mean_minus_z_times_paired_se_gt_zero",
                "supported_candidate_selection": "minimum_mean_log_loss_then_larger_alpha",
                "margin_interpretation": "descriptive_paired_standard_error_heuristic_not_a_finite_sample_or_adaptive_reuse_guarantee",
            }
        )
        full_index = alphas.index(1.0)
        full_losses = losses_by_candidate[full_index]
        supported: list[int] = []
        for index, alpha in enumerate(alphas):
            differences = tuple(
                (
                    full_loss - candidate_loss
                    for (full_loss, candidate_loss) in zip(full_losses, losses_by_candidate[index])
                )
            )
            mean_improvement = math.fsum(differences) / len(differences)
            if len(differences) > 1:
                centered = math.fsum(((value - mean_improvement) ** 2 for value in differences))
                standard_error = math.sqrt(centered / (len(differences) - 1)) / math.sqrt(
                    len(differences)
                )
                lower_margin = mean_improvement - margin_z * standard_error
                standard_error_hex: str | None = standard_error.hex()
                lower_margin_hex: str | None = lower_margin.hex()
                passes = (
                    alpha != 1.0 and mean_improvement > minimum_improvement and (lower_margin > 0.0)
                )
            else:
                standard_error_hex = None
                lower_margin_hex = None
                passes = False
            if passes:
                supported.append(index)
            decision_rows.append(
                {
                    **summaries[index],
                    "paired_improvement_vs_full_hex": mean_improvement.hex(),
                    "paired_standard_error_hex": standard_error_hex,
                    "heuristic_lower_margin_hex": lower_margin_hex,
                    "passes_supported_override": passes,
                }
            )
        if supported:
            selected_index = min(supported, key=lambda index: (means[index], -alphas[index]))
            status = "selected_by_paired_loss_safety_heuristic"
        else:
            selected_index = full_index
            status = "fallback_full_step_no_supported_override"
        for index, row in enumerate(decision_rows):
            row["selected"] = index == selected_index
    raw_table = copy.deepcopy(list(candidate_table))
    evidence = {
        "control_row_count": len(canonical_ids),
        "control_row_ids_sha256": canonical_sha256(list(canonical_ids)),
        "control_labels_sha256": canonical_sha256(list(canonical_labels)),
        "candidate_table_sha256": canonical_sha256(raw_table),
        "candidate_probability_payload_sha256": canonical_sha256(
            {
                "row_ids": list(canonical_ids),
                "labels": list(canonical_labels),
                "candidate_table": raw_table,
            }
        ),
    }
    receipt: dict[str, object] = {
        "schema": _identity("public_control_decision_receipt"),
        "status": "complete",
        "scope": scope.manifest(),
        "evidence_binding": binding.manifest(),
        "information_interface": "global_state_plus_single_aggregate_direction_plus_same_frozen_V_ctrl",
        "rule": rule,
        "evidence": evidence,
        "decision_rows": decision_rows,
        "selection_status": status,
        "selected_alpha_hex": alphas[selected_index].hex(),
        "outer_test_consumed": False,
        "clinical_safety_claim": False,
        "statistical_guarantee_claim": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def decide_public_argmin(
    *,
    scope: ControlDecisionScope,
    binding: ControlEvidenceBinding,
    row_ids: object,
    labels: object,
    candidate_table: object,
    expected_alphas: object,
) -> dict[str, object]:
    return _decision_payload(
        rule_name="public_argmin_mean_logloss",
        scope=scope,
        binding=binding,
        row_ids=row_ids,
        labels=labels,
        candidate_table=candidate_table,
        expected_alphas=expected_alphas,
    )


def decide_fedsift(
    *,
    scope: ControlDecisionScope,
    binding: ControlEvidenceBinding,
    row_ids: object,
    labels: object,
    candidate_table: object,
    expected_alphas: object,
    safety_margin_z: object,
    minimum_control_improvement: object,
) -> dict[str, object]:
    return _decision_payload(
        rule_name="fedsift_supported_override",
        scope=scope,
        binding=binding,
        row_ids=row_ids,
        labels=labels,
        candidate_table=candidate_table,
        expected_alphas=expected_alphas,
        safety_margin_z=safety_margin_z,
        minimum_control_improvement=minimum_control_improvement,
    )


def validate_control_decision_receipt(
    receipt: Mapping[str, object],
    *,
    expected_receipt_sha256: str,
    rule_name: str,
    scope: ControlDecisionScope,
    binding: ControlEvidenceBinding,
    row_ids: object,
    labels: object,
    candidate_table: object,
    expected_alphas: object,
    safety_margin_z: object = None,
    minimum_control_improvement: object = None,
) -> dict[str, object]:
    """Rebuild a decision from raw V_ctrl evidence and exact-compare the receipt."""
    expected_hash = _require_sha256(expected_receipt_sha256, "expected_receipt_sha256")
    if not isinstance(receipt, Mapping):
        raise ControlRuleError("receipt must be a mapping")
    rebuilt = _decision_payload(
        rule_name=rule_name,
        scope=scope,
        binding=binding,
        row_ids=row_ids,
        labels=labels,
        candidate_table=candidate_table,
        expected_alphas=expected_alphas,
        safety_margin_z=safety_margin_z,
        minimum_control_improvement=minimum_control_improvement,
    )
    if dict(receipt) != rebuilt:
        raise ControlRuleError("decision receipt differs from raw-evidence reconstruction")
    if rebuilt["receipt_sha256"] != expected_hash:
        raise ControlRuleError("decision receipt differs from the externally committed hash")
    return rebuilt


def validate_information_matched_pair(
    fedsift_receipt: Mapping[str, object],
    public_argmin_receipt: Mapping[str, object],
    *,
    expected_fedsift_receipt_sha256: str,
    expected_public_argmin_receipt_sha256: str,
    scope: ControlDecisionScope,
    binding: ControlEvidenceBinding,
    row_ids: object,
    labels: object,
    candidate_table: object,
    expected_alphas: object,
    safety_margin_z: object,
    minimum_control_improvement: object,
) -> dict[str, object]:
    """Rebuild both rules from one raw V_ctrl table, then compare information.

    Self-hashed dictionaries alone only prove internal consistency.  Exact raw
    reconstruction plus two externally committed receipt hashes is required
    before this function can attest an information-matched comparison.

    This is deliberately a *single-decision counterfactual* check.  It holds
    one bound global state, aggregate direction, candidate table and V_ctrl
    evidence fixed while changing only the public selection rule.  Once either
    rule is executed recursively, its selected alpha may change the next model
    state and every later candidate table.  Consequently this function cannot
    attest fairness, equality, or reproducibility of two independently run
    recursive trajectories; that requires the result-blind trajectory plan and
    validator in :mod:`fedsift.control_trajectory`.
    """
    fedsift_receipt = validate_control_decision_receipt(
        fedsift_receipt,
        expected_receipt_sha256=expected_fedsift_receipt_sha256,
        rule_name="fedsift_supported_override",
        scope=scope,
        binding=binding,
        row_ids=row_ids,
        labels=labels,
        candidate_table=candidate_table,
        expected_alphas=expected_alphas,
        safety_margin_z=safety_margin_z,
        minimum_control_improvement=minimum_control_improvement,
    )
    public_argmin_receipt = validate_control_decision_receipt(
        public_argmin_receipt,
        expected_receipt_sha256=expected_public_argmin_receipt_sha256,
        rule_name="public_argmin_mean_logloss",
        scope=scope,
        binding=binding,
        row_ids=row_ids,
        labels=labels,
        candidate_table=candidate_table,
        expected_alphas=expected_alphas,
    )
    for name, receipt in (("fedsift", fedsift_receipt), ("public_argmin", public_argmin_receipt)):
        if not isinstance(receipt, Mapping) or receipt.get("schema") != _identity(
            "public_control_decision_receipt"
        ):
            raise ControlRuleError(f"{name} receipt schema is invalid")
        payload = copy.deepcopy(dict(receipt))
        digest = payload.pop("receipt_sha256", None)
        if digest != canonical_sha256(payload):
            raise ControlRuleError(f"{name} receipt self-hash is invalid")
    f_scope = dict(fedsift_receipt.get("scope", {}))
    p_scope = dict(public_argmin_receipt.get("scope", {}))
    if f_scope != p_scope:
        raise ControlRuleError("information-matched receipts have different scopes")
    for field in ("evidence_binding", "information_interface", "evidence", "outer_test_consumed"):
        if fedsift_receipt.get(field) != public_argmin_receipt.get(field):
            raise ControlRuleError(f"information-matched receipts differ in {field}")
    f_rule = fedsift_receipt.get("rule")
    p_rule = public_argmin_receipt.get("rule")
    if not isinstance(f_rule, Mapping) or not isinstance(p_rule, Mapping):
        raise ControlRuleError("control rule manifests are missing")
    if f_rule.get("candidate_alphas_hex") != p_rule.get("candidate_alphas_hex"):
        raise ControlRuleError("information-matched receipts use different candidate grids")
    result: dict[str, object] = {
        "schema": _identity("information_matched_control_pair"),
        "status": "verified",
        "scope": f_scope,
        "candidate_table_sha256": dict(fedsift_receipt["evidence"])["candidate_table_sha256"],
        "fedsift_receipt_sha256": fedsift_receipt["receipt_sha256"],
        "public_argmin_receipt_sha256": public_argmin_receipt["receipt_sha256"],
        "same_control_information": True,
        "same_raw_control_table_and_declared_state_bindings": True,
        "attestation_scope": "single_decision_counterfactual_on_one_bound_state_direction_and_raw_table",
        "independent_recursive_trajectories_verified": False,
        "candidate_derivation_from_bound_states_verified": False,
        "provenance_boundary": "runner_must_rebuild_candidate_predictions_from_the_bound_global_state_and_aggregate_direction_and_use_control_trajectory_validation_for_independently_executed_recursive_arms",
    }
    result["pair_sha256"] = canonical_sha256(result)
    return result


__all__ = [
    "ControlDecisionScope",
    "ControlEvidenceBinding",
    "ControlRuleError",
    "canonical_sha256",
    "decide_fedsift",
    "decide_public_argmin",
    "validate_control_decision_receipt",
    "validate_information_matched_pair",
]
