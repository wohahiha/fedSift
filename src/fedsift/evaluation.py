"""Result-blind probability evaluation and technical working-point receipts.

This module deliberately separates three operations:

* HPO evidence consists only of threshold-free probability metrics.
* A technical working point is selected from an explicitly identified
  validation-selection split (``V_sel``) under a frozen sensitivity-target
  rule.
* An outer-test report consumes a previously committed threshold receipt and
  never calls the threshold selector.

The working point is an engineering evaluation device.  It is not a clinical
threshold and this module makes no clinical-utility claim.
"""

from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Integral, Real
from typing import Any, Mapping, Sequence
import numpy as np
from .calibration import compute_calibration_metrics
from .probability_contract import PROBABILITY_CLIP_EPSILON, PROBABILITY_CLIP_POLICY


class EvaluationError(ValueError):
    """Raised when evaluation evidence or a receipt fails closed."""


TECHNICAL_TARGET_SENSITIVITY = 0.85
_SHA256_HEX_LENGTH = 64
HPO_PROBABILITY_CONTRACT: dict[str, object] = {
    "schema": _identity("hpo_probability_contract"),
    "scope": "within_outer_repeat_outer_fold_only",
    "outer_test_access": "forbidden",
    "threshold_selection": "forbidden",
    "primary_metric": "log_loss",
    "secondary_probability_metrics": ["average_precision", "auroc", "brier_score"],
    "log_loss_probability_clip_epsilon": PROBABILITY_CLIP_EPSILON,
    "log_loss_probability_clip_policy": PROBABILITY_CLIP_POLICY,
    "hpo_tie_break_alignment": [
        {"criterion": "log_loss", "direction": "minimize"},
        {"criterion": "average_precision", "direction": "maximize"},
        {"criterion": "auroc", "direction": "maximize"},
        {"criterion": "brier_score", "direction": "minimize"},
        {"criterion": "communication_bytes", "direction": "minimize_external"},
        {"criterion": "candidate_id", "direction": "lexicographic_min_external"},
    ],
    "development_context": _identity("selection_development_context"),
    "average_precision_companion": "prevalence",
    "interpretation": "development_informed_native_probability_quality_selection",
}


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True
        )
    except (TypeError, ValueError) as exc:
        raise EvaluationError("value is not canonical-JSON serializable") from exc


def canonical_sha256(value: object) -> str:
    """Return a deterministic SHA-256 over strict canonical JSON."""
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise EvaluationError(f"{field} must be a lowercase SHA-256 hex string")
    if (
        len(value) != _SHA256_HEX_LENGTH
        or value.lower() != value
        or any((character not in "0123456789abcdef" for character in value))
    ):
        raise EvaluationError(f"{field} must be a lowercase SHA-256 hex string")
    return value


def _require_identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise EvaluationError(f"{field} must be a non-empty canonical string")
    return value


def _require_exact_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise EvaluationError(f"{field} must be an exact integer")
    result = int(value)
    if result < minimum:
        raise EvaluationError(f"{field} must be >= {minimum}")
    return result


def _one_dimensional_values(value: object, field: str) -> list[object]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise EvaluationError(f"{field} must be one-dimensional")
        return value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise EvaluationError(f"{field} must be a one-dimensional sequence")
    result = list(value)
    if any((isinstance(item, (list, tuple, np.ndarray)) for item in result)):
        raise EvaluationError(f"{field} must be one-dimensional")
    return result


@dataclass(frozen=True)
class _ProbabilityRows:
    row_ids: tuple[int, ...]
    labels: tuple[int, ...]
    probabilities: tuple[float, ...]

    @property
    def positive_count(self) -> int:
        return sum(self.labels)

    @property
    def negative_count(self) -> int:
        return len(self.labels) - self.positive_count

    @property
    def prevalence(self) -> float:
        return self.positive_count / len(self.labels)

    def bindings(self) -> dict[str, object]:
        probability_hex = [value.hex() for value in self.probabilities]
        return {
            "row_count": len(self.row_ids),
            "row_ids_sha256": canonical_sha256(list(self.row_ids)),
            "labels_sha256": canonical_sha256(list(self.labels)),
            "probabilities_sha256": canonical_sha256(probability_hex),
            "prediction_payload_sha256": canonical_sha256(
                [
                    {"row_id": row_id, "label": label, "probability_hex": probability.hex()}
                    for (row_id, label, probability) in zip(
                        self.row_ids, self.labels, self.probabilities
                    )
                ]
            ),
        }


def _validated_probability_rows(
    row_ids: object, labels: object, probabilities: object, *, require_two_classes: bool = True
) -> _ProbabilityRows:
    raw_row_ids = _one_dimensional_values(row_ids, "row_ids")
    raw_labels = _one_dimensional_values(labels, "labels")
    raw_probabilities = _one_dimensional_values(probabilities, "probabilities")
    if not raw_row_ids:
        raise EvaluationError("probability rows must be non-empty")
    if not len(raw_row_ids) == len(raw_labels) == len(raw_probabilities):
        raise EvaluationError("row_ids, labels, and probabilities must have equal length")
    canonical_row_ids = tuple(
        (
            _require_exact_int(value, f"row_ids[{index}]")
            for (index, value) in enumerate(raw_row_ids)
        )
    )
    if len(set(canonical_row_ids)) != len(canonical_row_ids):
        raise EvaluationError("row_ids must be unique")
    canonical_labels: list[int] = []
    for index, value in enumerate(raw_labels):
        label = _require_exact_int(value, f"labels[{index}]")
        if label not in (0, 1):
            raise EvaluationError("labels must contain only exact binary values 0 and 1")
        canonical_labels.append(label)
    canonical_probabilities: list[float] = []
    for index, value in enumerate(raw_probabilities):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise EvaluationError(f"probabilities[{index}] must be a real number")
        probability = float(value)
        if not math.isfinite(probability):
            raise EvaluationError(f"probabilities[{index}] must be finite")
        if probability < 0.0 or probability > 1.0:
            raise EvaluationError(f"probabilities[{index}] must be in [0, 1]")
        if probability == 0.0:
            probability = 0.0
        canonical_probabilities.append(probability)
    label_set = set(canonical_labels)
    if require_two_classes and label_set != {0, 1}:
        raise EvaluationError("both binary classes must be present")
    return _ProbabilityRows(
        row_ids=canonical_row_ids,
        labels=tuple(canonical_labels),
        probabilities=tuple(canonical_probabilities),
    )


def _average_precision(rows: _ProbabilityRows) -> float:
    """Uninterpolated AP, with tied scores handled as one threshold group."""
    order = sorted(
        range(len(rows.labels)), key=lambda index: rows.probabilities[index], reverse=True
    )
    positives = rows.positive_count
    true_positives = 0
    seen = 0
    previous_recall = 0.0
    result = 0.0
    position = 0
    while position < len(order):
        score = rows.probabilities[order[position]]
        group_end = position
        group_positives = 0
        while group_end < len(order) and rows.probabilities[order[group_end]] == score:
            group_positives += rows.labels[order[group_end]]
            group_end += 1
        true_positives += group_positives
        seen += group_end - position
        recall = true_positives / positives
        precision = true_positives / seen
        result += (recall - previous_recall) * precision
        previous_recall = recall
        position = group_end
    return float(result)


def _auroc(rows: _ProbabilityRows) -> float:
    favourable_pairs = 0.0
    positive_scores = [
        probability for (label, probability) in zip(rows.labels, rows.probabilities) if label == 1
    ]
    negative_scores = [
        probability for (label, probability) in zip(rows.labels, rows.probabilities) if label == 0
    ]
    for positive in positive_scores:
        for negative in negative_scores:
            if positive > negative:
                favourable_pairs += 1.0
            elif positive == negative:
                favourable_pairs += 0.5
    return favourable_pairs / (len(positive_scores) * len(negative_scores))


def _brier_score(rows: _ProbabilityRows) -> float:
    return math.fsum(
        (
            (probability - label) ** 2
            for (label, probability) in zip(rows.labels, rows.probabilities)
        )
    ) / len(rows.labels)


def _log_loss(rows: _ProbabilityRows) -> float:
    terms: list[float] = []
    for label, probability in zip(rows.labels, rows.probabilities):
        clipped = min(max(probability, PROBABILITY_CLIP_EPSILON), 1.0 - PROBABILITY_CLIP_EPSILON)
        terms.append(-math.log(clipped) if label == 1 else -math.log1p(-clipped))
    return math.fsum(terms) / len(terms)


def _probability_metrics(rows: _ProbabilityRows) -> dict[str, float]:
    return {
        "average_precision": _average_precision(rows),
        "auroc": _auroc(rows),
        "brier_score": _brier_score(rows),
        "log_loss": _log_loss(rows),
    }


def compute_hpo_probability_metrics(
    row_ids: object, labels: object, probabilities: object
) -> dict[str, object]:
    """Compute the frozen, threshold-free HPO metric unit.

    The returned artifact deliberately has no classification threshold or
    thresholded metric.  Log loss uses the common fixed probability contract;
    AP and prevalence are bound into the same hashable artifact so they cannot
    be reported independently by this API.
    """
    rows = _validated_probability_rows(row_ids, labels, probabilities)
    report: dict[str, object] = {
        "schema": _identity("hpo_probability_metrics"),
        "status": "complete",
        "contract": copy.deepcopy(HPO_PROBABILITY_CONTRACT),
        "bindings": rows.bindings(),
        "event_rate": {
            "positive_count": rows.positive_count,
            "negative_count": rows.negative_count,
            "prevalence": rows.prevalence,
        },
        "metrics": _probability_metrics(rows),
        "threshold_metrics_present": False,
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


@dataclass(frozen=True)
class ThresholdSelectionRule:
    """Frozen technical working-point rule for explicit ``V_sel`` evidence."""

    target_sensitivity: float
    candidate_thresholds: tuple[float, ...]

    def __post_init__(self) -> None:
        if isinstance(self.target_sensitivity, bool) or not isinstance(
            self.target_sensitivity, Real
        ):
            raise EvaluationError("target_sensitivity must be an explicit real number")
        target = float(self.target_sensitivity)
        if not math.isfinite(target) or target != TECHNICAL_TARGET_SENSITIVITY:
            raise EvaluationError(
                "target_sensitivity must be explicitly fixed to 0.85"
            )
        if not isinstance(self.candidate_thresholds, tuple):
            raise EvaluationError("candidate_thresholds must be an explicit tuple")
        if not self.candidate_thresholds:
            raise EvaluationError("candidate_thresholds must be non-empty")
        canonical: list[float] = []
        for index, value in enumerate(self.candidate_thresholds):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise EvaluationError(f"candidate_thresholds[{index}] must be a real number")
            threshold = float(value)
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise EvaluationError("candidate thresholds must be finite and in [0, 1]")
            if threshold == 0.0:
                threshold = 0.0
            canonical.append(threshold)
        if canonical != sorted(canonical) or len(set(canonical)) != len(canonical):
            raise EvaluationError("candidate_thresholds must be strictly increasing and unique")
        object.__setattr__(self, "target_sensitivity", target)
        object.__setattr__(self, "candidate_thresholds", tuple(canonical))

    def manifest(self) -> dict[str, object]:
        return {
            "schema": _identity("target_sensitivity_rule"),
            "role": "technical_working_point_only",
            "clinical_threshold_claim": False,
            "clinical_utility_claim": False,
            "target_sensitivity": self.target_sensitivity,
            "candidate_thresholds_hex": [
                threshold.hex() for threshold in self.candidate_thresholds
            ],
            "classification_rule": "predicted_positive_if_probability_ge_threshold",
            "feasible_definition": "sensitivity_ge_target",
            "feasible_integer_test": "20_times_tp_ge_17_times_actual_positives",
            "feasible_lexicographic_order": ["specificity_max", "ppv_max", "threshold_max"],
            "infeasible_lexicographic_order": [
                "sensitivity_max",
                "specificity_max",
                "ppv_max",
                "threshold_max",
            ],
            "undefined_ppv_ranking_value": 0.0,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.manifest())


@dataclass(frozen=True)
class EvaluationScope:
    """Exact nested-evaluation identity shared by V_sel and outer test."""

    study_id: str
    outer_repeat: int
    outer_fold: int
    method_id: str
    candidate_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "study_id", _require_identifier(self.study_id, "study_id"))
        object.__setattr__(
            self, "outer_repeat", _require_exact_int(self.outer_repeat, "outer_repeat")
        )
        object.__setattr__(self, "outer_fold", _require_exact_int(self.outer_fold, "outer_fold"))
        object.__setattr__(self, "method_id", _require_identifier(self.method_id, "method_id"))
        object.__setattr__(
            self, "candidate_id", _require_identifier(self.candidate_id, "candidate_id")
        )

    def manifest(self) -> dict[str, object]:
        return {
            "study_id": self.study_id,
            "outer_repeat": self.outer_repeat,
            "outer_fold": self.outer_fold,
            "method_id": self.method_id,
            "candidate_id": self.candidate_id,
        }


@dataclass(frozen=True)
class VSelBinding:
    scope: EvaluationScope
    membership_sha256: str
    prediction_artifact_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "membership_sha256",
            _require_sha256(self.membership_sha256, "V_sel membership_sha256"),
        )
        object.__setattr__(
            self,
            "prediction_artifact_sha256",
            _require_sha256(self.prediction_artifact_sha256, "V_sel prediction_artifact_sha256"),
        )

    def manifest(self) -> dict[str, object]:
        return {
            "role": "V_sel",
            "scope": self.scope.manifest(),
            "membership_sha256": self.membership_sha256,
            "prediction_artifact_sha256": self.prediction_artifact_sha256,
            "prediction_artifact_content_provenance": "external_runner_must_bind_artifact_to_prediction_payload_sha256",
            "prediction_artifact_content_verified_here": False,
        }


_VERIFIED_THRESHOLD_TOKEN = object()
_VERIFIED_THRESHOLD_SECRET = hashlib.sha256(
    f"{_identity('verified_threshold')}{id(_VERIFIED_THRESHOLD_TOKEN)}".encode("ascii")
).digest()


def _verified_threshold_seal(
    *,
    threshold: float,
    receipt_sha256: str,
    scope: EvaluationScope,
    v_sel_binding: VSelBinding,
    rule_sha256: str,
) -> str:
    payload = canonical_sha256(
        {
            "threshold_hex": float(threshold).hex(),
            "receipt_sha256": receipt_sha256,
            "scope": scope.manifest(),
            "v_sel_binding": v_sel_binding.manifest(),
            "rule_sha256": rule_sha256,
        }
    ).encode("ascii")
    return hashlib.sha256(_VERIFIED_THRESHOLD_SECRET + payload).hexdigest()


@dataclass(frozen=True)
class VerifiedThresholdReceipt:
    """Capability emitted only after exact reconstruction from raw V_sel rows."""

    threshold: float
    receipt_sha256: str
    scope: EvaluationScope
    v_sel_binding: VSelBinding
    rule_sha256: str
    _verification_token: object = field(repr=False, compare=False)
    _verification_seal: str = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        self._assert_valid()

    def _assert_valid(self) -> None:
        if self._verification_token is not _VERIFIED_THRESHOLD_TOKEN:
            raise EvaluationError(
                "VerifiedThresholdReceipt can only be created by raw-evidence validation"
            )
        if not isinstance(self.scope, EvaluationScope):
            raise EvaluationError("verified threshold scope is invalid")
        if not isinstance(self.v_sel_binding, VSelBinding):
            raise EvaluationError("verified threshold V_sel binding is invalid")
        if self.scope != self.v_sel_binding.scope:
            raise EvaluationError("verified threshold scope and V_sel binding differ")
        _require_sha256(self.receipt_sha256, "verified threshold receipt_sha256")
        _require_sha256(self.rule_sha256, "verified threshold rule_sha256")
        if (
            isinstance(self.threshold, bool)
            or not isinstance(self.threshold, Real)
            or (not math.isfinite(float(self.threshold)))
            or (not 0.0 <= float(self.threshold) <= 1.0)
        ):
            raise EvaluationError("verified threshold is invalid")
        expected_seal = _verified_threshold_seal(
            threshold=float(self.threshold),
            receipt_sha256=self.receipt_sha256,
            scope=self.scope,
            v_sel_binding=self.v_sel_binding,
            rule_sha256=self.rule_sha256,
        )
        if self._verification_seal != expected_seal:
            raise EvaluationError("verified threshold capability seal is invalid")


@dataclass(frozen=True)
class OuterTestBinding:
    scope: EvaluationScope
    membership_sha256: str
    prediction_artifact_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "membership_sha256",
            _require_sha256(self.membership_sha256, "outer-test membership_sha256"),
        )
        object.__setattr__(
            self,
            "prediction_artifact_sha256",
            _require_sha256(
                self.prediction_artifact_sha256, "outer-test prediction_artifact_sha256"
            ),
        )

    def manifest(self) -> dict[str, object]:
        return {
            "role": "outer_test",
            "scope": self.scope.manifest(),
            "membership_sha256": self.membership_sha256,
            "prediction_artifact_sha256": self.prediction_artifact_sha256,
            "prediction_artifact_content_provenance": "external_runner_must_bind_artifact_to_prediction_payload_sha256",
            "prediction_artifact_content_verified_here": False,
        }


def _confusion_counts(rows: _ProbabilityRows, threshold: float) -> dict[str, int]:
    true_positive = false_positive = true_negative = false_negative = 0
    for label, probability in zip(rows.labels, rows.probabilities):
        predicted_positive = probability >= threshold
        if label == 1 and predicted_positive:
            true_positive += 1
        elif label == 1:
            false_negative += 1
        elif predicted_positive:
            false_positive += 1
        else:
            true_negative += 1
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _threshold_metrics(counts: Mapping[str, int]) -> dict[str, float | None]:
    tp = counts["true_positive"]
    fp = counts["false_positive"]
    tn = counts["true_negative"]
    fn = counts["false_negative"]
    sensitivity = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    ppv = _ratio(tp, tp + fp)
    npv = _ratio(tn, tn + fn)
    f1 = _ratio(2 * tp, 2 * tp + fp + fn)
    balanced_accuracy = (
        None if sensitivity is None or specificity is None else (sensitivity + specificity) / 2.0
    )
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = _ratio(tp * tn - fp * fn, mcc_denominator)
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
    }


def compute_prediction_metrics(
    row_ids: object, labels: object, probabilities: object, *, threshold: float
) -> dict[str, float | None]:
    """Evaluate saved predictions using the same definitions as outer evaluation.

    Only log loss clips probabilities. Ranking, Brier score, and classification
    use the original probabilities, including exact zero and one.
    """
    rows = _validated_probability_rows(row_ids, labels, probabilities)
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise EvaluationError("threshold must be a real number")
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise EvaluationError("threshold must be finite and in [0, 1]")
    return {**_probability_metrics(rows), **_threshold_metrics(_confusion_counts(rows, threshold))}


def _threshold_denominators(counts: Mapping[str, int]) -> dict[str, object]:
    tp = counts["true_positive"]
    fp = counts["false_positive"]
    tn = counts["true_negative"]
    fn = counts["false_negative"]
    return {
        "sensitivity": tp + fn,
        "specificity": tn + fp,
        "ppv": tp + fp,
        "npv": tn + fn,
        "f1": 2 * tp + fp + fn,
        "balanced_accuracy_components": {"sensitivity": tp + fn, "specificity": tn + fp},
        "mcc_squared_product": (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn),
    }


def _threshold_metric_status(metrics: Mapping[str, float | None]) -> dict[str, str]:
    return {
        name: "defined" if value is not None else ZERO_DENOMINATOR_POLICY[name]
        for (name, value) in metrics.items()
    }


def _select_threshold(rows: _ProbabilityRows, rule: ThresholdSelectionRule) -> tuple[float, str]:
    operating_points: list[tuple[float, Fraction, Fraction, Fraction, bool]] = []
    for threshold in rule.candidate_thresholds:
        counts = _confusion_counts(rows, threshold)
        actual_positives = counts["true_positive"] + counts["false_negative"]
        actual_negatives = counts["true_negative"] + counts["false_positive"]
        predicted_positives = counts["true_positive"] + counts["false_positive"]
        target_reached = 20 * counts["true_positive"] >= 17 * actual_positives
        operating_points.append(
            (
                threshold,
                Fraction(counts["true_positive"], actual_positives),
                Fraction(counts["true_negative"], actual_negatives),
                (
                    Fraction(0, 1)
                    if predicted_positives == 0
                    else Fraction(counts["true_positive"], predicted_positives)
                ),
                target_reached,
            )
        )
    feasible = [point for point in operating_points if point[4]]
    if feasible:
        selected = max(feasible, key=lambda point: (point[2], point[3], point[0]))
        return (selected[0], "target_reached")
    selected = max(operating_points, key=lambda point: (point[1], point[2], point[3], point[0]))
    return (selected[0], "target_unreachable_fallback")


_THRESHOLD_RECEIPT_FIELDS = {
    "schema",
    "status",
    "selection_source",
    "scope",
    "v_sel_binding",
    "v_sel_rows",
    "rule",
    "rule_sha256",
    "selection_branch",
    "selected_threshold_hex",
    "technical_working_point_only",
    "receipt_sha256",
}


def threshold_receipt_sha256(receipt: Mapping[str, object]) -> str:
    """Recompute a threshold receipt digest without mutating the receipt."""
    payload = copy.deepcopy(dict(receipt))
    payload.pop("receipt_sha256", None)
    return canonical_sha256(payload)


def select_threshold_from_v_sel(
    *,
    v_sel_binding: VSelBinding,
    row_ids: object,
    labels: object,
    probabilities: object,
    rule: ThresholdSelectionRule,
) -> dict[str, object]:
    """Select a technical working point from explicit V_sel evidence only."""
    if not isinstance(v_sel_binding, VSelBinding):
        raise EvaluationError("v_sel_binding must be a VSelBinding")
    if not isinstance(rule, ThresholdSelectionRule):
        raise EvaluationError("rule must be a ThresholdSelectionRule")
    rows = _validated_probability_rows(row_ids, labels, probabilities)
    threshold, branch = _select_threshold(rows, rule)
    receipt: dict[str, object] = {
        "schema": _identity("threshold_receipt"),
        "status": "complete",
        "selection_source": "V_sel_only",
        "scope": v_sel_binding.scope.manifest(),
        "v_sel_binding": v_sel_binding.manifest(),
        "v_sel_rows": rows.bindings(),
        "rule": rule.manifest(),
        "rule_sha256": rule.sha256,
        "selection_branch": branch,
        "selected_threshold_hex": threshold.hex(),
        "technical_working_point_only": True,
    }
    receipt["receipt_sha256"] = threshold_receipt_sha256(receipt)
    return receipt


def validate_threshold_receipt(
    receipt: Mapping[str, object],
    *,
    expected_receipt_sha256: str,
    expected_v_sel_binding: VSelBinding,
    expected_rule: ThresholdSelectionRule,
    row_ids: object,
    labels: object,
    probabilities: object,
) -> VerifiedThresholdReceipt:
    """Rebuild a committed receipt from raw V_sel evidence.

    A self-hash plus an external commitment is insufficient when an invalid
    receipt could have been committed before outer-test access.  This gate
    therefore recomputes every row/prediction/rule binding and the selected
    threshold from the original V_sel values, then requires exact equality.
    """
    expected_digest = _require_sha256(expected_receipt_sha256, "expected_receipt_sha256")
    if not isinstance(receipt, Mapping) or set(receipt) != _THRESHOLD_RECEIPT_FIELDS:
        raise EvaluationError("threshold receipt has an unexpected schema")
    if receipt.get("schema") != _identity("threshold_receipt"):
        raise EvaluationError("threshold receipt schema is invalid")
    if receipt.get("status") != "complete":
        raise EvaluationError("threshold receipt is not complete")
    if receipt.get("selection_source") != "V_sel_only":
        raise EvaluationError("threshold receipt was not selected from V_sel")
    if receipt.get("technical_working_point_only") is not True:
        raise EvaluationError("threshold receipt has an invalid interpretation boundary")
    if receipt.get("scope") != expected_v_sel_binding.scope.manifest():
        raise EvaluationError("threshold receipt scope does not match")
    if receipt.get("v_sel_binding") != expected_v_sel_binding.manifest():
        raise EvaluationError("threshold receipt V_sel binding does not match")
    if receipt.get("rule") != expected_rule.manifest():
        raise EvaluationError("threshold receipt rule does not match")
    if receipt.get("rule_sha256") != expected_rule.sha256:
        raise EvaluationError("threshold receipt rule hash does not match")
    if receipt.get("selection_branch") not in {"target_reached", "target_unreachable_fallback"}:
        raise EvaluationError("threshold receipt selection branch is invalid")
    actual_digest = threshold_receipt_sha256(receipt)
    if receipt.get("receipt_sha256") != actual_digest:
        raise EvaluationError("threshold receipt self-hash does not match")
    if actual_digest != expected_digest:
        raise EvaluationError("threshold receipt is not the committed receipt")
    v_sel_rows = receipt.get("v_sel_rows")
    if not isinstance(v_sel_rows, Mapping) or set(v_sel_rows) != {
        "row_count",
        "row_ids_sha256",
        "labels_sha256",
        "probabilities_sha256",
        "prediction_payload_sha256",
    }:
        raise EvaluationError("threshold receipt row binding is invalid")
    _require_exact_int(v_sel_rows.get("row_count"), "V_sel row_count", minimum=1)
    for field in (
        "row_ids_sha256",
        "labels_sha256",
        "probabilities_sha256",
        "prediction_payload_sha256",
    ):
        _require_sha256(v_sel_rows.get(field), f"V_sel {field}")
    threshold_hex = receipt.get("selected_threshold_hex")
    if not isinstance(threshold_hex, str):
        raise EvaluationError("selected threshold must use canonical float hex")
    try:
        threshold = float.fromhex(threshold_hex)
    except ValueError as exc:
        raise EvaluationError("selected threshold hex is invalid") from exc
    if (
        not math.isfinite(threshold)
        or threshold not in expected_rule.candidate_thresholds
        or threshold.hex() != threshold_hex
    ):
        raise EvaluationError("selected threshold is outside the frozen rule")
    rebuilt = select_threshold_from_v_sel(
        v_sel_binding=expected_v_sel_binding,
        row_ids=row_ids,
        labels=labels,
        probabilities=probabilities,
        rule=expected_rule,
    )
    if dict(receipt) != rebuilt:
        raise EvaluationError("threshold receipt differs from raw V_sel evidence reconstruction")
    verification_seal = _verified_threshold_seal(
        threshold=threshold,
        receipt_sha256=actual_digest,
        scope=expected_v_sel_binding.scope,
        v_sel_binding=expected_v_sel_binding,
        rule_sha256=expected_rule.sha256,
    )
    return VerifiedThresholdReceipt(
        threshold=threshold,
        receipt_sha256=actual_digest,
        scope=expected_v_sel_binding.scope,
        v_sel_binding=expected_v_sel_binding,
        rule_sha256=expected_rule.sha256,
        _verification_token=_VERIFIED_THRESHOLD_TOKEN,
        _verification_seal=verification_seal,
    )


ZERO_DENOMINATOR_POLICY: dict[str, str] = {
    "sensitivity": "null_if_no_actual_positives",
    "specificity": "null_if_no_actual_negatives",
    "ppv": "null_if_no_predicted_positives",
    "npv": "null_if_no_predicted_negatives",
    "f1": "null_if_2tp_plus_fp_plus_fn_is_zero",
    "balanced_accuracy": "null_if_sensitivity_or_specificity_is_null",
    "mcc": "null_if_any_mcc_denominator_factor_is_zero",
}


def evaluate_outer_test(
    *,
    outer_test_binding: OuterTestBinding,
    row_ids: object,
    labels: object,
    probabilities: object,
    verified_threshold_receipt: VerifiedThresholdReceipt,
) -> dict[str, object]:
    """Evaluate outer-test predictions using only a committed V_sel threshold."""
    if not isinstance(outer_test_binding, OuterTestBinding):
        raise EvaluationError("outer_test_binding must be an OuterTestBinding")
    if not isinstance(verified_threshold_receipt, VerifiedThresholdReceipt):
        raise EvaluationError(
            "outer-test evaluation requires a raw-evidence-verified threshold receipt"
        )
    verified_threshold_receipt._assert_valid()
    if outer_test_binding.scope != verified_threshold_receipt.scope:
        raise EvaluationError("outer-test and V_sel scopes must match exactly")
    threshold = float(verified_threshold_receipt.threshold)
    rows = _validated_probability_rows(row_ids, labels, probabilities)
    counts = _confusion_counts(rows, threshold)
    threshold_metrics = _threshold_metrics(counts)
    report: dict[str, object] = {
        "schema": _identity("outer_test_evaluation"),
        "status": "complete",
        "outer_test_binding": outer_test_binding.manifest(),
        "outer_test_rows": rows.bindings(),
        "threshold_source": {
            "selection_source": "V_sel_only",
            "threshold_receipt_sha256": verified_threshold_receipt.receipt_sha256,
            "selected_threshold_hex": threshold.hex(),
            "rule_sha256": verified_threshold_receipt.rule_sha256,
            "technical_working_point_only": True,
            "clinical_utility_claim": False,
        },
        "event_rate": {
            "positive_count": rows.positive_count,
            "negative_count": rows.negative_count,
            "prevalence": rows.prevalence,
        },
        "probability_metrics": _probability_metrics(rows),
        "calibration_metrics": compute_calibration_metrics(
            list(rows.labels), list(rows.probabilities)
        ),
        "confusion_counts": counts,
        "threshold_metrics": threshold_metrics,
        "threshold_metric_denominators": _threshold_denominators(counts),
        "threshold_metric_status": _threshold_metric_status(threshold_metrics),
        "zero_denominator_policy": copy.deepcopy(ZERO_DENOMINATOR_POLICY),
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


__all__ = [
    "EvaluationError",
    "EvaluationScope",
    "HPO_PROBABILITY_CONTRACT",
    "OuterTestBinding",
    "PROBABILITY_CLIP_EPSILON",
    "PROBABILITY_CLIP_POLICY",
    "TECHNICAL_TARGET_SENSITIVITY",
    "ThresholdSelectionRule",
    "VerifiedThresholdReceipt",
    "VSelBinding",
    "ZERO_DENOMINATOR_POLICY",
    "canonical_sha256",
    "compute_hpo_probability_metrics",
    "compute_prediction_metrics",
    "evaluate_outer_test",
    "select_threshold_from_v_sel",
    "threshold_receipt_sha256",
    "validate_threshold_receipt",
]
