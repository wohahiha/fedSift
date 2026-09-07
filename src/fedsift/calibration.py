"""Calibration estimates for result-blind binary prediction evaluation.

The module reports calibration-in-the-large (offset intercept) and the
calibration slope with explicit failure statuses.  Wald intervals quantify
conditional estimation precision; they do not account for model-development,
cross-validation, or hyperparameter-selection uncertainty.
"""

from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import hashlib
import json
import math
from numbers import Integral, Real
from typing import Sequence
import numpy as np
from .probability_contract import PROBABILITY_CLIP_EPSILON


class CalibrationError(ValueError):
    """Raised when calibration inputs are malformed."""


_CLIP = PROBABILITY_CLIP_EPSILON
_NORMAL_975 = 1.959963984540054
_CITL_ROOT_BOUND = 80.0
_CITL_BISECTION_ITERATIONS = 180
_SLOPE_SCALE_MINIMUM = 1e-12
_IRLS_WEIGHT_MINIMUM = 1e-12
_IRLS_MAX_ITERATIONS = 200
_IRLS_CONDITION_MAXIMUM = 100000000000000.0
_IRLS_STEP_TOLERANCE = 1e-09


def _canonical_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise CalibrationError("calibration value is not canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def _validated_arrays(labels: object, probabilities: object) -> tuple[np.ndarray, np.ndarray]:
    if (
        isinstance(labels, np.ndarray)
        and labels.ndim == 1
        and isinstance(probabilities, np.ndarray)
        and (probabilities.ndim == 1)
    ):
        raw_labels = labels.tolist()
        raw_probabilities = probabilities.tolist()
    else:
        if (
            isinstance(labels, (str, bytes))
            or not isinstance(labels, Sequence)
            or isinstance(probabilities, (str, bytes))
            or (not isinstance(probabilities, Sequence))
        ):
            raise CalibrationError("labels and probabilities must be one-dimensional")
        raw_labels = list(labels)
        raw_probabilities = list(probabilities)
    if not raw_labels or len(raw_labels) != len(raw_probabilities):
        raise CalibrationError("labels and probabilities must be non-empty and aligned")
    canonical_labels: list[int] = []
    canonical_probabilities: list[float] = []
    for index, value in enumerate(raw_labels):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise CalibrationError(f"labels[{index}] must be an exact integer")
        label = int(value)
        if label not in (0, 1):
            raise CalibrationError("labels must be binary")
        canonical_labels.append(label)
    for index, value in enumerate(raw_probabilities):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise CalibrationError(f"probabilities[{index}] must be real")
        probability = float(value)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise CalibrationError("probabilities must be finite and in [0, 1]")
        if probability == 0.0:
            probability = 0.0
        canonical_probabilities.append(probability)
    return (
        np.asarray(canonical_labels, dtype=np.float64),
        np.asarray(canonical_probabilities, dtype=np.float64),
    )


def _expit(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def _interval(estimate: float, standard_error: float) -> dict[str, float]:
    return {
        "estimate": float(estimate),
        "standard_error": float(standard_error),
        "ci95_low": float(estimate - _NORMAL_975 * standard_error),
        "ci95_high": float(estimate + _NORMAL_975 * standard_error),
    }


def _calibration_in_the_large(
    labels: np.ndarray, linear_predictor: np.ndarray
) -> tuple[dict[str, object], float | None]:
    if len(np.unique(labels)) < 2:
        return ({"status": "single_class", "estimate": None}, None)
    target = float(np.mean(labels))
    low = -_CITL_ROOT_BOUND
    high = _CITL_ROOT_BOUND
    low_value = float(np.mean(_expit(linear_predictor + low)) - target)
    high_value = float(np.mean(_expit(linear_predictor + high)) - target)
    if low_value > 0.0 or high_value < 0.0:
        return ({"status": "root_not_bracketed", "estimate": None}, None)
    for _ in range(_CITL_BISECTION_ITERATIONS):
        midpoint = 0.5 * (low + high)
        value = float(np.mean(_expit(linear_predictor + midpoint)) - target)
        if value > 0.0:
            high = midpoint
        else:
            low = midpoint
    estimate = 0.5 * (low + high)
    fitted = _expit(linear_predictor + estimate)
    information = float(np.sum(fitted * (1.0 - fitted)))
    if not math.isfinite(estimate) or not math.isfinite(information) or information <= 0.0:
        return ({"status": "nonfinite_or_singular", "estimate": None}, None)
    standard_error = 1.0 / math.sqrt(information)
    return (
        {
            "status": "ok",
            **_interval(estimate, standard_error),
            "ideal_value": 0.0,
            "interval_kind": "wald_conditional_on_fixed_predictions",
        },
        estimate,
    )


def _calibration_slope(labels: np.ndarray, linear_predictor: np.ndarray) -> dict[str, object]:
    if len(np.unique(labels)) < 2:
        return {"status": "single_class", "estimate": None}
    scale = float(np.std(linear_predictor, ddof=0))
    if not math.isfinite(scale) or scale < _SLOPE_SCALE_MINIMUM:
        return {"status": "constant_predictions", "estimate": None}
    positive_scores = linear_predictor[labels == 1.0]
    negative_scores = linear_predictor[labels == 0.0]
    if float(np.min(positive_scores)) >= float(np.max(negative_scores)) or float(
        np.max(positive_scores)
    ) <= float(np.min(negative_scores)):
        return {"status": "complete_or_quasi_separation", "estimate": None}
    center = float(np.mean(linear_predictor))
    standardized = (linear_predictor - center) / scale
    design = np.column_stack([np.ones_like(standardized), standardized])
    prevalence = float(np.mean(labels))
    beta = np.array([math.log(prevalence / (1.0 - prevalence)), 0.0], dtype=np.float64)

    def log_likelihood(candidate: np.ndarray) -> float:
        eta = design @ candidate
        return float(np.sum(labels * eta - np.logaddexp(0.0, eta)))

    current_likelihood = log_likelihood(beta)
    converged = False
    information: np.ndarray | None = None
    for _ in range(_IRLS_MAX_ITERATIONS):
        eta = design @ beta
        fitted = _expit(eta)
        weights = np.maximum(fitted * (1.0 - fitted), _IRLS_WEIGHT_MINIMUM)
        score = design.T @ (labels - fitted)
        information = design.T @ (weights[:, None] * design)
        condition = float(np.linalg.cond(information))
        if not math.isfinite(condition) or condition > _IRLS_CONDITION_MAXIMUM:
            return {"status": "singular_information", "estimate": None}
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError:
            return {"status": "singular_information", "estimate": None}
        if not np.all(np.isfinite(step)):
            return {"status": "nonfinite_step", "estimate": None}
        multiplier = 1.0
        accepted = False
        for _ in range(40):
            proposal = beta + multiplier * step
            proposal_likelihood = log_likelihood(proposal)
            if (
                math.isfinite(proposal_likelihood)
                and proposal_likelihood >= current_likelihood - 1e-12
            ):
                beta = proposal
                current_likelihood = proposal_likelihood
                accepted = True
                break
            multiplier *= 0.5
        if not accepted:
            return {"status": "line_search_failed", "estimate": None}
        if float(np.max(np.abs(multiplier * step))) < _IRLS_STEP_TOLERANCE:
            converged = True
            break
        if float(np.max(np.abs(beta))) > 10000.0:
            return {"status": "divergent_estimate", "estimate": None}
    if not converged or information is None:
        return {"status": "nonconvergence", "estimate": None}
    fitted = _expit(design @ beta)
    weights = np.maximum(fitted * (1.0 - fitted), _IRLS_WEIGHT_MINIMUM)
    information = design.T @ (weights[:, None] * design)
    try:
        covariance = np.linalg.inv(information)
    except np.linalg.LinAlgError:
        return {"status": "singular_information", "estimate": None}
    slope = float(beta[1] / scale)
    slope_standard_error = float(math.sqrt(max(0.0, covariance[1, 1])) / scale)
    original_intercept = float(beta[0] - beta[1] * center / scale)
    if not all(
        (math.isfinite(value) for value in (slope, slope_standard_error, original_intercept))
    ):
        return {"status": "nonfinite", "estimate": None}
    if abs(slope) > 50.0:
        return {"status": "extreme_unstable_estimate", "estimate": None}
    return {
        "status": "ok",
        **_interval(slope, slope_standard_error),
        "fitted_intercept": original_intercept,
        "ideal_value": 1.0,
        "interval_kind": "wald_conditional_on_fixed_predictions",
    }


def compute_calibration_metrics(labels: object, probabilities: object) -> dict[str, object]:
    """Return CITL and slope with explicit conditional-uncertainty semantics."""
    canonical_labels, canonical_probabilities = _validated_arrays(labels, probabilities)
    clipped = np.clip(canonical_probabilities, _CLIP, 1.0 - _CLIP)
    linear_predictor = np.log(clipped / (1.0 - clipped))
    citl, _ = _calibration_in_the_large(canonical_labels, linear_predictor)
    slope = _calibration_slope(canonical_labels, linear_predictor)
    report: dict[str, object] = {
        "schema": _identity("calibration_metrics"),
        "status": "complete",
        "row_count": int(len(canonical_labels)),
        "positive_count": int(np.sum(canonical_labels)),
        "bindings": {
            "labels_sha256": _canonical_sha256([int(value) for value in canonical_labels]),
            "probabilities_sha256": _canonical_sha256(
                [float(value).hex() for value in canonical_probabilities]
            ),
            "label_probability_payload_sha256": _canonical_sha256(
                [
                    {"label": int(label), "probability_hex": float(probability).hex()}
                    for (label, probability) in zip(canonical_labels, canonical_probabilities)
                ]
            ),
        },
        "probability_clip": _CLIP,
        "estimation_contract": {
            "citl_definition": "offset_logit_intercept_with_slope_fixed_to_one",
            "citl_root_bound": _CITL_ROOT_BOUND,
            "citl_bisection_iterations": _CITL_BISECTION_ITERATIONS,
            "slope_definition": "unpenalized_logistic_intercept_plus_logit_probability",
            "slope_scale_minimum": _SLOPE_SCALE_MINIMUM,
            "irls_weight_minimum": _IRLS_WEIGHT_MINIMUM,
            "irls_max_iterations": _IRLS_MAX_ITERATIONS,
            "irls_condition_maximum": _IRLS_CONDITION_MAXIMUM,
            "irls_step_tolerance": _IRLS_STEP_TOLERANCE,
            "ci95_normal_quantile": _NORMAL_975,
            "separation_policy": "return_explicit_undefined_status",
        },
        "calibration_in_the_large": citl,
        "calibration_slope": slope,
        "hosmer_lemeshow_reported": False,
        "uncertainty_boundary": "conditional_on_fixed_predictions_not_full_pipeline_or_cross_validation_uncertainty",
    }
    report["report_sha256"] = _canonical_sha256(report)
    return report


__all__ = ["CalibrationError", "compute_calibration_metrics"]
