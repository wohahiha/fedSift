"""Numerical comparison of models, training evidence and accounting values."""

import math


def difference_rows(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [
            r
            for k in sorted(set(a) | set(b))
            for r in difference_rows(a.get(k), b.get(k), path + "/" + k)
        ]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [
            r
            for i, (x, y) in enumerate(zip(a, b))
            for r in difference_rows(x, y, path + "/" + str(i))
        ]
    if a == b:
        return []
    numeric = isinstance(a, (float, int)) and isinstance(b, (float, int))
    x, y = a, b
    if path.endswith("_hex") and isinstance(a, str) and isinstance(b, str):
        try:
            x, y = float.fromhex(a), float.fromhex(b)
            numeric = True
        except ValueError:
            pass
    return [
        {
            "path": path,
            "expected": a,
            "actual": b,
            "hash_field": any(part.endswith("_sha256") for part in path.split("/")),
            "numeric_close": numeric and math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12),
            "absolute_difference": abs(x - y) if numeric else None,
        }
    ]


def assess_training(expected, captured):
    left, right = expected["preopen_chain"], captured["chain"]
    differences = difference_rows(left, right)
    differences += difference_rows(
        expected["training_evidence"], captured["training"], "/training_evidence"
    )
    checks = {
        "capability_identical": left["capability_sha256"] == right["capability_sha256"],
        "model_state_identical": left["model_artifact"]["serialized_model_state"]
        == right["model_artifact"]["serialized_model_state"],
        "model_state_sha256_identical": left["model_artifact"]["final_model_state_sha256"]
        == right["model_artifact"]["final_model_state_sha256"],
        "preprocessing_identical": left["model_artifact"]["refit_preprocessing_artifact"]
        == right["model_artifact"]["refit_preprocessing_artifact"],
        "validation_prediction_payload_identical": all(
            left["prediction_artifact"][k] == right["prediction_artifact"][k]
            for k in ["row_ids", "labels", "probabilities", "prediction_payload_sha256"]
        ),
        "threshold_rule_identical": left["threshold_artifact"]["threshold_rule"]
        == right["threshold_artifact"]["threshold_rule"],
        "selected_threshold_identical": left["threshold_artifact"]["threshold_receipt"][
            "selected_threshold_hex"
        ]
        == right["threshold_artifact"]["threshold_receipt"]["selected_threshold_hex"],
        "resource_counts_identical": expected["training_evidence"]["resource_summary"]
        == captured["training"]["resource_summary"],
    }
    passed = all(checks.values()) and all(
        r["hash_field"] or r["numeric_close"] for r in differences
    )
    return {
        "numerically_equivalent": passed,
        "exact_checks": checks,
        "preopen_chain_identical": not differences,
        "maximum_numeric_difference": max(
            (r["absolute_difference"] or 0.0 for r in differences), default=0.0
        ),
        "differences": differences,
    }
