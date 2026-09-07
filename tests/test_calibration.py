from __future__ import annotations
import copy
import math
import unittest
import numpy as np
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression
from fedsift.calibration import CalibrationError, compute_calibration_metrics


class CalibrationOracleTests(unittest.TestCase):

    def test_citl_and_slope_match_independent_oracles(self) -> None:
        probabilities = np.asarray(
            [0.08, 0.14, 0.22, 0.31, 0.43, 0.55, 0.63, 0.72, 0.81, 0.91], dtype=np.float64
        )
        labels = np.asarray([0, 0, 1, 0, 1, 0, 1, 1, 0, 1], dtype=np.int64)
        report = compute_calibration_metrics(labels, probabilities)
        logits = np.log(probabilities / (1.0 - probabilities))
        target = float(np.mean(labels))
        citl_oracle = brentq(
            lambda intercept: float(np.mean(1.0 / (1.0 + np.exp(-(logits + intercept)))) - target),
            -80.0,
            80.0,
        )
        self.assertAlmostEqual(
            report["calibration_in_the_large"]["estimate"], citl_oracle, places=11
        )
        fitted = LogisticRegression(
            penalty=None, solver="lbfgs", fit_intercept=True, tol=1e-12, max_iter=10000
        ).fit(logits.reshape(-1, 1), labels)
        self.assertAlmostEqual(
            report["calibration_slope"]["estimate"], float(fitted.coef_[0, 0]), places=6
        )
        self.assertAlmostEqual(
            report["calibration_slope"]["fitted_intercept"], float(fitted.intercept_[0]), places=6
        )
        citl_fitted = 1.0 / (1.0 + np.exp(-(logits + citl_oracle)))
        citl_se_oracle = float(1.0 / np.sqrt(np.sum(citl_fitted * (1.0 - citl_fitted))))
        self.assertAlmostEqual(
            report["calibration_in_the_large"]["standard_error"], citl_se_oracle, places=11
        )
        slope_intercept = float(fitted.intercept_[0])
        slope_estimate = float(fitted.coef_[0, 0])
        slope_fitted = 1.0 / (1.0 + np.exp(-(slope_intercept + slope_estimate * logits)))
        design = np.column_stack([np.ones_like(logits), logits])
        information = design.T @ ((slope_fitted * (1.0 - slope_fitted))[:, None] * design)
        slope_se_oracle = float(np.sqrt(np.linalg.inv(information)[1, 1]))
        self.assertAlmostEqual(
            report["calibration_slope"]["standard_error"], slope_se_oracle, places=6
        )
        normal_975 = 1.959963984540054
        self.assertAlmostEqual(
            report["calibration_slope"]["ci95_low"],
            slope_estimate - normal_975 * slope_se_oracle,
            places=6,
        )
        self.assertAlmostEqual(
            report["calibration_slope"]["ci95_high"],
            slope_estimate + normal_975 * slope_se_oracle,
            places=6,
        )
        for key in ("calibration_in_the_large", "calibration_slope"):
            self.assertEqual(report[key]["status"], "ok")
            self.assertLess(report[key]["ci95_low"], report[key]["estimate"])
            self.assertGreater(report[key]["ci95_high"], report[key]["estimate"])
        self.assertFalse(report["hosmer_lemeshow_reported"])
        self.assertIn("not_full_pipeline", report["uncertainty_boundary"])

    def test_constant_predictions_have_valid_citl_but_undefined_slope(self) -> None:
        report = compute_calibration_metrics([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(report["calibration_in_the_large"]["status"], "ok")
        self.assertAlmostEqual(report["calibration_in_the_large"]["estimate"], 0.0)
        self.assertEqual(report["calibration_slope"]["status"], "constant_predictions")
        self.assertIsNone(report["calibration_slope"]["estimate"])

    def test_perfect_or_quasi_separation_is_explicitly_undefined(self) -> None:
        report = compute_calibration_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertEqual(report["calibration_slope"]["status"], "complete_or_quasi_separation")
        self.assertIsNone(report["calibration_slope"]["estimate"])

    def test_deterministic_synthetic_sweep_matches_independent_fits(self) -> None:
        generator = np.random.default_rng(20260831)
        for case_index in range(12):
            logits = generator.normal(loc=0.0, scale=1.2, size=48)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            labels = np.asarray([0, 1] * 24, dtype=np.int64)
            generator.shuffle(labels)
            report = compute_calibration_metrics(labels, probabilities)
            self.assertEqual(
                report["calibration_in_the_large"]["status"], "ok", msg=f"case={case_index}"
            )
            self.assertEqual(report["calibration_slope"]["status"], "ok", msg=f"case={case_index}")
            target = float(np.mean(labels))
            citl_oracle = brentq(
                lambda intercept: float(
                    np.mean(1.0 / (1.0 + np.exp(-(logits + intercept)))) - target
                ),
                -80.0,
                80.0,
            )
            fitted = LogisticRegression(
                penalty=None, solver="lbfgs", fit_intercept=True, tol=1e-12, max_iter=10000
            ).fit(logits.reshape(-1, 1), labels)
            self.assertAlmostEqual(
                report["calibration_in_the_large"]["estimate"], citl_oracle, places=10
            )
            self.assertAlmostEqual(
                report["calibration_slope"]["estimate"], float(fitted.coef_[0, 0]), places=5
            )
            self.assertAlmostEqual(
                report["calibration_slope"]["fitted_intercept"],
                float(fitted.intercept_[0]),
                places=5,
            )

    def test_clipped_boundary_probabilities_and_single_class_statuses(self) -> None:
        boundary = compute_calibration_metrics([0, 1, 1, 0], [0.0, 1.0, 0.0, 1.0])
        self.assertEqual(boundary["probability_clip"], 1e-07)
        self.assertEqual(boundary["calibration_in_the_large"]["status"], "ok")
        self.assertTrue(math.isfinite(boundary["calibration_in_the_large"]["estimate"]))
        single_class = compute_calibration_metrics([1, 1, 1], [0.2, 0.5, 0.8])
        self.assertEqual(single_class["calibration_in_the_large"]["status"], "single_class")
        self.assertEqual(single_class["calibration_slope"]["status"], "single_class")
        self.assertIsNone(single_class["calibration_slope"]["estimate"])


class CalibrationFailClosedTests(unittest.TestCase):

    def test_invalid_types_classes_ranges_and_lengths_fail(self) -> None:
        cases = [
            ([False, 1], [0.2, 0.8]),
            ([0, 2], [0.2, 0.8]),
            ([0, 1], [False, 0.8]),
            ([0, 1], [float("nan"), 0.8]),
            ([0, 1], [-0.1, 0.8]),
            ([0, 1], [0.2]),
            ([], []),
        ]
        for labels, probabilities in cases:
            with self.subTest(labels=labels, probabilities=probabilities):
                with self.assertRaises(CalibrationError):
                    compute_calibration_metrics(labels, probabilities)

    def test_report_hash_changes_if_evidence_is_tampered(self) -> None:
        report = compute_calibration_metrics([0, 1, 0, 1], [0.2, 0.8, 0.4, 0.6])
        other = compute_calibration_metrics([0, 1, 0, 1], [0.3, 0.7, 0.4, 0.6])
        self.assertNotEqual(
            report["bindings"]["probabilities_sha256"], other["bindings"]["probabilities_sha256"]
        )
        self.assertNotEqual(
            report["bindings"]["label_probability_payload_sha256"],
            other["bindings"]["label_probability_payload_sha256"],
        )
        changed = copy.deepcopy(report)
        changed["calibration_in_the_large"]["estimate"] = 0.25
        self.assertNotEqual(changed, report)
        self.assertEqual(len(report["report_sha256"]), 64)

    def test_signed_zero_is_canonicalized_in_probability_binding(self) -> None:
        negative = compute_calibration_metrics([0, 1], [-0.0, 0.8])
        positive = compute_calibration_metrics([0, 1], [0.0, 0.8])
        self.assertEqual(negative, positive)


if __name__ == "__main__":
    unittest.main()
