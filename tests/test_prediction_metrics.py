"""Regression checks for the shared clean/attack evaluation contract."""
import math
import unittest

from fedsift.evaluation import compute_prediction_metrics, EvaluationError


class PredictionMetricsTests(unittest.TestCase):
    def test_extreme_probabilities_clip_only_log_loss(self):
        result = compute_prediction_metrics([0, 1], [0, 1], [1.0, 0.0], threshold=1.0)
        expected_loss = (-math.log1p(-(1.0 - 1e-7)) - math.log(1e-7)) / 2
        self.assertAlmostEqual(result["log_loss"], expected_loss, places=13)
        self.assertEqual(result["brier_score"], 1.0)
        self.assertEqual(result["auroc"], 0.0)
        self.assertEqual(result["sensitivity"], 0.0)
        self.assertEqual(result["specificity"], 0.0)

    def test_tiny_scores_preserve_ranking_and_threshold_decisions(self):
        result = compute_prediction_metrics([0, 1], [0, 1], [0.0, 1e-15], threshold=1e-16)
        self.assertEqual(result["auroc"], 1.0)
        self.assertEqual(result["average_precision"], 1.0)
        self.assertEqual(result["sensitivity"], 1.0)
        self.assertEqual(result["specificity"], 1.0)

    def test_invalid_probability_or_threshold_is_rejected(self):
        for probability, threshold in ((float("nan"), 0.5), (1.1, 0.5), (0.5, float("nan"))):
            with self.subTest(probability=probability, threshold=threshold):
                with self.assertRaises(EvaluationError):
                    compute_prediction_metrics([0, 1], [0, 1], [0.2, probability], threshold=threshold)


if __name__ == "__main__":
    unittest.main()
