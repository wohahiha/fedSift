from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import copy
import inspect
import math
import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
import fedsift.evaluation as evaluation
from fedsift.evaluation import (
    EvaluationError,
    PROBABILITY_CLIP_EPSILON,
    EvaluationScope,
    OuterTestBinding,
    ThresholdSelectionRule,
    VSelBinding,
    compute_hpo_probability_metrics,
    evaluate_outer_test,
    select_threshold_from_v_sel,
    threshold_receipt_sha256,
    validate_threshold_receipt,
)


def _digest(label: str) -> str:
    return evaluation.canonical_sha256({"fixture": label})


def _fixed_contract_log_loss(labels, probabilities) -> float:
    terms = []
    for label, probability in zip(labels, probabilities):
        clipped = min(
            max(float(probability), PROBABILITY_CLIP_EPSILON), 1.0 - PROBABILITY_CLIP_EPSILON
        )
        terms.append(-math.log(clipped) if int(label) == 1 else -math.log1p(-clipped))
    return math.fsum(terms) / len(terms)


def _scope(*, fold: int = 2, method: str = "fedsift") -> EvaluationScope:
    return EvaluationScope(
        study_id=_identity("evaluation_test"),
        outer_repeat=1,
        outer_fold=fold,
        method_id=method,
        candidate_id="candidate_0001",
    )


def _v_sel_binding(*, fold: int = 2, method: str = "fedsift") -> VSelBinding:
    return VSelBinding(
        scope=_scope(fold=fold, method=method),
        membership_sha256=_digest(f"v-sel-membership-{fold}-{method}"),
        prediction_artifact_sha256=_digest(f"v-sel-predictions-{fold}-{method}"),
    )


def _outer_binding() -> OuterTestBinding:
    return OuterTestBinding(
        scope=_scope(),
        membership_sha256=_digest("outer-test-membership"),
        prediction_artifact_sha256=_digest("outer-test-predictions"),
    )


def _rule(thresholds: tuple[float, ...] = (0.3, 0.5, 0.55, 0.7)) -> ThresholdSelectionRule:
    return ThresholdSelectionRule(target_sensitivity=0.85, candidate_thresholds=thresholds)


def _v_sel_fixture() -> tuple[list[int], list[int], list[float]]:
    row_ids = list(range(100, 125))
    labels = [1] * 20 + [0] * 5
    probabilities = [0.6] * 17 + [0.4] * 3 + [0.2, 0.45, 0.55, 0.65, 0.75]
    return (row_ids, labels, probabilities)


def _receipt(
    *, rule: ThresholdSelectionRule | None = None, binding: VSelBinding | None = None
) -> tuple[dict[str, object], ThresholdSelectionRule, VSelBinding]:
    selected_rule = _rule() if rule is None else rule
    selected_binding = _v_sel_binding() if binding is None else binding
    row_ids, labels, probabilities = _v_sel_fixture()
    receipt = select_threshold_from_v_sel(
        v_sel_binding=selected_binding,
        row_ids=row_ids,
        labels=labels,
        probabilities=probabilities,
        rule=selected_rule,
    )
    return (receipt, selected_rule, selected_binding)


def _verify_fixture_receipt(
    receipt: dict[str, object], rule: ThresholdSelectionRule, binding: VSelBinding
):
    row_ids, labels, probabilities = _v_sel_fixture()
    return validate_threshold_receipt(
        receipt,
        expected_receipt_sha256=receipt["receipt_sha256"],
        expected_v_sel_binding=binding,
        expected_rule=rule,
        row_ids=row_ids,
        labels=labels,
        probabilities=probabilities,
    )


class ProbabilityMetricTests(unittest.TestCase):

    def test_hard_coded_oracle_and_prevalence_binding(self) -> None:
        row_ids = [11, 12, 13, 14]
        labels = [1, 0, 1, 0]
        probabilities = [0.9, 0.8, 0.7, 0.6]
        report = compute_hpo_probability_metrics(row_ids, labels, probabilities)
        metrics = report["metrics"]
        self.assertAlmostEqual(metrics["average_precision"], 5.0 / 6.0, places=15)
        self.assertAlmostEqual(metrics["auroc"], 3.0 / 4.0, places=15)
        self.assertAlmostEqual(metrics["brier_score"], 0.275, places=15)
        expected_log_loss = (
            -math.fsum([math.log(0.9), math.log(0.2), math.log(0.7), math.log(0.4)]) / 4.0
        )
        self.assertAlmostEqual(metrics["log_loss"], expected_log_loss, places=15)
        self.assertEqual(
            report["event_rate"], {"positive_count": 2, "negative_count": 2, "prevalence": 0.5}
        )
        self.assertEqual(report["contract"]["average_precision_companion"], "prevalence")
        self.assertFalse(report["threshold_metrics_present"])
        self.assertNotIn("threshold", report["metrics"])

    def test_independent_formulas_match_sklearn_with_ties_and_boundaries(self) -> None:
        cases = [
            ([0, 1, 1, 0], [0.1, 0.8, 0.4, 0.3]),
            ([1, 0, 1, 0], [0.7, 0.7, 0.4, 0.4]),
            ([0, 1, 0, 1, 1], [0.0, 1.0, 0.5, 0.5, 0.25]),
        ]
        for case_index, (labels, probabilities) in enumerate(cases):
            with self.subTest(case=case_index):
                report = compute_hpo_probability_metrics(
                    list(range(1000, 1000 + len(labels))), labels, probabilities
                )
                metrics = report["metrics"]
                expected = {
                    "average_precision": average_precision_score(labels, probabilities),
                    "auroc": roc_auc_score(labels, probabilities),
                    "brier_score": brier_score_loss(labels, probabilities),
                    "log_loss": _fixed_contract_log_loss(labels, probabilities),
                }
                for name, expected_value in expected.items():
                    self.assertAlmostEqual(
                        metrics[name], float(expected_value), places=14, msg=name
                    )

    def test_deterministic_synthetic_sweep_matches_sklearn_and_tie_permutations(self) -> None:
        generator = np.random.default_rng(20260831)
        for case_index in range(24):
            labels = generator.integers(0, 2, size=37, dtype=np.int64)
            labels[0] = 0
            labels[1] = 1
            probabilities = np.round(generator.random(37), decimals=2)
            probabilities[0] = 0.0
            probabilities[1] = 1.0
            permutation = generator.permutation(37)
            for order in (np.arange(37), permutation):
                ordered_labels = labels[order]
                ordered_probabilities = probabilities[order]
                report = compute_hpo_probability_metrics(
                    list(range(10000, 10037)), ordered_labels, ordered_probabilities
                )
                expected = {
                    "average_precision": average_precision_score(
                        ordered_labels, ordered_probabilities
                    ),
                    "auroc": roc_auc_score(ordered_labels, ordered_probabilities),
                    "brier_score": brier_score_loss(ordered_labels, ordered_probabilities),
                    "log_loss": _fixed_contract_log_loss(ordered_labels, ordered_probabilities),
                }
                for name, expected_value in expected.items():
                    self.assertAlmostEqual(
                        report["metrics"][name],
                        float(expected_value),
                        places=13,
                        msg=f"case={case_index}, metric={name}",
                    )

    def test_hpo_contract_matches_plan_and_contains_no_threshold_objective(self) -> None:
        report = compute_hpo_probability_metrics([1, 2, 3, 4], [0, 1, 0, 1], [0.2, 0.7, 0.4, 0.8])
        contract = report["contract"]
        self.assertEqual(contract["outer_test_access"], "forbidden")
        self.assertEqual(contract["threshold_selection"], "forbidden")
        self.assertEqual(contract["primary_metric"], "log_loss")
        self.assertEqual(contract["log_loss_probability_clip_epsilon"], PROBABILITY_CLIP_EPSILON)
        self.assertEqual(
            contract["log_loss_probability_clip_policy"],
            "fixed_symmetric_1e_minus_7_all_methods_and_roles",
        )
        self.assertEqual(
            contract["hpo_tie_break_alignment"],
            [
                {"criterion": "log_loss", "direction": "minimize"},
                {"criterion": "average_precision", "direction": "maximize"},
                {"criterion": "auroc", "direction": "maximize"},
                {"criterion": "brier_score", "direction": "minimize"},
                {"criterion": "communication_bytes", "direction": "minimize_external"},
                {"criterion": "candidate_id", "direction": "lexicographic_min_external"},
            ],
        )
        self.assertEqual(
            contract["development_context"],
            _identity("selection_development_context"),
        )
        self.assertEqual(
            set(report["metrics"]), {"average_precision", "auroc", "brier_score", "log_loss"}
        )

    def test_signed_zero_has_one_canonical_probability_and_threshold_encoding(self) -> None:
        negative_zero = compute_hpo_probability_metrics([1, 2], [0, 1], [-0.0, 0.8])
        positive_zero = compute_hpo_probability_metrics([1, 2], [0, 1], [0.0, 0.8])
        self.assertEqual(negative_zero, positive_zero)
        self.assertEqual(
            ThresholdSelectionRule(
                target_sensitivity=0.85, candidate_thresholds=(-0.0, 0.5)
            ).manifest(),
            ThresholdSelectionRule(
                target_sensitivity=0.85, candidate_thresholds=(0.0, 0.5)
            ).manifest(),
        )


class ThresholdReceiptTests(unittest.TestCase):

    def test_target_rule_uses_exact_17_of_20_boundary_and_higher_threshold_tie(self) -> None:
        receipt, rule, binding = _receipt()
        self.assertEqual(receipt["selection_source"], "V_sel_only")
        self.assertEqual(receipt["selection_branch"], "target_reached")
        self.assertEqual(float.fromhex(receipt["selected_threshold_hex"]), 0.55)
        self.assertEqual(receipt["rule"]["target_sensitivity"], 0.85)
        self.assertEqual(
            receipt["rule"]["feasible_integer_test"], "20_times_tp_ge_17_times_actual_positives"
        )
        self.assertEqual(receipt["rule_sha256"], rule.sha256)
        self.assertEqual(receipt["v_sel_binding"], binding.manifest())
        self.assertFalse(receipt["v_sel_binding"]["prediction_artifact_content_verified_here"])
        verified = _verify_fixture_receipt(receipt, rule, binding)
        self.assertEqual(verified.threshold, 0.55)
        self.assertEqual(verified.receipt_sha256, receipt["receipt_sha256"])

    def test_infeasible_candidates_use_frozen_fallback_order(self) -> None:
        rule = _rule((0.7, 0.8))
        row_ids = list(range(1, 26))
        labels = [1] * 20 + [0] * 5
        probabilities = [0.75] * 6 + [0.85] * 10 + [0.65] * 4 + [0.1] * 5
        receipt = select_threshold_from_v_sel(
            v_sel_binding=_v_sel_binding(),
            row_ids=row_ids,
            labels=labels,
            probabilities=probabilities,
            rule=rule,
        )
        self.assertEqual(receipt["selection_branch"], "target_unreachable_fallback")
        self.assertEqual(float.fromhex(receipt["selected_threshold_hex"]), 0.7)

    def test_probability_equal_to_threshold_is_predicted_positive(self) -> None:
        rule = _rule((0.5,))
        row_ids = list(range(1, 22))
        labels = [1] * 20 + [0]
        probabilities = [0.5] * 17 + [0.49] * 3 + [0.1]
        receipt = select_threshold_from_v_sel(
            v_sel_binding=_v_sel_binding(),
            row_ids=row_ids,
            labels=labels,
            probabilities=probabilities,
            rule=rule,
        )
        self.assertEqual(receipt["selection_branch"], "target_reached")

    def test_target_must_be_explicitly_085_and_candidates_frozen(self) -> None:
        with self.assertRaises(EvaluationError):
            ThresholdSelectionRule(target_sensitivity=0.8, candidate_thresholds=(0.3, 0.5))
        with self.assertRaises(EvaluationError):
            ThresholdSelectionRule(target_sensitivity=0.85, candidate_thresholds=(0.5, 0.3))
        with self.assertRaises(EvaluationError):
            ThresholdSelectionRule(target_sensitivity=0.85, candidate_thresholds=(0.5, 0.5))

    def test_receipt_tampering_fails_with_and_without_recomputed_self_hash(self) -> None:
        receipt, rule, binding = _receipt()
        expected_digest = receipt["receipt_sha256"]
        tampered = copy.deepcopy(receipt)
        tampered["selected_threshold_hex"] = (0.5).hex()
        with self.assertRaisesRegex(EvaluationError, "self-hash"):
            validate_threshold_receipt(
                tampered,
                expected_receipt_sha256=expected_digest,
                expected_v_sel_binding=binding,
                expected_rule=rule,
                row_ids=_v_sel_fixture()[0],
                labels=_v_sel_fixture()[1],
                probabilities=_v_sel_fixture()[2],
            )
        tampered["receipt_sha256"] = threshold_receipt_sha256(tampered)
        with self.assertRaisesRegex(EvaluationError, "committed"):
            validate_threshold_receipt(
                tampered,
                expected_receipt_sha256=expected_digest,
                expected_v_sel_binding=binding,
                expected_rule=rule,
                row_ids=_v_sel_fixture()[0],
                labels=_v_sel_fixture()[1],
                probabilities=_v_sel_fixture()[2],
            )

    def test_even_an_externally_committed_forgery_fails_raw_v_sel_rebuild(self) -> None:
        receipt, rule, binding = _receipt()
        forged = copy.deepcopy(receipt)
        forged["selected_threshold_hex"] = (0.5).hex()
        forged["receipt_sha256"] = threshold_receipt_sha256(forged)
        row_ids, labels, probabilities = _v_sel_fixture()
        with self.assertRaisesRegex(EvaluationError, "raw V_sel evidence"):
            validate_threshold_receipt(
                forged,
                expected_receipt_sha256=forged["receipt_sha256"],
                expected_v_sel_binding=binding,
                expected_rule=rule,
                row_ids=row_ids,
                labels=labels,
                probabilities=probabilities,
            )

    def test_verified_threshold_capability_cannot_be_replaced_or_constructed(self) -> None:
        receipt, rule, binding = _receipt()
        verified = _verify_fixture_receipt(receipt, rule, binding)
        with self.assertRaisesRegex(EvaluationError, "capability seal"):
            replace(verified, threshold=0.5)
        with self.assertRaisesRegex(EvaluationError, "only be created"):
            evaluation.VerifiedThresholdReceipt(
                threshold=0.55,
                receipt_sha256=receipt["receipt_sha256"],
                scope=binding.scope,
                v_sel_binding=binding,
                rule_sha256=rule.sha256,
                _verification_token=object(),
                _verification_seal="0" * 64,
            )

    def test_cross_fold_or_method_receipt_replay_is_rejected(self) -> None:
        receipt, rule, _ = _receipt()
        for other_binding in (_v_sel_binding(fold=3), _v_sel_binding(method="time_dpfedadam")):
            with self.assertRaises(EvaluationError):
                validate_threshold_receipt(
                    receipt,
                    expected_receipt_sha256=receipt["receipt_sha256"],
                    expected_v_sel_binding=other_binding,
                    expected_rule=rule,
                    row_ids=_v_sel_fixture()[0],
                    labels=_v_sel_fixture()[1],
                    probabilities=_v_sel_fixture()[2],
                )


class OuterTestEvaluationTests(unittest.TestCase):

    def test_hard_coded_confusion_oracle_and_probability_metrics(self) -> None:
        rule = _rule((0.5,))
        v_binding = _v_sel_binding()
        v_rows = [1, 2, 3, 4]
        receipt = select_threshold_from_v_sel(
            v_sel_binding=v_binding,
            row_ids=v_rows,
            labels=[1, 1, 0, 0],
            probabilities=[0.9, 0.8, 0.4, 0.2],
            rule=rule,
        )
        verified = validate_threshold_receipt(
            receipt,
            expected_receipt_sha256=receipt["receipt_sha256"],
            expected_v_sel_binding=v_binding,
            expected_rule=rule,
            row_ids=v_rows,
            labels=[1, 1, 0, 0],
            probabilities=[0.9, 0.8, 0.4, 0.2],
        )
        labels = [1, 1, 1, 0, 0, 0]
        probabilities = [0.9, 0.6, 0.4, 0.8, 0.3, 0.2]
        report = evaluate_outer_test(
            outer_test_binding=_outer_binding(),
            row_ids=[101, 102, 103, 104, 105, 106],
            labels=labels,
            probabilities=probabilities,
            verified_threshold_receipt=verified,
        )
        self.assertEqual(
            report["confusion_counts"],
            {"true_positive": 2, "false_positive": 1, "true_negative": 2, "false_negative": 1},
        )
        expected_threshold_metrics = {
            "sensitivity": 2.0 / 3.0,
            "specificity": 2.0 / 3.0,
            "ppv": 2.0 / 3.0,
            "npv": 2.0 / 3.0,
            "f1": 2.0 / 3.0,
            "balanced_accuracy": 2.0 / 3.0,
            "mcc": 1.0 / 3.0,
        }
        for name, expected in expected_threshold_metrics.items():
            self.assertAlmostEqual(report["threshold_metrics"][name], expected, places=15)
            self.assertEqual(report["threshold_metric_status"][name], "defined")
        expected_probability_metrics = {
            "average_precision": average_precision_score(labels, probabilities),
            "auroc": roc_auc_score(labels, probabilities),
            "brier_score": brier_score_loss(labels, probabilities),
            "log_loss": _fixed_contract_log_loss(labels, probabilities),
        }
        for name, expected in expected_probability_metrics.items():
            self.assertAlmostEqual(report["probability_metrics"][name], float(expected), places=14)
        self.assertEqual(report["event_rate"]["prevalence"], 0.5)
        self.assertTrue(report["threshold_source"]["technical_working_point_only"])
        self.assertFalse(report["threshold_source"]["clinical_utility_claim"])

    def test_zero_denominators_return_none_with_preregistered_status(self) -> None:
        rule = _rule((0.5,))
        v_binding = _v_sel_binding()
        receipt = select_threshold_from_v_sel(
            v_sel_binding=v_binding,
            row_ids=[1, 2],
            labels=[0, 1],
            probabilities=[0.1, 0.9],
            rule=rule,
        )
        verified = validate_threshold_receipt(
            receipt,
            expected_receipt_sha256=receipt["receipt_sha256"],
            expected_v_sel_binding=v_binding,
            expected_rule=rule,
            row_ids=[1, 2],
            labels=[0, 1],
            probabilities=[0.1, 0.9],
        )
        report = evaluate_outer_test(
            outer_test_binding=_outer_binding(),
            row_ids=[10, 11, 12, 13],
            labels=[0, 0, 1, 1],
            probabilities=[0.1, 0.2, 0.3, 0.4],
            verified_threshold_receipt=verified,
        )
        metrics = report["threshold_metrics"]
        self.assertIsNone(metrics["ppv"])
        self.assertEqual(metrics["f1"], 0.0)
        self.assertIsNone(metrics["mcc"])
        self.assertEqual(report["threshold_metric_status"]["ppv"], "null_if_no_predicted_positives")
        self.assertEqual(
            report["threshold_metric_status"]["mcc"], "null_if_any_mcc_denominator_factor_is_zero"
        )
        self.assertEqual(report["threshold_metric_denominators"]["ppv"], 0)
        self.assertEqual(report["threshold_metric_denominators"]["mcc_squared_product"], 0)

    def test_outer_test_never_calls_selector_or_changes_committed_threshold(self) -> None:
        receipt, rule, v_binding = _receipt(rule=_rule((0.5, 0.55)))
        verified = _verify_fixture_receipt(receipt, rule, v_binding)
        signature = inspect.signature(evaluate_outer_test)
        self.assertNotIn("selector", signature.parameters)
        with patch.object(
            evaluation,
            "select_threshold_from_v_sel",
            side_effect=AssertionError("outer test attempted threshold selection"),
        ):
            first = evaluate_outer_test(
                outer_test_binding=_outer_binding(),
                row_ids=[1, 2, 3, 4],
                labels=[0, 1, 0, 1],
                probabilities=[0.1, 0.9, 0.2, 0.8],
                verified_threshold_receipt=verified,
            )
            second = evaluate_outer_test(
                outer_test_binding=_outer_binding(),
                row_ids=[5, 6, 7, 8],
                labels=[1, 0, 1, 0],
                probabilities=[0.2, 0.9, 0.3, 0.8],
                verified_threshold_receipt=verified,
            )
        self.assertEqual(
            first["threshold_source"]["selected_threshold_hex"], receipt["selected_threshold_hex"]
        )
        self.assertEqual(
            second["threshold_source"]["selected_threshold_hex"], receipt["selected_threshold_hex"]
        )
        with self.assertRaisesRegex(EvaluationError, "raw-evidence-verified"):
            evaluate_outer_test(
                outer_test_binding=_outer_binding(),
                row_ids=[1, 2],
                labels=[0, 1],
                probabilities=[0.1, 0.9],
                verified_threshold_receipt=receipt,
            )


class FailClosedInputTests(unittest.TestCase):

    def test_invalid_probability_rows_fail_closed_in_hpo_and_v_sel(self) -> None:
        invalid_cases = [
            ([1, 1], [0, 1], [0.2, 0.8]),
            ([1, 2], [0, 1], [-0.1, 0.8]),
            ([1, 2], [0, 1], [0.2, 1.1]),
            ([1, 2], [0, 1], [float("nan"), 0.8]),
            ([1, 2], [0, 1], [float("inf"), 0.8]),
            ([1, 2], [1, 1], [0.2, 0.8]),
            ([1, 2], [0, 2], [0.2, 0.8]),
            ([1, 2], [0], [0.2, 0.8]),
        ]
        for row_ids, labels, probabilities in invalid_cases:
            with self.subTest(rows=row_ids, labels=labels, probabilities=probabilities):
                with self.assertRaises(EvaluationError):
                    compute_hpo_probability_metrics(row_ids, labels, probabilities)
                with self.assertRaises(EvaluationError):
                    select_threshold_from_v_sel(
                        v_sel_binding=_v_sel_binding(),
                        row_ids=row_ids,
                        labels=labels,
                        probabilities=probabilities,
                        rule=_rule(),
                    )

    def test_invalid_outer_rows_fail_closed_after_receipt_validation(self) -> None:
        receipt, rule, v_binding = _receipt()
        verified = _verify_fixture_receipt(receipt, rule, v_binding)
        invalid_cases = [
            ([1, 1], [0, 1], [0.2, 0.8]),
            ([1, 2], [0, 1], [float("nan"), 0.8]),
            ([1, 2], [1, 1], [0.2, 0.8]),
        ]
        for row_ids, labels, probabilities in invalid_cases:
            with self.subTest(rows=row_ids):
                with self.assertRaises(EvaluationError):
                    evaluate_outer_test(
                        outer_test_binding=_outer_binding(),
                        row_ids=row_ids,
                        labels=labels,
                        probabilities=probabilities,
                        verified_threshold_receipt=verified,
                    )

    def test_boolean_ids_labels_and_probabilities_are_not_silently_coerced(self) -> None:
        cases = [
            ([True, 2], [0, 1], [0.2, 0.8]),
            ([1, 2], [False, 1], [0.2, 0.8]),
            ([1, 2], [0, 1], [False, 0.8]),
        ]
        for row_ids, labels, probabilities in cases:
            with self.assertRaises(EvaluationError):
                compute_hpo_probability_metrics(row_ids, labels, probabilities)


if __name__ == "__main__":
    unittest.main()
