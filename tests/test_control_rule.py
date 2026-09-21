from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import copy
import math
import unittest
from fedsift.control_rule import (
    ControlDecisionScope,
    ControlEvidenceBinding,
    ControlRuleError,
    decide_fedsift,
    decide_public_argmin,
    validate_control_decision_receipt,
    validate_information_matched_pair,
)


def _scope() -> ControlDecisionScope:
    return ControlDecisionScope(
        study_id=_identity("artifact_identity"),
        dataset_id="retinopathy",
        outer_repeat=1,
        outer_fold=2,
        eval_seed=17,
        candidate_id="candidate_007",
        server_round=10,
        query_index=2,
    )


def _binding() -> ControlEvidenceBinding:
    return ControlEvidenceBinding(
        control_membership_sha256="1" * 64,
        preprocessing_artifact_sha256="2" * 64,
        model_manifest_sha256="3" * 64,
        candidate_parameters_sha256="4" * 64,
        global_state_sha256="5" * 64,
        aggregate_direction_sha256="6" * 64,
    )


def _losses(labels: list[int], probabilities: list[float]) -> list[float]:
    values: list[float] = []
    for label, probability in zip(labels, probabilities):
        clipped = min(max(probability, 1e-07), 1.0 - 1e-07)
        values.append(-math.log(clipped) if label == 1 else -math.log1p(-clipped))
    return values


def _candidate_table(labels: list[int]) -> list[dict[str, object]]:
    probability_rows = [[0.55, 0.45, 0.55, 0.45], [0.85, 0.15, 0.8, 0.2], [0.7, 0.3, 0.65, 0.35]]
    rows: list[dict[str, object]] = []
    for order, (alpha, probabilities) in enumerate(zip((0.0, 0.5, 1.0), probability_rows)):
        losses = _losses(labels, probabilities)
        rows.append(
            {
                "order": order,
                "alpha": alpha,
                "mean_log_loss": sum(losses) / len(losses),
                "per_record_log_loss": losses,
                "probability": probabilities,
            }
        )
    return rows


def _validate_pair(
    fedsift: dict[str, object],
    argmin: dict[str, object],
    *,
    row_ids: list[int],
    labels: list[int],
    table: list[dict[str, object]],
    alphas: tuple[float, ...],
    binding: ControlEvidenceBinding | None = None,
) -> dict[str, object]:
    return validate_information_matched_pair(
        fedsift,
        argmin,
        expected_fedsift_receipt_sha256=fedsift["receipt_sha256"],
        expected_public_argmin_receipt_sha256=argmin["receipt_sha256"],
        scope=_scope(),
        binding=_binding() if binding is None else binding,
        row_ids=row_ids,
        labels=labels,
        candidate_table=table,
        expected_alphas=alphas,
        safety_margin_z=0.0,
        minimum_control_improvement=0.0,
    )


class ControlRuleGoldenTests(unittest.TestCase):

    def test_small_control_sets_fall_back_without_inventing_uncertainty(self) -> None:
        for labels in ([], [1]):
            with self.subTest(labels=labels):
                table = []
                for order, (alpha, probability) in enumerate(((0.0, 0.5), (0.5, 0.99), (1.0, 0.1))):
                    probabilities = [probability] * len(labels)
                    losses = _losses(labels, probabilities)
                    table.append({
                        "order": order, "alpha": alpha, "probability": probabilities,
                        "per_record_log_loss": losses,
                        "mean_log_loss": sum(losses) / len(losses) if losses else None,
                    })
                receipt = decide_fedsift(
                    scope=_scope(), binding=_binding(), row_ids=list(range(len(labels))),
                    labels=labels, candidate_table=table, expected_alphas=(0.0, 0.5, 1.0),
                    safety_margin_z=0.0, minimum_control_improvement=0.0,
                )
                self.assertEqual(float.fromhex(receipt["selected_alpha_hex"]), 1.0)
                for row in receipt["decision_rows"]:
                    self.assertIsNone(row["paired_standard_error_hex"])
                    self.assertFalse(row["passes_supported_override"])
                validate_control_decision_receipt(
                    receipt, expected_receipt_sha256=receipt["receipt_sha256"],
                    rule_name="fedsift_supported_override", scope=_scope(), binding=_binding(),
                    row_ids=list(range(len(labels))), labels=labels, candidate_table=table,
                    expected_alphas=(0.0, 0.5, 1.0), safety_margin_z=0.0,
                    minimum_control_improvement=0.0,
                )

    def test_single_class_control_set_can_support_a_partial_step(self) -> None:
        labels = [1, 1, 1, 1]
        table = _candidate_table(labels)
        for row, probability in zip(table, (0.5, 0.9, 0.6)):
            row["probability"] = [probability] * 4
            row["per_record_log_loss"] = _losses(labels, row["probability"])
            row["mean_log_loss"] = sum(row["per_record_log_loss"]) / 4
        receipt = decide_fedsift(
            scope=_scope(), binding=_binding(), row_ids=[1, 2, 3, 4], labels=labels,
            candidate_table=table, expected_alphas=(0.0, 0.5, 1.0),
            safety_margin_z=1.96, minimum_control_improvement=0.0,
        )
        self.assertEqual(float.fromhex(receipt["selected_alpha_hex"]), 0.5)

    def setUp(self) -> None:
        self.row_ids = [101, 102, 103, 104]
        self.labels = [1, 0, 1, 0]
        self.table = _candidate_table(self.labels)
        self.alphas = (0.0, 0.5, 1.0)

    def test_argmin_and_fedsift_use_same_table_but_distinct_rules(self) -> None:
        argmin = decide_public_argmin(
            scope=_scope(),
            binding=_binding(),
            row_ids=self.row_ids,
            labels=self.labels,
            candidate_table=self.table,
            expected_alphas=self.alphas,
        )
        fedsift = decide_fedsift(
            scope=_scope(),
            binding=_binding(),
            row_ids=self.row_ids,
            labels=self.labels,
            candidate_table=self.table,
            expected_alphas=self.alphas,
            safety_margin_z=0.0,
            minimum_control_improvement=0.0,
        )
        self.assertEqual(float.fromhex(argmin["selected_alpha_hex"]), 0.5)
        self.assertEqual(float.fromhex(fedsift["selected_alpha_hex"]), 0.5)
        matched = _validate_pair(
            fedsift,
            argmin,
            row_ids=self.row_ids,
            labels=self.labels,
            table=self.table,
            alphas=self.alphas,
        )
        self.assertTrue(matched["same_control_information"])
        self.assertTrue(matched["same_raw_control_table_and_declared_state_bindings"])
        self.assertEqual(
            matched["attestation_scope"],
            "single_decision_counterfactual_on_one_bound_state_direction_and_raw_table",
        )
        self.assertFalse(matched["independent_recursive_trajectories_verified"])
        self.assertFalse(matched["candidate_derivation_from_bound_states_verified"])
        self.assertEqual(len(matched["pair_sha256"]), 64)
        self.assertEqual(
            matched["candidate_table_sha256"], argmin["evidence"]["candidate_table_sha256"]
        )
        self.assertFalse(fedsift["statistical_guarantee_claim"])
        self.assertIn("heuristic_not_a_finite_sample", fedsift["rule"]["margin_interpretation"])
        self.assertEqual(fedsift["rule"]["paired_standard_error"], "sample_sd_ddof_1_div_sqrt_n")
        self.assertIn("mean_minus_z", fedsift["rule"]["override_acceptance_rule"])

    def test_fedsift_falls_back_to_full_step_when_margin_is_not_supported(self) -> None:
        receipt = decide_fedsift(
            scope=_scope(),
            binding=_binding(),
            row_ids=self.row_ids,
            labels=self.labels,
            candidate_table=self.table,
            expected_alphas=self.alphas,
            safety_margin_z=100.0,
            minimum_control_improvement=0.0,
        )
        self.assertEqual(float.fromhex(receipt["selected_alpha_hex"]), 1.0)
        self.assertEqual(receipt["selection_status"], "fallback_full_step_no_supported_override")

    def test_paired_margin_fields_match_literal_sample_se_oracle(self) -> None:
        z_value = 1.25
        receipt = decide_fedsift(
            scope=_scope(),
            binding=_binding(),
            row_ids=self.row_ids,
            labels=self.labels,
            candidate_table=self.table,
            expected_alphas=self.alphas,
            safety_margin_z=z_value,
            minimum_control_improvement=0.0,
        )
        full_losses = _losses(self.labels, self.table[2]["probability"])
        candidate_losses = _losses(self.labels, self.table[1]["probability"])
        differences = [
            full_loss - candidate_loss
            for (full_loss, candidate_loss) in zip(full_losses, candidate_losses)
        ]
        mean = sum(differences) / len(differences)
        sample_variance = sum(((value - mean) ** 2 for value in differences)) / (
            len(differences) - 1
        )
        standard_error = math.sqrt(sample_variance) / math.sqrt(len(differences))
        row = receipt["decision_rows"][1]
        self.assertAlmostEqual(
            float.fromhex(row["paired_improvement_vs_full_hex"]), mean, places=15
        )
        self.assertAlmostEqual(
            float.fromhex(row["paired_standard_error_hex"]), standard_error, places=15
        )
        self.assertAlmostEqual(
            float.fromhex(row["heuristic_lower_margin_hex"]),
            mean - z_value * standard_error,
            places=15,
        )
        self.assertFalse(receipt["statistical_guarantee_claim"])

    def test_receipt_rebuild_and_external_commitment_are_both_required(self) -> None:
        receipt = decide_fedsift(
            scope=_scope(),
            binding=_binding(),
            row_ids=self.row_ids,
            labels=self.labels,
            candidate_table=self.table,
            expected_alphas=self.alphas,
            safety_margin_z=0.0,
            minimum_control_improvement=0.0,
        )
        rebuilt = validate_control_decision_receipt(
            receipt,
            expected_receipt_sha256=receipt["receipt_sha256"],
            rule_name="fedsift_supported_override",
            scope=_scope(),
            binding=_binding(),
            row_ids=self.row_ids,
            labels=self.labels,
            candidate_table=self.table,
            expected_alphas=self.alphas,
            safety_margin_z=0.0,
            minimum_control_improvement=0.0,
        )
        self.assertEqual(rebuilt, receipt)
        with self.assertRaisesRegex(ControlRuleError, "externally committed"):
            validate_control_decision_receipt(
                receipt,
                expected_receipt_sha256="f" * 64,
                rule_name="fedsift_supported_override",
                scope=_scope(),
                binding=_binding(),
                row_ids=self.row_ids,
                labels=self.labels,
                candidate_table=self.table,
                expected_alphas=self.alphas,
                safety_margin_z=0.0,
                minimum_control_improvement=0.0,
            )


class ControlRuleFailClosedTests(unittest.TestCase):

    def setUp(self) -> None:
        self.row_ids = [101, 102, 103, 104]
        self.labels = [1, 0, 1, 0]
        self.table = _candidate_table(self.labels)
        self.alphas = (0.0, 0.5, 1.0)

    def _argmin(self, **overrides):
        arguments = {
            "scope": _scope(),
            "binding": _binding(),
            "row_ids": self.row_ids,
            "labels": self.labels,
            "candidate_table": self.table,
            "expected_alphas": self.alphas,
        }
        arguments.update(overrides)
        return decide_public_argmin(**arguments)

    def test_probability_loss_label_and_mean_tampering_fail(self) -> None:
        cases = []
        changed_loss = copy.deepcopy(self.table)
        changed_loss[1]["per_record_log_loss"][0] += 0.1
        cases.append(changed_loss)
        changed_mean = copy.deepcopy(self.table)
        changed_mean[1]["mean_log_loss"] += 0.1
        cases.append(changed_mean)
        changed_probability = copy.deepcopy(self.table)
        changed_probability[1]["probability"][0] = float("nan")
        cases.append(changed_probability)
        for table in cases:
            with self.subTest(table=table):
                with self.assertRaises(ControlRuleError):
                    self._argmin(candidate_table=table)
        with self.assertRaises(ControlRuleError):
            self._argmin(labels=[0, 0, 1, 1])

    def test_grid_order_duplicates_boolean_ids_and_extra_fields_fail(self) -> None:
        with self.assertRaises(ControlRuleError):
            self._argmin(expected_alphas=(0.0, 1.0, 0.5))
        with self.assertRaises(ControlRuleError):
            self._argmin(expected_alphas=(0.0, 0.5, 0.5, 1.0))
        with self.assertRaises(ControlRuleError):
            self._argmin(row_ids=[True, 102, 103, 104])
        polluted = copy.deepcopy(self.table)
        polluted[0]["average_precision"] = 0.9
        with self.assertRaises(ControlRuleError):
            self._argmin(candidate_table=polluted)

    def test_information_pair_rejects_different_rows_or_direction(self) -> None:
        argmin = self._argmin()
        fedsift = decide_fedsift(
            scope=_scope(),
            binding=_binding(),
            row_ids=self.row_ids,
            labels=self.labels,
            candidate_table=self.table,
            expected_alphas=self.alphas,
            safety_margin_z=0.0,
            minimum_control_improvement=0.0,
        )
        changed_table = copy.deepcopy(self.table)
        changed_table[0]["probability"] = list(reversed(changed_table[0]["probability"]))
        changed_table[0]["per_record_log_loss"] = _losses(
            self.labels, changed_table[0]["probability"]
        )
        changed_table[0]["mean_log_loss"] = sum(changed_table[0]["per_record_log_loss"]) / len(
            self.labels
        )
        other = self._argmin(candidate_table=changed_table)
        with self.assertRaises(ControlRuleError):
            _validate_pair(
                fedsift,
                other,
                row_ids=self.row_ids,
                labels=self.labels,
                table=changed_table,
                alphas=self.alphas,
            )
        other_binding = ControlEvidenceBinding(
            control_membership_sha256="1" * 64,
            preprocessing_artifact_sha256="2" * 64,
            model_manifest_sha256="3" * 64,
            candidate_parameters_sha256="4" * 64,
            global_state_sha256="5" * 64,
            aggregate_direction_sha256="7" * 64,
        )
        other = self._argmin(binding=other_binding)
        with self.assertRaises(ControlRuleError):
            _validate_pair(
                fedsift,
                other,
                row_ids=self.row_ids,
                labels=self.labels,
                table=self.table,
                alphas=self.alphas,
                binding=other_binding,
            )
        self.assertIsNotNone(argmin)

    def test_information_pair_cannot_accept_one_self_hashed_rule_as_both_arms(self) -> None:
        fedsift = decide_fedsift(
            scope=_scope(),
            binding=_binding(),
            row_ids=self.row_ids,
            labels=self.labels,
            candidate_table=self.table,
            expected_alphas=self.alphas,
            safety_margin_z=0.0,
            minimum_control_improvement=0.0,
        )
        with self.assertRaises(ControlRuleError):
            _validate_pair(
                fedsift,
                fedsift,
                row_ids=self.row_ids,
                labels=self.labels,
                table=self.table,
                alphas=self.alphas,
            )


if __name__ == "__main__":
    unittest.main()
