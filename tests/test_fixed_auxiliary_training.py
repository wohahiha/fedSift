"""Exercise add/remove records through the complete training entry point."""
import unittest
from unittest.mock import patch

import fedsift.train_unit as training
from tests import test_train_unit as fixtures


class FixedAuxiliaryTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.TrainingUnitIntegrationTests.setUpClass()
        cls.base = fixtures.TrainingUnitIntegrationTests.base

    def test_reference_counts_and_accounting_survive_record_removal(self):
        capability = fixtures._method_capability(self.base, "fedsift")
        original = fixtures._sealed_data(capability)
        budget = fixtures._budget(original, capability)
        reference_count = len(original.role("client_0").row_ids)
        observed_counts = []
        weighted_delta = training._weighted_delta

        def record_counts(states, base, counts):
            observed_counts.append(tuple(counts))
            return weighted_delta(states, base, counts)

        reports = []
        for removed in (0, 1, reference_count):
            role_tables = {}
            for table in original.roles:
                start = removed if table.role == "client_0" else 0
                role_tables[table.role] = (table.row_ids[start:], table.features[start:], table.labels[start:])
            data = training.seal_training_role_tables(
                capability, role_tables,
                preprocessing_artifact_sha256=original.preprocessing_artifact_sha256,
                expected_capability_sha256=capability["capability_sha256"],
            )
            with patch("fedsift.train_unit._weighted_delta", side_effect=record_counts):
                result = training.execute_training_unit(
                    capability, data, budget,
                    expected_capability_sha256=capability["capability_sha256"],
                    expected_budget_sha256=budget.budget_sha256,
                )
            training.validate_training_unit_result(
                result, capability, data, budget,
                expected_capability_sha256=capability["capability_sha256"],
                expected_budget_sha256=budget.budget_sha256,
            )
            for receipt in result.local_step_receipts:
                self.assertEqual(receipt.population_size, reference_count)
                self.assertEqual(receipt.fixed_expected_batch_normalization, budget.poisson_sample_rate * reference_count)
                self.assertTrue(set(receipt.selected_row_ids).issubset(data.role("client_0").row_ids))
                if removed == reference_count:
                    self.assertEqual(receipt.sampled_record_count, 0)
            reports.append(result.parallel_privacy_report.epsilon)
        self.assertEqual(set(observed_counts), {(reference_count,)})
        self.assertEqual(len(set(reports)), 1)

    def test_public_membership_cannot_change_through_private_subset_entry(self):
        capability = fixtures._method_capability(self.base, "fedsift")
        original = fixtures._sealed_data(capability)
        roles = {t.role: (t.row_ids, t.features, t.labels) for t in original.roles}
        ids, features, labels = roles["v_ctrl"]
        roles["v_ctrl"] = (ids[1:], features[1:], labels[1:])
        with self.assertRaisesRegex(training.TrainingUnitError, "membership"):
            training.seal_training_role_tables(
                capability, roles,
                preprocessing_artifact_sha256=original.preprocessing_artifact_sha256,
                expected_capability_sha256=capability["capability_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
