"""Independent resampled-record oracle, including unequal and paired groups."""
import unittest

import numpy as np

from fedsift.paired_bootstrap import paired_group_intervals


class PairedBootstrapTests(unittest.TestCase):
    def test_matches_literal_resampled_records_and_preserves_pairing(self):
        values = np.array([[1., -1.], [3., -3.], [7., -7.]])
        result = paired_group_intervals(values, ["a", "a", "b"], seed=41, replicates=300)
        rng = np.random.default_rng(41)
        members = [[0, 1], [2]]
        samples = []
        for _ in range(300):
            records = [record for group in rng.integers(0, 2, 2) for record in members[group]]
            samples.append(values[records].mean(axis=0))
        lo, hi = np.quantile(samples, [.025, .975], axis=0)
        np.testing.assert_allclose(result.effect, values.mean(axis=0), atol=1e-15)
        np.testing.assert_allclose(result.lower, lo, atol=1e-15)
        np.testing.assert_allclose(result.upper, hi, atol=1e-15)
        self.assertEqual(result.lower[0], -result.upper[1])

    def test_identical_predictions_have_zero_interval(self):
        result = paired_group_intervals(np.zeros((4, 1)), ["a", "b", "b", "c"], seed=8)
        self.assertEqual(result.group_count, 3)
        self.assertEqual(result.lower[0], 0)
        self.assertEqual(result.upper[0], 0)

    def test_nonfinite_or_misaligned_data_is_rejected(self):
        for values, groups in (([[float("nan")]], ["a"]), ([[1.], [2.]], ["a"])):
            with self.assertRaises(ValueError):
                paired_group_intervals(values, groups, seed=1)


if __name__ == "__main__":
    unittest.main()
