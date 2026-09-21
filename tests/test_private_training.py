"""Private release must use fresh entropy and expose only permitted outputs."""
import json
import unittest
from random import SystemRandom
from unittest.mock import patch
import torch

from fedsift.private_training import train_private_model
from tests import test_train_unit as fixtures
from fedsift.local_training import _draw_full_parameter_gaussian_noise, _draw_internal_poisson_mask


class PrivateRandomnessTests(unittest.TestCase):
    def test_os_gaussian_combines_four_draws_and_preserves_tensor_scale(self):
        generator = SystemRandom()
        state = {"weight": torch.zeros(2, dtype=torch.float64)}
        with patch.object(generator, "normalvariate", side_effect=[999, 1, 2, 3, 4, -1, -2, -3, -4]), \
             patch("fedsift.local_training.torch.randn", side_effect=AssertionError("public RNG used")):
            noise = _draw_full_parameter_gaussian_noise(state, noise_std=3.0, generator=generator)
        torch.testing.assert_close(noise["weight"], torch.tensor([15.0, -15.0], dtype=torch.float64),
                                   rtol=0, atol=0)

    def test_os_poisson_sampling_uses_independent_uniform_draws(self):
        generator = SystemRandom()
        with patch.object(generator, "random", side_effect=[0.1, 0.8, 0.49, 0.5]), \
             patch("fedsift.local_training.torch.rand", side_effect=AssertionError("public RNG used")):
            mask = _draw_internal_poisson_mask(population_size=4, sample_rate=0.5,
                                               device=torch.device("cpu"), generator=generator)
        self.assertEqual(mask.tolist(), [True, False, True, False])


class PrivateTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.TrainingUnitIntegrationTests.setUpClass()
        cls.base = fixtures.TrainingUnitIntegrationTests.base

    def test_fresh_private_runs_have_distinct_models_and_no_audit_payload(self):
        capability = fixtures._method_capability(self.base, "dp_fedavg")
        data = fixtures._sealed_data(capability)
        budget = fixtures._budget(data, capability)
        releases = [train_private_model(
            capability, data, budget,
            expected_capability_sha256=capability["capability_sha256"],
            expected_budget_sha256=budget.budget_sha256,
        ) for _ in range(2)]
        self.assertNotEqual(releases[0]["model_state"], releases[1]["model_state"])
        self.assertEqual(releases[0]["privacy"], releases[1]["privacy"])
        for release in releases:
            self.assertEqual(set(release), {"method", "model_manifest", "model_state", "privacy"})
            text = json.dumps(release)
            for forbidden in ("seed", "row_ids", "gradient_lineage", "noise_rng", "execution_history"):
                self.assertNotIn(forbidden, text)
            self.assertLessEqual(release["privacy"]["epsilon"], budget.target_epsilon)


if __name__ == "__main__":
    unittest.main()
