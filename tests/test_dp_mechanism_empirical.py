"""Deterministic mechanism-level audits for the FedSift record-DP implementation.

These tests exercise production sampling, production Gaussian-noise drawing,
the joint clipping kernel, and one independently derived q=1 Gaussian
accounting special case.  They are implementation audits only.  They are not
a formal DP proof, an attack-based security evaluation, a CSPRNG/entropy
assessment, or evidence for end-to-end, client-level, or deployment security.
"""

from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import hashlib
import math
import unittest
import warnings
from collections import OrderedDict
import torch
from fedsift import local_training
from fedsift.dp_kernel import privatize_per_record_gradients
from fedsift.privacy_accounting import PoissonDPStage, account_poisson_dpsgd

_CPU = torch.device("cpu")
_FAMILY_TAIL_PROBABILITY = 1e-09


def _domain_separated_seed(purpose: str, stream_index: int) -> int:
    """Return fixed audit seeds from disjoint, explicit test-only domains."""
    material = f"{_identity('dp_mechanism_empirical_audit')}{purpose}/stream_{stream_index}".encode(
        "ascii"
    )
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 2**63


def _flatten_state(state: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(tuple((value.reshape(-1) for value in state.values())))


def _standard_normal_cdf(value: float) -> float:
    return 0.5 * math.erfc(-value / math.sqrt(2.0))


def _gaussian_composition_delta(epsilon: float, *, noise_multiplier: float, steps: int) -> float:
    """Exact hockey-stick divergence for composed full-sample Gaussians.

    For q=1, T Gaussian mechanisms with sensitivity-to-noise ratio 1/sigma
    have Gaussian privacy-loss parameter mu=sqrt(T)/sigma.  This expression is
    derived directly from the two shifted normal distributions and does not
    call Opacus or another accountant wrapper.
    """
    mu = math.sqrt(steps) / noise_multiplier
    return _standard_normal_cdf(mu / 2.0 - epsilon / mu) - math.exp(epsilon) * _standard_normal_cdf(
        -mu / 2.0 - epsilon / mu
    )


def _exact_q1_gaussian_epsilon(*, delta: float, noise_multiplier: float, steps: int) -> float:
    lower = 0.0
    upper = 64.0
    if _gaussian_composition_delta(lower, noise_multiplier=noise_multiplier, steps=steps) <= delta:
        return 0.0
    if _gaussian_composition_delta(upper, noise_multiplier=noise_multiplier, steps=steps) > delta:
        raise AssertionError("Gaussian oracle bracket does not contain the target")
    for _ in range(160):
        midpoint = (lower + upper) / 2.0
        if (
            _gaussian_composition_delta(midpoint, noise_multiplier=noise_multiplier, steps=steps)
            > delta
        ):
            lower = midpoint
        else:
            upper = midpoint
    return upper


class ProductionRandomDrawEmpiricalAuditTests(unittest.TestCase):

    def test_poisson_inclusion_rate_and_cross_record_covariance(self) -> None:
        """Audit iid Bernoulli(q) behavior under fixed separated streams."""
        sample_rate = 0.27
        stream_count = 4
        trials_per_stream = 4096
        record_count = 12
        draws: list[torch.Tensor] = []
        for stream_index in range(stream_count):
            generator = torch.Generator(device=_CPU).manual_seed(
                _domain_separated_seed("iid_poisson_sampling", stream_index)
            )
            flat_mask = local_training._draw_internal_poisson_mask(
                population_size=trials_per_stream * record_count,
                sample_rate=sample_rate,
                device=_CPU,
                generator=generator,
            )
            draws.append(flat_mask.reshape(trials_per_stream, record_count).to(torch.float64))
        observations = torch.cat(draws, dim=0)
        total_trials = int(observations.shape[0])
        pooled_count = observations.numel()
        pooled_rate_bound = math.sqrt(
            math.log(2.0 / _FAMILY_TAIL_PROBABILITY) / (2.0 * pooled_count)
        )
        observed_rate = float(observations.mean().item())
        self.assertLessEqual(abs(observed_rate - sample_rate), pooled_rate_bound)
        per_record_rate_bound = math.sqrt(
            math.log(2.0 * record_count / _FAMILY_TAIL_PROBABILITY) / (2.0 * total_trials)
        )
        maximum_rate_error = float(
            torch.max(torch.abs(observations.mean(dim=0) - sample_rate)).item()
        )
        self.assertLessEqual(maximum_rate_error, per_record_rate_bound)
        centered = observations - sample_rate
        covariance_moments = centered.T @ centered / total_trials
        upper_triangle = torch.triu_indices(record_count, record_count, offset=1)
        cross_record = covariance_moments[upper_triangle[0], upper_triangle[1]]
        pair_count = int(cross_record.numel())
        covariance_bound = max(sample_rate, 1.0 - sample_rate) * math.sqrt(
            math.log(2.0 * pair_count / _FAMILY_TAIL_PROBABILITY) / (2.0 * total_trials)
        )
        self.assertLessEqual(float(torch.max(torch.abs(cross_record)).item()), covariance_bound)

    def test_full_parameter_gaussian_mean_and_standard_deviation(self) -> None:
        """Audit production N(0, sigma^2 C^2) draws for every parameter."""
        noise_std = 1.7
        stream_count = 4
        state = OrderedDict(
            weight=torch.zeros((8192, 4), dtype=torch.float64),
            bias=torch.zeros((8192,), dtype=torch.float64),
        )
        by_parameter: dict[str, list[torch.Tensor]] = {name: [] for name in state}
        for stream_index in range(stream_count):
            generator = torch.Generator(device=_CPU).manual_seed(
                _domain_separated_seed("gaussian_parameter_noise", stream_index)
            )
            noise = local_training._draw_full_parameter_gaussian_noise(
                state, noise_std=noise_std, generator=generator
            )
            for name, value in noise.items():
                by_parameter[name].append(value.reshape(-1))
        parameter_count = len(by_parameter)
        gaussian_tail_x = math.log(2.0 * parameter_count / _FAMILY_TAIL_PROBABILITY)
        for name, chunks in by_parameter.items():
            with self.subTest(parameter=name):
                values = torch.cat(chunks)
                sample_count = int(values.numel())
                mean_bound = noise_std * math.sqrt(2.0 * gaussian_tail_x / sample_count)
                self.assertLessEqual(abs(float(values.mean().item())), mean_bound)
                degrees_of_freedom = sample_count - 1
                root_term = math.sqrt(gaussian_tail_x / degrees_of_freedom)
                lower_std = noise_std * math.sqrt(max(0.0, 1.0 - 2.0 * root_term))
                upper_std = noise_std * math.sqrt(
                    1.0 + 2.0 * root_term + 2.0 * gaussian_tail_x / degrees_of_freedom
                )
                observed_std = float(values.std(unbiased=True).item())
                self.assertGreaterEqual(observed_std, lower_std)
                self.assertLessEqual(observed_std, upper_std)


class JointClippingAndNormalizationInvariantTests(unittest.TestCase):

    def test_fixed_slot_add_remove_changes_output_by_at_most_c_over_qn(self) -> None:
        """Check active-versus-dummy fixed-slot sensitivity with fixed qN."""
        clip_norm = 2.6
        fixed_qn = 10.0
        dummy_slot = OrderedDict(
            weight=torch.tensor([[0.1, -0.2], [0.0, 0.0]], dtype=torch.float64),
            bias=torch.tensor([[0.3], [0.0]], dtype=torch.float64),
        )
        active_slot = OrderedDict(
            weight=torch.tensor([[0.1, -0.2], [3.0, 4.0]], dtype=torch.float64),
            bias=torch.tensor([[0.3], [12.0]], dtype=torch.float64),
        )
        zero_noise = OrderedDict(
            weight=torch.zeros((2,), dtype=torch.float64),
            bias=torch.zeros((1,), dtype=torch.float64),
        )
        common_output = privatize_per_record_gradients(
            dummy_slot, zero_noise, clip_norm=clip_norm, fixed_normalization=fixed_qn
        )
        adjacent_output = privatize_per_record_gradients(
            active_slot, zero_noise, clip_norm=clip_norm, fixed_normalization=fixed_qn
        )
        output_difference = OrderedDict(
            ((name, adjacent_output[name] - common_output[name]) for name in common_output)
        )
        observed_sensitivity = float(
            torch.linalg.vector_norm(_flatten_state(output_difference)).item()
        )
        self.assertLessEqual(observed_sensitivity, clip_norm / fixed_qn + 1e-12)
        self.assertAlmostEqual(observed_sensitivity, clip_norm / fixed_qn, places=12)

    def test_replace_one_can_require_two_c_over_qn_sensitivity(self) -> None:
        """Guard against reusing the add/remove C bound for replace-one."""
        clip_norm = 5.0
        fixed_qn = 10.0
        first_active_record = OrderedDict(
            weight=torch.tensor([[3.0, 4.0]], dtype=torch.float64),
            bias=torch.tensor([[0.0]], dtype=torch.float64),
        )
        replacement_active_record = OrderedDict(
            weight=torch.tensor([[-3.0, -4.0]], dtype=torch.float64),
            bias=torch.tensor([[0.0]], dtype=torch.float64),
        )
        zero_noise = OrderedDict(
            weight=torch.zeros((2,), dtype=torch.float64),
            bias=torch.zeros((1,), dtype=torch.float64),
        )
        first_output = privatize_per_record_gradients(
            first_active_record, zero_noise, clip_norm=clip_norm, fixed_normalization=fixed_qn
        )
        replacement_output = privatize_per_record_gradients(
            replacement_active_record, zero_noise, clip_norm=clip_norm, fixed_normalization=fixed_qn
        )
        difference = OrderedDict(
            ((name, replacement_output[name] - first_output[name]) for name in first_output)
        )
        observed = float(torch.linalg.vector_norm(_flatten_state(difference)).item())
        self.assertGreater(observed, clip_norm / fixed_qn)
        self.assertAlmostEqual(observed, 2.0 * clip_norm / fixed_qn, places=12)

    def test_realized_batch_count_does_not_replace_registered_qn_divisor(self) -> None:
        """A zero-gradient extra draw cannot change the fixed-normalized sum."""
        fixed_qn = 8.0
        one_record = OrderedDict(
            weight=torch.tensor([[1.0, 2.0]], dtype=torch.float64),
            bias=torch.tensor([[2.0]], dtype=torch.float64),
        )
        two_records_same_sum = OrderedDict(
            weight=torch.tensor([[1.0, 2.0], [0.0, 0.0]], dtype=torch.float64),
            bias=torch.tensor([[2.0], [0.0]], dtype=torch.float64),
        )
        zero_noise = OrderedDict(
            weight=torch.zeros((2,), dtype=torch.float64),
            bias=torch.zeros((1,), dtype=torch.float64),
        )
        one_output = privatize_per_record_gradients(
            one_record, zero_noise, clip_norm=10.0, fixed_normalization=fixed_qn
        )
        two_output = privatize_per_record_gradients(
            two_records_same_sum, zero_noise, clip_norm=10.0, fixed_normalization=fixed_qn
        )
        for name in one_output:
            torch.testing.assert_close(one_output[name], two_output[name], rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            one_output["weight"],
            torch.tensor([1.0 / 8.0, 2.0 / 8.0], dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            one_output["bias"], torch.tensor([2.0 / 8.0], dtype=torch.float64), rtol=0.0, atol=0.0
        )


class IndependentGaussianAccountantSpecialCaseTests(unittest.TestCase):

    def test_q1_prv_bound_agrees_with_exact_composed_gaussian_oracle(self) -> None:
        """Cross-check only the non-subsampled Gaussian special case."""
        delta = 1e-05
        noise_multiplier = 2.0
        steps = 3
        oracle_epsilon = _exact_q1_gaussian_epsilon(
            delta=delta, noise_multiplier=noise_multiplier, steps=steps
        )
        self.assertAlmostEqual(
            _gaussian_composition_delta(
                oracle_epsilon, noise_multiplier=noise_multiplier, steps=steps
            ),
            delta,
            delta=1e-12,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                module="opacus\\.accountants\\.analysis\\.prv\\.prvs",
            )
            report = account_poisson_dpsgd(
                (PoissonDPStage("full_sample_gaussian", 1.0, noise_multiplier, steps),),
                delta=delta,
                eps_error=0.001,
                delta_error=1e-08,
            )
        self.assertEqual(report.adjacency, "add_or_remove_one_record")
        self.assertGreaterEqual(report.epsilon_prv_upper + 1e-12, oracle_epsilon)
        self.assertLessEqual(report.epsilon_prv_upper - oracle_epsilon, 0.005)
        self.assertGreaterEqual(report.epsilon_rdp + 1e-12, oracle_epsilon)


if __name__ == "__main__":
    unittest.main()
