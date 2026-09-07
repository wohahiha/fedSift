from __future__ import annotations
import json
import unittest
from collections import OrderedDict
import numpy as np
import torch
from fedsift.baseline_math import (
    BaselineMathError,
    FedAdamState,
    FedYogiState,
    fedadam_init_like,
    fedadam_server_step,
    fedyogi_init_like,
    fedyogi_server_step,
    fedavg_server_average,
    local_sgd_explicit_batches,
    scaffold_local_step,
    scaffold_option2_control,
    scaffold_server_batched_weighted,
    scaffold_server_original,
    score_public_direction_grid,
    select_public_argmin,
)


def scalar(value: float) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(x=torch.tensor([value], dtype=torch.float64))


def scalar_value(state: OrderedDict[str, torch.Tensor]) -> float:
    return float(state["x"].item())


class FedAvgGoldenTests(unittest.TestCase):

    def test_fedavg_algorithm1_two_round_golden(self) -> None:
        targets = (1.0, -1.0)
        batch_plan = (((0,), (1,)),)

        def train(start: OrderedDict[str, torch.Tensor], target: float):

            def gradient(state, _batch):
                return scalar(scalar_value(state) - target)

            return local_sgd_explicit_batches(start, batch_plan, gradient, 0.1)

        global_state = scalar(2.0)
        local_r1 = [train(global_state, target) for target in targets]
        self.assertTrue(
            np.allclose(
                [scalar_value(value) for value in local_r1],
                np.array([1.81, 1.43]),
                rtol=0.0,
                atol=1e-12,
            )
        )
        global_r1 = fedavg_server_average(local_r1, [1, 3])
        self.assertAlmostEqual(scalar_value(global_r1), 1.525, places=12)
        local_r2 = [train(global_r1, target) for target in targets]
        self.assertTrue(
            np.allclose(
                [scalar_value(value) for value in local_r2],
                np.array([1.42525, 1.04525]),
                rtol=0.0,
                atol=1e-12,
            )
        )
        global_r2 = fedavg_server_average(local_r2, [1, 3])
        self.assertAlmostEqual(scalar_value(global_r2), 1.14025, places=12)


class FedAdamGoldenTests(unittest.TestCase):

    def test_fedadam_algorithm2_two_round_golden(self) -> None:
        state = OrderedDict(x=torch.tensor([1.0, -2.0], dtype=torch.float64))
        optimizer = fedadam_init_like(state, tau=0.001)
        self.assertTrue(
            torch.equal(optimizer.v["x"], torch.tensor([1e-06, 1e-06], dtype=torch.float64))
        )
        state, optimizer = fedadam_server_step(
            state,
            OrderedDict(x=torch.tensor([0.4, -0.2], dtype=torch.float64)),
            optimizer,
            beta1=0.9,
            beta2=0.99,
            server_learning_rate=0.05,
            tau=0.001,
        )
        self.assertTrue(np.allclose(optimizer.m["x"].numpy(), [0.04, -0.02], rtol=0.0, atol=1e-12))
        self.assertTrue(
            np.allclose(optimizer.v["x"].numpy(), [0.00160099, 0.00040099], rtol=0.0, atol=1e-12)
        )
        self.assertTrue(
            np.allclose(
                state["x"].numpy(), [1.0487657711439873, -2.0475630258377944], rtol=0.0, atol=1e-12
            )
        )
        state, optimizer = fedadam_server_step(
            state,
            OrderedDict(x=torch.tensor([0.1, 0.3], dtype=torch.float64)),
            optimizer,
            beta1=0.9,
            beta2=0.99,
            server_learning_rate=0.05,
            tau=0.001,
        )
        self.assertTrue(np.allclose(optimizer.m["x"].numpy(), [0.046, 0.012], rtol=0.0, atol=1e-12))
        self.assertTrue(
            np.allclose(
                optimizer.v["x"].numpy(), [0.0016849801, 0.0012969801], rtol=0.0, atol=1e-12
            )
        )
        self.assertTrue(
            np.allclose(
                state["x"].numpy(), [1.1034645000355534, -2.031352772334353], rtol=0.0, atol=1e-12
            )
        )


class FedYogiGoldenTests(unittest.TestCase):

    def test_fedyogi_algorithm2_two_round_golden(self) -> None:
        state = scalar(1.0)
        optimizer = fedyogi_init_like(state, tau=0.1)
        self.assertAlmostEqual(scalar_value(optimizer.v), 0.01, places=15)
        state, optimizer = fedyogi_server_step(
            state, scalar(0.4), optimizer, beta1=0.5, beta2=0.75, server_learning_rate=0.2, tau=0.1
        )
        self.assertAlmostEqual(scalar_value(optimizer.m), 0.2, places=15)
        self.assertAlmostEqual(scalar_value(optimizer.v), 0.05, places=15)
        self.assertAlmostEqual(scalar_value(state), 1.123606797749979, places=15)
        state, optimizer = fedyogi_server_step(
            state, scalar(0.1), optimizer, beta1=0.5, beta2=0.75, server_learning_rate=0.2, tau=0.1
        )
        self.assertAlmostEqual(scalar_value(optimizer.m), 0.15, places=15)
        self.assertAlmostEqual(scalar_value(optimizer.v), 0.0475, places=15)
        self.assertAlmostEqual(scalar_value(state), 1.217962755491606, places=15)


class ScaffoldGoldenTests(unittest.TestCase):

    def _client_round(self, x, server_c, client_cs):
        targets = (1.0, -1.0)
        local = []
        next_controls = []
        for target, client_c in zip(targets, client_cs):
            gradient = scalar(scalar_value(x) - target)
            y = scaffold_local_step(x, gradient, server_c, client_c, 0.1)
            next_control = scaffold_option2_control(
                client_c, server_c, x, y, actual_local_steps=1, local_learning_rate=0.1
            )
            local.append(y)
            next_controls.append(next_control)
        model_deltas = [scalar(scalar_value(value) - scalar_value(x)) for value in local]
        control_deltas = [
            scalar(scalar_value(new) - scalar_value(old))
            for (new, old) in zip(next_controls, client_cs)
        ]
        return (local, next_controls, model_deltas, control_deltas)

    def test_scaffold_original_algorithm1_golden(self) -> None:
        x = scalar(2.0)
        server_c = scalar(0.0)
        client_cs = [scalar(0.0), scalar(0.0)]
        expected = [([1.9, 1.7], [1.0, 3.0], 1.8, 2.0), ([1.62, 1.62], [0.8, 2.8], 1.62, 1.8)]
        for local_expected, controls_expected, x_expected, c_expected in expected:
            local, next_controls, model_deltas, control_deltas = self._client_round(
                x, server_c, client_cs
            )
            x, server_c = scaffold_server_original(
                x,
                server_c,
                model_deltas,
                control_deltas,
                total_client_count=2,
                server_learning_rate=1.0,
            )
            self.assertTrue(
                np.allclose(
                    [scalar_value(value) for value in local], local_expected, rtol=0.0, atol=1e-12
                )
            )
            self.assertTrue(
                np.allclose(
                    [scalar_value(value) for value in next_controls],
                    controls_expected,
                    rtol=0.0,
                    atol=1e-12,
                )
            )
            self.assertAlmostEqual(scalar_value(x), x_expected, places=12)
            self.assertAlmostEqual(scalar_value(server_c), c_expected, places=12)
            client_cs = next_controls

    def test_scaffold_batched_weighted_algorithm6_golden(self) -> None:
        x = scalar(2.0)
        server_c = scalar(0.0)
        client_cs = [scalar(0.0), scalar(0.0)]
        expected = [
            ([1.9, 1.7], [1.0, 3.0], 1.75, 2.5),
            ([1.525, 1.525], [0.75, 2.75], 1.525, 2.25),
        ]
        for local_expected, controls_expected, x_expected, c_expected in expected:
            local, next_controls, model_deltas, control_deltas = self._client_round(
                x, server_c, client_cs
            )
            x, server_c = scaffold_server_batched_weighted(
                x,
                server_c,
                model_deltas,
                control_deltas,
                [1, 3],
                selected_client_count=2,
                total_client_count=2,
                server_learning_rate=1.0,
            )
            self.assertTrue(
                np.allclose(
                    [scalar_value(value) for value in local], local_expected, rtol=0.0, atol=1e-12
                )
            )
            self.assertTrue(
                np.allclose(
                    [scalar_value(value) for value in next_controls],
                    controls_expected,
                    rtol=0.0,
                    atol=1e-12,
                )
            )
            self.assertAlmostEqual(scalar_value(x), x_expected, places=12)
            self.assertAlmostEqual(scalar_value(server_c), c_expected, places=12)
            client_cs = next_controls


class PublicControlBaselineTests(unittest.TestCase):

    def test_public_argmin_exact_tie_prefers_larger_step(self) -> None:
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        losses = [0.62, 0.55, 0.5, 0.5, 0.58]
        table = [{"alpha": alpha, "mean_log_loss": loss} for (alpha, loss) in zip(alphas, losses)]
        self.assertEqual(select_public_argmin(table), 0.75)

    def test_public_candidate_table_is_json_safe_and_rejects_invalid_probability(self) -> None:
        table = score_public_direction_grid(
            scalar(0.0),
            scalar(1.0),
            np.zeros((2, 1), dtype=np.float64),
            np.array([0, 1], dtype=np.float64),
            alphas=[0.0, 1.0],
            predict_fn=lambda _state, _features: np.array([0.2, 0.8]),
        )
        json.loads(json.dumps(table, allow_nan=False))
        with self.assertRaises(BaselineMathError):
            score_public_direction_grid(
                scalar(0.0),
                scalar(1.0),
                np.zeros((2, 1), dtype=np.float64),
                np.array([0, 1], dtype=np.float64),
                alphas=[0.0, 1.0],
                predict_fn=lambda _state, _features: np.array([-0.1, 1.1]),
            )
        with self.assertRaises(BaselineMathError):
            score_public_direction_grid(
                scalar(0.0),
                scalar(1.0),
                np.zeros((3, 1), dtype=np.float64),
                np.array([0, 1], dtype=np.float64),
                alphas=[0.0, 1.0],
                predict_fn=lambda _state, _features: np.array([0.2, 0.8]),
            )


class BaselineContractTests(unittest.TestCase):

    def test_counts_and_steps_must_be_exact_integers(self) -> None:
        with self.assertRaises(BaselineMathError):
            fedavg_server_average([scalar(1.0), scalar(2.0)], [1.9, 2])
        with self.assertRaises(BaselineMathError):
            fedavg_server_average([scalar(1.0), scalar(2.0)], [True, 2])
        with self.assertRaises(BaselineMathError):
            scaffold_option2_control(
                scalar(0.0),
                scalar(0.0),
                scalar(1.0),
                scalar(0.5),
                actual_local_steps=1.5,
                local_learning_rate=0.1,
            )

    def test_nonfinite_and_incompatible_tensors_fail_closed(self) -> None:
        with self.assertRaises(BaselineMathError):
            scaffold_option2_control(
                scalar(0.0),
                scalar(0.0),
                scalar(1.0),
                scalar(0.5),
                actual_local_steps=1,
                local_learning_rate=float("inf"),
            )
        with self.assertRaises(BaselineMathError):
            fedavg_server_average(
                [scalar(1.0), OrderedDict(x=torch.tensor([2], dtype=torch.int64))], [1, 1]
            )
        with self.assertRaises(BaselineMathError):
            fedavg_server_average(
                [scalar(1.0), OrderedDict(x=torch.tensor([2.0], dtype=torch.float32))], [1, 1]
            )

    def test_fedadam_rejects_negative_second_moment(self) -> None:
        invalid = FedAdamState(m=scalar(0.0), v=scalar(-1.0))
        with self.assertRaises(BaselineMathError):
            fedadam_server_step(
                scalar(0.0),
                scalar(0.1),
                invalid,
                beta1=0.9,
                beta2=0.99,
                server_learning_rate=0.1,
                tau=0.001,
            )

    def test_fedadam_rejects_invalid_beta_and_underflowed_initialization(self) -> None:
        with self.assertRaises(BaselineMathError):
            fedadam_server_step(
                scalar(0.0),
                scalar(0.1),
                FedAdamState(m=scalar(0.0), v=scalar(1e-06)),
                beta1="0.9",
                beta2=0.99,
                server_learning_rate=0.1,
                tau=0.001,
            )
        with self.assertRaises(BaselineMathError):
            fedadam_server_step(
                scalar(0.0),
                scalar(0.1),
                FedAdamState(m=scalar(0.0), v=scalar(1e-06)),
                beta1=True,
                beta2=0.99,
                server_learning_rate=0.1,
                tau=0.001,
            )
        with self.assertRaises(BaselineMathError):
            fedadam_init_like(OrderedDict(x=torch.tensor([0.0], dtype=torch.float16)), tau=1e-08)

    def test_fedyogi_rejects_negative_second_moment_and_invalid_beta(self) -> None:
        with self.assertRaises(BaselineMathError):
            fedyogi_server_step(
                scalar(0.0),
                scalar(0.1),
                FedYogiState(m=scalar(0.0), v=scalar(-1.0)),
                beta1=0.9,
                beta2=0.99,
                server_learning_rate=0.1,
                tau=0.001,
            )
        with self.assertRaises(BaselineMathError):
            fedyogi_server_step(
                scalar(0.0),
                scalar(0.1),
                FedYogiState(m=scalar(0.0), v=scalar(1e-06)),
                beta1=0.9,
                beta2=1.0,
                server_learning_rate=0.1,
                tau=0.001,
            )

    def test_non_tensor_first_state_fails_with_contract_error(self) -> None:
        with self.assertRaises(BaselineMathError):
            fedavg_server_average([OrderedDict(x=1.0), scalar(2.0)], [1, 1])


if __name__ == "__main__":
    unittest.main()
