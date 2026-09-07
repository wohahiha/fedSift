from __future__ import annotations
import copy
import io
import unittest
from collections import OrderedDict
import torch
from fedsift.sofim_math import (
    SOFIM_METHOD_ID,
    SOFIM_PROXY_INTERPRETATION,
    SofimMathError,
    SofimState,
    sofim_aggregate_normalized_client_proxy,
    sofim_aggregation_plan_sha256,
    sofim_deserialize_state,
    sofim_init_like,
    sofim_serialize_state,
    sofim_server_step_from_proxy,
    sofim_server_step_with_receipt,
    sofim_state_sha256,
    sofim_tensor_mapping_sha256,
    validate_sofim_normalized_proxy_receipt,
    validate_sofim_step_receipt,
)


def _state(
    a: list[float],
    b: float,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        weight=torch.tensor(a, dtype=dtype, device=device, requires_grad=requires_grad),
        bias=torch.tensor(b, dtype=dtype, device=device, requires_grad=requires_grad),
    )


def _flatten(state: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([state["weight"].reshape(-1), state["bias"].reshape(-1)])


def _one_client_proxy(
    gradient: OrderedDict[str, torch.Tensor], *, local_learning_rate: float
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, object]]:
    delta = OrderedDict(
        ((name, -local_learning_rate * value) for (name, value) in gradient.items())
    )
    return sofim_aggregate_normalized_client_proxy(
        expected_client_order=("client-a",),
        client_deltas=OrderedDict([("client-a", delta)]),
        actual_local_steps=OrderedDict([("client-a", 1)]),
        local_learning_rates=OrderedDict([("client-a", local_learning_rate)]),
        aggregation_weights=OrderedDict([("client-a", 1.0)]),
    )


class SofimNormalizedProxyTests(unittest.TestCase):

    def test_normalized_client_proxy_aggregation_golden(self):
        client_deltas = OrderedDict(
            [("client-a", _state([-0.2, 0.1], -0.04)), ("client-b", _state([-0.6, -0.3], 0.12))]
        )
        steps = OrderedDict([("client-a", 2), ("client-b", 3)])
        learning_rates = OrderedDict([("client-a", 0.1), ("client-b", 0.2)])
        weights = OrderedDict([("client-a", 0.25), ("client-b", 0.75)])
        proxy, receipt = sofim_aggregate_normalized_client_proxy(
            expected_client_order=("client-a", "client-b"),
            client_deltas=client_deltas,
            actual_local_steps=steps,
            local_learning_rates=learning_rates,
            aggregation_weights=weights,
        )
        torch.testing.assert_close(
            _flatten(proxy),
            torch.tensor([1.0, 0.25, -0.1], dtype=torch.float64),
            rtol=0.0,
            atol=1e-15,
        )
        self.assertEqual(receipt["method_id"], SOFIM_METHOD_ID)
        self.assertEqual(receipt["interpretation_boundary"], SOFIM_PROXY_INTERPRETATION)
        self.assertEqual(
            receipt["aggregation_plan_sha256"],
            sofim_aggregation_plan_sha256(("client-a", "client-b"), weights),
        )
        validate_sofim_normalized_proxy_receipt(
            receipt, expected_receipt_sha256=str(receipt["receipt_sha256"])
        )

    def test_client_order_runtime_values_and_receipt_tampering_fail(self):
        client_deltas = OrderedDict(
            [("client-a", _state([-0.1, 0.0], 0.0)), ("client-b", _state([-0.2, 0.0], 0.0))]
        )
        steps = OrderedDict([("client-a", 1), ("client-b", 1)])
        learning_rates = OrderedDict([("client-a", 0.1), ("client-b", 0.1)])
        weights = OrderedDict([("client-a", 0.5), ("client-b", 0.5)])
        _, receipt = sofim_aggregate_normalized_client_proxy(
            expected_client_order=("client-a", "client-b"),
            client_deltas=client_deltas,
            actual_local_steps=steps,
            local_learning_rates=learning_rates,
            aggregation_weights=weights,
        )
        with self.assertRaises(SofimMathError):
            sofim_aggregate_normalized_client_proxy(
                expected_client_order=("client-a", "client-b"),
                client_deltas=OrderedDict(reversed(tuple(client_deltas.items()))),
                actual_local_steps=steps,
                local_learning_rates=learning_rates,
                aggregation_weights=weights,
            )
        invalid_steps = OrderedDict(steps)
        invalid_steps["client-a"] = 0
        with self.assertRaises(SofimMathError):
            sofim_aggregate_normalized_client_proxy(
                expected_client_order=("client-a", "client-b"),
                client_deltas=client_deltas,
                actual_local_steps=invalid_steps,
                local_learning_rates=learning_rates,
                aggregation_weights=weights,
            )
        invalid_lrs = OrderedDict(learning_rates)
        invalid_lrs["client-a"] = 0.0
        with self.assertRaises(SofimMathError):
            sofim_aggregate_normalized_client_proxy(
                expected_client_order=("client-a", "client-b"),
                client_deltas=client_deltas,
                actual_local_steps=steps,
                local_learning_rates=invalid_lrs,
                aggregation_weights=weights,
            )
        invalid_weights = OrderedDict([("client-a", 0.6), ("client-b", 0.3)])
        with self.assertRaises(SofimMathError):
            sofim_aggregate_normalized_client_proxy(
                expected_client_order=("client-a", "client-b"),
                client_deltas=client_deltas,
                actual_local_steps=steps,
                local_learning_rates=learning_rates,
                aggregation_weights=invalid_weights,
            )
        tampered = copy.deepcopy(receipt)
        tampered["clients"][0]["actual_local_steps"] = 99
        with self.assertRaises(SofimMathError):
            validate_sofim_normalized_proxy_receipt(
                tampered, expected_receipt_sha256=str(receipt["receipt_sha256"])
            )


class SofimKernelGoldenTests(unittest.TestCase):

    def test_hard_coded_two_round_trace(self):
        beta = 0.73
        rho = 0.41
        server_lr = 0.17
        model = _state([0.2, -0.3], 0.7)
        state = sofim_init_like(model)
        proxies = (_state([0.11, -0.09], 0.04), _state([-0.03, 0.08], -0.12))
        expected = (
            (
                [0.0297, -0.024300000000000002, 0.0108],
                [0.2672567566273966, -0.21866461905877904, 0.09718427513723514],
                [0.15456635137334257, -0.2628270147600075, 0.68347867322667],
            ),
            (
                [0.013581, 0.003861, -0.024515999999999996],
                [-0.07340000339472351, 0.1950567706006273, -0.29226905268977516],
                [0.16704435195044556, -0.2959866657621142, 0.7331644121839318],
            ),
        )
        for proxy, (expected_m, expected_direction, expected_model) in zip(proxies, expected):
            model, state, direction = sofim_server_step_from_proxy(
                model, proxy, state, beta=beta, rho=rho, server_learning_rate=server_lr
            )
            torch.testing.assert_close(
                _flatten(state.moment),
                torch.tensor(expected_m, dtype=torch.float64),
                rtol=1e-14,
                atol=1e-14,
            )
            torch.testing.assert_close(
                _flatten(direction),
                torch.tensor(expected_direction, dtype=torch.float64),
                rtol=1e-14,
                atol=1e-14,
            )
            torch.testing.assert_close(
                _flatten(model),
                torch.tensor(expected_model, dtype=torch.float64),
                rtol=1e-14,
                atol=1e-14,
            )

    def test_delta_equals_negative_gradient_has_paper_sign(self):
        gradient = _state([0.3, -0.2], 0.1)
        proxy, _ = _one_client_proxy(gradient, local_learning_rate=1.0)
        base = _state([1.0, -2.0], 0.5)
        beta = 0.8
        rho = 0.7
        eta = 0.13
        model, next_state, direction = sofim_server_step_from_proxy(
            base, proxy, sofim_init_like(base), beta=beta, rho=rho, server_learning_rate=eta
        )
        gradient_vector = _flatten(gradient)
        expected_m = (1.0 - beta) * gradient_vector
        dense = rho * torch.eye(3, dtype=torch.float64) + torch.outer(expected_m, expected_m)
        expected_direction = torch.linalg.solve(dense, gradient_vector)
        expected_model = _flatten(base) - eta * expected_direction
        torch.testing.assert_close(_flatten(proxy), gradient_vector)
        torch.testing.assert_close(_flatten(next_state.moment), expected_m)
        torch.testing.assert_close(_flatten(direction), expected_direction)
        torch.testing.assert_close(_flatten(model), expected_model)

    def test_delta_gamma_normalization_matches_raw_unit_rescaling(self):
        gamma = 0.07
        beta = 0.6
        rho_gradient = 0.9
        eta_gradient = 0.2
        gradient = _state([0.4, -0.1], 0.25)
        proxy, _ = _one_client_proxy(gradient, local_learning_rate=gamma)
        base = _state([0.2, -0.4], 0.8)
        normalized_model, _, _ = sofim_server_step_from_proxy(
            base,
            proxy,
            sofim_init_like(base),
            beta=beta,
            rho=rho_gradient,
            server_learning_rate=eta_gradient,
        )
        gradient_vector = _flatten(gradient)
        gradient_moment = (1.0 - beta) * gradient_vector
        paper_direction = torch.linalg.solve(
            rho_gradient * torch.eye(3, dtype=torch.float64)
            + torch.outer(gradient_moment, gradient_moment),
            gradient_vector,
        )
        paper_model = _flatten(base) - eta_gradient * paper_direction
        delta_vector = -gamma * gradient_vector
        delta_moment = -gamma * gradient_moment
        rho_delta = gamma * gamma * rho_gradient
        eta_delta = gamma * eta_gradient
        raw_delta_direction = torch.linalg.solve(
            rho_delta * torch.eye(3, dtype=torch.float64) + torch.outer(delta_moment, delta_moment),
            delta_vector,
        )
        rescaled_raw_model = _flatten(base) + eta_delta * raw_delta_direction
        torch.testing.assert_close(_flatten(proxy), gradient_vector)
        torch.testing.assert_close(_flatten(normalized_model), paper_model)
        torch.testing.assert_close(rescaled_raw_model, paper_model)

    def test_float32_extreme_rho_and_parallel_cancellation_are_stable(self):
        zero = OrderedDict(weight=torch.zeros(1, dtype=torch.float32))
        model, _, direction = sofim_server_step_from_proxy(
            zero, zero, sofim_init_like(zero), beta=0.9, rho=1e-30, server_learning_rate=0.1
        )
        self.assertEqual(direction["weight"].item(), 0.0)
        self.assertEqual(model["weight"].item(), 0.0)
        base = OrderedDict(weight=torch.zeros(1, dtype=torch.float32))
        prior = SofimState(OrderedDict(weight=torch.tensor([9999.0], dtype=torch.float32)))
        proxy = OrderedDict(weight=torch.tensor([1.0], dtype=torch.float32))
        _, next_state, direction = sofim_server_step_from_proxy(
            base, proxy, prior, beta=0.9999, rho=0.0001, server_learning_rate=0.1
        )
        m = next_state.moment["weight"].double().item()
        expected = 1.0 / (0.0001 + m * m)
        self.assertGreater(direction["weight"].item(), 0.0)
        self.assertAlmostEqual(direction["weight"].item(), expected, delta=abs(expected) * 1e-05)

    def test_zero_proxy_does_not_move_model_and_only_decays_momentum(self):
        base = _state([1.0, -1.0], 2.0)
        prior = SofimState(_state([0.4, -0.2], 0.1))
        model, next_state, direction = sofim_server_step_from_proxy(
            base, _state([0.0, 0.0], 0.0), prior, beta=0.8, rho=1.0, server_learning_rate=0.2
        )
        torch.testing.assert_close(_flatten(model), _flatten(base))
        torch.testing.assert_close(_flatten(direction), torch.zeros(3, dtype=torch.float64))
        torch.testing.assert_close(_flatten(next_state.moment), 0.8 * _flatten(prior.moment))


class SofimStateAndReceiptTests(unittest.TestCase):

    def test_no_grad_no_alias_and_state_access_isolation(self):
        base = _state([1.0, -2.0], 0.5, requires_grad=True)
        proxy = _state([0.2, -0.1], 0.05, requires_grad=True)
        base_before = _flatten(base).detach().clone()
        proxy_before = _flatten(proxy).detach().clone()
        model, state, direction = sofim_server_step_from_proxy(
            base, proxy, sofim_init_like(base), beta=0.8, rho=1.0, server_learning_rate=0.2
        )
        for mapping in (model, direction, state.moment):
            self.assertTrue(all((not value.requires_grad for value in mapping.values())))
            self.assertTrue(all((value.grad_fn is None for value in mapping.values())))
        self.assertNotEqual(model["weight"].data_ptr(), base["weight"].data_ptr())
        self.assertNotEqual(direction["weight"].data_ptr(), proxy["weight"].data_ptr())
        self.assertNotEqual(model["weight"].data_ptr(), direction["weight"].data_ptr())
        self.assertNotEqual(state.moment["weight"].data_ptr(), direction["weight"].data_ptr())
        torch.testing.assert_close(_flatten(base).detach(), base_before)
        torch.testing.assert_close(_flatten(proxy).detach(), proxy_before)
        state_hash = sofim_state_sha256(state)
        exposed = state.moment
        exposed["weight"].add_(1000.0)
        exposed["new"] = torch.tensor(1.0)
        self.assertEqual(sofim_state_sha256(state), state_hash)
        self.assertNotIn("new", state.moment)
        with self.assertRaises(AttributeError):
            getattr(state, "_moment_items")

    def test_weights_only_safe_state_round_trip_replay_and_tamper(self):
        base = _state([0.1, 0.2], -0.3)
        state = SofimState(_state([0.4, -0.2], 0.1))
        payload = sofim_serialize_state(state)
        self.assertIs(type(payload), dict)
        self.assertIs(type(payload["moment"]), dict)
        with self.assertRaises(TypeError):
            torch.save(state, io.BytesIO())
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, weights_only=True, map_location="cpu")
        committed_state_hash = str(payload["state_sha256"])
        restored = sofim_deserialize_state(loaded, expected_state_sha256=committed_state_hash)
        self.assertEqual(sofim_state_sha256(restored), sofim_state_sha256(state))
        proxy = _state([0.05, -0.07], 0.02)
        first = sofim_server_step_from_proxy(
            base, proxy, state, beta=0.9, rho=0.8, server_learning_rate=0.1
        )
        replay = sofim_server_step_from_proxy(
            base, proxy, restored, beta=0.9, rho=0.8, server_learning_rate=0.1
        )
        torch.testing.assert_close(_flatten(first[0]), _flatten(replay[0]))
        torch.testing.assert_close(_flatten(first[2]), _flatten(replay[2]))
        self.assertEqual(sofim_state_sha256(first[1]), sofim_state_sha256(replay[1]))
        tampered = copy.deepcopy(loaded)
        tampered["moment"]["weight"][0].add_(1.0)
        with self.assertRaises(SofimMathError):
            sofim_deserialize_state(tampered, expected_state_sha256=committed_state_hash)
        tampered_state = SofimState(tampered["moment"])
        tampered["state_sha256"] = sofim_state_sha256(tampered_state)
        with self.assertRaises(SofimMathError):
            sofim_deserialize_state(tampered, expected_state_sha256=committed_state_hash)

    def test_step_receipt_binds_inputs_outputs_and_detects_tamper(self):
        gradient = _state([0.1, -0.2], 0.05)
        proxy, proxy_receipt = _one_client_proxy(gradient, local_learning_rate=0.1)
        base = _state([1.0, 2.0], -0.5)
        initial_state = sofim_init_like(base)
        output, next_state, direction, receipt = sofim_server_step_with_receipt(
            base,
            proxy,
            initial_state,
            beta=0.9,
            rho=0.7,
            server_learning_rate=0.15,
            normalized_proxy_receipt=proxy_receipt,
            expected_normalized_proxy_receipt_sha256=str(proxy_receipt["receipt_sha256"]),
        )
        validate_sofim_step_receipt(
            receipt,
            expected_receipt_sha256=str(receipt["receipt_sha256"]),
            normalized_proxy_receipt=proxy_receipt,
            expected_normalized_proxy_receipt_sha256=str(proxy_receipt["receipt_sha256"]),
        )
        self.assertEqual(
            receipt["input_model_state_sha256"],
            sofim_tensor_mapping_sha256(base, role="sofim_input_model"),
        )
        self.assertEqual(receipt["input_optimizer_state_sha256"], sofim_state_sha256(initial_state))
        self.assertEqual(
            receipt["aggregate_proxy_sha256"],
            sofim_tensor_mapping_sha256(proxy, role="aggregate_normalized_proxy"),
        )
        self.assertEqual(
            receipt["direction_sha256"],
            sofim_tensor_mapping_sha256(direction, role="sofim_preconditioned_direction"),
        )
        self.assertEqual(
            receipt["output_model_state_sha256"],
            sofim_tensor_mapping_sha256(output, role="sofim_output_model"),
        )
        self.assertEqual(receipt["output_optimizer_state_sha256"], sofim_state_sha256(next_state))
        tampered = copy.deepcopy(receipt)
        tampered["direction_sha256"] = "0" * 64
        with self.assertRaises(SofimMathError):
            validate_sofim_step_receipt(
                tampered,
                expected_receipt_sha256=str(receipt["receipt_sha256"]),
                normalized_proxy_receipt=proxy_receipt,
                expected_normalized_proxy_receipt_sha256=str(proxy_receipt["receipt_sha256"]),
            )
        other_proxy, other_receipt = _one_client_proxy(
            _state([0.2, 0.1], -0.05), local_learning_rate=0.1
        )
        self.assertFalse(torch.equal(_flatten(other_proxy), _flatten(proxy)))
        with self.assertRaises(SofimMathError):
            sofim_server_step_with_receipt(
                base,
                proxy,
                initial_state,
                beta=0.9,
                rho=0.7,
                server_learning_rate=0.15,
                normalized_proxy_receipt=other_receipt,
                expected_normalized_proxy_receipt_sha256=str(other_receipt["receipt_sha256"]),
            )
        with self.assertRaises(SofimMathError):
            sofim_server_step_with_receipt(
                base,
                proxy,
                initial_state,
                beta=0.9,
                rho=0.7,
                server_learning_rate=0.15,
                normalized_proxy_receipt=proxy_receipt,
                expected_normalized_proxy_receipt_sha256="0" * 64,
            )


class SofimContractTests(unittest.TestCase):

    def test_supported_dtypes_are_preserved(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=str(dtype)):
                base = _state([0.5, -0.25], 0.1, dtype=dtype)
                proxy = _state([0.1, -0.05], 0.02, dtype=dtype)
                model, state, direction = sofim_server_step_from_proxy(
                    base, proxy, sofim_init_like(base), beta=0.8, rho=1.0, server_learning_rate=0.1
                )
                for mapping in (model, state.moment, direction):
                    self.assertTrue(all((value.dtype == dtype for value in mapping.values())))

    def test_invalid_mapping_keys_dtype_device_and_structure_fail_closed(self):
        base = _state([1.0, 2.0], 0.0)
        proxy = _state([0.1, 0.2], 0.0)
        state = sofim_init_like(base)
        for invalid_base in (None, [], OrderedDict()):
            with self.subTest(invalid_base=type(invalid_base).__name__):
                with self.assertRaises(SofimMathError):
                    sofim_server_step_from_proxy(
                        invalid_base, proxy, state, beta=0.9, rho=1.0, server_learning_rate=0.1
                    )
        with self.assertRaises(SofimMathError):
            sofim_server_step_from_proxy(
                base, None, state, beta=0.9, rho=1.0, server_learning_rate=0.1
            )
        with self.assertRaises(SofimMathError):
            sofim_server_step_from_proxy(
                base, proxy, OrderedDict(), beta=0.9, rho=1.0, server_learning_rate=0.1
            )
        with self.assertRaises(SofimMathError):
            SofimState(None)
        with self.assertRaises(SofimMathError):
            sofim_init_like(OrderedDict(((1, torch.tensor([1.0])),)))
        mixed_dtype = OrderedDict(
            weight=torch.tensor([1.0], dtype=torch.float64),
            bias=torch.tensor(0.0, dtype=torch.float32),
        )
        with self.assertRaises(SofimMathError):
            sofim_init_like(mixed_dtype)
        if torch.cuda.is_available():
            mixed_device = OrderedDict(
                weight=torch.tensor([1.0], dtype=torch.float64),
                bias=torch.tensor(0.0, dtype=torch.float64, device="cuda"),
            )
            with self.assertRaises(SofimMathError):
                sofim_init_like(mixed_device)
        else:
            meta = OrderedDict(weight=torch.empty(1, dtype=torch.float64, device="meta"))
            with self.assertRaises(SofimMathError):
                sofim_init_like(meta)
        wrong_order = OrderedDict(bias=proxy["bias"], weight=proxy["weight"])
        with self.assertRaises(SofimMathError):
            sofim_server_step_from_proxy(
                base, wrong_order, state, beta=0.9, rho=1.0, server_learning_rate=0.1
            )
        wrong_shape = _state([0.1], 0.0)
        with self.assertRaises(SofimMathError):
            sofim_server_step_from_proxy(
                base, wrong_shape, state, beta=0.9, rho=1.0, server_learning_rate=0.1
            )

    def test_invalid_hyperparameters_and_nonfinite_values_fail_closed(self):
        base = _state([1.0, 2.0], 0.0)
        proxy = _state([0.1, 0.2], 0.0)
        state = sofim_init_like(base)
        invalid = (
            {"beta": -0.1, "rho": 1.0, "server_learning_rate": 1.0},
            {"beta": 1.0, "rho": 1.0, "server_learning_rate": 1.0},
            {"beta": 0.9, "rho": 0.0, "server_learning_rate": 1.0},
            {"beta": 0.9, "rho": 1.0, "server_learning_rate": float("nan")},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(SofimMathError):
                    sofim_server_step_from_proxy(base, proxy, state, **kwargs)
        nonfinite = _state([float("inf"), 0.0], 0.0)
        with self.assertRaises(SofimMathError):
            sofim_server_step_from_proxy(
                base, nonfinite, state, beta=0.9, rho=1.0, server_learning_rate=0.1
            )


if __name__ == "__main__":
    unittest.main()
