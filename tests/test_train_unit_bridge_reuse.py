from __future__ import annotations
import os
import copy
import json
from pathlib import Path
from collections import OrderedDict
import torch.nn.functional as F
import platform
import unittest
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import fields, is_dataclass, replace
from unittest.mock import patch
import opacus
import torch
from fedsift import train_unit as train_unit_module
from fedsift.candidate_space import canonical_sha256
from fedsift.modeling import SelectedBCEGradientBridge
from fedsift.train_unit import (
    ExplicitClientBatchPlan,
    _cached_minimal_noise_calibration,
    execute_training_unit,
    validate_training_unit_result,
)
from tests import test_train_unit as train_unit_fixtures

_budget = train_unit_fixtures._budget
_method_capability = train_unit_fixtures._method_capability
_sealed_data = train_unit_fixtures._sealed_data
_STATIC_REPLAY = json.loads((Path(__file__).parent / "fixtures/training_replay.json").read_text())
REGISTERED_REPLAY_ENVIRONMENT = _STATIC_REPLAY["environment"]
REGISTERED_REPLAY_SEMANTIC_GOLDENS = _STATIC_REPLAY["cases"]


def _runtime_environment() -> dict[str, object]:
    variable_names = (
        "BLIS_NUM_THREADS",
        "CUDA_VISIBLE_DEVICES",
        "MKL_DYNAMIC",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_DYNAMIC",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PYTHONHASHSEED",
        "PYTHONDONTWRITEBYTECODE",
        "VECLIB_MAXIMUM_THREADS",
    )
    return {
        **{name: os.environ.get(name) for name in variable_names},
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_version": torch.__version__,
        "opacus_version": opacus.__version__,
        "python_version": platform.python_version(),
    }


def _jsonable(value):
    if is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
            if not field.name.startswith("_")
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for (key, item) in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _private_budget(data, capability, *, model_family: str, sample_rate: float):
    budget = _budget(data, capability)
    calibration = _cached_minimal_noise_calibration(
        sample_rate,
        budget.server_rounds,
        budget.dp_optimizer_steps_by_round[0],
        1.0,
        1.0,
        budget.target_epsilon,
        budget.target_delta,
    )
    return replace(
        budget,
        poisson_sample_rate=sample_rate,
        base_noise_multiplier=calibration.base_noise_multiplier,
        model_family=model_family,
    )


def _shuffled_nonprivate_budget(data, capability, *, model_family: str):
    rows = tuple(sorted(data.role("client_0").row_ids))
    plan = ExplicitClientBatchPlan("client_0", epochs=(((rows[2], rows[0]), (rows[3], rows[1])),))
    return replace(
        _budget(data, capability),
        nonprivate_batch_plans_by_round=((plan,), (plan,)),
        model_family=model_family,
    )


def _semantic_payload(result) -> dict[str, object]:
    return {
        "artifact": result.artifact,
        "round_resource_receipts": result.round_resource_receipts,
        "resource_summary": result.resource_summary,
        "local_step_receipts": [_jsonable(receipt) for receipt in result.local_step_receipts],
        "sequential_client_reports": [
            _jsonable(report) for report in result.sequential_client_reports
        ],
        "parallel_privacy_report": (
            None
            if result.parallel_privacy_report is None
            else _jsonable(result.parallel_privacy_report)
        ),
    }


def _semantic_evidence(result) -> dict[str, object]:
    payload = _semantic_payload(result)
    return {
        "semantic_sha256": canonical_sha256(payload),
        "artifact_sha256": result.artifact["artifact_sha256"],
        "final_model_state_sha256": result.artifact["final_model_state_sha256"],
        "round_evidence_sha256": canonical_sha256(result.artifact["round_evidence"]),
        "resource_summary_sha256": canonical_sha256(result.resource_summary),
        "local_step_receipts_sha256": canonical_sha256(payload["local_step_receipts"]),
        "sampled_record_counts": [
            receipt.sampled_record_count for receipt in result.local_step_receipts
        ],
    }


def _fresh_bridge_from_bound(bridge: SelectedBCEGradientBridge) -> SelectedBCEGradientBridge:
    """Test-only reconstruction of the uncached per-call gradient bridge."""
    return SelectedBCEGradientBridge(
        bridge.model, bridge._features, bridge._labels, bridge._row_ids
    )


def _cases():
    return {
        "private_logistic_empty": (
            "dp_fedavg",
            lambda data, capability: _private_budget(
                data, capability, model_family="logistic_screening", sample_rate=0.001
            ),
        ),
        "private_mlp_nonempty": (
            "dp_fedavg",
            lambda data, capability: _private_budget(
                data, capability, model_family="screening_mlp", sample_rate=1.0
            ),
        ),
        "nonprivate_logistic_shuffled": (
            "fedavg_nonprivate",
            lambda data, capability: _shuffled_nonprivate_budget(
                data, capability, model_family="logistic_screening"
            ),
        ),
        "nonprivate_mlp_shuffled": (
            "fedavg_nonprivate",
            lambda data, capability: _shuffled_nonprivate_budget(
                data, capability, model_family="screening_mlp"
            ),
        ),
    }


class TrainingUnitBridgeReuseTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        train_unit_fixtures.TrainingUnitIntegrationTests.setUpClass()
        cls.base = train_unit_fixtures.TrainingUnitIntegrationTests.base

    def _execute(self, method: str, budget_factory, *, uncached_fresh: bool = False):
        capability = _method_capability(self.base, method)
        data = _sealed_data(capability)
        budget = budget_factory(data, capability)
        uncached_fresh_count = 0

        def fresh_bridge(bound_bridge):
            nonlocal uncached_fresh_count
            uncached_fresh_count += 1
            return _fresh_bridge_from_bound(bound_bridge)

        original_nonprivate = train_unit_module._execute_nonprivate_client
        original_private = train_unit_module._execute_private_client
        with ExitStack() as stack:
            bridge_factory = stack.enter_context(
                patch(
                    "fedsift.train_unit.SelectedBCEGradientBridge", wraps=SelectedBCEGradientBridge
                )
            )
            if uncached_fresh:

                def uncached_nonprivate(**kwargs):
                    arguments = dict(kwargs)
                    arguments["bridge"] = fresh_bridge(arguments["bridge"])
                    return original_nonprivate(**arguments)

                def uncached_private(**kwargs):
                    arguments = dict(kwargs)
                    arguments["bridge"] = fresh_bridge(arguments["bridge"])
                    return original_private(**arguments)

                stack.enter_context(
                    patch(
                        "fedsift.train_unit._execute_nonprivate_client",
                        side_effect=uncached_nonprivate,
                    )
                )
                stack.enter_context(
                    patch("fedsift.train_unit._execute_private_client", side_effect=uncached_private)
                )
            result = execute_training_unit(
                capability,
                data,
                budget,
                expected_capability_sha256=capability["capability_sha256"],
                expected_budget_sha256=budget.budget_sha256,
            )
        validate_training_unit_result(
            result,
            capability,
            data,
            budget,
            expected_capability_sha256=capability["capability_sha256"],
            expected_budget_sha256=budget.budget_sha256,
        )
        return (result, bridge_factory.call_count, uncached_fresh_count)

    def test_reuse_matches_same_process_uncached_fresh_reference(self) -> None:
        for name, (method, budget_factory) in _cases().items():
            with self.subTest(case=name):
                uncached, uncached_cache_count, uncached_fresh_count = self._execute(
                    method, budget_factory, uncached_fresh=True
                )
                reuse, reuse_bridge_count, reuse_fresh_count = self._execute(method, budget_factory)
                self.assertEqual(_semantic_payload(reuse), _semantic_payload(uncached))
                self.assertEqual(_semantic_evidence(reuse), _semantic_evidence(uncached))
                self.assertEqual(uncached_cache_count, 1)
                self.assertEqual(uncached_fresh_count, 2)
                self.assertEqual(reuse_bridge_count, 1)
                self.assertEqual(reuse_fresh_count, 0)

    def test_training_matches_independent_per_record_autograd(self) -> None:
        def literal_gradients(model, state, features, labels, row_ids, selected_row_ids):
            model = copy.deepcopy(model)
            model.load_state_dict(state)
            names = tuple(state)
            parameters = tuple(model.parameters())
            rows = {name: [] for name in names}
            positions = {int(row): index for index, row in enumerate(row_ids)}
            for row_id in selected_row_ids:
                index = positions[int(row_id)]
                logit = model(features[index:index + 1])[0]
                loss = F.binary_cross_entropy_with_logits(logit, labels[index], reduction="sum")
                for name, gradient in zip(names, torch.autograd.grad(loss, parameters)):
                    rows[name].append(gradient.detach())
            return OrderedDict((name, torch.stack(rows[name]) if rows[name] else
                                value.new_empty((0, *value.shape))) for name, value in state.items())

        for name, (method, budget_factory) in _cases().items():
            with self.subTest(case=name):
                actual, _, _ = self._execute(method, budget_factory)
                with patch("fedsift.modeling.selected_per_record_bce_gradients", side_effect=literal_gradients):
                    reference, _, _ = self._execute(method, budget_factory)
                for parameter in actual.model_state:
                    torch.testing.assert_close(actual.model_state[parameter], reference.model_state[parameter],
                                               rtol=1e-12, atol=1e-12)
                self.assertEqual(actual.parallel_privacy_report, reference.parallel_privacy_report)
                self.assertEqual([r.sampled_record_count for r in actual.local_step_receipts],
                                 [r.sampled_record_count for r in reference.local_step_receipts])

    def test_registered_environment_replays_static_goldens(self) -> None:
        environment = _runtime_environment()
        self.assertEqual(environment, REGISTERED_REPLAY_ENVIRONMENT, "run tests with the packaged runtime")
        observed = {}
        for name, (method, budget_factory) in _cases().items():
            result, bridge_count, fresh_count = self._execute(method, budget_factory)
            observed[name] = _semantic_evidence(result)
            self.assertEqual(bridge_count, 1)
            self.assertEqual(fresh_count, 0)
        self.assertEqual(observed, REGISTERED_REPLAY_SEMANTIC_GOLDENS)

    def test_five_client_roster_builds_one_bridge_per_client(self) -> None:
        clients = tuple((f"client_{index}" for index in range(5)))

        def budget_factory(data, capability):
            budget = _private_budget(
                data, capability, model_family="logistic_screening", sample_rate=0.001
            )
            plans = tuple(
                (
                    ExplicitClientBatchPlan(
                        client_id, epochs=((tuple(data.role(client_id).row_ids[:4]),),)
                    )
                    for client_id in clients
                )
            )
            return replace(
                budget,
                nonprivate_batch_plans_by_round=(plans, plans),
                participating_clients_by_round=(clients, clients),
            )

        result, bridge_count, fresh_count = self._execute("dp_fedavg", budget_factory)
        self.assertEqual(bridge_count, len(clients))
        self.assertEqual(fresh_count, 0)
        self.assertEqual(
            tuple((report.client_id for report in result.sequential_client_reports)), clients
        )


if __name__ == "__main__":
    unittest.main()
