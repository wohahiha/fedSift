from __future__ import annotations
import os
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
REGISTERED_REPLAY_ENVIRONMENT = {
    "BLIS_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
    "MKL_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "torch_num_threads": 1,
    "torch_num_interop_threads": 16,
    "torch_version": "2.5.1+cpu",
    "opacus_version": "1.5.4",
    "python_version": "3.10.12",
}
REGISTERED_REPLAY_SEMANTIC_GOLDENS = {
    "private_logistic_empty": {
        "semantic_sha256": "15e2a2feea1183ba046fa7efefeac3694eb3ad0baf7feeb6a4b3251f49f7167e",
        "artifact_sha256": "fa07215da992447d5885a85a779cc8fe6094ca0cf52fe1acb25ee317a1f86df7",
        "final_model_state_sha256": "02a70e00b0b90297e044b17c47a4503611d5aad27c21297ae0294d0e92afbd3e",
        "round_evidence_sha256": "6f248ef46ab6fe9702a118fd9f6867f4e4fe1e68082a43022f7964726be820ca",
        "resource_summary_sha256": "6f33448c825fec001d3a433bba42f81e7fcc3e8b8a0fa3fbb9c536af0865014b",
        "local_step_receipts_sha256": "caa9fc792f4ac1df45f21b9c24328c9dddfd8abb9101b656e2bcb05bbcf6eb79",
        "sampled_record_counts": [0, 0],
    },
    "private_mlp_nonempty": {
        "semantic_sha256": "9a44fef8dd4760ba9b9973fa62d4f3fa3d5bf05dfb743401fb9c74852a3e55c2",
        "artifact_sha256": "ed6fd778bfc2521b61fdcb3e6bdace522f27062add52140bbd3ed004110e6bfe",
        "final_model_state_sha256": "901e921dedd6fcc9c241d64f15fe3a897d985c24c811205796fcef76bffaf02f",
        "round_evidence_sha256": "56f8ff228b08e2adee0d0c36cecef613ff1f1356ebec8f71e36908de853e5443",
        "resource_summary_sha256": "cc48821732763ef5ad4eaf4d7306476cc0d683e257fbad1fd3ae7df8c9b917da",
        "local_step_receipts_sha256": "e87fdad8ed789ad3c6807694bb5eb50199a1b6c99eaa8a2793a70596ffde0eea",
        "sampled_record_counts": [43, 43],
    },
    "nonprivate_logistic_shuffled": {
        "semantic_sha256": "c083bf869218780792a9a045fa27c221a05c0ac9a055d696fbbc6755e89cd918",
        "artifact_sha256": "3cc1e578816ff994ec13d8a15c712855b9eadeb1101e9127745cb11a412fbeb3",
        "final_model_state_sha256": "f6b5d9bc1c9a27ff98c8dc4ca35fa31d92dc017dccb1670908f1e0cbce0849e9",
        "round_evidence_sha256": "a77b88c7eb4a984a8c3fb6ec31d00fe2f90edcf73543f6e2b638f0ab3dba248a",
        "resource_summary_sha256": "21242dbc166ee45149668a8af2ac2a0f12ab5fcd0b851b15439714ead2c3b291",
        "local_step_receipts_sha256": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
        "sampled_record_counts": [],
    },
    "nonprivate_mlp_shuffled": {
        "semantic_sha256": "b981ee939d02223350135349525dfb3b2d8f754aab8ebfec61cb68e5669493f4",
        "artifact_sha256": "adc235cdc661b41e99ccabe6a6b5ffeb4d6a2a00d91c21c74d2a9f4e94cf719c",
        "final_model_state_sha256": "4274e43952b5381b36a1503d14f40ce588e2f21f98603e19c581a55d04a70b41",
        "round_evidence_sha256": "d5acd04e1b58fdc10ea1f0b43e4420ea8767796360dbce4e1cad6416c791149e",
        "resource_summary_sha256": "33c4f1dffd2f8f3a5ce462a5b92132c89899e0f0cd320c6b12cb0d1550e31d06",
        "local_step_receipts_sha256": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
        "sampled_record_counts": [],
    },
}


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
    """Test-only reconstruction of the pre-reuse per-helper bridge."""
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

    def _execute(self, method: str, budget_factory, *, legacy_fresh: bool = False):
        capability = _method_capability(self.base, method)
        data = _sealed_data(capability)
        budget = budget_factory(data, capability)
        legacy_fresh_count = 0

        def fresh_bridge(bound_bridge):
            nonlocal legacy_fresh_count
            legacy_fresh_count += 1
            return _fresh_bridge_from_bound(bound_bridge)

        original_nonprivate = train_unit_module._execute_nonprivate_client
        original_private = train_unit_module._execute_private_client
        with ExitStack() as stack:
            bridge_factory = stack.enter_context(
                patch(
                    "fedsift.train_unit.SelectedBCEGradientBridge", wraps=SelectedBCEGradientBridge
                )
            )
            if legacy_fresh:

                def legacy_nonprivate(**kwargs):
                    arguments = dict(kwargs)
                    arguments["bridge"] = fresh_bridge(arguments["bridge"])
                    return original_nonprivate(**arguments)

                def legacy_private(**kwargs):
                    arguments = dict(kwargs)
                    arguments["bridge"] = fresh_bridge(arguments["bridge"])
                    return original_private(**arguments)

                stack.enter_context(
                    patch(
                        "fedsift.train_unit._execute_nonprivate_client",
                        side_effect=legacy_nonprivate,
                    )
                )
                stack.enter_context(
                    patch("fedsift.train_unit._execute_private_client", side_effect=legacy_private)
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
        return (result, bridge_factory.call_count, legacy_fresh_count)

    def test_reuse_matches_same_process_legacy_fresh_reference(self) -> None:
        for name, (method, budget_factory) in _cases().items():
            with self.subTest(case=name):
                legacy, legacy_cache_count, legacy_fresh_count = self._execute(
                    method, budget_factory, legacy_fresh=True
                )
                reuse, reuse_bridge_count, reuse_fresh_count = self._execute(method, budget_factory)
                self.assertEqual(_semantic_payload(reuse), _semantic_payload(legacy))
                self.assertEqual(_semantic_evidence(reuse), _semantic_evidence(legacy))
                self.assertEqual(legacy_cache_count, 1)
                self.assertEqual(legacy_fresh_count, 2)
                self.assertEqual(reuse_bridge_count, 1)
                self.assertEqual(reuse_fresh_count, 0)

    def test_registered_environment_replays_static_goldens(self) -> None:
        environment = _runtime_environment()
        if environment != REGISTERED_REPLAY_ENVIRONMENT:
            self.skipTest(
                f"static replay golden is restricted to the registered single-thread environment, observed={environment!r}"
            )
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
