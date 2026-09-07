from __future__ import annotations
import copy
import importlib.util
import unittest
from contextlib import ExitStack
from unittest import mock
from fedsift.candidate_space import assert_no_performance_fields, canonical_sha256
from fedsift.resource_protocol import (
    ResourceProtocolBoundaryError,
    ResourceProtocolError,
    ResourceProtocolExecutionError,
    assert_resource_protocol_result_blind,
    build_resource_protocol_capability,
    build_resource_protocol_order_manifest,
    run_resource_protocol,
    validate_resource_protocol_artifact,
    validate_resource_protocol_capability,
)
from fedsift.resource_accounting import build_round_resource_receipt

MODEL_HASH = "c" * 64
METHODS = ("fedsift", "dp_scaffold_adapted", "dp_fedavg")
SYNTHETIC_INPUT = {
    "dataset_id": "synthetic_cost_smoke",
    "record_count": 24,
    "v_ctrl_record_count": 5,
    "control_candidate_count": 3,
    "model_payload_bytes": 128,
}
PUBLIC_CONTROL_METHODS = {
    "public_argmin_time_dpfedadam",
    "fedsift",
    "fedsift_uniform_schedule",
    "fedsift_public_argmin_rule",
}


def _rehash(value: dict[str, object], field: str) -> None:
    payload = copy.deepcopy(value)
    payload.pop(field, None)
    value[field] = canonical_sha256(payload)


class SyntheticResourceExecutor:

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **request: object) -> list[dict[str, object]]:
        self.calls.append(copy.deepcopy(request))
        self.assert_resource_only_request(request)
        method_id = str(request["method_id"])
        synthetic_input = request["synthetic_input"]
        assert isinstance(synthetic_input, dict)
        receipts = []
        for server_round in (1, 2):
            query = method_id in PUBLIC_CONTROL_METHODS and server_round == 1
            arguments: dict[str, object] = {
                "method_id": method_id,
                "server_round": server_round,
                "participating_client_ids": ("client_0", "client_1"),
                "total_client_count": 3,
                "model_payload_bytes": synthetic_input["model_payload_bytes"],
                "model_manifest_sha256": MODEL_HASH,
                "local_optimizer_steps": 4,
                "sampled_record_gradient_evaluations": 16,
                "public_control_query_executed": query,
                "public_control_record_count": (
                    synthetic_input["v_ctrl_record_count"] if query else 0
                ),
                "public_control_candidate_count": (
                    synthetic_input["control_candidate_count"] if query else 0
                ),
            }
            if method_id == "dp_scaffold_adapted":
                arguments["control_payload_bytes"] = synthetic_input["model_payload_bytes"]
            receipts.append(build_round_resource_receipt(**arguments))
        return receipts

    @staticmethod
    def assert_resource_only_request(request: dict[str, object]) -> None:
        if request["candidate_id"] != 0:
            raise AssertionError("nonzero candidate reached executor")
        forbidden = {
            "inner_validation",
            "outer_capability",
            "outer_gate",
            "outer_test_gate",
            "hpo_selection",
            "oof",
        }
        if forbidden.intersection(request):
            raise AssertionError("forbidden authority reached executor")


def _capability(
    *,
    methods: tuple[str, ...] = METHODS,
    synthetic_input: object = SYNTHETIC_INPUT,
    repetitions: int = 1,
) -> dict[str, object]:
    return build_resource_protocol_capability(
        method_ids=methods,
        synthetic_input=synthetic_input,
        expected_rounds=2,
        model_manifest_sha256=MODEL_HASH,
        candidate_id=0,
        warmup_passes=1,
        latin_square_repetitions=repetitions,
        order_seed="cost-smoke-test-seed",
    )


def _run(
    capability: dict[str, object], executor: object, *, synthetic_input: object = SYNTHETIC_INPUT
) -> dict[str, object]:
    return run_resource_protocol(
        capability,
        synthetic_input=synthetic_input,
        expected_capability_sha256=str(capability["capability_sha256"]),
        resource_executor=executor,
    )


def _validate(
    artifact: dict[str, object],
    capability: dict[str, object],
    *,
    synthetic_input: object = SYNTHETIC_INPUT,
    expected_artifact_sha256: str | None = None,
) -> None:
    validate_resource_protocol_artifact(
        artifact,
        capability,
        synthetic_input=synthetic_input,
        expected_capability_sha256=str(capability["capability_sha256"]),
        expected_artifact_sha256=(
            str(artifact["artifact_sha256"])
            if expected_artifact_sha256 is None
            else expected_artifact_sha256
        ),
    )


class ResourceProtocolResultBlindTests(unittest.TestCase):

    def test_trapped_performance_selection_oof_and_outer_apis_are_never_called(self) -> None:
        self.assertIsNone(importlib.util.find_spec("fedsift.oof_inference"))
        capability = _capability()
        executor = SyntheticResourceExecutor()
        trap = mock.Mock(side_effect=AssertionError("forbidden API called"))
        targets = (
            "fedsift.evaluation.compute_hpo_probability_metrics",
            "fedsift.evaluation.select_threshold_from_v_sel",
            "fedsift.evaluation.evaluate_outer_test",
            "fedsift.hpo_select.build_hpo_selection_decision",
            "fedsift.outer_refit.build_outer_refit_capability",
            "fedsift.outer_refit.issue_outer_test_gate",
            "fedsift.outer_refit.open_outer_test_once",
        )
        with ExitStack() as stack:
            for target in targets:
                stack.enter_context(mock.patch(target, new=trap))
            artifact = _run(capability, executor)
        trap.assert_not_called()
        self.assertTrue(executor.calls)
        self.assertEqual(executor.calls[0]["phase"], "warmup")
        self.assertEqual(executor.calls[len(METHODS)]["phase"], "measured")
        self.assertNotIn("inner_validation", repr(artifact))
        self.assertNotIn("outer_test", repr(artifact))
        self.assertNotIn("oof", repr(artifact).lower())
        assert_no_performance_fields(artifact)
        assert_resource_protocol_result_blind(artifact)
        _validate(artifact, capability)
        fedsift_records = [
            record for record in artifact["execution_ledger"] if record["method_id"] == "fedsift"
        ]
        self.assertTrue(fedsift_records)
        for record in fedsift_records:
            self.assertEqual(
                record["resource_summary"]["totals"]["public_candidate_forward_record_evaluations"],
                15,
            )

    def test_shared_performance_gate_is_invoked_and_malicious_fields_fail(self) -> None:
        capability = _capability()
        executor = SyntheticResourceExecutor()
        with mock.patch(
            "fedsift.resource_protocol.assert_no_performance_fields",
            wraps=assert_no_performance_fields,
        ) as shared_gate:
            _run(capability, executor)
        self.assertGreater(shared_gate.call_count, 0)
        clean = SyntheticResourceExecutor()
        for malicious_key in ("prediction", "selected_alpha", "control-table"):
            with self.subTest(malicious_key=malicious_key):

                def malicious_executor(**request: object) -> list[dict[str, object]]:
                    receipts = clean(**request)
                    receipts[0][malicious_key] = 0.5
                    return receipts

                with self.assertRaises(ResourceProtocolBoundaryError):
                    _run(capability, malicious_executor)

    def test_executor_summary_is_not_accepted_as_resource_evidence(self) -> None:
        capability = _capability()
        clean = SyntheticResourceExecutor()

        def self_reporting_executor(**request: object) -> dict[str, object]:
            return {
                "round_resource_receipts": clean(**request),
                "resource_summary": {"totals": {"round_total_federated_bytes": 0}},
            }

        with self.assertRaises(ResourceProtocolExecutionError):
            _run(capability, self_reporting_executor)


class ResourceProtocolCapabilityAndOrderTests(unittest.TestCase):

    def test_candidate_zero_is_exact_and_nonzero_is_rejected(self) -> None:
        capability = _capability()
        self.assertEqual(capability["candidate_id"], 0)
        self.assertNotIn("inner_validation", repr(capability))
        self.assertNotIn("outer", repr(capability))
        with self.assertRaises(ResourceProtocolBoundaryError):
            build_resource_protocol_capability(
                method_ids=METHODS,
                synthetic_input=SYNTHETIC_INPUT,
                expected_rounds=2,
                model_manifest_sha256=MODEL_HASH,
                candidate_id=1,
            )
        with self.assertRaises(ResourceProtocolError):
            build_resource_protocol_capability(
                method_ids=METHODS,
                synthetic_input=SYNTHETIC_INPUT,
                expected_rounds=2,
                model_manifest_sha256=MODEL_HASH,
                candidate_id=False,
            )
        forged = copy.deepcopy(capability)
        forged["candidate_id"] = 1
        _rehash(forged, "capability_sha256")
        with self.assertRaises(ResourceProtocolBoundaryError):
            validate_resource_protocol_capability(
                forged,
                synthetic_input=SYNTHETIC_INPUT,
                expected_capability_sha256=str(forged["capability_sha256"]),
            )

    def test_outer_capability_and_gate_are_rejected_without_executor_access(self) -> None:
        with self.assertRaises(ResourceProtocolBoundaryError):
            build_resource_protocol_capability(
                method_ids=METHODS,
                synthetic_input=SYNTHETIC_INPUT,
                expected_rounds=2,
                model_manifest_sha256=MODEL_HASH,
                outer_capability={},
            )
        capability = _capability()
        executor = mock.Mock(side_effect=AssertionError("executor must not run"))
        with self.assertRaises(ResourceProtocolBoundaryError):
            run_resource_protocol(
                capability,
                synthetic_input=SYNTHETIC_INPUT,
                expected_capability_sha256=str(capability["capability_sha256"]),
                resource_executor=executor,
                outer_test_gate=object(),
            )
        executor.assert_not_called()
        injected = copy.deepcopy(capability)
        injected["outer_capability"] = {}
        _rehash(injected, "capability_sha256")
        with self.assertRaises(ResourceProtocolBoundaryError):
            validate_resource_protocol_capability(
                injected,
                synthetic_input=SYNTHETIC_INPUT,
                expected_capability_sha256=str(injected["capability_sha256"]),
            )

    def test_order_is_deterministic_rotation_balanced_and_input_bound(self) -> None:
        first = _capability(methods=METHODS, repetitions=2)
        second = _capability(methods=tuple(reversed(METHODS)), repetitions=2)
        self.assertEqual(first, second)
        first_order = build_resource_protocol_order_manifest(
            first,
            synthetic_input=SYNTHETIC_INPUT,
            expected_capability_sha256=str(first["capability_sha256"]),
        )
        second_order = build_resource_protocol_order_manifest(
            second,
            synthetic_input=SYNTHETIC_INPUT,
            expected_capability_sha256=str(second["capability_sha256"]),
        )
        self.assertEqual(first_order, second_order)
        self.assertEqual(len(first_order["warmup_orders"]), 1)
        self.assertEqual(len(first_order["measured_orders"]), 6)
        for position in first_order["position_balance"]:
            self.assertEqual(set(position["method_counts"].values()), {2})
        measured = first_order["measured_orders"]
        for block_index in (1, 2):
            block = [row["method_ids"] for row in measured if row["block_index"] == block_index]
            for position in range(len(METHODS)):
                self.assertEqual({row[position] for row in block}, set(METHODS))
        changed_input = dict(SYNTHETIC_INPUT)
        changed_input["record_count"] = 25
        with self.assertRaisesRegex(ResourceProtocolError, "input differs"):
            validate_resource_protocol_capability(
                first,
                synthetic_input=changed_input,
                expected_capability_sha256=str(first["capability_sha256"]),
            )


class ResourceProtocolTamperTests(unittest.TestCase):

    def setUp(self) -> None:
        self.capability = _capability()
        self.artifact = _run(self.capability, SyntheticResourceExecutor())

    def test_order_manifest_tamper_fails_even_after_attacker_rehashes(self) -> None:
        tampered = copy.deepcopy(self.artifact)
        row = tampered["order_manifest"]["measured_orders"][0]["method_ids"]
        row[0], row[1] = (row[1], row[0])
        _rehash(tampered["order_manifest"], "order_manifest_sha256")
        _rehash(tampered, "artifact_sha256")
        with self.assertRaisesRegex(ResourceProtocolError, "deterministic replay"):
            _validate(tampered, self.capability)

    def test_round_receipt_tamper_fails_exact_resource_reconstruction(self) -> None:
        tampered = copy.deepcopy(self.artifact)
        tampered["execution_ledger"][0]["round_resource_receipts"][0]["round_uplink_bytes"] += 1
        tampered["execution_ledger_sha256"] = canonical_sha256(tampered["execution_ledger"])
        _rehash(tampered, "artifact_sha256")
        with self.assertRaisesRegex(ResourceProtocolError, "exact reconstruction"):
            _validate(tampered, self.capability)

    def test_external_artifact_commitment_rejects_fully_rehashed_change(self) -> None:
        original_hash = str(self.artifact["artifact_sha256"])
        tampered = copy.deepcopy(self.artifact)
        tampered["status"] = "complete_result_blind"
        tampered["execution_ledger"][0]["resource_summary"]["totals"]["local_optimizer_steps"] += 1
        tampered["execution_ledger_sha256"] = canonical_sha256(tampered["execution_ledger"])
        _rehash(tampered, "artifact_sha256")
        with self.assertRaisesRegex(ResourceProtocolError, "external commitment"):
            _validate(tampered, self.capability, expected_artifact_sha256=original_hash)


if __name__ == "__main__":
    unittest.main()
