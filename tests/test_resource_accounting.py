from __future__ import annotations
import copy
import unittest
from fedsift.resource_accounting import (
    ResourceAccountingError,
    aggregate_resource_receipts,
    build_round_resource_receipt,
    validate_equal_information_control_resources,
)

MODEL_HASH = "a" * 64


def _round(method: str, round_number: int, **overrides):
    arguments = {
        "method_id": method,
        "server_round": round_number,
        "participating_client_ids": ("client_0", "client_1", "client_2"),
        "total_client_count": 5,
        "model_payload_bytes": 100,
        "model_manifest_sha256": MODEL_HASH,
        "local_optimizer_steps": 3,
        "sampled_record_gradient_evaluations": 17,
    }
    if method == "dp_scaffold_adapted":
        arguments["control_payload_bytes"] = 100
    if method in {
        "fedsift",
        "public_argmin_time_dpfedadam",
        "fedsift_uniform_schedule",
        "fedsift_public_argmin_rule",
    }:
        arguments["public_control_query_executed"] = True
        arguments["public_control_record_count"] = 20
        arguments["public_control_candidate_count"] = 5
    arguments.update(overrides)
    return build_round_resource_receipt(**arguments)


def _summary(method: str):
    receipts = [_round(method, 1), _round(method, 2)]
    return aggregate_resource_receipts(
        receipts,
        expected_method_id=method,
        expected_rounds=2,
        expected_model_manifest_sha256=MODEL_HASH,
    )


class ResourceAccountingGoldenTests(unittest.TestCase):

    def test_fedavg_like_and_scaffold_payload_oracles(self) -> None:
        fedavg = _round("dp_fedavg", 1)
        self.assertEqual(fedavg["round_downlink_bytes"], 300)
        self.assertEqual(fedavg["round_uplink_bytes"], 300)
        self.assertEqual(fedavg["round_total_federated_bytes"], 600)
        scaffold = _round("dp_scaffold_adapted", 1)
        self.assertEqual(scaffold["round_downlink_bytes"], 600)
        self.assertEqual(scaffold["round_uplink_bytes"], 600)
        self.assertEqual(scaffold["round_total_federated_bytes"], 1200)

    def test_public_control_work_is_explicit_and_pair_is_equal(self) -> None:
        fedsift = _summary("fedsift")
        argmin = _summary("fedsift_public_argmin_rule")
        self.assertEqual(fedsift["totals"]["public_candidate_forward_record_evaluations"], 200)
        check = validate_equal_information_control_resources(
            fedsift,
            argmin,
            expected_fedsift_summary_sha256=fedsift["report_sha256"],
            expected_public_argmin_summary_sha256=argmin["report_sha256"],
        )
        self.assertTrue(check["equal_network_and_public_control_resources"])
        self.assertEqual(
            check["comparison_identity"], "fedsift_vs_inherited_fedsift_public_argmin_rule"
        )

    def test_independently_tuned_public_argmin_is_not_the_exact_matched_arm(self) -> None:
        fedsift = _summary("fedsift")
        independent_argmin = _summary("public_argmin_time_dpfedadam")
        with self.assertRaises(ResourceAccountingError):
            validate_equal_information_control_resources(
                fedsift,
                independent_argmin,
                expected_fedsift_summary_sha256=fedsift["report_sha256"],
                expected_public_argmin_summary_sha256=independent_argmin["report_sha256"],
            )

    def test_server_persistent_state_is_not_misreported_as_network_payload(self) -> None:
        fedavg = _round("dp_fedavg", 1)
        self.assertEqual(fedavg["server_persistent_full_model_state_vector_count"], 0)
        self.assertEqual(fedavg["server_persistent_full_model_state_bytes"], 0)
        receipt = _round("dp_fedadam", 1)
        self.assertTrue(receipt["server_persistent_state_bytes_excluded_from_network"])
        self.assertEqual(receipt["server_persistent_full_model_state_vector_count"], 2)
        self.assertEqual(receipt["server_persistent_full_model_state_bytes"], 200)
        self.assertFalse(receipt["wall_clock_seconds_present"])
        scaffold = _round("dp_scaffold_adapted", 1)
        self.assertEqual(scaffold["server_persistent_full_model_state_vector_count"], 1)
        self.assertEqual(scaffold["server_persistent_full_model_state_bytes"], 100)
        sofim = _round("dp_fedsofim_delta_proxy_adapted", 1)
        self.assertEqual(sofim["round_total_federated_bytes"], 600)
        self.assertTrue(sofim["server_persistent_state_bytes_excluded_from_network"])
        self.assertEqual(sofim["server_persistent_full_model_state_vector_count"], 1)
        self.assertEqual(sofim["server_persistent_full_model_state_bytes"], 100)
        self.assertEqual(sofim["server_global_vector_inner_products"], 2)
        sofim_summary = _summary("dp_fedsofim_delta_proxy_adapted")
        self.assertEqual(sofim_summary["totals"]["server_global_vector_inner_products"], 4)
        self.assertEqual(
            sofim_summary["server_persistent_state_profile"]["peak_full_model_state_bytes"], 100
        )

    def test_public_control_method_can_have_explicit_non_query_round(self) -> None:
        receipt = _round(
            "fedsift",
            1,
            public_control_query_executed=False,
            public_control_record_count=0,
            public_control_candidate_count=0,
        )
        self.assertFalse(receipt["public_control_query_executed"])
        self.assertEqual(receipt["public_candidate_forward_record_evaluations"], 0)


class ResourceAccountingFailClosedTests(unittest.TestCase):

    def test_hidden_control_payload_and_unreported_public_query_fail(self) -> None:
        with self.assertRaises(ResourceAccountingError):
            _round("dp_fedavg", 1, control_payload_bytes=100)
        with self.assertRaises(ResourceAccountingError):
            _round("fedsift", 1, public_control_record_count=0)
        with self.assertRaises(ResourceAccountingError):
            _round("dp_fedavg", 1, public_control_record_count=20, public_control_candidate_count=5)
        with self.assertRaises(ResourceAccountingError):
            _round("dp_scaffold_adapted", 1, control_payload_bytes=99)

    def test_missing_reordered_or_tampered_round_receipt_fails(self) -> None:
        first = _round("dp_fedavg", 1)
        second = _round("dp_fedavg", 2)
        with self.assertRaises(ResourceAccountingError):
            aggregate_resource_receipts(
                [first],
                expected_method_id="dp_fedavg",
                expected_rounds=2,
                expected_model_manifest_sha256=MODEL_HASH,
            )
        with self.assertRaises(ResourceAccountingError):
            aggregate_resource_receipts(
                [second, first],
                expected_method_id="dp_fedavg",
                expected_rounds=2,
                expected_model_manifest_sha256=MODEL_HASH,
            )
        tampered = copy.deepcopy(first)
        tampered["round_uplink_bytes"] += 1
        with self.assertRaises(ResourceAccountingError):
            aggregate_resource_receipts(
                [tampered, second],
                expected_method_id="dp_fedavg",
                expected_rounds=2,
                expected_model_manifest_sha256=MODEL_HASH,
            )
        changed_payload = _round("dp_fedavg", 2, model_payload_bytes=101)
        with self.assertRaisesRegex(ResourceAccountingError, "model payload bytes changed"):
            aggregate_resource_receipts(
                [first, changed_payload],
                expected_method_id="dp_fedavg",
                expected_rounds=2,
                expected_model_manifest_sha256=MODEL_HASH,
            )

    def test_equal_information_pair_rejects_extra_control_work(self) -> None:
        fedsift = _summary("fedsift")
        argmin_receipts = [
            _round("fedsift_public_argmin_rule", 1),
            _round("fedsift_public_argmin_rule", 2, public_control_candidate_count=6),
        ]
        argmin = aggregate_resource_receipts(
            argmin_receipts,
            expected_method_id="fedsift_public_argmin_rule",
            expected_rounds=2,
            expected_model_manifest_sha256=MODEL_HASH,
        )
        with self.assertRaises(ResourceAccountingError):
            validate_equal_information_control_resources(
                fedsift,
                argmin,
                expected_fedsift_summary_sha256=fedsift["report_sha256"],
                expected_public_argmin_summary_sha256=argmin["report_sha256"],
            )

    def test_equal_information_pair_requires_external_summary_commitments(self) -> None:
        fedsift = _summary("fedsift")
        argmin = _summary("fedsift_public_argmin_rule")
        with self.assertRaises(ResourceAccountingError):
            validate_equal_information_control_resources(
                fedsift,
                argmin,
                expected_fedsift_summary_sha256="f" * 64,
                expected_public_argmin_summary_sha256=argmin["report_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
