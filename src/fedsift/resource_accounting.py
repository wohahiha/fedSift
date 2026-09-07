"""Deterministic communication and work accounting for neutral comparisons.

Wall-clock timing is intentionally excluded.  It requires a separate,
prewarmed and order-balanced benchmark.  This module counts actual protocol
payloads and public-control evaluations so FedSift does not receive an
unreported information or computation advantage.
"""

from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import copy
import hashlib
import json
from numbers import Integral
from typing import Mapping, Sequence


class ResourceAccountingError(ValueError):
    """Raised when a resource receipt or comparison is malformed."""


_METHODS = {
    "fedavg_nonprivate",
    "dp_fedavg",
    "dp_fedprox_adapted",
    "dp_scaffold_adapted",
    "dp_fedadam",
    "dp_fedyogi",
    "dp_fedsofim_delta_proxy_adapted",
    "time_dpfedadam",
    "public_argmin_time_dpfedadam",
    "fedsift",
    "fedsift_uniform_schedule",
    "fedsift_without_sift",
    "fedsift_public_argmin_rule",
}
_PUBLIC_CONTROL_METHODS = {
    "public_argmin_time_dpfedadam",
    "fedsift",
    "fedsift_uniform_schedule",
    "fedsift_public_argmin_rule",
}
_SCAFFOLD_METHODS = {"dp_scaffold_adapted"}
_SERVER_FULL_MODEL_STATE_VECTOR_COUNTS = {
    "fedavg_nonprivate": 0,
    "dp_fedavg": 0,
    "dp_fedprox_adapted": 0,
    "dp_scaffold_adapted": 1,
    "dp_fedadam": 2,
    "dp_fedyogi": 2,
    "dp_fedsofim_delta_proxy_adapted": 1,
    "time_dpfedadam": 2,
    "public_argmin_time_dpfedadam": 2,
    "fedsift": 2,
    "fedsift_uniform_schedule": 2,
    "fedsift_without_sift": 2,
    "fedsift_public_argmin_rule": 2,
}
_SERVER_GLOBAL_VECTOR_INNER_PRODUCTS_PER_ROUND = {"dp_fedsofim_delta_proxy_adapted": 2}
_ROUND_FIELDS = {
    "schema",
    "status",
    "method_id",
    "server_round",
    "participating_client_ids",
    "participating_client_count",
    "total_client_count",
    "model_manifest_sha256",
    "payload_definition",
    "model_payload_bytes",
    "control_payload_bytes",
    "per_client_downlink_bytes",
    "per_client_uplink_bytes",
    "round_downlink_bytes",
    "round_uplink_bytes",
    "round_total_federated_bytes",
    "local_optimizer_steps",
    "sampled_record_gradient_evaluations",
    "public_control_query_executed",
    "public_control_record_count",
    "public_control_candidate_count",
    "public_candidate_forward_record_evaluations",
    "server_persistent_full_model_state_definition",
    "server_persistent_full_model_state_vector_count",
    "server_persistent_full_model_state_bytes",
    "server_global_vector_inner_product_definition",
    "server_global_vector_inner_products",
    "server_persistent_state_bytes_excluded_from_network",
    "wall_clock_seconds_present",
    "secure_aggregation_claim",
    "receipt_sha256",
}
_SUMMARY_FIELDS = {
    "schema",
    "status",
    "method_id",
    "round_count",
    "model_manifest_sha256",
    "round_receipt_sha256",
    "public_control_schedule",
    "server_persistent_state_profile",
    "totals",
    "timing_interpretation",
    "report_sha256",
}


def _canonical_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ResourceAccountingError("resource value is not canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ResourceAccountingError(f"{name} must be an exact integer")
    result = int(value)
    if result < minimum:
        raise ResourceAccountingError(f"{name} must be >= {minimum}")
    return result


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value.lower() != value
        or any((character not in "0123456789abcdef" for character in value))
    ):
        raise ResourceAccountingError(f"{name} must be a lowercase SHA-256 hex string")
    return value


def _client_ids(values: object) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ResourceAccountingError("participating_client_ids must be a sequence")
    result = tuple(values)
    if not result or any(
        (not isinstance(value, str) or not value or value.strip() != value for value in result)
    ):
        raise ResourceAccountingError("participating client ids must be canonical strings")
    if len(set(result)) != len(result):
        raise ResourceAccountingError("participating client ids must be unique")
    return result


def build_round_resource_receipt(
    *,
    method_id: str,
    server_round: object,
    participating_client_ids: object,
    total_client_count: object,
    model_payload_bytes: object,
    model_manifest_sha256: str,
    local_optimizer_steps: object,
    sampled_record_gradient_evaluations: object,
    control_payload_bytes: object = 0,
    public_control_query_executed: object = False,
    public_control_record_count: object = 0,
    public_control_candidate_count: object = 0,
) -> dict[str, object]:
    """Build one round receipt from protocol payload sizes and executed work."""
    if method_id not in _METHODS:
        raise ResourceAccountingError("unknown method_id")
    round_number = _integer(server_round, "server_round", minimum=1)
    clients = _client_ids(participating_client_ids)
    total_clients = _integer(total_client_count, "total_client_count", minimum=1)
    if len(clients) > total_clients:
        raise ResourceAccountingError("participating clients exceed total client count")
    model_bytes = _integer(model_payload_bytes, "model_payload_bytes", minimum=1)
    control_bytes = _integer(control_payload_bytes, "control_payload_bytes")
    local_steps = _integer(local_optimizer_steps, "local_optimizer_steps")
    gradient_evaluations = _integer(
        sampled_record_gradient_evaluations, "sampled_record_gradient_evaluations"
    )
    control_records = _integer(public_control_record_count, "public_control_record_count")
    control_candidates = _integer(public_control_candidate_count, "public_control_candidate_count")
    if not isinstance(public_control_query_executed, bool):
        raise ResourceAccountingError("public_control_query_executed must be exact boolean")
    manifest_hash = _sha256(model_manifest_sha256, "model_manifest_sha256")
    server_state_vector_count = _SERVER_FULL_MODEL_STATE_VECTOR_COUNTS[method_id]
    server_state_bytes = server_state_vector_count * model_bytes
    server_inner_products = _SERVER_GLOBAL_VECTOR_INNER_PRODUCTS_PER_ROUND.get(method_id, 0)
    if method_id in _SCAFFOLD_METHODS:
        if control_bytes != model_bytes:
            raise ResourceAccountingError(
                "SCAFFOLD control payload must match the full model parameter payload"
            )
        per_client_downlink = model_bytes + control_bytes
        per_client_uplink = model_bytes + control_bytes
    else:
        if control_bytes != 0:
            raise ResourceAccountingError("non-SCAFFOLD methods cannot hide a control payload")
        per_client_downlink = model_bytes
        per_client_uplink = model_bytes
    if method_id in _PUBLIC_CONTROL_METHODS:
        if public_control_query_executed and (control_records == 0 or control_candidates < 2):
            raise ResourceAccountingError(
                "public-control methods must report records and at least two candidates"
            )
        if not public_control_query_executed and (control_records != 0 or control_candidates != 0):
            raise ResourceAccountingError("a non-query round cannot report public-control work")
    elif public_control_query_executed or control_records != 0 or control_candidates != 0:
        raise ResourceAccountingError(
            "methods without V_ctrl queries cannot report public-control work"
        )
    receipt: dict[str, object] = {
        "schema": _identity("round_resource_receipt"),
        "status": "complete",
        "method_id": method_id,
        "server_round": round_number,
        "participating_client_ids": list(clients),
        "participating_client_count": len(clients),
        "total_client_count": total_clients,
        "model_manifest_sha256": manifest_hash,
        "payload_definition": "parameter_tensor_numel_times_element_size",
        "model_payload_bytes": model_bytes,
        "control_payload_bytes": control_bytes,
        "per_client_downlink_bytes": per_client_downlink,
        "per_client_uplink_bytes": per_client_uplink,
        "round_downlink_bytes": per_client_downlink * len(clients),
        "round_uplink_bytes": per_client_uplink * len(clients),
        "round_total_federated_bytes": (per_client_downlink + per_client_uplink) * len(clients),
        "local_optimizer_steps": local_steps,
        "sampled_record_gradient_evaluations": gradient_evaluations,
        "public_control_query_executed": public_control_query_executed,
        "public_control_record_count": control_records,
        "public_control_candidate_count": control_candidates,
        "public_candidate_forward_record_evaluations": control_records * control_candidates,
        "server_persistent_full_model_state_definition": "full_model_equivalent_parameter_state_vectors_with_model_payload_dtype",
        "server_persistent_full_model_state_vector_count": server_state_vector_count,
        "server_persistent_full_model_state_bytes": server_state_bytes,
        "server_global_vector_inner_product_definition": "dot_product_over_the_full_flattened_parameter_vector",
        "server_global_vector_inner_products": server_inner_products,
        "server_persistent_state_bytes_excluded_from_network": True,
        "wall_clock_seconds_present": False,
        "secure_aggregation_claim": False,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


def aggregate_resource_receipts(
    receipts: Sequence[Mapping[str, object]],
    *,
    expected_method_id: str,
    expected_rounds: object,
    expected_model_manifest_sha256: str,
) -> dict[str, object]:
    """Validate a consecutive round ledger and sum deterministic resources."""
    if expected_method_id not in _METHODS:
        raise ResourceAccountingError("unknown expected_method_id")
    round_count = _integer(expected_rounds, "expected_rounds", minimum=1)
    manifest_hash = _sha256(expected_model_manifest_sha256, "expected_model_manifest_sha256")
    if isinstance(receipts, (str, bytes)) or not isinstance(receipts, Sequence):
        raise ResourceAccountingError("receipts must be a sequence")
    entries = [copy.deepcopy(dict(value)) for value in receipts]
    if len(entries) != round_count:
        raise ResourceAccountingError("resource ledger is not complete")
    for index, receipt in enumerate(entries, start=1):
        if set(receipt) != _ROUND_FIELDS:
            raise ResourceAccountingError("round resource receipt schema differs")
        rebuilt = build_round_resource_receipt(
            method_id=expected_method_id,
            server_round=index,
            participating_client_ids=receipt.get("participating_client_ids"),
            total_client_count=receipt.get("total_client_count"),
            model_payload_bytes=receipt.get("model_payload_bytes"),
            model_manifest_sha256=manifest_hash,
            local_optimizer_steps=receipt.get("local_optimizer_steps"),
            sampled_record_gradient_evaluations=receipt.get("sampled_record_gradient_evaluations"),
            control_payload_bytes=receipt.get("control_payload_bytes"),
            public_control_query_executed=receipt.get("public_control_query_executed"),
            public_control_record_count=receipt.get("public_control_record_count"),
            public_control_candidate_count=receipt.get("public_control_candidate_count"),
        )
        if receipt != rebuilt:
            raise ResourceAccountingError("resource ledger identity or order differs")
    if any(
        (
            receipt["model_payload_bytes"] != entries[0]["model_payload_bytes"]
            for receipt in entries[1:]
        )
    ):
        raise ResourceAccountingError("model payload bytes changed within one resource ledger")
    sum_fields = (
        "round_downlink_bytes",
        "round_uplink_bytes",
        "round_total_federated_bytes",
        "local_optimizer_steps",
        "sampled_record_gradient_evaluations",
        "public_candidate_forward_record_evaluations",
        "server_global_vector_inner_products",
    )
    totals = {
        field: sum((_integer(receipt[field], field) for receipt in entries)) for field in sum_fields
    }
    report: dict[str, object] = {
        "schema": _identity("resource_ledger_summary"),
        "status": "complete",
        "method_id": expected_method_id,
        "round_count": round_count,
        "model_manifest_sha256": manifest_hash,
        "round_receipt_sha256": [receipt["receipt_sha256"] for receipt in entries],
        "public_control_schedule": [
            {
                "server_round": receipt["server_round"],
                "record_count": receipt["public_control_record_count"],
                "candidate_count": receipt["public_control_candidate_count"],
            }
            for receipt in entries
            if receipt["public_control_query_executed"] is True
        ],
        "server_persistent_state_profile": {
            "scope": "persistent_full_parameter_shaped_server_state_only_excluding_ephemeral_working_tensors",
            "model_payload_bytes": entries[0]["model_payload_bytes"],
            "full_model_state_vector_count": entries[0][
                "server_persistent_full_model_state_vector_count"
            ],
            "peak_full_model_state_bytes": entries[0]["server_persistent_full_model_state_bytes"],
            "included_in_federated_network_bytes": False,
        },
        "totals": totals,
        "timing_interpretation": "deterministic_network_local_work_public_query_and_selected_server_work_counts_only_wall_clock_requires_separate_prewarmed_order_balanced_benchmark",
    }
    report["report_sha256"] = _canonical_sha256(report)
    return report


def validate_equal_information_control_resources(
    fedsift_summary: Mapping[str, object],
    matched_public_argmin_summary: Mapping[str, object],
    *,
    expected_fedsift_summary_sha256: str,
    expected_public_argmin_summary_sha256: str,
) -> dict[str, object]:
    """Require identical work for FedSift's exact inherited argmin ablation.

    The independently tuned ``public_argmin_time_dpfedadam`` main method is an
    equal-HPO-budget comparator, not an exact-resource matched arm: its selected
    query frequency or other candidate values may legitimately differ.  Exact
    equality is therefore enforced only against
    ``fedsift_public_argmin_rule``, which inherits the selected FedSift parent
    candidate and changes one control-rule component.
    """
    for name, summary, method, expected_hash in (
        (
            "fedsift",
            fedsift_summary,
            "fedsift",
            _sha256(expected_fedsift_summary_sha256, "expected_fedsift_summary_sha256"),
        ),
        (
            "matched_public_argmin",
            matched_public_argmin_summary,
            "fedsift_public_argmin_rule",
            _sha256(expected_public_argmin_summary_sha256, "expected_public_argmin_summary_sha256"),
        ),
    ):
        if (
            not isinstance(summary, Mapping)
            or set(summary) != _SUMMARY_FIELDS
            or summary.get("schema") != _identity("resource_ledger_summary")
            or (summary.get("status") != "complete")
            or (summary.get("method_id") != method)
        ):
            raise ResourceAccountingError(f"{name} summary identity differs")
        payload = copy.deepcopy(dict(summary))
        digest = payload.pop("report_sha256", None)
        if digest != _canonical_sha256(payload):
            raise ResourceAccountingError(f"{name} summary hash differs")
        if digest != expected_hash:
            raise ResourceAccountingError(f"{name} summary differs from external commitment")
        _integer(summary.get("round_count"), f"{name} round_count", minimum=1)
        _sha256(summary.get("model_manifest_sha256"), f"{name} model_manifest_sha256")
        receipt_hashes = summary.get("round_receipt_sha256")
        if not isinstance(receipt_hashes, list) or len(receipt_hashes) != summary.get(
            "round_count"
        ):
            raise ResourceAccountingError(f"{name} round receipt inventory differs")
        for digest_value in receipt_hashes:
            _sha256(digest_value, f"{name} round receipt sha256")
        totals = summary.get("totals")
        expected_total_fields = {
            "round_downlink_bytes",
            "round_uplink_bytes",
            "round_total_federated_bytes",
            "local_optimizer_steps",
            "sampled_record_gradient_evaluations",
            "public_candidate_forward_record_evaluations",
            "server_global_vector_inner_products",
        }
        if not isinstance(totals, Mapping) or set(totals) != expected_total_fields:
            raise ResourceAccountingError(f"{name} total fields differ")
        for field in expected_total_fields:
            _integer(totals.get(field), f"{name} {field}")
        state_profile = summary.get("server_persistent_state_profile")
        if not isinstance(state_profile, Mapping) or set(state_profile) != {
            "scope",
            "model_payload_bytes",
            "full_model_state_vector_count",
            "peak_full_model_state_bytes",
            "included_in_federated_network_bytes",
        }:
            raise ResourceAccountingError(f"{name} server state profile differs")
        if (
            state_profile.get("scope")
            != "persistent_full_parameter_shaped_server_state_only_excluding_ephemeral_working_tensors"
        ):
            raise ResourceAccountingError(f"{name} server state scope differs")
        expected_state_count = _SERVER_FULL_MODEL_STATE_VECTOR_COUNTS[method]
        model_payload_bytes = _integer(
            state_profile.get("model_payload_bytes"), f"{name} model payload bytes", minimum=1
        )
        state_count = _integer(
            state_profile.get("full_model_state_vector_count"),
            f"{name} full model state vector count",
        )
        state_bytes = _integer(
            state_profile.get("peak_full_model_state_bytes"), f"{name} peak full model state bytes"
        )
        if state_count != expected_state_count:
            raise ResourceAccountingError(f"{name} server state count differs")
        if state_bytes != state_count * model_payload_bytes:
            raise ResourceAccountingError(f"{name} server state bytes differ")
        if state_profile.get("included_in_federated_network_bytes") is not False:
            raise ResourceAccountingError(f"{name} server state network scope differs")
        schedule = summary.get("public_control_schedule")
        if not isinstance(schedule, list):
            raise ResourceAccountingError(f"{name} public-control schedule differs")
        previous_round = 0
        for entry in schedule:
            if not isinstance(entry, Mapping) or set(entry) != {
                "server_round",
                "record_count",
                "candidate_count",
            }:
                raise ResourceAccountingError(f"{name} public-control schedule schema differs")
            query_round = _integer(entry.get("server_round"), f"{name} query round", minimum=1)
            if query_round <= previous_round or query_round > summary.get("round_count"):
                raise ResourceAccountingError(f"{name} public-control schedule order differs")
            previous_round = query_round
            _integer(entry.get("record_count"), f"{name} query records", minimum=1)
            _integer(entry.get("candidate_count"), f"{name} query candidates", minimum=2)
    if fedsift_summary.get("round_count") != matched_public_argmin_summary.get(
        "round_count"
    ) or fedsift_summary.get("model_manifest_sha256") != matched_public_argmin_summary.get(
        "model_manifest_sha256"
    ):
        raise ResourceAccountingError("matched control summaries use different scopes")
    fields = (
        "round_downlink_bytes",
        "round_uplink_bytes",
        "round_total_federated_bytes",
        "local_optimizer_steps",
        "sampled_record_gradient_evaluations",
        "public_candidate_forward_record_evaluations",
        "server_global_vector_inner_products",
    )
    f_totals = fedsift_summary.get("totals")
    p_totals = matched_public_argmin_summary.get("totals")
    if not isinstance(f_totals, Mapping) or not isinstance(p_totals, Mapping):
        raise ResourceAccountingError("matched control totals are missing")
    if any((f_totals.get(field) != p_totals.get(field) for field in fields)):
        raise ResourceAccountingError(
            "FedSift and public argmin did not receive equal information/resources"
        )
    if fedsift_summary.get("server_persistent_state_profile") != matched_public_argmin_summary.get(
        "server_persistent_state_profile"
    ):
        raise ResourceAccountingError(
            "FedSift and public argmin used different persistent server resources"
        )
    if fedsift_summary.get("public_control_schedule") != matched_public_argmin_summary.get(
        "public_control_schedule"
    ):
        raise ResourceAccountingError(
            "FedSift and public argmin queried V_ctrl on different rounds or grids"
        )
    return {
        "schema": _identity("equal_information_resource_check"),
        "status": "verified",
        "fedsift_summary_sha256": fedsift_summary["report_sha256"],
        "matched_public_argmin_summary_sha256": matched_public_argmin_summary["report_sha256"],
        "equal_network_and_public_control_resources": True,
        "comparison_identity": "fedsift_vs_inherited_fedsift_public_argmin_rule",
    }


__all__ = [
    "ResourceAccountingError",
    "aggregate_resource_receipts",
    "build_round_resource_receipt",
    "validate_equal_information_control_resources",
]
