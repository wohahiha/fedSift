from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import copy
import hashlib
import json
import unittest
from fedsift.client_partition import (
    CLIENT_NAMES,
    ClientPartitionError,
    ClientPartitionInfeasibleError,
    generate_client_partition,
    validate_client_partition,
)
from fedsift.group_manifest import DatasetRows, build_group_manifest, group_manifest_fingerprint


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _synthetic_manifest() -> dict[str, object]:
    features: list[tuple[float, ...]] = []
    labels: list[int] = []
    for group_index in range(160):
        size = 2 if group_index < 10 else 1
        label = int(group_index % 5 in (0, 1))
        for _ in range(size):
            features.append((float(group_index), float(group_index % 7)))
            labels.append(label)
    rows = DatasetRows(
        dataset="synthetic_client_partition",
        source_path="synthetic://client-partition-v1",
        source_sha256="a" * 64,
        feature_names=("x0", "x1"),
        features=tuple(features),
        labels=tuple(labels),
        row_ids=tuple(range(len(labels))),
        target_name="label",
        predictor_allowlist_explicit=True,
    )
    return build_group_manifest(rows)


class RegisteredClientPartitionTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = _synthetic_manifest()
        cls.group_ids = tuple((str(group["group_id"]) for group in cls.manifest["groups"]))
        cls.kwargs = {
            "study_id": _identity("client_partition_test"),
            "scope": "inner_private_clients_fixed_label_skew",
            "outer_repeat": 1,
            "outer_fold": 2,
            "inner_fold": 0,
            "role": "private",
            "parent_split_sha256": "b" * 64,
        }
        cls.partition = generate_client_partition(cls.manifest, cls.group_ids, **cls.kwargs)

    def test_deterministic_registered_strength_constraints_and_diagnostics(self) -> None:
        repeated = generate_client_partition(self.manifest, self.group_ids, **self.kwargs)
        self.assertEqual(repeated, self.partition)
        validate_client_partition(self.partition, self.manifest, self.group_ids, **self.kwargs)
        policy = self.partition["policy"]
        self.assertEqual(policy["positive_share_percent_by_rank"], [10, 15, 20, 25, 30])
        self.assertEqual(policy["negative_share_percent_by_rank"], [30, 25, 20, 15, 10])
        self.assertEqual(policy["target_label_distribution_total_variation"], 0.3)
        self.assertEqual(policy["target_max_min_class_share_ratio"], 3.0)
        self.assertEqual(self.partition["candidate_attempts"], 64)
        self.assertEqual(len(self.partition["candidate_audit"]), 64)
        self.assertEqual(self.partition["performance_fields_used"], [])
        for client in CLIENT_NAMES:
            values = self.partition["clients"][client]
            self.assertGreaterEqual(values["row_count"], 24)
            self.assertGreaterEqual(values["positive"], 2)
            self.assertGreaterEqual(values["negative"], 2)
        diagnostics = self.partition["aggregate_diagnostics"]
        self.assertEqual(diagnostics["target_label_distribution_total_variation"], 0.3)
        self.assertGreater(diagnostics["attained_label_distribution_total_variation"], 0.0)
        self.assertGreater(diagnostics["client_prevalence_range"], 0.0)
        self.assertEqual(
            set(diagnostics["label_shift_diagnostic_fields"]),
            {
                "client_prevalence_range",
                "client_prevalence_population_std",
                "attained_label_distribution_total_variation",
                "positive_share_l1_error",
                "negative_share_l1_error",
            },
        )

    def test_duplicate_groups_are_indivisible_and_membership_is_exhaustive(self) -> None:
        group_to_client = self.partition["assignment"]
        seen: set[str] = set()
        for client in CLIENT_NAMES:
            ids = set(self.partition["clients"][client]["group_ids"])
            self.assertFalse(seen & ids)
            seen.update(ids)
            self.assertEqual(
                ids,
                {
                    group_id
                    for (group_id, assigned) in group_to_client.items()
                    if assigned == client
                },
            )
        self.assertEqual(seen, set(self.group_ids))
        duplicate_groups = [group for group in self.manifest["groups"] if group["n"] > 1]
        self.assertEqual(len(duplicate_groups), 10)
        for group in duplicate_groups:
            locations = sum(
                (
                    group["group_id"] in self.partition["clients"][client]["group_ids"]
                    for client in CLIENT_NAMES
                )
            )
            self.assertEqual(locations, 1)

    def test_tampering_is_rejected_even_when_outer_hash_is_recomputed(self) -> None:
        changed_membership = copy.deepcopy(self.partition)
        changed_membership["clients"]["client_0"]["positive"] += 1
        changed_membership["split_sha256"] = _canonical_sha256(
            {key: value for (key, value) in changed_membership.items() if key != "split_sha256"}
        )
        with self.assertRaises(ClientPartitionError):
            validate_client_partition(
                changed_membership, self.manifest, self.group_ids, **self.kwargs
            )
        changed_seed = copy.deepcopy(self.partition)
        changed_seed["seed_chain"]["role"] = "v_ctrl"
        changed_seed["split_sha256"] = _canonical_sha256(
            {key: value for (key, value) in changed_seed.items() if key != "split_sha256"}
        )
        with self.assertRaises(ClientPartitionError):
            validate_client_partition(changed_seed, self.manifest, self.group_ids, **self.kwargs)

    def test_infeasible_input_fails_closed_without_relaxation(self) -> None:
        with self.assertRaises(ClientPartitionInfeasibleError):
            generate_client_partition(self.manifest, self.group_ids[:40], **self.kwargs)

    def test_dataset_repeat_outer_inner_role_parent_and_membership_change_seeds(self) -> None:
        base = self.partition["seed_chain"]
        variants: list[dict[str, object]] = []
        for field, value in (
            ("outer_repeat", 2),
            ("outer_fold", 3),
            ("inner_fold", 1),
            ("role", "private_refit"),
            ("parent_split_sha256", "c" * 64),
        ):
            changed = dict(self.kwargs)
            changed[field] = value
            variants.append(
                generate_client_partition(self.manifest, self.group_ids, **changed)["seed_chain"]
            )
        changed_manifest = copy.deepcopy(self.manifest)
        changed_manifest["dataset"] = "synthetic_client_partition_variant"
        changed_manifest["group_manifest_sha256"] = group_manifest_fingerprint(changed_manifest)
        variants.append(
            generate_client_partition(changed_manifest, self.group_ids, **self.kwargs)["seed_chain"]
        )
        variants.append(
            generate_client_partition(self.manifest, self.group_ids[:-1], **self.kwargs)[
                "seed_chain"
            ]
        )
        for seed_chain in variants:
            self.assertNotEqual(seed_chain["role_seed_sha256"], base["role_seed_sha256"])
        self.assertNotEqual(variants[-2]["dataset_seed_sha256"], base["dataset_seed_sha256"])
        self.assertNotEqual(
            variants[-1]["selected_group_membership_sha256"],
            base["selected_group_membership_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
