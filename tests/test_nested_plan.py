from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import copy
import hashlib
import json
import unittest
from pathlib import Path
from fedsift.group_manifest import GroupManifestError, build_group_manifest, load_arff_rows
from fedsift.nested_plan import (
    CLIENT_NAMES,
    ForbiddenOuterTestAccess,
    NestedPlanError,
    freeze_nested_plan,
    generate_nested_plan,
    materialize_hpo_row_ids,
    nested_plan_fingerprint,
    validate_nested_plan,
)
from fedsift.client_partition import client_partition_policy_fingerprint

ROOT = Path(__file__).resolve().parents[1]
DEBRECEN_PREDICTORS = tuple((str(index) for index in range(19)))


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _rehash_nested_partition(partition: dict[str, object]) -> None:
    partition["split_sha256"] = _sha256_json(
        {key: value for (key, value) in partition.items() if key != "split_sha256"}
    )


class RealDebrecenNestedPlanTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = build_group_manifest(
            load_arff_rows(
                ROOT / "data" / "raw" / "diabetic_retinopathy_debrecen" / "messidor_features.arff",
                dataset="retinopathy",
                target="Class",
                predictors=DEBRECEN_PREDICTORS,
            )
        )
        cls.plan = generate_nested_plan(cls.manifest, study_id=_identity("nested_plan_test"))
        cls.frozen_plan = freeze_nested_plan(
            cls.plan,
            cls.manifest,
            protocol_sha256="2" * 64,
            implementation_sha256="3" * 64,
            client_policy_sha256=client_partition_policy_fingerprint(),
        )
        cls.duplicate_groups = {
            group["group_id"]: tuple(group["row_ids"])
            for group in cls.manifest["groups"]
            if group["n"] == 2
        }

    def test_fixed_shape_hash_bindings_and_explicit_stratification(self) -> None:
        self.assertEqual(self.plan["status"], "DRAFT_NOT_FROZEN")
        self.assertEqual(len(self.plan["repetitions"]), 3)
        self.assertTrue(self.plan["result_blind_selection"])
        self.assertEqual(self.plan["performance_fields_used"], [])
        self.assertTrue(self.plan["label_stratified"])
        self.assertEqual(self.plan["label_fields_used"], ["positive", "negative", "prevalence"])
        self.assertEqual(
            self.plan["source_binding"]["source_sha256"], self.manifest["source_sha256"]
        )
        self.assertEqual(
            self.plan["source_binding"]["group_manifest_sha256"],
            self.manifest["group_manifest_sha256"],
        )
        split_hashes: set[str] = set()
        for repeat in self.plan["repetitions"]:
            self.assertEqual(len(repeat["outer_folds"]), 5)
            split_hashes.add(repeat["outer_partition"]["split_sha256"])
            self.assertTrue(repeat["outer_partition"]["label_stratified"])
            for outer in repeat["outer_folds"]:
                self.assertEqual(len(outer["inner_folds"]), 3)
                split_hashes.add(outer["inner_partition"]["split_sha256"])
                self.assertTrue(outer["inner_partition"]["label_stratified"])
                for inner in outer["inner_folds"]:
                    for key in ("role_partition", "client_partition"):
                        split = inner[key]
                        self.assertTrue(split["label_stratified"])
                        self.assertEqual(split["candidate_attempts"], 64)
                        self.assertEqual(split["performance_fields_used"], [])
                        split_hashes.add(split["split_sha256"])
                    role_rows = {
                        name: inner["roles"][name]["row_count"]
                        for name in ("private", "v_ctrl", "v_sel")
                    }
                    total = sum(role_rows.values())
                    self.assertLessEqual(abs(role_rows["private"] - 0.8 * total), 2)
                    self.assertLessEqual(abs(role_rows["v_ctrl"] - 0.1 * total), 2)
                    self.assertLessEqual(abs(role_rows["v_sel"] - 0.1 * total), 2)
                for key in ("role_partition", "client_partition"):
                    split = outer["outer_refit"][key]
                    self.assertTrue(split["label_stratified"])
                    split_hashes.add(split["split_sha256"])
        self.assertEqual(len(split_hashes), 138)
        validate_nested_plan(self.plan, self.manifest)
        round_tripped = json.loads(json.dumps(self.plan, ensure_ascii=False))
        validate_nested_plan(round_tripped, self.manifest)

    def test_formal_consumer_requires_explicit_hash_bound_freeze(self) -> None:
        with self.assertRaises(NestedPlanError):
            validate_nested_plan(self.plan, self.manifest, require_frozen=True)
        frozen = freeze_nested_plan(
            self.plan,
            self.manifest,
            protocol_sha256="2" * 64,
            implementation_sha256="3" * 64,
            client_policy_sha256=client_partition_policy_fingerprint(),
        )
        self.assertEqual(frozen["status"], "FROZEN")
        self.assertEqual(
            frozen["freeze_binding"]["draft_nested_plan_sha256"], self.plan["nested_plan_sha256"]
        )
        self.assertEqual(frozen["freeze_binding"]["protocol_sha256"], "2" * 64)
        self.assertEqual(frozen["freeze_binding"]["implementation_sha256"], "3" * 64)
        validate_nested_plan(frozen, self.manifest, require_frozen=True)
        injected = copy.deepcopy(self.plan)
        injected["metrics"] = {"average_precision": 1.0}
        injected["nested_plan_sha256"] = nested_plan_fingerprint(injected)
        with self.assertRaises(NestedPlanError):
            validate_nested_plan(injected, self.manifest)
        with self.assertRaises(NestedPlanError):
            freeze_nested_plan(
                self.plan,
                self.manifest,
                protocol_sha256="2" * 64,
                implementation_sha256="3" * 64,
                client_policy_sha256="4" * 64,
            )

    def test_groups_are_exclusive_through_outer_inner_roles_and_clients(self) -> None:
        root_ids = {group["group_id"] for group in self.manifest["groups"]}
        self.assertEqual(len(self.duplicate_groups), 5)
        for repeat in self.plan["repetitions"]:
            test_counts = {group_id: 0 for group_id in root_ids}
            for outer in repeat["outer_folds"]:
                outer_test = set(outer["outer_test"]["group_ids"])
                outer_train = set(outer["outer_train"]["group_ids"])
                self.assertFalse(outer_test & outer_train)
                self.assertEqual(outer_test | outer_train, root_ids)
                for group_id in outer_test:
                    test_counts[group_id] += 1
                validation_counts = {group_id: 0 for group_id in outer_train}
                for inner in outer["inner_folds"]:
                    validation = set(inner["inner_validation"]["group_ids"])
                    inner_train = set(inner["inner_train"]["group_ids"])
                    self.assertFalse(validation & inner_train)
                    self.assertEqual(validation | inner_train, outer_train)
                    for group_id in validation:
                        validation_counts[group_id] += 1
                    role_sets = [
                        set(inner["roles"][name]["group_ids"])
                        for name in ("private", "v_ctrl", "v_sel")
                    ]
                    self.assertEqual(set.union(*role_sets), inner_train)
                    self.assertEqual(sum((len(values) for values in role_sets)), len(inner_train))
                    private = role_sets[0]
                    client_sets = [
                        set(inner["clients"][name]["group_ids"]) for name in CLIENT_NAMES
                    ]
                    self.assertEqual(set.union(*client_sets), private)
                    self.assertEqual(sum((len(values) for values in client_sets)), len(private))
                    for duplicate_group in self.duplicate_groups:
                        locations = sum(
                            (duplicate_group in values for values in [validation, *role_sets])
                        )
                        self.assertEqual(locations, int(duplicate_group in outer_train))
                        client_locations = sum(
                            (duplicate_group in values for values in client_sets)
                        )
                        self.assertEqual(client_locations, int(duplicate_group in private))
                self.assertTrue(all((value == 1 for value in validation_counts.values())))
            self.assertTrue(all((value == 1 for value in test_counts.values())))

    def test_hpo_access_fails_closed_before_outer_test_materialization(self) -> None:
        for forbidden in ("outer_test", "test", "outer-test"):
            with self.assertRaises(ForbiddenOuterTestAccess):
                materialize_hpo_row_ids(
                    self.plan,
                    self.manifest,
                    outer_repeat=0,
                    outer_fold=0,
                    inner_fold=0,
                    role=forbidden,
                )
        with self.assertRaises(NestedPlanError):
            materialize_hpo_row_ids(
                self.plan, self.manifest, outer_repeat=0, outer_fold=0, inner_fold=0, role="private"
            )
        outer_test_rows = {
            row_id
            for group_id in self.plan["repetitions"][0]["outer_folds"][0]["outer_test"]["group_ids"]
            for group in self.manifest["groups"]
            if group["group_id"] == group_id
            for row_id in group["row_ids"]
        }
        for allowed in ("private", "v_ctrl", "v_sel", "inner_validation", *CLIENT_NAMES):
            data_slice = materialize_hpo_row_ids(
                self.frozen_plan,
                self.manifest,
                outer_repeat=0,
                outer_fold=0,
                inner_fold=0,
                role=allowed,
            )
            self.assertFalse(set(data_slice.row_ids) & outer_test_rows)
            selected_groups = {
                group["group_id"]: tuple(group["row_ids"])
                for group in self.manifest["groups"]
                if group["group_id"] in data_slice.group_ids
            }
            expected_rows = sorted((row_id for rows in selected_groups.values() for row_id in rows))
            self.assertEqual(list(data_slice.row_ids), expected_rows)
            for group_id, rows in self.duplicate_groups.items():
                selected = [row_id in data_slice.row_ids for row_id in rows]
                self.assertIn(selected, ([False, False], [True, True]))

    def test_source_group_plan_and_split_tampering_is_rejected(self) -> None:
        changed_status = copy.deepcopy(self.plan)
        changed_status["status"] = "READY_ENOUGH"
        changed_status["nested_plan_sha256"] = nested_plan_fingerprint(changed_status)
        with self.assertRaises(NestedPlanError):
            validate_nested_plan(changed_status, self.manifest)
        changed_plan = copy.deepcopy(self.plan)
        changed_plan["source_binding"]["source_sha256"] = "0" * 64
        changed_plan["nested_plan_sha256"] = nested_plan_fingerprint(changed_plan)
        with self.assertRaises(NestedPlanError):
            validate_nested_plan(changed_plan, self.manifest)
        changed_split = copy.deepcopy(self.plan)
        outer = changed_split["repetitions"][0]["outer_partition"]
        first_group = next(iter(outer["assignment"]))
        outer["assignment"][first_group] = "fold_4"
        changed_split["nested_plan_sha256"] = nested_plan_fingerprint(changed_split)
        with self.assertRaises(NestedPlanError):
            validate_nested_plan(changed_split, self.manifest)
        changed_weight = copy.deepcopy(self.plan)
        weighted_outer = changed_weight["repetitions"][0]["outer_partition"]
        weighted_outer["weights"][0] = 2.0
        _rehash_nested_partition(weighted_outer)
        changed_weight["nested_plan_sha256"] = nested_plan_fingerprint(changed_weight)
        with self.assertRaises(NestedPlanError):
            validate_nested_plan(changed_weight, self.manifest)
        changed_constraint = copy.deepcopy(self.plan)
        constrained_outer = changed_constraint["repetitions"][0]["outer_partition"]
        constrained_outer["constraints"]["min_positive_per_part"] = 29
        _rehash_nested_partition(constrained_outer)
        changed_constraint["nested_plan_sha256"] = nested_plan_fingerprint(changed_constraint)
        with self.assertRaises(NestedPlanError):
            validate_nested_plan(changed_constraint, self.manifest)
        changed_scope = copy.deepcopy(self.plan)
        scope_outer = changed_scope["repetitions"][0]["outer_partition"]
        scope_outer["context"]["scope"] = "outer_tampered"
        scope_outer["context_sha256"] = _sha256_json(scope_outer["context"])
        _rehash_nested_partition(scope_outer)
        changed_scope["nested_plan_sha256"] = nested_plan_fingerprint(changed_scope)
        with self.assertRaises(NestedPlanError):
            validate_nested_plan(changed_scope, self.manifest)
        changed_audit = copy.deepcopy(self.plan)
        audit_outer = changed_audit["repetitions"][0]["outer_partition"]
        audit_outer["candidate_audit_sha256"] = "5" * 64
        _rehash_nested_partition(audit_outer)
        changed_audit["nested_plan_sha256"] = nested_plan_fingerprint(changed_audit)
        with self.assertRaises(NestedPlanError):
            validate_nested_plan(changed_audit, self.manifest)
        changed_attempt = copy.deepcopy(self.plan)
        attempt_outer = changed_attempt["repetitions"][0]["outer_partition"]
        attempt_outer["selected_attempt"] = (int(attempt_outer["selected_attempt"]) + 1) % 64
        _rehash_nested_partition(attempt_outer)
        changed_attempt["nested_plan_sha256"] = nested_plan_fingerprint(changed_attempt)
        with self.assertRaises(NestedPlanError):
            validate_nested_plan(changed_attempt, self.manifest)
        changed_manifest = copy.deepcopy(self.manifest)
        changed_manifest["source_sha256"] = "1" * 64
        with self.assertRaises(GroupManifestError):
            materialize_hpo_row_ids(
                self.frozen_plan,
                changed_manifest,
                outer_repeat=0,
                outer_fold=0,
                inner_fold=0,
                role="private",
            )


if __name__ == "__main__":
    unittest.main()
