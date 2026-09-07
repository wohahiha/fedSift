from __future__ import annotations
import copy
import inspect
import json
import unittest
from collections.abc import Mapping
from fedsift.candidate_space import canonical_sha256
from fedsift.hpo_capability import (
    HpoCapabilityError,
    OuterTestCapabilityError,
    build_hpo_unit_index,
    build_hpo_unit_capability,
    build_prevalidated_hpo_unit_capability,
    capability_candidate_parameters,
    materialize_capability_row_ids,
    validate_hpo_unit_capability,
)
from fedsift.nested_plan import CLIENT_NAMES
from tests.test_candidate_hpo_plan import SMALL_CANDIDATE_COUNT, cached_plan, frozen_nested_fixture


def _rehash_capability(capability: dict[str, object]) -> None:
    payload = copy.deepcopy(capability)
    payload.pop("capability_sha256", None)
    capability["capability_sha256"] = canonical_sha256(payload)


def _scalar_strings(value: object) -> set[str]:
    observed: set[str] = set()
    if isinstance(value, Mapping):
        for child in value.values():
            observed.update(_scalar_strings(child))
    elif isinstance(value, list):
        for child in value:
            observed.update(_scalar_strings(child))
    elif isinstance(value, str):
        observed.add(value)
    return observed


def _all_keys(value: object) -> set[str]:
    observed: set[str] = set()
    if isinstance(value, Mapping):
        observed.update((str(key) for key in value))
        for child in value.values():
            observed.update(_all_keys(child))
    elif isinstance(value, list):
        for child in value:
            observed.update(_all_keys(child))
    return observed


class HpoCapabilityTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.space, cls.nested, cls.manifest, cls.plan = cached_plan(SMALL_CANDIDATE_COUNT)
        cls.unit = cls.plan["units"][0]
        cls.capability = build_hpo_unit_capability(
            cls.plan, cls.space, cls.nested, cls.manifest, unit_id=cls.unit["unit_id"]
        )

    def _outer_test_membership(self, unit: Mapping[str, object]):
        fold = self.nested["repetitions"][unit["outer_repeat"]]["outer_folds"][unit["outer_fold"]]
        group_ids = set(fold["outer_test"]["group_ids"])
        group_rows = {
            group["group_id"]: tuple(group["row_ids"]) for group in self.manifest["groups"]
        }
        row_ids = {row_id for group_id in group_ids for row_id in group_rows[group_id]}
        return (group_ids, row_ids)

    def test_exact_single_unit_capability_rebuilds_from_upstreams(self) -> None:
        validate_hpo_unit_capability(
            self.capability,
            self.plan,
            self.space,
            self.nested,
            self.manifest,
            unit_id=self.unit["unit_id"],
        )
        self.assertEqual(self.capability["unit_id"], self.unit["unit_id"])
        self.assertEqual(
            self.capability["bindings"]["hpo_plan_sha256"], self.plan["hpo_plan_sha256"]
        )
        self.assertEqual(self.capability["split_bindings"], self.unit["split_bindings"])
        self.assertEqual(
            self.capability["candidate"]["candidate_sha256"], self.unit["candidate_sha256"]
        )
        slices = self.capability["data_slices"]
        self.assertEqual(set(slices["private_clients"]), set(CLIENT_NAMES))
        self.assertEqual(
            slices["inner_validation"]["membership_sha256"],
            self.unit["split_bindings"]["inner_validation_membership_sha256"],
        )
        for client in CLIENT_NAMES:
            self.assertEqual(
                slices["private_clients"][client]["membership_sha256"],
                self.unit["split_bindings"]["client_membership_sha256"][client],
            )

    def test_prevalidated_unit_index_matches_full_builder_and_is_plan_bound(self) -> None:
        index = build_hpo_unit_index(self.plan)
        indexed = build_prevalidated_hpo_unit_capability(
            self.plan,
            self.space,
            self.nested,
            self.manifest,
            unit_id=self.unit["unit_id"],
            unit_index=index,
            expected_hpo_plan_sha256=self.plan["hpo_plan_sha256"],
        )
        self.assertEqual(indexed, self.capability)
        with self.assertRaises(HpoCapabilityError):
            build_prevalidated_hpo_unit_capability(
                dict(self.plan),
                self.space,
                self.nested,
                self.manifest,
                unit_id=self.unit["unit_id"],
                unit_index=index,
                expected_hpo_plan_sha256=self.plan["hpo_plan_sha256"],
            )
        with self.assertRaises(HpoCapabilityError):
            build_prevalidated_hpo_unit_capability(
                self.plan,
                self.space,
                self.nested,
                self.manifest,
                unit_id=self.unit["unit_id"],
                unit_index=index,
                expected_hpo_plan_sha256="0" * 64,
            )
        duplicate_plan = dict(self.plan)
        duplicate_plan["units"] = [self.unit, self.unit]
        with self.assertRaises(HpoCapabilityError):
            build_hpo_unit_index(duplicate_plan)

    def test_serialized_artifact_has_no_outer_test_membership_or_other_unit(self) -> None:
        outer_groups, outer_rows = self._outer_test_membership(self.unit)
        serialized = json.dumps(
            self.capability, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )
        scalar_strings = _scalar_strings(self.capability)
        keys = _all_keys(self.capability)
        self.assertTrue(outer_groups.isdisjoint(scalar_strings))
        for row_id in outer_rows:
            self.assertNotIn(f"row_id:{row_id:012d}", serialized)
        self.assertNotIn("group_ids", keys)
        self.assertNotIn("row_to_group", keys)
        self.assertNotIn("repetitions", keys)
        self.assertNotIn("outer_folds", keys)
        self.assertNotIn("inner_folds", keys)
        self.assertNotIn("units", keys)
        other_unit_ids = {
            unit["unit_id"]
            for unit in self.plan["units"]
            if unit["unit_id"] != self.unit["unit_id"]
        }
        self.assertTrue(other_unit_ids.isdisjoint(scalar_strings))
        worker_rows = {
            row_id
            for role in (*CLIENT_NAMES, "v_ctrl", "v_sel", "inner_validation")
            for row_id in materialize_capability_row_ids(
                self.capability,
                role=role,
                expected_capability_sha256=self.capability["capability_sha256"],
            )
        }
        self.assertTrue(worker_rows.isdisjoint(outer_rows))

    def test_well_shaped_row_and_candidate_forgery_is_rejected(self) -> None:
        forged_rows = copy.deepcopy(self.capability)
        clients = forged_rows["data_slices"]["private_clients"]
        left = clients[CLIENT_NAMES[0]]["row_ids"]
        right = clients[CLIENT_NAMES[1]]["row_ids"]
        left[0], right[0] = (right[0], left[0])
        for values in (left, right):
            values.sort()
        for client in CLIENT_NAMES[:2]:
            client_slice = clients[client]
            client_slice["row_ids_sha256"] = canonical_sha256(client_slice["row_ids"])
        _rehash_capability(forged_rows)
        with self.assertRaises(HpoCapabilityError):
            validate_hpo_unit_capability(
                forged_rows,
                self.plan,
                self.space,
                self.nested,
                self.manifest,
                unit_id=self.unit["unit_id"],
            )
        forged_candidate = copy.deepcopy(self.capability)
        parameters = forged_candidate["candidate"]["parameters"]
        parameters["forged_parameter"] = True
        forged_candidate["candidate"]["parameters_sha256"] = canonical_sha256(parameters)
        _rehash_capability(forged_candidate)
        with self.assertRaises(HpoCapabilityError):
            validate_hpo_unit_capability(
                forged_candidate,
                self.plan,
                self.space,
                self.nested,
                self.manifest,
                unit_id=self.unit["unit_id"],
            )
        unknown_field = copy.deepcopy(self.capability)
        unknown_field["outer_test_rows"] = []
        _rehash_capability(unknown_field)
        with self.assertRaises(HpoCapabilityError):
            validate_hpo_unit_capability(
                unknown_field,
                self.plan,
                self.space,
                self.nested,
                self.manifest,
                unit_id=self.unit["unit_id"],
            )

    def test_wrong_unit_and_cross_outer_rebinding_are_rejected(self) -> None:
        other = next(
            (
                unit
                for unit in self.plan["units"]
                if unit["method"] == self.unit["method"]
                and unit["candidate_id"] == self.unit["candidate_id"]
                and (unit["hpo_seed"] == self.unit["hpo_seed"])
                and (unit["inner_fold"] == self.unit["inner_fold"])
                and (unit["outer_repeat"] == self.unit["outer_repeat"])
                and (unit["outer_fold"] != self.unit["outer_fold"])
            )
        )
        with self.assertRaises(HpoCapabilityError):
            validate_hpo_unit_capability(
                self.capability,
                self.plan,
                self.space,
                self.nested,
                self.manifest,
                unit_id=other["unit_id"],
            )
        forged = copy.deepcopy(self.capability)
        forged["unit_id"] = other["unit_id"]
        for field in forged["unit_identity"]:
            forged["unit_identity"][field] = other[field]
        forged["split_bindings"] = copy.deepcopy(other["split_bindings"])
        _rehash_capability(forged)
        with self.assertRaises(HpoCapabilityError):
            validate_hpo_unit_capability(
                forged, self.plan, self.space, self.nested, self.manifest, unit_id=other["unit_id"]
            )
        with self.assertRaises(HpoCapabilityError):
            validate_hpo_unit_capability(
                self.capability,
                self.plan,
                self.space,
                self.nested,
                self.manifest,
                unit_id="hpo_000000000000000000000000",
            )

    def test_worker_api_has_no_outer_test_or_upstream_materializer(self) -> None:
        parameters = inspect.signature(materialize_capability_row_ids).parameters
        self.assertEqual(set(parameters), {"capability", "role", "expected_capability_sha256"})
        for forbidden in ("outer_test", "outer-test", "test"):
            with self.assertRaises(OuterTestCapabilityError):
                materialize_capability_row_ids(
                    {}, role=forbidden, expected_capability_sha256="not-even-a-hash"
                )
        with self.assertRaises(HpoCapabilityError):
            materialize_capability_row_ids(
                self.capability,
                role="private",
                expected_capability_sha256=self.capability["capability_sha256"],
            )
        with self.assertRaises(HpoCapabilityError):
            materialize_capability_row_ids(
                self.capability, role="v_ctrl", expected_capability_sha256="0" * 64
            )
        parameters_copy = capability_candidate_parameters(
            self.capability, expected_capability_sha256=self.capability["capability_sha256"]
        )
        self.assertEqual(parameters_copy, self.capability["candidate"]["parameters"])
        parameters_copy["worker_mutation"] = True
        self.assertNotIn("worker_mutation", self.capability["candidate"]["parameters"])

    def test_draft_nested_plan_cannot_issue_a_capability(self) -> None:
        _, draft, _ = frozen_nested_fixture()
        with self.assertRaises(HpoCapabilityError):
            build_hpo_unit_capability(
                self.plan, self.space, draft, self.manifest, unit_id=self.unit["unit_id"]
            )


if __name__ == "__main__":
    unittest.main()
