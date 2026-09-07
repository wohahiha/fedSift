from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import copy
import unittest
from pathlib import Path
from fedsift.balanced_group_split import (
    BalanceConstraints,
    InfeasiblePartitionError,
    SplitPlanningError,
    generate_balanced_partition,
)
from fedsift.group_manifest import (
    DatasetRows,
    GroupRecord,
    GroupManifestError,
    build_group_manifest,
    group_manifest_fingerprint,
    load_arff_rows,
    load_csv_rows,
)

ROOT = Path(__file__).resolve().parents[1]
PIMA_PREDICTORS = (
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
)
RETINOPATHY_PREDICTORS = tuple((str(index) for index in range(19)))


def context(manifest: dict[str, object], *, repeat: int = 0) -> dict[str, object]:
    return {
        "study_id": _identity("test"),
        "dataset_sha256": manifest["source_sha256"],
        "group_manifest_sha256": manifest["group_manifest_sha256"],
        "scope": "outer",
        "outer_repeat": repeat,
        "outer_fold": -1,
        "inner_fold": -1,
    }


class GroupManifestTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.pima_rows = load_csv_rows(
            ROOT / "data" / "diabetes.csv",
            dataset="pima",
            target="Outcome",
            predictors=PIMA_PREDICTORS,
        )
        cls.retino_rows = load_arff_rows(
            ROOT / "data" / "raw" / "diabetic_retinopathy_debrecen" / "messidor_features.arff",
            dataset="retinopathy",
            target="Class",
            predictors=RETINOPATHY_PREDICTORS,
        )
        cls.pima = build_group_manifest(cls.pima_rows)
        cls.retino = build_group_manifest(cls.retino_rows)

    def test_real_dataset_group_counts(self) -> None:
        self.assertEqual(self.pima["row_count"], 768)
        self.assertEqual(self.pima["group_count"], 768)
        self.assertEqual(self.pima["max_group_size"], 1)
        self.assertEqual(self.pima["mixed_label_group_count"], 0)
        self.assertEqual(self.retino["row_count"], 1151)
        self.assertEqual(self.retino["group_count"], 1146)
        self.assertEqual(self.retino["max_group_size"], 2)
        self.assertEqual(self.retino["mixed_label_group_count"], 0)
        duplicate_groups = [group for group in self.retino["groups"] if group["n"] == 2]
        self.assertEqual(len(duplicate_groups), 5)
        self.assertEqual(
            sorted((tuple(group["row_ids"]) for group in duplicate_groups)),
            [(54, 171), (70, 680), (75, 782), (423, 708), (551, 627)],
        )

    def test_label_changes_do_not_change_feature_group_ids(self) -> None:
        original = DatasetRows(
            dataset="synthetic",
            source_path="synthetic",
            source_sha256="a" * 64,
            feature_names=("x",),
            features=((1.0,), (1.0,), (2.0,)),
            labels=(0, 1, 0),
            row_ids=(0, 1, 2),
        )
        changed = DatasetRows(
            dataset="synthetic",
            source_path="synthetic",
            source_sha256="a" * 64,
            feature_names=("x",),
            features=original.features,
            labels=(1, 0, 1),
            row_ids=original.row_ids,
        )
        first = build_group_manifest(original)
        second = build_group_manifest(changed)
        first_membership = sorted(
            ((tuple(group["row_ids"]), group["group_id"]) for group in first["groups"])
        )
        second_membership = sorted(
            ((tuple(group["row_ids"]), group["group_id"]) for group in second["groups"])
        )
        self.assertEqual(first_membership, second_membership)

    def test_subject_id_has_priority_over_features(self) -> None:
        rows = DatasetRows(
            dataset="synthetic",
            source_path="synthetic",
            source_sha256="b" * 64,
            feature_names=("x",),
            features=((1.0,), (2.0,), (1.0,)),
            labels=(0, 1, 0),
            row_ids=(0, 1, 2),
        )
        manifest = build_group_manifest(
            rows,
            subject_ids=("subject-a", "subject-a", "subject-b"),
            subject_salt="test-only-salt",
            subject_id_provenance="synthetic test subject column",
        )
        self.assertEqual(manifest["group_rule"], "salted_subject_id_sha256")
        self.assertEqual(manifest["group_count"], 2)
        self.assertIn((0, 1), [tuple(group["row_ids"]) for group in manifest["groups"]])

    def test_subject_id_requires_provenance_and_nonempty_values(self) -> None:
        rows = DatasetRows(
            dataset="synthetic",
            source_path="synthetic",
            source_sha256="d" * 64,
            feature_names=("x",),
            features=((1.0,), (2.0,)),
            labels=(0, 1),
            row_ids=(0, 1),
        )
        with self.assertRaises(GroupManifestError):
            build_group_manifest(rows, subject_ids=("a", "b"), subject_salt="salt")
        with self.assertRaises(GroupManifestError):
            build_group_manifest(
                rows, subject_ids=("a", ""), subject_salt="salt", subject_id_provenance="test"
            )

    def test_serialized_manifest_rows_must_be_exclusive_and_exhaustive(self) -> None:
        poisoned = copy.deepcopy(self.retino)
        first = poisoned["groups"][0]
        second = poisoned["groups"][1]
        copied_row = first["row_ids"][0]
        second["row_ids"].append(copied_row)
        second["row_ids"].sort()
        second["n"] += 1
        second["negative"] += 1
        with self.assertRaises(GroupManifestError):
            generate_balanced_partition(
                poisoned, part_names=["a", "b"], weights=[1.0, 1.0], context=context(poisoned)
            )
        missing = copy.deepcopy(self.pima)
        removed = missing["groups"].pop()
        self.assertEqual(len(removed["row_ids"]), 1)
        with self.assertRaises(GroupManifestError):
            generate_balanced_partition(
                missing, part_names=["a", "b"], weights=[1.0, 1.0], context=context(missing)
            )

    def test_manifest_fingerprint_and_derived_fields_are_recomputed(self) -> None:
        changed_group = copy.deepcopy(self.pima)
        changed_group["groups"][0]["group_id"] = "0" * 64
        with self.assertRaises(GroupManifestError):
            generate_balanced_partition(
                changed_group,
                part_names=["a", "b"],
                weights=[1.0, 1.0],
                context=context(changed_group),
            )
        changed_provenance = copy.deepcopy(self.pima)
        changed_provenance["proxy_or_provenance_audit_required"] = False
        with self.assertRaises(GroupManifestError):
            generate_balanced_partition(
                changed_provenance,
                part_names=["a", "b"],
                weights=[1.0, 1.0],
                context=context(changed_provenance),
            )
        changed_stat = copy.deepcopy(self.pima)
        changed_stat["group_count"] += 1
        with self.assertRaises(GroupManifestError):
            generate_balanced_partition(
                changed_stat,
                part_names=["a", "b"],
                weights=[1.0, 1.0],
                context=context(changed_stat),
            )

    def test_manifest_schema_and_mapping_entrypoint_are_mandatory(self) -> None:
        missing_row_count = copy.deepcopy(self.pima)
        del missing_row_count["row_count"]
        with self.assertRaises(GroupManifestError):
            generate_balanced_partition(
                missing_row_count,
                part_names=["a", "b"],
                weights=[1.0, 1.0],
                context=context(missing_row_count),
            )
        with self.assertRaises(SplitPlanningError):
            generate_balanced_partition(
                [GroupRecord("g0", (0,), 1, 1, 0), GroupRecord("g1", (1,), 1, 0, 1)],
                part_names=["a", "b"],
                weights=[1.0, 1.0],
                context=context(self.pima),
            )
        extra_top_level = copy.deepcopy(self.pima)
        extra_top_level["outcome_nonce"] = "unregistered"
        extra_top_level["group_manifest_sha256"] = group_manifest_fingerprint(extra_top_level)
        with self.assertRaises(GroupManifestError):
            generate_balanced_partition(
                extra_top_level,
                part_names=["a", "b"],
                weights=[1.0, 1.0],
                context=context(extra_top_level),
            )
        extra_group_field = copy.deepcopy(self.pima)
        extra_group_field["groups"][0]["outcome_nonce"] = "unregistered"
        extra_group_field["group_manifest_sha256"] = group_manifest_fingerprint(extra_group_field)
        with self.assertRaises(GroupManifestError):
            generate_balanced_partition(
                extra_group_field,
                part_names=["a", "b"],
                weights=[1.0, 1.0],
                context=context(extra_group_field),
            )


class BalancedPartitionTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.pima = build_group_manifest(
            load_csv_rows(
                ROOT / "data" / "diabetes.csv",
                dataset="pima",
                target="Outcome",
                predictors=PIMA_PREDICTORS,
            )
        )
        cls.retino = build_group_manifest(
            load_arff_rows(
                ROOT / "data" / "raw" / "diabetic_retinopathy_debrecen" / "messidor_features.arff",
                dataset="retinopathy",
                target="Class",
                predictors=RETINOPATHY_PREDICTORS,
            )
        )

    def _outer(self, manifest: dict[str, object], repeat: int = 0) -> dict[str, object]:
        return generate_balanced_partition(
            manifest,
            part_names=[f"fold_{index}" for index in range(5)],
            weights=[1, 1, 1, 1, 1],
            context=context(manifest, repeat=repeat),
            constraints=BalanceConstraints(min_positive_per_part=30, min_negative_per_part=30),
        )

    def test_real_outer_partitions_are_balanced_and_group_safe(self) -> None:
        pima = self._outer(self.pima)
        retino = self._outer(self.retino)
        self.assertEqual(pima["candidate_attempts"], 64)
        self.assertEqual(retino["candidate_attempts"], 64)
        self.assertTrue(self.pima["predictor_allowlist_explicit"])
        self.assertTrue(self.retino["predictor_allowlist_explicit"])
        self.assertEqual(len(pima["candidate_audit"]), 64)
        self.assertEqual(len(retino["candidate_audit"]), 64)
        self.assertTrue(retino["label_stratified"])
        self.assertEqual(retino["label_fields_used"], ["positive", "negative", "prevalence"])
        pima_stats = list(pima["parts"].values())
        self.assertTrue(all((row["n"] in (153, 154) for row in pima_stats)))
        self.assertTrue(all((row["positive"] in (53, 54) for row in pima_stats)))
        retino_stats = list(retino["parts"].values())
        self.assertTrue(all((row["n"] in (230, 231) for row in retino_stats)))
        self.assertTrue(all((row["positive"] in (122, 123) for row in retino_stats)))
        assignment = retino["assignment"]
        row_to_part: dict[int, str] = {}
        for group in self.retino["groups"]:
            self.assertIn(group["group_id"], assignment)
            for row_id in group["row_ids"]:
                self.assertNotIn(row_id, row_to_part)
                row_to_part[row_id] = assignment[group["group_id"]]
            if group["n"] == 2:
                self.assertEqual(len({row_to_part[row_id] for row_id in group["row_ids"]}), 1)
        self.assertEqual(set(row_to_part), set(range(self.retino["row_count"])))

    def test_same_input_is_deterministic_and_repeats_differ(self) -> None:
        first = self._outer(self.retino, repeat=0)
        second = self._outer(copy.deepcopy(self.retino), repeat=0)
        third = self._outer(self.retino, repeat=1)
        self.assertEqual(first["selected_assignment_sha256"], second["selected_assignment_sha256"])
        self.assertEqual(first["assignment"], second["assignment"])
        self.assertNotEqual(
            first["selected_assignment_sha256"], third["selected_assignment_sha256"]
        )

    def test_attempt_count_is_fail_closed(self) -> None:
        with self.assertRaises(SplitPlanningError):
            generate_balanced_partition(
                self.pima,
                part_names=["fold_0", "fold_1"],
                weights=[1, 1],
                context=context(self.pima),
                candidate_attempts=65,
            )

    def test_infeasible_constraints_return_certificate(self) -> None:
        manifest = build_group_manifest(
            DatasetRows(
                dataset="infeasible",
                source_path="synthetic",
                source_sha256="c" * 64,
                feature_names=("x",),
                features=tuple(((float(index),) for index in range(4))),
                labels=(1, 1, 1, 1),
                row_ids=(0, 1, 2, 3),
            )
        )
        with self.assertRaises(InfeasiblePartitionError) as caught:
            generate_balanced_partition(
                manifest, part_names=["a", "b"], weights=[1, 1], context=context(manifest)
            )
        certificate = caught.exception.certificate
        self.assertEqual(certificate["candidate_attempts"], 64)
        self.assertFalse(certificate["automatic_relaxation"])
        self.assertEqual(len(certificate["candidate_summaries"]), 64)
        self.assertTrue(certificate["label_stratified"])
        self.assertEqual(certificate["label_fields_used"], ["positive", "negative", "prevalence"])

    def test_context_hashes_must_match_manifest(self) -> None:
        wrong_dataset = context(self.pima)
        wrong_dataset["dataset_sha256"] = "0" * 64
        with self.assertRaises(SplitPlanningError):
            generate_balanced_partition(
                self.pima, part_names=["a", "b"], weights=[1.0, 1.0], context=wrong_dataset
            )
        wrong_manifest = context(self.pima)
        wrong_manifest["group_manifest_sha256"] = "1" * 64
        with self.assertRaises(SplitPlanningError):
            generate_balanced_partition(
                self.pima, part_names=["a", "b"], weights=[1.0, 1.0], context=wrong_manifest
            )

    def test_context_serialization_has_no_pipe_collision(self) -> None:
        first_context = context(self.pima)
        first_context["study_id"] = "a|b"
        first_context["scope"] = "c"
        second_context = context(self.pima)
        second_context["study_id"] = "a"
        second_context["scope"] = "b|c"
        first = generate_balanced_partition(
            self.pima, part_names=["a", "b"], weights=[1.0, 1.0], context=first_context
        )
        second = generate_balanced_partition(
            self.pima, part_names=["a", "b"], weights=[1.0, 1.0], context=second_context
        )
        self.assertNotEqual(first["context_sha256"], second["context_sha256"])

    def test_context_and_weights_use_strict_registered_types(self) -> None:
        extra = context(self.pima)
        extra["unregistered"] = "metadata"
        with self.assertRaises(SplitPlanningError):
            generate_balanced_partition(
                self.pima, part_names=["a", "b"], weights=[1.0, 1.0], context=extra
            )
        bad_fold = context(self.pima)
        bad_fold["outer_fold"] = 0.0
        with self.assertRaises(SplitPlanningError):
            generate_balanced_partition(
                self.pima, part_names=["a", "b"], weights=[1.0, 1.0], context=bad_fold
            )
        with self.assertRaises(SplitPlanningError):
            generate_balanced_partition(
                self.pima, part_names=["a", "b"], weights=["1", "1"], context=context(self.pima)
            )
        with self.assertRaises(SplitPlanningError):
            generate_balanced_partition(
                self.pima, part_names=["a", "b"], weights=[1e308, 1e308], context=context(self.pima)
            )


if __name__ == "__main__":
    unittest.main()
