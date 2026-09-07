from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import copy
import inspect
import math
import unittest
from pathlib import Path
from fedsift.group_manifest import DatasetRows, build_group_manifest, load_arff_rows, load_csv_rows
from fedsift.candidate_space import canonical_sha256
from fedsift.hpo_capability import build_hpo_unit_capability
from fedsift.preprocessing import (
    DEBRECEN_PROFILE,
    PIMA_PRIMARY_PROFILE,
    PIMA_SENSITIVITY_PROFILE,
    PreprocessedRoles,
    PreprocessingError,
    HPO_ROLE_NAMES,
    preprocess_roles,
    preprocess_hpo_capability,
    preprocessing_artifact_fingerprint,
    validate_hpo_preprocessed_roles,
    validate_preprocessed_roles,
)
from tests.test_candidate_hpo_plan import SMALL_CANDIDATE_COUNT, cached_plan, hpo_rows_fixture

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
DEBRECEN_PREDICTORS = tuple((str(index) for index in range(19)))


def _synthetic_pima() -> DatasetRows:
    features: list[tuple[float, ...]] = []
    labels: list[int] = []
    for index in range(40):
        features.append(
            (
                float(index % 7),
                0.0 if index % 4 == 0 else 90.0 + index,
                0.0 if index % 5 == 0 else 60.0 + index / 2.0,
                0.0 if index % 6 == 0 else 15.0 + index / 3.0,
                0.0 if index % 3 == 0 else 70.0 + index * 2.0,
                0.0 if index % 7 == 0 else 20.0 + index / 4.0,
                0.1 + index / 100.0,
                20.0 + index,
            )
        )
        labels.append(index % 2)
    return DatasetRows(
        dataset="pima",
        source_path="synthetic://pima-preprocessing-v1",
        source_sha256="a" * 64,
        feature_names=PIMA_PREDICTORS,
        features=tuple(features),
        labels=tuple(labels),
        row_ids=tuple(range(40)),
        target_name="Outcome",
        predictor_allowlist_explicit=True,
    )


def _synthetic_roles() -> dict[str, tuple[int, ...]]:
    return {
        "private": tuple(range(10, 25)),
        "v_ctrl": tuple(range(0, 10)),
        "v_sel": tuple(range(25, 32)),
        "inner_validation": tuple(range(32, 40)),
    }


def _replace_features(
    rows: DatasetRows,
    features: tuple[tuple[float, ...], ...],
    *,
    feature_names: tuple[str, ...] | None = None,
    allowlist_explicit: bool | None = None,
) -> DatasetRows:
    return DatasetRows(
        dataset=rows.dataset,
        source_path=rows.source_path,
        source_sha256=rows.source_sha256,
        feature_names=feature_names or rows.feature_names,
        features=features,
        labels=rows.labels,
        row_ids=rows.row_ids,
        target_name=rows.target_name,
        predictor_allowlist_explicit=(
            rows.predictor_allowlist_explicit if allowlist_explicit is None else allowlist_explicit
        ),
    )


class SyntheticPreprocessingTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = _synthetic_pima()
        cls.roles = _synthetic_roles()
        cls.group_sha256 = build_group_manifest(cls.rows)["group_manifest_sha256"]

    def test_primary_is_fitted_only_on_v_ctrl_and_outputs_are_finite(self) -> None:
        result = preprocess_roles(
            self.rows,
            self.roles,
            profile=PIMA_PRIMARY_PROFILE,
            group_manifest_sha256=self.group_sha256,
        )
        self.assertEqual(result.artifact["schema"], _identity("preprocessing_artifact"))
        self.assertEqual(result.artifact["status"], "RESULT_BLIND_V_CTRL_FEATURES_ONLY")
        self.assertEqual(result.artifact["fit"]["fit_role"], "v_ctrl")
        self.assertFalse(result.artifact["fit"]["labels_read"])
        self.assertFalse(result.artifact["fit"]["non_v_ctrl_features_used_for_fit"])
        self.assertFalse(result.artifact["fit"]["outer_test_features_used"])
        self.assertEqual(
            result.artifact["privacy_boundary"]["claim"],
            "conditional_record_level_dp_given_fixed_auxiliary_Z",
        )
        self.assertEqual(set(result.matrices), set(self.roles))
        for role, row_ids in self.roles.items():
            matrix = result.matrices[role]
            self.assertEqual(len(matrix), len(row_ids))
            self.assertTrue(all((len(row) == 8 for row in matrix)))
            self.assertTrue(all((math.isfinite(value) for row in matrix for value in row)))
            self.assertEqual(result.artifact["roles"][role]["matrix_shape"], [len(row_ids), 8])
        validate_preprocessed_roles(
            result,
            self.rows,
            self.roles,
            profile=PIMA_PRIMARY_PROFILE,
            group_manifest_sha256=self.group_sha256,
        )

    def test_registered_pima_profiles_differ_only_by_predeclared_zero_rule(self) -> None:
        primary = preprocess_roles(
            self.rows,
            self.roles,
            profile=PIMA_PRIMARY_PROFILE,
            group_manifest_sha256=self.group_sha256,
        )
        sensitivity = preprocess_roles(
            self.rows,
            self.roles,
            profile=PIMA_SENSITIVITY_PROFILE,
            group_manifest_sha256=self.group_sha256,
        )
        self.assertNotEqual(
            primary.artifact["profile"]["profile_sha256"],
            sensitivity.artifact["profile"]["profile_sha256"],
        )
        primary_glucose = primary.artifact["parameters"][1]
        sensitivity_glucose = sensitivity.artifact["parameters"][1]
        self.assertTrue(primary_glucose["zero_as_missing"])
        self.assertFalse(sensitivity_glucose["zero_as_missing"])
        self.assertLess(
            primary_glucose["v_ctrl_observed_count"], sensitivity_glucose["v_ctrl_observed_count"]
        )
        self.assertNotEqual(primary_glucose["median"], sensitivity_glucose["median"])

    def test_non_v_ctrl_feature_changes_cannot_change_fit_parameters(self) -> None:
        original = preprocess_roles(
            self.rows,
            self.roles,
            profile=PIMA_PRIMARY_PROFILE,
            group_manifest_sha256=self.group_sha256,
        )
        changed_features = [list(row) for row in self.rows.features]
        for row_id in (
            *self.roles["private"],
            *self.roles["v_sel"],
            *self.roles["inner_validation"],
        ):
            changed_features[row_id] = [
                100000.0 + row_id + column for column in range(len(PIMA_PREDICTORS))
            ]
        changed_rows = _replace_features(self.rows, tuple((tuple(row) for row in changed_features)))
        changed = preprocess_roles(
            changed_rows,
            self.roles,
            profile=PIMA_PRIMARY_PROFILE,
            group_manifest_sha256=self.group_sha256,
        )
        self.assertEqual(original.artifact["parameters"], changed.artifact["parameters"])
        self.assertEqual(
            original.artifact["fit"]["fit_raw_feature_matrix_sha256"],
            changed.artifact["fit"]["fit_raw_feature_matrix_sha256"],
        )
        self.assertEqual(original.matrices["v_ctrl"], changed.matrices["v_ctrl"])
        self.assertNotEqual(original.matrices["private"], changed.matrices["private"])

    def test_outer_test_plan_duplicate_overlap_and_bad_ids_fail_closed(self) -> None:
        outer = dict(self.roles)
        outer["outer_test"] = (39,)
        with self.assertRaises(PreprocessingError):
            preprocess_roles(
                self.rows,
                outer,
                profile=PIMA_PRIMARY_PROFILE,
                group_manifest_sha256=self.group_sha256,
            )
        with self.assertRaises(PreprocessingError):
            preprocess_roles(
                self.rows,
                {"schema": _identity("nested_plan")},
                profile=PIMA_PRIMARY_PROFILE,
                group_manifest_sha256=self.group_sha256,
            )
        duplicate = dict(self.roles)
        duplicate["private"] = (10, 10, *range(11, 25))
        with self.assertRaises(PreprocessingError):
            preprocess_roles(
                self.rows,
                duplicate,
                profile=PIMA_PRIMARY_PROFILE,
                group_manifest_sha256=self.group_sha256,
            )
        overlap = dict(self.roles)
        overlap["v_sel"] = (24, *range(25, 32))
        with self.assertRaises(PreprocessingError):
            preprocess_roles(
                self.rows,
                overlap,
                profile=PIMA_PRIMARY_PROFILE,
                group_manifest_sha256=self.group_sha256,
            )
        out_of_range = dict(self.roles)
        out_of_range["inner_validation"] = (32, 40)
        with self.assertRaises(PreprocessingError):
            preprocess_roles(
                self.rows,
                out_of_range,
                profile=PIMA_PRIMARY_PROFILE,
                group_manifest_sha256=self.group_sha256,
            )

    def test_unknown_profile_columns_and_implicit_allowlist_fail_closed(self) -> None:
        with self.assertRaises(PreprocessingError):
            preprocess_roles(
                self.rows,
                self.roles,
                profile="choose_whichever_wins",
                group_manifest_sha256=self.group_sha256,
            )
        renamed = tuple(["UnknownColumn", *PIMA_PREDICTORS[1:]])
        changed_columns = _replace_features(self.rows, self.rows.features, feature_names=renamed)
        with self.assertRaises(PreprocessingError):
            preprocess_roles(
                changed_columns,
                self.roles,
                profile=PIMA_PRIMARY_PROFILE,
                group_manifest_sha256=self.group_sha256,
            )
        implicit = _replace_features(self.rows, self.rows.features, allowlist_explicit=False)
        with self.assertRaises(PreprocessingError):
            preprocess_roles(
                implicit,
                self.roles,
                profile=PIMA_PRIMARY_PROFILE,
                group_manifest_sha256=self.group_sha256,
            )

    def test_all_missing_v_ctrl_and_non_finite_transform_fail_closed(self) -> None:
        all_missing = [list(row) for row in self.rows.features]
        for row_id in self.roles["v_ctrl"]:
            all_missing[row_id][1] = 0.0
        missing_rows = _replace_features(self.rows, tuple((tuple(row) for row in all_missing)))
        with self.assertRaises(PreprocessingError):
            preprocess_roles(
                missing_rows,
                self.roles,
                profile=PIMA_PRIMARY_PROFILE,
                group_manifest_sha256=self.group_sha256,
            )
        infinite = [list(row) for row in self.rows.features]
        infinite[self.roles["private"][0]][0] = float("inf")
        infinite_rows = _replace_features(self.rows, tuple((tuple(row) for row in infinite)))
        with self.assertRaises(PreprocessingError):
            preprocess_roles(
                infinite_rows,
                self.roles,
                profile=PIMA_PRIMARY_PROFILE,
                group_manifest_sha256=self.group_sha256,
            )
        finite_overflow = [list(row) for row in self.rows.features]
        for offset, row_id in enumerate(self.roles["v_ctrl"], start=1):
            finite_overflow[row_id][0] = offset * 1e-100
        finite_overflow[self.roles["private"][0]][0] = 1e308
        overflow_rows = _replace_features(self.rows, tuple((tuple(row) for row in finite_overflow)))
        with self.assertRaises(PreprocessingError):
            preprocess_roles(
                overflow_rows,
                self.roles,
                profile=PIMA_PRIMARY_PROFILE,
                group_manifest_sha256=self.group_sha256,
            )

    def test_artifact_or_matrix_tampering_is_rejected_after_rehash(self) -> None:
        result = preprocess_roles(
            self.rows,
            self.roles,
            profile=PIMA_PRIMARY_PROFILE,
            group_manifest_sha256=self.group_sha256,
        )
        artifact = copy.deepcopy(result.artifact)
        artifact["parameters"][0]["median"] += 1.0
        artifact["artifact_sha256"] = preprocessing_artifact_fingerprint(artifact)
        tampered_artifact = PreprocessedRoles(
            artifact=artifact, matrices=copy.deepcopy(result.matrices)
        )
        with self.assertRaises(PreprocessingError):
            validate_preprocessed_roles(
                tampered_artifact,
                self.rows,
                self.roles,
                profile=PIMA_PRIMARY_PROFILE,
                group_manifest_sha256=self.group_sha256,
            )
        matrices = copy.deepcopy(result.matrices)
        private = [list(row) for row in matrices["private"]]
        private[0][0] += 1.0
        matrices["private"] = tuple((tuple(row) for row in private))
        tampered_matrix = PreprocessedRoles(
            artifact=copy.deepcopy(result.artifact), matrices=matrices
        )
        with self.assertRaises(PreprocessingError):
            validate_preprocessed_roles(
                tampered_matrix,
                self.rows,
                self.roles,
                profile=PIMA_PRIMARY_PROFILE,
                group_manifest_sha256=self.group_sha256,
            )


class RealDataPreprocessingTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.pima_rows = load_csv_rows(
            ROOT / "data" / "diabetes.csv",
            dataset="pima",
            target="Outcome",
            predictors=PIMA_PREDICTORS,
        )
        cls.pima_manifest = build_group_manifest(cls.pima_rows)
        cls.retino_rows = load_arff_rows(
            ROOT / "data" / "raw" / "diabetic_retinopathy_debrecen" / "messidor_features.arff",
            dataset="retinopathy",
            target="Class",
            predictors=DEBRECEN_PREDICTORS,
        )
        cls.retino_manifest = build_group_manifest(cls.retino_rows)

    def test_current_pima_profiles_have_bound_finite_shapes_without_target(self) -> None:
        roles = {
            "v_ctrl": tuple(range(0, 100)),
            "private": tuple(range(100, 500)),
            "v_sel": tuple(range(500, 600)),
            "inner_validation": tuple(range(600, 768)),
        }
        for profile in (PIMA_PRIMARY_PROFILE, PIMA_SENSITIVITY_PROFILE):
            result = preprocess_roles(
                self.pima_rows,
                roles,
                profile=profile,
                group_manifest_sha256=self.pima_manifest["group_manifest_sha256"],
            )
            self.assertEqual(
                result.artifact["input_binding"]["source_sha256"], self.pima_rows.source_sha256
            )
            self.assertNotIn(
                self.pima_rows.target_name, result.artifact["input_binding"]["predictor_allowlist"]
            )
            self.assertTrue(
                all(
                    (
                        math.isfinite(value)
                        for matrix in result.matrices.values()
                        for row in matrix
                        for value in row
                    )
                )
            )
            validate_preprocessed_roles(
                result,
                self.pima_rows,
                roles,
                profile=profile,
                group_manifest_sha256=self.pima_manifest["group_manifest_sha256"],
            )

    def test_current_debrecen_group_safe_roles_are_finite_and_bound(self) -> None:
        groups = self.retino_manifest["groups"]
        role_groups = {
            "v_ctrl": groups[:200],
            "private": groups[200:800],
            "v_sel": groups[800:950],
            "inner_validation": groups[950:],
        }
        roles = {
            role: tuple(
                sorted((row_id for group in selected_groups for row_id in group["row_ids"]))
            )
            for (role, selected_groups) in role_groups.items()
        }
        result = preprocess_roles(
            self.retino_rows,
            roles,
            profile=DEBRECEN_PROFILE,
            group_manifest_sha256=self.retino_manifest["group_manifest_sha256"],
        )
        self.assertEqual(len(result.artifact["parameters"]), 19)
        self.assertEqual(
            result.artifact["input_binding"]["group_manifest_sha256"],
            self.retino_manifest["group_manifest_sha256"],
        )
        self.assertTrue(
            all(
                (
                    math.isfinite(value)
                    for matrix in result.matrices.values()
                    for row in matrix
                    for value in row
                )
            )
        )
        for group in groups:
            if group["n"] == 2:
                locations = sum(
                    (all((row_id in roles[role] for row_id in group["row_ids"])) for role in roles)
                )
                self.assertEqual(locations, 1)
        validate_preprocessed_roles(
            result,
            self.retino_rows,
            roles,
            profile=DEBRECEN_PROFILE,
            group_manifest_sha256=self.retino_manifest["group_manifest_sha256"],
        )


class CapabilityScopedPreprocessingTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.space, cls.nested, cls.manifest, cls.plan = cached_plan(SMALL_CANDIDATE_COUNT)
        cls.rows = hpo_rows_fixture()
        duplicate = next((group for group in cls.manifest["groups"] if group["n"] > 1))
        cls.duplicate_group = duplicate
        chosen = None
        for unit in cls.plan["units"]:
            if (
                unit["method"] != "fedavg_nonprivate"
                or unit["candidate_id"] != "candidate_0000"
                or unit["hpo_seed"] != 11
            ):
                continue
            fold = cls.nested["repetitions"][unit["outer_repeat"]]["outer_folds"][
                unit["outer_fold"]
            ]
            if duplicate["group_id"] not in fold["outer_test"]["group_ids"]:
                chosen = unit
                break
        if chosen is None:
            raise AssertionError("no capability scope contains the duplicate group")
        cls.unit = chosen
        cls.capability = build_hpo_unit_capability(
            cls.plan, cls.space, cls.nested, cls.manifest, unit_id=cls.unit["unit_id"]
        )
        cls.result = preprocess_hpo_capability(
            cls.capability,
            cls.rows,
            cls.manifest,
            expected_capability_sha256=cls.capability["capability_sha256"],
        )

    @staticmethod
    def _rehash_capability(capability) -> None:
        payload = copy.deepcopy(capability)
        payload.pop("capability_sha256", None)
        capability["capability_sha256"] = canonical_sha256(payload)

    @staticmethod
    def _slice(capability, role):
        data = capability["data_slices"]
        if role.startswith("client_"):
            return data["private_clients"][role]
        return data[role]

    def test_formal_api_derives_exact_capability_roles_and_rebuilds(self) -> None:
        self.assertEqual(set(self.result.matrices), set(HPO_ROLE_NAMES))
        self.assertEqual(self.result.artifact["schema"], _identity("hpo_preprocessing_artifact"))
        self.assertEqual(
            self.result.artifact["input_binding"]["capability_sha256"],
            self.capability["capability_sha256"],
        )
        self.assertEqual(
            self.result.artifact["input_binding"]["preprocessing_profile_sha256"],
            self.plan["bindings"]["preprocessing_profile_sha256"],
        )
        self.assertEqual(set(self.result.artifact["capability_role_bindings"]), set(HPO_ROLE_NAMES))
        validate_hpo_preprocessed_roles(
            self.result,
            self.capability,
            self.rows,
            self.manifest,
            expected_capability_sha256=self.capability["capability_sha256"],
        )
        parameters = inspect.signature(preprocess_hpo_capability).parameters
        self.assertEqual(
            set(parameters), {"capability", "rows", "group_manifest", "expected_capability_sha256"}
        )
        self.assertNotIn("role_row_ids", parameters)
        self.assertNotIn("nested_plan", parameters)
        self.assertNotIn("profile", parameters)

    def test_fake_manifest_and_feature_or_label_drift_fail_closed(self) -> None:
        fake_manifest = copy.deepcopy(self.manifest)
        fake_manifest["group_manifest_sha256"] = "0" * 64
        with self.assertRaises(PreprocessingError):
            preprocess_hpo_capability(
                self.capability,
                self.rows,
                fake_manifest,
                expected_capability_sha256=self.capability["capability_sha256"],
            )
        features = [list(row) for row in self.rows.features]
        features[10][0] += 0.5
        changed_features = _replace_features(self.rows, tuple((tuple(row) for row in features)))
        with self.assertRaises(PreprocessingError):
            preprocess_hpo_capability(
                self.capability,
                changed_features,
                self.manifest,
                expected_capability_sha256=self.capability["capability_sha256"],
            )
        labels = list(self.rows.labels)
        labels[10] = 1 - labels[10]
        changed_labels = DatasetRows(
            dataset=self.rows.dataset,
            source_path=self.rows.source_path,
            source_sha256=self.rows.source_sha256,
            feature_names=self.rows.feature_names,
            features=self.rows.features,
            labels=tuple(labels),
            row_ids=self.rows.row_ids,
            target_name=self.rows.target_name,
            predictor_allowlist_explicit=True,
        )
        with self.assertRaises(PreprocessingError):
            preprocess_hpo_capability(
                self.capability,
                changed_labels,
                self.manifest,
                expected_capability_sha256=self.capability["capability_sha256"],
            )
        subject_manifest = build_group_manifest(
            self.rows,
            subject_ids=tuple((f"subject-{index}" for index in self.rows.row_ids)),
            subject_salt="test-only-salt",
            subject_id_provenance="synthetic test subject IDs",
        )
        with self.assertRaises(PreprocessingError):
            preprocess_hpo_capability(
                self.capability,
                self.rows,
                subject_manifest,
                expected_capability_sha256=self.capability["capability_sha256"],
            )

    def test_outer_row_capability_and_expected_hash_tampering_fail(self) -> None:
        fold = self.nested["repetitions"][self.unit["outer_repeat"]]["outer_folds"][
            self.unit["outer_fold"]
        ]
        outer_group = fold["outer_test"]["group_ids"][0]
        outer_row = next(
            (
                row_id
                for group in self.manifest["groups"]
                if group["group_id"] == outer_group
                for row_id in group["row_ids"]
            )
        )
        forged = copy.deepcopy(self.capability)
        v_ctrl = forged["data_slices"]["v_ctrl"]
        v_ctrl["row_ids"][0] = f"row_id:{outer_row:012d}"
        v_ctrl["row_ids"].sort()
        v_ctrl["row_ids_sha256"] = canonical_sha256(v_ctrl["row_ids"])
        self._rehash_capability(forged)
        with self.assertRaises(PreprocessingError):
            preprocess_hpo_capability(
                forged,
                self.rows,
                self.manifest,
                expected_capability_sha256=self.capability["capability_sha256"],
            )
        with self.assertRaises(PreprocessingError):
            preprocess_hpo_capability(
                self.capability, self.rows, self.manifest, expected_capability_sha256="0" * 64
            )

    def test_cross_role_duplicate_group_is_rejected_even_after_rehash(self) -> None:
        forged = copy.deepcopy(self.capability)
        duplicate_rows = set(self.duplicate_group["row_ids"])
        source_role = next(
            (
                role
                for role in HPO_ROLE_NAMES
                if duplicate_rows.issubset(
                    {int(token.split(":", 1)[1]) for token in self._slice(forged, role)["row_ids"]}
                )
            )
        )
        group_sizes = {
            row_id: group["n"] for group in self.manifest["groups"] for row_id in group["row_ids"]
        }
        destination_role = next(
            (
                role
                for role in HPO_ROLE_NAMES
                if role != source_role
                and any(
                    (
                        group_sizes[int(token.split(":", 1)[1])] == 1
                        for token in self._slice(forged, role)["row_ids"]
                    )
                )
            )
        )
        source = self._slice(forged, source_role)
        destination = self._slice(forged, destination_role)
        duplicate_token = f"row_id:{next(iter(duplicate_rows)):012d}"
        destination_token = next(
            (
                token
                for token in destination["row_ids"]
                if group_sizes[int(token.split(":", 1)[1])] == 1
            )
        )
        source["row_ids"].remove(duplicate_token)
        destination["row_ids"].remove(destination_token)
        source["row_ids"].append(destination_token)
        destination["row_ids"].append(duplicate_token)
        for value in (source, destination):
            value["row_ids"].sort()
            value["row_ids_sha256"] = canonical_sha256(value["row_ids"])
        self._rehash_capability(forged)
        with self.assertRaises(PreprocessingError):
            preprocess_hpo_capability(
                forged,
                self.rows,
                self.manifest,
                expected_capability_sha256=forged["capability_sha256"],
            )

    def test_formal_artifact_or_output_tampering_is_rejected(self) -> None:
        artifact = copy.deepcopy(self.result.artifact)
        artifact["output_binding"]["role_matrix_set_sha256"] = "f" * 64
        artifact["artifact_sha256"] = preprocessing_artifact_fingerprint(artifact)
        tampered = PreprocessedRoles(
            artifact=artifact, matrices=copy.deepcopy(self.result.matrices)
        )
        with self.assertRaises(PreprocessingError):
            validate_hpo_preprocessed_roles(
                tampered,
                self.capability,
                self.rows,
                self.manifest,
                expected_capability_sha256=self.capability["capability_sha256"],
            )
        matrices = copy.deepcopy(self.result.matrices)
        role = HPO_ROLE_NAMES[0]
        changed = [list(row) for row in matrices[role]]
        changed[0][0] += 1.0
        matrices[role] = tuple((tuple(row) for row in changed))
        with self.assertRaises(PreprocessingError):
            validate_hpo_preprocessed_roles(
                PreprocessedRoles(artifact=copy.deepcopy(self.result.artifact), matrices=matrices),
                self.capability,
                self.rows,
                self.manifest,
                expected_capability_sha256=self.capability["capability_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
