from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import copy
import unittest
from functools import lru_cache
from pathlib import Path
from fedsift.preprocessing import DEBRECEN_PROFILE, PIMA_PRIMARY_PROFILE
from fedsift.study_factory import (
    PROPOSAL_STATUS,
    StudyConstruction,
    StudyProposal,
    StudyFactoryError,
    build_study_bundle,
    build_study_construction,
    propose_study,
    study_bundle_fingerprint,
    validate_study_bundle,
    validate_study_bundle_integrity,
    validate_study_construction,
    validate_study_proposal,
)

ROOT = Path(__file__).resolve().parents[1]
STUDY_ID = _identity("real_data_factory_test")
PROTOCOL_SHA256 = "a" * 64
IMPLEMENTATION_SHA256 = "b" * 64
CLIENT_POLICY_SHA256 = "356fac82397469a91b4da89e27d7557590236d4fdf0df20259881c6bbff423c7"
CANDIDATE_COUNT = 10
HPO_SEEDS = (101,)
MAX_STEPS = 9
EXPECTED = {
    "pima": {
        "profile": PIMA_PRIMARY_PROFILE,
        "bundle_sha256": "1acbf5cbd7e299b5c358bfef2ce3adf7103254829f4fd3cebda83ea47f7b26fb",
        "source_sha256": "b78029447fae2743b3218bb2b76ef0d04afe8d7e55ce2faf4d1ec82d8f8ae8ac",
        "row_count": 768,
        "positive_count": 268,
        "negative_count": 500,
        "feature_count": 8,
        "group_count": 768,
        "max_group_size": 1,
        "group_manifest_sha256": "6c195bfdc9fa3ad339ae2a95917574759da05ecfdf87d6c775f8480581fd26fa",
        "row_to_group_sha256": "ce842396aac37d0df78f5e7dc884d89b38d66fd1878047115cb33b83cc1ee29b",
        "nested_plan_sha256": "1bdce67c9cd948b90ade10b92e82ffde2e908c15b4ee9ea6b29e46a4a9d02962",
        "freeze_binding_sha256": "43c57baea1c1633544f9b188077157012b708a87518ae6828295ec2dbba766bd",
        "candidate_space_sha256": "594a26ebe5b63c7ec694b50af03c6f8f98eeee38b1053b57b3bdc3f86b499c2a",
        "hpo_plan_sha256": "9078754d9a77b1db6f3f4440299bdbd92b930eae37b7aed7897f282553d518d0",
    },
    "retinopathy": {
        "profile": DEBRECEN_PROFILE,
        "bundle_sha256": "59c6a59260d40bc893d3af7a32528b234c4838e08ff7951d4dd876f33172c2f5",
        "source_sha256": "b83485dd519127ac1ba06a95da5f1d12d4bde2a362eb32d7d353bdcabfa08a93",
        "row_count": 1151,
        "positive_count": 611,
        "negative_count": 540,
        "feature_count": 19,
        "group_count": 1146,
        "max_group_size": 2,
        "group_manifest_sha256": "fe1b8c7d326eeece9010b78965a854cf078f0cad19a6e60171607fb540167c7c",
        "row_to_group_sha256": "6fc6e1f1c0c0dd75374178a3bf8a92504ea166fdb850294e0ef91e60ab1b18ac",
        "nested_plan_sha256": "4d834ebed1f617a0dca5987f948a0f0c356e94f3ddfef232c5b80661bfec35b1",
        "freeze_binding_sha256": "8546b2bcf7ca454f3d120da5985990da204ba8e04fd513abbc7e6df578a4a6b5",
        "candidate_space_sha256": "594a26ebe5b63c7ec694b50af03c6f8f98eeee38b1053b57b3bdc3f86b499c2a",
        "hpo_plan_sha256": "2b6fc8591635a225183e57aa5eab62b279ce6bc7e442a4c68fa18bbf76abfb6c",
    },
}


def _kwargs(dataset_id: str) -> dict[str, object]:
    return {
        "study_id": STUDY_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "implementation_sha256": IMPLEMENTATION_SHA256,
        "client_policy_sha256": CLIENT_POLICY_SHA256,
        "candidate_count": CANDIDATE_COUNT,
        "hpo_seeds": HPO_SEEDS,
        "max_steps": MAX_STEPS,
        "preprocessing_profile_name": EXPECTED[dataset_id]["profile"],
        "expected_bundle_sha256": EXPECTED[dataset_id]["bundle_sha256"],
    }


@lru_cache(maxsize=2)
def _real_construction(dataset_id: str) -> StudyConstruction:
    return build_study_construction(ROOT, dataset_id, **_kwargs(dataset_id))


def _real_bundle(dataset_id: str) -> dict[str, object]:
    return _real_construction(dataset_id).bundle


@lru_cache(maxsize=1)
def _real_pima_proposal() -> StudyProposal:
    arguments = _kwargs("pima")
    arguments.pop("expected_bundle_sha256")
    return propose_study(ROOT, "pima", **arguments)


def _all_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(key.strip().lower().replace("-", "_"))
            keys.update(_all_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_all_keys(child))
    return keys


class RealDataStudyFactoryTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.constructions = {
            dataset_id: _real_construction(dataset_id) for dataset_id in ("pima", "retinopathy")
        }
        cls.bundles = {
            dataset_id: construction.bundle
            for (dataset_id, construction) in cls.constructions.items()
        }

    def test_exact_real_source_group_plan_counts_and_deterministic_hashes(self) -> None:
        for dataset_id, expected in EXPECTED.items():
            with self.subTest(dataset_id=dataset_id):
                bundle = self.bundles[dataset_id]
                source = bundle["source_binding"]
                group = bundle["group_binding"]
                nested = bundle["nested_plan_binding"]
                candidate = bundle["candidate_space_binding"]
                hpo = bundle["hpo_plan_binding"]
                self.assertEqual(bundle["bundle_sha256"], expected["bundle_sha256"])
                self.assertEqual(bundle["bundle_sha256"], study_bundle_fingerprint(bundle))
                self.assertEqual(source["source_sha256"], expected["source_sha256"])
                self.assertEqual(
                    source["registered_source_uri"],
                    f"registered://{dataset_id}/{expected['source_sha256']}",
                )
                for field in ("row_count", "positive_count", "negative_count", "feature_count"):
                    self.assertEqual(source[field], expected[field])
                for field in (
                    "group_count",
                    "max_group_size",
                    "group_manifest_sha256",
                    "row_to_group_sha256",
                ):
                    self.assertEqual(group[field], expected[field])
                self.assertEqual(group["mixed_label_group_count"], 0)
                self.assertEqual(nested["nested_plan_sha256"], expected["nested_plan_sha256"])
                self.assertEqual(nested["freeze_binding_sha256"], expected["freeze_binding_sha256"])
                self.assertEqual(
                    (
                        candidate["candidate_space_sha256"],
                        candidate["method_count"],
                        candidate["candidate_count_per_method"],
                        candidate["total_candidate_count"],
                    ),
                    (expected["candidate_space_sha256"], 10, CANDIDATE_COUNT, 100),
                )
                self.assertEqual(
                    (
                        nested["outer_repeats"],
                        nested["outer_folds_per_repeat"],
                        nested["inner_folds_per_outer_fold"],
                        nested["outer_scope_count"],
                        nested["inner_scope_count"],
                        nested["split_plan_count"],
                        nested["unique_split_plan_count"],
                    ),
                    (3, 5, 3, 15, 45, 138, 138),
                )
                self.assertEqual(hpo["hpo_plan_sha256"], expected["hpo_plan_sha256"])
                self.assertEqual(
                    (
                        hpo["expected_unit_count"],
                        hpo["expected_units_per_method"],
                        hpo["expected_units_per_outer_method"],
                    ),
                    (4500, 450, 30),
                )
                validate_study_bundle_integrity(
                    bundle, expected_bundle_sha256=expected["bundle_sha256"]
                )

    def test_bundle_is_compact_and_has_no_runtime_or_observed_output_fields(self) -> None:
        forbidden = {
            "accuracy",
            "average_precision",
            "auroc",
            "brier_score",
            "gate",
            "gates",
            "loss",
            "metric",
            "metrics",
            "outer_data",
            "outer_features",
            "outer_labels",
            "outer_rows",
            "outer_test",
            "prediction",
            "predictions",
            "score",
            "scores",
            "selection",
            "selection_gate",
        }
        for dataset_id, bundle in self.bundles.items():
            with self.subTest(dataset_id=dataset_id):
                self.assertFalse(_all_keys(bundle) & forbidden)
                boundary = bundle["preexecution_boundary"]
                self.assertEqual(boundary["returned_payload"], "compact_hashes_and_counts_only")
                for field, value in boundary.items():
                    if field not in {"source_loader", "returned_payload"}:
                        self.assertIs(value, False)

    def test_closed_construction_api_and_rebuild_validator(self) -> None:
        construction = self.constructions["pima"]
        self.assertIsInstance(construction, StudyConstruction)
        self.assertEqual(construction.dataset_rows.dataset, "pima")
        self.assertEqual(
            construction.group_manifest["group_manifest_sha256"],
            construction.bundle["group_binding"]["group_manifest_sha256"],
        )
        self.assertEqual(construction.frozen_nested_plan["status"], "FROZEN")
        self.assertEqual(construction.frozen_candidate_space["status"], "FROZEN")
        self.assertEqual(
            construction.preprocessing_profile_spec["profile_sha256"],
            construction.bundle["preprocessing_binding"]["profile_sha256"],
        )
        self.assertEqual(len(construction.complete_hpo_plan["units"]), 4500)
        validate_study_construction(construction, ROOT, "pima", **_kwargs("pima"))
        forged = copy.deepcopy(construction)
        forged.preprocessing_profile_spec["dataset"] = "retinopathy"
        with self.assertRaises(StudyFactoryError):
            validate_study_construction(forged, ROOT, "pima", **_kwargs("pima"))

    def test_proposal_bootstrap_is_compact_and_cannot_authorize_execution(self) -> None:
        proposal = _real_pima_proposal()
        self.assertEqual(proposal.status, PROPOSAL_STATUS)
        self.assertEqual(proposal.proposed_bundle_sha256, EXPECTED["pima"]["bundle_sha256"])
        self.assertEqual(proposal.bundle, self.bundles["pima"])
        validate_study_proposal(proposal)
        with self.assertRaises(StudyFactoryError):
            validate_study_construction(proposal, ROOT, "pima", **_kwargs("pima"))
        with self.assertRaises(StudyFactoryError):
            StudyConstruction()

    def test_wrong_workspace_root_fails_before_any_plan_can_be_built(self) -> None:
        with self.assertRaises(StudyFactoryError):
            build_study_bundle(ROOT / "code", "pima", **_kwargs("pima"))

    def test_dataset_profile_mismatch_fails_closed(self) -> None:
        arguments = _kwargs("retinopathy")
        arguments["preprocessing_profile_name"] = PIMA_PRIMARY_PROFILE
        with self.assertRaises(StudyFactoryError):
            build_study_bundle(ROOT, "retinopathy", **arguments)
        with self.assertRaises(StudyFactoryError):
            validate_study_bundle(self.bundles["pima"], ROOT, "retinopathy", **_kwargs("pima"))

    def test_explicit_hash_candidate_seed_and_step_drift_fail_fast(self) -> None:
        base = self.bundles["pima"]
        drifts = (
            {"protocol_sha256": "c" * 64},
            {"implementation_sha256": "d" * 64},
            {"client_policy_sha256": "e" * 64},
            {"candidate_count": CANDIDATE_COUNT + 1},
            {"hpo_seeds": (307,)},
            {"max_steps": MAX_STEPS + 1},
        )
        for change in drifts:
            with self.subTest(change=change), self.assertRaises(StudyFactoryError):
                arguments = _kwargs("pima")
                arguments.update(change)
                validate_study_bundle(base, ROOT, "pima", **arguments)

    def test_self_hash_and_external_expected_hash_drift_fail_closed(self) -> None:
        bundle = self.bundles["pima"]
        changed = copy.deepcopy(bundle)
        changed["group_binding"]["group_manifest_sha256"] = "0" * 64
        with self.assertRaises(StudyFactoryError):
            validate_study_bundle_integrity(
                changed, expected_bundle_sha256=EXPECTED["pima"]["bundle_sha256"]
            )
        with self.assertRaises(StudyFactoryError):
            validate_study_bundle_integrity(bundle, expected_bundle_sha256="0" * 64)

    def test_performance_field_pollution_is_rejected_even_when_rehashed(self) -> None:
        polluted = copy.deepcopy(self.bundles["pima"])
        polluted["metrics"] = {"loss": 0.0}
        polluted["bundle_sha256"] = study_bundle_fingerprint(polluted)
        with self.assertRaises(StudyFactoryError):
            validate_study_bundle_integrity(
                polluted, expected_bundle_sha256=polluted["bundle_sha256"]
            )


if __name__ == "__main__":
    unittest.main()
