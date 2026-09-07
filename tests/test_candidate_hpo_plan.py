from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import copy
import hashlib
import unittest
from functools import lru_cache
from fedsift.candidate_space import (
    CONTROL_QUERY_EVERY,
    DEFAULT_CANDIDATE_COUNT,
    LOCAL_LR,
    MAIN_METHODS,
    METHOD_SOURCE_SENTINELS,
    METHOD_DIMENSIONS,
    MIN_CANDIDATE_COUNT,
    SIFT_DISABLE_OVERRIDE_THRESHOLD,
    SIFT_MIN_IMPROVEMENT,
    CandidateSpaceError,
    build_candidate_space,
    candidate_by_id,
    canonical_sha256,
    freeze_candidate_space,
    validate_candidate_space,
)
from fedsift.client_partition import client_partition_policy_fingerprint
from fedsift.group_manifest import DatasetRows, build_group_manifest
from fedsift.hpo_attempt_receipt import NONPRIVATE_PRIVACY_NOT_APPLICABLE, build_hpo_attempt_receipt
from fedsift.hpo_capability import _build_capability_payload
from fedsift.hpo_plan import (
    HpoPlanError,
    MATCHED_ABLATIONS,
    build_hpo_plan,
    close_hpo_ledger,
    inherit_matched_ablation,
    validate_hpo_plan,
    validate_selection_authorization,
)
from fedsift.nested_plan import freeze_nested_plan, generate_nested_plan
from fedsift.preprocessing import (
    PIMA_PRIMARY_PROFILE,
    PIMA_SENSITIVITY_PROFILE,
    preprocessing_profile,
)

HEX_A = "a" * 64
HEX_B = "b" * 64
STUDY_ID = _identity("hpo_test")
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
SMALL_CANDIDATE_COUNT = MIN_CANDIDATE_COUNT


@lru_cache(maxsize=1)
def hpo_rows_fixture():
    row_count = 500
    features = []
    for index in range(row_count):
        value = 0 if index == 1 else index
        features.append(
            (
                float(value % 12 + 1),
                float(80 + value),
                float(50 + value / 5),
                float(10 + value / 7),
                float(40 + value * 2),
                float(18 + value / 20),
                float(0.1 + value / 1000),
                float(20 + value % 70),
            )
        )
    return DatasetRows(
        dataset="pima",
        source_path=_identity("synthetic_hpo_pima_test"),
        source_sha256="c" * 64,
        feature_names=PIMA_PREDICTORS,
        features=tuple(features),
        labels=tuple((index % 2 for index in range(row_count))),
        row_ids=tuple(range(row_count)),
        target_name="Outcome",
        predictor_allowlist_explicit=True,
    )


@lru_cache(maxsize=1)
def hpo_profile_fixture():
    return preprocessing_profile(PIMA_PRIMARY_PROFILE)


@lru_cache(maxsize=1)
def frozen_nested_fixture():
    rows = hpo_rows_fixture()
    manifest = build_group_manifest(rows)
    draft = generate_nested_plan(manifest, study_id=STUDY_ID)
    frozen = freeze_nested_plan(
        draft,
        manifest,
        protocol_sha256=HEX_A,
        implementation_sha256=HEX_B,
        client_policy_sha256=client_partition_policy_fingerprint(),
    )
    return (manifest, draft, frozen)


def frozen_space(count: int = SMALL_CANDIDATE_COUNT):
    return freeze_candidate_space(
        build_candidate_space(STUDY_ID, candidate_count=count),
        protocol_sha256=HEX_A,
        implementation_contract_sha256=HEX_B,
    )


@lru_cache(maxsize=None)
def cached_plan(count: int = SMALL_CANDIDATE_COUNT):
    manifest, _, nested = frozen_nested_fixture()
    space = frozen_space(count)
    plan = build_hpo_plan(
        space,
        nested,
        manifest,
        preprocessing_profile_spec=hpo_profile_fixture(),
        hpo_seeds=(11,),
        max_steps=30,
    )
    return (space, nested, manifest, plan)


def small_plan(count: int = SMALL_CANDIDATE_COUNT):
    space, nested, manifest, plan = cached_plan(count)
    return (space, nested, manifest, copy.deepcopy(plan))


def _evidence_hash(kind: str, unit_id: str, attempt_index: int) -> str:
    return hashlib.sha256(f"{kind}:{unit_id}:{attempt_index}".encode("ascii")).hexdigest()


def _seal_entry_attempts(plan, space, nested, manifest, entry):
    unit_id = entry["unit_id"]
    capability = _build_capability_payload(plan, space, nested, manifest, unit_id)
    method = capability["unit_identity"]["method"]
    receipts = []
    for attempt in entry["attempts"]:
        attempt_index = attempt["attempt_index"]
        outcome = attempt["outcome"]
        nonprivate = method == "fedavg_nonprivate"
        privacy_report = (
            NONPRIVATE_PRIVACY_NOT_APPLICABLE
            if nonprivate
            else _evidence_hash("privacy-report", unit_id, attempt_index)
        )
        privacy_schedule = (
            NONPRIVATE_PRIVACY_NOT_APPLICABLE
            if nonprivate
            else _evidence_hash("privacy-schedule", unit_id, attempt_index)
        )
        receipt = build_hpo_attempt_receipt(
            capability,
            expected_capability_sha256=capability["capability_sha256"],
            attempt_index=attempt_index,
            outcome=outcome,
            failure_code=attempt.get("failure_code"),
            code_sha256=_evidence_hash("code", unit_id, attempt_index),
            environment_sha256=_evidence_hash("environment", unit_id, attempt_index),
            dependency_lock_sha256=_evidence_hash("dependencies", unit_id, attempt_index),
            privacy_accounting_report_sha256=privacy_report,
            privacy_schedule_sha256=privacy_schedule,
            exclusive_output_artifact_manifest_sha256=(
                _evidence_hash("output", unit_id, attempt_index) if outcome == "complete" else None
            ),
            failure_incident_sha256=(
                _evidence_hash("incident", unit_id, attempt_index) if outcome == "failed" else None
            ),
        )
        attempt["attempt_receipt_sha256"] = receipt["attempt_receipt_sha256"]
        receipts.append(receipt)
    return receipts


@lru_cache(maxsize=None)
def cached_success_evidence(count: int = SMALL_CANDIDATE_COUNT):
    space, nested, manifest, plan = cached_plan(count)
    ledger = []
    receipts = []
    for unit in plan["units"]:
        entry = {
            "unit_id": unit["unit_id"],
            "attempts": [{"attempt_index": 0, "outcome": "complete"}],
        }
        receipts.extend(_seal_entry_attempts(plan, space, nested, manifest, entry))
        ledger.append(entry)
    return (ledger, receipts)


def successful_evidence(plan, space, nested, manifest):
    count = space["candidate_count_per_method"]
    cached_plan_value = cached_plan(count)[3]
    if cached_plan_value["hpo_plan_sha256"] != plan["hpo_plan_sha256"]:
        raise AssertionError("test evidence fixture received a different plan")
    ledger, receipts = cached_success_evidence(count)
    return (copy.deepcopy(ledger), copy.deepcopy(receipts))


def refresh_entry_receipts(receipts, plan, space, nested, manifest, entry):
    receipts[:] = [receipt for receipt in receipts if receipt["unit_id"] != entry["unit_id"]]
    receipts.extend(_seal_entry_attempts(plan, space, nested, manifest, entry))


def close(plan, ledger, receipts, space, nested, manifest):
    return close_hpo_ledger(
        plan,
        ledger,
        attempt_receipts=receipts,
        candidate_space=space,
        nested_plan=nested,
        group_manifest=manifest,
    )


class CandidateSpaceTests(unittest.TestCase):

    def test_literature_driven_prefreeze_ranges_are_symmetric_and_cover_fedprox(self) -> None:
        self.assertEqual((LOCAL_LR.low, LOCAL_LR.high), (0.001, 3.1622776601683795))
        for method in MAIN_METHODS:
            self.assertIs(METHOD_DIMENSIONS[method][0], LOCAL_LR)
        prox_dimensions = {
            dimension.coupling_key: dimension
            for dimension in METHOD_DIMENSIONS["dp_fedprox_adapted"]
        }
        prox_mu = prox_dimensions["dp_fedprox/prox_mu"]
        self.assertEqual((prox_mu.low, prox_mu.high), (0.0001, 1.0))

    def test_primary_source_baseline_grids_are_not_narrowed(self) -> None:
        fedopt = {
            dimension.coupling_key: dimension for dimension in METHOD_DIMENSIONS["dp_fedadam"]
        }
        self.assertEqual(
            fedopt["shared/fedopt/server_learning_rate"].choices,
            (0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 1.0, 3.1622776601683795, 10.0),
        )
        self.assertEqual(
            fedopt["shared/fedopt/beta1"].choices, (0.0, 0.1, 0.25, 0.5, 0.8, 0.9, 0.95)
        )
        self.assertEqual(
            fedopt["shared/fedopt/beta2"].choices, (0.5, 0.8, 0.9, 0.99, 0.999, 0.9999)
        )
        self.assertEqual(
            fedopt["shared/fedopt/tau"].choices, (1e-05, 0.0001, 0.001, 0.01, 0.05, 0.1)
        )
        scaffold = {
            dimension.coupling_key: dimension
            for dimension in METHOD_DIMENSIONS["dp_scaffold_adapted"]
        }["dp_scaffold/server_learning_rate"]
        self.assertEqual((scaffold.low, scaffold.high), (0.01, 5.0))

    def test_result_informed_boundary_refinement_is_symmetric_and_covered(self) -> None:
        self.assertEqual(
            SIFT_MIN_IMPROVEMENT.choices,
            (
                0.0,
                0.001,
                0.0025,
                0.005,
                0.01,
                0.02,
                0.05,
                0.1,
                0.2,
                0.5,
                1.0,
                SIFT_DISABLE_OVERRIDE_THRESHOLD,
            ),
        )
        self.assertEqual(CONTROL_QUERY_EVERY.choices, (1, 2, 3, 5, 6, 10))
        space = build_candidate_space(STUDY_ID)
        fedadam_family = (
            "dp_fedadam",
            "dp_fedyogi",
            "time_dpfedadam",
            "public_argmin_time_dpfedadam",
            "fedsift",
        )
        expected_beta1 = {0.0, 0.1, 0.25, 0.5, 0.8, 0.9, 0.95}
        for method in fedadam_family:
            lhs = [
                candidate
                for candidate in space["methods"][method]["candidates"]
                if candidate["generation"]["kind"] == "lhs"
            ]
            self.assertEqual(
                {candidate["parameters"]["backend"]["beta1"] for candidate in lhs}, expected_beta1
            )
        fedsift_lhs = [
            candidate
            for candidate in space["methods"]["fedsift"]["candidates"]
            if candidate["generation"]["kind"] == "lhs"
        ]
        self.assertEqual(
            {
                candidate["parameters"]["sift"]["minimum_control_improvement"]
                for candidate in fedsift_lhs
            },
            set(SIFT_MIN_IMPROVEMENT.choices),
        )
        for method in ("public_argmin_time_dpfedadam", "fedsift"):
            lhs = [
                candidate
                for candidate in space["methods"][method]["candidates"]
                if candidate["generation"]["kind"] == "lhs"
            ]
            self.assertEqual(
                {
                    candidate["parameters"]["control_rule"]["query_every_rounds"]
                    for candidate in lhs
                },
                set(CONTROL_QUERY_EVERY.choices),
            )

    def test_default_rosters_have_source_sentinels_and_remaining_lhs(self) -> None:
        space = build_candidate_space(STUDY_ID)
        self.assertEqual(space["candidate_count_per_method"], DEFAULT_CANDIDATE_COUNT)
        for method in MAIN_METHODS:
            roster = space["methods"][method]
            self.assertEqual(roster["candidate_count"], DEFAULT_CANDIDATE_COUNT)
            self.assertEqual(len(roster["candidates"]), DEFAULT_CANDIDATE_COUNT)
            sentinel_count = len(METHOD_SOURCE_SENTINELS[method])
            self.assertEqual(roster["source_sentinel_count"], sentinel_count)
            self.assertEqual(roster["lhs_count"], DEFAULT_CANDIDATE_COUNT - sentinel_count - 1)
            self.assertGreater(roster["lhs_count"], 0)
            self.assertEqual(
                [candidate["generation"]["kind"] for candidate in roster["candidates"][1:]],
                ["source_sentinel"] * sentinel_count + ["lhs"] * roster["lhs_count"],
            )

    def test_every_source_sentinel_is_exactly_one_factor_from_candidate0(self) -> None:

        def leaf_values(value, path=()):
            if isinstance(value, dict):
                result = {}
                for key, child in value.items():
                    result.update(leaf_values(child, (*path, key)))
                return result
            return {path: value}

        space = build_candidate_space(STUDY_ID)
        for method in MAIN_METHODS:
            roster = space["methods"][method]
            reference = leaf_values(roster["candidates"][0]["parameters"])
            for candidate in roster["candidates"][1:]:
                generation = candidate["generation"]
                if generation["kind"] != "source_sentinel":
                    continue
                observed = leaf_values(candidate["parameters"])
                differences = {path for path in reference if reference[path] != observed[path]}
                expected_path = tuple(generation["changed_path"])
                self.assertEqual(differences, {expected_path})
                self.assertEqual(reference[expected_path], generation["reference_value"])
                self.assertEqual(observed[expected_path], generation["sentinel_value"])

    def test_required_primary_source_sentinel_values_are_explicit(self) -> None:
        space = build_candidate_space(STUDY_ID)
        for method in MAIN_METHODS:
            candidates = space["methods"][method]["candidates"]
            self.assertEqual(candidates[1]["parameters"]["local_optimizer"]["learning_rate"], 0.001)
            self.assertEqual(candidates[2]["parameters"]["local_optimizer"]["learning_rate"], 0.5)
        prox = space["methods"]["dp_fedprox_adapted"]["candidates"]
        prox_mu = {
            candidate["parameters"]["local_objective"]["prox_mu"]
            for candidate in prox
            if candidate["candidate_id"] == "candidate_0000"
            or candidate["generation"]["kind"] == "source_sentinel"
        }
        self.assertTrue({0.001, 0.01, 0.1, 1.0}.issubset(prox_mu))
        scaffold = space["methods"]["dp_scaffold_adapted"]["candidates"]
        self.assertEqual(
            {
                candidate["parameters"]["backend"]["server_learning_rate"]
                for candidate in scaffold
                if candidate["generation"]["kind"] == "source_sentinel"
                and candidate["generation"]["changed_path"] == ["backend", "server_learning_rate"]
            },
            {0.01, 5.0},
        )
        sofim = space["methods"]["dp_fedsofim_delta_proxy_adapted"]["candidates"]
        sofim_sentinels = [
            candidate for candidate in sofim if candidate["generation"]["kind"] == "source_sentinel"
        ]
        by_path = {}
        for candidate in sofim_sentinels:
            path = tuple(candidate["generation"]["changed_path"])
            by_path.setdefault(path, set()).add(candidate["generation"]["sentinel_value"])
        self.assertEqual(by_path["backend", "server_learning_rate"], {0.001, 5.0})
        self.assertEqual(by_path["backend", "rho"], {0.01, 20.0})
        self.assertEqual(by_path["backend", "beta"], {0.8, 0.99})

    def test_candidate_count_below_minimum_fails_closed(self) -> None:
        with self.assertRaises(CandidateSpaceError):
            build_candidate_space(STUDY_ID, candidate_count=MIN_CANDIDATE_COUNT - 1)
        space = build_candidate_space(STUDY_ID, candidate_count=MIN_CANDIDATE_COUNT)
        for method in MAIN_METHODS:
            self.assertEqual(len(space["methods"][method]["candidates"]), MIN_CANDIDATE_COUNT)
            self.assertGreaterEqual(space["methods"][method]["lhs_count"], 1)

    def test_exact_ten_equal_rosters_and_transparent_candidate_zero(self) -> None:
        space = build_candidate_space(STUDY_ID)
        self.assertEqual(tuple(space["method_order"]), MAIN_METHODS)
        self.assertEqual(len(space["methods"]), 10)
        self.assertIn("dp_fedyogi", MAIN_METHODS)
        self.assertIn("dp_fedsofim_delta_proxy_adapted", MAIN_METHODS)
        for method in MAIN_METHODS:
            roster = space["methods"][method]
            self.assertEqual(roster["candidate_count"], 24)
            self.assertEqual(len(roster["candidates"]), 24)
            candidate0 = roster["candidates"][0]
            self.assertNotIn(candidate0["generation"]["kind"], {"source_sentinel", "lhs"})
            self.assertTrue(candidate0["generation"]["basis"])

    def test_lhs_is_deterministic_and_each_dimension_uses_every_stratum(self) -> None:
        first = build_candidate_space(STUDY_ID)
        second = build_candidate_space(STUDY_ID)
        changed = build_candidate_space(STUDY_ID, design_seed="different-seed")
        self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
        self.assertNotEqual(first["manifest_sha256"], changed["manifest_sha256"])
        for method in MAIN_METHODS:
            candidates = [
                candidate
                for candidate in first["methods"][method]["candidates"]
                if candidate["generation"]["kind"] == "lhs"
            ]
            self.assertTrue(candidates)
            keys = candidates[0]["generation"]["strata"]
            for key in keys:
                observed = {candidate["generation"]["strata"][key] for candidate in candidates}
                self.assertEqual(observed, set(range(len(candidates))))

    def test_adam_yogi_time_control_and_fedsift_coordinates_are_paired(self) -> None:
        space = build_candidate_space(STUDY_ID)
        adam_family = ("dp_fedadam", "time_dpfedadam", "public_argmin_time_dpfedadam", "fedsift")
        numeric_backend_keys = ("server_learning_rate", "beta1", "beta2", "tau")
        for index in range(1, 9):
            cid = f"candidate_{index:04d}"
            adam_candidate = candidate_by_id(space, "dp_fedadam", cid)
            adam = adam_candidate["parameters"]
            for method in adam_family[1:]:
                paired = candidate_by_id(space, method, cid)
                self.assertEqual(adam_candidate["generation"], paired["generation"])
                self.assertEqual(adam["local_optimizer"], paired["parameters"]["local_optimizer"])
                for key in numeric_backend_keys:
                    self.assertEqual(adam["backend"][key], paired["parameters"]["backend"][key])
            yogi_candidate = candidate_by_id(space, "dp_fedyogi", cid)
            yogi = yogi_candidate["parameters"]
            self.assertEqual(adam["local_optimizer"], yogi["local_optimizer"])
            for key in numeric_backend_keys:
                if index == 8 and key == "beta2":
                    self.assertEqual(adam["backend"][key], 0.8)
                    self.assertEqual(yogi["backend"][key], 0.5)
                else:
                    self.assertEqual(adam["backend"][key], yogi["backend"][key])
            if index == 8:
                self.assertEqual(
                    yogi_candidate["generation"]["pairing_scope"],
                    "dp_fedyogi_method_specific_candidate_0008",
                )
            timed = candidate_by_id(space, "time_dpfedadam", cid)["parameters"]
            public = candidate_by_id(space, "public_argmin_time_dpfedadam", cid)["parameters"]
            fedsift = candidate_by_id(space, "fedsift", cid)["parameters"]
            self.assertEqual(timed["privacy_schedule"], public["privacy_schedule"])
            self.assertEqual(public["privacy_schedule"], fedsift["privacy_schedule"])
            self.assertEqual(
                public["control_rule"]["query_every_rounds"],
                fedsift["control_rule"]["query_every_rounds"],
            )
            self.assertEqual(
                public["control_rule"]["step_candidates"],
                fedsift["control_rule"]["step_candidates"],
            )
            self.assertEqual(yogi["backend"]["name"], "fedyogi_paper_aligned")
        for index in range(9, DEFAULT_CANDIDATE_COUNT):
            cid = f"candidate_{index:04d}"
            adam_candidate = candidate_by_id(space, "dp_fedadam", cid)
            self.assertEqual(adam_candidate["generation"]["kind"], "lhs")
            for method in (*adam_family[1:], "dp_fedyogi"):
                paired = candidate_by_id(space, method, cid)
                self.assertEqual(paired["generation"]["kind"], "lhs")
                for key, coordinate in adam_candidate["generation"]["coordinates"].items():
                    self.assertEqual(paired["generation"]["coordinates"][key], coordinate)
                self.assertEqual(
                    paired["parameters"]["local_optimizer"],
                    adam_candidate["parameters"]["local_optimizer"],
                )
                for key in numeric_backend_keys:
                    self.assertEqual(
                        paired["parameters"]["backend"][key],
                        adam_candidate["parameters"]["backend"][key],
                    )

    def test_fedsift_joint_candidate_contains_backend_schedule_and_sift(self) -> None:
        space = build_candidate_space(STUDY_ID)
        for candidate in space["methods"]["fedsift"]["candidates"]:
            parameters = candidate["parameters"]
            self.assertIn("backend", parameters)
            self.assertIn("privacy_schedule", parameters)
            self.assertIn("sift", parameters)

    def test_sofim_adaptation_is_explicit_and_gets_the_shared_local_lr(self) -> None:
        space = build_candidate_space(STUDY_ID)
        for index in range(3):
            cid = f"candidate_{index:04d}"
            sofim = candidate_by_id(space, "dp_fedsofim_delta_proxy_adapted", cid)["parameters"]
            fedavg = candidate_by_id(space, "dp_fedavg", cid)["parameters"]
            self.assertEqual(sofim["local_optimizer"], fedavg["local_optimizer"])
        for index in range(DEFAULT_CANDIDATE_COUNT):
            cid = f"candidate_{index:04d}"
            sofim = candidate_by_id(space, "dp_fedsofim_delta_proxy_adapted", cid)["parameters"]
            backend = sofim["backend"]
            self.assertEqual(
                backend["preconditioned_vector"], "current_normalized_trajectory_proxy"
            )
            self.assertFalse(backend["bias_correction"])
            self.assertEqual(backend["warmup_rounds"], 0)
            self.assertIn("adapted_proxy", backend["claim_boundary"])

    def test_performance_fields_hash_drift_and_unknown_overrides_fail_closed(self) -> None:
        space = build_candidate_space(STUDY_ID)
        polluted = copy.deepcopy(space)
        polluted["average_precision"] = 0.9
        with self.assertRaises(CandidateSpaceError):
            validate_candidate_space(polluted)
        drifted = copy.deepcopy(space)
        drifted["methods"]["dp_fedavg"]["candidates"][0]["parameters"]["local_optimizer"][
            "learning_rate"
        ] = 0.5
        with self.assertRaises(CandidateSpaceError):
            validate_candidate_space(drifted)
        frozen = frozen_space()
        override = copy.deepcopy(frozen)
        override["runner_override"] = {"clip_norm": 999.0}
        override.pop("manifest_sha256")
        override["manifest_sha256"] = canonical_sha256(override)
        with self.assertRaises(CandidateSpaceError):
            validate_candidate_space(override, require_frozen=True)


class HpoPlanTests(unittest.TestCase):

    def test_without_sift_has_one_frozen_no_query_full_step_semantics(self) -> None:
        switch = MATCHED_ABLATIONS["fedsift_without_sift"]["mechanism_switch"]
        self.assertEqual(switch["replacement_value"], "no_public_control_fixed_full_step")
        self.assertFalse(switch["sift_enabled"])
        self.assertFalse(switch["public_control_queries_enabled"])
        self.assertEqual(switch["fixed_alpha"], 1.0)
        self.assertEqual(switch["non_query_round_behavior"], "fixed_full_step")

    def test_hpo_rejects_unfrozen_candidate_or_nested_plan(self) -> None:
        manifest, draft_nested, frozen_nested = frozen_nested_fixture()
        with self.assertRaises(HpoPlanError):
            build_hpo_plan(
                build_candidate_space(STUDY_ID, candidate_count=SMALL_CANDIDATE_COUNT),
                frozen_nested,
                manifest,
                preprocessing_profile_spec=hpo_profile_fixture(),
                hpo_seeds=(11,),
                max_steps=30,
            )
        with self.assertRaises(HpoPlanError):
            build_hpo_plan(
                frozen_space(),
                draft_nested,
                manifest,
                preprocessing_profile_spec=hpo_profile_fixture(),
                hpo_seeds=(11,),
                max_steps=30,
            )

    def test_plan_derives_exact_nested_dimensions_and_cartesian_inventory(self) -> None:
        space, nested, manifest, plan = small_plan()
        expected_per_method = 3 * 5 * 3 * SMALL_CANDIDATE_COUNT * 1
        self.assertEqual(plan["method_order"], list(MAIN_METHODS))
        self.assertEqual(plan["expected_unit_count"], 10 * expected_per_method)
        self.assertEqual(plan["bindings"]["nested_plan_sha256"], nested["nested_plan_sha256"])
        self.assertEqual(
            plan["bindings"]["group_manifest_sha256"], manifest["group_manifest_sha256"]
        )
        for contract in plan["method_contracts"].values():
            self.assertEqual(contract["expected_units"], expected_per_method)
            self.assertEqual(contract["expected_units_per_outer_scope"], 3 * SMALL_CANDIDATE_COUNT)
            self.assertEqual(contract["max_steps"], 30)
            self.assertEqual(contract["hpo_seeds"], [11])
        self.assertEqual(plan["selection_contract"]["scope"], "within_outer_repeat_outer_fold_only")
        self.assertEqual(
            plan["selection_contract"]["lexicographic_objectives"],
            [
                {"criterion": "log_loss", "direction": "minimize"},
                {"criterion": "average_precision", "direction": "maximize"},
                {"criterion": "auroc", "direction": "maximize"},
                {"criterion": "brier_score", "direction": "minimize"},
                {"criterion": "communication_bytes", "direction": "minimize"},
                {"criterion": "candidate_id", "direction": "lexicographic_min"},
            ],
        )
        self.assertEqual(
            plan["selection_contract"]["development_context"],
            "log_loss_first_for_all_methods_after_disclosed_v4_pilot_not_independent_confirmation",
        )
        validate_hpo_plan(plan, space, nested, manifest)

    def test_every_unit_binds_exact_candidate_nested_and_membership_hashes(self) -> None:
        space, nested, manifest, plan = small_plan()
        unit = plan["units"][0]
        self.assertEqual(unit["candidate_space_sha256"], space["manifest_sha256"])
        self.assertEqual(unit["nested_plan_sha256"], nested["nested_plan_sha256"])
        expected = nested["repetitions"][0]["outer_folds"][0]["inner_folds"][0]
        self.assertEqual(
            unit["split_bindings"]["inner_validation_membership_sha256"],
            expected["inner_validation"]["membership_sha256"],
        )
        self.assertEqual(
            unit["split_bindings"]["v_ctrl_membership_sha256"],
            expected["roles"]["v_ctrl"]["membership_sha256"],
        )
        self.assertNotIn("outer_test", unit)
        validate_hpo_plan(plan, space, nested, manifest)

    def test_preprocessing_profile_is_symmetric_and_changes_plan_and_capability(self) -> None:
        space, nested, manifest, primary_plan = small_plan()
        primary = preprocessing_profile(PIMA_PRIMARY_PROFILE)
        sensitivity = preprocessing_profile(PIMA_SENSITIVITY_PROFILE)
        for unit in primary_plan["units"]:
            self.assertEqual(unit["preprocessing_profile_name"], primary["name"])
            self.assertEqual(unit["preprocessing_profile_sha256"], primary["profile_sha256"])
            self.assertEqual(unit["preprocessing_profile"], primary)
        sensitivity_plan = build_hpo_plan(
            space,
            nested,
            manifest,
            preprocessing_profile_spec=sensitivity,
            hpo_seeds=(11,),
            max_steps=30,
        )
        self.assertNotEqual(primary_plan["hpo_plan_sha256"], sensitivity_plan["hpo_plan_sha256"])
        primary_capability = _build_capability_payload(
            primary_plan, space, nested, manifest, primary_plan["units"][0]["unit_id"]
        )
        sensitivity_capability = _build_capability_payload(
            sensitivity_plan, space, nested, manifest, sensitivity_plan["units"][0]["unit_id"]
        )
        self.assertNotEqual(
            primary_capability["capability_sha256"], sensitivity_capability["capability_sha256"]
        )
        method_drift = copy.deepcopy(primary_plan)
        for unit in method_drift["units"]:
            if unit["method"] == "dp_fedavg":
                unit["preprocessing_profile_name"] = sensitivity["name"]
                unit["preprocessing_profile_sha256"] = sensitivity["profile_sha256"]
                unit["preprocessing_profile"] = copy.deepcopy(sensitivity)
                identity = dict(unit)
                identity.pop("unit_id")
                digest = hashlib.sha256(
                    (_identity("hpo_unit_v3") + canonical_sha256(identity)).encode("ascii")
                ).hexdigest()
                unit["unit_id"] = f"hpo_{digest[:24]}"
        method_drift.pop("hpo_plan_sha256")
        method_drift["hpo_plan_sha256"] = canonical_sha256(method_drift)
        with self.assertRaises(HpoPlanError):
            validate_hpo_plan(method_drift, space, nested, manifest)

    def test_unknown_overrides_contract_drift_and_unit_drift_fail_closed(self) -> None:
        space, nested, manifest, plan = small_plan()
        override = copy.deepcopy(plan)
        override["runner_override"] = {"selection_scope": "global_cross_outer"}
        override.pop("hpo_plan_sha256")
        override["hpo_plan_sha256"] = canonical_sha256(override)
        with self.assertRaises(HpoPlanError):
            validate_hpo_plan(override, space, nested, manifest)
        contracts = copy.deepcopy(plan)
        for contract in contracts["method_contracts"].values():
            contract["minimum_fully_closed_candidates_per_outer_scope"] = 0
            contract["max_failed_candidates_per_outer_scope"] = 4
        contracts.pop("hpo_plan_sha256")
        contracts["hpo_plan_sha256"] = canonical_sha256(contracts)
        with self.assertRaises(HpoPlanError):
            validate_hpo_plan(contracts, space, nested, manifest)
        unit_drift = copy.deepcopy(plan)
        unit_drift["units"][0]["candidate_sha256"] = "f" * 64
        identity = dict(unit_drift["units"][0])
        identity.pop("unit_id")
        digest = hashlib.sha256(
            (_identity("hpo_unit_v3") + canonical_sha256(identity)).encode("ascii")
        ).hexdigest()
        unit_drift["units"][0]["unit_id"] = f"hpo_{digest[:24]}"
        unit_drift.pop("hpo_plan_sha256")
        unit_drift["hpo_plan_sha256"] = canonical_sha256(unit_drift)
        with self.assertRaises(HpoPlanError):
            validate_hpo_plan(unit_drift, space, nested, manifest)

    def test_open_polluted_or_receiptless_ledger_cannot_close(self) -> None:
        space, nested, manifest, plan = small_plan()
        ledger, receipts = successful_evidence(plan, space, nested, manifest)
        with self.assertRaises(HpoPlanError):
            close(plan, ledger[:-1], receipts, space, nested, manifest)
        polluted = copy.deepcopy(ledger)
        polluted[0]["average_precision"] = 0.8
        with self.assertRaises(HpoPlanError):
            close(plan, polluted, receipts, space, nested, manifest)
        receiptless = copy.deepcopy(ledger)
        del receiptless[0]["attempts"][0]["attempt_receipt_sha256"]
        with self.assertRaises(HpoPlanError):
            close(plan, receiptless, receipts, space, nested, manifest)

    def test_forged_closure_without_original_ledger_is_rejected(self) -> None:
        space, nested, manifest, plan = small_plan()
        ledger, receipts = successful_evidence(plan, space, nested, manifest)
        closure = close(plan, ledger, receipts, space, nested, manifest)
        forged = copy.deepcopy(closure)
        forged["ledger_sha256"] = "e" * 64
        forged.pop("closure_sha256")
        forged["closure_sha256"] = canonical_sha256(forged)
        with self.assertRaises(HpoPlanError):
            validate_selection_authorization(
                plan,
                forged,
                ledger,
                attempt_receipts=receipts,
                candidate_space=space,
                nested_plan=nested,
                group_manifest=manifest,
            )
        with self.assertRaises(HpoPlanError):
            validate_selection_authorization(
                plan,
                closure,
                [],
                attempt_receipts=receipts,
                candidate_space=space,
                nested_plan=nested,
                group_manifest=manifest,
            )

    def test_candidate_failure_is_isolated_to_one_outer_scope(self) -> None:
        space, nested, manifest, plan = small_plan()
        ledger, receipts = successful_evidence(plan, space, nested, manifest)
        target = next(
            (
                unit
                for unit in plan["units"]
                if unit["method"] == "dp_fedavg"
                and unit["outer_repeat"] == 0
                and (unit["outer_fold"] == 0)
                and (unit["candidate_id"] == "candidate_0003")
            )
        )
        entry = next((row for row in ledger if row["unit_id"] == target["unit_id"]))
        entry["attempts"] = [
            {"attempt_index": 0, "outcome": "failed", "failure_code": "numerical_instability"}
        ]
        refresh_entry_receipts(receipts, plan, space, nested, manifest, entry)
        closure = close(plan, ledger, receipts, space, nested, manifest)
        scope00 = next(
            (
                row
                for row in closure["outer_authorizations"]
                if row["outer_repeat"] == 0 and row["outer_fold"] == 0
            )
        )
        scope01 = next(
            (
                row
                for row in closure["outer_authorizations"]
                if row["outer_repeat"] == 0 and row["outer_fold"] == 1
            )
        )
        self.assertEqual(
            scope00["methods"]["dp_fedavg"]["failed_candidate_ids"], ["candidate_0003"]
        )
        self.assertIn("candidate_0003", scope01["methods"]["dp_fedavg"]["eligible_candidate_ids"])

    def test_scientific_failure_cannot_be_retried(self) -> None:
        space, nested, manifest, plan = small_plan()
        ledger, receipts = successful_evidence(plan, space, nested, manifest)
        unit_id = ledger[0]["unit_id"]
        ledger[0]["attempts"] = [
            {"attempt_index": 0, "outcome": "failed", "failure_code": "nonfinite_update"},
            {"attempt_index": 1, "outcome": "complete"},
        ]
        refresh_entry_receipts(receipts, plan, space, nested, manifest, ledger[0])
        with self.assertRaises(HpoPlanError):
            close(plan, ledger, receipts, space, nested, manifest)

    def test_infrastructure_retry_may_end_in_scientific_failure(self) -> None:
        space, nested, manifest, plan = small_plan()
        ledger, receipts = successful_evidence(plan, space, nested, manifest)
        target = next(
            (
                unit
                for unit in plan["units"]
                if unit["method"] == "fedsift"
                and unit["outer_repeat"] == 0
                and (unit["outer_fold"] == 0)
                and (unit["candidate_id"] == "candidate_0003")
            )
        )
        entry = next((row for row in ledger if row["unit_id"] == target["unit_id"]))
        entry["attempts"] = [
            {"attempt_index": 0, "outcome": "failed", "failure_code": "infrastructure_failure"},
            {"attempt_index": 1, "outcome": "failed", "failure_code": "privacy_accounting_failure"},
        ]
        refresh_entry_receipts(receipts, plan, space, nested, manifest, entry)
        closure = close(plan, ledger, receipts, space, nested, manifest)
        scope = next(
            (
                row
                for row in closure["outer_authorizations"]
                if row["outer_repeat"] == 0 and row["outer_fold"] == 0
            )
        )
        self.assertEqual(scope["methods"]["fedsift"]["failed_candidate_ids"], ["candidate_0003"])

    def test_matched_ablation_is_bound_to_one_outer_selected_parent(self) -> None:
        space, nested, manifest, plan = small_plan()
        ledger, receipts = successful_evidence(plan, space, nested, manifest)
        closure = close(plan, ledger, receipts, space, nested, manifest)
        parent = candidate_by_id(space, "fedsift", "candidate_0000")
        inheritance = inherit_matched_ablation(
            plan,
            closure,
            ledger,
            attempt_receipts=receipts,
            candidate_space=space,
            nested_plan=nested,
            group_manifest=manifest,
            ablation_id="fedsift_uniform_schedule",
            outer_repeat=0,
            outer_fold=0,
            parent_candidate_id=parent["candidate_id"],
            parent_candidate_sha256=parent["candidate_sha256"],
            parent_selection_receipt_sha256="d" * 64,
        )
        self.assertEqual(inheritance["outer_repeat"], 0)
        self.assertEqual(inheritance["outer_fold"], 0)
        self.assertEqual(
            inheritance["configuration_source"], "inherit_outer_selected_parent_candidate_exactly"
        )
        self.assertFalse(inheritance["independent_candidate_roster"])
        self.assertEqual(inheritance["additional_hpo_units"], 0)
        public_rule = inherit_matched_ablation(
            plan,
            closure,
            ledger,
            attempt_receipts=receipts,
            candidate_space=space,
            nested_plan=nested,
            group_manifest=manifest,
            ablation_id="fedsift_public_argmin_rule",
            outer_repeat=0,
            outer_fold=0,
            parent_candidate_id=parent["candidate_id"],
            parent_candidate_sha256=parent["candidate_sha256"],
            parent_selection_receipt_sha256="d" * 64,
        )
        self.assertEqual(public_rule["method_identity"], "fedsift_public_argmin_rule")
        self.assertEqual(public_rule["parent_method"], "fedsift")
        self.assertEqual(
            public_rule["mechanism_switch"],
            {
                "changed_component": "control_rule.selection_rule",
                "parent_value": "fedsift_supported_override",
                "replacement_value": "public_argmin_mean_logloss",
                "replacement_contract": {
                    "tie_break": "larger_step",
                    "safety_margin": "none",
                    "uses_sift_safety_terms": False,
                },
            },
        )
        self.assertIn("candidate_grid", public_rule["inherited_parent_fields"])
        self.assertIn("parameters.backend", public_rule["inherited_parent_fields"])
        self.assertIn("parameters.privacy_schedule", public_rule["inherited_parent_fields"])
        self.assertIn(
            "parameters.control_rule.query_every_rounds", public_rule["inherited_parent_fields"]
        )
        self.assertIn(
            "parameters.control_rule.step_candidates", public_rule["inherited_parent_fields"]
        )
        self.assertFalse(public_rule["independent_candidate_roster"])
        self.assertEqual(public_rule["additional_hpo_units"], 0)
        self.assertIn("public_argmin_time_dpfedadam", plan["method_order"])


if __name__ == "__main__":
    unittest.main()
