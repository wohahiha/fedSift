from __future__ import annotations
import copy
import unittest
from fedsift import baseline_math
from fedsift.candidate_space import MAIN_METHODS, _base_parameters
from fedsift.hpo_plan import MATCHED_ABLATIONS
from fedsift.method_dispatch import (
    ALL_METHODS,
    MethodDispatchError,
    build_method_execution_spec,
    resolve_server_kernel,
    validate_method_execution_spec,
)
from fedsift.sofim_math import sofim_server_step_with_receipt


class MethodDispatchTests(unittest.TestCase):

    def _main(self, method: str):
        return build_method_execution_spec(
            method_id=method, candidate_parameters=_base_parameters(method)
        )

    def _ablation(self, method: str):
        return build_method_execution_spec(
            method_id=method,
            candidate_parameters=_base_parameters("fedsift"),
            mechanism_switch=MATCHED_ABLATIONS[method]["mechanism_switch"],
        )

    def test_complete_roster_has_an_explicit_nonfallback_dispatch(self) -> None:
        self.assertEqual(ALL_METHODS, (*MAIN_METHODS, *tuple(MATCHED_ABLATIONS)))
        specs = {method: self._main(method) for method in MAIN_METHODS}
        specs.update({method: self._ablation(method) for method in MATCHED_ABLATIONS})
        self.assertEqual(set(specs), set(ALL_METHODS))
        self.assertEqual(len({spec.dispatch_sha256 for spec in specs.values()}), 13)
        for method, spec in specs.items():
            self.assertEqual(spec.method_id, method)
            self.assertEqual(spec.resource_accounting_method_id, method)
            self.assertNotEqual(spec.server_backend_handler, "fallback")

    def test_server_kernel_mapping_is_exact_for_every_backend_family(self) -> None:
        expected = {
            "fedavg_nonprivate": baseline_math.fedavg_server_average,
            "dp_fedavg": baseline_math.fedavg_server_average,
            "dp_fedprox_adapted": baseline_math.fedavg_server_average,
            "dp_scaffold_adapted": baseline_math.scaffold_server_batched_weighted,
            "dp_fedadam": baseline_math.fedadam_server_step,
            "dp_fedyogi": baseline_math.fedyogi_server_step,
            "dp_fedsofim_delta_proxy_adapted": sofim_server_step_with_receipt,
            "time_dpfedadam": baseline_math.fedadam_server_step,
            "public_argmin_time_dpfedadam": baseline_math.fedadam_server_step,
            "fedsift": baseline_math.fedadam_server_step,
        }
        for method, kernel in expected.items():
            with self.subTest(method=method):
                self.assertIs(resolve_server_kernel(self._main(method)), kernel)

    def test_local_correction_and_schedule_paths_are_not_collapsed(self) -> None:
        nonprivate = self._main("fedavg_nonprivate")
        fedprox = self._main("dp_fedprox_adapted")
        scaffold = self._main("dp_scaffold_adapted")
        sofim = self._main("dp_fedsofim_delta_proxy_adapted")
        timed = self._main("time_dpfedadam")
        self.assertEqual(
            nonprivate.local_update_handler, "canonical_nonprivate_explicit_minibatches"
        )
        self.assertEqual(
            fedprox.correction_handler, "fedprox_post_privacy_data_independent_correction"
        )
        self.assertEqual(
            scaffold.correction_handler, "scaffold_post_privacy_prior_state_correction"
        )
        self.assertEqual(
            _base_parameters("dp_scaffold_adapted")["backend"]["name"],
            "scaffold_option2_batched_weighted_adapted",
        )
        self.assertEqual(
            scaffold.scaffold_variant, "option_ii_batched_weighted_registered_client_weights"
        )
        self.assertIn("actual_steps", sofim.sofim_proxy_contract)
        self.assertEqual(
            timed.privacy_schedule_handler, "two_phase_time_candidate_specific_calibration"
        )

    def test_three_ablations_have_frozen_one_component_semantics(self) -> None:
        uniform = self._ablation("fedsift_uniform_schedule")
        no_sift = self._ablation("fedsift_without_sift")
        argmin = self._ablation("fedsift_public_argmin_rule")
        self.assertEqual(uniform.privacy_schedule_handler, "uniform_candidate_specific_calibration")
        self.assertEqual(no_sift.public_control_handler, "none")
        self.assertFalse(no_sift.public_control_queries_enabled)
        self.assertEqual(no_sift.non_query_round_handler, "fixed_full_step_alpha_1")
        self.assertEqual(argmin.public_control_handler, "decide_public_argmin")
        self.assertTrue(argmin.public_control_queries_enabled)
        self.assertEqual(
            argmin.server_backend_handler, self._main("fedsift").server_backend_handler
        )

    def test_unknown_or_incompatible_names_parameters_and_switches_fail_closed(self) -> None:
        with self.assertRaises(MethodDispatchError):
            build_method_execution_spec(
                method_id="fedsift_better_because_hidden_fallback",
                candidate_parameters=_base_parameters("fedsift"),
            )
        tampered = _base_parameters("dp_fedyogi")
        tampered["backend"]["name"] = "weighted_average"
        with self.assertRaises(MethodDispatchError):
            build_method_execution_spec(method_id="dp_fedyogi", candidate_parameters=tampered)
        with self.assertRaises(MethodDispatchError):
            build_method_execution_spec(
                method_id="fedsift_without_sift",
                candidate_parameters=_base_parameters("fedsift"),
                mechanism_switch={"sift_enabled": False},
            )
        with self.assertRaises(MethodDispatchError):
            build_method_execution_spec(
                method_id="fedsift",
                candidate_parameters=_base_parameters("fedsift"),
                mechanism_switch=MATCHED_ABLATIONS["fedsift_without_sift"]["mechanism_switch"],
            )
        missing_fedopt_parameter = _base_parameters("dp_fedadam")
        del missing_fedopt_parameter["backend"]["tau"]
        with self.assertRaises(MethodDispatchError):
            build_method_execution_spec(
                method_id="dp_fedadam", candidate_parameters=missing_fedopt_parameter
            )

    def test_zero_safety_margin_registered_candidate_is_valid(self) -> None:
        parameters = _base_parameters("fedsift")
        parameters["sift"]["safety_margin_z"] = 0.0
        spec = build_method_execution_spec(method_id="fedsift", candidate_parameters=parameters)
        self.assertEqual(spec.public_control_handler, "decide_fedsift")

    def test_fedopt_zero_decay_sentinels_follow_the_paper_domain(self) -> None:
        for method in ("dp_fedadam", "dp_fedyogi"):
            with self.subTest(method=method, beta="beta1"):
                parameters = _base_parameters(method)
                parameters["backend"]["beta1"] = 0.0
                spec = build_method_execution_spec(
                    method_id=method, candidate_parameters=parameters
                )
                self.assertEqual(spec.method_id, method)
            with self.subTest(method=method, beta="beta2"):
                parameters = _base_parameters(method)
                parameters["backend"]["beta2"] = 0.0
                spec = build_method_execution_spec(
                    method_id=method, candidate_parameters=parameters
                )
                self.assertEqual(spec.method_id, method)
        for invalid in (-0.01, 1.0):
            parameters = _base_parameters("dp_fedadam")
            parameters["backend"]["beta1"] = invalid
            with self.assertRaises(MethodDispatchError):
                build_method_execution_spec(method_id="dp_fedadam", candidate_parameters=parameters)

    def test_spec_validation_rebuilds_external_commitment(self) -> None:
        parameters = _base_parameters("fedsift")
        switch = MATCHED_ABLATIONS["fedsift_public_argmin_rule"]["mechanism_switch"]
        spec = build_method_execution_spec(
            method_id="fedsift_public_argmin_rule",
            candidate_parameters=parameters,
            mechanism_switch=switch,
        )
        validate_method_execution_spec(
            spec,
            method_id="fedsift_public_argmin_rule",
            candidate_parameters=parameters,
            mechanism_switch=switch,
            expected_dispatch_sha256=spec.dispatch_sha256,
        )
        modified = copy.deepcopy(parameters)
        modified["local_optimizer"]["learning_rate"] = 0.07
        with self.assertRaises(MethodDispatchError):
            validate_method_execution_spec(
                spec,
                method_id="fedsift_public_argmin_rule",
                candidate_parameters=modified,
                mechanism_switch=switch,
                expected_dispatch_sha256=spec.dispatch_sha256,
            )


if __name__ == "__main__":
    unittest.main()
