from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import hashlib
import inspect
import unittest
from collections import OrderedDict
from dataclasses import replace
from unittest.mock import patch
import torch
import fedsift.local_training as local_training
from fedsift.local_training import (
    CorrectionPolicy,
    LocalTrainingContext,
    LocalTrainingGateError,
    LocalTrainingGatePoisoned,
    PoissonLocalTrainingGate,
    PoissonStepResult,
    build_correction_source_binding,
    build_fedprox_correction,
    build_scaffold_correction,
    canonical_nonprivate_explicit_minibatches,
    poisson_step_receipt_fingerprint,
)
from fedsift.modeling import (
    InitializationDomain,
    LogisticScreening,
    SelectedBCEGradientBridge,
    extract_model_state,
    model_state_sha256,
)
from fedsift.privacy_accounting import PoissonDPStage, PrivacyAccountingError

CANDIDATE_SHA256 = hashlib.sha256(b"candidate-fixture").hexdigest()
ANCHOR_EVIDENCE_SHA256 = hashlib.sha256(b"round-global-anchor").hexdigest()
CONTROL_EVIDENCE_SHA256 = hashlib.sha256(b"fixed-control-variates").hexdigest()
INITIAL_STATE_SOURCE_SHA256 = hashlib.sha256(b"fixed-model-initialization").hexdigest()


def _context(
    *, client_id: str = "client_0", method_id: str = "dp_fedavg", federated_round: int = 4
) -> LocalTrainingContext:
    return LocalTrainingContext(
        study_id=_identity("local_training_test"),
        dataset_id="synthetic_binary",
        outer_repeat=0,
        outer_fold=1,
        inner_fold=2,
        seed_repeat=3,
        client_id=client_id,
        federated_round=federated_round,
        method_id=method_id,
        candidate_sha256=CANDIDATE_SHA256,
    )


def _bridge_and_state(*, client_id: str = "client_0"):
    model = LogisticScreening(
        2,
        initialization_seed=53,
        initialization_domain=InitializationDomain(
            study_id=_identity("local_training_test"),
            outer_repeat=0,
            outer_fold=1,
            client_id=client_id,
            model_role="private_local_screening",
        ),
    )
    features = torch.tensor([[0.2, 0.5], [1.0, -0.4], [-0.3, 0.8], [0.7, 0.1]], dtype=torch.float64)
    labels = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float64)
    row_ids = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
    bridge = SelectedBCEGradientBridge(model, features, labels, row_ids)
    return (bridge, extract_model_state(model))


def _policy(kind: str, bridge, state, context) -> CorrectionPolicy:
    if kind == "none":
        return CorrectionPolicy("none", CANDIDATE_SHA256, None, ())
    if kind == "fedprox":
        anchor = OrderedDict(((name, torch.zeros_like(value)) for (name, value) in state.items()))
        source = build_correction_source_binding(
            bridge,
            anchor,
            secondary_state=None,
            source_kind="round_global_anchor",
            source_evidence_sha256=ANCHOR_EVIDENCE_SHA256,
            source_context=_context(
                client_id="server",
                method_id=context.method_id,
                federated_round=context.federated_round,
            ),
        )
        return CorrectionPolicy("fedprox", CANDIDATE_SHA256, 0.25, (source,))
    if kind == "scaffold":
        server = OrderedDict(((name, torch.ones_like(value)) for (name, value) in state.items()))
        client = OrderedDict(((name, torch.zeros_like(value)) for (name, value) in state.items()))
        source = build_correction_source_binding(
            bridge,
            server,
            secondary_state=client,
            source_kind="fixed_scaffold_controls",
            source_evidence_sha256=CONTROL_EVIDENCE_SHA256,
            source_context=context,
        )
        return CorrectionPolicy("scaffold", CANDIDATE_SHA256, None, (source,))
    raise AssertionError(kind)


def _fixture(
    *,
    q: float = 0.5,
    sigma: float = 0.5,
    steps: int = 1,
    sampling_seed: int = 101,
    noise_seed: int = 202,
    client_id: str = "client_0",
    correction_kind: str = "none",
    method_id: str | None = None,
    federated_round: int = 4,
    initial_model_state_sha256: str | None = None,
    initial_model_state_source_sha256: str = INITIAL_STATE_SOURCE_SHA256,
):
    bridge, state = _bridge_and_state(client_id=client_id)
    method = (
        method_id
        or {"none": "dp_fedavg", "fedprox": "dp_fedprox", "scaffold": "dp_scaffold"}[
            correction_kind
        ]
    )
    context = _context(client_id=client_id, method_id=method, federated_round=federated_round)
    gate = PoissonLocalTrainingGate(
        (PoissonDPStage("local_train", q, sigma, steps),),
        gradient_bridge=bridge,
        population_row_ids=bridge.canonical_population_row_ids,
        clip_norm=1.0,
        learning_rate=0.1,
        delta=1e-05,
        sampling_seed=sampling_seed,
        noise_seed=noise_seed,
        context=context,
        correction_policy=_policy(correction_kind, bridge, state, context),
        initial_model_state_sha256=(
            model_state_sha256(bridge.model, state)
            if initial_model_state_sha256 is None
            else initial_model_state_sha256
        ),
        initial_model_state_source_sha256=initial_model_state_source_sha256,
        eps_error=0.01,
        delta_error=1e-08,
    )
    return (gate, bridge, state)


def _mask(values: tuple[bool, ...]):

    def draw(**kwargs):
        return torch.tensor(values, dtype=torch.bool, device=kwargs["device"])

    return draw


def _zero_randn(shape, *, dtype, device, generator):
    del generator
    return torch.zeros(shape, dtype=dtype, device=device)


class InternalPoissonIdentityTests(unittest.TestCase):

    def test_gate_owns_mask_q_sigma_gradients_and_generators(self) -> None:
        gate, bridge, state = _fixture()
        signature = inspect.signature(gate.execute_poisson_optimizer_step)
        for forbidden in (
            "poisson_mask",
            "selected_per_record_gradients",
            "executed_sample_rate",
            "executed_noise_multiplier",
            "generator",
        ):
            self.assertNotIn(forbidden, signature.parameters)
        batch = bridge.selected_gradient_batch(state, (10, 30))
        with self.assertRaises(TypeError):
            gate.execute_poisson_optimizer_step(
                state,
                batch.gradients,
                torch.tensor([True, False, True, False]),
                optimizer_step_index=1,
            )

    def test_internal_mask_drives_exact_selected_ids_and_receipt(self) -> None:
        gate, bridge, state = _fixture()
        observed: list[tuple[int, ...]] = []
        original = bridge.selected_gradient_batch

        def capture(current_state, selected_row_ids):
            observed.append(tuple(selected_row_ids))
            return original(current_state, selected_row_ids)

        with (
            patch.object(
                local_training,
                "_draw_internal_poisson_mask",
                side_effect=_mask((True, False, True, False)),
            ),
            patch.object(bridge, "selected_gradient_batch", side_effect=capture),
            patch.object(local_training.torch, "randn", side_effect=_zero_randn),
        ):
            result = gate.execute_poisson_optimizer_step(state, optimizer_step_index=1)
        self.assertEqual(observed, [(10, 30)])
        self.assertEqual(result.receipt.schema, _identity("poisson_optimizer_step_receipt"))
        self.assertEqual(result.receipt.selected_row_ids, (10, 30))
        self.assertEqual(result.receipt.sampled_record_count, 2)
        self.assertNotEqual(result.receipt.sampling_rng_domain, result.receipt.noise_rng_domain)
        self.assertNotEqual(
            result.receipt.sampling_seed_commitment_sha256,
            result.receipt.noise_seed_commitment_sha256,
        )
        self.assertIn("not_csprng", result.receipt.rng_security_claim)
        self.assertEqual(result.receipt.learning_rate, 0.1)
        self.assertEqual(len(result.receipt.gaussian_noise_tensors_sha256), 64)
        self.assertTrue(result.receipt.receipt_confidentiality.endswith("do_not_publish"))
        self.assertFalse(result.receipt.performance_metrics_consumed)
        self.assertEqual(
            poisson_step_receipt_fingerprint(result.receipt), result.receipt.receipt_sha256
        )
        gate.validate_step_result(result, state)

    def test_same_count_different_internal_masks_have_distinct_identity(self) -> None:
        outputs = []
        for values in ((True, False, True, False), (False, True, False, True)):
            gate, _, state = _fixture()
            with (
                patch.object(
                    local_training, "_draw_internal_poisson_mask", side_effect=_mask(values)
                ),
                patch.object(local_training.torch, "randn", side_effect=_zero_randn),
            ):
                outputs.append(gate.execute_poisson_optimizer_step(state, optimizer_step_index=1))
        left, right = (value.receipt for value in outputs)
        self.assertEqual(left.sampled_record_count, right.sampled_record_count)
        self.assertNotEqual(left.selected_row_ids, right.selected_row_ids)
        self.assertNotEqual(left.selected_row_ids_sha256, right.selected_row_ids_sha256)
        self.assertNotEqual(left.poisson_mask_sha256, right.poisson_mask_sha256)
        self.assertNotEqual(left.gradient_lineage_sha256, right.gradient_lineage_sha256)

    def test_duplicate_or_wrong_selected_ids_from_bridge_fail_closed(self) -> None:
        for replacement in ((10, 10), (20, 40)):
            gate, bridge, state = _fixture()
            original = bridge.selected_gradient_batch

            def forged(current_state, selected_row_ids, replacement=replacement):
                batch = original(current_state, selected_row_ids)
                return replace(batch, selected_row_ids=replacement)

            with (
                self.subTest(replacement=replacement),
                patch.object(
                    local_training,
                    "_draw_internal_poisson_mask",
                    side_effect=_mask((True, False, True, False)),
                ),
                patch.object(bridge, "selected_gradient_batch", side_effect=forged),
                patch.object(local_training.torch, "randn") as gaussian,
            ):
                with self.assertRaises(LocalTrainingGatePoisoned):
                    gate.execute_poisson_optimizer_step(state, optimizer_step_index=1)
                gaussian.assert_not_called()
                self.assertTrue(gate.poisoned)

    def test_row_gradient_swap_is_detected_before_noise(self) -> None:
        gate, bridge, state = _fixture()
        original = bridge.selected_gradient_batch

        def swapped(current_state, selected_row_ids):
            batch = original(current_state, selected_row_ids)
            gradients = OrderedDict(
                ((name, value.flip(0)) for (name, value) in batch.gradients.items())
            )
            return replace(batch, gradients=gradients)

        with (
            patch.object(
                local_training,
                "_draw_internal_poisson_mask",
                side_effect=_mask((True, False, True, False)),
            ),
            patch.object(bridge, "selected_gradient_batch", side_effect=swapped),
            patch.object(local_training.torch, "randn") as gaussian,
        ):
            with self.assertRaises(LocalTrainingGatePoisoned):
                gate.execute_poisson_optimizer_step(state, optimizer_step_index=1)
        gaussian.assert_not_called()

    def test_population_mismatch_duplicate_and_noncanonical_fail_at_construction(self) -> None:
        bridge, _ = _bridge_and_state()
        invalid = ((10, 20, 30, 99), (10, 20, 20, 40), (20, 10, 30, 40))
        for population in invalid:
            with self.subTest(population=population), self.assertRaises(LocalTrainingGateError):
                PoissonLocalTrainingGate(
                    (PoissonDPStage("local_train", 0.5, 1.0, 1),),
                    gradient_bridge=bridge,
                    population_row_ids=population,
                    clip_norm=1.0,
                    learning_rate=0.1,
                    delta=1e-05,
                    sampling_seed=1,
                    noise_seed=2,
                    context=_context(),
                    correction_policy=_policy("none", bridge, _, _context()),
                    initial_model_state_sha256=model_state_sha256(bridge.model, _),
                    initial_model_state_source_sha256=INITIAL_STATE_SOURCE_SHA256,
                )


class RandomnessDomainTests(unittest.TestCase):

    def test_equal_sampling_and_noise_seed_is_rejected(self) -> None:
        bridge, _ = _bridge_and_state()
        with self.assertRaisesRegex(LocalTrainingGateError, "seeds must differ"):
            PoissonLocalTrainingGate(
                (PoissonDPStage("local_train", 0.5, 1.0, 1),),
                gradient_bridge=bridge,
                population_row_ids=bridge.canonical_population_row_ids,
                clip_norm=1.0,
                learning_rate=0.1,
                delta=1e-05,
                sampling_seed=7,
                noise_seed=7,
                context=_context(),
                correction_policy=_policy("none", bridge, _, _context()),
                initial_model_state_sha256=model_state_sha256(bridge.model, _),
                initial_model_state_source_sha256=INITIAL_STATE_SOURCE_SHA256,
            )

    def test_forced_rng_stream_collision_fails_before_sampling(self) -> None:
        gate, _, state = _fixture()
        generator = torch.Generator(device="cpu").manual_seed(3)
        collision = "a" * 64
        with (
            patch.object(gate, "_rng_stream", return_value=(generator, collision, "b" * 64, 4)),
            patch.object(local_training, "_draw_internal_poisson_mask") as sampling,
        ):
            with self.assertRaises(LocalTrainingGatePoisoned):
                gate.execute_poisson_optimizer_step(state, optimizer_step_index=1)
        sampling.assert_not_called()
        self.assertTrue(gate.poisoned)

    def test_same_context_is_reproducible_and_different_seed_changes_commitment(self) -> None:
        gate_a, _, state_a = _fixture(q=1.0)
        gate_b, _, state_b = _fixture(q=1.0)
        result_a = gate_a.execute_poisson_optimizer_step(state_a, optimizer_step_index=1)
        result_b = gate_b.execute_poisson_optimizer_step(state_b, optimizer_step_index=1)
        self.assertEqual(result_a.receipt, result_b.receipt)
        for name in result_a.updated_state:
            self.assertTrue(torch.equal(result_a.updated_state[name], result_b.updated_state[name]))
        gate_c, _, state_c = _fixture(q=1.0, sampling_seed=303)
        result_c = gate_c.execute_poisson_optimizer_step(state_c, optimizer_step_index=1)
        self.assertNotEqual(
            result_a.receipt.sampling_seed_commitment_sha256,
            result_c.receipt.sampling_seed_commitment_sha256,
        )

    def test_client_identity_domain_separates_streams(self) -> None:
        gate_a, _, state_a = _fixture(client_id="client_0", q=1.0)
        gate_b, _, state_b = _fixture(client_id="client_1", q=1.0)
        result_a = gate_a.execute_poisson_optimizer_step(state_a, optimizer_step_index=1)
        result_b = gate_b.execute_poisson_optimizer_step(state_b, optimizer_step_index=1)
        self.assertNotEqual(
            result_a.receipt.local_training_context_sha256,
            result_b.receipt.local_training_context_sha256,
        )
        self.assertNotEqual(
            result_a.receipt.noise_seed_commitment_sha256,
            result_b.receipt.noise_seed_commitment_sha256,
        )

    def test_seed_repeat_is_paired_index_not_common_random_numbers(self) -> None:
        gate_a, _, state_a = _fixture(q=1.0, method_id="dp_fedavg")
        gate_b, _, state_b = _fixture(q=1.0, method_id="fedsift")
        result_a = gate_a.execute_poisson_optimizer_step(state_a, optimizer_step_index=1)
        result_b = gate_b.execute_poisson_optimizer_step(state_b, optimizer_step_index=1)
        self.assertNotEqual(
            result_a.receipt.sampling_seed_commitment_sha256,
            result_b.receipt.sampling_seed_commitment_sha256,
        )
        self.assertNotEqual(
            result_a.receipt.noise_seed_commitment_sha256,
            result_b.receipt.noise_seed_commitment_sha256,
        )
        self.assertIn("not_common_random_numbers", result_a.receipt.randomness_pairing_claim)


class TypedCorrectionTests(unittest.TestCase):

    def test_fixed_method_roster_requires_exact_correction_semantics(self) -> None:
        no_correction_methods = (
            "dp_fedavg",
            "dp_fedadam",
            "dp_fedyogi",
            "dp_fedsofim_delta_proxy_adapted",
            "time_dpfedadam",
            "public_argmin_time_dpfedadam",
            "fedsift",
            "fedsift_uniform_schedule",
            "fedsift_without_sift",
            "fedsift_public_argmin_rule",
        )
        for method_id in no_correction_methods:
            with self.subTest(method_id=method_id):
                gate, _, _ = _fixture(correction_kind="none", method_id=method_id)
                self.assertEqual(gate.committed_steps, 0)
        for method_id, kind in (
            ("dp_fedprox", "fedprox"),
            ("dp_fedprox_adapted", "fedprox"),
            ("dp_scaffold", "scaffold"),
            ("dp_scaffold_adapted", "scaffold"),
        ):
            with self.subTest(method_id=method_id):
                gate, _, _ = _fixture(correction_kind=kind, method_id=method_id)
                self.assertEqual(gate.committed_steps, 0)
        for method_id, kind in (
            ("unknown_private_method", "none"),
            ("fedavg_nonprivate", "none"),
            ("dp_fedprox_adapted", "none"),
            ("dp_scaffold_adapted", "none"),
            ("dp_fedadam", "fedprox"),
        ):
            with (
                self.subTest(method_id=method_id, kind=kind),
                self.assertRaises(LocalTrainingGateError),
            ):
                _fixture(correction_kind=kind, method_id=method_id)

    def test_arbitrary_correction_mapping_is_rejected(self) -> None:
        gate, _, state = _fixture()
        forged = OrderedDict(((name, torch.zeros_like(value)) for (name, value) in state.items()))
        with (
            patch.object(
                local_training,
                "_draw_internal_poisson_mask",
                side_effect=_mask((False, False, False, False)),
            ),
            patch.object(local_training.torch, "randn") as gaussian,
        ):
            with self.assertRaises(LocalTrainingGatePoisoned):
                gate.execute_poisson_optimizer_step(
                    state, optimizer_step_index=1, correction=forged
                )
        gaussian.assert_not_called()

    def test_fedprox_is_internally_derived_and_binds_mu_anchor_and_source(self) -> None:
        gate, bridge, state = _fixture(correction_kind="fedprox")
        anchor = OrderedDict(((name, torch.zeros_like(value)) for (name, value) in state.items()))
        correction = build_fedprox_correction(
            bridge, anchor, mu=0.25, round_global_anchor_receipt_sha256=ANCHOR_EVIDENCE_SHA256
        )
        for value in anchor.values():
            value.add_(999.0)
        with (
            patch.object(
                local_training,
                "_draw_internal_poisson_mask",
                side_effect=_mask((False, False, False, False)),
            ),
            patch.object(local_training.torch, "randn", side_effect=_zero_randn),
        ):
            result = gate.execute_poisson_optimizer_step(
                state, optimizer_step_index=1, correction=correction
            )
        for name in state:
            torch.testing.assert_close(
                result.updated_state[name],
                state[name] - 0.1 * 0.25 * state[name],
                rtol=0.0,
                atol=1e-12,
            )
        self.assertEqual(result.receipt.correction_kind, "fedprox")

    def test_fedprox_wrong_mu_anchor_or_untrusted_source_fails_closed(self) -> None:
        for mode in ("mu", "anchor", "source"):
            gate, bridge, state = _fixture(correction_kind="fedprox")
            anchor = OrderedDict(
                (
                    (name, torch.ones_like(value) if mode == "anchor" else torch.zeros_like(value))
                    for (name, value) in state.items()
                )
            )
            source = (
                ANCHOR_EVIDENCE_SHA256
                if mode in {"mu", "anchor"}
                else hashlib.sha256(b"untrusted-anchor").hexdigest()
            )
            correction = build_fedprox_correction(
                bridge, anchor, mu=0.25, round_global_anchor_receipt_sha256=source
            )
            if mode == "mu":
                correction = replace(correction, mu=0.5)
            with (
                self.subTest(mode=mode),
                patch.object(
                    local_training,
                    "_draw_internal_poisson_mask",
                    side_effect=_mask((False, False, False, False)),
                ),
                patch.object(local_training.torch, "randn") as gaussian,
            ):
                with self.assertRaises(LocalTrainingGatePoisoned):
                    gate.execute_poisson_optimizer_step(
                        state, optimizer_step_index=1, correction=correction
                    )
                gaussian.assert_not_called()

    def test_scaffold_is_server_minus_client_and_control_tamper_fails(self) -> None:
        gate, bridge, state = _fixture(correction_kind="scaffold")
        server = OrderedDict(((name, torch.ones_like(value)) for (name, value) in state.items()))
        client = OrderedDict(((name, torch.zeros_like(value)) for (name, value) in state.items()))
        correction = build_scaffold_correction(
            bridge,
            server,
            client,
            provenance_kind="fixed_before_private_training",
            source_evidence_sha256=CONTROL_EVIDENCE_SHA256,
        )
        with (
            patch.object(
                local_training,
                "_draw_internal_poisson_mask",
                side_effect=_mask((False, False, False, False)),
            ),
            patch.object(local_training.torch, "randn", side_effect=_zero_randn),
        ):
            result = gate.execute_poisson_optimizer_step(
                state, optimizer_step_index=1, correction=correction
            )
        for name in state:
            torch.testing.assert_close(
                result.updated_state[name],
                state[name] - 0.1 * torch.ones_like(state[name]),
                rtol=0.0,
                atol=1e-12,
            )
        self.assertEqual(result.receipt.correction_kind, "scaffold")
        gate_bad, bridge_bad, state_bad = _fixture(correction_kind="scaffold")
        forged = build_scaffold_correction(
            bridge_bad,
            OrderedDict(((name, torch.ones_like(value)) for (name, value) in state_bad.items())),
            OrderedDict(((name, torch.zeros_like(value)) for (name, value) in state_bad.items())),
            provenance_kind="prior_dp_control_release",
            source_evidence_sha256=CONTROL_EVIDENCE_SHA256,
        )
        next(iter(forged.server_control.values())).add_(1.0)
        with (
            patch.object(
                local_training,
                "_draw_internal_poisson_mask",
                side_effect=_mask((False, False, False, False)),
            ),
            patch.object(local_training.torch, "randn") as gaussian,
        ):
            with self.assertRaises(LocalTrainingGatePoisoned):
                gate_bad.execute_poisson_optimizer_step(
                    state_bad, optimizer_step_index=1, correction=forged
                )
        gaussian.assert_not_called()

    def test_prior_dp_scaffold_source_must_precede_current_round(self) -> None:
        for source_round, succeeds in ((3, True), (5, False)):
            bridge, state = _bridge_and_state()
            current_context = _context(method_id="dp_scaffold")
            source_context = replace(current_context, federated_round=source_round)
            server = OrderedDict(
                ((name, torch.ones_like(value)) for (name, value) in state.items())
            )
            client = OrderedDict(
                ((name, torch.zeros_like(value)) for (name, value) in state.items())
            )
            binding = build_correction_source_binding(
                bridge,
                server,
                secondary_state=client,
                source_kind="prior_dp_scaffold_controls",
                source_evidence_sha256=CONTROL_EVIDENCE_SHA256,
                source_context=source_context,
            )
            policy = CorrectionPolicy("scaffold", CANDIDATE_SHA256, None, (binding,))
            gate = PoissonLocalTrainingGate(
                (PoissonDPStage("local_train", 0.5, 0.5, 1),),
                gradient_bridge=bridge,
                population_row_ids=bridge.canonical_population_row_ids,
                clip_norm=1.0,
                learning_rate=0.1,
                delta=1e-05,
                sampling_seed=101,
                noise_seed=202,
                context=current_context,
                correction_policy=policy,
                initial_model_state_sha256=model_state_sha256(bridge.model, state),
                initial_model_state_source_sha256=INITIAL_STATE_SOURCE_SHA256,
            )
            correction = build_scaffold_correction(
                bridge,
                server,
                client,
                provenance_kind="prior_dp_control_release",
                source_evidence_sha256=CONTROL_EVIDENCE_SHA256,
            )
            with (
                self.subTest(source_round=source_round),
                patch.object(
                    local_training,
                    "_draw_internal_poisson_mask",
                    side_effect=_mask((False, False, False, False)),
                ),
                patch.object(local_training.torch, "randn", side_effect=_zero_randn) as gaussian,
            ):
                if succeeds:
                    result = gate.execute_poisson_optimizer_step(
                        state, optimizer_step_index=1, correction=correction
                    )
                    self.assertEqual(result.receipt.correction_kind, "scaffold")
                    self.assertGreater(gaussian.call_count, 0)
                else:
                    with self.assertRaises(LocalTrainingGatePoisoned):
                        gate.execute_poisson_optimizer_step(
                            state, optimizer_step_index=1, correction=correction
                        )
                    gaussian.assert_not_called()


class ReceiptAndAccountingOrderTests(unittest.TestCase):

    def test_first_input_state_is_preregistered_before_sampling(self) -> None:
        gate, _, state = _fixture()
        forged = OrderedDict(((name, value.clone()) for (name, value) in state.items()))
        next(iter(forged.values())).add_(1.0)
        with patch.object(local_training, "_draw_internal_poisson_mask") as sampling:
            with self.assertRaises(LocalTrainingGatePoisoned):
                gate.execute_poisson_optimizer_step(forged, optimizer_step_index=1)
        sampling.assert_not_called()

    def test_accounting_occurs_after_noise_kernel_and_update(self) -> None:
        gate, _, state = _fixture()
        events: list[str] = []

        def observed(name, function):

            def wrapper(*args, **kwargs):
                events.append(name)
                return function(*args, **kwargs)

            return wrapper

        with (
            patch.object(
                local_training,
                "_draw_internal_poisson_mask",
                side_effect=_mask((True, False, True, False)),
            ),
            patch.object(
                local_training,
                "_draw_full_parameter_gaussian_noise",
                side_effect=observed("noise", local_training._draw_full_parameter_gaussian_noise),
            ),
            patch.object(
                local_training,
                "privatize_per_record_gradients",
                side_effect=observed("kernel", local_training.privatize_per_record_gradients),
            ),
            patch.object(
                local_training,
                "_apply_gradient_update",
                side_effect=observed("update", local_training._apply_gradient_update),
            ),
            patch.object(
                gate._accountant,
                "record_optimizer_step",
                side_effect=observed("account", gate._accountant.record_optimizer_step),
            ),
        ):
            gate.execute_poisson_optimizer_step(state, optimizer_step_index=1)
        self.assertEqual(events, ["noise", "kernel", "update", "account"])

    def test_accounting_failure_after_update_poisons_and_returns_no_state(self) -> None:
        gate, _, state = _fixture()
        before = OrderedDict(((name, value.clone()) for (name, value) in state.items()))
        update_calls = 0
        original_update = local_training._apply_gradient_update

        def count_update(*args, **kwargs):
            nonlocal update_calls
            update_calls += 1
            return original_update(*args, **kwargs)

        with (
            patch.object(
                local_training,
                "_draw_internal_poisson_mask",
                side_effect=_mask((True, False, True, False)),
            ),
            patch.object(local_training, "_apply_gradient_update", side_effect=count_update),
            patch.object(
                gate._accountant,
                "record_optimizer_step",
                side_effect=PrivacyAccountingError("injected accounting failure"),
            ),
        ):
            with self.assertRaises(LocalTrainingGatePoisoned):
                gate.execute_poisson_optimizer_step(state, optimizer_step_index=1)
        self.assertEqual(update_calls, 1)
        self.assertEqual(gate.committed_steps, 0)
        self.assertTrue(gate.poisoned)
        for name in state:
            self.assertTrue(torch.equal(state[name], before[name]))

    def test_empty_draw_still_adds_all_parameter_noise_and_accounts(self) -> None:
        gate, _, state = _fixture(q=0.25)
        with (
            patch.object(
                local_training,
                "_draw_internal_poisson_mask",
                side_effect=_mask((False, False, False, False)),
            ),
            patch.object(local_training.torch, "randn", side_effect=_zero_randn) as gaussian,
        ):
            result = gate.execute_poisson_optimizer_step(state, optimizer_step_index=1)
        self.assertEqual(gaussian.call_count, len(state))
        self.assertTrue(result.receipt.empty_poisson_draw)
        self.assertEqual(result.receipt.sampled_record_count, 0)
        report = gate.close(executed_optimizer_steps=1)
        self.assertTrue(report.complete)
        self.assertEqual(report.empty_sample_steps, 1)
        self.assertEqual(
            report.execution_history[0].mechanism_evidence_sha256,
            result.receipt.mechanism_evidence_sha256,
        )
        self.assertTrue(report.evidence_confidentiality.endswith("do_not_publish"))

    def test_receipt_tamper_rehash_and_output_mutation_fail_ledger_validation(self) -> None:
        gate, _, state = _fixture()
        with patch.object(
            local_training,
            "_draw_internal_poisson_mask",
            side_effect=_mask((True, False, True, False)),
        ):
            result = gate.execute_poisson_optimizer_step(state, optimizer_step_index=1)
        tampered = replace(result.receipt, sample_rate=0.25)
        with self.assertRaises(LocalTrainingGateError):
            poisson_step_receipt_fingerprint(tampered)
        rebuilt = replace(
            tampered,
            receipt_sha256=local_training._sha256_json(local_training._receipt_payload(tampered)),
        )
        poisson_step_receipt_fingerprint(rebuilt)
        with self.assertRaisesRegex(LocalTrainingGateError, "committed evidence ledger"):
            gate.validate_step_result(PoissonStepResult(result.updated_state, rebuilt), state)
        next(iter(result.updated_state.values())).add_(1.0)
        with self.assertRaisesRegex(LocalTrainingGateError, "state lineage"):
            gate.validate_step_result(result, state)

    def test_two_step_state_and_receipt_chain_is_enforced(self) -> None:
        gate, _, state = _fixture(steps=2)
        with patch.object(
            local_training,
            "_draw_internal_poisson_mask",
            side_effect=_mask((False, False, False, False)),
        ):
            first = gate.execute_poisson_optimizer_step(state, optimizer_step_index=1)
            second = gate.execute_poisson_optimizer_step(
                first.updated_state, optimizer_step_index=2
            )
        self.assertEqual(second.receipt.previous_receipt_sha256, first.receipt.receipt_sha256)
        bad_gate, _, bad_state = _fixture(steps=2)
        with patch.object(
            local_training,
            "_draw_internal_poisson_mask",
            side_effect=_mask((False, False, False, False)),
        ):
            bad_gate.execute_poisson_optimizer_step(bad_state, optimizer_step_index=1)
            with self.assertRaises(LocalTrainingGatePoisoned):
                bad_gate.execute_poisson_optimizer_step(bad_state, optimizer_step_index=2)

    def test_next_round_gate_binds_prior_output_and_receipt(self) -> None:
        first_gate, first_bridge, state = _fixture(federated_round=4)
        with patch.object(
            local_training,
            "_draw_internal_poisson_mask",
            side_effect=_mask((False, False, False, False)),
        ):
            first = first_gate.execute_poisson_optimizer_step(state, optimizer_step_index=1)
        prior_output_hash = model_state_sha256(first_bridge.model, first.updated_state)
        second_gate, _, _ = _fixture(
            federated_round=5,
            initial_model_state_sha256=prior_output_hash,
            initial_model_state_source_sha256=first.receipt.receipt_sha256,
        )
        with patch.object(
            local_training,
            "_draw_internal_poisson_mask",
            side_effect=_mask((False, False, False, False)),
        ):
            second = second_gate.execute_poisson_optimizer_step(
                first.updated_state, optimizer_step_index=1
            )
        self.assertEqual(second.receipt.initial_model_state_sha256, prior_output_hash)
        self.assertEqual(
            second.receipt.initial_model_state_source_sha256, first.receipt.receipt_sha256
        )


class PrivateAndNonprivateSeparationTests(unittest.TestCase):

    def test_nonprivate_explicit_minibatches_never_enter_dp_path(self) -> None:
        start = OrderedDict(x=torch.tensor([1.0], dtype=torch.float64))

        def gradient(state, batch):
            del state
            self.assertEqual(batch, "registered_batch")
            return OrderedDict(x=torch.tensor([2.0], dtype=torch.float64))

        with (
            patch.object(
                local_training,
                "privatize_per_record_gradients",
                side_effect=AssertionError("non-private path reached DP kernel"),
            ) as private_kernel,
            patch.object(
                local_training,
                "RuntimePrefixAccountant",
                side_effect=AssertionError("non-private path created accountant"),
            ) as accountant,
        ):
            observed = canonical_nonprivate_explicit_minibatches(
                start, (("registered_batch",),), gradient, 0.1
            )
        torch.testing.assert_close(
            observed["x"], torch.tensor([0.8], dtype=torch.float64), rtol=0.0, atol=1e-12
        )
        private_kernel.assert_not_called()
        accountant.assert_not_called()


if __name__ == "__main__":
    unittest.main()
