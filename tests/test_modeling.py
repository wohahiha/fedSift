from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
import hashlib
import unittest
from collections import OrderedDict
from dataclasses import replace
from unittest.mock import patch
import torch
import torch.nn.functional as F
import fedsift.modeling as modeling
from fedsift.local_training import CorrectionPolicy, LocalTrainingContext, PoissonLocalTrainingGate
from fedsift.modeling import (
    InitializationDomain,
    LogisticScreening,
    ModelingError,
    SelectedBCEGradientBridge,
    ScreeningMLP,
    build_fixed_model,
    extract_model_state,
    fixed_model_manifest,
    fixed_model_manifest_sha256,
    model_state_sha256,
    predict_logits,
    predict_probabilities,
    selected_per_record_bce_gradients,
    validate_fixed_model,
    validate_model_state,
)
from fedsift.privacy_accounting import PoissonDPStage


def _domain(*, client_id: str = "client_0") -> InitializationDomain:
    return InitializationDomain(
        study_id=_identity("modeling_test"),
        outer_repeat=1,
        outer_fold=2,
        client_id=client_id,
        model_role="private_local_screening",
    )


def _table() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.tensor(
        [[0.2, -0.1, 0.5], [1.0, 0.3, -0.7], [-0.4, 0.9, 0.1], [0.8, -0.2, 0.6]],
        dtype=torch.float64,
    )
    labels = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float64)
    row_ids = torch.tensor([101, 205, 309, 412], dtype=torch.int64)
    return (features, labels, row_ids)


def _literal_autograd_loop(
    model: ScreeningMLP | LogisticScreening,
    features: torch.Tensor,
    labels: torch.Tensor,
    positions: tuple[int, ...],
) -> OrderedDict[str, torch.Tensor]:
    names = tuple((name for (name, _) in model.named_parameters()))
    rows: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    parameters = tuple(model.parameters())
    for position in positions:
        logit = model(features[position : position + 1])[0]
        loss = F.binary_cross_entropy_with_logits(logit, labels[position], reduction="sum")
        gradients = torch.autograd.grad(loss, parameters)
        for name, value in zip(names, gradients):
            rows[name].append(value.detach().clone())
    return OrderedDict(((name, torch.stack(rows[name], dim=0)) for name in names))


class FixedModelManifestAndPredictionTests(unittest.TestCase):

    def test_architecture_manifest_hash_and_domain_initialization_are_deterministic(self) -> None:
        rng_before = torch.random.get_rng_state().clone()
        first = ScreeningMLP(3, initialization_seed=17, initialization_domain=_domain())
        rng_after = torch.random.get_rng_state().clone()
        second = ScreeningMLP(3, initialization_seed=17, initialization_domain=_domain())
        other_domain = ScreeningMLP(
            3, initialization_seed=17, initialization_domain=_domain(client_id="client_1")
        )
        self.assertTrue(torch.equal(rng_before, rng_after))
        self.assertEqual(first.model_manifest_sha256, second.model_manifest_sha256)
        self.assertNotEqual(first.model_manifest_sha256, other_domain.model_manifest_sha256)
        first_state = extract_model_state(first)
        second_state = extract_model_state(second)
        other_state = extract_model_state(other_domain)
        for name in first_state:
            self.assertTrue(torch.equal(first_state[name], second_state[name]))
        self.assertTrue(
            any((not torch.equal(first_state[name], other_state[name]) for name in first_state))
        )
        manifest = first.model_manifest()
        self.assertEqual(fixed_model_manifest(first), manifest)
        self.assertEqual(fixed_model_manifest_sha256(first), first.model_manifest_sha256)
        self.assertEqual(manifest["family"], "screening_mlp")
        self.assertEqual(manifest["architecture"]["hidden_widths"], [16, 8])
        self.assertEqual(manifest["dtype"], "torch.float64")
        self.assertEqual(manifest["performance_fields_used"], [])
        self.assertTrue(manifest["result_blind_model_choice"])

    def test_logistic_manifest_and_logits_probabilities(self) -> None:
        model = build_fixed_model(
            "logistic_screening", 3, initialization_seed=23, initialization_domain=_domain()
        )
        self.assertIs(type(model), LogisticScreening)
        manifest = model.model_manifest()
        self.assertEqual(manifest["architecture"]["hidden_widths"], [])
        self.assertEqual(
            manifest["architecture"]["convex_objective"], "binary_cross_entropy_with_logits"
        )
        features, _, _ = _table()
        logits = predict_logits(model, features)
        probabilities = predict_probabilities(model, features)
        self.assertEqual(logits.shape, (4,))
        self.assertEqual(probabilities.shape, (4,))
        torch.testing.assert_close(probabilities, torch.sigmoid(logits))
        self.assertTrue(bool(((probabilities >= 0) & (probabilities <= 1)).all()))

    def test_manifest_parameter_or_state_tampering_fails_closed(self) -> None:
        model = LogisticScreening(3, initialization_seed=29, initialization_domain=_domain())
        state = extract_model_state(model)
        reversed_state = OrderedDict(reversed(tuple(state.items())))
        with self.assertRaises(ModelingError):
            validate_model_state(model, reversed_state)
        wrong_dtype = OrderedDict(((name, value.float()) for (name, value) in state.items()))
        with self.assertRaises(ModelingError):
            validate_model_state(model, wrong_dtype)
        nonfinite = OrderedDict(((name, value.clone()) for (name, value) in state.items()))
        nonfinite["output.weight"][0, 0] = float("nan")
        with self.assertRaises(ModelingError):
            validate_model_state(model, nonfinite)
        model._manifest_sha256 = "0" * 64
        with self.assertRaises(ModelingError):
            validate_fixed_model(model)


class SelectedPerRecordGradientOracleTests(unittest.TestCase):

    def _compare_family(self, model: ScreeningMLP | LogisticScreening) -> None:
        features, labels, row_ids = _table()
        selected_ids = torch.tensor([309, 101], dtype=torch.int64)
        observed = selected_per_record_bce_gradients(
            model, extract_model_state(model), features, labels, row_ids, selected_ids
        )
        literal = _literal_autograd_loop(model, features, labels, (2, 0))
        self.assertEqual(tuple(observed), tuple(literal))
        for name in observed:
            torch.testing.assert_close(observed[name], literal[name], rtol=1e-11, atol=1e-12)

    def test_mlp_vmap_gradients_match_independent_literal_autograd_loop(self) -> None:
        self._compare_family(
            ScreeningMLP(3, initialization_seed=31, initialization_domain=_domain())
        )

    def test_logistic_vmap_gradients_match_independent_literal_autograd_loop(self) -> None:
        self._compare_family(
            LogisticScreening(3, initialization_seed=37, initialization_domain=_domain())
        )

    def test_empty_selection_returns_zero_row_shapes_without_functional_call(self) -> None:
        model = ScreeningMLP(3, initialization_seed=41, initialization_domain=_domain())
        features = torch.full((3, 3), float("nan"), dtype=torch.float64)
        labels = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
        row_ids = torch.tensor([10, 20, 30], dtype=torch.int64)
        with patch.object(
            modeling,
            "functional_call",
            side_effect=AssertionError("empty selection evaluated a row"),
        ) as call:
            gradients = selected_per_record_bce_gradients(
                model,
                extract_model_state(model),
                features,
                labels,
                row_ids,
                torch.empty((0,), dtype=torch.int64),
            )
        call.assert_not_called()
        for name, parameter in model.named_parameters():
            self.assertEqual(tuple(gradients[name].shape), (0, *tuple(parameter.shape)))
            self.assertEqual(gradients[name].dtype, torch.float64)

    def test_unselected_invalid_sentinel_is_not_touched_but_selected_invalid_is_rejected(
        self,
    ) -> None:
        model = LogisticScreening(3, initialization_seed=43, initialization_domain=_domain())
        features = torch.tensor([[0.1, 0.2, 0.3], [float("nan"), 1.0, 2.0]], dtype=torch.float64)
        labels = torch.tensor([1.0, 2.0], dtype=torch.float64)
        row_ids = torch.tensor([5, 9], dtype=torch.int64)
        gradients = selected_per_record_bce_gradients(
            model,
            extract_model_state(model),
            features,
            labels,
            row_ids,
            torch.tensor([5], dtype=torch.int64),
        )
        self.assertTrue(all((value.shape[0] == 1 for value in gradients.values())))
        with self.assertRaises(ModelingError):
            selected_per_record_bce_gradients(
                model,
                extract_model_state(model),
                features,
                labels,
                row_ids,
                torch.tensor([9], dtype=torch.int64),
            )

    def test_bound_bridge_batch_rebuilds_ids_state_table_and_gradient_hashes(self) -> None:
        model = LogisticScreening(3, initialization_seed=47, initialization_domain=_domain())
        features, labels, row_ids = _table()
        state = extract_model_state(model)
        bridge = SelectedBCEGradientBridge(model, features, labels, row_ids)
        first = bridge.selected_gradient_batch(state, (101, 309))
        validated = bridge.validate_batch(first, state, (101, 309))
        self.assertEqual(first.selected_row_ids, (101, 309))
        self.assertEqual(first.parameter_names, tuple(state))
        self.assertEqual(tuple(validated), tuple(state))
        features.add_(999.0)
        labels.zero_()
        row_ids.add_(1000)
        second = bridge.selected_gradient_batch(state, (101, 309))
        self.assertEqual(first.source_table_sha256, second.source_table_sha256)
        self.assertEqual(first.gradient_tensor_sha256, second.gradient_tensor_sha256)
        self.assertEqual(first.gradient_lineage_sha256, second.gradient_lineage_sha256)
        swapped_gradients = OrderedDict(
            ((name, value.flip(0)) for (name, value) in first.gradients.items())
        )
        with self.assertRaises(ModelingError):
            bridge.validate_batch(replace(first, gradients=swapped_gradients), state, (101, 309))

    def test_bridge_accepts_explicit_shuffled_selection_and_binds_its_order(self) -> None:
        model = LogisticScreening(3, initialization_seed=47, initialization_domain=_domain())
        features, labels, row_ids = _table()
        bridge = SelectedBCEGradientBridge(model, features, labels, row_ids)
        state = extract_model_state(model)
        canonical = bridge.selected_gradient_batch(state, (101, 309))
        shuffled = bridge.selected_gradient_batch(state, (309, 101))
        validated = bridge.validate_batch(shuffled, state, (309, 101))
        self.assertEqual(shuffled.selected_row_ids, (309, 101))
        self.assertNotEqual(shuffled.selected_row_ids_sha256, canonical.selected_row_ids_sha256)
        self.assertNotEqual(shuffled.gradient_lineage_sha256, canonical.gradient_lineage_sha256)
        for name in canonical.gradients:
            torch.testing.assert_close(
                shuffled.gradients[name], canonical.gradients[name].flip(0), rtol=0.0, atol=0.0
            )
            torch.testing.assert_close(
                validated[name], shuffled.gradients[name], rtol=0.0, atol=0.0
            )
        with self.assertRaises(ModelingError):
            bridge.validate_batch(shuffled, state, (101, 309))

    def test_bridge_rejects_malformed_duplicate_and_out_of_population_selection(self) -> None:
        model = LogisticScreening(3, initialization_seed=47, initialization_domain=_domain())
        features, labels, row_ids = _table()
        bridge = SelectedBCEGradientBridge(model, features, labels, row_ids)
        state = extract_model_state(model)
        invalid_selections = ((101, True), (101, 309.0), (-1, 101), (101, 101), (101, 999))
        for selected in invalid_selections:
            with self.subTest(selected=selected), self.assertRaises(ModelingError):
                bridge.selected_gradient_batch(state, selected)


class ModelingInputAndLocalGateIntegrationTests(unittest.TestCase):

    def test_invalid_model_rows_labels_features_and_selection_fail_closed(self) -> None:
        with self.assertRaises(ModelingError):
            build_fixed_model(
                "performance_selected_family",
                3,
                initialization_seed=1,
                initialization_domain=_domain(),
            )
        with self.assertRaises(ModelingError):
            ScreeningMLP(0, initialization_seed=1, initialization_domain=_domain())
        with self.assertRaises(ModelingError):
            InitializationDomain(
                study_id="",
                outer_repeat=0,
                outer_fold=0,
                client_id="client_0",
                model_role="screening",
            )
        model = LogisticScreening(3, initialization_seed=47, initialization_domain=_domain())
        features, labels, row_ids = _table()
        state = extract_model_state(model)
        bad_cases = (
            (features, labels, torch.tensor([1, 1, 2, 3]), torch.tensor([1])),
            (features, labels, row_ids, torch.tensor([999])),
            (features, labels, row_ids, torch.tensor([101, 101])),
            (features.float(), labels, row_ids, torch.tensor([101])),
            (features, labels, torch.tensor([-1, 2, 3, 4]), torch.tensor([2])),
        )
        for feature_value, label_value, ids, selected in bad_cases:
            with self.subTest(selected=selected), self.assertRaises(ModelingError):
                selected_per_record_bce_gradients(
                    model, state, feature_value, label_value, ids, selected
                )

    def test_toy_selected_gradients_feed_poisson_gate_and_bind_receipt(self) -> None:
        model = LogisticScreening(2, initialization_seed=53, initialization_domain=_domain())
        state = extract_model_state(model)
        features = torch.tensor(
            [[0.2, 0.5], [1.0, -0.4], [-0.3, 0.8], [0.7, 0.1]], dtype=torch.float64
        )
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float64)
        row_ids = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
        bridge = SelectedBCEGradientBridge(model, features, labels, row_ids)
        candidate_hash = hashlib.sha256(b"modeling-integration-candidate").hexdigest()
        gate = PoissonLocalTrainingGate(
            (PoissonDPStage("toy_train", 0.5, 2.0, 1),),
            gradient_bridge=bridge,
            population_row_ids=bridge.canonical_population_row_ids,
            clip_norm=1.0,
            learning_rate=0.05,
            delta=1e-05,
            sampling_seed=17,
            noise_seed=19,
            context=LocalTrainingContext(
                study_id=_identity("modeling_test"),
                dataset_id="toy",
                outer_repeat=1,
                outer_fold=2,
                inner_fold=0,
                seed_repeat=0,
                client_id="client_0",
                federated_round=0,
                method_id="dp_fedavg",
                candidate_sha256=candidate_hash,
            ),
            correction_policy=CorrectionPolicy("none", candidate_hash, None, ()),
            initial_model_state_sha256=model_state_sha256(model, state),
            initial_model_state_source_sha256=hashlib.sha256(
                b"fixed-model-initialization"
            ).hexdigest(),
        )
        with (
            patch(
                "fedsift.local_training._draw_internal_poisson_mask",
                return_value=torch.tensor([True, False, True, False]),
            ),
            patch(
                "fedsift.local_training.torch.randn",
                side_effect=tuple((torch.zeros_like(value) for value in state.values())),
            ),
        ):
            result = gate.execute_poisson_optimizer_step(state, optimizer_step_index=1)
        self.assertEqual(result.receipt.sampled_record_count, 2)
        self.assertEqual(result.receipt.parameter_names, tuple(state))
        self.assertEqual(result.receipt.fixed_expected_batch_normalization, 2.0)
        self.assertEqual(result.receipt.accountant_steps_after, 1)
        self.assertEqual(tuple(result.updated_state), tuple(state))
        self.assertEqual(result.receipt.input_model_state_sha256, model_state_sha256(model, state))
        gate.validate_step_result(result, state)


if __name__ == "__main__":
    unittest.main()
