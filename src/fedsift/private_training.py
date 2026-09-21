"""Private-model release entry point with an explicit output allowlist.

Public benchmark replay is a separate operation. This entry point always uses
fresh operating-system randomness and returns no record memberships, seeds,
raw-gradient commitments, execution histories, or internal audit receipts.
"""
from .train_unit import TrainingUnitError, execute_training_unit
from .local_training import PRIVATE_RNG_CLAIM


def train_private_model(capability, sealed_data, budget, *,
                        expected_capability_sha256, expected_budget_sha256):
    """Release one model trained under fixed public auxiliary configuration.

    The caller fixes preprocessing, client reference slots, and hyperparameters
    before training. The reported budget covers this training run; selecting
    those inputs from private records or releasing other runs requires its own
    accounting. Internal training objects must remain within the trusted process.
    """
    result = execute_training_unit(
        capability, sealed_data, budget,
        expected_capability_sha256=expected_capability_sha256,
        expected_budget_sha256=expected_budget_sha256,
        randomness_mode="private",
    )
    if result.parallel_privacy_report is None or not result.local_step_receipts:
        raise TrainingUnitError("private release requires an accounted private method")
    if any(r.rng_security_claim != PRIVATE_RNG_CLAIM for r in result.local_step_receipts):
        raise TrainingUnitError("reproducible benchmark noise cannot be privately released")
    report = result.parallel_privacy_report
    return {
        "method": result.dispatch_spec.method_id,
        "model_manifest": {
            key: result.artifact["model_manifest"][key]
            for key in ("architecture", "device", "dtype", "family", "parameters", "training_loss")
        },
        "model_state": {name: value.detach().cpu().tolist() for name, value in result.model_state.items()},
        "privacy": {
            "epsilon": report.epsilon,
            "delta": report.delta,
            "adjacency": "add_remove_one_record_with_fixed_public_auxiliary_information",
            "release": "one_final_model",
        },
    }


__all__ = ["train_private_model"]
