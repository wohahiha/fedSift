from __future__ import annotations
from fedsift.artifact_contract import identity as _identity
from fedsift.experiment_io import (
    registered_study,
    results_root,
    resolve_record_path,
    verify_release_sources,
)
import functools
import hashlib
import json
from pathlib import Path
from fedsift.candidate_space import canonical_sha256
from fedsift.cost_benchmark import measure_linux_process_rss
from fedsift.resource_study import (
    ResourceStudyScope,
    build_resource_study,
    run_real_data_cost_benchmark,
)
from fedsift.runtime_environment import RuntimeEnvironmentGuard
from fedsift.study_factory import build_study_construction, propose_study
from fedsift.training_budget_factory import SharedTrainingBudgetPolicy
from fedsift.study_design import load_study_design

ROOT = Path(__file__).resolve().parents[1]
TOP = results_root()
OUT = TOP / "resource_benchmark"
OUT.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


environment = json.loads((ROOT / "environment/runtime_contract.json").read_text())
environment_sha256 = canonical_sha256(environment)
guard = RuntimeEnvironmentGuard(environment, expected_sha256=environment_sha256)
checks = [guard.check("before_resource_study_construction")]
implementation = registered_study()["implementation_sha256"]
protocol = registered_study()["resource_protocol"]
protocol_sha256 = canonical_sha256(protocol)
client_policy_sha256 = "356fac82397469a91b4da89e27d7557590236d4fdf0df20259881c6bbff423c7"
study_kwargs = dict(
    study_id=_identity("pima_result_blind_cost_benchmark"),
    protocol_sha256=protocol_sha256,
    implementation_sha256=implementation,
    client_policy_sha256=client_policy_sha256,
    candidate_count=10,
    hpo_seeds=(101,),
    max_steps=5,
    preprocessing_profile_name="pima_primary_zero_as_missing_v1",
)
proposal = propose_study(ROOT, "pima", **study_kwargs)
(OUT / "proposal.json").write_text(
    json.dumps(
        {"bundle": proposal.bundle, "proposed_bundle_sha256": proposal.proposed_bundle_sha256},
        ensure_ascii=False,
        indent=2,
    )
)
construction = build_study_construction(
    ROOT, "pima", **study_kwargs, expected_bundle_sha256=proposal.proposed_bundle_sha256
)
scope = ResourceStudyScope(outer_repeat=0, outer_fold=0, inner_fold=0, hpo_seed=101)
policy = SharedTrainingBudgetPolicy(
    server_rounds=5,
    local_epochs=1,
    poisson_sample_rate=1.0,
    clip_norm=1.0,
    target_epsilon=8.0,
    target_delta=1e-05,
    model_family="logistic_screening",
    participating_client_ids=tuple((f"client_{i}" for i in range(5))),
    paired_initialization_seed=20260901,
    nonprivate_order_seed=_identity("pima_result_blind_cost_benchmark_nonprivate_order"),
    partition_mode="fixed_label_driven_auxiliary_condition",
    fixed_auxiliary_partition_condition_sha256=client_policy_sha256,
)
preparation = build_resource_study(
    construction,
    ROOT,
    expected_bundle_sha256=proposal.proposed_bundle_sha256,
    scope=scope,
    expected_scope_sha256=scope.scope_sha256,
    shared_policy=policy,
    expected_policy_sha256=policy.policy_sha256,
    warmup_passes=1,
    latin_square_repetitions=1,
    method_ids=load_study_design()["main_methods"],
    order_seed=_identity("pima_result_blind_cost_benchmark_order"),
)
preparation_record = {
    "environment_sha256": environment_sha256,
    "implementation_sha256": implementation,
    "protocol_sha256": protocol_sha256,
    "study_bundle_sha256": proposal.proposed_bundle_sha256,
    "scope_sha256": scope.scope_sha256,
    "policy_sha256": policy.policy_sha256,
    "preparation_manifest": preparation.manifest(),
    "preparation_sha256": preparation.preparation_sha256,
    "cost_capability": preparation.cost_capability,
    "cost_capability_sha256": preparation.cost_capability["capability_sha256"],
    "result_blind": True,
}
(OUT / "preparation.json").write_text(json.dumps(preparation_record, ensure_ascii=False, indent=2))
checks.append(guard.check("after_resource_study_preparation"))
backend = guard.measurement_backend(
    functools.partial(measure_linux_process_rss, sample_interval_ns=5000000)
)
artifact = run_real_data_cost_benchmark(
    preparation,
    expected_preparation_sha256=preparation.preparation_sha256,
    expected_capability_sha256=preparation.cost_capability["capability_sha256"],
    measurement_backend=backend,
)
checks.append(guard.check("after_resource_benchmark"))
result = {
    "schema": _identity("guarded_pima_cost_benchmark_record"),
    "status": "complete_result_blind_cost_only",
    "environment_checks": checks,
    "environment_sha256": environment_sha256,
    "implementation_sha256": implementation,
    "protocol_sha256": protocol_sha256,
    "study_bundle_sha256": proposal.proposed_bundle_sha256,
    "preparation_sha256": preparation.preparation_sha256,
    "cost_capability_sha256": preparation.cost_capability["capability_sha256"],
    "artifact": artifact,
    "predictions_accessed": False,
    "performance_metrics_accessed": False,
    "outer_test_accessed": False,
}
(OUT / "benchmark.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(
    json.dumps(
        {
            "status": result["status"],
            "benchmark_sha256": canonical_sha256(result),
            "environment_check_count": guard.check_count,
            "artifact_sha256": artifact["artifact_sha256"],
        }
    ),
    flush=True,
)
