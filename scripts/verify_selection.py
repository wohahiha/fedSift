"""Recompute the manuscript's hyperparameter decisions from saved inner predictions."""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

from fedsift.candidate_space import build_candidate_space
from fedsift.hpo_select import _candidate_evaluation, _sort_key
from fedsift.study_design import load_study_design, paper_training_units, validate_candidate_pairing

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_selection(reference):
    design = load_study_design()
    evidence = read(reference / "selection_evidence.manifest.json")
    manifest = {row["path"]: row for row in evidence["files"]}
    comparisons = []
    maximum = 0.0
    verified_units = 0
    with ZipFile(ROOT / evidence["archive"]) as archive:
        for dataset in design["datasets"]:
            closed = reference / "selection" / dataset / "closed_evidence"
            receipts = read(closed / "attempt_receipts.json")
            units = paper_training_units({"units": [dict(receipt["unit_binding"], unit_id=receipt["unit_id"])
                                                     for receipt in receipts]})
            scopes = defaultdict(lambda: defaultdict(list))
            for unit in units:
                scopes[(unit["method"], unit["outer_repeat"], unit["outer_fold"])][unit["candidate_id"]].append(unit)
            proposal = read(reference / "selection" / dataset / "proposal.json")
            space = build_candidate_space(proposal["bundle"]["source_binding"]["study_id"],
                                          candidate_count=design["candidates_per_method"])
            validate_candidate_pairing(space, design)
            for (method, repeat, fold), candidates in sorted(scopes.items()):
                decision = read(closed / "selection_decisions" / f"r{repeat}_f{fold}_{method}.json")
                recorded = {row["candidate_id"]: row for row in decision["candidate_evaluations"]}
                evaluations = []
                for candidate, candidate_units in sorted(candidates.items()):
                    outputs = {}
                    for unit in candidate_units:
                        name = f"{dataset}/{unit['unit_id']}.json"
                        payload = archive.read(name)
                        if hashlib.sha256(payload).hexdigest() != manifest[name]["sha256"]:
                            raise ValueError(f"selection evidence changed: {name}")
                        artifact = json.loads(payload)
                        if artifact["unit_id"] != unit["unit_id"] or artifact["outer_test_accessed"]:
                            raise ValueError(f"selection input identity or boundary differs: {name}")
                        outputs[unit["unit_id"]] = artifact["hpo_output_manifest"]
                        verified_units += 1
                    evaluation = _candidate_evaluation(
                        {}, space, outputs, method=method, outer_repeat=repeat, outer_fold=fold,
                        candidate_id=candidate, hpo_seeds=tuple(design["training_seeds"]),
                        inner_folds=tuple(design["inner_folds"]), candidate_units=candidate_units,
                    )
                    for metric, value in evaluation["aggregate_metrics"].items():
                        expected = recorded[candidate]["aggregate_metrics"][metric]
                        difference = abs(value - expected)
                        maximum = max(maximum, difference)
                        if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12):
                            raise ValueError(f"selection metric differs: {dataset}/{method}/{fold}/{candidate}/{metric}")
                    evaluations.append(evaluation)
                selected = min(evaluations, key=_sort_key)["candidate_id"]
                if selected != decision["selected_candidate"]["candidate_id"]:
                    raise ValueError(f"selected candidate differs: {dataset}/{method}/{fold}")
                comparisons.append({"dataset": dataset, "method": method, "outer_fold": fold,
                                    "candidates": len(evaluations), "selected_candidate_id": selected})
                print(json.dumps(comparisons[-1]), flush=True)
    return {"status": "PASS", "training_units": verified_units, "decisions": comparisons,
            "decision_count": len(comparisons), "maximum_metric_difference": maximum,
            "outer_test_predictions_used": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_selection(ROOT / "output/reference")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "decisions"}), flush=True)


if __name__ == "__main__":
    main()
