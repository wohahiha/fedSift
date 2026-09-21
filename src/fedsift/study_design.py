"""The experiment matrix used by the manuscript and active command line."""
import json
from pathlib import Path


def load_study_design():
    return json.loads((Path(__file__).resolve().parents[2] / "config/study.json").read_text(encoding="utf-8"))


def validate_candidate_pairing(candidate_space, design=None):
    """Check the declared optimizer pairs before reading training outcomes."""
    design = load_study_design() if design is None else design
    from .candidate_space import candidate_by_id

    for left, right in design["candidate_pairing"]:
        components = ("local_optimizer", "backend") if left == "dp_fedadam" else ("local_optimizer",)
        for index in range(design["candidates_per_method"]):
            candidate = f"candidate_{index:04}"
            first = candidate_by_id(candidate_space, left, candidate)["parameters"]
            second = candidate_by_id(candidate_space, right, candidate)["parameters"]
            if any(first[name] != second[name] for name in components):
                raise ValueError(f"paired optimizer coordinates differ: {left}, {right}, {candidate}")


def paper_training_units(plan, design=None):
    """Select complete inner-training scopes without consulting any results."""
    design = load_study_design() if design is None else design
    selected = [unit for unit in plan["units"]
                if unit["method"] in design["main_methods"]
                and unit["outer_repeat"] in design["outer_repeats"]
                and unit["outer_fold"] in design["outer_folds"]]
    expected = (len(design["main_methods"]) * len(design["outer_repeats"])
                * len(design["outer_folds"]) * len(design["inner_folds"])
                * design["candidates_per_method"] * len(design["training_seeds"]))
    keys = {(u["method"], u["outer_repeat"], u["outer_fold"], u["inner_fold"],
             u["candidate_id"], u["hpo_seed"]) for u in selected}
    if len(selected) != expected or len(keys) != expected:
        raise ValueError("paper training matrix is incomplete or duplicated")
    expected_pairs = {(fold, seed) for fold in design["inner_folds"] for seed in design["training_seeds"]}
    candidates = {}
    for method, repeat, fold, inner, candidate, seed in keys:
        candidates.setdefault((method, repeat, fold, candidate), set()).add((inner, seed))
    if any(pairs != expected_pairs for pairs in candidates.values()):
        raise ValueError("candidate has a missing or unexpected inner fold or seed")
    for method in design["main_methods"]:
        for repeat in design["outer_repeats"]:
            for fold in design["outer_folds"]:
                count = sum(key[:3] == (method, repeat, fold) for key in candidates)
                if count != design["candidates_per_method"]:
                    raise ValueError("candidate count differs between paper scopes")
    return selected
