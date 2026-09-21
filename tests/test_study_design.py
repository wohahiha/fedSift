"""The active search must cover exactly the paper matrix."""
import unittest
from fedsift.study_design import load_study_design, paper_training_units, validate_candidate_pairing


class StudyDesignTests(unittest.TestCase):
    def test_declared_candidate_pairs_share_optimizer_coordinates(self):
        from fedsift.candidate_space import build_candidate_space
        from fedsift.artifact_contract import identity
        for study_id in (identity("pima_hpo_24c_3seed_50r"), identity("debrecen_hpo_24c_3seed_50r")):
            with self.subTest(study_id=study_id):
                validate_candidate_pairing(build_candidate_space(study_id, candidate_count=24))

    def test_scope_excludes_other_methods_and_repeats(self):
        design = load_study_design()
        units = [{"method": method, "outer_repeat": repeat, "outer_fold": fold,
                  "inner_fold": inner, "candidate_id": f"candidate_{candidate:04}", "hpo_seed": seed}
                 for method in [*design["main_methods"], "other_method"]
                 for repeat in range(3) for fold in range(5) for inner in range(3)
                 for candidate in range(24) for seed in design["training_seeds"]]
        selected = paper_training_units({"units": units}, design)
        self.assertEqual(len(selected), 4320)
        self.assertEqual({u["outer_repeat"] for u in selected}, {0})
        self.assertEqual({u["method"] for u in selected}, set(design["main_methods"]))
        with self.assertRaises(ValueError):
            paper_training_units({"units": selected[:-1]}, design)
        corrupted = [dict(u) for u in selected]
        corrupted[0]["hpo_seed"] = 999
        with self.assertRaises(ValueError):
            paper_training_units({"units": corrupted}, design)


if __name__ == "__main__":
    unittest.main()
