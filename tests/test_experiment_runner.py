"""The attack stage must see a closed, successful main experiment."""
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools import experiment


class ExperimentRunnerTests(unittest.TestCase):
    def test_replay_closes_both_main_shards_before_any_poisoning_worker(self):
        completed = set()
        closed = False
        attacked = []

        def execute(directory, script, *arguments):
            nonlocal closed
            if script == "run_experiment.py":
                completed.add(arguments[-1])
            elif script == "summarize_results.py":
                self.assertEqual(completed, {"0", "1"})
                closed = True
            elif script == "evaluate_poisoning.py":
                self.assertTrue(closed, "attack requires the main experiment closure")
                attacked.append(arguments[-1])

        with patch.object(experiment, "runtime_check"), patch.object(experiment, "source_check"), \
             patch.object(experiment, "new_run", return_value=Path("unused")), \
             patch.object(experiment, "execute", side_effect=execute), \
             patch.object(experiment, "summarize"), patch.object(experiment, "compare_tables", return_value=[]), \
             patch.object(experiment, "report"):
            experiment.rebuild(SimpleNamespace(command="replay"))
        self.assertEqual(attacked, ["pima", "retinopathy"])

    def test_main_training_failure_prevents_attack_and_success_report(self):
        with patch.object(experiment, "runtime_check"), patch.object(experiment, "source_check"), \
             patch.object(experiment, "new_run", return_value=Path("unused")), \
             patch.object(experiment, "execute", side_effect=RuntimeError("training failed")) as execute, \
             patch.object(experiment, "summarize") as summarize, \
             patch.object(experiment, "report") as report:
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                experiment.rebuild(SimpleNamespace(command="replay"))
        self.assertEqual(execute.call_count, 1)
        summarize.assert_not_called()
        report.assert_not_called()


if __name__ == "__main__":
    unittest.main()
