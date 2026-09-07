from unittest import TestCase, mock
from fedsift.runtime_environment import (
    SCHEMA,
    RuntimeEnvironmentError,
    RuntimeEnvironmentGuard,
    canonical_hash,
)


class RuntimeEnvironmentTests(TestCase):

    def setUp(self):
        self.snapshot = {
            "schema": SCHEMA,
            "torch_intraop_threads": 1,
            "affinity": [0, 1],
            "packages": {"torch": "2.5.1+cpu"},
        }
        self.guard = RuntimeEnvironmentGuard(
            self.snapshot, expected_sha256=canonical_hash(self.snapshot)
        )

    def test_changed_manifest_rejected_before_execution(self):
        with self.assertRaises(RuntimeEnvironmentError):
            RuntimeEnvironmentGuard(self.snapshot, expected_sha256="a" * 64)

    def test_each_runtime_dimension_checked_before_operation(self):
        for key, value in [
            ("torch_intraop_threads", 2),
            ("affinity", [0]),
            ("packages", {"torch": "other"}),
        ]:
            with (
                self.subTest(key=key),
                mock.patch(
                    "fedsift.runtime_environment.observe_environment",
                    return_value={**self.snapshot, key: value},
                ),
            ):
                operation = mock.Mock()
                with self.assertRaises(RuntimeEnvironmentError):
                    self.guard.measurement_backend(lambda op, ctx: op())(operation, {})
                operation.assert_not_called()

    def test_drift_during_operation_prevents_success_artifact(self):
        with mock.patch(
            "fedsift.runtime_environment.observe_environment",
            side_effect=[self.snapshot, {**self.snapshot, "affinity": [0]}],
        ):
            with self.assertRaises(RuntimeEnvironmentError):
                self.guard.measurement_backend(lambda op, ctx: op())(lambda: "receipt", {})

    def test_guard_is_outside_measurement_interval(self):
        events = []

        def observe():
            events.append("observe")
            return self.snapshot

        def measure(op, context):
            events.append("start_clock")
            value = op()
            events.append("stop_clock")
            return (value, {"elapsed": 1})

        with mock.patch("fedsift.runtime_environment.observe_environment", side_effect=observe):
            result = self.guard.measurement_backend(measure)(lambda: "receipt", {})
        self.assertEqual(events, ["observe", "start_clock", "stop_clock", "observe"])
        self.assertEqual(result, ("receipt", {"elapsed": 1}))
