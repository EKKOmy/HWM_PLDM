"""Standard-library tests: python -m unittest discover -s tests -p test_compare_planning_runs.py."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_planning_runs.py"
SPEC = importlib.util.spec_from_file_location("compare_planning_runs", SCRIPT)
compare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compare
SPEC.loader.exec_module(compare)


class ComparePlanningRunsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def summary(self, data, filename="summary.json"):
        path = self.directory / filename
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def run_from(self, data, label="run", filename="summary.json", prefix="auto"):
        path = self.summary(data, filename)
        return compare.load_run(f"{label}={path}", prefix)

    def test_delta_and_explicit_counts_and_elapsed(self):
        old = self.run_from(
            {"d4rl_medium_planning_success_rate": 0.675}, "old", "old.json"
        )
        new = self.run_from(
            {
                "l2_d4rl_medium_planning_success_rate": 0.8,
                "l2_d4rl_medium_planning_num_tasks": 15,
                "l2_d4rl_medium_planning_successes": 12,
                "l2_d4rl_medium_planning_elapsed_seconds": 2.125,
            },
            "new", "new.json",
        )
        report = compare.format_report([old, new])
        self.assertIn("+12.50", report)
        self.assertIn("12 / 15", report)
        self.assertIn("2.125", report)
        self.assertIn("hard", report)
        self.assertIn("NA", report)

    def test_missing_counts_are_not_inferred(self):
        run = self.run_from(
            {"d4rl_medium_planning_success_rate": 0.675,
             "d4rl_medium_planning_successes": 27}
        )
        report = compare.format_report([run])
        self.assertNotIn("Successes / tasks", report)
        self.assertNotIn("40", report)
        self.assertNotIn("Elapsed (s)", report)

    def test_task_count_without_success_count_does_not_infer_successes(self):
        run = self.run_from(
            {"d4rl_medium_planning_success_rate": 0.8,
             "d4rl_medium_planning_num_tasks": 15}
        )
        self.assertIn("NA / 15", compare.format_report([run]))

    def test_missing_baseline_difficulty_has_no_delta(self):
        old = self.run_from({"d4rl_medium_planning_success_rate": 1}, "old")
        new = self.run_from({"d4rl_hard_planning_success_rate": 0.5}, "new")
        hard_row = next(
            line for line in compare.format_report([old, new]).splitlines()
            if line.startswith("new") and "hard" in line
        )
        self.assertIn("50.00%", hard_row)
        self.assertIn("NA", hard_row)

    def test_auto_uses_one_prefix_without_mixing_difficulties(self):
        data = {
            "l2_d4rl_medium_planning_success_rate": 0.8,
            "d4rl_medium_planning_success_rate": 0.5,
            "d4rl_hard_planning_success_rate": 0.4,
        }
        run = self.run_from(data)
        self.assertEqual(run.prefix, "l2_d4rl")
        self.assertNotIn("hard", run.measurements)
        flat = self.run_from(data, prefix="d4rl")
        self.assertEqual(flat.measurements["medium"].success_rate, 0.5)
        self.assertEqual(flat.measurements["hard"].success_rate, 0.4)

    def test_missing_metrics_and_nonobject_json_fail(self):
        for data in ({}, {"loss": 0.1}, [], None):
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.run_from(data)

    def test_bad_json_and_missing_file_fail(self):
        path = self.directory / "broken.json"
        path.write_text("{invalid", encoding="utf-8")
        for filename in (path, self.directory / "missing.json"):
            with self.subTest(filename=filename):
                with self.assertRaisesRegex(ValueError, "Cannot read JSON"):
                    compare.load_run(f"bad={filename}")

    def test_rate_range_and_types(self):
        for value in (-0.1, 1.01, 67.5, float("nan"), float("inf"), True, "0.5", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.run_from({"d4rl_medium_planning_success_rate": value})
        for value in (0, 1, 0.675):
            with self.subTest(value=value):
                run = self.run_from({"d4rl_medium_planning_success_rate": value})
                self.assertEqual(run.measurements["medium"].success_rate, value)

    def test_invalid_optional_metrics_fail(self):
        invalid = [
            {"num_tasks": 0}, {"num_tasks": -1}, {"num_tasks": 1.5},
            {"num_tasks": True}, {"num_tasks": 3, "successes": 4},
            {"num_tasks": 3, "successes": -1},
            {"elapsed_seconds": -1}, {"elapsed_seconds": float("nan")},
        ]
        for fields in invalid:
            data = {"d4rl_medium_planning_success_rate": 0.5}
            data.update({"d4rl_medium_planning_" + k: v for k, v in fields.items()})
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.run_from(data)

    def test_historical_values_are_separate_and_explicitly_unpaired(self):
        run = self.run_from({"d4rl_medium_planning_success_rate": 0.5})
        report = compare.format_report([run], historical=True)
        measured, historical = report.split("历史参考(未配对)")
        self.assertNotIn("67.50%", measured)
        for value in ("67.50%", "45.00%", "80.00%", "62.50%"):
            self.assertIn(value, historical)
        self.assertIn("not re-measured", historical)
        self.assertNotIn("Delta", historical)

    def test_bad_run_spec_fails(self):
        for spec in ("summary.json", "=summary.json", "label="):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                compare.load_run(spec)

    def test_cli_accepts_repeated_runs_and_fails_cleanly(self):
        first = self.summary({"d4rl_medium_planning_success_rate": 0.5}, "one.json")
        second = self.summary({"l2_d4rl_medium_planning_success_rate": 0.8}, "two.json")
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--run", f"old={first}",
             "--run", f"new={second}", "--historical"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("+30.00", result.stdout)
        self.assertIn("历史参考(未配对)", result.stdout)
        empty = self.summary({}, "empty.json")
        failed = subprocess.run(
            [sys.executable, str(SCRIPT), "--run", f"empty={empty}"],
            capture_output=True, text=True,
        )
        self.assertEqual(failed.returncode, 2)
        self.assertIn("no planning success-rate metrics", failed.stderr)
        self.assertNotIn("Traceback", failed.stderr)


if __name__ == "__main__":
    unittest.main()
