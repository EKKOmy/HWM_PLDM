"""Launch-command tests: no model imports, GPU jobs, or checkpoint loading."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "feedback_launcher", ROOT / "scripts/eval_subgoal_feedback.py"
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "epoch=5_sample_step=10789632.ckpt"
        self.checkpoint.touch()  # A filename fixture; never deserialized.
        self.output = self.root / "outputs"

    def invoke(self, mode="feedback", *extra):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return launcher.main([
                mode, "--checkpoint", str(self.checkpoint),
                "--output-root", str(self.output), *extra,
            ])

    def test_checkpoint_directory_selects_numeric_epoch_and_sample_step(self):
        (self.root / "epoch=10_sample_step=10.ckpt").touch()
        latest = self.root / "epoch=10_sample_step=20.ckpt"
        latest.touch()
        (self.root / "unrelated.ckpt").touch()
        self.assertEqual(launcher.resolve_checkpoint(self.root), latest.resolve())
        self.assertEqual(launcher.resolve_checkpoint(self.checkpoint), self.checkpoint.resolve())

    def test_dry_run_never_launches_or_creates_output(self):
        with mock.patch.object(launcher.subprocess, "call") as call:
            self.assertEqual(self.invoke("feedback", "--dry-run"), 0)
            call.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_modes_keep_the_same_checkpoint_seed_and_protect_complete_weights(self):
        for mode in ("baseline", "feedback", "candidates", "candidates_safe", "feedback_safe"):
            with mock.patch.object(launcher.subprocess, "call", return_value=0) as call:
                self.assertEqual(self.invoke(mode), 0)
                command = call.call_args.args[0]
                values = dict(x.split("=", 1) for x in command[command.index("--values") + 1:])
                self.assertEqual(values["eval_only"], "true")
                self.assertEqual(values["load_l1_only"], "false")
                self.assertEqual(values["require_full_checkpoint"], "true")
                self.assertEqual(values["eval_cfg.planning_seed"], "246")
                self.assertEqual(json.loads(values["load_checkpoint_path"]), str(self.checkpoint.resolve()))
                prefix = "eval_cfg.h_d4rl_planning.subgoal_feedback."
                self.assertEqual(values[prefix + "enabled"], str(mode != "baseline").lower())
                self.assertEqual(float(values[prefix + "weight"]), 0 if mode.startswith("candidates") else 1)
                self.assertEqual(values[prefix + "selection_mode"], {
                    "candidates_safe": "tolerance", "feedback_safe": "conservative",
                }.get(mode, "legacy"))
                self.assertEqual(values["output_dir"], f"{mode}_seed246_p246")
                self.assertEqual(call.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2")

    def test_smoke_has_separate_output_and_tiny_evaluation(self):
        with mock.patch.object(launcher.subprocess, "call", return_value=0) as call:
            self.invoke("feedback", "--smoke")
            command = call.call_args.args[0]
            self.assertIn("output_dir=feedback_seed246_p246_smoke", command)
            self.assertIn("eval_cfg.h_d4rl_planning.medium.n_envs=2", command)
            self.assertIn("eval_cfg.h_d4rl_planning.medium.n_steps=8", command)
            self.assertIn("eval_cfg.h_d4rl_planning.final_trans_steps=0", command)

    def test_existing_output_is_rejected_without_launching(self):
        (self.output / "feedback_seed246_p246").mkdir(parents=True)
        with mock.patch.object(launcher.subprocess, "call") as call:
            with self.assertRaises(SystemExit) as error:
                self.invoke()
            self.assertEqual(error.exception.code, 2)
            call.assert_not_called()

    def test_invalid_arguments_are_rejected(self):
        for args in (
            ("--weight", "nan"), ("--weight", "-1"),
            ("--num-candidates", "0"), ("--workers", "-1"),
            ("--seed", "-1"), ("--gpu", "2,3"),
            ("--run-name", "../oops"), ("--run-name", "test"),
            ("--score-atol", "-1"), ("--score-rtol", "nan"),
            ("--reach-trigger", "inf"), ("--reach-improvement-rtol", "1"),
            ("--goal-degradation-rtol", "-1"),
        ):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                self.invoke("feedback", *args)

    def test_safe_threshold_overrides_are_forwarded(self):
        with mock.patch.object(launcher.subprocess, "call", return_value=0) as call:
            self.invoke("feedback_safe", "--reach-trigger", "0.25", "--score-atol", "0.0002")
            command = call.call_args.args[0]
            prefix = "eval_cfg.h_d4rl_planning.subgoal_feedback."
            self.assertIn(prefix + "reach_trigger=0.25", command)
            self.assertIn(prefix + "score_atol=0.0002", command)
            self.assertIn(prefix + "goal_degradation_rtol=0.0", command)

    def test_missing_checkpoint_and_empty_directory_fail(self):
        with self.assertRaises(ValueError):
            launcher.resolve_checkpoint(self.root / "missing.ckpt")
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(ValueError):
            launcher.resolve_checkpoint(empty)

    def test_child_failure_is_propagated(self):
        with mock.patch.object(launcher.subprocess, "call", return_value=17):
            self.assertEqual(self.invoke(), 17)


if __name__ == "__main__":
    unittest.main()
