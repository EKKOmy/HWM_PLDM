"""Test the actual reporting helpers without importing MuJoCo/probing modules.

Only the relevant AST definitions are compiled, unchanged, from their source.
This deliberately does not claim full evaluator/CUDA integration coverage.
"""

import ast
import contextlib
import dataclasses
from enum import Enum
import io
import json
import math
from pathlib import Path
import random
import tempfile
import types
from typing import NamedTuple, Optional
import unittest
import zlib

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def definition(path, class_name, methods=None):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    if methods is not None:
        node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    namespace = dict(
        torch=torch, np=np, Path=Path, Enum=Enum, dataclasses=dataclasses,
        math=math, json=json, random=random, zlib=zlib, NamedTuple=NamedTuple,
        Optional=Optional,
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / path), "exec"), namespace)
    return namespace[class_name]


Evaluator = definition(
    "pldm/evaluation/evaluator.py", "Evaluator",
    {"_reset_planning_rng", "_json_value", "_save_planning_details"},
)
Report = definition("pldm/planning/d4rl/enums.py", "MPCReport")


class PlanningRecordsTest(unittest.TestCase):
    def setUp(self):
        self.evaluator = Evaluator()
        self.evaluator.config = types.SimpleNamespace(planning_seed=246)
        py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        self.addCleanup(random.setstate, py_state)
        self.addCleanup(np.random.set_state, np_state)
        self.addCleanup(torch.set_rng_state, torch_state)

    def report(self):
        return Report(
            success_rate=0.5, success=torch.tensor([1, 0]),
            avg_steps_to_goal=1, median_steps_to_goal=1, terminations=[1, 4],
            one_turn_success_rate=1, two_turn_success_rate=0, three_turn_success_rate=-1,
            num_one_turns=1, num_two_turns=1, num_three_turns=0,
            num_turns=[1, 2], block_dists=[9, 10], ood_report={}, elapsed_seconds=2.5,
        )

    def test_explicit_seed_is_independent_of_prior_difficulty_and_random_consumption(self):
        self.assertEqual(self.evaluator._reset_planning_rng("hard"), 246 + 2018)
        expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
        self.evaluator._reset_planning_rng("medium")
        torch.rand(20)
        self.evaluator._reset_planning_rng("hard")
        self.assertEqual(expected, (random.random(), float(np.random.rand()), float(torch.rand(()))))

    def test_default_seed_does_not_change_rng(self):
        self.evaluator.config.planning_seed = None
        before = torch.get_rng_state()
        self.assertIsNone(self.evaluator._reset_planning_rng("hard"))
        torch.testing.assert_close(before, torch.get_rng_state())

    def test_report_exports_actual_counts_and_optional_time(self):
        report = self.report()
        log = report.build_log_dict("l2_d4rl_medium")
        self.assertEqual(log["l2_d4rl_medium_planning_num_tasks"], 2)
        self.assertEqual(log["l2_d4rl_medium_planning_successes"], 1)
        self.assertEqual(log["l2_d4rl_medium_planning_elapsed_seconds"], 2.5)
        old = report._replace(elapsed_seconds=None).build_log_dict("d4rl_medium")
        self.assertNotIn("d4rl_medium_planning_elapsed_seconds", old)

    def test_json_records_preserve_tasks_config_checkpoint_and_diagnostics(self):
        @dataclasses.dataclass
        class Config:
            planning_seed: int = 246

        self.evaluator.config = Config()
        self.evaluator.epoch = 0
        self.evaluator.load_checkpoint_path = "/tmp/checkpoint.ckpt"
        result = types.SimpleNamespace(
            locations=[torch.tensor([[0., 0.], [1., 1.]])],
            targets=torch.tensor([[2., 3.], [4., 5.]]), reward_history=[0, 1, 2, 3],
        )
        planner = types.SimpleNamespace(
            prefix="l2_d4rl_medium", config=Config(),
            trials={"map_layouts": [np.array([[1, 0]]), np.array([[0, 1]])]},
            feedback_records=[{"env_index": 0, "selected_candidate": 1}],
        )
        with tempfile.TemporaryDirectory() as directory:
            self.evaluator.output_path = directory
            with contextlib.redirect_stdout(io.StringIO()):
                self.evaluator._save_planning_details(planner, result, self.report(), 1255)
            data = json.loads((Path(directory) / "planning_details_l2_d4rl_medium.json").read_text())
        self.assertEqual(data["planning_seed"], 1255)
        self.assertEqual(data["checkpoint_path"], str(Path("/tmp/checkpoint.ckpt").resolve()))
        self.assertEqual(data["tasks"][0]["start"], [0., 0.])
        self.assertEqual(data["tasks"][1]["target"], [4., 5.])
        self.assertIsNone(data["tasks"][1]["first_success_step_index"])
        self.assertEqual(data["num_tasks"], 2)
        self.assertEqual(data["feedback_records"][0]["selected_candidate"], 1)
        self.assertIsNone(Evaluator._json_value(float("nan")))


if __name__ == "__main__":
    unittest.main()
