"""CPU tests for feedback selection and the real MPPI scoring interfaces.

Run with: python3 -m unittest discover -s tests -p test_subgoal_feedback.py -v
No checkpoints, simulator, training, or network access are needed.
"""

import importlib.util
from contextlib import nullcontext
import math
import sys
import types
import unittest
from unittest import mock

import torch


# The planner imports config dataclasses, but these tests never parse configs.
# Permit running in a torch-only environment without installing OmegaConf.
if importlib.util.find_spec("omegaconf") is None:
    _omega_stub = types.ModuleType("omegaconf")
    _omega_stub.OmegaConf = types.SimpleNamespace(
        register_new_resolver=lambda *args, **kwargs: None
    )
    _imports = mock.patch.dict(sys.modules, {"omegaconf": _omega_stub})
else:
    _imports = nullcontext()

with _imports:
    from pldm.planning.planners.mppi_planner import MPPIPlanner, RunningCost
    from pldm.planning.planners.planner import PlanningResult
    from pldm.planning.planners.enums import MPPIConfig, SubgoalFeedbackConfig
    from pldm.planning.planners.two_lvl_planner import TwoLvlPlanner
    from pldm.models.encoders.enums import BackboneOutput
    from pldm.planning.objectives_v2 import ReprTargetMPCObjective, ReprTargetMPCObjective2
    from pldm.planning.planners.subgoal_feedback import (
        plan_with_feedback,
        select_results,
        snapshot,
    )


def make_result(tag, batch=3, horizon=3, ensembles=2):
    """Unique values across every axis expose incorrect per-environment gathers."""
    fields = {}
    for index, name in enumerate(PlanningResult._fields):
        if name == "losses":
            fields[name] = [0]
            continue
        if name == "actions":
            shape = (batch, horizon, 2)
        elif name.startswith("ensemble_"):
            shape = (horizon + 1, ensembles, batch, 5)
        else:
            shape = (horizon + 1, batch, 5)
        fields[name] = (
            torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
            + tag * 1000 + index * 10000
        )
    return PlanningResult(**fields)


class FakeLowPlanner:
    def __init__(self, high_results, reach_costs, fail_on_call=None):
        self.high_results = high_results
        self.reach_costs = reach_costs
        batch = reach_costs.shape[1]
        self.ctrls = [
            types.SimpleNamespace(U=torch.full((2, 2), float(b + 1)), T=2, K=7)
            for b in range(batch)
        ]
        self.last_plan_size = 2
        self.objective = types.SimpleNamespace(target_enc=torch.full((batch, 5), -3.0))
        self.calls = []
        self.fail_on_call = fail_on_call

    def reset_targets(self, target, repr_input=True):
        self.objective.target_enc = target

    def plan(self, current_state, plan_size, repr_input=True, **kwargs):
        candidate = next(
            c for c, result in enumerate(self.high_results)
            if torch.equal(self.objective.target_enc, result.pred_obs[1])
        )
        initial = snapshot(self)
        noise = torch.randn(len(self.ctrls), plan_size, 2)
        for b, ctrl in enumerate(self.ctrls):
            ctrl.U = noise[b] + candidate + initial[0][b].mean() / 10
            ctrl.T = plan_size
        self.last_plan_size = plan_size
        result = make_result(candidate, batch=len(self.ctrls), horizon=plan_size)
        result = result._replace(actions=torch.stack([ctrl.U * 2 for ctrl in self.ctrls]))
        self.calls.append(
            {"candidate": candidate, "noise": noise, "initial": initial,
             "saved": snapshot(self), "result": result, "kwargs": kwargs}
        )
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("synthetic planning failure")
        return result

    def score_plan(self, result, terminal_only=False):
        assert terminal_only
        return self.reach_costs[self.calls[-1]["candidate"]].clone()


class FakeHighPlanner:
    candidate_actions = MPPIPlanner.candidate_actions

    def __init__(self, results, goal_costs):
        self.results = results
        self.goal_costs = goal_costs
        self.ctrls = []
        self.last_plan_size = results[0].actions.shape[1]
        self.objective = types.SimpleNamespace(
            target_enc=torch.zeros(goal_costs.shape[1], 5)
        )
        for b in range(goal_costs.shape[1]):
            self.ctrls.append(types.SimpleNamespace(
                U=results[0].actions[b].clone(), T=self.last_plan_size, K=len(results) - 1,
                perturbed_action=torch.stack([r.actions[b] for r in results[1:]])
                if len(results) > 1 else torch.empty(0, self.last_plan_size, 2),
                cost_total=torch.arange(len(results) - 1, dtype=torch.float32),
            ))
        self.rollout_calls = 0

    def rollout_actions(self, backbone, actions):
        self.rollout_calls += 1
        return next(r for r in self.results if torch.equal(r.actions, actions))

    def score_plan(self, result):
        index = next(i for i, r in enumerate(self.results) if result is r)
        return self.goal_costs[index].clone()


def fixture(goal_costs=None, reach_costs=None, count=3, duplicate=False):
    if goal_costs is None:
        goal_costs = torch.tensor([[3., 1., 5.], [1., 4., 3.], [4., 3., 1.]])[:count]
    batch = goal_costs.shape[1]
    high_results = [make_result(0 if duplicate else c, batch=batch) for c in range(count)]
    if reach_costs is None:
        reach_costs = torch.zeros(count, batch)
    low = FakeLowPlanner(high_results, reach_costs)
    high = FakeHighPlanner(high_results, goal_costs)
    backbone = types.SimpleNamespace(encodings=torch.zeros(batch, 5))
    return low, high, backbone


class FeedbackTest(unittest.TestCase):
    def setUp(self):
        state = torch.get_rng_state()
        self.addCleanup(torch.set_rng_state, state)

    def assert_result_equal(self, left, right):
        for name in PlanningResult._fields:
            actual, expected = getattr(left, name), getattr(right, name)
            if isinstance(expected, torch.Tensor):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            else:
                self.assertEqual(actual, expected)

    def assert_snapshot_equal(self, planner, saved):
        current = snapshot(planner)
        for actual, expected in zip(current[0], saved[0]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(current[1:3], saved[1:3])
        self.assertIs(current[3], saved[3])

    def test_selects_all_planning_result_fields_on_the_correct_batch_axis(self):
        results = [make_result(c) for c in range(3)]
        selected = torch.tensor([2, 0, 1])
        actual = select_results(results, selected)
        for name in PlanningResult._fields:
            if name == "losses":
                self.assertEqual(actual.losses, [0])
                continue
            value = getattr(actual, name)
            self.assertEqual(value.shape, getattr(results[0], name).shape)
            for b, c in enumerate(selected.tolist()):
                expected = getattr(results[c], name)
                if name == "actions":
                    torch.testing.assert_close(value[b], expected[b])
                elif name.startswith("ensemble_"):
                    torch.testing.assert_close(value[:, :, b], expected[:, :, b])
                else:
                    torch.testing.assert_close(value[:, b], expected[:, b])

    def test_optional_result_fields_remain_none(self):
        results = [make_result(c)._replace(
            pred_proprio=None, pred_location=None, raw_locations=None, locations=None,
            ensemble_proprio_component=None, ensemble_raw_locations=None,
        ) for c in range(2)]
        result = select_results(results, torch.tensor([0, 1, 0]))
        self.assertIsNone(result.locations)
        self.assertIsNone(result.ensemble_proprio_component)
        self.assertIsNone(result.raw_locations)

    def test_per_environment_winner_restores_selected_warm_starts_and_targets(self):
        low, high, backbone = fixture()
        low_before = snapshot(low)
        config = types.SimpleNamespace(num_candidates=3, weight=1.0)
        actual_low, actual_high, records = plan_with_feedback(
            low, high, backbone, high.results[0], 4, config, curr_proprio_pos=None
        )
        selected = torch.tensor([1, 0, 2])
        self.assert_result_equal(
            actual_low, select_results([c["result"] for c in low.calls], selected)
        )
        self.assert_result_equal(actual_high, select_results(high.results, selected))
        for b, c in enumerate(selected.tolist()):
            torch.testing.assert_close(low.ctrls[b].U, low.calls[c]["saved"][0][b])
            torch.testing.assert_close(high.ctrls[b].U, high.results[c].actions[b])
            self.assertEqual(low.ctrls[b].T, 4)
        self.assertEqual(low.last_plan_size, 4)
        torch.testing.assert_close(low.objective.target_enc, actual_high.pred_obs[1])
        self.assertEqual([r["selected_candidate"] for r in records], [1, 0, 2])
        self.assertEqual(len(low.calls), 3)  # No extra optimization for the winner.
        for call in low.calls:
            for actual, expected in zip(call["initial"][0], low_before[0]):
                torch.testing.assert_close(actual, expected)

    def test_one_candidate_is_identical_to_the_original_low_level_solve(self):
        direct_low, direct_high, direct_backbone = fixture(count=1)
        torch.manual_seed(409)
        direct_low.reset_targets(direct_high.results[0].pred_obs[1])
        expected = direct_low.plan(direct_backbone, plan_size=3)
        expected_rng = torch.get_rng_state()
        low, high, backbone = fixture(count=1)
        torch.manual_seed(409)
        actual, actual_high, records = plan_with_feedback(
            low, high, backbone, high.results[0], 3,
            types.SimpleNamespace(num_candidates=1, weight=1.0),
        )
        self.assert_result_equal(actual, expected)
        self.assert_result_equal(actual_high, high.results[0])
        self.assertEqual(high.rollout_calls, 0)
        self.assertEqual([r["selected_candidate"] for r in records], [0, 0, 0])
        torch.testing.assert_close(torch.get_rng_state(), expected_rng)
        for actual_ctrl, expected_ctrl in zip(low.ctrls, direct_low.ctrls):
            torch.testing.assert_close(actual_ctrl.U, expected_ctrl.U)

    def test_duplicate_candidates_share_noise_and_advance_rng_only_once(self):
        low, high, backbone = fixture(duplicate=True)
        torch.manual_seed(42)
        expected_noise = torch.randn(3, 4, 2)
        expected_rng = torch.get_rng_state()
        torch.manual_seed(42)
        result, _, records = plan_with_feedback(
            low, high, backbone, high.results[0], 4,
            types.SimpleNamespace(num_candidates=3, weight=2.0),
        )
        for call in low.calls:
            torch.testing.assert_close(call["noise"], expected_noise, rtol=0, atol=0)
            self.assert_result_equal(call["result"], result)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng)
        self.assertEqual([r["selected_candidate"] for r in records], [0, 0, 0])

    def test_zero_weight_selects_only_by_goal_cost(self):
        goals = torch.tensor([[1., 4., 3.], [4., 1., 2.], [2., 3., 1.]])
        reaches = torch.tensor([[100., 0., 0.], [0., 100., 0.], [0., 0., 100.]])
        low, high, backbone = fixture(goals, reaches)
        _, _, records = plan_with_feedback(
            low, high, backbone, high.results[0], 3,
            types.SimpleNamespace(num_candidates=3, weight=0.0),
        )
        self.assertEqual([r["selected_candidate"] for r in records], [0, 1, 2])
        for record in records:
            self.assertEqual(record["scores"], record["goal_costs"])

    def test_reachability_cost_can_change_a_goal_only_winner(self):
        goals = torch.tensor([[0., 0., 0.], [1., 1., 1.]])
        reaches = torch.tensor([[10., 10., 10.], [0., 0., 0.]])
        low, high, backbone = fixture(goals, reaches, count=2)
        _, _, records = plan_with_feedback(
            low, high, backbone, high.results[0], 3,
            types.SimpleNamespace(num_candidates=2, weight=1.0),
        )
        self.assertEqual([r["selected_candidate"] for r in records], [1, 1, 1])

    def test_no_finite_candidate_restores_both_planners(self):
        goals = torch.tensor([[1., float("nan"), 3.], [2., float("inf"), 1.]])
        low, high, backbone = fixture(goals, count=2)
        before_low, before_high = snapshot(low), snapshot(high)
        with self.assertRaisesRegex(RuntimeError, "No finite subgoal candidate"):
            plan_with_feedback(
                low, high, backbone, high.results[0], 4,
                types.SimpleNamespace(num_candidates=2, weight=1.0),
            )
        self.assert_snapshot_equal(low, before_low)
        self.assert_snapshot_equal(high, before_high)

    def test_planning_exception_restores_targets_actions_and_horizons(self):
        low, high, backbone = fixture()
        low.fail_on_call = 2
        before_low, before_high = snapshot(low), snapshot(high)
        with self.assertRaisesRegex(RuntimeError, "synthetic planning failure"):
            plan_with_feedback(
                low, high, backbone, high.results[0], 4,
                types.SimpleNamespace(num_candidates=3, weight=1.0),
            )
        self.assert_snapshot_equal(low, before_low)
        self.assert_snapshot_equal(high, before_high)


class MPPIInterfaceTest(unittest.TestCase):
    def scoring_planner(self, target, sum_all_diffs=False, sum_last_n=2):
        planner = MPPIPlanner.__new__(MPPIPlanner)
        planner.cost_entity = "obs_component"
        planner.model = types.SimpleNamespace(
            predictor=types.SimpleNamespace(ensemble_size=1)
        )
        objective = types.SimpleNamespace(
            target_enc=target, sum_all_diffs=sum_all_diffs, sum_last_n=sum_last_n
        )
        planner.mppi_costs = [RunningCost(objective, idx=b) for b in range(len(target))]
        return planner

    def test_real_score_plan_returns_one_cost_per_env_and_excludes_initial_state(self):
        target = torch.tensor([[0., 0.], [10., 10.]])
        planner = self.scoring_planner(target)
        states = torch.tensor([
            [[100., 100.], [-100., -100.]],
            [[1., 1.], [11., 11.]],
            [[2., 2.], [12., 12.]],
            [[3., 3.], [13., 13.]],
        ])
        result = make_result(0, batch=2)._replace(pred_obs=states)
        torch.testing.assert_close(planner.score_plan(result), torch.tensor([6.5, 6.5]))
        torch.testing.assert_close(
            planner.score_plan(result, terminal_only=True), torch.tensor([9., 9.])
        )
        for cost in planner.mppi_costs:
            cost.sum_all_diffs = True
        torch.testing.assert_close(
            planner.score_plan(result), torch.tensor([14. / 3, 14. / 3])
        )

    def test_real_score_plan_supports_conv_latents_and_cost_dimension_slice(self):
        target = torch.zeros(2, 8)
        planner = self.scoring_planner(target, sum_all_diffs=True)
        states = torch.arange(4 * 2 * 8, dtype=torch.float32).reshape(4, 2, 2, 2, 2)
        for cost in planner.mppi_costs:
            cost.cost_dim_range = [2, 5]
        result = make_result(0, batch=2)._replace(pred_obs=states)
        expected = states[1:].reshape(3, 2, 8)[:, :, 2:5].square().mean((0, 2))
        torch.testing.assert_close(planner.score_plan(result), expected)

    def test_real_score_plan_rejects_empty_horizon_or_unsupported_objective(self):
        planner = self.scoring_planner(torch.zeros(2, 5))
        empty = make_result(0, batch=2, horizon=0)
        with self.assertRaisesRegex(ValueError, "at least one scored step"):
            planner.score_plan(empty)
        result = make_result(0, batch=2)
        planner.model.predictor.ensemble_size = 2
        with self.assertRaisesRegex(ValueError, "ensemble_size=1"):
            planner.score_plan(result)
        planner.model.predictor.ensemble_size = 1
        planner.cost_entity = "proprio_component"
        with self.assertRaisesRegex(ValueError, "obs_component"):
            planner.score_plan(result)

    def test_real_candidates_rank_per_env_preserve_shape_and_do_not_alias(self):
        planner = MPPIPlanner.__new__(MPPIPlanner)
        planner.ctrls = []
        for b, costs in enumerate(([9., 1., 4.], [0., 7., 3.])):
            planner.ctrls.append(types.SimpleNamespace(
                K=3, U=torch.full((4, 2), -float(b + 1)),
                perturbed_action=torch.arange(24.).reshape(3, 4, 2) + b * 100,
                cost_total=torch.tensor(costs),
            ))
        candidates = planner.candidate_actions(3, batch_size=2)
        self.assertEqual(candidates.shape, (3, 2, 4, 2))
        torch.testing.assert_close(candidates[0], torch.stack([c.U for c in planner.ctrls]))
        torch.testing.assert_close(candidates[1, 0], planner.ctrls[0].perturbed_action[1])
        torch.testing.assert_close(candidates[1, 1], planner.ctrls[1].perturbed_action[0])
        torch.testing.assert_close(candidates[2, 0], planner.ctrls[0].perturbed_action[2])
        candidates[0, 0].fill_(99)
        self.assertTrue(torch.equal(planner.ctrls[0].U, torch.full((4, 2), -1.)))

    def test_real_candidates_fall_back_to_nominal_for_nonfinite_samples(self):
        planner = MPPIPlanner.__new__(MPPIPlanner)
        planner.ctrls = [types.SimpleNamespace(
            K=3, U=torch.ones(2, 2), cost_total=torch.tensor([0., float("nan"), 2.]),
            perturbed_action=torch.tensor([
                [[float("nan"), 0.], [0., 0.]],
                [[2., 2.], [2., 2.]],
                [[3., 3.], [3., 3.]],
            ]),
        )]
        candidates = planner.candidate_actions(4, 1)
        self.assertEqual(candidates.shape, (4, 1, 2, 2))
        torch.testing.assert_close(candidates[1, 0], torch.full((2, 2), 3.))
        torch.testing.assert_close(candidates[2:, 0], torch.ones(2, 2, 2))
        with self.assertRaisesRegex(ValueError, "samples \\+ 1"):
            planner.candidate_actions(5, 1)


class ToyBackbone(torch.nn.Module):
    output_proprio_dim = 0
    output_obs_dim = 2
    using_proprio = False

    def forward(self, obs, **kwargs):
        return BackboneOutput(obs)


class ToyPredictor(torch.nn.Module):
    ensemble_size = 1
    action_dim = 2

    def _prepare_ensemble_input(self, value):
        return value.unsqueeze(0) if value is not None else None

    def forward_multiple(self, state, actions, T, latents=None, **kwargs):
        actions = actions if actions is not None else latents
        states = torch.cat([state, state[0:1] + actions[:T].cumsum(0)])
        return types.SimpleNamespace(
            predictions=states, obs_component=states,
            proprio_component=None, location_component=None, raw_locations=None,
            ensemble_predictions=states.unsqueeze(1),
            ensemble_obs_component=states.unsqueeze(1),
            ensemble_proprio_component=None, ensemble_location_component=None,
            ensemble_raw_locations=None,
        )


class ToyModel(torch.nn.Module):
    spatial_repr_dim = 2

    def __init__(self, l2=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.backbone = ToyBackbone()
        self.predictor = ToyPredictor()
        self.config = types.SimpleNamespace(
            action_dim=0 if l2 else 2, predictor=types.SimpleNamespace(z_dim=2 if l2 else 0)
        )


class ToyIntegrationTest(unittest.TestCase):
    def make_planner(self, enabled, candidates=1):
        low_model, high_model = ToyModel(), ToyModel(l2=True)
        low_model.eval()
        high_model.eval()
        normalizer = types.SimpleNamespace(unnormalize_action=lambda x: x * 2)
        low = MPPIPlanner(
            MPPIConfig(num_samples=8, lambda_=0.5), low_model, normalizer,
            ReprTargetMPCObjective(model=low_model, sum_all_diffs=True,
                                  pred_encoder=high_model.backbone), n_envs=2,
        )
        high = MPPIPlanner(
            MPPIConfig(num_samples=8, lambda_=0.5), high_model, normalizer,
            ReprTargetMPCObjective2(model=high_model, sum_all_diffs=True),
            n_envs=2, l2=True,
        )
        planner = TwoLvlPlanner(low, high, 2, SubgoalFeedbackConfig(enabled, candidates, 1.0))
        planner.reset_targets(torch.tensor([[2., 1.], [1., -2.]]))
        return planner

    def run_sequence(self, enabled, candidates=1):
        torch.manual_seed(781)
        planner = self.make_planner(enabled, candidates)
        results = []
        for horizon in (4, 3):
            result = planner.plan(torch.zeros(2, 2), plan_size=horizon)
            results.append(result)
            self.assertEqual(result.level1.actions.shape, (2, 2, 2))
            self.assertEqual(result.level2.pred_obs.shape, (horizon + 1, 2, 2))
            self.assertEqual(planner.l2_planner.last_plan_size, horizon)
            self.assertEqual(planner.l2_planner.ctrls[0].T, horizon)
            # Returned physical actions differ from the normalized warm starts.
            torch.testing.assert_close(
                result.level1.actions,
                2 * torch.stack([ctrl.U for ctrl in planner.l1_planner.ctrls]),
            )
        return results, planner, torch.get_rng_state()

    def test_real_mppi_two_level_single_candidate_matches_disabled_on_shrinking_horizon(self):
        baseline, _, baseline_rng = self.run_sequence(False)
        feedback, planner, feedback_rng = self.run_sequence(True)
        for left, right in zip(baseline, feedback):
            for lhs, rhs in zip(left, right):
                for name in PlanningResult._fields:
                    expected, actual = getattr(lhs, name), getattr(rhs, name)
                    if isinstance(expected, torch.Tensor):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    else:
                        self.assertEqual(actual, expected)
        torch.testing.assert_close(baseline_rng, feedback_rng)
        self.assertEqual(len(planner.feedback_records), 4)

    def test_three_candidates_real_mppi_keep_rng_consumption_and_valid_plans(self):
        _, _, baseline_rng = self.run_sequence(False)
        results, planner, feedback_rng = self.run_sequence(True, candidates=3)
        torch.testing.assert_close(baseline_rng, feedback_rng)
        for result in results:
            self.assertTrue(torch.isfinite(result.level1.actions).all())
            self.assertTrue(torch.isfinite(result.level2.pred_obs).all())
        self.assertEqual(len(planner.feedback_records), 4)
        self.assertTrue(all(len(r["scores"]) == 3 for r in planner.feedback_records))


if __name__ == "__main__":
    unittest.main()
