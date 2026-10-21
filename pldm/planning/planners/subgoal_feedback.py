"""Inference-only candidate reranking using low-level predicted reach error.

This is a model-based consistency check, NOT a guarantee of real executability.
No simulator state, test outcomes, extra training data or learned critic is used.
"""

import torch


def select_results(results, selected):
    """Select a whole PlanningResult per environment, preserving tensor axes."""
    values = {}
    for name in results[0]._fields:
        value = getattr(results[0], name)
        if value is None or name == "losses":
            values[name] = value
            continue
        batch_axis = 0 if name == "actions" else (2 if name.startswith("ensemble_") else 1)
        # Avoid stacking all candidates' long rollouts into another large tensor.
        values[name] = torch.cat(
            [getattr(results[c], name).narrow(batch_axis, b, 1)
             for b, c in enumerate(selected.tolist())],
            dim=batch_axis,
        )
    return type(results[0])(**values)


def snapshot(planner):
    return (
        [ctrl.U.detach().clone() for ctrl in planner.ctrls],
        [ctrl.T for ctrl in planner.ctrls],
        planner.last_plan_size,
        planner.objective.target_enc,
    )


def restore(planner, saved):
    actions, horizons, plan_size, target = saved
    for ctrl, action, horizon in zip(planner.ctrls, actions, horizons):
        ctrl.U = action.clone()
        ctrl.T = horizon
    planner.last_plan_size = plan_size
    planner.objective.target_enc = target


def choose_candidates(goals, reaches, scores, valid, config):
    """Choose independently per environment; candidate zero is the nominal plan.

    Tolerance requires a meaningful score gain over zero. Conservative selection
    additionally requires a high nominal reach error, a bounded goal-cost loss,
    and a meaningful reach-error reduction. These are model-score guards, not
    guarantees of better control. Invalid nominal plans fall back to finite
    candidates rather than being protected by the guards.
    """
    mode = getattr(config, "selection_mode", "legacy")
    if mode not in ("legacy", "tolerance", "conservative"):
        raise ValueError(f"Unknown feedback selection mode: {mode}")
    valid = valid & torch.isfinite(goals) & torch.isfinite(reaches) & torch.isfinite(scores)
    if not valid.any(dim=0).all():
        raise RuntimeError("No finite subgoal candidate for at least one environment")
    raw_best = scores.masked_fill(~valid, float("inf")).argmin(dim=0)
    nominal_valid = valid[0]
    eligible = valid.clone()
    score_pass = torch.ones_like(valid)
    goal_pass = torch.ones_like(valid)
    reach_pass = torch.ones_like(valid)
    triggered = torch.ones_like(nominal_valid)
    score_tolerance = torch.zeros_like(scores[0])

    if mode != "legacy":
        score_tolerance = config.score_atol + config.score_rtol * scores[0].abs()
        score_pass = scores[0].unsqueeze(0) - scores > score_tolerance.unsqueeze(0)
        eligible &= score_pass
        if mode == "conservative":
            triggered = reaches[0] > config.reach_trigger
            goal_limit = config.goal_degradation_atol + config.goal_degradation_rtol * goals[0].abs()
            reach_gain = config.reach_improvement_atol + config.reach_improvement_rtol * reaches[0].abs()
            goal_pass = goals - goals[0].unsqueeze(0) <= goal_limit.unsqueeze(0)
            reach_pass = reaches[0].unsqueeze(0) - reaches > reach_gain.unsqueeze(0)
            eligible &= triggered.unsqueeze(0) & goal_pass & reach_pass
        eligible[0] = nominal_valid
        # Avoid NaN thresholds stranding an environment whose original plan failed.
        eligible = torch.where(nominal_valid.unsqueeze(0), eligible, valid)

    selected = scores.masked_fill(~eligible, float("inf")).argmin(dim=0)
    return selected, {
        "raw_best": raw_best,
        "nominal_valid": nominal_valid,
        "eligible": eligible,
        "score_pass": score_pass,
        "goal_pass": goal_pass,
        "reach_pass": reach_pass,
        "triggered": triggered,
        "score_tolerance": score_tolerance,
    }


def candidate_differences(l1_results, l2_results, actions):
    """Diagnostic distances, never used for ranking; no simulator input needed."""
    def rms(delta):
        return delta.flatten(1).square().mean(1).sqrt()

    return {
        "subgoal_rms_from_nominal": torch.stack([
            rms(result.pred_obs[1] - l2_results[0].pred_obs[1])
            for result in l2_results
        ]),
        "l2_action_rms_from_nominal": torch.stack([
            rms(action - actions[0]) for action in actions
        ]),
        "l1_action_rms_from_nominal": torch.stack([
            rms(result.actions - l1_results[0].actions) for result in l1_results
        ]),
        "l1_first_action_l2_from_nominal": torch.stack([
            (result.actions[:, 0] - l1_results[0].actions[:, 0]).norm(dim=-1)
            for result in l1_results
        ]),
    }


@torch.no_grad()
def plan_with_feedback(l1, l2, backbone, nominal, step_skip, config, **plan_kwargs):
    """Keep the original high-level search, inspect C low-level plans.

    Every candidate gets the SAME L1 sample budget, initial warm start and random
    noise. Additional checks do not consume the global RNG stream. At exit only
    the selected candidates become warm starts; the selected L1 plan is executed
    directly (no unfair additional solve for the winner).
    """
    device = backbone.encodings.device
    batch_size = backbone.encodings.shape[0]
    candidates = l2.candidate_actions(config.num_candidates, batch_size)
    l1_before = snapshot(l1)
    l2_before = snapshot(l2)
    cpu_rng = torch.get_rng_state()
    cuda_devices = [device.index] if device.type == "cuda" else []
    cuda_rng = torch.cuda.get_rng_state(device) if cuda_devices else None
    l1_results, l2_results, l1_starts = [], [], []
    goal_costs, reach_costs = [], []

    def check(candidate):
        restore(l1, l1_before)
        l1.reset_targets(candidate.pred_obs[1].detach(), repr_input=True)
        result = l1.plan(
            current_state=backbone, plan_size=step_skip, repr_input=True, **plan_kwargs
        )
        return result, snapshot(l1), l1.score_plan(result, terminal_only=True)

    try:
        for c in range(config.num_candidates):
            high = nominal if c == 0 else l2.rollout_actions(backbone, candidates[c])
            if c == 0:
                low, warm_start, reach_cost = check(high)
            else:
                with torch.random.fork_rng(devices=cuda_devices):
                    torch.set_rng_state(cpu_rng)
                    if cuda_rng is not None:
                        torch.cuda.set_rng_state(cuda_rng, device)
                    low, warm_start, reach_cost = check(high)
            l1_results.append(low)
            l2_results.append(high)
            l1_starts.append(warm_start)
            goal_costs.append(l2.score_plan(high))
            reach_costs.append(reach_cost)

        goals = torch.stack(goal_costs)
        reaches = torch.stack(reach_costs)
        scores = goals + config.weight * reaches
        valid = torch.isfinite(goals) & torch.isfinite(reaches) & torch.isfinite(scores)
        for c, result in enumerate(l1_results):
            valid[c] &= torch.isfinite(result.actions).flatten(1).all(1)
        selected, selection_info = choose_candidates(goals, reaches, scores, valid, config)
        low = select_results(l1_results, selected)
        high = select_results(l2_results, selected)

        for b, c in enumerate(selected.tolist()):
            l1.ctrls[b].U = l1_starts[c][0][b].clone()
            l1.ctrls[b].T = step_skip
            l2.ctrls[b].U = candidates[c, b].clone()
        l1.last_plan_size = step_skip
        l1.reset_targets(high.pred_obs[1].detach(), repr_input=True)
    except Exception:
        restore(l1, l1_before)
        restore(l2, l2_before)
        raise

    goals, reaches, scores = [x.detach().cpu() for x in (goals, reaches, scores)]
    selection_info = {key: value.detach().cpu() for key, value in selection_info.items()}
    differences = {
        key: value.detach().cpu()
        for key, value in candidate_differences(l1_results, l2_results, candidates).items()
    }
    mode = getattr(config, "selection_mode", "legacy")

    def finite_list(values):
        return [float(x) if torch.isfinite(x) else None for x in values]

    records = [
        {
            "record_version": 2,
            "env_index": b,
            "selected_candidate": c,
            "raw_best_candidate": int(selection_info["raw_best"][b]),
            "selection_mode": mode,
            "selection_reason": (
                "invalid_nominal_fallback" if not selection_info["nominal_valid"][b]
                else "legacy_minimum" if mode == "legacy"
                else "accepted_alternative" if c != 0
                else "nominal_retained"
            ),
            "reach_gate_triggered": bool(selection_info["triggered"][b]),
            "score_tolerance": finite_list(selection_info["score_tolerance"][b:b + 1])[0],
            "eligible_candidates": selection_info["eligible"][:, b].tolist(),
            "score_margin_passed": selection_info["score_pass"][:, b].tolist(),
            "goal_guard_passed": selection_info["goal_pass"][:, b].tolist(),
            "reach_guard_passed": selection_info["reach_pass"][:, b].tolist(),
            "goal_costs": finite_list(goals[:, b]),
            "reach_costs": finite_list(reaches[:, b]),
            "scores": finite_list(scores[:, b]),
            "feedback_weight": config.weight,
            "l1_samples_per_candidate": l1.ctrls[b].K,
            "l2_samples": l2.ctrls[b].K,
            "l1_horizon": step_skip,
            "l2_horizon": l2.ctrls[b].T,
            **{key: finite_list(value[:, b]) for key, value in differences.items()},
        }
        for b, c in enumerate(selected.tolist())
    ]
    return low, high, records
