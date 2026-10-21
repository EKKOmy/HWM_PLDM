#!/usr/bin/env python3
"""Launch a frozen-world-model evaluation; the existing evaluator may fit probes."""

import argparse
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]


def resolve_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        candidates = []
        for item in path.glob("*.ckpt"):
            match = re.search(r"epoch=(\d+)_sample_step=(\d+)\.ckpt$", item.name)
            if match:
                candidates.append((tuple(map(int, match.groups())), item))
        if not candidates:
            raise ValueError(f"No epoch=N_sample_step=M.ckpt found in {path}")
        path = max(candidates, key=lambda entry: entry[0])[1]
    if not path.is_file():
        raise ValueError(f"Checkpoint does not exist: {path}")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("baseline", "feedback", "candidates", "candidates_safe", "feedback_safe")
    )
    parser.add_argument("--checkpoint", required=True, help="Your trained HWM file or run directory")
    parser.add_argument("--gpu", default="2", help="Physical GPU ID; one GPU per evaluation")
    parser.add_argument("--seed", type=int, default=246)
    parser.add_argument("--planning-seed", type=int, default=246)
    parser.add_argument("--num-candidates", type=int, default=3)
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--score-atol", type=float, default=1e-5,
                        help="Safe modes: absolute minimum score improvement")
    parser.add_argument("--score-rtol", type=float, default=1e-4,
                        help="Safe modes: relative minimum score improvement")
    parser.add_argument("--reach-trigger", type=float, default=0.1,
                        help="feedback_safe: intervene only above this nominal predicted reach cost; uncalibrated default")
    parser.add_argument("--reach-improvement-atol", type=float, default=1e-5)
    parser.add_argument("--reach-improvement-rtol", type=float, default=0.1)
    parser.add_argument("--goal-degradation-atol", type=float, default=1e-6)
    parser.add_argument("--goal-degradation-rtol", type=float, default=0.0,
                        help="feedback_safe: permitted relative goal-cost increase (default none)")
    parser.add_argument("--workers", type=int, default=0, help="Data loader workers (0 avoids I/O fan-out)")
    parser.add_argument("--output-root", type=Path, default=PROJECT / "checkpoint/maze2d_large_diverse/feedback_experiment")
    parser.add_argument("--run-name", help="New directory name; existing directories are rejected")
    parser.add_argument("--smoke", action="store_true", help="Tiny wiring check, NOT a benchmark result")
    parser.add_argument("--dry-run", action="store_true", help="Print command without importing or running the model")
    args = parser.parse_args(argv)
    try:
        checkpoint = resolve_checkpoint(args.checkpoint)
        if args.num_candidates < 1 or args.workers < 0:
            raise ValueError("num-candidates >= 1 and workers >= 0 required")
        if not math.isfinite(args.weight) or args.weight < 0:
            raise ValueError("weight must be finite and non-negative")
        for name in ("score_atol", "score_rtol", "reach_trigger", "reach_improvement_atol",
                     "reach_improvement_rtol", "goal_degradation_atol", "goal_degradation_rtol"):
            value = getattr(args, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name.replace('_', '-')} must be finite and non-negative")
        if args.reach_improvement_rtol >= 1:
            raise ValueError("reach-improvement-rtol must be < 1")
        if not 0 <= args.seed < 2**32 or not 0 <= args.planning_seed < 2**32:
            raise ValueError("seeds must be in [0, 2**32)")
        if not re.fullmatch(r"\d+", args.gpu):
            raise ValueError("--gpu must be one physical GPU index, e.g. 2")
        run_name = args.run_name or f"{args.mode}_seed{args.seed}_p{args.planning_seed}"
        if args.smoke:
            run_name += "_smoke"
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name) or run_name in (".", ".."):
            raise ValueError("run-name must be a simple directory name")
        # The legacy TrainConfig has special destructive handling for 'test'.
        if "test" in run_name:
            raise ValueError("Avoid 'test' in run-name; use 'smoke' for small checks")
        output_root = args.output_root.expanduser().resolve()
        if (output_root / run_name).exists():
            raise ValueError(f"Output already exists: {output_root / run_name}; choose --run-name")
    except ValueError as exc:
        parser.error(str(exc))

    feedback = args.mode != "baseline"
    weight = 0.0 if args.mode in ("candidates", "candidates_safe") else args.weight
    selection_mode = {
        "candidates_safe": "tolerance", "feedback_safe": "conservative",
    }.get(args.mode, "legacy")
    # Quote string override values for OmegaConf, including checkpoint filenames
    # containing '='. subprocess receives an argument list, never shell text.
    import json

    values = [
        "eval_only=true", "wandb=false", "load_l1_only=false",
        "require_full_checkpoint=true", "eval_cfg.probing.full_finetune=false",
        f"seed={args.seed}", f"eval_cfg.planning_seed={args.planning_seed}",
        f"load_checkpoint_path={json.dumps(str(checkpoint))}",
        f"output_root={json.dumps(str(output_root))}", f"output_dir={run_name}",
        f"data.num_workers={args.workers}", "data.prefetch_factor=2",
        f"data.d4rl_config.num_workers={args.workers}",
        "eval_cfg.h_d4rl_planning.visualize_planning=false",
        "eval_cfg.probing.visualize_probing=false",
        f"eval_cfg.h_d4rl_planning.subgoal_feedback.enabled={str(feedback).lower()}",
        f"eval_cfg.h_d4rl_planning.subgoal_feedback.num_candidates={args.num_candidates}",
        f"eval_cfg.h_d4rl_planning.subgoal_feedback.weight={weight}",
        f"eval_cfg.h_d4rl_planning.subgoal_feedback.selection_mode={selection_mode}",
    ]
    for name in ("score_atol", "score_rtol", "reach_trigger", "reach_improvement_atol",
                 "reach_improvement_rtol", "goal_degradation_atol", "goal_degradation_rtol"):
        values.append(f"eval_cfg.h_d4rl_planning.subgoal_feedback.{name}={getattr(args, name)}")
    if args.smoke:
        values += [
            "eval_cfg.probing.epochs=1", "eval_cfg.probing.epochs_enc=1",
            "eval_cfg.probing.epochs_latent=1",
            "eval_cfg.h_d4rl_planning.levels=medium",
            "eval_cfg.h_d4rl_planning.medium.n_envs=2",
            "eval_cfg.h_d4rl_planning.medium.n_steps=8",
            "eval_cfg.h_d4rl_planning.medium.max_plan_length_l2=2",
            "eval_cfg.h_d4rl_planning.medium.level1.mppi.num_samples=32",
            "eval_cfg.h_d4rl_planning.medium.level2.mppi.num_samples=64",
            "eval_cfg.h_d4rl_planning.n_envs_batch_size=2",
            "eval_cfg.h_d4rl_planning.final_trans_steps=0",
        ]
    command = [
        sys.executable, "-m", "pldm.train", "--configs",
        str(PROJECT / "pldm/configs/diverse_maze/icml/large_diverse_25maps_l2.yaml"),
        "--values", *values,
    ]
    print(f"Checkpoint: {checkpoint}\nOutput: {output_root / run_name}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES={args.gpu} " + shlex.join(command), flush=True)
    if selection_mode == "conservative":
        print(
            "Conservative gates use predicted errors, not true executability. "
            "Thresholds are uncalibrated defaults; use validation tasks to choose them.",
            flush=True,
        )
    if args.smoke:
        print("SMOKE CHECK ONLY: do not compare these rates with the 40-task experiment.", flush=True)
    if args.dry_run:
        return 0
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONUNBUFFERED="1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")
    try:
        return subprocess.call(command, cwd=PROJECT, env=env)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
