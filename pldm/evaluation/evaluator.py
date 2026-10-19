from pldm.configs import ConfigBase
import torch
from typing import Optional
from dataclasses import dataclass
import dataclasses
from enum import Enum
import json
import math
from pathlib import Path
import random
import zlib

import numpy as np

from pldm.probing.evaluator import ProbingConfig, ProbingEvaluator
from pldm.data.enums import ProbingDatasets
from pldm.planning.d4rl.enums import D4RLMPCConfig, HierarchicalD4RLMPCConfig
from pldm.planning.enums import LevelConfig, MPCConfig
from omegaconf import MISSING

from pldm_envs.utils.normalizer import Normalizer

from pldm_envs.diverse_maze.utils import PixelMapper as D4RLPixelMapper


@dataclass
class EvalConfig(ConfigBase):
    env_name: str = MISSING
    probing: ProbingConfig = ProbingConfig()
    eval_l1: bool = True
    eval_l2: bool = False
    log_heatmap: bool = True
    disable_planning: bool = False
    disable_l2_planning: bool = False
    d4rl_planning: D4RLMPCConfig = D4RLMPCConfig()
    h_d4rl_planning: HierarchicalD4RLMPCConfig = HierarchicalD4RLMPCConfig()
    manispace_planning: MPCConfig = MPCConfig()
    l2_latent_bounds_percentile: float = 0
    l2_use_latent_mean_std: bool = False
    planning_seed: Optional[int] = None

    def __post_init__(self):
        self.d4rl_planning.env_name = self.env_name
        self.h_d4rl_planning.env_name = self.env_name
        self.manispace_planning.env_name = self.env_name


class Evaluator:
    def __init__(
        self,
        config: EvalConfig,
        model: torch.nn.Module,
        quick_debug: bool,
        normalizer: Normalizer,
        epoch: int,
        probing_datasets: Optional[ProbingDatasets],
        l2_probing_datasets: Optional[ProbingDatasets],
        load_checkpoint_path: "",
        output_path: "",
        data_config=None,
    ):
        self.config = config
        self.model = model
        self.quick_debug = quick_debug
        self.normalizer = normalizer
        self.epoch = epoch
        self.output_path = output_path
        self.load_checkpoint_path = load_checkpoint_path
        self.data_config = data_config  # wall_config

        self.probing_evaluator = ProbingEvaluator(
            model=self.model,
            config=self.config.probing,
            quick_debug=self.quick_debug,
            probing_datasets=probing_datasets,
            l2_probing_datasets=l2_probing_datasets,
            load_checkpoint_path=load_checkpoint_path,
            output_path=output_path,
        )

        self.pixel_mapper = self._create_pixel_mapper()
        self.planning_config = self._get_planning_config()
        # Get hierarchical planning config based on environment
        if "diverse" in self.config.env_name or "maze" in self.config.env_name:
            self.h_planning_config = getattr(self.config, "h_d4rl_planning", None)
        else:
            self.h_planning_config = None

    @staticmethod
    def _merge_dataclass_overrides(base_cfg, override_cfg):
        """
        Merge nested dataclass overrides by applying only values that differ
        from that dataclass type's defaults.
        """
        if override_cfg is None:
            return base_cfg

        if not dataclasses.is_dataclass(base_cfg) or not dataclasses.is_dataclass(
            override_cfg
        ):
            return override_cfg

        default_cfg = type(override_cfg)()
        updates = {}

        for field in dataclasses.fields(override_cfg):
            name = field.name
            override_value = getattr(override_cfg, name)
            default_value = getattr(default_cfg, name)
            base_value = getattr(base_cfg, name)

            if dataclasses.is_dataclass(override_value):
                merged_value = Evaluator._merge_dataclass_overrides(
                    base_value, override_value
                )
                if merged_value != base_value:
                    updates[name] = merged_value
            elif override_value != default_value:
                updates[name] = override_value

        if not updates:
            return base_cfg

        return dataclasses.replace(base_cfg, **updates)

    def _get_planning_config(self):
        if "diverse" in self.config.env_name or "maze" in self.config.env_name:
            config = self.config.d4rl_planning
        else:
            raise NotImplementedError
        return config

    def _reset_planning_rng(self, level):
        """Make each difficulty independent of probe RNG and evaluation order."""
        if self.config.planning_seed is None:
            return None
        offsets = {"easy": 0, "medium": 1009, "hard": 2018}
        offset = offsets.get(level, zlib.crc32(level.encode("utf-8")))
        seed = (int(self.config.planning_seed) + offset) % (2**32)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return seed

    @staticmethod
    def _json_value(value):
        if dataclasses.is_dataclass(value):
            return {
                field.name: Evaluator._json_value(getattr(value, field.name))
                for field in dataclasses.fields(value)
            }
        if isinstance(value, torch.Tensor):
            return Evaluator._json_value(value.detach().cpu().tolist())
        if isinstance(value, (np.ndarray, np.generic)):
            return Evaluator._json_value(value.tolist())
        if isinstance(value, Enum):
            return f"{type(value).__name__}.{value.name}"
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): Evaluator._json_value(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [Evaluator._json_value(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    def _save_planning_details(self, planning_evaluator, result, report, planning_seed):
        if self.output_path is None:
            return
        successes = self._json_value(report.success)
        starts = self._json_value(result.locations[0]) if result.locations else None
        final_positions = self._json_value(result.locations[-1]) if result.locations else None
        targets = self._json_value(result.targets)
        trials = getattr(planning_evaluator, "trials", {})
        tasks = []
        for index, success in enumerate(successes):
            tasks.append({
                "env_index": index,
                "success": bool(success),
                "start": starts[index] if starts is not None else None,
                "target": targets[index],
                "final_position": final_positions[index] if final_positions is not None else None,
                "first_success_step_index": report.terminations[index] if success else None,
                "num_turns": report.num_turns[index],
                "block_distance": report.block_dists[index],
                "map_layout": trials.get("map_layouts", [None] * len(successes))[index],
            })
        checkpoint_path = (
            str(Path(self.load_checkpoint_path).expanduser().resolve())
            if self.load_checkpoint_path else None
        )
        details = {
            "schema_version": 1,
            "checkpoint_path": checkpoint_path,
            "epoch": self.epoch,
            "prefix": planning_evaluator.prefix,
            "planning_seed": planning_seed,
            "planning_seed_base": self.config.planning_seed,
            "config": planning_evaluator.config,
            "eval_config": self.config,
            "num_tasks": len(successes),
            "successes": sum(successes),
            "success_rate": report.success_rate,
            "elapsed_seconds": report.elapsed_seconds,
            "timing_scope": (
                "synchronized chunk planner construction, planning and environment "
                "rollout; excluding evaluator construction, probing and final plotting"
            ),
            "executed_steps_per_task": len(result.reward_history),
            "tasks": tasks,
            "feedback_records": getattr(planning_evaluator, "feedback_records", []),
        }
        path = Path(self.output_path) / f"planning_details_{planning_evaluator.prefix}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self._json_value(details), handle, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"saved planning details to {path}")

    def evaluate_loc_probing(self):
        probers = {}
        probers_l2 = {}

        probers = self.probing_evaluator.train_pred_prober(
            epoch=self.epoch,
            train_probers=self.config.eval_l1,
        )

        if self.config.eval_l1:
            if self.config.probing.probe_preds:
                self.probing_evaluator.evaluate_all(
                    probers=probers,
                    epoch=self.epoch,
                    pixel_mapper=self.pixel_mapper.obs_coord_to_pixel_coord,
                )

            if self.config.probing.probe_encoder:
                enc_probers = self.probing_evaluator.train_encoder_prober(
                    epoch=self.epoch,
                )

                enc_probe_loss = self.probing_evaluator.eval_probe_enc_position(
                    probers=enc_probers,
                    epoch=self.epoch,
                )

        # L2 Probing
        if self.config.eval_l2:
            probers_l2 = self.probing_evaluator.train_pred_prober(
                epoch=self.epoch,
                l2=True,
            )

            if self.config.probing.probe_preds:
                self.probing_evaluator.evaluate_all(
                    probers=probers_l2,
                    epoch=self.epoch,
                    l2=True,
                    pixel_mapper=self.pixel_mapper.obs_coord_to_pixel_coord,
                )

        return probers, probers_l2

    def _create_pixel_mapper(self):
        if "diverse" in self.config.env_name or "maze2d" in self.config.env_name:
            pixel_mapper = D4RLPixelMapper(env_name=self.config.env_name)
        else:

            class IdPixelMapper:
                def obs_coord_to_pixel_coord(self, x):
                    return x

                def pixel_coord_to_obs_coord(self, x):
                    return x

            pixel_mapper = IdPixelMapper()

        return pixel_mapper

    def _create_l1_planning_evaluator(
        self,
        level: str,
        level_config: Optional[LevelConfig],
    ):
        planner_config = self.planning_config.level1

        if level_config is not None and level_config.override_config:
            max_plan_length = level_config.max_plan_length
            n_envs = level_config.n_envs
            n_steps = level_config.n_steps
            offline_T = level_config.offline_T
            plot_every = level_config.plot_every
            set_start_target_path = level_config.set_start_target_path
            planner_config = self._merge_dataclass_overrides(
                planner_config, level_config.level1
            )

            if self.quick_debug:
                # max_plan_length = self.planning_config.level1.max_plan_length
                n_envs = self.planning_config.n_envs
                n_steps = self.planning_config.n_steps
        else:
            max_plan_length = self.planning_config.level1.max_plan_length
            n_envs = self.planning_config.n_envs
            n_steps = self.planning_config.n_steps
            offline_T = self.planning_config.offline_T
            set_start_target_path = self.planning_config.set_start_target_path
            plot_every = self.planning_config.plot_every

        planner_config = dataclasses.replace(
            planner_config,
            max_plan_length=max_plan_length,
        )

        mpc_config = dataclasses.replace(
            self.planning_config,
            level=level,
            n_envs=n_envs,
            n_steps=n_steps,
            level1=planner_config,
            offline_T=offline_T,
            plot_every=plot_every,
            set_start_target_path=set_start_target_path,
        )

        if "diverse" in self.config.env_name or "maze2d" in self.config.env_name:
            from pldm.planning.d4rl.mpc import MazeMPCEvaluator

            planning_evaluator = MazeMPCEvaluator(
                config=mpc_config,
                normalizer=self.normalizer,
                model=self.model,
                pixel_mapper=self.pixel_mapper,
                prober=self.probers["locations"],
                prefix=f"d4rl_{level}",
                quick_debug=self.quick_debug,
                l2_use_latent_mean_std=self.config.l2_use_latent_mean_std,
            )
        else:
            raise NotImplementedError

        return planning_evaluator

    def _create_l2_planning_evaluator(
        self,
        level: str,
        level_config: Optional[LevelConfig],
    ):
        planner_config_l1 = self.h_planning_config.level1
        planner_config_l2 = self.h_planning_config.level2

        if level_config is not None and level_config.override_config:
            max_plan_length_l1 = level_config.max_plan_length
            max_plan_length_l2 = level_config.max_plan_length_l2
            n_envs = level_config.n_envs
            n_steps = level_config.n_steps
            set_start_target_path = level_config.set_start_target_path
            plot_every = level_config.plot_every
            planner_config_l1 = self._merge_dataclass_overrides(
                planner_config_l1, level_config.level1
            )
            planner_config_l2 = self._merge_dataclass_overrides(
                planner_config_l2, level_config.level2
            )

            if self.quick_debug:
                max_plan_length_l2 = self.h_planning_config.level2.max_plan_length
                max_plan_length_l1 = self.h_planning_config.level1.max_plan_length
                n_envs = self.h_planning_config.n_envs
                n_steps = self.h_planning_config.n_steps
        else:
            max_plan_length_l2 = self.h_planning_config.level2.max_plan_length
            max_plan_length_l1 = self.h_planning_config.level1.max_plan_length
            n_envs = self.h_planning_config.n_envs
            n_steps = self.h_planning_config.n_steps
            set_start_target_path = self.h_planning_config.set_start_target_path
            plot_every = self.h_planning_config.plot_every

        planner_config_l2 = dataclasses.replace(
            planner_config_l2,
            max_plan_length=max_plan_length_l2,
        )

        planner_config_l1 = dataclasses.replace(
            planner_config_l1,
            max_plan_length=max_plan_length_l1,
        )

        mpc_config = dataclasses.replace(
            self.h_planning_config,
            level=level,
            n_envs=n_envs,
            n_steps=n_steps,
            level2=planner_config_l2,
            level1=planner_config_l1,
            set_start_target_path=set_start_target_path,
            plot_every=plot_every,
        )

        if "maze2d" in self.config.env_name:
            from pldm.planning.d4rl.hmpc import HierarchicalD4RLMPCEvaluator
            from pldm.planning.d4rl.enums import HierarchicalD4RLMPCConfig

            planning_evaluator = HierarchicalD4RLMPCEvaluator(
                config=mpc_config,
                normalizer=self.normalizer,
                model=self.model,
                pixel_mapper=self.pixel_mapper,
                prober=self.probers["locations"],
                prober_l2=(
                    self.probers_l2.get("l2_locations", None)
                    if self.probers_l2
                    else None
                ),
                prefix=f"l2_d4rl_{level}",
                quick_debug=self.quick_debug,
                l2_use_latent_mean_std=self.config.l2_use_latent_mean_std,
            )

        return planning_evaluator

    def _get_planning_levels(self, l2=False):
        if l2:
            levels = self.h_planning_config.levels.split(",")
            level_configs = [
                getattr(self.h_planning_config, level, None) for level in levels
            ]
        else:
            levels = self.planning_config.levels.split(",")
            level_configs = [
                getattr(self.planning_config, level, None) for level in levels
            ]

        # if self.quick_debug:
        #     levels = [levels[0]]

        return (levels, level_configs)

    def evaluate(self):
        """
        Evaluation consists of both probing and planning
        """

        log_dict = {}

        self.probers, self.probers_l2 = self.evaluate_loc_probing()

        # Planning
        if not self.config.disable_planning and self.config.eval_l1:
            levels, level_configs = self._get_planning_levels()

            for i, level in enumerate(levels):
                level_config = level_configs[i]

                planning_seed = self._reset_planning_rng(level)
                planning_evaluator = self._create_l1_planning_evaluator(
                    level=level,
                    level_config=level_config,
                )

                print(
                    f"evaluating planning level {level} for {planning_evaluator.config.n_envs} envs"
                )

                mpc_result, mpc_report = planning_evaluator.evaluate()

                self._save_planning_details(
                    planning_evaluator, mpc_result, mpc_report, planning_seed
                )
                planning_evaluator.close()

                log_dict.update(
                    mpc_report.build_log_dict(prefix=planning_evaluator.prefix)
                )

                torch.save(
                    mpc_result,
                    f"{self.output_path}/planning_l1_mpc_result_{planning_evaluator.prefix}",
                )
                torch.save(
                    mpc_report,
                    f"{self.output_path}/planning_l1_mpc_report_{planning_evaluator.prefix}",
                )

        if not self.config.disable_l2_planning and self.config.eval_l2:
            levels, level_configs = self._get_planning_levels(l2=True)

            for i, level in enumerate(levels):
                level_config = level_configs[i]

                planning_seed = self._reset_planning_rng(level)
                planning_evaluator = self._create_l2_planning_evaluator(
                    level=level,
                    level_config=level_config,
                )
                print(
                    f"evaluating l2 planning level {level} for {planning_evaluator.config.n_envs} envs"
                )

                l2_mpc_result, l2_mpc_report = planning_evaluator.evaluate()
                self._save_planning_details(
                    planning_evaluator, l2_mpc_result, l2_mpc_report, planning_seed
                )
                planning_evaluator.close()
                log_dict.update(
                    l2_mpc_report.build_log_dict(prefix=planning_evaluator.prefix)
                )

                torch.save(
                    l2_mpc_result,
                    f"{self.output_path}/planning_l2_mpc_result_{planning_evaluator.prefix}",
                )
                torch.save(
                    l2_mpc_report,
                    f"{self.output_path}/planning_l2_mpc_report_{planning_evaluator.prefix}",
                )

        self.model.train()

        return log_dict
