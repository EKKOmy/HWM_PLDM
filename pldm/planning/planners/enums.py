import enum
from dataclasses import dataclass
from pldm.configs import ConfigBase


class PlannerType(enum.Enum):
    SGD = enum.auto()
    MPPI = enum.auto()
    BeamSearch = enum.auto()


@dataclass
class SubgoalFeedbackConfig(ConfigBase):
    # Opt-in inference-only experiment. Candidate 0 is the unchanged MPPI plan.
    enabled: bool = False
    num_candidates: int = 3
    weight: float = 1.0
    # Keep v1 behavior unless a new policy is explicitly requested.
    selection_mode: str = "legacy"  # legacy / tolerance / conservative
    score_atol: float = 1e-5
    score_rtol: float = 1e-4
    # Uncalibrated engineering defaults, not values optimized on test outcomes.
    reach_trigger: float = 0.1
    reach_improvement_atol: float = 1e-5
    reach_improvement_rtol: float = 0.1
    goal_degradation_atol: float = 1e-6
    goal_degradation_rtol: float = 0.0

    def __post_init__(self):
        import math

        if self.num_candidates < 1:
            raise ValueError("feedback.num_candidates must be >= 1")
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("feedback.weight must be finite and >= 0")
        if self.selection_mode not in ("legacy", "tolerance", "conservative"):
            raise ValueError("Unknown feedback.selection_mode")
        for name in (
            "score_atol", "score_rtol", "reach_trigger",
            "reach_improvement_atol", "reach_improvement_rtol",
            "goal_degradation_atol", "goal_degradation_rtol",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"feedback.{name} must be finite and >= 0")
        if self.reach_improvement_rtol >= 1:
            raise ValueError("feedback.reach_improvement_rtol must be < 1")


@dataclass
class SGDConfig(ConfigBase):
    n_iters: int = 100
    lr: float = 1e-2
    l2_reg: float = 0.0
    action_change_reg: float = 0.0
    z_reg_coeff: float = 0.0


@dataclass
class MPPIConfig(ConfigBase):
    noise_sigma: float = 5
    num_samples: int = 500
    lambda_: float = 0.005
    z_reg_coeff: float = 0
    var_samples: int = 0
    rollout_var_cost: float = 0.0
    rollout_obs_var_cost: float = 0.0
    rollout_proprio_var_cost: float = 0.0
    rollout_var_discount: float = 0.95


@dataclass
class LFBGSConfig(ConfigBase):
    history_size: int = 10
    max_iter: int = 20


@dataclass
class PlannerConfig(ConfigBase):
    planner_type: PlannerType = PlannerType.SGD
    sgd: SGDConfig = SGDConfig()
    mppi: MPPIConfig = MPPIConfig()
    lfgbs: LFBGSConfig = LFBGSConfig()
    clamp_actions: bool = False
    min_step: float = 0.1
    max_step: float = 1.0
    repr_target: bool = False
    loss_coeff_first: float = 0.1
    loss_coeff_last: float = 1
    sum_all_diffs: bool = False
    min_plan_length: int = 1
    max_plan_length: int = 26
    depth_probe_threshold: int = 25
    probe_depth: bool = False
    sum_last_n: int = 3
    proprio_cost: bool = False
    projected_cost: bool = False
    cost_entity: str = "obs_component"
    cost_dim_range: str = (
        "0:99999999"  # only taking cost in this range of the representation
    )
