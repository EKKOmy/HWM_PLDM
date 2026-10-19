# 实验：低层预测执行误差驱动的子目标重排序

## 假设与范围

高层目标代价小，不代表低层能在给定时长和搜索预算内实现该子目标。
本实验检验：在相同 HWM checkpoint 下，使用低层预测的执行误差重排
高层候选，是否改善困难迷宫任务的成功率。它是一个待验证假设，不保证提升。

本版本只修改推理，不重新训练世界模型、不使用测试环境状态作为规划输入，
也没有新增可达性网络。误差由冻结的低层模型估计，不是真实可达性的保证。
只支持当前 MPPI、latent L2 action、obs_component、单模型 ensemble 的配置。

## 方法

每次原有 L2 MPPI 搜索完成后，保留原 nominal 序列，以及该次搜索中
代价最低的 C−1 条已采样序列。默认 C=3。不额外进行 L2 随机搜索。
原采样总成本仅用于筛选；全部候选（含 nominal）重新用同一个纯目标代价评分。

对每个候选：

1. 用高层序列的第一个预测状态作为子目标。
2. 低层从相同 warm start 出发，使用相同样本数、相同随机噪声，规划 10 步。
3. 计算高层原目标函数所覆盖的未来时刻的平均成本 G。
4. 计算低层预测终点到子目标的成本 R（使用低层 RunningCost 的表示映射）。
5. 每个环境独立选择 G + λR 最小的候选，默认 λ=1。

执行被选候选已经求得的低层动作，不再为胜者额外搜索。保存被选高低层
归一化动作序列供下一次 warm start；额外候选检查不推进全局随机流。
num_candidates=1 应保持 nominal 行为，是一个回归检查。
权重的量级需要在独立验证任务上选择，不能根据最终测试成功率反复调参。

默认功能关闭，旧命令不启用反馈。此次没有顺手修复原 MPPI 的 z_reg 等算法问题，
避免把多个改动混在一次实验中。

## 三组对照

| 模式 | 候选与选择 | 每次高层规划后的低层搜索次数 |
|---|---|---|
| baseline | 原 HWM nominal | 1 |
| feedback | C 个候选，G + λR | C |
| candidates | 相同 C 个候选，λ=0，仅按 G 选择 | C |

三组使用同一个完整分层 checkpoint、相同起终点文件、环境种子与显式 planning seed。
默认 medium/hard 各 40 个任务；不改变各组原有的规划 horizon、环境步数及末段平层策略。
当前 HWM 与旧 flat 的环境步数/搜索预算并不完全一致，flat 仅作为历史参考。

feedback 对 baseline 的差异包含额外计算；feedback 对 candidates 的差异才能更好地
隔离执行误差项的作用。后者算子预算相同，但具体运行时间仍应实测。
本版本没有实现严格等耗时的自适应调度，不应宣称同算力胜过 baseline。

## 在服务器上运行

把此次修改同步到服务器的 /home/my/HWM_PLDM；无需重新渲染数据。
以下入口只在服务器运行。它使用当前 Python 解释器启动 pldm.train，绕过 conda activate。

checkpoint 可以是训练输出目录（按 epoch、sample_step 数值选最新文件），也可以是
明确的 .ckpt 文件。**正式比较应固定同一个文件**，不要在训练仍写入的目录中动态选取。
入口会打印解析后的路径，并校验完整模型权重；不能使用仅有 L1 的预训练文件。

先检查命令，不启动模型：

```bash
cd /home/my/HWM_PLDM
/home/my/miniconda3/envs/hwm/bin/python scripts/eval_subgoal_feedback.py feedback \
  --checkpoint /home/my/HWM_PLDM/checkpoint/maze2d_large_diverse/l2_wo_encoder/hwm_l2_seed246 \
  --gpu 2 --dry-run
```

先做小规模接线检查（2 个 medium 任务、8 步、小采样数；不是正式结果）：

```bash
/home/my/miniconda3/envs/hwm/bin/python scripts/eval_subgoal_feedback.py feedback \
  --checkpoint /home/my/HWM_PLDM/checkpoint/maze2d_large_diverse/l2_wo_encoder/hwm_l2_seed246 \
  --gpu 2 --smoke
```

正式实验，建议在同一张卡上顺序执行，降低并行任务对计时的影响：

```bash
/home/my/miniconda3/envs/hwm/bin/python scripts/eval_subgoal_feedback.py baseline \
  --checkpoint /home/my/HWM_PLDM/checkpoint/maze2d_large_diverse/l2_wo_encoder/hwm_l2_seed246 \
  --gpu 2

/home/my/miniconda3/envs/hwm/bin/python scripts/eval_subgoal_feedback.py feedback \
  --checkpoint /home/my/HWM_PLDM/checkpoint/maze2d_large_diverse/l2_wo_encoder/hwm_l2_seed246 \
  --gpu 2

/home/my/miniconda3/envs/hwm/bin/python scripts/eval_subgoal_feedback.py candidates \
  --checkpoint /home/my/HWM_PLDM/checkpoint/maze2d_large_diverse/l2_wo_encoder/hwm_l2_seed246 \
  --gpu 2
```

默认 seed=246，planning-seed=246，数据 worker=0，绘图关闭。评估入口仍会训练位置探针，
但不会更新世界模型；探针耗时不算入 planning_elapsed_seconds。
如需重复实验可指定 --planning-seed 247 / 248；它只检验同一模型的规划随机性，
不等同于独立训练种子实验。世界模型多种子实验需使用相应独立训练的 checkpoint。

输出目录已存在时入口会拒绝运行，避免覆盖。重试用新 --run-name，例如 feedback_retry1。
使用 --smoke 时自动加 _smoke 后缀。不要把 smoke 结果与完整 40 任务实验比较。

## 查看结果与历史比较

```bash
/home/my/miniconda3/envs/hwm/bin/python scripts/compare_planning_runs.py \
  --run baseline=checkpoint/maze2d_large_diverse/feedback_experiment/baseline_seed246_p246/summary.json \
  --run feedback=checkpoint/maze2d_large_diverse/feedback_experiment/feedback_seed246_p246/summary.json \
  --run candidates=checkpoint/maze2d_large_diverse/feedback_experiment/candidates_seed246_p246/summary.json \
  --historical
```

也可增加已有的真实 summary：

```bash
/home/my/miniconda3/envs/hwm/bin/python scripts/compare_planning_runs.py \
  --run old_hwm=checkpoint/maze2d_large_diverse/l2_wo_encoder/hwm_l2_seed246/summary.json \
  --run feedback=checkpoint/maze2d_large_diverse/feedback_experiment/feedback_seed246_p246/summary.json \
  --historical
```

历史参考是用户已报告的数据：flat medium=67.5%、hard=45%；HWM medium=80%、hard=62.5%。
这些数值没有被重新测量。旧训练末尾评估受 epoch 和探针 RNG 消耗影响；新入口显式固定
planning seed，因此重新评估的 baseline 不保证恰好等于历史数值。主要结论应基于新配对实验。

每个正式输出目录包含：

- summary.json：成功率、真实任务数、成功数，以及 planning_elapsed_seconds。
- planning_details_l2_d4rl_medium.json / planning_details_l2_d4rl_hard.json：实际 checkpoint、
  配置、规划种子、逐任务成败与起终点，以及 feedback_records。
- 原有 tensor 结果文件与探针结果。

计时包含分块规划器构建、模型规划、环境执行和记录，不含之前的探针训练与之后的绘图；
它不是纯 GPU kernel 时间。各模式计时范围相同，额外候选 rollout/check 开销也包含在内。
feedback_records 中记录每次选择、每候选 G/R/score、采样数与 horizon。
可据此检查是否真的更换过子目标、R 是否降低，而不是仅看最终成功率。

第一轮成功标准是：代码可运行、旧行为保留、记录完整、能判断反馈是否有价值。
40 个任务上的一次提升不是论文结论；下一步需要独立验证集选参数、多种子统计、
更多未见地图及第二类任务。不要为追求超过历史数字而在同一测试集上反复选 λ。

## 本地轻量验证

```bash
python3 -m unittest discover -s tests -v
git diff --check
```

单元测试使用标准库和 CPU toy tensors，不训练模型、不加载数据、不运行 MuJoCo。
完整 CUDA / MuJoCo 路径仍需在服务器执行 --smoke 后验证。
