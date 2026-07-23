---
title: 统一 Tracking Metrics v2
slug: /reference/tracking-metrics-diff
---

# 统一 Tracking Metrics v2

本文记录 `sim2real.metrics.tracking` 实现的正式指标契约；兼容命令行入口仍为
`scripts/tracking_experiment/compute_tracking_metrics.py`。本文也逐项说明它与
当前仓库旧指标、Spider G1 WBC 指标和 SONIC 评测指标的差异。目标是在同一条
motion 或同一个 motion 数据集上，对多个 controller 产生公平、可审计、可复算的
rollout 对比结果。

v2 的层次结构为：

```text
outcome
tracking
├── global_start_aligned
├── local_heading
└── joint_space
smoothness
metadata
```

v2 不计算综合分数，也不自动给 controller 排名。输出保留各个独立测量量，
由人类结合任务需求比较 controller，避免用一组权重掩盖取舍。

## 坐标系契约

设一条轨迹初始时刻的 root 位置和朝向为 \(r(0)\) 与 \(q_r(0)\)。robot 和
reference 分别使用各自的初始 root 状态完成变换。

### Global start-aligned

对 body 位置 \(p_i(t)\) 和朝向 \(q_i(t)\)：

```text
p_i^SA(t) = R(q_r(0))^T [p_i^W(t) - r^W(0)]
q_i^SA(t) = q_r(0)^-1 ⊗ q_i^W(t)
```

robot 与 reference 各自先减去自己的初始 root 平移，并移除自己的初始完整
root 朝向，然后再计算两者误差。这会消除任意的初始世界坐标平移和朝向偏差，
但保留后续 root 漂移、转向、高度变化与全身轨迹误差。

这就是 v2 中 `global` 的含义；它不是 raw-world MPJPE。

### Local heading

每一帧中，robot 和 reference 各自使用当前 pelvis 位置及其投影 heading：

```text
a(t) = [pelvis_x(t), pelvis_y(t), 0]
h(t) = projected_yaw(pelvis_orientation(t))

p_i^LH(t) = R(h(t))^T [p_i^W(t) - a(t)]
q_i^LH(t) = h(t)^-1 ⊗ q_i^W(t)
```

这里只消除 pelvis 的 XY 平移和 yaw。世界 Z、pelvis 高度、roll、pitch 以及
重力方向都被保留。因此，该坐标系能够在不受全局水平漂移和 heading 影响的
情况下测量 pose 质量，同时不会掩盖跌倒和高度误差。

### Joint space

robot 与 reference 的 joint 按名称匹配，并以 G1 robot config 的标准顺序
保存。位置、速度误差在单个标量关节坐标上做归约，而不是先对整条 joint
向量取一个 L2 norm。这样结果具有直接物理含义，也不会仅仅因为增加 joint
数量而改变指标尺度。

## 评测窗口与 outcome

tracking 和 smoothness 指标使用 pre-failure 窗口。rollout 第一次满足下列
连续条件时被分析性地判定为 termination：

| 原因 | 阈值 | 连续 policy 帧 |
| --- | ---: | ---: |
| `root_ori_error` | torso 朝向误差 >= 1.2 rad | 25 |
| `body_pos_error` | heading-local key-body 最大位置误差 >= 0.4 m | 5 |
| `body_ori_error` | heading-local key-body 最大朝向误差 >= 1.2 rad | 5 |

确认 failure 的那一帧不进入 tracking 和 smoothness 指标。如果保存的轨迹
没有到达 reference 最后一帧，则记为 `truncated`；否则记为 `motion_end`。

任何 tracking 误差都必须与以下 outcome 一起阅读：

- `completion_ratio`：第一次确认 failure 时的 reference 进度；正常到达
  motion end 时为 1.0；
- `termination_rate` 与 `success_rate`：由分析性 termination 状态得到的
  数据集比例；
- `termination_reasons`：各类原因的计数；
- `runs.failed`：子进程失败或轨迹文件无效的数量，这些 rollout 不会被静默
  混入 tracking error 平均值。

这种配对可以防止一个很早失败的 controller 仅仅因为只在简单前缀上计算
误差而显得指标很好。

## Body 与 end-effector 集合

保存轨迹时的 body layout 固定来自 G1 robot config，不再取决于具体 controller
YAML。trajectory metrics 的 reference FK 始终使用锁定的 canonical G1 MJCF；
即使 BFM-0 等 controller 为自身 observation 或 reward FK 使用 policy-specific
MJCF，也不会改变正式评测的 body 集合。最终使用 simulation model 与这个
canonical reference 共有的交集。

如果不同 controller 产生不同的 key-body、end-effector 或 joint layout，batch
evaluator 会直接拒绝结果，不会静默聚合不可比的 rows。

key-body pattern 选择：

- pelvis 与 torso；
- 左右 hip yaw、knee 和 toe link；
- 左右 shoulder yaw、elbow 和 wrist yaw link。

end-effector 固定为左右 toe 和 wrist yaw link。一次评测实际观察到的 body
集合、end-effector 集合和 joint 集合都会写入 `metric_schema`，便于发现
不同 controller 之间意外的选择差异。

## 指标目录

所有朝向误差均使用四元数测地角，单位为 rad；位置误差是欧氏距离。每个
rollout 的 `mean` 和 `p95` 在评测窗口内对全部有效 frame/body 样本归约。

### Global start-aligned tracking

| JSON 指标 | 单 rollout CSV 字段 | 单位 |
| --- | --- | --- |
| root XYZ 位置 mean/p95 | `global_root_pos_xyz_{mean,p95}_m` | m |
| root XY 位置 mean/p95 | `global_root_pos_xy_{mean,p95}_m` | m |
| root 朝向 mean/p95 | `global_root_ori_{mean,p95}_rad` | rad |
| key-body 位置 mean/p95 | `global_key_body_pos_{mean,p95}_m` | m |
| key-body 朝向 mean/p95 | `global_key_body_ori_{mean,p95}_rad` | rad |
| end-effector 位置 mean/p95 | `global_end_effector_pos_{mean,p95}_m` | m |
| end-effector 朝向 mean/p95 | `global_end_effector_ori_{mean,p95}_rad` | rad |
| key-body 速度跟踪 mean/p95 | `global_key_body_vel_error_{mean,p95}_mps` | m/s |
| key-body 加速度跟踪 mean/p95 | `global_key_body_acc_error_{mean,p95}_mps2` | m/s² |

速度和加速度是 start-aligned key-body 位置的有限差分。每次求导都使用轨迹
中真实记录的 `sim_time`，指标代码内部不假定固定频率。

### Heading-local tracking

| JSON 指标 | 单 rollout CSV 字段 | 单位 |
| --- | --- | --- |
| key-body 位置 mean/p95 | `local_key_body_pos_{mean,p95}_m` | m |
| key-body 朝向 mean/p95 | `local_key_body_ori_{mean,p95}_rad` | rad |
| end-effector 位置 mean/p95 | `local_end_effector_pos_{mean,p95}_m` | m |
| end-effector 朝向 mean/p95 | `local_end_effector_ori_{mean,p95}_rad` | rad |

### Joint-space tracking

| JSON 指标 | 单 rollout CSV 字段 | 单位 |
| --- | --- | --- |
| 位置 MAE/RMSE/p95 绝对误差 | `joint_pos_{mae,rmse,p95_abs}_rad` | rad |
| 速度 MAE/RMSE/p95 绝对误差 | `joint_vel_{mae,rmse,p95_abs}_rad_s` | rad/s |

MAE 表示典型误差，RMSE 对大误差更敏感，p95 则在避免单个不稳定最大值的
同时揭示尾部行为。

### Smoothness

| JSON 指标 | 单 rollout CSV 字段 | 单位 |
| --- | --- | --- |
| robot joint acceleration RMS/p95 absolute | `joint_acc_{rms,p95_abs}_rad_s2` | rad/s² |
| robot joint jerk RMS/p95 absolute | `joint_jerk_{rms,p95_abs}_rad_s3` | rad/s³ |

定义为：

```text
joint_acceleration = d(qdot) / dt
joint_jerk         = d(joint_acceleration) / dt
```

smoothness 与 reference tracking 分开报告。高动态 reference 本身可能需要
较大的加速度，因此必须结合 dynamic tracking error 解释 smoothness。

## 数据集聚合

`tracking_metrics.csv` 的一行对应一个
`(controller, motion, seed)` rollout。每个 rollout 先把自己的全部 frame
归约为 mean 或 p95 标量；数据集 summary 再对这些 rollout 标量做 macro
average，因此不论 motion 长短，每个 rollout 权重相同。

每个 summary 叶子节点为：

```json
{
  "mean": 0.0,
  "std": 0.0,
  "valid_count": 1
}
```

`std` 是 rollout 间的总体标准差。如果高阶导数对极短轨迹没有定义，单
rollout 值写成 JSON `null`，汇总时跳过，并通过 `valid_count` 显示有效
分母。JSON 不会输出非标准 `NaN`。

对于名称中带 `p95` 的 summary 指标，其中的 `mean` 是“各 rollout p95
的平均值”，而不是把全数据集所有帧混在一起计算出的 pooled p95。

## 来源对比与取舍

| 方面 | 当前仓库旧指标 | Spider G1 WBC | SONIC eval | 统一 v2 |
| --- | --- | --- | --- | --- |
| Outcome | progress 和 post-hoc failure | 带阈值的 score/success | terminated、progress、success rate | 明确保留完成率、成功/终止率和原因；不做 score |
| Global 位置 | start-aligned root trajectory | raw-world root/body/EE 距离 | `mpjpe_g` raw 选定 body 距离 | 保留 start alignment，扩展到 root、key body 和 EE |
| Root 朝向 | 只参与 failure，没有正式输出 | raw-world root quaternion error | 不是主要 pose 输出 | 新增 start-aligned 测地角 mean/p95 |
| Local pose | heading-local key-body position | 当前 anchor 的完整 SE(3)-local 位置/朝向 | `mpjpe_l` 和 body 子集 | 保留 heading-local 语义，新增朝向和 EE |
| Joint tracking | 无 | 全 joint 向量 L2 位置/速度 norm | 不是 callback 主要输出 | 标量关节坐标 MAE/RMSE/p95 |
| Body 子集 | 一组 key bodies | all body、task EE、hands | all、legs、VR points、upper body、feet | 固定 semantic key set 和四个 EE，并记录实际集合 |
| Dynamic tracking | 无 | 只有 joint velocity error | SMPLSim velocity/acceleration 派生指标 | 用真实时间新增 key-body 速度与加速度跟踪 |
| Smoothness | 无 | action/control delta、joint acceleration 和类 jerk 项 | 不是 eval_agent 核心输出 | 只保留物理量纲正确的 joint acceleration 与 jerk |
| Contact | 无 | 多个 reference-contact/force 指标 | 主要用于 termination | 在接触标签、geom mapping、阈值和归一化形成统一契约前不纳入 v2 |
| 聚合 | rollout macro mean/std | 单 rollout 归约与 weighted score | summary micro-average 使长 motion 权重更大 | 等 rollout 权重的 macro aggregation，并保留逐 rollout 行 |
| 排名 | 无 | weighted score 和 success threshold | 可为渲染排序 motion | 不给 controller 排名，不做 composite score |

Spider 的关键差异：

- Spider global error 直接比较世界坐标，因此把初始放置误差和后续 tracking
  drift 混在一起。
- Spider local error 会移除当前 anchor 的完整 SE(3)，包括高度、roll 和
  pitch。v2 只移除 XY 和 heading，保留与重力有关的失败。
- Spider joint position/velocity 对整条 joint vector 取 L2 norm；v2 使用
  单坐标 MAE/RMSE/p95。
- Spider 的类 jerk 项对 joint velocity 做二阶差分，却只除以一次 `dt`。
  v2 对 acceleration 再求一次时间导数，得到预期的 rad/s³。
- 不同 controller 的 action 参数化、scale 和底层 gain 可能不同，因此
  action/control delta 不具有可移植的横向可比性，v2 不纳入它们。

SONIC 值得借鉴的部分是逐 motion outcome、固定语义 body 子集，以及分开的
global/local 诊断。raw `mpjpe_g` 会受到初始世界位置影响，因此没有直接
采用；完整 Procrustes 对齐也不作为主要控制指标，因为它可能掩盖平移、
heading 和 scale 失败。

## 兼容字段

详细 CSV 继续保留下列旧字段：

| 旧字段 | v2 含义 |
| --- | --- |
| `progress` | `completion_ratio` 的别名 |
| `global_root_tracking_error` | start-aligned root XYZ mean 的别名 |
| `global_root_tracking_error_xy` | start-aligned root XY mean 的别名 |
| `local_body_tracking_error` | heading-local key-body position mean 的别名 |
| `mpjpe` | `local_body_tracking_error` 的别名；仅为兼容保留这个名称 |
| `root_final_error_norm` / `root_final_error_xy_norm` | 旧版最终 start-aligned root displacement error |

v2 需要 trajectory NPZ 包含 joint 数组、joint names 和 `sim_time`。因此
`--skip-existing` 会拒绝不能生成完整 v2 指标的旧 trajectory 文件。

## 完整 AMASS-corrected 评测

本地目标 manifest 包含 12,273 条 motion，总 reference 时长 40.60 小时。
一个 seed、五类 controller 共 61,365 次 rollout。应先用小子集 smoke test，
确认无误后省略 `--num-motions` 运行完整数据集。

```bash
ANY4HDMI_CACHE_BUILD_DEVICE=cpu uv run --no-sync python \
  scripts/tracking_experiment/run_tracking_metrics_eval.py \
  --motions-root outputs/any4hdmi_datasets/amass_corrected \
  --policy mimic_lite=checkpoints/mimic-lite/32x8192-huge/policy.yaml \
  --policy heft=checkpoints/heft/pmg/policy.yaml \
  --policy humanoid_gpt=checkpoints/humanoid-gpt/policy.yaml \
  --policy teleopit=checkpoints/teleopit/policy.yaml \
  --policy twist2=checkpoints/twist2/policy.yaml \
  --seeds 0 \
  --initial-pause-s 0 \
  --retention summary-only \
  --output-dir outputs/tracking_eval/amass_corrected_metrics_v2 \
  --skip-existing
```

如果目标是另一个 HEFT 变体，把路径替换成
`checkpoints/heft/wujs/policy.yaml`。controller alias 只是标签，其书写顺序
不会产生排名。

正式 batch 评测默认离线运行：runner 会向 rollout 子进程设置
`HF_HUB_OFFLINE=1`，并使用 `third_party/prebuilt/g1_xmls/` 中经过 checksum
固定的 G1 资源。policy 专属的 `motion.mjcf_path` 仍然具有最高优先级；
`--allow-network-assets` 只是显式的诊断或初次准备开关。

把资源从远程 URI/cache 移到字节完全相同的本地路径，不会使已完成 metrics
失效。应保留现有 checkpoint，只补跑失败或缺失的 rollout key。只有
MJCF/mesh 内容、policy checkpoint/config、motion 数据、simulator 设置、
trajectory schema 或 metrics 实现发生变化时，才需要完整重跑。

完整命令使用 `--retention summary-only`：每条 rollout 结束后立即计算
标量 metrics，持久追加到 `checkpoints/rollout_metrics.jsonl`，再删除已成功
计算 metrics 的 trajectory。正常到达 motion end 的日志会删除；tracking
终止和执行失败的日志会保留用于诊断。`--skip-existing` 会从 metrics
checkpoint 恢复，不重复执行已完成 rollout。

完整评测前先运行固定的十来源子集：

```bash
ANY4HDMI_CACHE_BUILD_DEVICE=cpu uv run --no-sync python \
  scripts/tracking_experiment/run_tracking_metrics_eval.py \
  --motions-root outputs/any4hdmi_datasets/amass_corrected \
  --motion-list scripts/tracking_experiment/motion_lists/amass_corrected_smoke10.txt \
  --policy humanoid_gpt=checkpoints/humanoid-gpt/policy.yaml \
  --seeds 0 \
  --initial-pause-s 0 \
  --retention summary-only \
  --checkpoint-every 1 \
  --output-dir outputs/tracking_eval/amass_corrected_smoke10_humanoid_gpt
```

summary-only 模式的输出契约：

```text
summary.json
  metric_schema
  all_rollouts
  per_controller
  runs

tracking_metrics.json
  metric_schema
  summary
  per_policy_config
  per_controller

checkpoints/rollout_metrics.jsonl
  用于恢复的紧凑持久状态，不是 trajectory archive

runs.csv / failed_runs.csv
  每个请求 rollout 的执行状态
```

日常对比从 `summary.json` 开始；正式比较 controller 前先用 `runs.csv`
审计覆盖率。只有需要逐 motion trajectory 回放或
`tracking_metrics.csv` 详细分析时，才省略 `--retention summary-only`。
