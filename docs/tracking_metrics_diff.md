---
title: Unified Tracking Metrics v2
slug: /reference/tracking-metrics-diff
---

# Unified Tracking Metrics v2

This document records the metric contract implemented by
`scripts/tracking_experiment/compute_tracking_metrics.py` and the differences
from the previous repository metrics, Spider G1 WBC metrics, and SONIC
evaluation metrics. The target use case is a fair, auditable comparison of
multiple controllers on the same single motion or motion dataset.

The v2 hierarchy is:

```text
outcome
tracking
├── global_start_aligned
├── local_heading
└── joint_space
smoothness
metadata
```

It intentionally does not calculate a composite score or controller ranking.
The result preserves the individual measurements so a human can compare
controllers without hiding trade-offs behind weights.

## Coordinate-frame contract

Let \(r(0)\) and \(q_r(0)\) be the initial root position and orientation of one
trajectory. Robot and reference are transformed independently using their own
initial root states.

### Global start-aligned

For body position \(p_i(t)\) and orientation \(q_i(t)\):

```text
p_i^SA(t) = R(q_r(0))^T [p_i^W(t) - r^W(0)]
q_i^SA(t) = q_r(0)^-1 ⊗ q_i^W(t)
```

The same transform is applied separately to the robot and reference before
their difference is measured. This removes an arbitrary initial world
translation and full initial root orientation, while preserving all subsequent
root drift, turning, height change, and whole-body trajectory error.

This is the meaning of `global` in v2. It is not raw-world MPJPE.

### Local heading

At every frame, robot and reference each use their own current pelvis position
and projected heading:

```text
a(t) = [pelvis_x(t), pelvis_y(t), 0]
h(t) = projected_yaw(pelvis_orientation(t))

p_i^LH(t) = R(h(t))^T [p_i^W(t) - a(t)]
q_i^LH(t) = h(t)^-1 ⊗ q_i^W(t)
```

Only pelvis XY translation and yaw are removed. World Z, pelvis height, roll,
pitch, and the gravity direction remain observable. Therefore this frame
measures pose quality independent of global horizontal drift and heading
without hiding falls or height errors.

### Joint space

Robot and reference joints are matched by name and stored in the canonical G1
robot order. Position and velocity errors are reduced per scalar joint
coordinate, rather than taking one L2 norm over the complete joint vector.
This keeps the values interpretable and avoids changing the scale merely by
adding joints.

## Evaluation window and outcome

Tracking metrics use the pre-failure window. A rollout terminates
analytically at the first confirmed condition:

| Reason | Threshold | Consecutive policy frames |
| --- | ---: | ---: |
| `root_ori_error` | torso orientation error >= 1.2 rad | 25 |
| `body_pos_error` | maximum heading-local key-body position error >= 0.4 m | 5 |
| `body_ori_error` | maximum heading-local key-body orientation error >= 1.2 rad | 5 |

The frame that confirms a failure is excluded from tracking and smoothness
metrics. If the recorded trajectory does not reach the final reference frame,
it is marked `truncated`. Otherwise it ends with `motion_end`.

Every tracking result must be read together with:

- `completion_ratio`: reference-frame progress at the first confirmed
  failure, or 1.0 at motion end;
- `termination_rate` and `success_rate`: dataset rates derived from the
  analytical termination state;
- `termination_reasons`: counts by reason;
- `runs.failed`: subprocess or invalid-trajectory failures, which are not
  silently mixed into tracking-error averages.

This pairing prevents an early-failing controller from looking good merely
because its error is averaged over a short easy prefix.

## Body and end-effector sets

The recorded body layout now comes from the G1 robot configuration and is
independent of controller YAML. Metrics use the intersection shared by the
robot model and reference motion.

The key-body patterns select:

- pelvis and torso;
- left/right hip yaw, knee, and toe links;
- left/right shoulder yaw, elbow, and wrist yaw links.

The end-effectors are the left/right toe and wrist yaw links. The exact body
and joint sets observed in a run are stored in `metric_schema`; this makes
unexpected selection differences auditable.

## Metric catalog

All orientation errors use quaternion geodesic angle in radians. Position
errors are Euclidean distances. For each rollout, `mean` and `p95` reduce
all valid frame/body samples in the evaluation window.

### Global start-aligned tracking

| JSON metric | Per-rollout CSV field | Unit |
| --- | --- | --- |
| root position XYZ mean/p95 | `global_root_pos_xyz_{mean,p95}_m` | m |
| root position XY mean/p95 | `global_root_pos_xy_{mean,p95}_m` | m |
| root orientation mean/p95 | `global_root_ori_{mean,p95}_rad` | rad |
| key-body position mean/p95 | `global_key_body_pos_{mean,p95}_m` | m |
| key-body orientation mean/p95 | `global_key_body_ori_{mean,p95}_rad` | rad |
| end-effector position mean/p95 | `global_end_effector_pos_{mean,p95}_m` | m |
| end-effector orientation mean/p95 | `global_end_effector_ori_{mean,p95}_rad` | rad |
| key-body velocity tracking mean/p95 | `global_key_body_vel_error_{mean,p95}_mps` | m/s |
| key-body acceleration tracking mean/p95 | `global_key_body_acc_error_{mean,p95}_mps2` | m/s² |

Velocity and acceleration are finite differences of start-aligned key-body
positions. Every derivative uses the recorded `sim_time`; no fixed frequency
is assumed inside the metric implementation.

### Heading-local tracking

| JSON metric | Per-rollout CSV field | Unit |
| --- | --- | --- |
| key-body position mean/p95 | `local_key_body_pos_{mean,p95}_m` | m |
| key-body orientation mean/p95 | `local_key_body_ori_{mean,p95}_rad` | rad |
| end-effector position mean/p95 | `local_end_effector_pos_{mean,p95}_m` | m |
| end-effector orientation mean/p95 | `local_end_effector_ori_{mean,p95}_rad` | rad |

### Joint-space tracking

| JSON metric | Per-rollout CSV field | Unit |
| --- | --- | --- |
| position MAE/RMSE/p95 absolute error | `joint_pos_{mae,rmse,p95_abs}_rad` | rad |
| velocity MAE/RMSE/p95 absolute error | `joint_vel_{mae,rmse,p95_abs}_rad_s` | rad/s |

MAE is the primary typical-error quantity, RMSE emphasizes larger errors, and
p95 exposes tail behavior without using a single unstable maximum.

### Smoothness

| JSON metric | Per-rollout CSV field | Unit |
| --- | --- | --- |
| robot joint acceleration RMS/p95 absolute | `joint_acc_{rms,p95_abs}_rad_s2` | rad/s² |
| robot joint jerk RMS/p95 absolute | `joint_jerk_{rms,p95_abs}_rad_s3` | rad/s³ |

The definitions are:

```text
joint_acceleration = d(qdot) / dt
joint_jerk         = d(joint_acceleration) / dt
```

Smoothness is intentionally reported separately from reference tracking. A
highly dynamic reference can require high acceleration, so these values should
be interpreted alongside dynamic tracking error.

## Dataset aggregation

One row in `tracking_metrics.csv` represents one
`(controller, motion, seed)` rollout. Each rollout first reduces its own
frames to scalar means or p95 values. Dataset summaries then macro-average
those rollout scalars, so every rollout has equal weight regardless of motion
duration.

Each summary leaf contains:

```json
{
  "mean": 0.0,
  "std": 0.0,
  "valid_count": 1
}
```

`std` is population standard deviation across rollouts. A derivative that is
undefined for a very short trajectory is written as JSON `null` in the
per-rollout result and excluded from aggregation; `valid_count` exposes the
remaining denominator. JSON never emits non-standard `NaN`.

For a summary metric whose name contains `p95`, the reported `mean` is the
mean of the per-rollout p95 values. It is not a pooled frame-level p95.

## Source comparison and decisions

| Area | Previous repository metrics | Spider G1 WBC | SONIC evaluation | Unified v2 decision |
| --- | --- | --- | --- | --- |
| Outcome | progress and post-hoc failure | thresholded score/success | terminated, progress, success rate | keep explicit completion, success/termination rates and reasons; no score |
| Global position | start-aligned root trajectory | raw-world root/body/EE distance | `mpjpe_g` raw selected-body distance | retain start alignment; expand to root, key body, and EE |
| Root orientation | used for failure, not reported | raw-world root quaternion error | not a primary reported pose metric | add start-aligned geodesic mean/p95 |
| Local pose | heading-local key-body position | full current anchor SE(3)-local position/orientation | `mpjpe_l` and body subsets | retain heading-local semantics; add orientation and EE |
| Joint tracking | absent | full joint-vector L2 position/velocity norm | not the primary callback output | add scalar-coordinate MAE/RMSE/p95 |
| Body subsets | one key-body set | all bodies, task EE, hands | all, legs, VR points, upper body, feet | keep a fixed semantic key set plus explicit four EEs; record exact sets |
| Dynamic tracking | absent | joint velocity error only | SMPLSim velocity/acceleration-derived metrics | add key-body velocity and acceleration tracking with true time |
| Smoothness | absent | action/control delta, joint acceleration and jerk-like terms | not central to eval_agent output | add physically dimensioned joint acceleration and jerk only |
| Contact | absent | many reference-contact/force metrics | termination-oriented | exclude until contact labels, geom mapping, thresholds, and normalization share a validated contract |
| Aggregation | rollout macro mean/std | single-rollout reductions and weighted score | summary micro-average weights longer clips more | use equal-rollout macro aggregation and retain detailed rows |
| Ranking | none | weighted score and success thresholds | can sort motions for rendering | no controller ranking or composite score |

Important Spider differences:

- Spider global errors directly compare world coordinates, so they mix initial
  placement error with subsequent tracking drift.
- Spider local errors remove the anchor's full current SE(3), including height,
  roll, and pitch. V2 removes only XY and heading so gravity-related failures
  remain visible.
- Spider joint position/velocity metrics take an L2 norm over the full joint
  vector. V2 reports per-coordinate MAE/RMSE/p95.
- Spider computes its jerk-like second difference of joint velocity with only
  one division by `dt`. V2 differentiates acceleration again, giving the
  expected rad/s³ unit.
- Action and control deltas are not portable across controllers with different
  action parameterizations, scales, or low-level gains, so v2 does not compare
  them.

Useful SONIC ideas retained are per-motion outcome reporting, fixed semantic
body subsets, and separate global/local diagnostics. Raw `mpjpe_g` is not
adopted because it is sensitive to initial world placement; full
Procrustes-aligned pose error is also not a primary control metric because it
can hide translation, heading, and scale failures.

## Compatibility fields

The detailed CSV retains these old names:

| Old field | v2 meaning |
| --- | --- |
| `progress` | alias of `completion_ratio` |
| `global_root_tracking_error` | alias of start-aligned root XYZ mean |
| `global_root_tracking_error_xy` | alias of start-aligned root XY mean |
| `local_body_tracking_error` | alias of heading-local key-body position mean |
| `mpjpe` | alias of `local_body_tracking_error`; the name is retained only for compatibility |
| `root_final_error_norm` / `root_final_error_xy_norm` | legacy final start-aligned root displacement errors |

New v2 metric computation requires joint arrays, joint names, and
`sim_time` in trajectory NPZ files. Consequently, `--skip-existing` rejects
older trajectory files that cannot produce the complete v2 metric set.

## Full AMASS-corrected evaluation

The local target manifest contains 12,273 motions (40.60 reference hours). One
seed across five controller families produces 61,365 rollouts. Run a small
smoke subset first, then omit `--num-motions` for the complete dataset.

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

Use `checkpoints/heft/wujs/policy.yaml` instead if that is the intended HEFT
variant. Controller aliases are labels only; no ranking follows their order.

Formal batch evaluation is offline by default: the runner exports
`HF_HUB_OFFLINE=1` to rollout subprocesses and uses the checksum-pinned G1
asset from `third_party/prebuilt/g1_xmls/`. A policy-specific
`motion.mjcf_path` remains authoritative. `--allow-network-assets` is an
explicit diagnostic/bootstrap escape hatch.

Moving an asset from a remote URI/cache to a byte-identical local path does
not invalidate completed metrics. Preserve existing checkpoints and rerun only
failed or missing rollout keys. A complete rerun is required only when the
MJCF/mesh contents, policy checkpoint/configuration, motion data, simulator
settings, trajectory schema, or metric implementation changes.

The full command uses `--retention summary-only`: after each rollout, the
runner computes its scalar metrics, durably appends them to
`checkpoints/rollout_metrics.jsonl`, and deletes the successfully processed
trajectory. Logs for motion-end successes are deleted; tracking-termination and
execution-failure logs remain for diagnosis. `--skip-existing` resumes from
the metric checkpoint without rerunning completed rollouts.

Before the full run, use the fixed ten-source subset:

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

Output contract in summary-only mode:

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
  compact durable recovery state; not a trajectory archive

runs.csv / failed_runs.csv
  execution status for every requested rollout
```

`summary.json` is the normal human entry point. Use `runs.csv` to audit
coverage before comparing controllers. Omit `--retention summary-only` only
when motion-level trajectory replay or detailed `tracking_metrics.csv`
analysis is required.
