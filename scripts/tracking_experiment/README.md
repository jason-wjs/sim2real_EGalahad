# Integrated MuJoCo Tracking Evaluation

This directory contains the scripts used to evaluate tracking policies in one
MuJoCo process. The evaluator runs the exported policy, the MuJoCo simulation,
and the motion reference in the same Python process, then saves trajectory data
for metric computation.

Run all commands from the `sim2real/` repository root.

## Core Files

- `sim2real/sim_env/integrated_sim2sim.py`
  - Runs one policy on one motion in MuJoCo.
  - Initializes the robot from motion frame 0, waits for `--initial-pause-s`,
    starts policy tracking, and can stop when the motion reaches the last frame.
  - Saves root-only trajectories with `--root-trajectory-output`.
  - Saves root/body/joint trajectories and timestamps with
    `--trajectory-output`; the batch evaluator enables policy-rate-only
    storage.
- `run_tracking_metrics_eval.py`
  - Batch runner for multiple policies, motions, and seeds.
  - Calls `integrated_sim2sim.py` once per `(policy, motion, seed)`.
  - By default writes per-rollout logs, resumable `runs.csv` checkpoints,
    trajectory `.npz` files, detailed `tracking_metrics.csv`, and compact
    `tracking_metrics.json` / `summary.json` aggregates.
  - `--retention summary-only` durably checkpoints each rollout's metric
    scalars, then deletes its trajectory; completed-motion logs are also
    removed, while tracking-termination and execution-failure logs remain.
- `compute_tracking_metrics.py`
  - Compatibility CLI for the public `sim2real.metrics` API.
  - Computes outcome, start-aligned global tracking, heading-local tracking,
    joint-space tracking, dynamic tracking, and smoothness from saved
    trajectory `.npz` files.
- `run_root_final_error_eval.py` and `compute_root_final_error.py`
  - Legacy root-final-displacement-only evaluation path. Keep this for older
    plots, but use `run_tracking_metrics_eval.py` for new tracking comparisons.
- `convert_xrobot_raw_zip_to_any4hdmi.py`
  - Converts XRobot raw G1 motion dumps into any4hdmi `.npz` motion files.
- `convert_to_any4hdmi.py`
  - Converts supported corrected IsaacLab G1, `mjlab_g1_native`, and MuJoCo
    qpos NPZ schemas into an any4hdmi dataset.
- `convert_mjlab_g1_native_to_any4hdmi.py`
  - Backward-compatible wrapper for `--source-format mjlab-g1-native`.
- `visualize_root_trajectory.py` and `plot_root_final_error_bars.py`
  - Helpers for inspecting root trajectories and old root-final-error outputs.

## Metrics

`sim2real.metrics` provides the reusable Python API, while
`compute_tracking_metrics.py` reports one detailed row per rollout and
structured dataset summaries from the command line:

- `outcome`: completion, termination/success rate, and termination reasons;
- `tracking.global_start_aligned`: root, key-body, end-effector, velocity,
  and acceleration tracking after independent initial-pose alignment;
- `tracking.local_heading`: key-body and end-effector pose after removing
  current pelvis XY and yaw while retaining world Z, roll, and pitch;
- `tracking.joint_space`: joint position/velocity MAE, RMSE, and p95;
- `smoothness`: robot joint acceleration and jerk using recorded time in
  seconds.

Legacy fields remain in the detailed CSV as compatibility aliases. See
`docs/tracking_metrics_diff.md` for the exact formulas, body sets, units,
aggregation contract, Spider/SONIC comparison, and full AMASS-corrected
evaluation command.

## Run One Rollout

Use `integrated_sim2sim.py` directly when debugging a single policy/motion pair:

```bash
uv run python -m sim2real.sim_env.integrated_sim2sim \
  --robot g1 \
  --policy-config checkpoints/example_policy/policy.yaml \
  --motion-path ../any4hdmi/output/lafan/motions/example_motion.npz \
  --headless \
  --run-once \
  --initial-pause-s 5.0 \
  --trajectory-output outputs/example_eval/trajectory.npz \
  --seed 0
```

Drop `--headless` to inspect the rollout in the MuJoCo viewer. In viewer mode,
pressing space after the final-frame hold restarts the motion from frame 0.

## Run Batched Tracking Metrics

Evaluate one or more policies over a motion directory:

```bash
uv run python scripts/tracking_experiment/run_tracking_metrics_eval.py \
  --motions-root ../any4hdmi/output/lafan/motions \
  --policy mimic_lite_ppo=checkpoints/mimic_lite_ppo/policy.yaml \
  --policy sonic=checkpoints/sonic/release/g1/policy.yaml \
  --num-motions 40 \
  --seeds 0 1 2 \
  --output-dir outputs/tracking_eval/lafan40
```

Omit `--num-motions` to evaluate every discovered motion. Use it only to select
a deterministic prefix for smoke tests or subsets. The runner checkpoints
`runs.csv` and prints one progress line every 100 rollouts by default; change
that interval with `--checkpoint-every`.

Use the fixed ten-source AMASS-corrected smoke subset before a full run:

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

In `summary-only` mode, add `--skip-existing` to resume from
`checkpoints/rollout_metrics.jsonl`. The checkpoint retains only the compact
per-rollout scalars needed to reconstruct the exact macro-average and standard
deviation; successfully processed trajectories are not retained.

Recompute tables from existing trajectory files without rerunning MuJoCo:

```bash
uv run python scripts/tracking_experiment/run_tracking_metrics_eval.py \
  --motions-root ../any4hdmi/output/lafan/motions \
  --policy mimic_lite_ppo=checkpoints/mimic_lite_ppo/policy.yaml \
  --policy sonic=checkpoints/sonic/release/g1/policy.yaml \
  --num-motions 40 \
  --seeds 0 1 2 \
  --output-dir outputs/tracking_eval/lafan40 \
  --skip-existing
```

Output layout:

```text
outputs/tracking_eval/lafan40/
  runs.csv                  # one rollout status row per policy/motion/seed
  failed_runs.csv
  tracking_metrics.csv      # detailed per-rollout metrics
  tracking_metrics.json     # compact aggregate metrics
  summary.json              # compact policy/run summary
  logs/
    <policy>/
      seed_<seed>/
        <motion_index>_<motion_slug>.log
  trajectories/
    <policy>/
      seed_<seed>/
        <motion_index>_<motion_slug>.npz
```

With `--retention summary-only`, `tracking_metrics.csv` is intentionally
omitted, successfully processed trajectory files are deleted, and the compact
recovery state lives at `checkpoints/rollout_metrics.jsonl`.
`tracking_metrics.json`, `summary.json`, `runs.csv`, and
`failed_runs.csv` remain part of the output contract.

## Compute Metrics From Existing Trajectories

Use this when another script already produced full trajectory `.npz` files.
For large evaluations, pass the run manifest so the shell never expands thousands
of paths into one command line:

```bash
uv run python scripts/tracking_experiment/compute_tracking_metrics.py \
  --manifest outputs/tracking_eval/lafan40/runs.csv \
  --output-csv outputs/tracking_eval/lafan40/tracking_metrics.csv \
  --output-json outputs/tracking_eval/lafan40/tracking_metrics.json
```

Direct paths and glob patterns remain supported for small ad-hoc runs. Standard
stdout contains only progress and aggregate metrics; add `--print-rows` only
when explicitly debugging a small result set.

## Legacy Root Final Error

The older root-final-error path is still available:

```bash
uv run python scripts/tracking_experiment/run_root_final_error_eval.py \
  --no-default-policies \
  --policy mimic_lite_ppo=checkpoints/mimic_lite_ppo/policy.yaml \
  --motions-root ../any4hdmi/output/xrobot_raw_20260524/motions \
  --num-motions 8 \
  --seeds 0 1 2 \
  --output-dir outputs/root_final_error_eval/mimic_lite_ppo
```

This path writes root-only trajectories and computes
`root_final_error_norm` / `root_final_error_xy_norm`; it does not compute
motion progress or local body tracking.

Failed rollouts are recorded and the remaining matrix continues by default.
Use `--fail-fast` when debugging and `--skip-existing` to reuse only
trajectory files that pass the evaluator's integrity check. Full trajectory
files remain on disk because they are the auditable source for recomputing
metrics; budget storage per policy and seed before a large run.

## Convert G1 Motion Datasets

`convert_to_any4hdmi.py` accepts these concrete source schemas:

- `isaaclab-g1-corrected`: 29-joint corrected G1 arrays plus a validated
  `motion.diff.json` contract.
- `mjlab-g1-native`: NPZ files carrying `mjlab_g1_body_names`.
- `mujoco-qpos`: `qpos`, `qpos_names`, and `fps` or `timestep`.
- `auto`: only selects a format when the NPZ has an unambiguous signature.

`isaaclab` and `mujoco` are CLI aliases for the corresponding concrete
formats.

### Inspect corrected source motions directly

Use the diagnostic Viser before conversion when source body transforms need to
be compared with the MuJoCo reconstruction produced by the current converter.
It selects only the first five motions by default and lazily loads the selected
clip:

```bash
uv run --no-sync python scripts/view_raw_motion.py \
  --input /path/to/amass_filtered_0.05_40k-segmented_2k \
  --num-motions 5 \
  --loop
```

The magenta points, skeleton, and axes are the source
`body_pos_w/body_quat_w` values. The robot mesh is reconstructed from the
source pelvis pose and `joint_pos` using the exact assumptions in
`convert_to_any4hdmi.py`. The sidebar reports aggregate FK mismatch for
the selected clip.

Single file:

```bash
uv run --no-sync python scripts/tracking_experiment/convert_to_any4hdmi.py \
  --input /path/to/motion.npz \
  --source-format mjlab-g1-native \
  --out-dir outputs/any4hdmi_datasets/example_clip \
  --dataset-name example_clip
```

Directory (optional smoke with `--max-files`):

```bash
uv run --no-sync python scripts/tracking_experiment/convert_to_any4hdmi.py \
  --input /path/to/amass_filtered_0.05_40k-segmented_2k \
  --source-format isaaclab-g1-corrected \
  --out-dir outputs/any4hdmi_datasets/amass_filtered_0.05_40k-segmented_2k \
  --dataset-name amass_filtered_0.05_40k-segmented_2k \
  --skip-existing \
  --continue-on-error
```

The converter appends one durable record per completed motion to
`conversion_records.jsonl`. It also writes `conversion_report.json`,
`failed_motions.json`, and a manifest covering all output motions, including
files reused by `--skip-existing`.

After conversion, point offline replay or batch eval at the output motions:

```bash
ANY4HDMI_CACHE_BUILD_DEVICE=cpu uv run --no-sync python -m sim2real.sim_env.integrated_sim2sim \
  --robot g1 \
  --policy-config checkpoints/twist2/policy.yaml \
  --motion-path outputs/any4hdmi_datasets/example_clip/motions/motion.npz \
  --headless --run-once --initial-pause-s 1.0 --inference-backend onnx-cpu

ANY4HDMI_CACHE_BUILD_DEVICE=cpu uv run --no-sync python scripts/tracking_experiment/run_tracking_metrics_eval.py \
  --motions-root outputs/any4hdmi_datasets/amass_filtered_0.05_40k-segmented_2k \
  --policy twist2=checkpoints/twist2/policy.yaml \
  --num-motions 8 \
  --seeds 0 \
  --output-dir outputs/tracking_eval/amass_smoke
```
