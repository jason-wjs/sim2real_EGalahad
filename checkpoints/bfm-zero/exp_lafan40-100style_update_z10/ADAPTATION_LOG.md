# BFM-Zero exp_lafan40-100style_update_z10 Adaptation Log

## Source

- Remote host: `rp-2` (`zp-nc68`)
- Run directory: `exp_lafan40-100style_update_z10`
- Source code inspected: `BFM-Zero upstream at b87916f`
- Training model: `FBcprAuxAgent` / `FBcprAuxModel`
- Exported actor ONNX: `exported/FBcprAuxModel.onnx`
- Backward encoder ONNX: `exported/backward_encoder.onnx`

## Runtime Layout

- Actor ONNX input: `actor_obs[1,721]`
- Actor ONNX output: `action[1,29]`
- `actor_obs` layout:
  - `state[64] = dof_pos[29] | dof_vel[29] | projected_gravity[3] | base_ang_vel[3]`
  - `last_action[29]`
  - `history_actor[372]`
  - `z[256]`
- `history_actor` layout follows upstream sorted keys:
  - `actions[4,29]`
  - `base_ang_vel[4,3]`
  - `dof_pos[4,29]`
  - `dof_vel[4,29]`
  - `projected_gravity[4,3]`
- History is newest-to-oldest and excludes the current frame until after the observation is computed.
- Policy joint order is HumanoidVerse G1 29DOF order from `robot.dof_names`.
- Effective deploy action scale is `normalize_action_to / normalize_action_from * action_scale * effort_limit / stiffness`.

## Local Artifacts

- Runtime policy YAML: `policy.yaml`
- Runtime policy ONNX: `policy.onnx`
- Build input retained for regenerating the streaming graph: `policy-bfm-zero-exp_lafan40-100style_update_z10-fused.onnx`
- Source configs: `config.yaml`, `config.json`
- The deploy path does not use precomputed latent files or a separate backward encoder ONNX.
- The YAML sets `motion.mjcf_path` to `mjcf/g1_for_reward_inference.xml`, copied from BFM-Zero `minimal_inference`, so any4hdmi builds the MotionData cache with the same body tree as the source inference code.

## Verification

Commands run from the sim2real repository root:

Current smoke command:

```bash
MUJOCO_GL=egl uv run sim2real/sim_env/integrated_sim2sim.py --robot g1 --policy-config checkpoints/bfm-zero/exp_lafan40-100style_update_z10/policy.yaml --motion-path ../any4hdmi/output/g1/lafan/motions/fallAndGetUp1_subject5.npz --headless --run-once --initial-pause-s 0 --seed 0 --max-runtime-s 4 --root-trajectory-output outputs/bfm_zero_exp_lafan40_100style_update_z10/fallAndGetUp1_subject5_fused_window_motion_data_bfm_xml_smoke_root.npz
```

Results:

- any4hdmi loaded `.cache/motion/qpos_online_v2/5c9de8dfe543a983`, whose `cache_meta.json` pointed to the BFM minimal XML.
- The cached body tree has 42 bodies, includes `head_link`, and does not include the older any4hdmi toe links.
- Integrated sim2sim ran headless and saved `outputs/bfm_zero_exp_lafan40_100style_update_z10/fallAndGetUp1_subject5_fused_window_motion_data_bfm_xml_smoke_root.npz`.
- Observed `prepare_obs` was about 14-17 ms for the smoke, not the old per-step MuJoCo FK path around 90 ms.

## 2026-06-29 Obs/Fused ONNX Fixes

- Upstream `obs_buf_dict_raw` is already scaled by observation scales. The sim2real observations therefore apply `base_ang_vel * 0.25`.
- Upstream stores previous actions after `normalize_action_to / normalize_action_from = 5` and clip to `[-5, 5]`. The sim2real last-action and history observations now store `clip(raw_action * 5, -5, 5)`, while the deploy PD target still uses raw ONNX action times YAML `action_scale`.
- The deploy path uses fused-window-streaming only. Precomputed `z` files and per-window MuJoCo `mj_forward` FK have been removed from the runtime code.
- Fused-window ONNX I/O:
  - inputs: `actor_state[1,64]`, `last_action[1,29]`, `history_actor[1,372]`, `encoder_state[8,64]`, `privileged_state[8,463]`, `encoder_window_weight[8,1]`
  - output: `action[1,29]`
- `encoder_state` and `privileged_state` are generated directly from any4hdmi `MotionData` built with the BFM XML manifest override.
