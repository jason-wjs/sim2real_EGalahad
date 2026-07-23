# BFM-Zero

Source: https://github.com/LeCAR-Lab/BFM-Zero

Deployable checkpoint:

- `exp_lafan40-100style_update_z10/policy.yaml`
- `exp_lafan40-100style_update_z10/policy.onnx`
- `exp_lafan40-100style_update_z10/mjcf/g1_for_reward_inference.xml`

Notes:

- This policy needs the BFM-Zero MJCF override for motion FK. Direct `npz`
  tracking reads it from `policy.yaml`.
- For ZMQ motion streams, pass the same MJCF override to the publisher:
  `--mjcf-path checkpoints/bfm-zero/exp_lafan40-100style_update_z10/mjcf/g1_for_reward_inference.xml`.
- Real-robot smoke tests looked stable with the override. Without it, body-name
  and FK mismatches can produce incorrect motion observations.

Example:

```bash
uv run sim2real/rl_policy/tracking.py --robot g1 --policy_config checkpoints/bfm-zero/exp_lafan40-100style_update_z10/policy.yaml --inference_backend onnx-cpu --robot_io inline --motion_backend npz --motion_path ../any4hdmi/output/g1/lafan/motions/walk1_subject1.npz
```
