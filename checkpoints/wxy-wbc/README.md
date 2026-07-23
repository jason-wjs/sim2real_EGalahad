# WXY WBC

Source checkpoint: `WXY_8000/model_8000.pt`, iteration 8000.

Source implementation:

- `spider.tasks.g1_wbc.policy.WbcActor`
- `spider.tasks.g1_wbc.obs.G1WbcObservationBuilder`

Files:

- `policy.onnx`
- `policy.yaml`
- `policy.json`
- `source-metadata.json`

ONNX signature:

- `motion_command[1, 241]`
- `proprioception[1, 645]`
- `action[1, 29]`

The graph preserves the source actor and its embedded observation normalizer.
Its original flat `obs[1, 886]` input is assembled inside the graph from the
two semantic inputs above.

Deploy notes:

- Uses standard G1 any4hdmi reference motions through either NPZ or ZMQ.
- Uses only the current reference frame and five-frame oldest-to-newest
  observation histories.
- Reconstructs robot wrist and ankle poses from measured joints with the
  standard sim2real G1 MuJoCo model.
- The first observation backfills every history slot with the current value.
- Paused motion zeros reference joint and torso angular velocity while keeping
  the policy and observation histories active.

Example:

```bash
uv run sim2real/sim_env/integrated_sim2sim.py \
  --robot g1 \
  --policy_config checkpoints/wxy-wbc/policy.yaml \
  --motion_path hf://elijahgalahad/any4hdmi-g1-lafan/motions/walk1_subject1.npz \
  --inference_backend onnx-cpu
```

Robot deployment with a realtime motion stream:

```bash
uv run sim2real/rl_policy/tracking.py \
  --robot g1 \
  --policy_config checkpoints/wxy-wbc/policy.yaml \
  --inference_backend onnx-cpu \
  --robot_io inline \
  --motion_backend zmq
```
