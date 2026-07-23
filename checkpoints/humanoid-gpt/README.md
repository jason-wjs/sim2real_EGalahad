# Humanoid-GPT

Source: https://github.com/GalaxyGeneralRobotics/Humanoid-GPT

Files:

- `policy.yaml`
- `policy.onnx`
- `policy.json`

Notes:

- Uses the normal G1 motion stream from `npz` or ZMQ.
- The policy follows the stream directly; default-pose and pause behavior should
  be handled by the motion source.
- Real-robot testing showed hand and leg response, but locomotion translation was
  not reliable. Validate in sim before robot runs.

Example:

```bash
uv run sim2real/rl_policy/tracking.py --robot g1 --policy_config checkpoints/humanoid-gpt/policy.yaml --inference_backend onnx-cpu --robot_io inline --motion_backend zmq
```
