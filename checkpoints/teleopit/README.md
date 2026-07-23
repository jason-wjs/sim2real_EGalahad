# TeleopIT

Source: external TeleopIT checkpoint adapted into sim2real.

Files:

- `policy.yaml`
- `policy.onnx`
- `policy.json`

Notes:

- Uses the normal G1 motion stream from `npz` or ZMQ.
- Real-robot tests showed good walking behavior.
- Aggressive low or double-knee kneeling motions caused strong joint noise; keep
  first robot tests conservative.

Example:

```bash
uv run sim2real/rl_policy/tracking.py --robot g1 --policy_config checkpoints/teleopit/policy.yaml --inference_backend onnx-cpu --robot_io inline --motion_backend zmq
```
