# TWIST2 checkpoint

Source: https://github.com/amazon-far/TWIST2

Upstream commit: `d5c7108e9ef82d1b8770e5b692f27a1294f3aa8a`

SHA256:

```text
2d1fb3a31e4e967f70ecfefc3ad1e7b2ac491677068b89f60b565a94e7735061
```

Copied checkpoint:

- `policy.onnx`
- `policy.yaml`
- `policy.json`

ONNX signature:

- inputs: `current_motion[35]`, `proprioception[92]`,
  `observation_history[1270]`, and `future_motion[35]`
- output: `action[29]`

TWIST2 observation layout from the upstream training/export code:

- `num_actions = 29`
- `n_mimic_obs = 35`
- `n_proprio = 92`
- `n_obs_single = 127`
- `history_len = 10`
- `num_observations = 127 * 11 + 35 = 1432`

The upstream low-level deployment code builds:

```text
obs_buf = concat(current_mimic_plus_proprio, history_10_frames, future_mimic)
```

The sim2real adapter lives in
`sim2real/rl_policy/observations/twist2.py`. For integrated sim2sim it builds
TWIST2's mimic command from the configured motion dataset and exposes each
semantic input as its own observation group.

Deploy note:

- Use the normal G1 motion stream from `npz` or ZMQ.
- Validate in sim before real-robot use; this checkpoint has not had the same
  recent real-robot pass as BFM-Zero, HEFT, and TeleopIT.

Example:

```bash
uv run sim2real/rl_policy/tracking.py --robot g1 --policy_config checkpoints/twist2/policy.yaml --inference_backend onnx-cpu --robot_io inline --motion_backend zmq
```
