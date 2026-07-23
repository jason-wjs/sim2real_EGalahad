# SONIC low-latency policies

These are complete, mode-specific ONNX policies composed from NVIDIA's
low-latency `model_encoder.onnx` and `model_decoder.onnx` release.

| Mode | ONNX inputs | Semantic horizon |
| --- | --- | --- |
| G1 | `g1_input[640]`, `proprioception[930]` | 10 frames at 20 ms |
| SMPL | `smpl_input[336]`, `proprioception[930]` | 4 frames at 20 ms |

Both models return `action[29]` and the encoder's unbatched `token[64]`.
Current policy execution consumes only `action`.

The mode selector, zero-filled inactive encoder fields, encoder/decoder bridge,
and batch dimensions are embedded in each ONNX. The runtime does not need to
know the released encoder's combined 1247D layout.

## Validation commands

Run the G1 policy in integrated sim2sim:

```bash
uv run sim2real/sim_env/integrated_sim2sim.py \
  --policy-config checkpoints/sonic/low_latency/g1/policy.yaml \
  --motion-path ../any4hdmi/output/root_tracking_test/motions/forward_1.npz \
  --headless --run-once --initial-pause-s 0 --inference-backend onnx-cpu
```

Replay NVIDIA's paired SMPL/G1 walk sample for the SMPL policy:

```bash
uv run sim2real/teleop/sonic_smpl_pkl_pub.py \
  --smpl-path ../GR00T-WholeBodyControl/sample_data/smpl_filtered/walk_forward_amateur_001__A001.pkl \
  --robot-path ../GR00T-WholeBodyControl/sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl
```

The publisher converts the official 50 Hz SMPL joint/root data into the
sim2real SMPL stream and resamples the paired 30 Hz G1 wrist targets to 50 Hz.
