# SONIC

Source: https://github.com/NVlabs/GR00T-WholeBodyControl

Deployable checkpoints:

- `release/g1/policy.yaml`: released G1 reference-motion policy.
- `release/smpl/policy.yaml`: released SMPL-mode policy.
- `low_latency/g1/policy.yaml`: low-latency G1 policy.
- `low_latency/smpl/policy.yaml`: low-latency SMPL policy.

Notes:

- Every ONNX exposes semantic multi-key inputs. There are no legacy flat-input
  SONIC checkpoints in this tree.
- Every ONNX returns `action[29]` and the encoder's `token[64]`. Current policy
  execution consumes `action`; `token` is retained for future integrations.
- Use the G1 policies with regular `npz` or ZMQ G1 motion streams. Use the SMPL
  policies only with SMPL motion messages, for example Pico retargeting with
  `--publish-smpl`.
