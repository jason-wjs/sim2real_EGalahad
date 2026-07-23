---
title: Download Artifacts
slug: /reference/artifacts
---

# Download Artifacts

Large runtime artifacts are stored outside git. Download the shared
[sim2real artifacts](https://drive.google.com/drive/folders/1lrPyiiy7anyG3P4wHNIQQQlydboLPd9e)
folder before running deploy or onboard setup commands.

After download, the repo root should contain:

```text
checkpoints/
third_party/
  wheels/
  prebuilt/
    g1_xmls/
      g1-mode_13_15.xml
      meshes/
```

`checkpoints/` contains exported policy YAML and ONNX files. Tutorial commands
assume these paths exist locally, for example
`checkpoints/mimic-lite/32x8192-huge/policy.yaml`.

`third_party/wheels/` contains deployment-only wheels that are resolved by
`uv` through `find-links = ["third_party/wheels"]`. On G1, use:

```bash
uv sync --group g1
```

Use the GPU group when the onboard environment should install the JetPack 6
ONNX Runtime GPU wheel:

```bash
uv sync --group g1-gpu
```

`third_party/prebuilt/` contains prebuilt packages that are not Python wheels.
The JetPack 5 Pico onboard setup expects:

```text
third_party/prebuilt/jetpack5-aarch64/
```

Formal G1 rollout evaluation expects the canonical simulation asset under
`third_party/prebuilt/g1_xmls/`. For artifact distribution, package this tree
as `g1_xmls-a57ffbdf.zip` with `g1_xmls/` as its top-level directory. The
pinned `g1-mode_13_15.xml` SHA-256 is
`29a7ad71803d37d09f564bb1c9ae15e348a8c82b815c5d1ccbdde3f2f0521513`.
The runtime selects this local copy only when the checksum matches. All
controllers use it as the default physical simulation model; an explicit
policy `motion.mjcf_path` still takes precedence for policy-specific reference
or reward FK.

The tracking batch evaluator disables Hugging Face network access by default.
Use `--allow-network-assets` only for other policy bootstrap or diagnosis, not
for a formal dataset run. This flag does not replace the required local G1
asset.

Do not commit downloaded artifact contents. Keep refreshed artifacts in the
Google Drive folder and keep local paths matching the layout above.
