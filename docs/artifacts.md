---
title: Download Artifacts
slug: /reference/artifacts
---

# Download Artifacts

Large runtime artifacts are stored outside git. The versioned
`artifacts/lock.v1.json` records their BCE BOS URIs, byte sizes, and SHA-256
digests. Restore the eight G1 reference policies and runtime assets with:

```bash
uv run python scripts/artifact_tool.py fetch --profile reference
uv run python scripts/artifact_tool.py verify --profile reference
```

Use `--profile benchmark` for the Any4HDMI walk/jump tuning set and
`--profile validation` for the frozen AMASS validation dataset. The latter is
about 1 GB and is intentionally not part of the default download.

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

`checkpoints/` contains Git-tracked policy contracts and BOS-restored ONNX
files. The reference profile covers HEFT PMG/WUJS, Humanoid-GPT, SONIC
low-latency G1, TWIST2, TeleopIT, BFM-0, and WXY-WBC.

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
`third_party/prebuilt/g1_xmls/`. The reference profile restores this tree from
the locked `g1_xmls-a57ffbdf.tar.gz` bundle. The pinned
`g1-mode_13_15.xml` SHA-256 is
`29a7ad71803d37d09f564bb1c9ae15e348a8c82b815c5d1ccbdde3f2f0521513`.
The runtime selects this local copy only when the checksum matches. All
controllers use it as the default physical simulation model; an explicit
policy `motion.mjcf_path` still takes precedence for policy-specific reference
or reward FK.

The tracking batch evaluator disables Hugging Face network access by default.
Use `--allow-network-assets` only for other policy bootstrap or diagnosis, not
for a formal dataset run. This flag does not replace the required local G1
asset.

Do not commit downloaded artifact contents. Publish refreshed assets under a
new versioned BCE BOS prefix, update the lock, and verify a clean round-trip
download before changing the default profile.
