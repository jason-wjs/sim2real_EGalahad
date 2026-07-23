# Runtime artifacts

Git stores policy contracts and provenance. BCE BOS stores model binaries,
runtime meshes, and motion data. `lock.v1.json` pins byte sizes and SHA-256
digests for every reference artifact.

```bash
# Eight G1 policy binaries plus the G1 and BFM runtime assets.
uv run python scripts/artifact_tool.py fetch --profile reference

# The two Any4HDMI TTS tuning motions: walk and jump.
uv run python scripts/artifact_tool.py fetch --profile benchmark

# The frozen AMASS validation dataset (about 1 GB).
uv run python scripts/artifact_tool.py fetch --profile validation

# The pre-matrix Mimic-Lite checkpoint, when reproducing legacy commands.
uv run python scripts/artifact_tool.py fetch --profile legacy

uv run python scripts/artifact_tool.py verify --profile reference
```

The tool requires an authenticated `bcecmd` installation. Downloads are
checksum-verified before installation. Archive contents are also checked with
a deterministic tree digest after extraction.
