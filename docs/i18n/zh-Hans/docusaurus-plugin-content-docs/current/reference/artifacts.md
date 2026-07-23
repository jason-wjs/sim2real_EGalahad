---
title: Download Artifacts
slug: /reference/artifacts
---

# Download Artifacts

大文件不放在 git 里。版本化的 `artifacts/lock.v1.json` 记录 BCE BOS URI、
字节数和 SHA-256。恢复八个 G1 reference policy 及其 runtime 资产：

```bash
uv run python scripts/artifact_tool.py fetch --profile reference
uv run python scripts/artifact_tool.py verify --profile reference
```

Any4HDMI walk/jump 调参集使用 `--profile benchmark`；冻结的 AMASS 验证集使用
`--profile validation`。后者约 1 GB，因此不会包含在默认下载中。

下载后，repo 根目录应该长这样：

```text
checkpoints/
third_party/
  wheels/
  prebuilt/
    g1_xmls/
      g1-mode_13_15.xml
      meshes/
```

`checkpoints/` 里同时放 Git 跟踪的 policy contract 和从 BOS 恢复的 ONNX。
reference profile 覆盖 HEFT PMG/WUJS、Humanoid-GPT、SONIC low-latency G1、
TWIST2、TeleopIT、BFM-0 和 WXY-WBC。

`third_party/wheels/` 里放部署用 wheel。`uv` 会通过
`find-links = ["third_party/wheels"]` 解析这些包。G1 上安装基础真机依赖：

```bash
uv sync --group g1
```

如果 onboard 环境还要安装 JetPack 6 的 ONNX Runtime GPU wheel，用：

```bash
uv sync --group g1-gpu
```

`third_party/prebuilt/` 里放非 Python wheel 的预编译包。JetPack 5 的 Pico onboard
setup 需要：

```text
third_party/prebuilt/jetpack5-aarch64/
```

正式 G1 rollout 评测要求公共仿真资源位于
`third_party/prebuilt/g1_xmls/`。reference profile 会从锁定的
`g1_xmls-a57ffbdf.tar.gz` bundle 恢复这棵目录。固定版本
`g1-mode_13_15.xml` 的 SHA-256 是
`29a7ad71803d37d09f564bb1c9ae15e348a8c82b815c5d1ccbdde3f2f0521513`。
运行时仅在 checksum 匹配时选择本地副本。所有 controller 默认使用它作为
物理仿真模型；如果 policy 显式设置了 `motion.mjcf_path`，该设置仍然优先，
用于 policy 专属的 reference 或 reward FK。

tracking batch evaluator 默认禁用 Hugging Face 网络访问。
`--allow-network-assets` 只应用于其他 policy 的初次准备或诊断，不应用于正式
数据集评测；该开关不能替代必需的本地 G1 资源。

不要把下载下来的 artifact 内容提交进 git。更新时应发布到新的版本化 BCE BOS
前缀，更新 lock，并在切换默认 profile 前完成一次 clean round-trip 下载校验。
