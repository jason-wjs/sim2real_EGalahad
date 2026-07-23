# sim2real

root project 负责 inference、tracking policy，以及 MuJoCo 的 sim / sim2real runtime。Pico / XR teleoperation 工具请使用 `venv/teleop`。

English version: [README.md](./README.md)

Full documentation: [https://egalahad.github.io/sim2real/](https://egalahad.github.io/sim2real/)

如果你在找 HDMI 的部署栈，请看 [hdmi tag](https://github.com/EGalahad/sim2real/tree/hdmi)。

## Runtime Artifacts

大文件不放在 git 里。配置好 `bcecmd` 后，从 BCE BOS 恢复锁定的 G1
reference 资产：

```bash
uv run python scripts/artifact_tool.py fetch --profile reference
uv run python scripts/artifact_tool.py verify --profile reference
```

目录结构、benchmark/validation profile 和 onboard 依赖说明见
[Download Artifacts](./docs/artifacts.md)。

## 快速开始

```bash
uv sync
uv run python scripts/artifact_tool.py fetch --profile reference
```

运行离线动作跟踪（sim2sim）：

```bash
uv run sim2real/sim_env/base_sim.py --robot g1
uv run sim2real/rl_policy/tracking.py \
  --robot g1 \
  --policy_config checkpoints/humanoid-gpt/policy.yaml \
  --motion_path hf://elijahgalahad/any4hdmi-g1-lafan/motions/walk1_subject1.npz
```

两个进程都启动后，在 policy 终端按 `]` 开始跟踪，然后打开 `base_sim.py` 打印出来的 mjviser URL。虚拟 gantry / elastic band 的开关和长度在 viewer UI 里调。

## Migrating to sim2real

这个 repo 内置了一个 Codex skill，用来把外部训练 codebase 里的 policy 适配到 `sim2real`：

```text
skills/adapt-policy-to-sim2real
```

已经转好的 checkpoint 二进制统一通过上面的 BCE BOS reference profile
分发。

目前已经支持的 adapted checkpoint：

- BFM-Zero: `checkpoints/bfm-zero/exp_lafan40-100style_update_z10/policy.yaml`
- HEFT: `checkpoints/heft/pmg/policy.yaml`, `checkpoints/heft/wujs/policy.yaml`
- Humanoid-GPT: `checkpoints/humanoid-gpt/policy.yaml`
- SONIC low-latency G1: `checkpoints/sonic/low_latency/g1/policy.yaml`
- TeleopIT: `checkpoints/teleopit/policy.yaml`
- TWIST2: `checkpoints/twist2/policy.yaml`
- WXY WBC: `checkpoints/wxy-wbc/policy.yaml`

安装到本机 Codex skills 目录：

```bash
mkdir -p ~/.codex/skills
cp -r skills/adapt-policy-to-sim2real ~/.codex/skills/
```

安装后重新打开一个 Codex session，即可通过 policy adaptation 相关请求触发；也可以显式提到 `adapt-policy-to-sim2real`。

## 下一步

- [文档首页](https://egalahad.github.io/sim2real/zh-Hans/)
- [快速上手](https://egalahad.github.io/sim2real/zh-Hans/getting-started/overview)
- [Root Project Setup](https://egalahad.github.io/sim2real/zh-Hans/getting-started/root-project)
- [离线动作跟踪教程](https://egalahad.github.io/sim2real/zh-Hans/tutorials/offline-motion-tracking)
- [Pico Teleoperation 教程](https://egalahad.github.io/sim2real/zh-Hans/tutorials/pico-teleoperation)

## Citation

如果 sim2real 对你的研究有所帮助，请引用：

```bibtex
@misc{sim2real2026,
  author       = {{RoboParty Lab Team}},
  title        = {sim2real: A Lightweight and Modular Sim2sim and Sim2real Deployment Stack},
  year         = {2026},
  howpublished = {\url{https://github.com/EGalahad/sim2real}},
  note         = {Documentation: \url{https://egalahad.github.io/sim2real/}}
}
```
