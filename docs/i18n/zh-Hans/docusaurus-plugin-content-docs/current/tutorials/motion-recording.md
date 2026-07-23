# Motion Recording

这个教程把 `sim2real/teleop/pico_retarget_pub.py` 发布的 retargeted G1 motion stream 录成 any4hdmi 的 qpos motion clip。

## 1. 启动 live publisher

```bash
uv --project venv/teleop run sim2real/teleop/pico_retarget_pub.py
```

## 2. 录制 motion stream

```bash
uv run scripts/record_motion.py
```

用 `Ctrl-C` 停止录制并写出数据。

## Output

默认会生成一个时间戳目录，例如 `g1_motion_YYYYMMDD_HHMMSS/`，里面会写出：

- `manifest.json`
- `motions/motion.npz`

这个输出目录就是 any4hdmi dataset root。终端会打印最终输出目录、frame 数、invalid frame 数，以及推断出的 FPS。

## 3. 可选：用 any4hdmi 回看保存的 motion

```bash
uv run scripts/view_motion.py \
  --motion g1_motion_YYYYMMDD_HHMMSS/motions/motion.npz
```

默认 backend 是原生 MuJoCo viewer。远程通过浏览器检查纯 reference 时，
使用 Viser 并循环播放：

```bash
uv run scripts/view_motion.py \
  --motion g1_motion_YYYYMMDD_HHMMSS/motions/motion.npz \
  --viewer viser \
  --loop
```

Viser backend 会打印访问地址，并等待首个浏览器客户端连接后再开始播放；
按 `Ctrl-C` 结束。不加 `--loop` 时会停在最后一帧。

如需浏览一个 any4hdmi 数据集中的所有 motion，可传入 dataset root，而不是
单条 motion：

```bash
uv run scripts/view_motion.py \
  --dataset outputs/any4hdmi_datasets/amass_corrected \
  --viewer viser \
  --loop
```

在 Viser 侧栏的 `Motion dataset` 中，可以直接选择 motion，或切换到上一条、
下一条。每次选择都会从该 motion 所包含的第一帧重新开始播放。

实时 retarget viewer 已经内置在 `sim2real/teleop/pico_retarget_pub.py` 里；它不回放录好的 `.npz` 文件。
