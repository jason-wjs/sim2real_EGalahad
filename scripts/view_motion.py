#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import mujoco
from any4hdmi.core.format import find_dataset_root, load_manifest, load_motion
from mjhub import temp_mjcf_with_floor
from mujoco import viewer as mujoco_viewer
from tqdm import tqdm

from sim2real.utils.mjviser_viewer import MjviserMujocoViewer


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a qpos-only any4hdmi motion with MuJoCo.")
    parser.add_argument(
        "--motion",
        required=True,
        help="Path to a converted motion .npz file. The dataset root is inferred from manifest.json.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Playback FPS override. Defaults to 1 / manifest.timestep.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start frame index.")
    parser.add_argument("--end", type=int, default=-1, help="End frame index.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    parser.add_argument("--loop", action="store_true", help="Loop playback.")
    parser.add_argument(
        "--viewer",
        choices=("mujoco", "viser"),
        default="mujoco",
        help="Interactive viewer backend (default: mujoco).",
    )
    parser.add_argument("--headless", action="store_true", help="Run without opening a viewer.")
    return parser.parse_args(argv)


def _iter_frame_indices(length: int, start: int, end: int, stride: int) -> range:
    resolved_end = end if end >= 0 else length
    return range(start, min(length, resolved_end), max(1, stride))


def _apply_qpos_frame(data: mujoco.MjData, qpos_frame) -> None:
    data.qpos[:] = qpos_frame
    data.qvel[:] = 0.0


def _tracked_body_id(model: mujoco.MjModel) -> int | None:
    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"))
    return body_id if body_id >= 0 else None


def _sleep_until(deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining > 0.0:
        time.sleep(remaining)


def _play_viser(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos,
    frame_indices: list[int],
    frame_dt: float,
    *,
    loop: bool,
) -> None:
    _apply_qpos_frame(data, qpos[frame_indices[0]])
    mujoco.mj_forward(model, data)
    viewer = MjviserMujocoViewer(
        model,
        data,
        label="any4hdmi-reference-viewer",
        tracked_body_id=_tracked_body_id(model),
    )
    try:
        while viewer.is_running() and not viewer.has_clients():
            time.sleep(0.05)

        while viewer.is_running():
            deadline = time.monotonic()
            for frame_idx in frame_indices:
                if not viewer.is_running():
                    break
                _apply_qpos_frame(data, qpos[frame_idx])
                mujoco.mj_forward(model, data)
                viewer.sync()
                deadline += frame_dt
                _sleep_until(deadline)
            if not loop:
                while viewer.is_running():
                    time.sleep(0.1)
                break
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    motion_path = Path(args.motion).expanduser().resolve()
    dataset_root = find_dataset_root(motion_path)
    manifest = load_manifest(dataset_root)
    qpos = load_motion(motion_path)

    with temp_mjcf_with_floor(manifest.mjcf_path) as viewer_mjcf_path:
        model = mujoco.MjModel.from_xml_path(str(viewer_mjcf_path))
    data = mujoco.MjData(model)

    if qpos.shape[1] != model.nq:
        raise ValueError(f"Motion qpos width {qpos.shape[1]} does not match model.nq={model.nq}")

    frame_indices = list(_iter_frame_indices(qpos.shape[0], args.start, args.end, args.stride))
    if not frame_indices:
        raise ValueError("No frames selected. Check --start/--end/--stride.")

    fps = float(args.fps) if args.fps is not None else 1.0 / manifest.timestep
    frame_dt = 1.0 / fps

    if args.headless:
        for frame_idx in tqdm(frame_indices, desc="Playing", unit="frame"):
            _apply_qpos_frame(data, qpos[frame_idx])
            mujoco.mj_forward(model, data)
        return

    if args.viewer == "viser":
        _play_viser(model, data, qpos, frame_indices, frame_dt, loop=args.loop)
        return

    deadline = time.monotonic()
    with mujoco_viewer.launch_passive(
        model,
        data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        while viewer.is_running():
            for frame_idx in frame_indices:
                if not viewer.is_running():
                    break
                _apply_qpos_frame(data, qpos[frame_idx])
                mujoco.mj_forward(model, data)
                viewer.sync()
                deadline += frame_dt
                _sleep_until(deadline)
            if not args.loop:
                break
            deadline = time.monotonic()


if __name__ == "__main__":
    main()
