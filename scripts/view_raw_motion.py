#!/usr/bin/env python3
"""Compare corrected-source body transforms with the current MuJoCo reconstruction."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, SimpleQueue

import mujoco
import numpy as np
from any4hdmi.utils.mjcf import qpos_names_from_model
from mjhub import temp_mjcf_with_floor

from sim2real.config.robots import G1_CFG
from sim2real.utils.mjviser_viewer import MjviserMujocoViewer
from tracking_experiment.convert_to_any4hdmi import (
    ISAACLAB_G1_BODY_NAMES,
    _iter_inputs,
    _load_isaaclab_corrected,
    _scan_root,
)


RAW_COLOR = (255, 0, 190)


@dataclass(frozen=True)
class MismatchSummary:
    position_mean_m: float
    position_p95_m: float
    orientation_mean_rad: float
    orientation_p95_rad: float


@dataclass(frozen=True)
class RawMotionClip:
    path: Path
    qpos: np.ndarray
    body_pos_w: np.ndarray
    body_quat_wxyz: np.ndarray
    frame_indices: list[int]
    fps: float
    mismatch: MismatchSummary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a small corrected G1 source sample directly and overlay "
            "the MuJoCo reconstruction used by convert_to_any4hdmi.py."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "One corrected .npz, origin_interp10_NPZ, or its parent dataset root."
        ),
    )
    parser.add_argument(
        "--num-motions",
        type=int,
        default=5,
        help="Select at most this many motions without scanning the full dataset.",
    )
    parser.add_argument(
        "--body-quat-order",
        choices=("wxyz", "xyzw"),
        default="wxyz",
        help="Interpretation of source body_quat_w (default: wxyz).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Playback FPS override. Defaults to the selected source motion FPS.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start frame index.")
    parser.add_argument("--end", type=int, default=-1, help="End frame index.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    parser.add_argument("--loop", action="store_true", help="Loop the selected motion.")
    return parser.parse_args(argv)


def _resolve_source_paths(input_path: Path, num_motions: int) -> tuple[Path, list[Path]]:
    if num_motions <= 0:
        raise ValueError("--num-motions must be positive")
    input_path = input_path.expanduser().resolve()
    scan_root = _scan_root(input_path)
    scan_input = input_path if input_path.is_file() else scan_root
    return scan_root, _iter_inputs(scan_input, max_files=num_motions)


def _motion_label(path: Path, scan_root: Path) -> str:
    try:
        return path.relative_to(scan_root).as_posix()
    except ValueError:
        return path.name


def _frame_indices(length: int, start: int, end: int, stride: int) -> list[int]:
    resolved_end = length if end < 0 else min(length, end)
    indices = list(range(start, resolved_end, max(1, stride)))
    if not indices:
        raise ValueError("No frames selected. Check --start/--end/--stride.")
    return indices


def _body_ids(model: mujoco.MjModel, body_names: Sequence[str]) -> np.ndarray:
    ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in body_names
        ],
        dtype=np.int32,
    )
    if np.any(ids < 0):
        missing = [name for name, body_id in zip(body_names, ids) if body_id < 0]
        raise ValueError(f"Target MJCF is missing bodies: {missing}")
    return ids


def _body_edges(model: mujoco.MjModel, body_names: Sequence[str]) -> np.ndarray:
    index_by_name = {name: index for index, name in enumerate(body_names)}
    edges: list[tuple[int, int]] = []
    for child_name, child_index in index_by_name.items():
        child_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            child_name,
        )
        if child_id < 0:
            raise ValueError(f"Target MJCF is missing body: {child_name}")
        parent_id = int(model.body_parentid[child_id])
        parent_name = model.body(parent_id).name if parent_id >= 0 else None
        if parent_name in index_by_name:
            edges.append((index_by_name[parent_name], child_index))
    return np.asarray(edges, dtype=np.int32).reshape(-1, 2)


def _body_quats_to_wxyz(quat: np.ndarray, order: str) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    if quat.ndim != 3 or quat.shape[-1] != 4:
        raise ValueError(f"body_quat_w must have shape (T, B, 4), got {quat.shape}")
    if order == "xyzw":
        quat = quat[..., [3, 0, 1, 2]]
    norms = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("body_quat_w contains a zero quaternion")
    return quat / norms


def _mismatch_summary(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    body_pos_w: np.ndarray,
    body_quat_wxyz: np.ndarray,
    body_ids: np.ndarray,
    frame_indices: Sequence[int],
) -> MismatchSummary:
    data = mujoco.MjData(model)
    position_errors: list[np.ndarray] = []
    orientation_errors: list[np.ndarray] = []
    for frame_index in frame_indices:
        data.qpos[:] = qpos[frame_index]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        position_errors.append(
            np.linalg.norm(data.xpos[body_ids] - body_pos_w[frame_index], axis=-1)
        )
        quat_dot = np.abs(
            np.sum(
                data.xquat[body_ids] * body_quat_wxyz[frame_index],
                axis=-1,
            )
        )
        orientation_errors.append(
            2.0 * np.arccos(np.clip(quat_dot, 0.0, 1.0))
        )

    position_error = np.concatenate(position_errors)
    orientation_error = np.concatenate(orientation_errors)
    return MismatchSummary(
        position_mean_m=float(np.mean(position_error)),
        position_p95_m=float(np.percentile(position_error, 95)),
        orientation_mean_rad=float(np.mean(orientation_error)),
        orientation_p95_rad=float(np.percentile(orientation_error, 95)),
    )


def _load_clip(
    path: Path,
    *,
    model: mujoco.MjModel,
    qpos_names: list[str],
    body_ids: np.ndarray,
    body_quat_order: str,
    start: int,
    end: int,
    stride: int,
) -> RawMotionClip:
    qpos, fps = _load_isaaclab_corrected(
        path,
        qpos_names=qpos_names,
        body_quat_order=body_quat_order,
    )
    with np.load(path, allow_pickle=False) as raw:
        body_pos_w = np.asarray(raw["body_pos_w"], dtype=np.float32)
        body_quat_wxyz = _body_quats_to_wxyz(
            raw["body_quat_w"],
            body_quat_order,
        )
    if body_pos_w.shape != body_quat_wxyz.shape[:-1] + (3,):
        raise ValueError(
            f"body position/quaternion shape mismatch: "
            f"{body_pos_w.shape} vs {body_quat_wxyz.shape}"
        )
    if body_pos_w.shape[1] != len(ISAACLAB_G1_BODY_NAMES):
        raise ValueError(
            f"Expected {len(ISAACLAB_G1_BODY_NAMES)} source bodies, "
            f"got {body_pos_w.shape[1]}"
        )
    selected_frames = _frame_indices(qpos.shape[0], start, end, stride)
    return RawMotionClip(
        path=path,
        qpos=qpos,
        body_pos_w=body_pos_w,
        body_quat_wxyz=body_quat_wxyz,
        frame_indices=selected_frames,
        fps=float(fps),
        mismatch=_mismatch_summary(
            model,
            qpos,
            body_pos_w,
            body_quat_wxyz,
            body_ids,
            selected_frames,
        ),
    )


def _latest_selection(requests: SimpleQueue[int], current_index: int) -> int:
    selected_index = current_index
    while True:
        try:
            selected_index = requests.get_nowait()
        except Empty:
            return selected_index


def _adjacent_index(current_index: int, count: int, offset: int) -> int:
    return (current_index + offset) % count


def _status_content(label: str, clip: RawMotionClip) -> str:
    mismatch = clip.mismatch
    return "\n".join(
        [
            f"**Motion:** {label}",
            "",
            f"Frames: {len(clip.frame_indices)} at {clip.fps:g} FPS",
            "",
            "**Raw body vs MuJoCo FK**",
            "",
            f"- position mean: {mismatch.position_mean_m:.4f} m",
            f"- position p95: {mismatch.position_p95_m:.4f} m",
            f"- orientation mean: {mismatch.orientation_mean_rad:.4f} rad",
            f"- orientation p95: {mismatch.orientation_p95_rad:.4f} rad",
        ]
    )


def _play(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    source_paths: list[Path],
    scan_root: Path,
    *,
    fps_override: float | None,
    body_quat_order: str,
    start: int,
    end: int,
    stride: int,
    loop: bool,
) -> None:
    qpos_names = [str(name) for name in qpos_names_from_model(model)]
    body_ids = _body_ids(model, ISAACLAB_G1_BODY_NAMES)
    body_edges = _body_edges(model, ISAACLAB_G1_BODY_NAMES)
    labels = [_motion_label(path, scan_root) for path in source_paths]
    label_to_index = {label: index for index, label in enumerate(labels)}
    selection_requests: SimpleQueue[int] = SimpleQueue()

    selected_index = 0
    clip = _load_clip(
        source_paths[selected_index],
        model=model,
        qpos_names=qpos_names,
        body_ids=body_ids,
        body_quat_order=body_quat_order,
        start=start,
        end=end,
        stride=stride,
    )
    first_frame = clip.frame_indices[0]
    data.qpos[:] = clip.qpos[first_frame]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    viewer = MjviserMujocoViewer(
        model,
        data,
        label="corrected-source-diagnostic",
        tracked_body_id=int(body_ids[0]),
    )
    raw_points = viewer.server.scene.add_point_cloud(
        "/diagnostic/raw/body_points",
        clip.body_pos_w[first_frame],
        RAW_COLOR,
        point_size=0.035,
        point_shape="circle",
        precision="float32",
    )
    raw_skeleton = viewer.server.scene.add_line_segments(
        "/diagnostic/raw/skeleton",
        clip.body_pos_w[first_frame][body_edges],
        RAW_COLOR,
        line_width=4.0,
    )
    raw_axes = viewer.server.scene.add_batched_axes(
        "/diagnostic/raw/body_axes",
        clip.body_quat_wxyz[first_frame],
        clip.body_pos_w[first_frame],
        axes_length=0.08,
        axes_radius=0.003,
    )

    with viewer.server.gui.add_folder("Corrected source diagnostic"):
        viewer.server.gui.add_markdown(
            "**Magenta:** raw body points/skeleton. "
            "**Robot mesh:** current converter reconstruction."
        )
        status = viewer.server.gui.add_markdown(
            _status_content(labels[selected_index], clip)
        )
        motion_dropdown = viewer.server.gui.add_dropdown(
            "Motion",
            labels,
            initial_value=labels[selected_index],
            disabled=len(labels) == 1,
        )
        previous_button = viewer.server.gui.add_button(
            "Previous",
            disabled=len(labels) == 1,
        )
        next_button = viewer.server.gui.add_button(
            "Next",
            disabled=len(labels) == 1,
        )
        show_axes = viewer.server.gui.add_checkbox(
            "Show raw body axes",
            initial_value=True,
        )

    @motion_dropdown.on_update
    def _select_motion(_) -> None:
        selection_requests.put(label_to_index[motion_dropdown.value])

    @previous_button.on_click
    def _select_previous(_) -> None:
        current_index = label_to_index[motion_dropdown.value]
        motion_dropdown.value = labels[
            _adjacent_index(current_index, len(labels), -1)
        ]

    @next_button.on_click
    def _select_next(_) -> None:
        current_index = label_to_index[motion_dropdown.value]
        motion_dropdown.value = labels[
            _adjacent_index(current_index, len(labels), 1)
        ]

    @show_axes.on_update
    def _show_axes(_) -> None:
        raw_axes.visible = show_axes.value

    try:
        while viewer.is_running() and not viewer.has_clients():
            time.sleep(0.05)

        frame_cursor = 0
        playing = True
        deadline = time.monotonic()
        frame_dt = 1.0 / (fps_override or clip.fps)
        while viewer.is_running():
            requested_index = _latest_selection(selection_requests, selected_index)
            if requested_index != selected_index:
                selected_index = requested_index
                clip = _load_clip(
                    source_paths[selected_index],
                    model=model,
                    qpos_names=qpos_names,
                    body_ids=body_ids,
                    body_quat_order=body_quat_order,
                    start=start,
                    end=end,
                    stride=stride,
                )
                status.content = _status_content(labels[selected_index], clip)
                frame_cursor = 0
                playing = True
                deadline = time.monotonic()
                frame_dt = 1.0 / (fps_override or clip.fps)

            if not playing:
                time.sleep(0.05)
                continue

            frame_index = clip.frame_indices[frame_cursor]
            data.qpos[:] = clip.qpos[frame_index]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            raw_positions = clip.body_pos_w[frame_index]
            raw_points.points = raw_positions
            raw_skeleton.points = raw_positions[body_edges]
            raw_axes.batched_positions = raw_positions
            raw_axes.batched_wxyzs = clip.body_quat_wxyz[frame_index]
            viewer.sync()

            deadline += frame_dt
            remaining = deadline - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)

            frame_cursor += 1
            if frame_cursor == len(clip.frame_indices):
                if loop:
                    frame_cursor = 0
                else:
                    playing = False
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.fps is not None and args.fps <= 0.0:
        raise ValueError("--fps must be positive")

    scan_root, source_paths = _resolve_source_paths(
        Path(args.input),
        args.num_motions,
    )
    print(f"Selected {len(source_paths)} corrected source motions from {scan_root}")
    for source_path in source_paths:
        print(f"- {_motion_label(source_path, scan_root)}")

    with temp_mjcf_with_floor(G1_CFG.resolve_mjcf_path()) as viewer_mjcf_path:
        model = mujoco.MjModel.from_xml_path(str(viewer_mjcf_path))
    data = mujoco.MjData(model)
    _play(
        model,
        data,
        source_paths,
        scan_root,
        fps_override=args.fps,
        body_quat_order=args.body_quat_order,
        start=args.start,
        end=args.end,
        stride=args.stride,
        loop=args.loop,
    )


if __name__ == "__main__":
    main()
