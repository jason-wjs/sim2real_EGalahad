#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from queue import Empty, SimpleQueue

import mujoco
from any4hdmi.core.format import find_dataset_root, load_manifest, load_motion
from mjhub import temp_mjcf_with_floor
from mujoco import viewer as mujoco_viewer
from tqdm import tqdm

from sim2real.utils.mjviser_viewer import MjviserMujocoViewer


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a qpos-only any4hdmi motion with MuJoCo.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--motion",
        help="Path to a converted motion .npz file. The dataset root is inferred from manifest.json.",
    )
    source.add_argument(
        "--dataset",
        help="Path to an any4hdmi dataset root. Requires --viewer viser.",
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


def _discover_motion_paths(
    dataset_root: Path,
    motions_subdir: str,
    expected_count: int | None = None,
) -> list[Path]:
    motions_root = dataset_root / motions_subdir
    if not motions_root.is_dir():
        raise FileNotFoundError(f"Motions directory not found: {motions_root}")

    records_path = dataset_root / "conversion_records.jsonl"
    if records_path.is_file():
        indexed_paths: list[Path] = []
        for line_number, line in enumerate(
            records_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            output = json.loads(line).get("output")
            if not isinstance(output, str) or not output:
                raise ValueError(f"Missing output path in {records_path}:{line_number}")
            relative_path = Path(output)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"Invalid output path in {records_path}:{line_number}")
            motion_path = dataset_root / relative_path
            try:
                motion_path.relative_to(motions_root)
            except ValueError as exc:
                raise ValueError(
                    f"Output path is outside {motions_root}: {output}"
                ) from exc
            indexed_paths.append(motion_path)
        if indexed_paths and (
            expected_count is None or len(indexed_paths) == expected_count
        ):
            return sorted(indexed_paths)

    motion_paths = sorted(motions_root.rglob("*.npz"))
    if not motion_paths:
        raise FileNotFoundError(f"No .npz motions found under {motions_root}")
    return motion_paths


def _motion_label(motion_path: Path, motions_root: Path) -> str:
    return motion_path.relative_to(motions_root).as_posix()


def _adjacent_motion_index(current_index: int, count: int, offset: int) -> int:
    return (current_index + offset) % count


def _latest_motion_selection(requests: SimpleQueue[int], current_index: int) -> int:
    selected_index = current_index
    while True:
        try:
            selected_index = requests.get_nowait()
        except Empty:
            return selected_index


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


def _load_motion_frames(
    motion_path: Path,
    *,
    model_nq: int,
    start: int,
    end: int,
    stride: int,
):
    qpos = load_motion(motion_path)
    if qpos.shape[1] != model_nq:
        raise ValueError(f"Motion qpos width {qpos.shape[1]} does not match model.nq={model_nq}")
    frame_indices = list(_iter_frame_indices(qpos.shape[0], start, end, stride))
    if not frame_indices:
        raise ValueError(f"No frames selected for {motion_path}. Check --start/--end/--stride.")
    return qpos, frame_indices


def _play_viser(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    motion_paths: list[Path],
    motions_root: Path,
    frame_dt: float,
    *,
    start: int,
    end: int,
    stride: int,
    loop: bool,
) -> None:
    labels = [_motion_label(path, motions_root) for path in motion_paths]
    label_to_index = {label: index for index, label in enumerate(labels)}
    selection_requests: SimpleQueue[int] = SimpleQueue()
    selected_index = 0
    qpos, frame_indices = _load_motion_frames(
        motion_paths[selected_index],
        model_nq=model.nq,
        start=start,
        end=end,
        stride=stride,
    )
    _apply_qpos_frame(data, qpos[frame_indices[0]])
    mujoco.mj_forward(model, data)
    viewer = MjviserMujocoViewer(
        model,
        data,
        label="any4hdmi-reference-viewer",
        tracked_body_id=_tracked_body_id(model),
    )
    with viewer.server.gui.add_folder("Motion dataset"):
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

    @motion_dropdown.on_update
    def _select_motion(_) -> None:
        selection_requests.put(label_to_index[motion_dropdown.value])

    @previous_button.on_click
    def _select_previous(_) -> None:
        current_index = label_to_index[motion_dropdown.value]
        motion_dropdown.value = labels[
            _adjacent_motion_index(current_index, len(labels), -1)
        ]

    @next_button.on_click
    def _select_next(_) -> None:
        current_index = label_to_index[motion_dropdown.value]
        motion_dropdown.value = labels[
            _adjacent_motion_index(current_index, len(labels), 1)
        ]

    try:
        while viewer.is_running() and not viewer.has_clients():
            time.sleep(0.05)

        frame_cursor = 0
        playing = True
        deadline = time.monotonic()
        while viewer.is_running():
            requested_index = _latest_motion_selection(selection_requests, selected_index)
            if requested_index != selected_index:
                selected_index = requested_index
                qpos, frame_indices = _load_motion_frames(
                    motion_paths[selected_index],
                    model_nq=model.nq,
                    start=start,
                    end=end,
                    stride=stride,
                )
                frame_cursor = 0
                playing = True
                deadline = time.monotonic()

            if not playing:
                time.sleep(0.05)
                continue

            _apply_qpos_frame(data, qpos[frame_indices[frame_cursor]])
            mujoco.mj_forward(model, data)
            viewer.sync()
            deadline += frame_dt
            _sleep_until(deadline)

            frame_cursor += 1
            if frame_cursor == len(frame_indices):
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

    if args.dataset is not None:
        if args.viewer != "viser" or args.headless:
            raise ValueError("--dataset requires --viewer viser without --headless")
        dataset_root = Path(args.dataset).expanduser().resolve()
    else:
        motion_path = Path(args.motion).expanduser().resolve()
        dataset_root = find_dataset_root(motion_path)
    manifest = load_manifest(dataset_root)
    motions_subdir = str(manifest.payload.get("motions_subdir", "motions"))
    motions_root = dataset_root / motions_subdir
    motion_paths = (
        _discover_motion_paths(
            dataset_root,
            motions_subdir,
            expected_count=int(manifest.payload["num_motions"]),
        )
        if args.dataset is not None
        else [motion_path]
    )

    with temp_mjcf_with_floor(manifest.mjcf_path) as viewer_mjcf_path:
        model = mujoco.MjModel.from_xml_path(str(viewer_mjcf_path))
    data = mujoco.MjData(model)

    fps = float(args.fps) if args.fps is not None else 1.0 / manifest.timestep
    frame_dt = 1.0 / fps

    if args.viewer == "viser" and not args.headless:
        _play_viser(
            model,
            data,
            motion_paths,
            motions_root,
            frame_dt,
            start=args.start,
            end=args.end,
            stride=args.stride,
            loop=args.loop,
        )
        return

    qpos, frame_indices = _load_motion_frames(
        motion_paths[0],
        model_nq=model.nq,
        start=args.start,
        end=args.end,
        stride=args.stride,
    )

    if args.headless:
        for frame_idx in tqdm(frame_indices, desc="Playing", unit="frame"):
            _apply_qpos_frame(data, qpos[frame_idx])
            mujoco.mj_forward(model, data)
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
