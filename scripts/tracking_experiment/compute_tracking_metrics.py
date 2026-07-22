from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from sim2real.utils.math import (
    projected_yaw_quat,
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse_numpy,
)


METRIC_SCHEMA_VERSION = "2.0.0"
TRACKING_BODY_PATTERNS = (
    "pelvis",
    "torso_link",
    ".*_hip_yaw_link",
    ".*_knee_link",
    ".*_toe_link",
    ".*_shoulder_yaw_link",
    ".*_elbow_link",
    ".*_wrist_yaw_link",
)
END_EFFECTOR_BODY_PATTERNS = (
    ".*_toe_link",
    ".*_wrist_yaw_link",
)
TERMINATION_ROOT_BODY_NAME = "torso_link"
ANCHOR_BODY_NAME = "pelvis"
REQUIRED_TRAJECTORY_KEYS = {
    "robot_root_pos_w",
    "robot_root_quat_w",
    "motion_root_pos_w",
    "motion_root_quat_w",
    "robot_body_pos_w",
    "robot_body_quat_w",
    "motion_body_pos_w",
    "motion_body_quat_w",
    "body_names",
    "robot_joint_pos",
    "robot_joint_vel",
    "motion_joint_pos",
    "motion_joint_vel",
    "joint_names",
    "sim_time",
    "motion_t",
    "motion_length",
}


SUMMARY_FIELD_GROUPS = (
    (
        ("outcome",),
        {
            "completion_ratio": "completion_ratio",
            "success_rate": "success",
            "termination_rate": "terminated",
        },
    ),
    (
        ("tracking", "global_start_aligned"),
        {
            "root_position_xyz_mean_m": "global_root_pos_xyz_mean_m",
            "root_position_xyz_p95_m": "global_root_pos_xyz_p95_m",
            "root_position_xy_mean_m": "global_root_pos_xy_mean_m",
            "root_position_xy_p95_m": "global_root_pos_xy_p95_m",
            "root_orientation_mean_rad": "global_root_ori_mean_rad",
            "root_orientation_p95_rad": "global_root_ori_p95_rad",
            "key_body_position_mean_m": "global_key_body_pos_mean_m",
            "key_body_position_p95_m": "global_key_body_pos_p95_m",
            "key_body_orientation_mean_rad": "global_key_body_ori_mean_rad",
            "key_body_orientation_p95_rad": "global_key_body_ori_p95_rad",
            "end_effector_position_mean_m": "global_end_effector_pos_mean_m",
            "end_effector_position_p95_m": "global_end_effector_pos_p95_m",
            "end_effector_orientation_mean_rad": "global_end_effector_ori_mean_rad",
            "end_effector_orientation_p95_rad": "global_end_effector_ori_p95_rad",
            "key_body_velocity_error_mean_mps": "global_key_body_vel_error_mean_mps",
            "key_body_velocity_error_p95_mps": "global_key_body_vel_error_p95_mps",
            "key_body_acceleration_error_mean_mps2": "global_key_body_acc_error_mean_mps2",
            "key_body_acceleration_error_p95_mps2": "global_key_body_acc_error_p95_mps2",
        },
    ),
    (
        ("tracking", "local_heading"),
        {
            "key_body_position_mean_m": "local_key_body_pos_mean_m",
            "key_body_position_p95_m": "local_key_body_pos_p95_m",
            "key_body_orientation_mean_rad": "local_key_body_ori_mean_rad",
            "key_body_orientation_p95_rad": "local_key_body_ori_p95_rad",
            "end_effector_position_mean_m": "local_end_effector_pos_mean_m",
            "end_effector_position_p95_m": "local_end_effector_pos_p95_m",
            "end_effector_orientation_mean_rad": "local_end_effector_ori_mean_rad",
            "end_effector_orientation_p95_rad": "local_end_effector_ori_p95_rad",
        },
    ),
    (
        ("tracking", "joint_space"),
        {
            "position_mae_rad": "joint_pos_mae_rad",
            "position_rmse_rad": "joint_pos_rmse_rad",
            "position_p95_abs_rad": "joint_pos_p95_abs_rad",
            "velocity_mae_rad_s": "joint_vel_mae_rad_s",
            "velocity_rmse_rad_s": "joint_vel_rmse_rad_s",
            "velocity_p95_abs_rad_s": "joint_vel_p95_abs_rad_s",
        },
    ),
    (
        ("smoothness",),
        {
            "joint_acceleration_rms_rad_s2": "joint_acc_rms_rad_s2",
            "joint_acceleration_p95_abs_rad_s2": "joint_acc_p95_abs_rad_s2",
            "joint_jerk_rms_rad_s3": "joint_jerk_rms_rad_s3",
            "joint_jerk_p95_abs_rad_s3": "joint_jerk_p95_abs_rad_s3",
        },
    ),
    (
        ("legacy",),
        {
            "progress": "progress",
            "global_root_tracking_error": "global_root_tracking_error",
            "global_root_tracking_error_xy": "global_root_tracking_error_xy",
            "local_body_tracking_error": "local_body_tracking_error",
            "mpjpe": "mpjpe",
            "root_final_error_norm": "root_final_error_norm",
            "root_final_error_xy_norm": "root_final_error_xy_norm",
        },
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute outcome, start-aligned global tracking, heading-local tracking, "
            "joint-space tracking, dynamic tracking, and smoothness metrics from "
            "trajectory NPZ files saved by the integrated MuJoCo evaluator."
        )
    )
    parser.add_argument("paths", nargs="*", help="Full trajectory .npz files or glob patterns.")
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="runs.csv manifest containing successful trajectory_path entries.",
    )
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-json", default=None, help="Summary-only JSON output.")
    parser.add_argument(
        "--print-rows",
        action="store_true",
        help="Print per-rollout rows to stdout in addition to the compact summary.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print metric progress every N trajectories.",
    )
    return parser.parse_args()


def _manifest_paths(manifest_paths: list[str]) -> list[Path]:
    paths: list[Path] = []
    for manifest_text in manifest_paths:
        manifest = Path(manifest_text).expanduser().resolve()
        with manifest.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "trajectory_path" not in reader.fieldnames:
                raise ValueError(f"Manifest is missing trajectory_path: {manifest}")
            for line_number, row in enumerate(reader, start=2):
                status = str(row.get("status", "")).strip()
                if status and status not in {"succeeded", "skipped_existing"}:
                    continue
                trajectory_text = str(row.get("trajectory_path", "")).strip()
                if not trajectory_text:
                    raise ValueError(
                        f"Manifest has empty trajectory_path at {manifest}:{line_number}"
                    )
                trajectory = Path(trajectory_text).expanduser()
                if not trajectory.is_absolute():
                    trajectory = manifest.parent / trajectory
                trajectory = trajectory.resolve()
                if not trajectory.is_file():
                    raise FileNotFoundError(
                        f"Manifest trajectory does not exist at {manifest}:{line_number}: "
                        f"{trajectory}"
                    )
                paths.append(trajectory)
    return paths


def _expand_paths(
    patterns: list[str],
    manifest_paths: list[str] | None = None,
) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        path = Path(pattern).expanduser()
        expanded = (
            sorted(Path(item) for item in glob.glob(str(path), recursive=True))
            if any(char in pattern for char in "*?[]")
            else [path]
        )
        for item in expanded:
            resolved = item.resolve()
            if resolved.is_file() and resolved not in seen:
                paths.append(resolved)
                seen.add(resolved)
    for resolved in _manifest_paths(manifest_paths or []):
        if resolved not in seen:
            paths.append(resolved)
            seen.add(resolved)
    if not paths:
        raise FileNotFoundError(
            f"No trajectory files matched paths={patterns} manifests={manifest_paths or []}"
        )
    return paths


def _scalar(value: np.ndarray | str | bytes | object) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def _select_policy_frames(data: dict[str, np.ndarray]) -> np.ndarray:
    motion_t = np.asarray(data["motion_t"], dtype=np.int32)
    if motion_t.size == 0:
        raise ValueError("empty motion_t")
    return np.flatnonzero(np.r_[True, motion_t[1:] != motion_t[:-1]])


def _indices_for_patterns(names: list[str], patterns: tuple[str, ...]) -> list[int]:
    indices: list[int] = []
    for pattern in patterns:
        for idx, name in enumerate(names):
            if idx in indices:
                continue
            if name == pattern or re.fullmatch(pattern, name):
                indices.append(idx)
    if not indices:
        raise ValueError(f"No body names matched patterns: {patterns}")
    return indices


def _quat_angle_magnitude(quat: np.ndarray, eps: float = 1.0e-9) -> np.ndarray:
    xyz_norm = np.linalg.norm(quat[..., 1:], axis=-1)
    return 2.0 * np.arctan2(xyz_norm, np.maximum(np.abs(quat[..., 0]), eps))


def _orientation_error(reference: np.ndarray, robot: np.ndarray) -> np.ndarray:
    return _quat_angle_magnitude(quat_mul(quat_conjugate(reference), robot))


def _local_tracking_state(
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    anchor_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    anchor_pos = body_pos_w[:, anchor_idx].copy()
    anchor_pos[:, 2] = 0.0
    anchor_yaw = projected_yaw_quat(body_quat_w[:, anchor_idx])
    anchor_yaw_expanded = np.broadcast_to(anchor_yaw[:, None, :], body_quat_w.shape)
    body_pos_local = quat_rotate_inverse_numpy(
        anchor_yaw_expanded,
        body_pos_w - anchor_pos[:, None, :],
    )
    body_quat_local = quat_mul(
        quat_conjugate(anchor_yaw_expanded),
        body_quat_w,
    )
    return body_pos_local, body_quat_local


def _start_aligned_tracking_state(
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    root_pos_w: np.ndarray,
    root_quat_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    initial_root_quat = np.broadcast_to(root_quat_w[0], body_quat_w.shape)
    body_pos_start = quat_rotate_inverse_numpy(
        initial_root_quat,
        body_pos_w - root_pos_w[0].reshape(1, 1, 3),
    )
    body_quat_start = quat_mul(
        quat_conjugate(initial_root_quat),
        body_quat_w,
    )
    return body_pos_start, body_quat_start


def _first_cumulative_failure(
    error: np.ndarray,
    *,
    threshold: float,
    min_steps: int,
) -> int | None:
    count = 0
    for idx, value in enumerate(error):
        if float(value) >= threshold:
            count += 1
        else:
            count = 0
        if count >= min_steps:
            return idx
    return None


def _relative_translation(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    return quat_rotate_inverse_numpy(
        quat[0].reshape(1, 4),
        (pos[-1] - pos[0]).reshape(1, 3),
    )[0]


def _relative_translation_series(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    return quat_rotate_inverse_numpy(
        np.broadcast_to(quat[0].reshape(1, 4), quat.shape),
        pos - pos[0].reshape(1, 3),
    )


def _relative_orientation_series(quat: np.ndarray) -> np.ndarray:
    return quat_mul(
        np.broadcast_to(quat_conjugate(quat[0].reshape(1, 4)), quat.shape),
        quat,
    )


def _differentiate(values: np.ndarray, time_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(values) < 2:
        return np.empty((0, *values.shape[1:]), dtype=np.float64), np.empty(0)
    dt = np.diff(time_s.astype(np.float64))
    if np.any(dt <= 0.0):
        raise ValueError("sim_time must be strictly increasing at selected policy frames")
    reshape = (len(dt),) + (1,) * (values.ndim - 1)
    derivative = np.diff(values.astype(np.float64), axis=0) / dt.reshape(reshape)
    derivative_time = 0.5 * (time_s[:-1] + time_s[1:])
    return derivative, derivative_time


def _finite_flat(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def _mean(values: np.ndarray) -> float | None:
    array = _finite_flat(values)
    return None if array.size == 0 else float(np.mean(array))


def _p95(values: np.ndarray) -> float | None:
    array = _finite_flat(values)
    return None if array.size == 0 else float(np.percentile(array, 95))


def _rmse(values: np.ndarray) -> float | None:
    array = _finite_flat(values)
    return None if array.size == 0 else float(np.sqrt(np.mean(np.square(array))))


def _mean_p95(values: np.ndarray) -> tuple[float | None, float | None]:
    return _mean(values), _p95(values)


def _compute_one(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as loaded:
        missing = REQUIRED_TRAJECTORY_KEYS.difference(loaded.files)
        if missing:
            raise ValueError(
                f"Trajectory {path} uses an older/incomplete schema; missing keys: "
                f"{sorted(missing)}"
            )
        data = {key: loaded[key] for key in loaded.files}

    frame_idx = _select_policy_frames(data)
    names = [str(name) for name in np.asarray(data["body_names"]).tolist()]
    joint_names = [str(name) for name in np.asarray(data["joint_names"]).tolist()]
    tracking_indices = _indices_for_patterns(names, TRACKING_BODY_PATTERNS)
    end_effector_indices = _indices_for_patterns(names, END_EFFECTOR_BODY_PATTERNS)
    root_idx = names.index(TERMINATION_ROOT_BODY_NAME)
    anchor_idx = names.index(ANCHOR_BODY_NAME)

    robot_pos = np.asarray(data["robot_body_pos_w"], dtype=np.float64)[frame_idx]
    robot_quat = np.asarray(data["robot_body_quat_w"], dtype=np.float64)[frame_idx]
    motion_pos = np.asarray(data["motion_body_pos_w"], dtype=np.float64)[frame_idx]
    motion_quat = np.asarray(data["motion_body_quat_w"], dtype=np.float64)[frame_idx]
    robot_root_pos = np.asarray(data["robot_root_pos_w"], dtype=np.float64)[frame_idx]
    robot_root_quat = np.asarray(data["robot_root_quat_w"], dtype=np.float64)[frame_idx]
    motion_root_pos = np.asarray(data["motion_root_pos_w"], dtype=np.float64)[frame_idx]
    motion_root_quat = np.asarray(data["motion_root_quat_w"], dtype=np.float64)[frame_idx]
    robot_joint_pos = np.asarray(data["robot_joint_pos"], dtype=np.float64)[frame_idx]
    robot_joint_vel = np.asarray(data["robot_joint_vel"], dtype=np.float64)[frame_idx]
    motion_joint_pos = np.asarray(data["motion_joint_pos"], dtype=np.float64)[frame_idx]
    motion_joint_vel = np.asarray(data["motion_joint_vel"], dtype=np.float64)[frame_idx]
    sim_time = np.asarray(data["sim_time"], dtype=np.float64)[frame_idx]
    motion_t = np.asarray(data["motion_t"], dtype=np.int32)[frame_idx]

    robot_pos_local, robot_quat_local = _local_tracking_state(
        robot_pos,
        robot_quat,
        anchor_idx,
    )
    motion_pos_local, motion_quat_local = _local_tracking_state(
        motion_pos,
        motion_quat,
        anchor_idx,
    )
    termination_root_ori_error = _orientation_error(
        motion_quat[:, root_idx],
        robot_quat[:, root_idx],
    )
    body_pos_error_local = np.linalg.norm(
        motion_pos_local[:, tracking_indices] - robot_pos_local[:, tracking_indices],
        axis=-1,
    )
    body_ori_error_local = _orientation_error(
        motion_quat_local[:, tracking_indices],
        robot_quat_local[:, tracking_indices],
    )

    failures = {
        "root_ori_error": _first_cumulative_failure(
            termination_root_ori_error,
            threshold=1.2,
            min_steps=25,
        ),
        "body_pos_error": _first_cumulative_failure(
            body_pos_error_local.max(axis=1),
            threshold=0.4,
            min_steps=5,
        ),
        "body_ori_error": _first_cumulative_failure(
            body_ori_error_local.max(axis=1),
            threshold=1.2,
            min_steps=5,
        ),
    }
    valid_failures = {name: idx for name, idx in failures.items() if idx is not None}
    motion_length = int(np.asarray(data["motion_length"]).reshape(()))
    if valid_failures:
        termination_reason, termination_idx = min(
            valid_failures.items(),
            key=lambda item: item[1],
        )
        terminated = True
    elif int(motion_t[-1]) < motion_length - 1:
        termination_reason = "truncated"
        termination_idx = int(len(motion_t) - 1)
        terminated = True
    else:
        termination_reason = "motion_end"
        termination_idx = int(len(motion_t) - 1)
        terminated = False

    pre_end = max(
        1,
        termination_idx if termination_reason in valid_failures else termination_idx + 1,
    )
    motion_denominator = max(1, motion_length - 1)
    completion_ratio = min(
        1.0,
        max(0.0, float(motion_t[termination_idx]) / float(motion_denominator)),
    )
    eval_time = sim_time[:pre_end]

    robot_root_pos_eval = robot_root_pos[:pre_end]
    robot_root_quat_eval = robot_root_quat[:pre_end]
    motion_root_pos_eval = motion_root_pos[:pre_end]
    motion_root_quat_eval = motion_root_quat[:pre_end]
    robot_root_rel = _relative_translation_series(
        robot_root_pos_eval,
        robot_root_quat_eval,
    )
    motion_root_rel = _relative_translation_series(
        motion_root_pos_eval,
        motion_root_quat_eval,
    )
    root_position_error = robot_root_rel - motion_root_rel
    root_position_error_xyz = np.linalg.norm(root_position_error, axis=-1)
    root_position_error_xy = np.linalg.norm(root_position_error[:, :2], axis=-1)
    root_orientation_error = _orientation_error(
        _relative_orientation_series(motion_root_quat_eval),
        _relative_orientation_series(robot_root_quat_eval),
    )

    robot_pos_start, robot_quat_start = _start_aligned_tracking_state(
        robot_pos[:pre_end],
        robot_quat[:pre_end],
        robot_root_pos_eval,
        robot_root_quat_eval,
    )
    motion_pos_start, motion_quat_start = _start_aligned_tracking_state(
        motion_pos[:pre_end],
        motion_quat[:pre_end],
        motion_root_pos_eval,
        motion_root_quat_eval,
    )
    global_body_pos_error = np.linalg.norm(
        motion_pos_start - robot_pos_start,
        axis=-1,
    )
    global_body_ori_error = _orientation_error(
        motion_quat_start,
        robot_quat_start,
    )

    local_body_ori_error = _orientation_error(
        motion_quat_local[:pre_end],
        robot_quat_local[:pre_end],
    )
    local_body_pos_error = np.linalg.norm(
        motion_pos_local[:pre_end] - robot_pos_local[:pre_end],
        axis=-1,
    )

    joint_pos_error = robot_joint_pos[:pre_end] - motion_joint_pos[:pre_end]
    joint_vel_error = robot_joint_vel[:pre_end] - motion_joint_vel[:pre_end]

    robot_key_body_vel, velocity_time = _differentiate(
        robot_pos_start[:, tracking_indices],
        eval_time,
    )
    motion_key_body_vel, _ = _differentiate(
        motion_pos_start[:, tracking_indices],
        eval_time,
    )
    key_body_velocity_error = np.linalg.norm(
        robot_key_body_vel - motion_key_body_vel,
        axis=-1,
    )
    robot_key_body_acc, _ = _differentiate(robot_key_body_vel, velocity_time)
    motion_key_body_acc, _ = _differentiate(motion_key_body_vel, velocity_time)
    key_body_acceleration_error = np.linalg.norm(
        robot_key_body_acc - motion_key_body_acc,
        axis=-1,
    )

    robot_joint_acc, joint_acc_time = _differentiate(
        robot_joint_vel[:pre_end],
        eval_time,
    )
    robot_joint_jerk, _ = _differentiate(robot_joint_acc, joint_acc_time)

    root_xyz_mean, root_xyz_p95 = _mean_p95(root_position_error_xyz)
    root_xy_mean, root_xy_p95 = _mean_p95(root_position_error_xy)
    root_ori_mean, root_ori_p95 = _mean_p95(root_orientation_error)
    global_key_pos_mean, global_key_pos_p95 = _mean_p95(
        global_body_pos_error[:, tracking_indices]
    )
    global_key_ori_mean, global_key_ori_p95 = _mean_p95(
        global_body_ori_error[:, tracking_indices]
    )
    global_ee_pos_mean, global_ee_pos_p95 = _mean_p95(
        global_body_pos_error[:, end_effector_indices]
    )
    global_ee_ori_mean, global_ee_ori_p95 = _mean_p95(
        global_body_ori_error[:, end_effector_indices]
    )
    local_key_pos_mean, local_key_pos_p95 = _mean_p95(
        local_body_pos_error[:, tracking_indices]
    )
    local_key_ori_mean, local_key_ori_p95 = _mean_p95(
        local_body_ori_error[:, tracking_indices]
    )
    local_ee_pos_mean, local_ee_pos_p95 = _mean_p95(
        local_body_pos_error[:, end_effector_indices]
    )
    local_ee_ori_mean, local_ee_ori_p95 = _mean_p95(
        local_body_ori_error[:, end_effector_indices]
    )
    key_vel_mean, key_vel_p95 = _mean_p95(key_body_velocity_error)
    key_acc_mean, key_acc_p95 = _mean_p95(key_body_acceleration_error)

    root_final_error = _relative_translation(
        robot_root_pos,
        robot_root_quat,
    ) - _relative_translation(
        motion_root_pos,
        motion_root_quat,
    )
    termination_time_s = float(sim_time[termination_idx] - sim_time[0])
    evaluated_duration_s = float(eval_time[-1] - eval_time[0])

    return {
        "path": str(path),
        "policy_config": _scalar(data["policy_config"]) if "policy_config" in data else "",
        "motion_path": _scalar(data["motion_path"]) if "motion_path" in data else "",
        "seed": int(np.asarray(data["seed"]).reshape(())) if "seed" in data else -1,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "key_body_names": "|".join(names[idx] for idx in tracking_indices),
        "end_effector_names": "|".join(names[idx] for idx in end_effector_indices),
        "joint_names": "|".join(joint_names),
        "frames": int(len(frame_idx)),
        "evaluated_frames": int(pre_end),
        "evaluated_duration_s": evaluated_duration_s,
        "motion_start": int(motion_t[0]),
        "motion_end": int(motion_t[-1]),
        "motion_length": motion_length,
        "termination_idx": int(termination_idx),
        "termination_motion_t": int(motion_t[termination_idx]),
        "termination_time_s": termination_time_s,
        "termination_reason": termination_reason,
        "terminated": int(terminated),
        "success": int(not terminated),
        "completion_ratio": completion_ratio,
        "global_root_pos_xyz_mean_m": root_xyz_mean,
        "global_root_pos_xyz_p95_m": root_xyz_p95,
        "global_root_pos_xy_mean_m": root_xy_mean,
        "global_root_pos_xy_p95_m": root_xy_p95,
        "global_root_ori_mean_rad": root_ori_mean,
        "global_root_ori_p95_rad": root_ori_p95,
        "global_key_body_pos_mean_m": global_key_pos_mean,
        "global_key_body_pos_p95_m": global_key_pos_p95,
        "global_key_body_ori_mean_rad": global_key_ori_mean,
        "global_key_body_ori_p95_rad": global_key_ori_p95,
        "global_end_effector_pos_mean_m": global_ee_pos_mean,
        "global_end_effector_pos_p95_m": global_ee_pos_p95,
        "global_end_effector_ori_mean_rad": global_ee_ori_mean,
        "global_end_effector_ori_p95_rad": global_ee_ori_p95,
        "global_key_body_vel_error_mean_mps": key_vel_mean,
        "global_key_body_vel_error_p95_mps": key_vel_p95,
        "global_key_body_acc_error_mean_mps2": key_acc_mean,
        "global_key_body_acc_error_p95_mps2": key_acc_p95,
        "local_key_body_pos_mean_m": local_key_pos_mean,
        "local_key_body_pos_p95_m": local_key_pos_p95,
        "local_key_body_ori_mean_rad": local_key_ori_mean,
        "local_key_body_ori_p95_rad": local_key_ori_p95,
        "local_end_effector_pos_mean_m": local_ee_pos_mean,
        "local_end_effector_pos_p95_m": local_ee_pos_p95,
        "local_end_effector_ori_mean_rad": local_ee_ori_mean,
        "local_end_effector_ori_p95_rad": local_ee_ori_p95,
        "joint_pos_mae_rad": _mean(np.abs(joint_pos_error)),
        "joint_pos_rmse_rad": _rmse(joint_pos_error),
        "joint_pos_p95_abs_rad": _p95(np.abs(joint_pos_error)),
        "joint_vel_mae_rad_s": _mean(np.abs(joint_vel_error)),
        "joint_vel_rmse_rad_s": _rmse(joint_vel_error),
        "joint_vel_p95_abs_rad_s": _p95(np.abs(joint_vel_error)),
        "joint_acc_rms_rad_s2": _rmse(robot_joint_acc),
        "joint_acc_p95_abs_rad_s2": _p95(np.abs(robot_joint_acc)),
        "joint_jerk_rms_rad_s3": _rmse(robot_joint_jerk),
        "joint_jerk_p95_abs_rad_s3": _p95(np.abs(robot_joint_jerk)),
        # Compatibility aliases for the original metric output.
        "progress": completion_ratio,
        "global_root_tracking_error": root_xyz_mean,
        "global_root_tracking_error_xy": root_xy_mean,
        "local_body_tracking_error": local_key_pos_mean,
        "mpjpe": local_key_pos_mean,
        "root_final_error_norm": float(np.linalg.norm(root_final_error)),
        "root_final_error_xy_norm": float(np.linalg.norm(root_final_error[:2])),
    }


# Public entry point for runners that consume and then discard a trajectory.
compute_trajectory_metrics = _compute_one


def _numeric_values(rows: list[dict[str, object]], field: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        if value is None or value == "":
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            values.append(numeric)
    return np.asarray(values, dtype=np.float64)


def _aggregate_field(rows: list[dict[str, object]], field: str) -> dict[str, float | int | None]:
    values = _numeric_values(rows, field)
    if values.size == 0:
        return {"mean": None, "std": None, "valid_count": 0}
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "valid_count": int(values.size),
    }


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {"count": len(rows)}
    for group_path, fields in SUMMARY_FIELD_GROUPS:
        node = summary
        for group_name in group_path:
            child = node.setdefault(group_name, {})
            assert isinstance(child, dict)
            node = child
        for output_name, field_name in fields.items():
            node[output_name] = _aggregate_field(rows, field_name)

    outcome = summary["outcome"]
    assert isinstance(outcome, dict)
    outcome["termination_reasons"] = dict(
        sorted(Counter(str(row.get("termination_reason", "")) for row in rows).items())
    )
    return summary


def _observed_name_sets(
    rows: list[dict[str, object]],
    field: str,
) -> list[list[str]]:
    values = {
        tuple(item for item in str(row.get(field, "")).split("|") if item)
        for row in rows
    }
    return [list(value) for value in sorted(values)]


def metric_schema(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "version": METRIC_SCHEMA_VERSION,
        "tracking_window": (
            "From the first recorded policy frame through motion end, or up to but "
            "excluding the frame that confirms a tracking-failure criterion."
        ),
        "rollout_aggregation": (
            "Each rollout first reduces frame/body/joint errors to a scalar mean or "
            "p95; dataset summaries report the mean, population std, and valid rollout "
            "count of each per-rollout scalar."
        ),
        "coordinate_frames": {
            "global_start_aligned": (
                "Robot and reference each subtract their own initial root translation "
                "and remove their own initial full root orientation."
            ),
            "local_heading": (
                "Robot and reference each subtract the current pelvis XY translation "
                "and remove the current pelvis yaw; world Z and gravity alignment remain."
            ),
            "joint_space": "Matched named robot/reference joints in the G1 robot order.",
        },
        "dynamic_tracking": (
            "Finite differences use recorded sim_time in seconds; velocity and "
            "acceleration errors are computed on start-aligned key-body positions."
        ),
        "smoothness": (
            "Robot joint acceleration is d(qdot)/dt and joint jerk is "
            "d²(qdot)/dt², both using recorded sim_time."
        ),
        "failure_criteria": {
            "root_ori_error": {"threshold_rad": 1.2, "consecutive_policy_frames": 25},
            "body_pos_error": {"threshold_m": 0.4, "consecutive_policy_frames": 5},
            "body_ori_error": {"threshold_rad": 1.2, "consecutive_policy_frames": 5},
        },
        "key_body_patterns": list(TRACKING_BODY_PATTERNS),
        "end_effector_patterns": list(END_EFFECTOR_BODY_PATTERNS),
        "observed_key_body_sets": _observed_name_sets(rows, "key_body_names"),
        "observed_end_effector_sets": _observed_name_sets(rows, "end_effector_names"),
        "observed_joint_sets": _observed_name_sets(rows, "joint_names"),
        "excluded_from_v2": [
            "contact metrics",
            "composite score",
            "automatic controller ranking",
        ],
    }


# Keep the private name for callers of the previous script revision.
_summary = summarize_rows


def main() -> None:
    args = _parse_args()
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    paths = _expand_paths(args.paths, args.manifest)
    rows: list[dict[str, object]] = []
    for index, path in enumerate(paths, start=1):
        rows.append(_compute_one(path))
        if index % args.progress_every == 0 or index == len(paths):
            print(f"[metrics] processed={index}/{len(paths)}", file=sys.stderr, flush=True)
    summary = summarize_rows(rows)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["policy_config"])].append(row)
    per_policy_config = {
        policy_config: summarize_rows(policy_rows)
        for policy_config, policy_rows in grouped.items()
    }
    summary_payload = {
        "metric_schema": metric_schema(rows),
        "summary": summary,
        "per_policy_config": per_policy_config,
    }
    print_payload = dict(summary_payload)
    if args.print_rows:
        print_payload["rows"] = rows
    print(json.dumps(print_payload, indent=2, allow_nan=False))

    if args.output_csv:
        output_csv = Path(args.output_csv).expanduser()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if args.output_json:
        output_json = Path(args.output_json).expanduser()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(summary_payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
