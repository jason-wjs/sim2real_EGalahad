from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

import compute_tracking_metrics as metrics


class ComputeTrackingMetricsTest(unittest.TestCase):
    def _write_trajectory(
        self,
        path: Path,
        *,
        motion_length: int,
        robot_initial_offset_m: float = 0.0,
        robot_root_drift_m: float = 0.0,
        robot_joint_vel: np.ndarray | None = None,
    ) -> None:
        frames = 4
        body_names = np.asarray(
            ["pelvis", "torso_link", "left_toe_link", "right_wrist_yaw_link"]
        )
        joint_names = np.asarray(["left_hip_pitch_joint", "right_hip_pitch_joint"])
        body_pos = np.zeros((frames, len(body_names), 3), dtype=np.float32)
        body_quat = np.zeros((frames, len(body_names), 4), dtype=np.float32)
        body_quat[..., 0] = 1.0
        root_pos = np.zeros((frames, 3), dtype=np.float32)
        root_quat = np.zeros((frames, 4), dtype=np.float32)
        root_quat[:, 0] = 1.0
        robot_body_pos = body_pos.copy()
        robot_root_pos = root_pos.copy()
        drift = np.arange(frames, dtype=np.float32) * robot_root_drift_m
        robot_body_pos[:, :, 0] += robot_initial_offset_m + drift[:, None]
        robot_root_pos[:, 0] += robot_initial_offset_m + drift
        joint_pos = np.zeros((frames, len(joint_names)), dtype=np.float32)
        motion_joint_vel = np.zeros_like(joint_pos)
        if robot_joint_vel is None:
            robot_joint_vel = motion_joint_vel.copy()
        np.savez(
            path,
            robot_root_pos_w=robot_root_pos,
            robot_root_quat_w=root_quat,
            motion_root_pos_w=root_pos,
            motion_root_quat_w=root_quat,
            robot_body_pos_w=robot_body_pos,
            robot_body_quat_w=body_quat,
            motion_body_pos_w=body_pos,
            motion_body_quat_w=body_quat,
            body_names=body_names,
            robot_joint_pos=joint_pos,
            robot_joint_vel=robot_joint_vel,
            motion_joint_pos=joint_pos,
            motion_joint_vel=motion_joint_vel,
            joint_names=joint_names,
            sim_time=np.arange(frames, dtype=np.float32) * 0.02,
            motion_t=np.arange(frames, dtype=np.int32),
            motion_length=np.asarray(motion_length, dtype=np.int32),
        )

    def test_incomplete_trajectory_is_truncated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trajectory.npz"
            self._write_trajectory(path, motion_length=10)
            result = metrics._compute_one(path)
        self.assertEqual(result["termination_reason"], "truncated")
        self.assertEqual(result["terminated"], 1)
        self.assertAlmostEqual(result["progress"], 3.0 / 9.0)

    def test_last_motion_frame_is_motion_end(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trajectory.npz"
            self._write_trajectory(path, motion_length=4)
            result = metrics._compute_one(path)
        self.assertEqual(result["termination_reason"], "motion_end")
        self.assertEqual(result["terminated"], 0)
        self.assertEqual(result["progress"], 1.0)

    def test_root_drift_affects_global_but_not_heading_local_tracking(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trajectory.npz"
            self._write_trajectory(
                path,
                motion_length=4,
                robot_root_drift_m=0.1,
            )
            result = metrics._compute_one(path)

        self.assertAlmostEqual(result["global_root_pos_xyz_mean_m"], 0.15)
        self.assertAlmostEqual(result["global_key_body_pos_mean_m"], 0.15)
        self.assertAlmostEqual(result["local_key_body_pos_mean_m"], 0.0)

    def test_start_alignment_removes_constant_initial_world_offset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trajectory.npz"
            self._write_trajectory(
                path,
                motion_length=4,
                robot_initial_offset_m=2.0,
            )
            result = metrics._compute_one(path)

        self.assertAlmostEqual(result["global_root_pos_xyz_mean_m"], 0.0)
        self.assertAlmostEqual(result["global_key_body_pos_mean_m"], 0.0)

    def test_joint_jerk_uses_recorded_seconds(self) -> None:
        time_s = np.arange(4, dtype=np.float32) * 0.02
        joint_vel = np.repeat((time_s**2)[:, None], 2, axis=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trajectory.npz"
            self._write_trajectory(
                path,
                motion_length=4,
                robot_joint_vel=joint_vel,
            )
            result = metrics._compute_one(path)

        self.assertAlmostEqual(result["joint_jerk_rms_rad_s3"], 2.0, places=4)

    def test_manifest_filters_failures_and_resolves_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            successful = root / "successful.npz"
            failed = root / "failed.npz"
            successful.touch()
            failed.touch()
            manifest = root / "runs.csv"
            with manifest.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["trajectory_path", "status"])
                writer.writeheader()
                writer.writerow({"trajectory_path": successful.name, "status": "succeeded"})
                writer.writerow({"trajectory_path": failed.name, "status": "failed"})

            paths = metrics._expand_paths([], [str(manifest)])

        self.assertEqual(paths, [successful.resolve()])

    def test_main_prints_and_writes_summary_without_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trajectory = root / "trajectory.npz"
            output_csv = root / "metrics.csv"
            output_json = root / "metrics.json"
            self._write_trajectory(trajectory, motion_length=4)
            argv = [
                "compute_tracking_metrics.py",
                str(trajectory),
                "--output-csv",
                str(output_csv),
                "--output-json",
                str(output_json),
            ]
            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch.object(sys, "argv", argv), redirect_stdout(stdout), redirect_stderr(stderr):
                metrics.main()

            printed = json.loads(stdout.getvalue())
            saved = json.loads(output_json.read_text(encoding="utf-8"))
            with output_csv.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertNotIn("rows", printed)
        self.assertNotIn("rows", saved)
        self.assertEqual(saved["summary"]["count"], 1)
        self.assertIn("global_start_aligned", saved["summary"]["tracking"])
        self.assertEqual(
            saved["summary"]["smoothness"]["joint_jerk_rms_rad_s3"]["valid_count"],
            1,
        )
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
