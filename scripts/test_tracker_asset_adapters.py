from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from sim2real.rl_policy.observations.teleopit import _resolve_xml_path
from sim2real.rl_policy.observations.wxy_wbc import (
    WXY_WBC_ACTION_DIM,
    WXY_WBC_HISTORY_LENGTH,
    WXY_WBC_MOTION_COMMAND_DIM,
    WXY_WBC_OBS_DIM,
    WXY_WBC_PROPRIOCEPTION_DIM,
    wxy_wbc_policy_obs,
)
from sim2real.rl_policy.utils.motion import _any4hdmi_manifest_override_view


class TrackerAssetAdaptersTest(unittest.TestCase):
    def test_teleopit_robot_cfg_keeps_logical_symlink_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot = root / "snapshot"
            blob = root / "blobs" / "robot.xml"
            snapshot.mkdir()
            blob.parent.mkdir()
            blob.write_text("<mujoco/>", encoding="utf-8")
            logical_path = snapshot / "robot.xml"
            logical_path.symlink_to(blob)

            robot_cfg = Mock()
            robot_cfg.resolve_mjcf_path.return_value = logical_path
            env = SimpleNamespace(robot_cfg=robot_cfg)

            selected = _resolve_xml_path(env, "robot_cfg")

            self.assertEqual(selected, logical_path.absolute())
            self.assertNotEqual(selected, blob.resolve())

    def test_teleopit_robot_cfg_requires_runtime_robot_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires env.robot_cfg"):
            _resolve_xml_path(SimpleNamespace(), "robot_cfg")

    def test_manifest_override_accepts_symlinked_motion_blob(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            motion_dir = dataset / "motions" / "walk"
            blob = root / "blobs" / "walk.npz"
            motion_dir.mkdir(parents=True)
            blob.parent.mkdir()
            blob.write_bytes(b"npz-placeholder")
            (motion_dir / "motion.npz").symlink_to(blob)
            (dataset / "manifest.json").write_text(
                json.dumps({"motions_subdir": "motions"}),
                encoding="utf-8",
            )
            mjcf = root / "g1.xml"
            mjcf.write_text("<mujoco/>", encoding="utf-8")

            view_path = Path(
                _any4hdmi_manifest_override_view(
                    root_path=dataset,
                    base_dir=root,
                    mjcf_path=mjcf,
                )
            )

            materialized = view_path / "motions" / "walk" / "motion.npz"
            self.assertTrue(materialized.is_file())
            self.assertEqual(materialized.read_bytes(), blob.read_bytes())
            view_manifest = json.loads(
                (view_path / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(view_manifest["mjcf"], str(mjcf))

    def test_wxy_dimensions_match_actor_contract(self) -> None:
        limb_pose_dim = 4 * 9
        expected_motion_command_dim = (
            WXY_WBC_ACTION_DIM * 2
            + WXY_WBC_HISTORY_LENGTH * limb_pose_dim
            + 3
        )
        expected_proprioception_dim = WXY_WBC_HISTORY_LENGTH * (
            limb_pose_dim
            + 3
            + 3
            + WXY_WBC_ACTION_DIM
            + WXY_WBC_ACTION_DIM
            + WXY_WBC_ACTION_DIM
        )

        self.assertEqual(WXY_WBC_MOTION_COMMAND_DIM, expected_motion_command_dim)
        self.assertEqual(WXY_WBC_PROPRIOCEPTION_DIM, expected_proprioception_dim)
        self.assertEqual(
            WXY_WBC_OBS_DIM,
            expected_motion_command_dim + expected_proprioception_dim,
        )

    def test_wxy_history_initializes_then_shifts_oldest_to_newest(self) -> None:
        history = np.zeros((WXY_WBC_HISTORY_LENGTH, 2), dtype=np.float32)
        initial = np.asarray([1.0, 2.0], dtype=np.float32)
        update = np.asarray([3.0, 4.0], dtype=np.float32)

        wxy_wbc_policy_obs._append_history(history, initial, initialize=True)
        np.testing.assert_array_equal(
            history,
            np.broadcast_to(initial, history.shape),
        )

        wxy_wbc_policy_obs._append_history(history, update, initialize=False)
        np.testing.assert_array_equal(
            history[:-1],
            np.broadcast_to(initial, history[:-1].shape),
        )
        np.testing.assert_array_equal(history[-1], update)


if __name__ == "__main__":
    unittest.main()
