from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock

from sim2real.sim_env.integrated_sim2sim import _resolve_motion_mjcf_path


class IntegratedSim2SimAssetsTest(unittest.TestCase):
    def test_policy_specific_motion_mjcf_takes_precedence(self) -> None:
        robot_cfg = Mock()

        selected = _resolve_motion_mjcf_path(
            robot_cfg,
            {"mjcf_path": "checkpoints/bfm-zero/mjcf/g1_for_reward_inference.xml"},
        )

        self.assertEqual(
            selected,
            "checkpoints/bfm-zero/mjcf/g1_for_reward_inference.xml",
        )
        robot_cfg.resolve_mjcf_path.assert_not_called()

    def test_common_robot_mjcf_is_default_for_motion_fk(self) -> None:
        robot_cfg = Mock()
        robot_cfg.resolve_mjcf_path.return_value = Path("/runtime/g1.xml")

        selected = _resolve_motion_mjcf_path(robot_cfg, {})

        self.assertEqual(selected, Path("/runtime/g1.xml"))
        robot_cfg.resolve_mjcf_path.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
