from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

import view_motion


class ViewMotionTest(unittest.TestCase):
    def test_parse_viser_viewer(self) -> None:
        args = view_motion._parse_args(["--motion", "motion.npz", "--viewer", "viser"])
        self.assertEqual(args.viewer, "viser")
        self.assertFalse(args.headless)

    def test_frame_selection_matches_existing_behavior(self) -> None:
        self.assertEqual(list(view_motion._iter_frame_indices(10, 2, 9, 3)), [2, 5, 8])
        self.assertEqual(list(view_motion._iter_frame_indices(4, 0, -1, 0)), [0, 1, 2, 3])

    def test_apply_qpos_frame_zeros_velocity(self) -> None:
        data = SimpleNamespace(qpos=np.zeros(3), qvel=np.ones(2))
        view_motion._apply_qpos_frame(data, np.asarray([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(data.qpos, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(data.qvel, [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
