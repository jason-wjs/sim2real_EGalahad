from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

import view_raw_motion


class ViewRawMotionTest(unittest.TestCase):
    def test_parse_defaults_to_five_motions(self) -> None:
        args = view_raw_motion._parse_args(["--input", "dataset"])
        self.assertEqual(args.num_motions, 5)
        self.assertEqual(args.body_quat_order, "wxyz")

    def test_source_discovery_uses_origin_directory_and_stops_early(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            scan_root = dataset_root / "origin_interp10_NPZ"
            for name in ("c", "a", "b", "d"):
                path = scan_root / name / "motion.npz"
                path.parent.mkdir(parents=True)
                path.touch()

            resolved_root, paths = view_raw_motion._resolve_source_paths(
                dataset_root,
                3,
            )

            self.assertEqual(resolved_root, scan_root)
            self.assertEqual(
                [path.relative_to(scan_root).as_posix() for path in paths],
                ["a/motion.npz", "b/motion.npz", "c/motion.npz"],
            )

    def test_body_edges_follow_target_hierarchy(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <worldbody>
                <body name="pelvis">
                  <body name="hip">
                    <body name="knee"/>
                  </body>
                </body>
              </worldbody>
            </mujoco>
            """
        )
        edges = view_raw_motion._body_edges(
            model,
            ("pelvis", "hip", "knee"),
        )
        np.testing.assert_array_equal(edges, [[0, 1], [1, 2]])

    def test_body_quaternion_conversion_and_normalization(self) -> None:
        xyzw = np.asarray([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)
        actual = view_raw_motion._body_quats_to_wxyz(xyzw, "xyzw")
        expected = np.asarray([[[4.0, 1.0, 2.0, 3.0]]], dtype=np.float32)
        expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
        np.testing.assert_allclose(actual, expected)

    def test_frame_indices_match_requested_slice(self) -> None:
        self.assertEqual(view_raw_motion._frame_indices(10, 2, 9, 3), [2, 5, 8])


if __name__ == "__main__":
    unittest.main()
