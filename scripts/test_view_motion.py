from __future__ import annotations

import unittest
from pathlib import Path
from queue import SimpleQueue
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

import view_motion


class ViewMotionTest(unittest.TestCase):
    def test_parse_viser_viewer(self) -> None:
        args = view_motion._parse_args(["--motion", "motion.npz", "--viewer", "viser"])
        self.assertEqual(args.viewer, "viser")
        self.assertFalse(args.headless)

    def test_parse_viser_dataset(self) -> None:
        args = view_motion._parse_args(["--dataset", "dataset", "--viewer", "viser"])
        self.assertEqual(args.dataset, "dataset")
        self.assertIsNone(args.motion)

    def test_frame_selection_matches_existing_behavior(self) -> None:
        self.assertEqual(list(view_motion._iter_frame_indices(10, 2, 9, 3)), [2, 5, 8])
        self.assertEqual(list(view_motion._iter_frame_indices(4, 0, -1, 0)), [0, 1, 2, 3])

    def test_discover_motion_paths_recursively_and_sort(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            first = dataset_root / "motions" / "a" / "motion.npz"
            second = dataset_root / "motions" / "b" / "motion.npz"
            second.parent.mkdir(parents=True)
            first.parent.mkdir(parents=True)
            second.touch()
            first.touch()
            (first.parent / "ignored.txt").touch()

            self.assertEqual(
                view_motion._discover_motion_paths(dataset_root, "motions"),
                [first, second],
            )
            self.assertEqual(
                view_motion._motion_label(second, dataset_root / "motions"),
                "b/motion.npz",
            )

    def test_discover_motion_paths_uses_conversion_records(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir)
            motions_root = dataset_root / "motions"
            motions_root.mkdir()
            records = [
                '{"output": "motions/b/motion.npz"}',
                '{"output": "motions/a/motion.npz"}',
            ]
            (dataset_root / "conversion_records.jsonl").write_text(
                chr(10).join(records) + chr(10),
                encoding="utf-8",
            )

            self.assertEqual(
                view_motion._discover_motion_paths(
                    dataset_root,
                    "motions",
                    expected_count=2,
                ),
                [
                    motions_root / "a" / "motion.npz",
                    motions_root / "b" / "motion.npz",
                ],
            )

    def test_adjacent_motion_index_wraps(self) -> None:
        self.assertEqual(view_motion._adjacent_motion_index(0, 3, -1), 2)
        self.assertEqual(view_motion._adjacent_motion_index(2, 3, 1), 0)

    def test_latest_motion_selection_drains_queue(self) -> None:
        requests: SimpleQueue[int] = SimpleQueue()
        requests.put(1)
        requests.put(4)
        self.assertEqual(view_motion._latest_motion_selection(requests, 0), 4)
        self.assertEqual(view_motion._latest_motion_selection(requests, 4), 4)

    def test_apply_qpos_frame_zeros_velocity(self) -> None:
        data = SimpleNamespace(qpos=np.zeros(3), qvel=np.ones(2))
        view_motion._apply_qpos_frame(data, np.asarray([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(data.qpos, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(data.qvel, [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
