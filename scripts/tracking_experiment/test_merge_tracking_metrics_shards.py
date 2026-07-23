from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from merge_tracking_metrics_shards import merge_shards


RUN_FIELDNAMES = [
    "policy",
    "policy_config",
    "motion_index",
    "motion_path",
    "seed",
    "trajectory_path",
    "status",
    "exit_code",
    "failure_reason",
    "log_path",
    "trajectory_retained",
]


def _metric_row(root: Path, controller: str, motion_index: int) -> dict[str, object]:
    motion_path = root / "motions" / f"{motion_index}.npz"
    return {
        "path": str(root / "temporary.npz"),
        "policy": controller,
        "policy_config": str(root / controller / "policy.yaml"),
        "motion_index": motion_index,
        "motion_path": str(motion_path),
        "seed": 0,
        "key_body_names": "pelvis",
        "end_effector_names": "left_toe_link",
        "joint_names": "left_hip_pitch_joint",
        "termination_reason": "motion_end",
        "terminated": 0,
        "success": 1,
        "completion_ratio": 1.0,
    }


def _run_row(metric_row: dict[str, object]) -> dict[str, object]:
    return {
        "policy": metric_row["policy"],
        "policy_config": metric_row["policy_config"],
        "motion_index": metric_row["motion_index"],
        "motion_path": metric_row["motion_path"],
        "seed": metric_row["seed"],
        "trajectory_path": metric_row["path"],
        "status": "succeeded",
        "exit_code": 0,
        "failure_reason": "",
        "log_path": "",
        "trajectory_retained": 0,
    }


def _write_shard(shard_dir: Path, metric_rows: list[dict[str, object]]) -> None:
    checkpoint_path = shard_dir / "checkpoints" / "rollout_metrics.jsonl"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_text(
        "".join(json.dumps(row) + "\n" for row in metric_rows),
        encoding="utf-8",
    )
    with (shard_dir / "runs.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_FIELDNAMES)
        writer.writeheader()
        writer.writerows(_run_row(row) for row in metric_rows)


class MergeTrackingMetricsShardsTest(unittest.TestCase):
    def test_merges_complete_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shards_root = root / "shards"
            output_dir = root / "merged"
            first = _metric_row(root, "humanoid_gpt", 0)
            second = _metric_row(root, "twist2", 1)
            _write_shard(shards_root / "humanoid_gpt" / "shard_00", [first])
            _write_shard(shards_root / "twist2" / "shard_00", [second])

            summary = merge_shards(
                shards_root,
                output_dir,
                expected_rollouts=2,
                expected_controllers={"humanoid_gpt", "twist2"},
            )

            merged_rows = [
                json.loads(line)
                for line in (
                    output_dir / "checkpoints" / "rollout_metrics.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(summary["runs"], {"total": 2, "successful": 2, "failed": 0})
            self.assertEqual(set(summary["per_controller"]), {"humanoid_gpt", "twist2"})
            self.assertEqual(len(merged_rows), 2)

    def test_rejects_duplicate_rollout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shards_root = root / "shards"
            row = _metric_row(root, "humanoid_gpt", 0)
            _write_shard(shards_root / "shard_00", [row])
            _write_shard(shards_root / "shard_01", [row])

            with self.assertRaisesRegex(RuntimeError, "Duplicate metric rollout"):
                merge_shards(
                    shards_root,
                    root / "merged",
                    expected_rollouts=2,
                    expected_controllers={"humanoid_gpt"},
                )


if __name__ == "__main__":
    unittest.main()
