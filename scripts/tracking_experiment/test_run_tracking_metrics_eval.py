from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import run_tracking_metrics_eval as evaluator


class RunTrackingMetricsEvalTest(unittest.TestCase):
    def test_rejects_noncomparable_metric_body_layouts(self) -> None:
        rows = [
            {
                "key_body_names": "pelvis|left_toe_link",
                "end_effector_names": "left_toe_link",
                "joint_names": "left_hip_pitch_joint",
            },
            {
                "key_body_names": "pelvis",
                "end_effector_names": "",
                "joint_names": "left_hip_pitch_joint",
            },
        ]

        with self.assertRaisesRegex(RuntimeError, "shared key_body_names layout"):
            evaluator._validate_metric_layouts(rows)

    def test_batch_eval_disables_hugging_face_network_by_default(self) -> None:
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "0"}, clear=False):
            evaluator._configure_asset_network(allow_network_assets=False)
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")

    def test_network_assets_can_be_explicitly_allowed(self) -> None:
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}, clear=False):
            evaluator._configure_asset_network(allow_network_assets=True)
            self.assertNotIn("HF_HUB_OFFLINE", os.environ)

    def test_motion_paths_default_to_all_and_allow_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            motions = root / "motions"
            (motions / "a").mkdir(parents=True)
            (motions / "b").mkdir(parents=True)
            first = motions / "a" / "motion.npz"
            second = motions / "b" / "motion.npz"
            first.touch()
            second.touch()

            all_paths = evaluator._motion_paths(root, None)
            limited_paths = evaluator._motion_paths(root, 1)

        self.assertEqual(all_paths, [first.resolve(), second.resolve()])
        self.assertEqual(limited_paths, [first.resolve()])

    def test_policy_summary_reads_detail_rows_from_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trajectory = root / "trajectory.npz"
            trajectory.touch()
            result_json = root / "metrics.json"
            result_csv = root / "metrics.csv"
            summary = {
                "count": 1,
                "outcome": {
                    "completion_ratio": {"mean": 1.0, "std": 0.0, "valid_count": 1}
                },
            }
            result_json.write_text(
                json.dumps(
                    {
                        "metric_schema": {"version": "2.0.0"},
                        "summary": summary,
                        "per_policy_config": {},
                    }
                ),
                encoding="utf-8",
            )
            metric_row = {
                "path": str(trajectory),
                "completion_ratio": 1.0,
                "success": 1,
                "terminated": 0,
                "termination_reason": "motion_end",
                "progress": 1.0,
                "global_root_tracking_error": 0.1,
                "global_root_tracking_error_xy": 0.1,
                "local_body_tracking_error": 0.1,
                "mpjpe": 0.1,
                "root_final_error_norm": 0.1,
                "root_final_error_xy_norm": 0.1,
            }
            with result_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(metric_row))
                writer.writeheader()
                writer.writerow(metric_row)
            run_rows = [
                {
                    "policy": "test_policy",
                    "motion_index": 0,
                    "trajectory_path": str(trajectory),
                }
            ]

            result = evaluator._add_policy_summary(result_json, result_csv, run_rows)
            saved_json = json.loads(result_json.read_text(encoding="utf-8"))
            with result_csv.open("r", encoding="utf-8", newline="") as f:
                saved_rows = list(csv.DictReader(f))

        self.assertEqual(result["per_controller"]["test_policy"]["count"], 1)
        self.assertNotIn("rows", saved_json)
        self.assertIn("per_controller", saved_json)
        self.assertEqual(saved_rows[0]["policy"], "test_policy")

    def test_summary_only_checkpoints_metrics_and_removes_success_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            motion = root / "motions" / "clip" / "motion.npz"
            motion.parent.mkdir(parents=True)
            motion.touch()
            output_dir = root / "output"
            args = Namespace(
                motions_root=str(root),
                motion_list=None,
                output_dir=str(output_dir),
                num_motions=None,
                seeds=[0],
                initial_pause_s=0.0,
                max_runtime_s=None,
                robot="g1",
                policy=["humanoid_gpt=checkpoints/humanoid-gpt/policy.yaml"],
                skip_existing=False,
                retention="summary-only",
                fail_fast=False,
                motion_index_offset=0,
                checkpoint_every=1,
            )

            def fake_rollout(cmd: list[str], log_path: Path) -> int:
                trajectory = Path(cmd[cmd.index("--trajectory-output") + 1])
                trajectory.parent.mkdir(parents=True, exist_ok=True)
                trajectory.touch()
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text("ok", encoding="utf-8")
                return 0

            metric_row = {
                "path": "temporary.npz",
                "policy_config": "",
                "motion_path": str(motion),
                "seed": 0,
                "key_body_names": "pelvis",
                "end_effector_names": "left_toe_link",
                "joint_names": "left_hip_pitch_joint",
                "termination_reason": "motion_end",
                "terminated": 0,
                "success": 1,
                "completion_ratio": 1.0,
            }
            with (
                patch.object(evaluator, "_parse_args", return_value=args),
                patch.object(evaluator, "_run_rollout", side_effect=fake_rollout) as run_rollout,
                patch.object(evaluator, "_valid_trajectory", return_value=True),
                patch.object(
                    evaluator,
                    "compute_trajectory_metrics",
                    return_value=dict(metric_row),
                ),
                redirect_stdout(io.StringIO()),
            ):
                evaluator.main()

            trajectory_files = list((output_dir / "trajectories").rglob("*.npz"))
            log_files = list((output_dir / "logs").rglob("*.log"))
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            checkpoints = evaluator._load_metric_checkpoints(
                output_dir / "checkpoints" / "rollout_metrics.jsonl"
            )

            self.assertEqual(run_rollout.call_count, 1)
            self.assertEqual(trajectory_files, [])
            self.assertEqual(log_files, [])
            self.assertEqual(len(checkpoints), 1)
            self.assertEqual(summary["retention"]["mode"], "summary-only")
            self.assertEqual(
                summary["all_rollouts"]["outcome"]["completion_ratio"]["mean"],
                1.0,
            )
            self.assertEqual(summary["runs"], {"total": 1, "successful": 1, "failed": 0})
            self.assertFalse((output_dir / "tracking_metrics.csv").exists())

            args.skip_existing = True
            with (
                patch.object(evaluator, "_parse_args", return_value=args),
                patch.object(evaluator, "_run_rollout") as resumed_rollout,
                redirect_stdout(io.StringIO()),
            ):
                evaluator.main()
            resumed_rollout.assert_not_called()


if __name__ == "__main__":
    unittest.main()
