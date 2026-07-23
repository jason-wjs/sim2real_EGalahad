from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

from compute_tracking_metrics import (
    REQUIRED_TRAJECTORY_KEYS,
    compute_trajectory_metrics,
    metric_schema,
    summarize_rows,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the integrated MuJoCo evaluator and compute unified outcome, "
            "tracking, and smoothness metrics."
        )
    )
    parser.add_argument("--motions-root", required=True, help="Motion directory or dataset root.")
    parser.add_argument("--motion-list", default=None, help="Optional newline-delimited motion path list.")
    parser.add_argument("--output-dir", required=True, help="Evaluation output dir.")
    parser.add_argument(
        "--num-motions",
        type=int,
        default=None,
        help="Evaluate first N motions (default: all motions).",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="Seeds to run.")
    parser.add_argument("--initial-pause-s", type=float, default=5.0)
    parser.add_argument(
        "--max-runtime-s",
        type=float,
        default=None,
        help="Optional simulated-time limit forwarded to integrated_sim2sim.",
    )
    parser.add_argument("--robot", default="g1")
    parser.add_argument("--policy", action="append", default=[], help="Policy as name=path.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--retention",
        choices=("trajectories", "summary-only"),
        default="trajectories",
        help=(
            "Keep every trajectory (default), or checkpoint per-rollout metric scalars "
            "and delete trajectories after successful metric computation."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first failed rollout instead of recording it and continuing.",
    )
    parser.add_argument(
        "--motion-index-offset",
        type=int,
        default=0,
        help="Offset added to motion indices when naming trajectories for sharded motion lists.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Checkpoint runs.csv and print progress every N rollouts.",
    )
    parser.add_argument(
        "--allow-network-assets",
        action="store_true",
        help=(
            "Allow rollout subprocesses to fetch Hugging Face assets. Batch evaluation "
            "is offline by default so missing runtime artifacts fail deterministically."
        ),
    )
    return parser.parse_args()


def _configure_asset_network(*, allow_network_assets: bool) -> None:
    if allow_network_assets:
        os.environ.pop("HF_HUB_OFFLINE", None)
        return
    os.environ["HF_HUB_OFFLINE"] = "1"


def _policy_map(items: list[str]) -> dict[str, str]:
    policies: dict[str, str] = {}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"--policy must be name=path, got {item!r}")
        policies[name] = path
    if not policies:
        raise ValueError("At least one --policy name=path is required.")
    return policies


def _motion_paths(
    motions_root: Path,
    count: int | None,
    motion_list: str | None = None,
) -> list[Path]:
    if motion_list is not None:
        paths: list[Path] = []
        for line in Path(motion_list).expanduser().read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            path = Path(stripped).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.append(path)
        if count is not None and len(paths) < count:
            raise RuntimeError(f"Expected at least {count} motions in {motion_list}, got {len(paths)}")
        return paths if count is None else paths[:count]

    root = motions_root.expanduser().resolve()
    scan_root = root / "motions" if (root / "motions").is_dir() else root
    motions = sorted(scan_root.rglob("*.npz"))
    if count is not None and len(motions) < count:
        raise RuntimeError(f"Expected at least {count} motions under {scan_root}, got {len(motions)}")
    return motions if count is None else motions[:count]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _run_rollout(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {shlex.join(cmd)}\n")
        log_file.flush()
        result = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(result.returncode)


def _valid_trajectory(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            if not REQUIRED_TRAJECTORY_KEYS.issubset(data.files):
                return False
            motion_t = np.asarray(data["motion_t"])
            return motion_t.ndim == 1 and motion_t.size > 0
    except (OSError, ValueError, KeyError):
        return False


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def _rollout_key(
    policy: str,
    policy_config: str,
    motion_index: int,
    motion_path: str,
    seed: int,
) -> tuple[str, str, int, str, int]:
    return (
        policy,
        str(Path(policy_config).expanduser().resolve()),
        int(motion_index),
        str(Path(motion_path).expanduser().resolve()),
        int(seed),
    )


def _metric_row_key(row: dict[str, object]) -> tuple[str, str, int, str, int]:
    return _rollout_key(
        str(row["policy"]),
        str(row["policy_config"]),
        int(row["motion_index"]),
        str(row["motion_path"]),
        int(row["seed"]),
    )


def _append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _load_metric_checkpoints(
    path: Path,
) -> dict[tuple[str, str, int, str, int], dict[str, object]]:
    rows: dict[tuple[str, str, int, str, int], dict[str, object]] = {}
    if not path.is_file():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid metric checkpoint at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Metric checkpoint is not an object at {path}:{line_number}")
        rows[_metric_row_key(row)] = row
    return rows


def _summary_only_outputs(
    output_dir: Path,
    metric_rows: list[dict[str, object]],
    run_rows: list[dict[str, object]],
) -> dict[str, object]:
    controller_groups: dict[str, list[dict[str, object]]] = {}
    config_groups: dict[str, list[dict[str, object]]] = {}
    for metric_row in metric_rows:
        controller_groups.setdefault(str(metric_row["policy"]), []).append(metric_row)
        config_groups.setdefault(str(metric_row["policy_config"]), []).append(metric_row)

    schema = metric_schema(metric_rows)
    all_rollouts = summarize_rows(metric_rows)
    per_controller = {
        policy_name: summarize_rows(rows)
        for policy_name, rows in controller_groups.items()
    }
    tracking_payload = {
        "metric_schema": schema,
        "summary": all_rollouts,
        "per_policy_config": {
            policy_config: summarize_rows(rows)
            for policy_config, rows in config_groups.items()
        },
        "per_controller": per_controller,
    }
    (output_dir / "tracking_metrics.json").write_text(
        json.dumps(tracking_payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    failed = sum(str(row["status"]) == "failed" for row in run_rows)
    summary = {
        "metric_schema": schema,
        "all_rollouts": all_rollouts,
        "per_controller": per_controller,
        "runs": {
            "total": len(run_rows),
            "successful": len(run_rows) - failed,
            "failed": failed,
        },
        "retention": {
            "mode": "summary-only",
            "successful_trajectories_retained": 0,
            "metric_checkpoint": "checkpoints/rollout_metrics.jsonl",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _add_policy_summary(
    result_json: Path,
    result_csv: Path,
    run_rows: list[dict[str, object]],
) -> dict[str, object]:
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    with result_csv.open("r", encoding="utf-8", newline="") as f:
        result_rows = list(csv.DictReader(f))
    if len(result_rows) != len(run_rows):
        raise RuntimeError(f"Expected {len(run_rows)} result rows, got {len(result_rows)}")

    groups: dict[str, list[dict[str, object]]] = {}
    for result_row, run_row in zip(result_rows, run_rows, strict=True):
        result_path = Path(str(result_row["path"])).resolve()
        run_path = Path(str(run_row["trajectory_path"])).resolve()
        if result_path != run_path:
            raise RuntimeError(
                "Result row order does not match run manifest: "
                f"result={result_path}, run={run_path}"
            )
        result_row["policy"] = run_row["policy"]
        result_row["motion_index"] = run_row["motion_index"]
        result_row["trajectory_path"] = run_row["trajectory_path"]
        groups.setdefault(str(run_row["policy"]), []).append(result_row)

    per_controller = {
        policy_name: summarize_rows(rows)
        for policy_name, rows in groups.items()
    }

    payload["per_controller"] = per_controller
    result_json.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    _write_csv(result_csv, result_rows, list(result_rows[0].keys()))

    return {
        "metric_schema": payload["metric_schema"],
        "all_rollouts": payload["summary"],
        "per_controller": per_controller,
    }


def main() -> None:
    args = _parse_args()
    _configure_asset_network(
        allow_network_assets=bool(getattr(args, "allow_network_assets", False))
    )
    if args.num_motions is not None and args.num_motions <= 0:
        raise ValueError("--num-motions must be positive")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    policies = _policy_map(args.policy)
    motions = _motion_paths(Path(args.motions_root), args.num_motions, args.motion_list)

    rows: list[dict[str, object]] = []
    successful_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    metric_checkpoint_path = output_dir / "checkpoints" / "rollout_metrics.jsonl"
    metric_checkpoints: dict[
        tuple[str, str, int, str, int], dict[str, object]
    ] = {}
    if args.retention == "summary-only" and metric_checkpoint_path.exists():
        if not args.skip_existing:
            raise RuntimeError(
                f"Summary-only checkpoint already exists: {metric_checkpoint_path}. "
                "Use --skip-existing to resume or choose a new output directory."
            )
        metric_checkpoints = _load_metric_checkpoints(metric_checkpoint_path)
    fieldnames = [
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
    total_rollouts = len(policies) * len(motions) * len(args.seeds)
    print(
        f"[eval] motions={len(motions)} policies={len(policies)} "
        f"seeds={len(args.seeds)} rollouts={total_rollouts}",
        flush=True,
    )
    for policy_name, policy_config in policies.items():
        for motion_index, motion_path in enumerate(motions):
            output_motion_index = args.motion_index_offset + motion_index
            motion_slug = motion_path.stem
            for seed in args.seeds:
                rollout_key = _rollout_key(
                    policy_name,
                    policy_config,
                    output_motion_index,
                    str(motion_path),
                    seed,
                )
                traj_path = (
                    output_dir
                    / "trajectories"
                    / policy_name
                    / f"seed_{seed}"
                    / f"{output_motion_index:02d}_{motion_slug}.npz"
                )
                log_path = (
                    output_dir
                    / "logs"
                    / policy_name
                    / f"seed_{seed}"
                    / f"{output_motion_index:02d}_{motion_slug}.log"
                )
                status = "skipped_existing"
                exit_code = 0
                failure_reason = ""
                checkpoint_metric = metric_checkpoints.get(rollout_key)
                if args.retention == "summary-only" and checkpoint_metric is not None:
                    metric_row = dict(checkpoint_metric)
                    metric_rows.append(metric_row)
                    traj_path.unlink(missing_ok=True)
                    if int(metric_row["success"]):
                        log_path.unlink(missing_ok=True)
                elif not (args.skip_existing and _valid_trajectory(traj_path)):
                    cmd = [
                        sys.executable,
                        "-m",
                        "sim2real.sim_env.integrated_sim2sim",
                        "--robot",
                        args.robot,
                        "--policy-config",
                        policy_config,
                        "--motion-path",
                        str(motion_path),
                        "--headless",
                        "--run-once",
                        "--initial-pause-s",
                        str(args.initial_pause_s),
                        "--trajectory-output",
                        str(traj_path),
                        "--trajectory-policy-frames-only",
                        "--seed",
                        str(seed),
                    ]
                    if args.max_runtime_s is not None:
                        cmd.extend(["--max-runtime-s", str(args.max_runtime_s)])
                    exit_code = _run_rollout(cmd, log_path)
                    if exit_code != 0:
                        status = "failed"
                        failure_reason = "subprocess_exit"
                    elif not _valid_trajectory(traj_path):
                        status = "failed"
                        failure_reason = "invalid_trajectory"
                    else:
                        status = "succeeded"
                metric_row = None
                if (
                    args.retention == "summary-only"
                    and checkpoint_metric is None
                    and status != "failed"
                ):
                    try:
                        metric_row = compute_trajectory_metrics(traj_path)
                    except Exception as exc:  # Preserve diagnostics and continue the batch.
                        status = "failed"
                        failure_reason = f"metric_compute:{type(exc).__name__}"
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        with log_path.open("a", encoding="utf-8") as log_file:
                            log_file.write("\n[metrics] Failed to compute metrics:\n")
                            log_file.write(traceback.format_exc())
                    else:
                        metric_row.update(
                            {
                                "policy": policy_name,
                                "policy_config": policy_config,
                                "motion_index": output_motion_index,
                                "motion_path": str(motion_path),
                                "seed": seed,
                                "log_path": str(log_path),
                            }
                        )
                        _append_jsonl(metric_checkpoint_path, metric_row)
                        metric_checkpoints[rollout_key] = metric_row
                        metric_rows.append(metric_row)
                        traj_path.unlink(missing_ok=True)
                        if int(metric_row["success"]):
                            log_path.unlink(missing_ok=True)
                row = {
                    "policy": policy_name,
                    "policy_config": policy_config,
                    "motion_index": output_motion_index,
                    "motion_path": str(motion_path),
                    "seed": seed,
                    "trajectory_path": str(traj_path),
                    "status": status,
                    "exit_code": exit_code,
                    "failure_reason": failure_reason,
                    "log_path": str(log_path),
                    "trajectory_retained": int(traj_path.is_file()),
                }
                rows.append(row)
                if status == "failed":
                    failed_rows.append(row)
                    print(
                        f"[FAIL] policy={policy_name} motion={motion_path} seed={seed} "
                        f"exit={exit_code} log={log_path}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if args.fail_fast:
                        _write_csv(output_dir / "runs.csv", rows, fieldnames)
                        _write_csv(output_dir / "failed_runs.csv", failed_rows, fieldnames)
                        if args.retention == "summary-only" and metric_rows:
                            _summary_only_outputs(output_dir, metric_rows, rows)
                        raise RuntimeError(f"Rollout failed; see {log_path}")
                else:
                    successful_rows.append(row)
                completed = len(rows)
                if completed % args.checkpoint_every == 0 or completed == total_rollouts:
                    _write_csv(output_dir / "runs.csv", rows, fieldnames)
                    _write_csv(output_dir / "failed_runs.csv", failed_rows, fieldnames)
                    if args.retention == "summary-only" and metric_rows:
                        _summary_only_outputs(output_dir, metric_rows, rows)
                    print(
                        f"[eval] completed={completed}/{total_rollouts} "
                        f"successful={len(successful_rows)} failed={len(failed_rows)}",
                        flush=True,
                    )

    manifest_path = output_dir / "runs.csv"
    _write_csv(manifest_path, rows, fieldnames)
    _write_csv(output_dir / "failed_runs.csv", failed_rows, fieldnames)

    if not successful_rows:
        raise RuntimeError(f"No successful rollouts; see {output_dir / 'failed_runs.csv'}")

    if args.retention == "summary-only":
        if not metric_rows:
            raise RuntimeError(f"No metrics were computed; see {output_dir / 'failed_runs.csv'}")
        summary = _summary_only_outputs(output_dir, metric_rows, rows)
        print(json.dumps(summary, indent=2, allow_nan=False))
        return

    result_json = output_dir / "tracking_metrics.json"
    result_csv = output_dir / "tracking_metrics.csv"
    _run(
        [
            sys.executable,
            str(SCRIPT_DIR / "compute_tracking_metrics.py"),
            "--manifest",
            str(manifest_path),
            "--output-json",
            str(result_json),
            "--output-csv",
            str(result_csv),
        ]
    )

    summary = _add_policy_summary(result_json, result_csv, successful_rows)
    summary["runs"] = {
        "total": len(rows),
        "successful": len(successful_rows),
        "failed": len(failed_rows),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
