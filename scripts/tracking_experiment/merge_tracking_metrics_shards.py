from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from run_tracking_metrics_eval import _summary_only_outputs, _write_csv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge and validate summary-only tracking-evaluation shards."
    )
    parser.add_argument(
        "--shards-root",
        required=True,
        help="Directory containing shard output directories.",
    )
    parser.add_argument("--output-dir", required=True, help="Merged output directory.")
    parser.add_argument(
        "--expected-rollouts",
        type=int,
        required=True,
        help="Exact number of rollout metric rows required.",
    )
    parser.add_argument(
        "--expected-controller",
        action="append",
        default=[],
        help="Controller name required in the merged output; repeat for each controller.",
    )
    return parser.parse_args()


def _semantic_key(row: dict[str, object]) -> tuple[str, str, str, int]:
    return (
        str(row["policy"]),
        str(Path(str(row["policy_config"])).expanduser().resolve()),
        str(Path(str(row["motion_path"])).expanduser().resolve()),
        int(row["seed"]),
    )


def _motion_index_key(row: dict[str, object]) -> tuple[str, str, int, int]:
    return (
        str(row["policy"]),
        str(Path(str(row["policy_config"])).expanduser().resolve()),
        int(row["motion_index"]),
        int(row["seed"]),
    )


def _read_metric_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Metric checkpoint is not an object at {path}:{line_number}")
        rows.append(row)
    return rows


def _read_run_rows(path: Path) -> tuple[list[str], list[dict[str, object]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def _require_unique(
    rows: list[dict[str, object]],
    key_fn,
    *,
    label: str,
) -> dict[tuple[object, ...], dict[str, object]]:
    unique: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        key = key_fn(row)
        if key in unique:
            raise RuntimeError(f"Duplicate {label}: {key}")
        unique[key] = row
    return unique


def merge_shards(
    shards_root: Path,
    output_dir: Path,
    *,
    expected_rollouts: int,
    expected_controllers: set[str] | None = None,
) -> dict[str, object]:
    if expected_rollouts <= 0:
        raise ValueError("expected_rollouts must be positive")

    checkpoint_paths = sorted(shards_root.rglob("checkpoints/rollout_metrics.jsonl"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No rollout metric checkpoints under {shards_root}")

    metric_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    run_fieldnames: list[str] | None = None
    for checkpoint_path in checkpoint_paths:
        shard_dir = checkpoint_path.parent.parent
        runs_path = shard_dir / "runs.csv"
        if not runs_path.is_file():
            raise FileNotFoundError(f"Missing runs.csv beside {checkpoint_path}")
        metric_rows.extend(_read_metric_rows(checkpoint_path))
        fieldnames, shard_run_rows = _read_run_rows(runs_path)
        if run_fieldnames is None:
            run_fieldnames = fieldnames
        elif fieldnames != run_fieldnames:
            raise RuntimeError(
                f"Inconsistent runs.csv schema at {runs_path}: "
                f"expected {run_fieldnames}, got {fieldnames}"
            )
        run_rows.extend(shard_run_rows)

    metrics_by_key = _require_unique(metric_rows, _semantic_key, label="metric rollout")
    runs_by_key = _require_unique(run_rows, _semantic_key, label="run")
    _require_unique(metric_rows, _motion_index_key, label="metric motion index")
    _require_unique(run_rows, _motion_index_key, label="run motion index")

    failed_rows = [row for row in run_rows if str(row["status"]) == "failed"]
    if failed_rows:
        raise RuntimeError(
            f"Cannot produce a complete merge: {len(failed_rows)} shard rollouts failed"
        )
    if metrics_by_key.keys() != runs_by_key.keys():
        missing_metrics = runs_by_key.keys() - metrics_by_key.keys()
        orphan_metrics = metrics_by_key.keys() - runs_by_key.keys()
        raise RuntimeError(
            "Metric/run mismatch: "
            f"missing_metrics={len(missing_metrics)} orphan_metrics={len(orphan_metrics)}"
        )
    if len(metric_rows) != expected_rollouts:
        raise RuntimeError(
            f"Expected {expected_rollouts} rollouts, found {len(metric_rows)}"
        )

    controllers = {str(row["policy"]) for row in metric_rows}
    required_controllers = expected_controllers or set()
    if controllers != required_controllers:
        raise RuntimeError(
            f"Expected controllers {sorted(required_controllers)}, found {sorted(controllers)}"
        )

    sort_key = lambda row: (
        str(row["policy"]),
        int(row["motion_index"]),
        int(row["seed"]),
        str(row["motion_path"]),
    )
    metric_rows.sort(key=sort_key)
    run_rows.sort(key=sort_key)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoints" / "rollout_metrics.jsonl"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_temp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    with checkpoint_temp.open("w", encoding="utf-8") as f:
        for row in metric_rows:
            f.write(json.dumps(row, allow_nan=False) + "\n")
    checkpoint_temp.replace(checkpoint_path)

    assert run_fieldnames is not None
    _write_csv(output_dir / "runs.csv", run_rows, run_fieldnames)
    _write_csv(output_dir / "failed_runs.csv", [], run_fieldnames)
    return _summary_only_outputs(output_dir, metric_rows, run_rows)


def main() -> None:
    args = _parse_args()
    summary = merge_shards(
        Path(args.shards_root).expanduser().resolve(),
        Path(args.output_dir).expanduser().resolve(),
        expected_rollouts=args.expected_rollouts,
        expected_controllers=set(args.expected_controller),
    )
    print(
        json.dumps(
            {
                "rollouts": summary["runs"]["total"],
                "controllers": sorted(summary["per_controller"]),
                "output_dir": str(Path(args.output_dir).expanduser().resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
