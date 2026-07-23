"""Compatibility CLI for the public Unified Tracking Metrics v2 API."""

from sim2real.metrics.tracking import (
    METRIC_ROW_FIELDS,
    METRIC_SCHEMA_VERSION,
    REQUIRED_TRAJECTORY_KEYS,
    _compute_one,
    _expand_paths,
    _summary,
    compute_trajectory_metrics,
    main,
    metric_schema,
    summarize_rows,
)

__all__ = [
    "METRIC_ROW_FIELDS",
    "METRIC_SCHEMA_VERSION",
    "REQUIRED_TRAJECTORY_KEYS",
    "compute_trajectory_metrics",
    "metric_schema",
    "summarize_rows",
]


if __name__ == "__main__":
    main()
