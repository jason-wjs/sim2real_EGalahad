#!/usr/bin/env python3
"""Convert supported G1 motion NPZ schemas into an any4hdmi qpos dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from any4hdmi.core.format import MOTION_DTYPE, MOTIONS_SUBDIR, load_motion, save_motion, write_manifest
from any4hdmi.utils.mjcf import qpos_names_from_model
from sim2real.config.robots import G1_CFG


ORIGIN_INTERP_MARKER = "origin_interp10_NPZ"
SOURCE_FORMATS = (
    "auto",
    "isaaclab-g1-corrected",
    "mjlab-g1-native",
    "mujoco-qpos",
)
SOURCE_FORMAT_ALIASES = {
    "isaaclab": "isaaclab-g1-corrected",
    "mujoco": "mujoco-qpos",
}
EXPECTED_DOF = 29
EXPECTED_BODY_COUNT = 30
ISAACLAB_G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)
ISAACLAB_G1_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
CORRECTED_REQUIRED_KEYS = (
    "fps",
    "joint_pos",
    "body_pos_w",
    "body_quat_w",
)
NATIVE_REQUIRED_KEYS = (*CORRECTED_REQUIRED_KEYS, "mjlab_g1_body_names")
CORRECTED_CONTRACT = {
    "joint_order": "IsaacLab Data10k G1 order",
    "body_order": "IsaacLab Entity.body_names order",
    "quaternion_order": "wxyz",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert supported IsaacLab/mjlab/MuJoCo G1 NPZ motions to an "
            "any4hdmi qpos dataset. Input may be one .npz or a directory tree."
        )
    )
    parser.add_argument("--input", required=True, help="Source .npz or directory tree.")
    parser.add_argument(
        "--source-format",
        required=True,
        choices=(*SOURCE_FORMATS, *SOURCE_FORMAT_ALIASES),
        help="Concrete source NPZ schema. 'isaaclab' and 'mujoco' are aliases.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output any4hdmi dataset root (motions/ + manifest.json).",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset name written into manifest.json (default: out-dir name).",
    )
    parser.add_argument(
        "--reference-manifest",
        default=None,
        help="Optional any4hdmi manifest.json providing mjcf and qpos_names.",
    )
    parser.add_argument(
        "--body-quat-order",
        choices=["wxyz", "xyzw"],
        default=None,
        help="Override body_quat_w order for IsaacLab/mjlab inputs.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Convert only first N files.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failures and continue when converting a directory.",
    )
    return parser.parse_args(argv)


def _load_reference_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "qpos_names" not in payload or "mjcf" not in payload:
        raise ValueError(f"Reference manifest is missing qpos_names or mjcf: {path}")
    return payload


def _resolve_qpos_meta(reference_manifest: Path | None) -> tuple[Any, list[str]]:
    if reference_manifest is not None:
        payload = _load_reference_manifest(reference_manifest)
        return payload["mjcf"], [str(name) for name in payload["qpos_names"]]

    mjcf_path = G1_CFG.resolve_mjcf_path()
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    qpos_names = [str(name) for name in qpos_names_from_model(model)]
    mjcf = G1_CFG.mjcf_path if G1_CFG.mjcf_path is not None else mjcf_path
    return mjcf, qpos_names


def _scalar_fps(raw_fps: object) -> float:
    value = float(np.asarray(raw_fps).reshape(()))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"fps must be finite and positive, got {value}")
    return value


def _source_fps(raw: Any) -> float:
    if "fps" in raw.files:
        return _scalar_fps(raw["fps"])
    if "timestep" in raw.files:
        timestep = float(np.asarray(raw["timestep"]).reshape(()))
        if not np.isfinite(timestep) or timestep <= 0:
            raise ValueError(f"timestep must be finite and positive, got {timestep}")
        return 1.0 / timestep
    raise ValueError("source must contain fps or timestep")


def _resolve_quat_order(path: Path, explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    if "_xyzw" in path.stem.lower():
        return "xyzw"
    return "wxyz"


def _quat_to_wxyz(quat: np.ndarray, order: str, *, label: str) -> np.ndarray:
    arr = np.asarray(quat, dtype=MOTION_DTYPE)
    if arr.ndim != 2 or arr.shape[-1] != 4:
        raise ValueError(f"{label} must have shape (T, 4), got {arr.shape}")
    if order == "xyzw":
        arr = arr[:, [3, 0, 1, 2]]
    elif order != "wxyz":
        raise ValueError(f"Unsupported quaternion order: {order}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{label} contains NaN or Inf")
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError(f"{label} contains a zero quaternion")
    return arr / norms


def _validate_finite(name: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")


def _load_sidecar_contract(path: Path) -> dict[str, Any]:
    sidecar_path = path.with_suffix(".diff.json")
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"Corrected IsaacLab source requires sidecar contract: {sidecar_path}"
        )
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    contract = payload.get("output_contract")
    if not isinstance(contract, dict):
        raise ValueError(f"Sidecar is missing output_contract: {sidecar_path}")
    for key, expected in CORRECTED_CONTRACT.items():
        actual = contract.get(key)
        if actual != expected:
            raise ValueError(
                f"{sidecar_path}: expected output_contract.{key}={expected!r}, got {actual!r}"
            )
    return payload


def _validate_joint_and_body_arrays(
    *,
    path: Path,
    joint_pos: np.ndarray,
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    body_count: int,
) -> None:
    if joint_pos.ndim != 2 or joint_pos.shape[-1] != EXPECTED_DOF:
        raise ValueError(f"{path}: joint_pos must have shape (T, {EXPECTED_DOF}), got {joint_pos.shape}")
    if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
        raise ValueError(f"{path}: body_pos_w must have shape (T, B, 3), got {body_pos_w.shape}")
    if body_quat_w.ndim != 3 or body_quat_w.shape[-1] != 4:
        raise ValueError(f"{path}: body_quat_w must have shape (T, B, 4), got {body_quat_w.shape}")
    if body_pos_w.shape[0] != joint_pos.shape[0] or body_quat_w.shape[0] != joint_pos.shape[0]:
        raise ValueError(
            f"{path}: frame count mismatch joint_pos={joint_pos.shape}, "
            f"body_pos_w={body_pos_w.shape}, body_quat_w={body_quat_w.shape}"
        )
    if body_pos_w.shape[1] != body_count or body_quat_w.shape[1] != body_count:
        raise ValueError(
            f"{path}: expected {body_count} bodies, got "
            f"pos={body_pos_w.shape[1]} quat={body_quat_w.shape[1]}"
        )
    _validate_finite("joint_pos", joint_pos)
    _validate_finite("body_pos_w", body_pos_w)


def _joint_pos_in_qpos_order(
    joint_pos: np.ndarray,
    *,
    source_joint_names: tuple[str, ...],
    qpos_names: list[str],
) -> np.ndarray:
    target_joint_names = qpos_names[7:]
    if len(source_joint_names) != joint_pos.shape[1]:
        raise ValueError(
            f"Expected {joint_pos.shape[1]} source joint names, got {len(source_joint_names)}"
        )
    if len(set(source_joint_names)) != len(source_joint_names):
        raise ValueError("Source joint names contain duplicates")
    if len(set(target_joint_names)) != len(target_joint_names):
        raise ValueError("Target qpos joint names contain duplicates")
    if set(source_joint_names) != set(target_joint_names):
        missing = sorted(set(target_joint_names) - set(source_joint_names))
        extra = sorted(set(source_joint_names) - set(target_joint_names))
        raise ValueError(
            f"Source and target joint names differ: missing={missing}, extra={extra}"
        )
    source_index = {name: index for index, name in enumerate(source_joint_names)}
    return joint_pos[:, [source_index[name] for name in target_joint_names]]


def _build_qpos(
    *,
    path: Path,
    joint_pos: np.ndarray,
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    pelvis_index: int,
    qpos_names: list[str],
    quat_order: str,
) -> np.ndarray:
    if len(qpos_names) != 7 + EXPECTED_DOF:
        raise ValueError(f"Expected {7 + EXPECTED_DOF} qpos_names, got {len(qpos_names)}")
    _validate_joint_and_body_arrays(
        path=path,
        joint_pos=joint_pos,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_count=body_pos_w.shape[1],
    )
    root_quat = _quat_to_wxyz(
        body_quat_w[:, pelvis_index, :],
        quat_order,
        label="pelvis quaternion",
    )
    qpos = np.empty((joint_pos.shape[0], len(qpos_names)), dtype=MOTION_DTYPE)
    qpos[:, 0:3] = body_pos_w[:, pelvis_index, :]
    qpos[:, 3:7] = root_quat
    qpos[:, 7:] = joint_pos
    return qpos


def _load_isaaclab_corrected(
    path: Path,
    *,
    qpos_names: list[str],
    body_quat_order: str | None,
) -> tuple[np.ndarray, float]:
    raw = np.load(path, allow_pickle=False)
    missing = [key for key in CORRECTED_REQUIRED_KEYS if key not in raw.files]
    if missing:
        raise ValueError(f"{path}: missing required keys {missing}")
    sidecar = _load_sidecar_contract(path)
    joint_pos = np.asarray(raw["joint_pos"], dtype=MOTION_DTYPE)
    body_pos_w = np.asarray(raw["body_pos_w"], dtype=MOTION_DTYPE)
    body_quat_w = np.asarray(raw["body_quat_w"], dtype=MOTION_DTYPE)
    _validate_joint_and_body_arrays(
        path=path,
        joint_pos=joint_pos,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_count=EXPECTED_BODY_COUNT,
    )
    frames = sidecar.get("frames")
    if frames is not None and int(frames) != int(joint_pos.shape[0]):
        raise ValueError(f"{path}: sidecar frames={frames} != NPZ frames={joint_pos.shape[0]}")
    fps = _source_fps(raw)
    sidecar_fps = sidecar.get("fps")
    if sidecar_fps is not None and not np.isclose(float(sidecar_fps), fps):
        raise ValueError(f"{path}: sidecar fps={sidecar_fps} != NPZ fps={fps}")
    joint_pos = _joint_pos_in_qpos_order(
        joint_pos,
        source_joint_names=ISAACLAB_G1_JOINT_NAMES,
        qpos_names=qpos_names,
    )
    qpos = _build_qpos(
        path=path,
        joint_pos=joint_pos,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        pelvis_index=ISAACLAB_G1_BODY_NAMES.index("pelvis"),
        qpos_names=qpos_names,
        quat_order=_resolve_quat_order(path, body_quat_order),
    )
    return qpos, fps


def _load_mjlab_native(
    path: Path,
    *,
    qpos_names: list[str],
    body_quat_order: str | None,
) -> tuple[np.ndarray, float]:
    raw = np.load(path, allow_pickle=True)
    missing = [key for key in NATIVE_REQUIRED_KEYS if key not in raw.files]
    if missing:
        raise ValueError(f"{path}: missing required keys {missing}")
    if "motion_format" in raw.files:
        motion_format = str(np.asarray(raw["motion_format"]).reshape(()).item())
        if motion_format != "mjlab_g1_native":
            raise ValueError(f"{path}: expected motion_format=mjlab_g1_native, got {motion_format!r}")
    body_names = [str(name) for name in np.asarray(raw["mjlab_g1_body_names"]).tolist()]
    if "pelvis" not in body_names:
        raise ValueError(f"{path}: mjlab_g1_body_names missing pelvis")
    joint_pos = np.asarray(raw["joint_pos"], dtype=MOTION_DTYPE)
    body_pos_w = np.asarray(raw["body_pos_w"], dtype=MOTION_DTYPE)
    body_quat_w = np.asarray(raw["body_quat_w"], dtype=MOTION_DTYPE)
    _validate_joint_and_body_arrays(
        path=path,
        joint_pos=joint_pos,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_count=len(body_names),
    )
    qpos = _build_qpos(
        path=path,
        joint_pos=joint_pos,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        pelvis_index=body_names.index("pelvis"),
        qpos_names=qpos_names,
        quat_order=_resolve_quat_order(path, body_quat_order),
    )
    return qpos, _source_fps(raw)


def _load_mujoco_qpos(path: Path, *, qpos_names: list[str]) -> tuple[np.ndarray, float]:
    raw = np.load(path, allow_pickle=False)
    if "qpos" not in raw.files:
        raise ValueError(f"{path}: mujoco-qpos source must contain qpos")
    if "qpos_names" not in raw.files:
        raise ValueError(f"{path}: mujoco-qpos source must contain qpos_names")
    source_names = [str(name) for name in np.asarray(raw["qpos_names"]).tolist()]
    if source_names != qpos_names:
        raise ValueError(f"{path}: qpos_names do not match target manifest order")
    qpos = np.asarray(raw["qpos"], dtype=MOTION_DTYPE)
    if qpos.ndim == 1:
        qpos = qpos[None, :]
    if qpos.ndim != 2 or qpos.shape[1] != len(qpos_names):
        raise ValueError(f"{path}: qpos must have shape (T, {len(qpos_names)}), got {qpos.shape}")
    _validate_finite("qpos", qpos)
    qpos = qpos.copy()
    qpos[:, 3:7] = _quat_to_wxyz(qpos[:, 3:7], "wxyz", label="root quaternion")
    return qpos, _source_fps(raw)


def _detect_source_format(path: Path) -> str:
    raw = np.load(path, allow_pickle=True)
    files = set(raw.files)
    if {"qpos", "qpos_names"}.issubset(files):
        return "mujoco-qpos"
    if "mjlab_g1_body_names" in files:
        return "mjlab-g1-native"
    if set(CORRECTED_REQUIRED_KEYS).issubset(files) and path.with_suffix(".diff.json").is_file():
        return "isaaclab-g1-corrected"
    raise ValueError(f"Could not auto-detect a supported source format for {path}")


def _load_source(
    path: Path,
    *,
    source_format: str,
    qpos_names: list[str],
    body_quat_order: str | None,
) -> tuple[np.ndarray, float, str]:
    resolved_format = _detect_source_format(path) if source_format == "auto" else source_format
    if resolved_format == "isaaclab-g1-corrected":
        qpos, fps = _load_isaaclab_corrected(
            path,
            qpos_names=qpos_names,
            body_quat_order=body_quat_order,
        )
    elif resolved_format == "mjlab-g1-native":
        qpos, fps = _load_mjlab_native(
            path,
            qpos_names=qpos_names,
            body_quat_order=body_quat_order,
        )
    elif resolved_format == "mujoco-qpos":
        qpos, fps = _load_mujoco_qpos(path, qpos_names=qpos_names)
    else:  # pragma: no cover - argparse and aliases constrain this value.
        raise ValueError(f"Unsupported source format: {resolved_format}")
    return qpos, fps, resolved_format


def _scan_root(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.parent
    if input_path.name == ORIGIN_INTERP_MARKER:
        return input_path
    nested = input_path / ORIGIN_INTERP_MARKER
    if nested.is_dir():
        return nested
    for candidate in input_path.parents:
        if candidate.name == ORIGIN_INTERP_MARKER:
            return candidate
    return input_path


def _iter_inputs(input_path: Path, *, max_files: int | None = None) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".npz":
            raise ValueError(f"Input file must be .npz, got {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    motions: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(input_path):
        dirnames.sort()
        for name in sorted(filenames):
            if not name.endswith(".npz"):
                continue
            path = Path(dirpath) / name
            if not path.is_file():
                continue
            motions.append(path)
            if max_files is not None and len(motions) >= int(max_files):
                return motions
    if not motions:
        raise RuntimeError(f"No .npz files found under {input_path}")
    return motions


def _relative_output(src: Path, scan_root: Path, *, single_file: bool) -> Path:
    if single_file:
        return Path(f"{src.stem}.npz")
    try:
        return src.resolve().relative_to(scan_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Source {src} is not under scan root {scan_root}") from exc


def _read_jsonl_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL record at {path}:{line_number}") from exc
        records[str(record["output"])] = record
    return records


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _record_for_existing(
    *,
    src: Path,
    out_path: Path,
    out_dir: Path,
    source_format: str,
) -> dict[str, Any]:
    qpos = load_motion(out_path)
    raw = np.load(src, allow_pickle=False)
    fps = _source_fps(raw)
    resolved_format = _detect_source_format(src) if source_format == "auto" else source_format
    return {
        "source": str(src),
        "output": str(out_path.relative_to(out_dir)),
        "source_format": resolved_format,
        "frames": int(qpos.shape[0]),
        "fps": float(fps),
    }


def _write_reports(
    *,
    out_dir: Path,
    input_path: Path,
    scan_root: Path,
    requested_format: str,
    records: dict[str, dict[str, Any]],
    failures: list[dict[str, str]],
    skipped: int,
) -> None:
    report = {
        "input": str(input_path),
        "scan_root": str(scan_root),
        "requested_source_format": requested_format,
        "num_motions": len(records),
        "skipped_this_run": skipped,
        "failures_this_run": failures,
        "source_format_counts": {},
    }
    format_counts: dict[str, int] = {}
    for record in records.values():
        name = str(record["source_format"])
        format_counts[name] = format_counts.get(name, 0) + 1
    report["source_format_counts"] = format_counts
    (out_dir / "conversion_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "failed_motions.json").write_text(
        json.dumps(failures, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = args.dataset_name or out_dir.name
    source_format = SOURCE_FORMAT_ALIASES.get(args.source_format, args.source_format)
    reference_manifest = (
        Path(args.reference_manifest).expanduser().resolve() if args.reference_manifest else None
    )
    mjcf, qpos_names = _resolve_qpos_meta(reference_manifest)
    sources = _iter_inputs(input_path, max_files=args.max_files)
    scan_root = _scan_root(input_path)
    single_file = input_path.is_file()
    records_path = out_dir / "conversion_records.jsonl"
    records = _read_jsonl_records(records_path)
    failures: list[dict[str, str]] = []
    skipped = 0

    for index, src in enumerate(sources, start=1):
        rel = _relative_output(src, scan_root, single_file=single_file)
        out_path = out_dir / MOTIONS_SUBDIR / rel
        record_key = str(out_path.relative_to(out_dir))
        try:
            if args.skip_existing and out_path.is_file():
                skipped += 1
                if record_key not in records:
                    record = _record_for_existing(
                        src=src,
                        out_path=out_path,
                        out_dir=out_dir,
                        source_format=source_format,
                    )
                    _append_jsonl(records_path, record)
                    records[record_key] = record
                print(f"[{index}/{len(sources)}] skip {src}", flush=True)
                continue
            qpos, fps, resolved_format = _load_source(
                src,
                source_format=source_format,
                qpos_names=qpos_names,
                body_quat_order=args.body_quat_order,
            )
            save_motion(out_path, qpos)
            record = {
                "source": str(src),
                "output": record_key,
                "source_format": resolved_format,
                "frames": int(qpos.shape[0]),
                "fps": float(fps),
            }
            _append_jsonl(records_path, record)
            records[record_key] = record
            print(f"[{index}/{len(sources)}] converted {src} -> {out_path}", flush=True)
        except Exception as exc:  # noqa: BLE001 - directory conversion may continue by request.
            failure = {"path": str(src), "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            print(f"[FAIL] {src}: {exc}", file=sys.stderr, flush=True)
            if single_file or not args.continue_on_error:
                _write_reports(
                    out_dir=out_dir,
                    input_path=input_path,
                    scan_root=scan_root,
                    requested_format=source_format,
                    records=records,
                    failures=failures,
                    skipped=skipped,
                )
                raise

    output_paths = sorted((out_dir / MOTIONS_SUBDIR).rglob("*.npz"))
    output_keys = {str(path.relative_to(out_dir)) for path in output_paths}
    records = {key: value for key, value in records.items() if key in output_keys}
    if not records:
        _write_reports(
            out_dir=out_dir,
            input_path=input_path,
            scan_root=scan_root,
            requested_format=source_format,
            records=records,
            failures=failures,
            skipped=skipped,
        )
        raise RuntimeError(f"No motions converted under {input_path}")
    if output_keys != set(records):
        missing_records = sorted(output_keys - set(records))
        raise RuntimeError(f"Output motions are missing conversion records: {missing_records[:5]}")

    fps_values = np.asarray([float(record["fps"]) for record in records.values()])
    fps = float(np.median(fps_values))
    if not np.allclose(fps_values, fps, rtol=0.0, atol=1e-6):
        unique_fps = sorted({float(value) for value in fps_values})
        raise ValueError(f"any4hdmi dataset requires one timestep; found fps values {unique_fps}")
    total_hours = sum(
        int(record["frames"]) / float(record["fps"]) / 3600.0
        for record in records.values()
    )
    write_manifest(
        out_dir,
        dataset_name=dataset_name,
        mjcf=mjcf,
        timestep=1.0 / fps,
        qpos_names=qpos_names,
        num_motions=len(records),
        total_hours=total_hours,
        source={
            "input": str(input_path),
            "scan_root": str(scan_root),
            "source_format": source_format,
            "reference_manifest": str(reference_manifest) if reference_manifest else None,
            "body_quat_order": args.body_quat_order or "auto",
            "failures": len(failures),
        },
    )
    _write_reports(
        out_dir=out_dir,
        input_path=input_path,
        scan_root=scan_root,
        requested_format=source_format,
        records=records,
        failures=failures,
        skipped=skipped,
    )
    print(
        f"converted={len(records)} skipped_this_run={skipped} failures={len(failures)} "
        f"fps={fps:g} total_hours={total_hours:.3f} -> {out_dir}",
        flush=True,
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
