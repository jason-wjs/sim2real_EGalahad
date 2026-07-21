#!/usr/bin/env python3
"""Convert mjlab_g1_native / isaaclab_g1 NPZ motions to any4hdmi qpos datasets.

Supports a single .npz file or a directory tree (rglob *.npz). Writes
motions/<relative_path>.npz plus manifest.json under --out-dir.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from any4hdmi.core.format import MOTION_DTYPE, MOTIONS_SUBDIR, save_motion, write_manifest
from any4hdmi.utils.mjcf import qpos_names_from_model
from sim2real.config.robots import G1_CFG

ORIGIN_INTERP_MARKER = "origin_interp10_NPZ"
REQUIRED_KEYS = (
    "fps",
    "joint_pos",
    "body_pos_w",
    "body_quat_w",
    "mjlab_g1_body_names",
)
EXPECTED_DOF = 29


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert mjlab_g1_native (isaaclab_g1) NPZ motions to any4hdmi qpos format. "
            "Input may be a single .npz or a directory."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to one .npz file or a directory tree of mjlab_g1_native motions.",
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
        help=(
            "Quaternion order of body_quat_w. Default: wxyz, or xyzw when the "
            "filename contains '_xyzw' and this flag is omitted."
        ),
    )
    parser.add_argument("--max-files", type=int, default=None, help="Convert only the first N files.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip sources whose output motion.npz already exists.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Log failures and continue when converting a directory.",
    )
    return parser.parse_args(argv)


def _load_reference_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if "qpos_names" not in manifest or "mjcf" not in manifest:
        raise ValueError(f"Reference manifest is missing qpos_names or mjcf: {path}")
    return manifest


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
    if value <= 0:
        raise ValueError(f"fps must be positive, got {value}")
    return value


def _resolve_quat_order(path: Path, explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    if "_xyzw" in path.stem.lower():
        return "xyzw"
    return "wxyz"


def _body_quat_to_wxyz(quat: np.ndarray, order: str) -> np.ndarray:
    arr = np.asarray(quat, dtype=MOTION_DTYPE)
    if arr.ndim != 2 or arr.shape[-1] != 4:
        raise ValueError(f"body quat must have shape (T, 4), got {arr.shape}")
    if order == "xyzw":
        arr = arr[:, [3, 0, 1, 2]]
    elif order != "wxyz":
        raise ValueError(f"Unsupported body quat order: {order}")
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.clip(norms, 1e-12, None)


def _load_mjlab_native(path: Path) -> dict[str, Any]:
    raw = np.load(path, allow_pickle=True)
    missing = [key for key in REQUIRED_KEYS if key not in raw.files]
    if missing:
        raise ValueError(f"{path}: missing required keys {missing}")

    motion_format = None
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
    fps = _scalar_fps(raw["fps"])

    if joint_pos.ndim != 2 or joint_pos.shape[-1] != EXPECTED_DOF:
        raise ValueError(f"{path}: joint_pos must have shape (T, {EXPECTED_DOF}), got {joint_pos.shape}")
    if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
        raise ValueError(f"{path}: body_pos_w must have shape (T, B, 3), got {body_pos_w.shape}")
    if body_quat_w.ndim != 3 or body_quat_w.shape[-1] != 4:
        raise ValueError(f"{path}: body_quat_w must have shape (T, B, 4), got {body_quat_w.shape}")
    if body_pos_w.shape[0] != joint_pos.shape[0] or body_quat_w.shape[0] != joint_pos.shape[0]:
        raise ValueError(
            f"{path}: frame count mismatch "
            f"joint_pos={joint_pos.shape}, body_pos_w={body_pos_w.shape}, body_quat_w={body_quat_w.shape}"
        )
    if body_pos_w.shape[1] != len(body_names) or body_quat_w.shape[1] != len(body_names):
        raise ValueError(
            f"{path}: body count mismatch names={len(body_names)} "
            f"pos={body_pos_w.shape[1]} quat={body_quat_w.shape[1]}"
        )

    return {
        "fps": fps,
        "joint_pos": joint_pos,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_names": body_names,
        "motion_format": motion_format,
        "source_format": (
            str(np.asarray(raw["source_format"]).reshape(()).item())
            if "source_format" in raw.files
            else None
        ),
    }


def _build_qpos(payload: dict[str, Any], qpos_names: list[str], quat_order: str) -> np.ndarray:
    if len(qpos_names) != 7 + EXPECTED_DOF:
        raise ValueError(f"Expected {7 + EXPECTED_DOF} qpos_names, got {len(qpos_names)}")

    pelvis_i = payload["body_names"].index("pelvis")
    joint_pos = payload["joint_pos"]
    root_pos = payload["body_pos_w"][:, pelvis_i, :]
    root_quat = _body_quat_to_wxyz(payload["body_quat_w"][:, pelvis_i, :], quat_order)

    qpos = np.zeros((joint_pos.shape[0], len(qpos_names)), dtype=MOTION_DTYPE)
    qpos[:, 0:3] = root_pos
    qpos[:, 3:7] = root_quat
    qpos[:, 7:] = joint_pos
    if qpos.shape[1] != len(qpos_names):
        raise ValueError(f"qpos dim mismatch: {qpos.shape[1]} vs {len(qpos_names)}")
    return qpos


def _scan_root(input_path: Path) -> Path:
    """Directory used as the relative-path root for output motion names."""
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
    # Lexicographic walk so --max-files can stop early on large trees.
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


def _rel_motion_out(src: Path, scan_root: Path, *, single_file: bool) -> Path:
    if single_file:
        return Path(f"{src.stem}.npz")
    try:
        rel = src.resolve().relative_to(scan_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Source {src} is not under scan root {scan_root}") from exc
    return rel


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    dataset_name = args.dataset_name or out_dir.name
    reference_manifest = (
        Path(args.reference_manifest).expanduser().resolve() if args.reference_manifest else None
    )

    mjcf, qpos_names = _resolve_qpos_meta(reference_manifest)
    sources = _iter_inputs(input_path, max_files=args.max_files)
    single_file = input_path.is_file()
    scan_root = _scan_root(input_path)

    total_frames = 0
    fps_values: list[float] = []
    successes = 0
    failures: list[tuple[Path, str]] = []
    skipped = 0

    for src in sources:
        rel = _rel_motion_out(src, scan_root, single_file=single_file)
        out_path = out_dir / MOTIONS_SUBDIR / rel
        if args.skip_existing and out_path.is_file():
            skipped += 1
            continue
        try:
            payload = _load_mjlab_native(src)
            quat_order = _resolve_quat_order(src, args.body_quat_order)
            qpos = _build_qpos(payload, qpos_names, quat_order)
            save_motion(out_path, qpos)
            fps_values.append(float(payload["fps"]))
            total_frames += int(qpos.shape[0])
            successes += 1
        except Exception as exc:  # noqa: BLE001 - batch convert should keep going when requested
            if single_file or not args.continue_on_error:
                raise
            failures.append((src, f"{type(exc).__name__}: {exc}"))
            print(f"[FAIL] {src}: {exc}", file=sys.stderr)

    if successes == 0:
        raise RuntimeError(
            f"No motions converted under {input_path} "
            f"(skipped={skipped}, failures={len(failures)})"
        )

    fps = float(np.median(np.asarray(fps_values, dtype=np.float64)))
    write_manifest(
        out_dir,
        dataset_name=dataset_name,
        mjcf=mjcf,
        timestep=1.0 / fps,
        qpos_names=qpos_names,
        num_motions=successes,
        total_hours=total_frames / fps / 3600.0,
        source={
            "input": str(input_path),
            "scan_root": str(scan_root),
            "reference_manifest": str(reference_manifest) if reference_manifest else None,
            "body_quat_order": args.body_quat_order or "auto",
            "fps_values_median": fps,
            "successes": successes,
            "skipped": skipped,
            "failures": [{"path": str(path), "error": err} for path, err in failures],
        },
    )
    print(
        f"converted successes={successes} skipped={skipped} failures={len(failures)} "
        f"frames={total_frames} -> {out_dir}"
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
