from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import convert_to_any4hdmi as converter
from any4hdmi.core.format import load_motion
from sim2real.config.robots import G1_CFG


ROOT_QPOS_NAMES = (
    "root_tx",
    "root_ty",
    "root_tz",
    "root_qw",
    "root_qx",
    "root_qy",
    "root_qz",
)
QPOS_NAMES = [*ROOT_QPOS_NAMES, *G1_CFG.joint_names]


class ConvertToAny4HdmiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.reference_manifest = self.root / "reference_manifest.json"
        self.reference_manifest.write_text(
            json.dumps(
                {
                    "mjcf": "hf://elijahgalahad/g1_xmls@main/g1-mode_13_15.xml",
                    "qpos_names": QPOS_NAMES,
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def _body_arrays(frames: int = 3) -> tuple[np.ndarray, np.ndarray]:
        body_pos = np.zeros((frames, converter.EXPECTED_BODY_COUNT, 3), dtype=np.float32)
        body_pos[:, 0, 0] = np.arange(frames, dtype=np.float32)
        body_pos[:, 0, 2] = 0.8
        body_quat = np.zeros((frames, converter.EXPECTED_BODY_COUNT, 4), dtype=np.float32)
        body_quat[..., 0] = 1.0
        return body_pos, body_quat

    def _write_corrected(self, path: Path, *, frames: int = 3) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        body_pos, body_quat = self._body_arrays(frames)
        joint_pos = np.arange(frames * 29, dtype=np.float32).reshape(frames, 29) / 100.0
        np.savez(
            path,
            fps=np.asarray([50]),
            joint_pos=joint_pos,
            joint_vel=np.zeros_like(joint_pos),
            body_pos_w=body_pos,
            body_quat_w=body_quat,
            body_lin_vel_w=np.zeros_like(body_pos),
            body_ang_vel_w=np.zeros_like(body_pos),
        )
        path.with_suffix(".diff.json").write_text(
            json.dumps(
                {
                    "frames": frames,
                    "fps": 50.0,
                    "output_contract": {
                        **converter.CORRECTED_CONTRACT,
                        "npz_fields": ["fps", "joint_pos", "body_pos_w", "body_quat_w"],
                    },
                }
            ),
            encoding="utf-8",
        )

    def _base_args(self, source: Path, output: Path, source_format: str) -> list[str]:
        return [
            "--input",
            str(source),
            "--source-format",
            source_format,
            "--out-dir",
            str(output),
            "--reference-manifest",
            str(self.reference_manifest),
        ]

    def test_corrected_isaaclab_conversion_and_resume(self) -> None:
        source = self.root / "input" / "origin_interp10_NPZ" / "clip" / "motion.npz"
        output = self.root / "output"
        self._write_corrected(source)

        self.assertEqual(converter.main(self._base_args(source, output, "isaaclab")), 0)
        motion = load_motion(output / "motions" / "motion.npz")
        self.assertEqual(motion.shape, (3, 36))
        np.testing.assert_allclose(motion[:, :3], [[0.0, 0.0, 0.8], [1.0, 0.0, 0.8], [2.0, 0.0, 0.8]])
        np.testing.assert_allclose(motion[:, 3:7], [[1.0, 0.0, 0.0, 0.0]] * 3)
        source_joint_pos = np.arange(3 * 29, dtype=np.float32).reshape(3, 29) / 100.0
        source_index = {
            name: index for index, name in enumerate(converter.ISAACLAB_G1_JOINT_NAMES)
        }
        expected_joint_pos = source_joint_pos[
            :, [source_index[name] for name in G1_CFG.joint_names]
        ]
        np.testing.assert_allclose(motion[:, 7:], expected_joint_pos)

        resume_args = [*self._base_args(source, output, "isaaclab"), "--skip-existing"]
        self.assertEqual(converter.main(resume_args), 0)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["num_motions"], 1)
        self.assertEqual(manifest["timestep"], 0.02)

    def test_mjlab_native_conversion(self) -> None:
        source = self.root / "native.npz"
        output = self.root / "native_output"
        body_pos, body_quat = self._body_arrays(2)
        np.savez(
            source,
            fps=np.asarray([50]),
            joint_pos=np.zeros((2, 29), dtype=np.float32),
            body_pos_w=body_pos,
            body_quat_w=body_quat,
            motion_format=np.asarray("mjlab_g1_native"),
            mjlab_g1_body_names=np.asarray(converter.ISAACLAB_G1_BODY_NAMES),
        )

        self.assertEqual(converter.main(self._base_args(source, output, "mjlab-g1-native")), 0)
        self.assertEqual(load_motion(output / "motions" / "native.npz").shape, (2, 36))

    def test_mujoco_qpos_conversion(self) -> None:
        source = self.root / "mujoco.npz"
        output = self.root / "mujoco_output"
        qpos = np.zeros((2, 36), dtype=np.float32)
        qpos[:, 2] = 0.8
        qpos[:, 3] = 2.0
        np.savez(source, qpos=qpos, qpos_names=np.asarray(QPOS_NAMES), timestep=np.asarray(0.02))

        self.assertEqual(converter.main(self._base_args(source, output, "mujoco")), 0)
        converted = load_motion(output / "motions" / "mujoco.npz")
        np.testing.assert_allclose(converted[:, 3], 1.0)

    def test_corrected_contract_mismatch_is_rejected(self) -> None:
        source = self.root / "bad" / "motion.npz"
        output = self.root / "bad_output"
        self._write_corrected(source)
        sidecar = json.loads(source.with_suffix(".diff.json").read_text(encoding="utf-8"))
        sidecar["output_contract"]["quaternion_order"] = "xyzw"
        source.with_suffix(".diff.json").write_text(json.dumps(sidecar), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "quaternion_order"):
            converter.main(self._base_args(source, output, "isaaclab-g1-corrected"))


if __name__ == "__main__":
    unittest.main()
