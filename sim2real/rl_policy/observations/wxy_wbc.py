from __future__ import annotations

from typing import Any, Dict, Sequence

import mujoco
import numpy as np

from sim2real.rl_policy.observations.base import Observation
from sim2real.rl_policy.observations.motion import motion_obs
from sim2real.utils.math import (
    matrix_from_quat,
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse_numpy,
)


WXY_WBC_ACTION_DIM = 29
WXY_WBC_HISTORY_LENGTH = 5
WXY_WBC_MOTION_COMMAND_DIM = 241
WXY_WBC_PROPRIOCEPTION_DIM = 645
WXY_WBC_OBS_DIM = WXY_WBC_MOTION_COMMAND_DIM + WXY_WBC_PROPRIOCEPTION_DIM

WXY_WBC_COMMAND_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
WXY_WBC_LIMB_BODY_NAMES = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)


def _safe_unit_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (quat / norm).astype(np.float32)


class wxy_wbc_policy_obs(motion_obs, namespace="wxy_wbc"):
    """Exact 886D observation used by the WXY G1 WBC actor.

    The actor was trained with one current reference frame and five-frame
    oldest-to-newest histories. Robot limb poses are reconstructed from the
    measured joint state with the normal sim2real G1 MuJoCo model, so the same
    implementation works in integrated sim2sim and on the robot.
    """

    def __init__(
        self,
        joint_names: Sequence[str] | None = None,
        command_body_names: Sequence[str] = WXY_WBC_COMMAND_BODY_NAMES,
        limb_body_names: Sequence[str] = WXY_WBC_LIMB_BODY_NAMES,
        anchor_body_name: str = "pelvis",
        tracking_anchor_body_name: str = "torso_link",
        history_length: int = WXY_WBC_HISTORY_LENGTH,
        **kwargs: Any,
    ) -> None:
        resolved_joint_names = (
            list(joint_names)
            if joint_names is not None
            else list(kwargs["env"].policy_joint_names)
        )
        self.command_body_names = [str(name) for name in command_body_names]
        self.limb_body_names = [str(name) for name in limb_body_names]
        self.tracking_anchor_body_name = str(tracking_anchor_body_name)
        self.history_length = int(history_length)
        if self.history_length != WXY_WBC_HISTORY_LENGTH:
            raise ValueError(
                "WXY WBC actor requires history_length="
                f"{WXY_WBC_HISTORY_LENGTH}, got {self.history_length}"
            )
        if len(resolved_joint_names) != WXY_WBC_ACTION_DIM:
            raise ValueError(
                f"WXY WBC actor requires {WXY_WBC_ACTION_DIM} joints, "
                f"got {len(resolved_joint_names)}"
            )
        if anchor_body_name not in self.command_body_names:
            raise ValueError(
                f"WXY WBC anchor body {anchor_body_name!r} is absent from command_body_names"
            )
        if tracking_anchor_body_name not in self.command_body_names:
            raise ValueError(
                "WXY WBC tracking anchor body "
                f"{tracking_anchor_body_name!r} is absent from command_body_names"
            )
        missing_limbs = [
            name for name in self.limb_body_names if name not in self.command_body_names
        ]
        if missing_limbs:
            raise ValueError(
                f"WXY WBC limb bodies are absent from command_body_names: {missing_limbs}"
            )

        super().__init__(
            future_steps=[0],
            joint_names=resolved_joint_names,
            body_names=self.command_body_names,
            root_body_name=anchor_body_name,
            anchor_body_name=anchor_body_name,
            joint_order="given",
            body_order="given",
            **kwargs,
        )

        self._state_joint_indices = [
            self.state_processor.joint_names.index(name) for name in self.joint_names
        ]
        self._default_joint_pos = np.asarray(
            [
                self.env.default_dof_angles[
                    self.env.joint_names_simulation.index(joint_name)
                ]
                for joint_name in self.joint_names
            ],
            dtype=np.float32,
        )
        self._limb_indices = np.asarray(
            [self.command_body_names.index(name) for name in self.limb_body_names],
            dtype=np.int64,
        )
        self._anchor_index = self.command_body_names.index(self.anchor_body_name)
        self._tracking_anchor_index = self.command_body_names.index(
            self.tracking_anchor_body_name
        )

        self._fk_model = mujoco.MjModel.from_xml_path(
            str(self.env.robot_cfg.resolve_mjcf_path())
        )
        self._fk_data = mujoco.MjData(self._fk_model)
        self._fk_joint_qpos_adrs = np.asarray(
            [self._joint_qpos_address(name) for name in self.joint_names],
            dtype=np.int64,
        )
        self._fk_body_ids = np.asarray(
            [self._body_id(name) for name in self.command_body_names],
            dtype=np.int64,
        )
        self._fk_root_qpos_adr = self._root_qpos_address()

        limb_pose_dim = len(self.limb_body_names) * 9
        self._histories = {
            "ref_limb_ee_pose_b": np.zeros(
                (self.history_length, limb_pose_dim), dtype=np.float32
            ),
            "robot_limb_ee_pose_b": np.zeros(
                (self.history_length, limb_pose_dim), dtype=np.float32
            ),
            "projected_gravity": np.zeros(
                (self.history_length, 3), dtype=np.float32
            ),
            "base_ang_vel": np.zeros(
                (self.history_length, 3), dtype=np.float32
            ),
            "joint_pos": np.zeros(
                (self.history_length, WXY_WBC_ACTION_DIM), dtype=np.float32
            ),
            "joint_vel": np.zeros(
                (self.history_length, WXY_WBC_ACTION_DIM), dtype=np.float32
            ),
            "actions": np.zeros(
                (self.history_length, WXY_WBC_ACTION_DIM), dtype=np.float32
            ),
        }
        self._history_initialized = False
        self._parts = {
            "motion_command": np.zeros(
                WXY_WBC_MOTION_COMMAND_DIM, dtype=np.float32
            ),
            "proprioception": np.zeros(
                WXY_WBC_PROPRIOCEPTION_DIM, dtype=np.float32
            ),
        }
        self._obs = np.zeros((1, WXY_WBC_OBS_DIM), dtype=np.float32)

    def _joint_qpos_address(self, joint_name: str) -> int:
        joint_id = mujoco.mj_name2id(
            self._fk_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if joint_id < 0:
            raise ValueError(f"WXY WBC FK model is missing joint {joint_name!r}")
        return int(self._fk_model.jnt_qposadr[joint_id])

    def _body_id(self, body_name: str) -> int:
        body_id = mujoco.mj_name2id(
            self._fk_model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        if body_id < 0:
            raise ValueError(f"WXY WBC FK model is missing body {body_name!r}")
        return int(body_id)

    def _root_qpos_address(self) -> int:
        for joint_name in self.env.robot_cfg.root_joint_names:
            joint_id = mujoco.mj_name2id(
                self._fk_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id >= 0:
                return int(self._fk_model.jnt_qposadr[joint_id])
        raise ValueError(
            "WXY WBC FK model is missing all configured root joints: "
            f"{self.env.robot_cfg.root_joint_names}"
        )

    @staticmethod
    def _append_history(
        history: np.ndarray,
        value: np.ndarray,
        *,
        initialize: bool,
    ) -> None:
        value = np.asarray(value, dtype=np.float32).reshape(history.shape[1:])
        if initialize:
            history[:] = value
            return
        history[:-1] = history[1:]
        history[-1] = value

    def _limb_pose_in_anchor_frame(
        self,
        body_pos_w: np.ndarray,
        body_quat_w: np.ndarray,
    ) -> np.ndarray:
        body_pos_w = np.asarray(body_pos_w, dtype=np.float32)
        body_quat_w = np.asarray(body_quat_w, dtype=np.float32)
        limb_pos_w = body_pos_w[self._limb_indices]
        limb_quat_w = body_quat_w[self._limb_indices]
        anchor_pos_w = body_pos_w[self._anchor_index]
        anchor_quat_w = _safe_unit_quat(body_quat_w[self._anchor_index])
        anchor_quat_batch = np.broadcast_to(
            anchor_quat_w, (len(self._limb_indices), 4)
        )
        limb_pos_b = quat_rotate_inverse_numpy(
            anchor_quat_batch,
            limb_pos_w - anchor_pos_w,
        )
        limb_quat_b = quat_mul(
            quat_conjugate(anchor_quat_batch),
            limb_quat_w,
        )
        limb_rot6d = matrix_from_quat(limb_quat_b)[..., :, :2].reshape(
            len(self._limb_indices), 6
        )
        return np.concatenate([limb_pos_b, limb_rot6d], axis=-1).reshape(-1)

    def _robot_limb_pose(self, joint_pos: np.ndarray) -> np.ndarray:
        self._fk_data.qpos[:] = 0.0
        root_qpos_adr = self._fk_root_qpos_adr
        self._fk_data.qpos[root_qpos_adr + 3] = 1.0
        self._fk_data.qpos[self._fk_joint_qpos_adrs] = np.asarray(
            joint_pos, dtype=np.float64
        )
        mujoco.mj_kinematics(self._fk_model, self._fk_data)
        body_pos_w = np.asarray(
            self._fk_data.xpos[self._fk_body_ids], dtype=np.float32
        )
        body_quat_w = np.asarray(
            self._fk_data.xquat[self._fk_body_ids], dtype=np.float32
        )
        return self._limb_pose_in_anchor_frame(body_pos_w, body_quat_w)

    def _current_projected_gravity(self) -> np.ndarray:
        root_quat_w = _safe_unit_quat(self.state_processor.root_quat_w)
        return quat_rotate_inverse_numpy(
            root_quat_w.reshape(1, 4),
            np.asarray([[0.0, 0.0, -1.0]], dtype=np.float32),
        )[0].astype(np.float32)

    def reset(self) -> None:
        super().reset()
        for history in self._histories.values():
            history[:] = 0.0
        for part in self._parts.values():
            part[:] = 0.0
        self._obs[:] = 0.0
        self._history_initialized = False

    def update(self, data: Dict[str, Any]) -> None:
        super().update(data)
        paused = bool(data.get("paused", False))

        ref_joint_pos = np.asarray(
            self.ref_joint_pos_future[0, self.obs_current_step_index],
            dtype=np.float32,
        )
        ref_joint_vel = np.asarray(
            self.ref_joint_vel_future[0, self.obs_current_step_index],
            dtype=np.float32,
        ).copy()
        ref_body_pos_w = np.asarray(
            self.ref_body_pos_future_w[0, self.obs_current_step_index],
            dtype=np.float32,
        )
        ref_body_quat_w = np.asarray(
            self.ref_body_quat_future_w[0, self.obs_current_step_index],
            dtype=np.float32,
        )
        motion_ref_ang_vel = np.asarray(
            self.ref_body_ang_vel_future_w[
                0, self.obs_current_step_index, self._tracking_anchor_index
            ],
            dtype=np.float32,
        ).copy()
        if paused:
            ref_joint_vel[:] = 0.0
            motion_ref_ang_vel[:] = 0.0

        joint_pos = np.asarray(
            self.state_processor.joint_pos[self._state_joint_indices],
            dtype=np.float32,
        )
        joint_vel = np.asarray(
            self.state_processor.joint_vel[self._state_joint_indices],
            dtype=np.float32,
        )
        previous_action = np.asarray(
            data.get(
                "action",
                np.zeros(WXY_WBC_ACTION_DIM, dtype=np.float32),
            ),
            dtype=np.float32,
        ).reshape(-1)
        if previous_action.shape != (WXY_WBC_ACTION_DIM,):
            raise ValueError(
                "WXY WBC previous action dim mismatch: "
                f"{previous_action.shape} != ({WXY_WBC_ACTION_DIM},)"
            )

        ref_limb_pose = self._limb_pose_in_anchor_frame(
            ref_body_pos_w, ref_body_quat_w
        )
        robot_limb_pose = self._robot_limb_pose(joint_pos)
        projected_gravity = self._current_projected_gravity()
        base_ang_vel = np.asarray(
            self.state_processor.root_ang_vel_b, dtype=np.float32
        )

        initialize = not self._history_initialized
        values = {
            "ref_limb_ee_pose_b": ref_limb_pose,
            "robot_limb_ee_pose_b": robot_limb_pose,
            "projected_gravity": projected_gravity,
            "base_ang_vel": base_ang_vel,
            "joint_pos": joint_pos - self._default_joint_pos,
            "joint_vel": joint_vel,
            "actions": previous_action,
        }
        for name, value in values.items():
            self._append_history(
                self._histories[name],
                value,
                initialize=initialize,
            )
        self._history_initialized = True

        self._parts["motion_command"][:] = np.concatenate(
            [
                ref_joint_pos,
                ref_joint_vel,
                self._histories["ref_limb_ee_pose_b"].reshape(-1),
                motion_ref_ang_vel,
            ],
            axis=0,
        )
        self._parts["proprioception"][:] = np.concatenate(
            [
                self._histories["robot_limb_ee_pose_b"].reshape(-1),
                self._histories["projected_gravity"].reshape(-1),
                self._histories["base_ang_vel"].reshape(-1),
                self._histories["joint_pos"].reshape(-1),
                self._histories["joint_vel"].reshape(-1),
                self._histories["actions"].reshape(-1),
            ],
            axis=0,
        )
        self._obs[0] = np.concatenate(
            [
                self._parts["motion_command"],
                self._parts["proprioception"],
            ],
            axis=0,
        )
        if not np.all(np.isfinite(self._obs)):
            raise ValueError("WXY WBC observation contains non-finite values")

    def _build_obs_parts(self) -> Dict[str, np.ndarray]:
        return {
            name: value.astype(np.float32, copy=True)
            for name, value in self._parts.items()
        }

    def compute(self) -> np.ndarray:
        return self._obs


class wxy_wbc_component_obs(Observation, namespace="wxy_wbc"):
    """One semantic ONNX input backed by the shared WXY observation core."""

    def __init__(self, component: str, **kwargs: Any) -> None:
        if component not in {"motion_command", "proprioception"}:
            raise ValueError(f"Unsupported WXY WBC component: {component!r}")
        super().__init__(env=kwargs["env"])
        self.component = component
        cache_name = "_wxy_wbc_semantic_observation"
        self._owns_core = not hasattr(self.env, cache_name)
        if self._owns_core:
            setattr(self.env, cache_name, wxy_wbc_policy_obs(**kwargs))
        self._core: wxy_wbc_policy_obs = getattr(self.env, cache_name)

    def reset(self) -> None:
        if self._owns_core:
            self._core.reset()

    def update(self, data: Dict[str, Any]) -> None:
        if self._owns_core:
            self._core.update(data)

    def compute(self) -> np.ndarray:
        return self._core._build_obs_parts()[self.component][None, :]
