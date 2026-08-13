# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pinocchio-based FK / IK / gravity for one side of an OpenArm bimanual model.

Quaternion convention follows ROS (geometry_msgs/Quaternion): [x, y, z, w].
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin


class PinocchioModel:
    """Wraps a Pinocchio model built from the OpenArm bimanual URDF.

    Only the 7 joints of the requested arm side are exposed; the other side and
    finger (mimic) joints are held at neutral. EE frame is ``openarm_<side>_hand_tcp``.
    """

    def __init__(self, urdf_path: str, side: str) -> None:
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        self.side = side

        # mimic=True so finger_joint2 follows finger_joint1 (matches URDF)
        try:
            self.model = pin.buildModelFromUrdf(urdf_path, True)
        except TypeError:
            self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        # Indices of this side's 7 joints within the full q vector.
        self.joint_names = [f"openarm_{side}_joint{i}" for i in range(1, 8)]
        self.q_idx: list[int] = []
        for name in self.joint_names:
            jid = self.model.getJointId(name)
            if jid >= self.model.njoints:
                raise ValueError(f"joint {name!r} not found in URDF model")
            self.q_idx.append(int(self.model.idx_qs[jid]))

        self.ee_frame = f"openarm_{side}_hand_tcp"
        self.ee_fid = self.model.getFrameId(self.ee_frame)
        if self.ee_fid >= len(self.model.frames):
            raise ValueError(f"EE frame {self.ee_frame!r} not found")

        # Per-joint position limits (rad) taken straight from the URDF <limit>.
        self.lower = np.array(
            [self.model.lowerPositionLimit[i] for i in self.q_idx], dtype=float
        )
        self.upper = np.array(
            [self.model.upperPositionLimit[i] for i in self.q_idx], dtype=float
        )

    def _full_q(self, q7: np.ndarray) -> np.ndarray:
        q = pin.neutral(self.model)
        for i, qi in enumerate(self.q_idx):
            q[qi] = q7[i]
        return q

    def fk(self, q7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics: 7 joint angles (rad) -> (position[3], quat_xyzw[4])."""
        q = self._full_q(np.asarray(q7, dtype=float))
        pin.framesForwardKinematics(self.model, self.data, q)
        oMf = self.data.oMf[self.ee_fid]
        quat_xyzw = np.asarray(pin.Quaternion(oMf.rotation).coeffs(), dtype=float)
        return oMf.translation.copy(), quat_xyzw

    def gravity(self, q7: np.ndarray) -> np.ndarray:
        """Gravity torques (Nm) for this side's 7 joints at configuration q7."""
        q = self._full_q(np.asarray(q7, dtype=float))
        g = pin.computeGeneralizedGravity(self.model, self.data, q)
        return np.array([g[i] for i in self.q_idx], dtype=float)

    def ik(
        self,
        target_pos: np.ndarray,
        target_quat_xyzw: np.ndarray,
        q_init: np.ndarray,
        max_iters: int = 50,
        tol: float = 1e-4,
        damping: float = 1e-2,
    ) -> np.ndarray | None:
        """Damped-least-squares IK (Pinocchio 3.x has no built-in IK solver).

        Returns 7 joint angles (rad) within limits, or None if not converged.
        """
        R = pin.Quaternion(np.asarray(target_quat_xyzw, dtype=float)).matrix()
        target_se3 = pin.SE3(R, np.asarray(target_pos, dtype=float))

        q7 = np.clip(
            np.asarray(q_init, dtype=float).copy(), self.lower, self.upper
        )
        for _ in range(max_iters):
            q = self._full_q(q7)
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacement(self.model, self.data, self.ee_fid)
            oMf = self.data.oMf[self.ee_fid]
            err = pin.log(oMf.inverse() * target_se3).vector  # 6D (pos+ori)
            if np.linalg.norm(err) < tol:
                return q7
            J = pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                self.ee_fid,
                pin.ReferenceFrame.LOCAL,
            )
            J7 = J[:, self.q_idx]  # 6 x 7
            JJt = J7 @ J7.T + (damping**2) * np.eye(6)
            dq = J7.T @ np.linalg.solve(JJt, err)
            q7 = np.clip(q7 + dq, self.lower, self.upper)
        return None
