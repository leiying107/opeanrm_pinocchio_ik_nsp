# Copyright 2026 Enactic, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gravity compensation G(q) for OpenArm v1.0 — mirrors the validated
``openarm_panel/hardware_patch/openarm_simple_hardware.cpp`` (KDL JntToGravity).

Algorithm (identical numbers to KDL on the same URDF):
  * model = v1_simple.urdf (19 masses, simplified, no hand_tcp). Resolved from
    $OPENARM_GRAVITY_URDF -> /tmp/v1_simple.urdf -> bundled copy.
  * motor_q used DIRECTLY as URDF joint positions — v1_simple.urdf joint zeros
    already match the motor encoders (verified: FK matches example/v1.urdf).
    NO motor->URDF offset (the cpp's offset is for a different URDF).
  * G(urdf_q) computed with Pinocchio computeGeneralizedGravity (== KDL
    JntToGravity; same rigid-body gravity term, gravity vector (0,0,-9.81)).
  * return clamp(scale * G, +/- TMAX) with TMAX = [54,54,28,28,10,10,10]
    (DM8009 / DM4340 / DM4310).

Purpose: feed scale*G(q) into the MIT tau term so the PD loop doesn't have to
fight gravity (cancels steady-state sag = G(q)/kp).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# motor->URDF joint offsets (urdf_q = motor_q - offset). From the cpp.
OFFSETS = {
    "left":  np.array([0.0, 0.506145, -1.570796, -1.745329, 0.0, -0.331612, 1.570796]),
    "right": np.array([0.0, -0.506145, 1.570796, -1.745329, 0.0, 0.331612, -1.570796]),
}
# per-joint torque clamp (Nm): DM8009, DM8009, DM4340, DM4340, DM4310 x3
TMAX = np.array([54.0, 54.0, 28.0, 28.0, 10.0, 10.0, 10.0])
GRAV = np.array([0.0, 0.0, -9.81])


def _resolve_urdf() -> str | None:
    for p in (os.environ.get("OPENARM_GRAVITY_URDF"), "/tmp/v1_simple.urdf",
              str(Path(__file__).parent / "v1_simple.urdf")):
        if p and os.path.isfile(p):
            return p
    return None


class GravityComp:
    """Per-joint gravity torque G(q) on v1_simple.urdf (KDL-equivalent)."""

    def __init__(self):
        import pinocchio as pin  # numpy<2 via the dashboard venv
        self._pin = pin
        path = _resolve_urdf()
        if not path:
            raise FileNotFoundError(
                "v1_simple.urdf not found (set OPENARM_GRAVITY_URDF or place /tmp/v1_simple.urdf)")
        self.model = pin.buildModelFromUrdf(path)
        # match KDL's gravity vector exactly (pinocchio default is already (0,0,-9.81),
        # set explicitly to remove any ambiguity)
        self.model.gravity.linear = GRAV
        self.data = self.model.createData()
        self.qidx: dict[str, list[int]] = {}
        for side in ("left", "right"):
            self.qidx[side] = [
                int(self.model.idx_qs[self.model.getJointId(f"openarm_{side}_joint{k}")])
                for k in range(1, 8)
            ]

    def torque(self, side: str, motor_q, scale: float) -> np.ndarray:
        """Return the 7-vector MIT tau feedforward = clamp(scale*G(q), +/-TMAX).

        ``motor_q`` is the sensed 7 motor angles (rad); the motor->URDF offset is
        applied internally. Sign convention matches the cpp: add this to tau.
        """
        pin = self._pin
        q = pin.neutral(self.model)
        qi = self.qidx[side]
        # v1_simple.urdf joint zeros already match the motor encoders, so motor_q
        # is used DIRECTLY (NO motor->URDF offset) — verified by FK matching
        # example/v1.urdf at home/hang/reach. (The cpp's offset is for a different URDF.)
        for i in range(7):
            q[qi[i]] = float(motor_q[i])
        g = pin.computeGeneralizedGravity(self.model, self.data, q)
        tau = np.array([g[qi[i]] for i in range(7)])
        return np.clip(float(scale) * tau, -TMAX, TMAX)


if __name__ == "__main__":
    # sanity self-check: print G(q) at home vs elbow-extended; magnitudes should
    # be bounded and grow on the shoulder/elbow when the arm reaches out.
    gc = GravityComp()
    print("gravity vector:", gc.model.gravity.linear)
    for cfg_name, q in [("home (zeros)", [0]*7),
                        ("elbow out j4=1.5", [0, 0, 0, 1.5, 0, 0, 0]),
                        ("reach out", [0.5, 0.5, 0, 1.8, 0, 0.5, 0])]:
        for side in ("left", "right"):
            t = gc.torque(side, np.array(q, dtype=float), 1.0)
            print(f"{side:5s} {cfg_name:16s} G(q)=", np.round(t, 2))
