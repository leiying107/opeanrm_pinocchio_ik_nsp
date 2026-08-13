#!/usr/bin/env python3
# Copyright 2026 Enactic, Inc.
# Offline validation of PinocchioModel: FK<->IK round-trip + gravity sanity.

import math
import os
import sys

import numpy as np
import pinocchio as pin

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from openarm_pinocchio_ik.kinematics import PinocchioModel  # noqa: E402

URDF = (
    "/ros2_ws/install/openarm_description/share/openarm_description/"
    "assets/robot/openarm_v1.0/urdf/example/v1.urdf"
)


def quat_angle_err(q1_xyzw, q2_xyzw) -> float:
    R1 = pin.Quaternion(np.asarray(q1_xyzw, float)).matrix()
    R2 = pin.Quaternion(np.asarray(q2_xyzw, float)).matrix()
    R = R1.T @ R2
    return math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))


def main() -> None:
    model = PinocchioModel(URDF, "right")
    print(f"right arm q_idx={model.q_idx}")
    print(f"  lower={np.round(model.lower, 3)}")
    print(f"  upper={np.round(model.upper, 3)}")

    home = (model.lower + model.upper) / 2.0
    pos0, quat0 = model.fk(home)
    print(f"home FK pos={np.round(pos0, 4)}  quat(xyzw)={np.round(quat0, 4)}")
    print(f"home gravity (Nm)={np.round(model.gravity(home), 3)}")

    rng = np.random.default_rng(0)
    n, fail, max_pe, max_oe = 20, 0, 0.0, 0.0
    for _ in range(n):
        q = rng.uniform(model.lower, model.upper)
        pos, quat = model.fk(q)
        qs = model.ik(pos, quat, q_init=home)
        if qs is None:
            fail += 1
            continue
        pos2, quat2 = model.fk(qs)
        max_pe = max(max_pe, float(np.linalg.norm(pos - pos2)))
        max_oe = max(max_oe, quat_angle_err(quat, quat2))
        assert np.all(qs >= model.lower - 1e-6) and np.all(qs <= model.upper + 1e-6), (
            "IK result out of limits"
        )

    print(
        f"FK->IK->FK: {n - fail}/{n} converged, "
        f"max pos err={max_pe * 1000:.3f} mm, max ori err={math.degrees(max_oe):.3f} deg"
    )
    if fail > n // 2:
        print("WARNING: many IK failures (targets may be unreachable from home seed)")


if __name__ == "__main__":
    main()
