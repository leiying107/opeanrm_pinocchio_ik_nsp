#!/usr/bin/env python3
# Copyright 2026 Enactic, Inc.
# SPDX-License-Identifier: Apache-2.0
"""P1 validation: closed-loop check against fake_hardware.

Requires the bimanual bringup running in simulation::

    ros2 launch openarm_bringup openarm.bimanual.launch.py \
        arm_type:=openarm_v1.0 description_file:=v10.urdf.xacro \
        use_fake_hardware:=true robot_controller:=joint_trajectory_controller

Reads the arm's *current* pose from ``/joint_states``, plans a short Cartesian
step from there, publishes it, then reads back the achieved joints and
FK-compares commanded vs achieved EE. Under fake_hardware the residual is the
IK+FK internal consistency (sub-mm).

Run::
    python src/openarm_pinocchio_nsp/validation/v_closed_loop_fk_compare.py --side right
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from openarm_pinocchio_nsp.cartesian_planner import Waypoint, plan_cartesian
from openarm_pinocchio_nsp.kinematics import PinocchioModel
from openarm_pinocchio_nsp.urdf_path import resolve_urdf_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="right")
    ap.add_argument("--urdf", default=None)
    ap.add_argument("--step", default="0.06,0,0.02", help="EE delta xyz (m)")
    args = ap.parse_args()

    model = PinocchioModel(resolve_urdf_path(args.urdf), args.side)
    joint_names = [f"openarm_{args.side}_joint{i}" for i in range(1, 8)]
    delta = np.array([float(v) for v in args.step.split(",")])

    rclpy.init()
    node = Node("closed_loop_fk_compare")
    cur = {"q": None}

    def on_js(msg: JointState):
        try:
            cur["q"] = np.array([msg.position[msg.name.index(n)] for n in joint_names])
        except ValueError:
            pass

    node.create_subscription(JointState, "/joint_states", on_js, qos_profile_sensor_data)
    pub = node.create_publisher(
        JointTrajectory, f"/{args.side}_joint_trajectory_controller/joint_trajectory", 10
    )

    # read current arm pose — seed the path from where the arm actually is
    t0 = time.time()
    while cur["q"] is None and time.time() - t0 < 5.0:
        rclpy.spin_once(node, timeout_sec=0.1)
    if cur["q"] is None:
        print("no /joint_states — is fake_hardware bringup running?")
        node.destroy_node(); rclpy.shutdown(); return 1

    q_seed = cur["q"]
    p0, quat0 = model.fk(q_seed)
    print(f"current EE: {np.round(p0, 3).tolist()}  "
          f"σ_min={model.singular_values(q_seed)[-1]:.3f}")

    result = plan_cartesian(
        model,
        [Waypoint(p0, quat0), Waypoint(p0 + delta, quat0)],
        q_seed,
    )
    if not result.success:
        print(f"plan failed at sample {result.break_index}; aborting")
        node.destroy_node(); rclpy.shutdown(); return 1

    traj = JointTrajectory(joint_names=joint_names)
    for q, t in zip(result.q_path, result.times):
        pt = JointTrajectoryPoint(positions=q.tolist())
        pt.time_from_start = Duration(sec=int(t), nanosec=int((t - int(t)) * 1e9))
        traj.points.append(pt)

    time.sleep(0.8)  # let publisher connect
    pub.publish(traj)
    print(f"sent {len(traj.points)}-point trajectory ({result.times[-1]:.2f}s); "
          f"gate={'PASS' if result.passed_gate else 'REVIEW'}")

    # wait for execution, sampling the final pose
    cur["q"] = None
    end = time.time() + result.times[-1] + 2.0
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
        if cur["q"] is not None:
            q_got = cur["q"].copy()

    q_cmd = result.q_path[-1]
    joint_err = float(np.max(np.abs(q_cmd - q_got)))
    pos_cmd, _ = model.fk(q_cmd)
    pos_got, _ = model.fk(q_got)
    ee_err_mm = float(np.linalg.norm(pos_cmd - pos_got) * 1000)

    print(f"final joint tracking error: {joint_err:.5f} rad")
    print(f"final EE position error:     {ee_err_mm:.3f} mm")
    ok = ee_err_mm < 5.0
    print("PASS — closed-loop EE tracks command" if ok else "REVIEW")
    node.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
