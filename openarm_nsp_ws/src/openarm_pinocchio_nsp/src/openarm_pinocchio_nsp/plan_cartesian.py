#!/usr/bin/env python3
# Copyright 2026 Enactic, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CLI: plan (and optionally execute) a Cartesian trajectory via NSP-IK.

Default mode is offline: it solves the path, prints a per-sample safety report
and the go/no-go gate, and writes the joint trajectory to JSON. ``--publish``
additionally sends the trajectory to ``/<side>_joint_trajectory_controller`` on
the real/simulated arm.

Examples:
    # offline plan a 10 cm forward straight line, right arm
    ros2 run openarm_pinocchio_nsp plan_cartesian --side right \
        --line "0.35,-0.1,0.5 -> 0.45,-0.1,0.5"

    # plan + execute (requires bimanual launch running)
    ros2 run openarm_pinocchio_nsp plan_cartesian --side right --publish \
        --line "0.35,-0.1,0.5 -> 0.40,-0.05,0.55"
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np

from openarm_pinocchio_nsp.cartesian_planner import Waypoint, plan_cartesian
from openarm_pinocchio_nsp.kinematics import PinocchioModel, SIGMA_WARN
from openarm_pinocchio_nsp.urdf_path import resolve_urdf_path

_LINE_RE = re.compile(r"\[?([-\d.,\s]+)\]?\s*->\s*\[?([-\d.,\s]+)\]?")


def parse_line(spec: str) -> tuple[np.ndarray, np.ndarray]:
    m = _LINE_RE.search(spec)
    if not m:
        raise SystemExit(f"bad --line spec: {spec!r} (use 'x,y,z -> x,y,z')")
    p1 = np.array([float(v) for v in m.group(1).split(",")])
    p2 = np.array([float(v) for v in m.group(2).split(",")])
    if len(p1) != 3 or len(p2) != 3:
        raise SystemExit("--line endpoints must be x,y,z")
    return p1, p2


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline/online Cartesian trajectory planner")
    ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--urdf", default=None)
    ap.add_argument("--line", required=True, help="'x,y,z -> x,y,z'")
    ap.add_argument("--quat", default=None, help="constant orientation xyzw (default: start pose)")
    ap.add_argument("--publish", action="store_true", help="send trajectory to the controller")
    ap.add_argument("--output", default=None, help="write trajectory JSON to this path")
    args = ap.parse_args()

    urdf = resolve_urdf_path(args.urdf)
    model = PinocchioModel(urdf, args.side)

    # seed: well-conditioned hands_up pose
    q_init = np.zeros(7)
    q_init[3] = 2.0
    q_init = np.clip(q_init, model.lower, model.upper)

    start_pos, end_pos = parse_line(args.line)
    # orientation: constant; default = orientation at the seed
    _, start_quat = model.fk(q_init)
    quat = np.array([float(v) for v in args.quat.split(",")]) if args.quat else start_quat

    waypoints = [Waypoint(start_pos, quat), Waypoint(end_pos, quat)]
    print(f"planning {args.side} arm: {start_pos.round(3).tolist()} -> "
          f"{end_pos.round(3).tolist()}, quat={np.round(quat, 3).tolist()}")

    result = plan_cartesian(model, waypoints, q_init)

    # ---- report --------------------------------------------------------
    print(f"samples: {len(result.q_path)}  success: {result.success}")
    if not result.success:
        print(f"!! path broke at sample {result.break_index} (unreachable / σ_min too low)")
        return 1

    sig = [d["sigma_min"] for d in result.diagnostics]
    mar = [d["joint_margin"] for d in result.diagnostics]
    print(f"σ_min along path: min={min(sig):.4f}  med={np.median(sig):.4f}  "
          f"(σ_warn={SIGMA_WARN})")
    print(f"joint margin:     min={min(mar):.3f} rad")
    print(f"duration:         {result.times[-1]:.2f} s")
    gate = result.passed_gate
    print(f"GO/NO-GO gate:    {'PASS (safe to run)' if gate else 'REVIEW (near singularity/limits)'}")

    if args.output:
        traj = {
            "side": args.side,
            "joint_names": [f"openarm_{args.side}_joint{i}" for i in range(1, 8)],
            "times": result.times,
            "positions": [q.tolist() for q in result.q_path],
            "sigma_min": sig,
        }
        with open(args.output, "w") as f:
            json.dump(traj, f, indent=2)
        print(f"trajectory written: {args.output}")

    # ---- optional execution -------------------------------------------
    if args.publish:
        rc = _publish(model, args.side, result)
        if rc != 0:
            return rc

    return 0 if gate else 2  # 2 = planned but gate not passed (review)


def _publish(model: PinocchioModel, side: str, result) -> int:
    import rclpy
    from builtin_interfaces.msg import Duration
    from rclpy.node import Node
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    if not result.passed_gate:
        print("refusing to publish: go/no-go gate not passed")
        return 1

    rclpy.init()
    node = Node("plan_cartesian")
    pub = node.create_publisher(
        JointTrajectory, f"/{side}_joint_trajectory_controller/joint_trajectory", 10
    )
    joint_names = [f"openarm_{side}_joint{i}" for i in range(1, 8)]

    traj = JointTrajectory(joint_names=joint_names)
    for q, t in zip(result.q_path, result.times):
        pt = JointTrajectoryPoint(positions=q.tolist())
        sec = int(t)
        pt.time_from_start = Duration(sec=sec, nanosec=int((t - sec) * 1e9))
        traj.points.append(pt)

    # wait a moment for the publisher to connect, then send
    node.get_logger().info(f"publishing {len(traj.points)}-point trajectory ({result.times[-1]:.2f}s)")
    import time
    time.sleep(0.5)
    pub.publish(traj)
    time.sleep(result.times[-1] + 0.5)
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
