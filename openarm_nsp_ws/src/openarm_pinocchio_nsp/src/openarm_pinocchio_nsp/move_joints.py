#!/usr/bin/env python3
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

"""Joint-space move CLI: send the real arm(s) to given joint angles.

Reads current joint positions from /joint_states, builds a linearly interpolated
multi-point trajectory from there to the target, and publishes it to
``/<side>_joint_trajectory_controller/joint_trajectory`` for each requested side.
Requires the bimanual launch to be running.

Examples:
    ros2 run openarm_pinocchio_nsp move_joints --side right --deg 0,-30,0,90,0,45,0
    ros2 run openarm_pinocchio_nsp move_joints --side both --joints 0,0,0,0,0,0,0 --time 3.0
"""

import argparse
import re
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def sides_from_arg(side: str) -> list[str]:
    if side == "both":
        return ["left", "right"]
    return [side]


class MoveJoints(Node):
    def __init__(self, sides: list[str], target: np.ndarray, t_sec: float, n_pts: int) -> None:
        super().__init__("move_joints")
        self.sides = sides
        self.target = target
        self.t_sec = t_sec
        self.n_pts = n_pts
        self.joint_names: dict[str, list[str]] = {}
        self.pubs: dict[str, rclpy.publisher.Publisher] = {}
        self.current: dict[str, np.ndarray | None] = {}
        self.received_joint_names: set[str] = set()
        for side in sides:
            self.joint_names[side] = [f"openarm_{side}_joint{i}" for i in range(1, 8)]
            self.pubs[side] = self.create_publisher(
                JointTrajectory,
                f"/{side}_joint_trajectory_controller/joint_trajectory",
                10,
            )
            self.current[side] = None
        self.declare_parameter("joint_state_timeout_sec", 10.0)
        self.joint_state_timeout = self.get_parameter("joint_state_timeout_sec").value
        self.create_subscription(JointState, "/joint_states", self._on_js, qos_profile_sensor_data)
        self.create_timer(0.1, self._tick)
        self.sent = False
        self.t0 = time.time()
        self.last_log_time = self.t0

    def _on_js(self, msg: JointState) -> None:
        self.received_joint_names.update(msg.name)
        for side in self.sides:
            try:
                self.current[side] = np.array(
                    [msg.position[msg.name.index(n)] for n in self.joint_names[side]]
                )
            except ValueError:
                pass

    def _tick(self) -> None:
        if self.sent:
            return
        missing = [s for s in self.sides if self.current[s] is None]
        if missing:
            now = time.time()
            elapsed = now - self.t0
            if elapsed > self.joint_state_timeout:
                for side in missing:
                    required = set(self.joint_names[side])
                    received = self.received_joint_names.intersection(required)
                    missing_joints = required - received
                    self.get_logger().error(
                        f"timeout after {elapsed:.1f}s waiting for /joint_states for side '{side}': "
                        f"received {sorted(received)}, missing {sorted(missing_joints)}"
                    )
                raise SystemExit(1)
            if now - self.last_log_time >= 1.0:
                for side in missing:
                    self.get_logger().info(
                        f"waiting for /joint_states for side {side}..."
                    )
                self.last_log_time = now
            return
        for side in self.sides:
            cur = self.current[side]
            assert cur is not None
            traj = JointTrajectory()
            traj.joint_names = self.joint_names[side]
            for i in range(self.n_pts + 1):
                s = i / self.n_pts
                q = cur + s * (self.target - cur)
                pt = JointTrajectoryPoint()
                pt.positions = q.tolist()
                ts = s * self.t_sec
                pt.time_from_start = Duration(
                    sec=int(ts), nanosec=int((ts - int(ts)) * 1e9)
                )
                traj.points.append(pt)
            self.pubs[side].publish(traj)
            self.get_logger().info(
                f"move {side}: {np.round(cur, 2).tolist()} -> "
                f"{np.round(self.target, 2).tolist()} over {self.t_sec}s"
            )
        self.sent = True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move real OpenArm to given joint angles (joint-space control)"
    )
    parser._negative_number_matcher = re.compile(r"^-?[0-9]")
    parser.add_argument("--side", default="right", choices=["left", "right", "both"])
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--joints", default="0,0,0,0,0,0,0", help="7 target angles in rad"
    )
    group.add_argument("--deg", help="7 target angles in degrees")
    parser.add_argument("--time", type=float, default=2.0, help="time to reach (s)")
    parser.add_argument("--n_pts", type=int, default=30, help="trajectory waypoints")
    args = parser.parse_args()

    raw = args.deg if args.deg else args.joints
    q = np.array([float(x) for x in raw.split(",")], dtype=float)
    if len(q) != 7:
        raise SystemExit(f"expected 7 joint angles, got {len(q)}")
    if args.deg:
        q = np.deg2rad(q)

    rclpy.init()
    node = MoveJoints(sides_from_arg(args.side), q, args.time, args.n_pts)
    end = time.time() + args.time + 1.0
    try:
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.sent and time.time() - node.t0 > args.time + 0.3:
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
