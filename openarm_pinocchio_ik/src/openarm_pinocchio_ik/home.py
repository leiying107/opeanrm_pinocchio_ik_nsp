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

"""Home command: move arm to the zero pose (all 7 joints at 0 rad).

Shortcut for ``move_joints --joints 0,0,0,0,0,0,0``. Reads current joint
positions from /joint_states and smoothly interpolates to zero.

Example:
    ros2 run openarm_pinocchio_ik home --side right
    ros2 run openarm_pinocchio_ik home --side left --time 3.0
"""

import argparse
import re
import time

import numpy as np
import rclpy

from openarm_pinocchio_ik.move_joints import MoveJoints, sides_from_arg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move real OpenArm to HOME (all joints at 0 rad)"
    )
    parser._negative_number_matcher = re.compile(r"^-?[0-9]")
    parser.add_argument("--side", default="right", choices=["left", "right", "both"])
    parser.add_argument("--time", type=float, default=2.0, help="time to reach (s)")
    args = parser.parse_args()

    rclpy.init()
    node = MoveJoints(sides_from_arg(args.side), np.zeros(7), args.time, n_pts=30)
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
