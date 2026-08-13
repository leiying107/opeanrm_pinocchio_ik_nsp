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

"""Forward-kinematics CLI: 7 joint angles (rad) -> EE pose (position + xyzw quat).

Examples:
    ros2 run openarm_pinocchio_nsp fk --side right --joints 0,0,0,0,0,0,0
    ros2 run openarm_pinocchio_nsp fk --side right --deg 0,-30,0,90,0,45,0
"""

import argparse
import re

import numpy as np

from openarm_pinocchio_nsp.kinematics import PinocchioModel
from openarm_pinocchio_nsp.urdf_path import resolve_urdf_path

_DEFAULT_URDF = resolve_urdf_path()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenArm forward kinematics: joints -> EE pose"
    )
    parser._negative_number_matcher = re.compile(r"^-?[0-9]")
    parser.add_argument("--side", default="right", choices=["left", "right", "both"])
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--joints",
        default="0,0,0,0,0,0,0",
        help="7 joint angles, comma-separated, in rad (default)",
    )
    group.add_argument("--deg", help="7 joint angles, comma-separated, in degrees")
    parser.add_argument("--urdf", default=_DEFAULT_URDF)
    args = parser.parse_args()

    raw = args.deg if args.deg else args.joints
    q = np.array([float(x) for x in raw.split(",")], dtype=float)
    if len(q) != 7:
        raise SystemExit(f"expected 7 joint angles, got {len(q)}")
    if args.deg:
        q = np.deg2rad(q)

    sides = ["left", "right"] if args.side == "both" else [args.side]
    for side in sides:
        model = PinocchioModel(args.urdf, side)
        pos, quat = model.fk(q)
        print(f"side = {side}")
        print(f"joints (rad)      : {np.round(q, 4).tolist()}")
        print(f"EE position   xyz : [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        print(f"EE orientation xyzw: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")


if __name__ == "__main__":
    main()
