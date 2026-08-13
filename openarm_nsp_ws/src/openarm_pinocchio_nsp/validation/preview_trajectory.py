#!/usr/bin/env python3
# Copyright 2026 Enactic, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Offline 3D preview of a planned trajectory — ZERO hardware risk.

Reads the JSON written by ``plan_cartesian --output`` and animates the arm
following the joint path in a Meshcat 3D viewer (browser). No ROS, no CAN,
no controller — the real arm cannot move because nothing is published to it.
This is the safe dry-run before real-hardware deployment.

Usage:
    # 1. plan offline, write trajectory
    ros2 run openarm_pinocchio_nsp plan_cartesian --side right \
        --line "0.22,-0.04,0.51 -> 0.30,-0.04,0.51" --output /tmp/traj.json
    # 2. preview in browser (no hardware)
    python preview_trajectory.py /tmp/traj.json
    #    -> open the printed URL in a browser; loops until Ctrl-C
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pinocchio as pin

# Meshcat needs the URDF *with meshes*. The v1.urdf references mesh files; we
# build the model from it and feed mesh paths to the visualizer.
from openarm_pinocchio_nsp.urdf_path import resolve_urdf_path

try:
    from pinocchio.visualize import MeshcatVisualizer
except ImportError as e:
    print("meshcat not installed: pip install meshcat ;", e)
    sys.exit(1)


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    traj_path = sys.argv[1]
    with open(traj_path) as f:
        traj = json.load(f)

    positions = [np.array(p) for p in traj["positions"]]
    times = traj["times"]
    side = traj.get("side", "right")
    n = len(positions)
    duration = times[-1] if times else 1.0

    urdf = resolve_urdf_path()
    # bimanual model; we animate only `side`'s 7 joints, rest at neutral
    model = pin.buildModelFromUrdf(urdf, True)
    # load visual meshes (dae/stl); package://openarm_description needs package_dirs
    PACKAGE_DIRS = [
        "/ros2_ws/openarm_ros2/install/openarm_description/share",
        "/ros2_ws/openarm_ros2/openarm_description",
    ]
    try:
        geom_model = pin.buildGeomFromUrdf(
            model, urdf, pin.GeometryType.VISUAL, PACKAGE_DIRS
        )
    except Exception as e:
        print(f"warning: mesh load failed ({type(e).__name__}); frames-only", flush=True)
        geom_model = pin.GeometryModel()

    data = model.createData()
    q_idx = []
    for i in range(1, 8):
        jid = model.getJointId(f"openarm_{side}_joint{i}")
        q_idx.append(int(model.idx_qs[jid]))

    viz = MeshcatVisualizer(model, geom_model, geom_model)
    viz.initViewer(open=False)
    viz.loadViewerModel(rootNodeName="openarm_preview")
    url = viz.viewer.url()
    print("=" * 60)
    print("3D PREVIEW (offline, no hardware)")
    print("=" * 60)
    print(f"trajectory: {traj_path}  ({n} samples, {duration:.2f}s, {side})")
    print(f"σ_min along path: min={min(traj.get('sigma_min',[0])):.3f}")
    print()
    print(f">>> open this URL in a browser to see the arm move <<<")
    print(f"    {url}")
    print("    (animation loops; Ctrl-C to quit)")
    print("=" * 60)

    def full_q(q7):
        q = pin.neutral(model)
        for i, qi in enumerate(q_idx):
            q[qi] = q7[i]
        return q

    # show start pose first
    viz.display(full_q(positions[0]))
    time.sleep(2.0)

    # loop the animation
    try:
        while True:
            for i in range(n):
                viz.display(full_q(positions[i]))
                # hold each sample proportional to its time slice
                dt = (times[i] - times[i - 1]) if i > 0 else 0.05
                time.sleep(max(0.02, min(dt, 0.2)))
            time.sleep(1.0)  # pause between loops
    except KeyboardInterrupt:
        print("\npreview stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
