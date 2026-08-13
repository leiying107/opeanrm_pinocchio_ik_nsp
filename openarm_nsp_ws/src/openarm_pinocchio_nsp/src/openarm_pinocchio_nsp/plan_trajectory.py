#!/usr/bin/env python3
# Copyright 2026 Enactic, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Standalone trajectory PLANNER — no execution, no CAN, no hardware.

Splits planning from execution: this tool takes recorded points + a max joint
speed + a control frequency and OUTPUTS a joint pose chain (time-stamped joint
angles) to a JSON file. Execution is a separate part — the chain is
``joint_trajectory_controller``-shaped so any executor (ros2_control, the
dashboard, a custom node) can consume it.

Pipeline (pure computation):
    recorded joint points  →  FK → 6D poses
                            →  pose_replay_traj: pose interp at control freq
                              + per-point IK, time by max-speed
                            →  joint pose chain (times, q_path)

Input points file (JSON) — a list of 7-element joint-angle arrays (rad), e.g.
    [[0,0,0,1.57,0,0,0], [0.3,-0.2,0,1.2,0,0.3,0], [0.6,-0.4,0.3,0.9,0.2,0.6,0]]

Run:
    ros2 run openarm_pinocchio_nsp plan_trajectory \\
        --points pts.json --max-speed 1.0 --freq 100 --side right \\
        --output chain.json
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .cartesian_planner import Waypoint, check_traj_smoothness, pose_replay_traj
from .kinematics import PinocchioModel
from .urdf_path import resolve_urdf_path


def joint_names(side: str) -> list[str]:
    return [f"openarm_{side}_joint{i}" for i in range(1, 8)]


def load_points(path: str) -> list[np.ndarray]:
    """Load a JSON file of 7-element joint-angle arrays (rad)."""
    with open(path) as f:
        pts = json.load(f)
    out = [np.asarray(p, dtype=float).ravel() for p in pts]
    if any(len(p) != 7 for p in out):
        raise ValueError("each point must have 7 joint angles")
    return out


def build_chain(
    model: PinocchioModel,
    points: list[np.ndarray],
    max_speed: float,
    freq: float,
    q_seed: np.ndarray | None = None,
    null_iters: int = 6,
) -> dict | None:
    """Plan a joint pose chain from recorded points. Pure planner.

    Returns a dict with ``times`` and ``q_path`` (list of 7 joint angles per
    point), or None if any IK fails. No execution.
    """
    if len(points) < 2:
        raise ValueError("need at least 2 points")
    poses = [Waypoint(*model.fk(np.asarray(q, dtype=float))) for q in points]
    seed = np.asarray(q_seed if q_seed is not None else points[0], dtype=float)
    traj = pose_replay_traj(model, poses, seed, max_speed, freq, null_iters=null_iters)
    if traj is None:
        return None
    times, q_path = traj
    return {"times": list(times), "q_path": [list(q) for q in q_path]}


def chain_to_joint_trajectory(chain: dict, side: str, max_speed: float, freq: float) -> dict:
    """Serialize a chain as a ros2_control JointTrajectory-shaped dict.

    Keeps both the ROS shape (joint_names + points[{positions,time_from_start}])
    and the raw times/positions, so any executor can consume it.
    """
    times, q_path = chain["times"], chain["q_path"]
    return {
        "joint_names": joint_names(side),
        "points": [
            {
                "positions": q_path[i],
                "time_from_start": {"sec": int(t), "nanosec": int(round((t - int(t)) * 1e9))},
            }
            for i, t in enumerate(times)
        ],
        # raw form for non-ROS / dashboard executors
        "times": times,
        "positions": q_path,
        "duration": times[-1] if times else 0.0,
        "max_speed": max_speed,
        "freq": freq,
        "n_points": len(q_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Plan a joint pose chain (points+speed+freq) → JSON. No execution.")
    ap.add_argument("--points", required=True, help="JSON file: list of 7 joint angles (rad)")
    ap.add_argument("--max-speed", type=float, required=True, help="max joint angular speed (rad/s)")
    ap.add_argument("--freq", type=float, default=100.0, help="control frequency / sampling (Hz)")
    ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--q-seed", default=None,
                    help="JSON '7 angles' IK warm-start seed (default: first point)")
    ap.add_argument("--null-iters", type=int, default=6, help="IK null-space iters (6 fast, 12 thorough)")
    ap.add_argument("--output", required=True, help="output chain JSON path")
    args = ap.parse_args()

    model = PinocchioModel(resolve_urdf_path(), args.side)
    points = load_points(args.points)
    q_seed = np.asarray(json.loads(args.q_seed), dtype=float) if args.q_seed else None

    chain = build_chain(model, points, args.max_speed, args.freq, q_seed, args.null_iters)
    if chain is None:
        print("PLANNING FAILED: some point unreachable / IK branch break", file=sys.stderr)
        return 1

    # warn-only smoothness check (does not block)
    chk = check_traj_smoothness(chain["times"], [np.asarray(q) for q in chain["q_path"]])
    for w in chk.warnings:
        print(w, file=sys.stderr)

    out = chain_to_joint_trajectory(chain, args.side, args.max_speed, args.freq)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    peak = chk.max_velocity
    print(f"planned {out['n_points']}-point chain | duration {out['duration']:.2f}s | "
          f"peak {peak:.2f} rad/s | max_step {chk.max_step:.4f} rad -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
