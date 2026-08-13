#!/usr/bin/env python3
# Copyright 2026 Enactic, Inc.
# SPDX-License-Identifier: Apache-2.0
"""P1 validation: plan + audit standard Cartesian paths (continuity, σ_min,
joint margin, FK round-trip) and apply the go/no-go gate.

Runs three canonical paths from a well-conditioned start pose:
  * straight line (10 cm forward)
  * horizontal circle (radius 6 cm)
  * orientation scan (yaw ±30° in place)

A path passes only if every sample is solved, σ_min ≥ σ_warn along the whole
path, joint margin > 0.1 rad, and the joint chain is continuous (no jumps).

Run::
    python src/openarm_pinocchio_nsp/validation/v_trajectory_validation.py
"""

from __future__ import annotations

import sys

import numpy as np
import pinocchio as pin

from openarm_pinocchio_nsp.cartesian_planner import Waypoint, plan_cartesian
from openarm_pinocchio_nsp.kinematics import PinocchioModel, SIGMA_WARN
from openarm_pinocchio_nsp.urdf_path import DEFAULT_URDF


def _circle_waypoints(center, radius, quat, n=12):
    return [
        Waypoint(center + radius * np.array([np.cos(a), np.sin(a), 0.0]), quat)
        for a in np.linspace(0, 2 * np.pi, n, endpoint=False)
    ]


def _yaw_scan_waypoints(pos, quat_xyzw, deg=30, n=7):
    q0 = pin.Quaternion(np.asarray(quat_xyzw, float))
    out = []
    for d in np.linspace(-deg, deg, n):
        rz = pin.Quaternion(np.array([0.0, 0.0, np.sin(np.deg2rad(d) / 2),
                                       np.cos(np.deg2rad(d) / 2)]))
        out.append(Waypoint(pos, (rz * q0).coeffs()))
    return out


def _audit(model, name, waypoints, q_init):
    r = plan_cartesian(model, waypoints, q_init)
    if not r.success:
        print(f"  [{name}] FAILED at sample {r.break_index}")
        return False
    sig = [d["sigma_min"] for d in r.diagnostics]
    max_dq = max(
        np.max(np.abs(r.q_path[i + 1] - r.q_path[i])) for i in range(len(r.q_path) - 1)
    )
    max_fk_mm = max(d["pos_err_mm"] for d in r.diagnostics)
    ok = r.passed_gate and max_dq < 0.2 and max_fk_mm < 1.0
    print(
        f"  [{name}] samples={len(r.q_path):3d}  "
        f"σ_min(min)={min(sig):.3f}  margin(min)={r.min_margin:.3f}  "
        f"maxΔq={max_dq:.3f}  fk_err={max_fk_mm:.2f}mm  "
        f"{'PASS' if ok else 'REVIEW'}"
    )
    return ok


def main() -> int:
    model = PinocchioModel(DEFAULT_URDF, "right")
    q0 = np.zeros(7)
    q0[3] = 2.0
    q0 = np.clip(q0, model.lower, model.upper)
    p0, quat0 = model.fk(q0)

    print("=" * 70)
    print("Cartesian trajectory validation (σ_warn=%.2f)" % SIGMA_WARN)
    print("=" * 70)

    results = []
    # 1. straight line forward 10 cm
    results.append(_audit(
        model, "line +x 10cm",
        [Waypoint(p0, quat0), Waypoint(p0 + np.array([0.10, 0, 0]), quat0)], q0))
    # 2. horizontal circle radius 6 cm around start
    results.append(_audit(
        model, "circle r=6cm",
        _circle_waypoints(p0 + np.array([0.05, 0, 0]), 0.06, quat0), q0))
    # 3. in-place yaw scan ±30°
    results.append(_audit(
        model, "yaw scan ±30°",
        _yaw_scan_waypoints(p0, quat0), q0))

    print()
    allok = all(results)
    print(f"OVERALL: {'PASS — all paths safe' if allok else 'REVIEW — some path near singularity/limits'}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
