#!/usr/bin/env python3
# Copyright 2026 Enactic, Inc.
# SPDX-License-Identifier: Apache-2.0
"""P0 validation: map the kinematic singularity structure of the arm.

Samples a 2-D slice of the 7-D joint space — elbow (joint4) × shoulder-yaw
(joint1) — with the other joints at their midpoint, and reports sigma_min /
manipulability / condition number. Writes a CSV (for plotting) and an ASCII
heatmap, and confirms the home (midpoint) and zero poses are singular.

Run::
    python src/openarm_pinocchio_nsp/validation/v_singularity_map.py
"""

from __future__ import annotations

import csv
import sys

import numpy as np

from openarm_pinocchio_nsp.kinematics import PinocchioModel, SIGMA_WARN
from openarm_pinocchio_nsp.urdf_path import DEFAULT_URDF

CSV_OUT = "/ros2_ws/openarm_nsp_ws/singularity_map_j1j4.csv"


def main() -> int:
    m = PinocchioModel(DEFAULT_URDF, "right")
    mid = (m.lower + m.upper) / 2.0
    # Base pose = a well-conditioned working posture (σ_min ~0.1); sweeping
    # joint1 (shoulder yaw) x joint4 (elbow) around it reveals real safe/unsafe
    # regions (sweeping around the singular midpoint makes everything singular).
    base = np.deg2rad([0.0, -30.0, 0.0, 90.0, 0.0, 45.0, 0.0])

    # 2-D slice: joint1 (shoulder yaw) x joint4 (elbow); rest at working posture.
    j1_grid = np.linspace(m.lower[0], m.upper[0], 25)
    j4_grid = np.linspace(m.lower[3], m.upper[3], 17)

    rows = []
    safe = 0
    total = 0
    for j4 in j4_grid:
        line = []
        for j1 in j1_grid:
            q = base.copy()
            q[0] = j1
            q[3] = j4
            sv = m.singular_values(q)
            smin = float(sv[-1])
            w = m.manipulability(q)
            rows.append((j1, j4, smin, w, float(sv[0] / smin) if smin > 0 else 1e12))
            line.append(smin)
            total += 1
            if smin >= SIGMA_WARN:
                safe += 1
        # ASCII heatmap row (joint4 = row). # = singular, . = safe
        bar = "".join("#" if s < SIGMA_WARN else ("+" if s < 0.08 else " ") for s in line)
        print(f"j4={np.degrees(j4):6.1f}° |{bar}|")

    # write CSV
    with open(CSV_OUT, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["joint1_rad", "joint4_rad", "sigma_min", "manipulability", "cond"])
        wr.writerows(rows)

    print()
    print(f"safe region (σ_min≥{SIGMA_WARN}): {safe}/{total} = {safe/total:.0%} of slice")
    print(f"CSV written: {CSV_OUT}")
    print()

    # verify known singular poses
    checks = {
        "home (midpoint)": mid,
        "zero": np.zeros(7),
        "hands_up (j4=2)": np.array([0, 0, 0, 2.0, 0, 0, 0.0]),
    }
    print("known-pose check:")
    all_ok = True
    for name, q in checks.items():
        smin = float(m.singular_values(q)[-1])
        is_sing = smin < SIGMA_WARN
        flag = "SINGULAR" if is_sing else "ok"
        if name.startswith("home") or name == "zero":
            all_ok &= is_sing  # these MUST be singular
        else:
            all_ok &= not is_sing  # hands_up must be safe
        print(f"  {name:18s} σ_min={smin:.4f}  {flag}")

    print()
    print("PASS" if all_ok else "REVIEW", "— home/zero singular, hands_up safe")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
