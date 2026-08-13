#!/usr/bin/env python3
# Copyright 2026 Enactic, Inc.
# Offline FK→IK→FK validation for openarm_pinocchio_ik.
# This script validates the kinematics implementation without modifying source code.

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin

# Import the actual kinematics module from source
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from openarm_pinocchio_ik.kinematics import PinocchioModel


def quat_angle_err(q1_xyzw, q2_xyzw) -> float:
    """Calculate orientation error between two quaternions (xyzw)."""
    R1 = pin.Quaternion(np.asarray(q1_xyzw, float)).matrix()
    R2 = pin.Quaternion(np.asarray(q2_xyzw, float)).matrix()
    R = R1.T @ R2
    # Handle quaternion double-cover (q and -q represent same rotation)
    trace_val = np.trace(R)
    angle = math.acos(max(-1.0, min(1.0, (trace_val - 1.0) / 2.0)))
    return angle


def validate_fk_ik_fk(
    urdf_path: str,
    side: str,
    q_source: np.ndarray,
    pos_tol_mm: float = 1.0,
    ori_tol_deg: float = 0.1,
    verbose: bool = True,
) -> int:
    """
    Perform FK→IK→FK validation.

    Returns 0 on success, non-zero on failure.
    """
    if verbose:
        print("=" * 70)
        print("FK→IK→FK VALIDATION")
        print("=" * 70)
        print(f"URDF path: {urdf_path}")
        print(f"Side: {side}")
        print()

    # Initialize model
    model = PinocchioModel(urdf_path, side)

    # Validate input
    if len(q_source) != 7:
        print(f"ERROR: Expected 7 joint angles, got {len(q_source)}")
        return 1

    if not np.all(np.isfinite(q_source)):
        print("ERROR: q_source contains NaN or Inf")
        return 1

    # Check joint limits
    in_limits = np.all(q_source >= model.lower) and np.all(q_source <= model.upper)
    if verbose:
        print(f"q_source (rad): {np.round(q_source, 4)}")
        print(f"Joint limits: {'SATISFIED' if in_limits else 'VIOLATED'}")
        if not in_limits:
            print(f"  lower: {np.round(model.lower, 3)}")
            print(f"  upper: {np.round(model.upper, 3)}")

    # First FK
    pos1, quat1 = model.fk(q_source)
    if verbose:
        print(f"\nFirst FK:")
        print(f"  Position (m): [{pos1[0]:.4f}, {pos1[1]:.4f}, {pos1[2]:.4f}]")
        print(f"  Orientation (xyzw): [{quat1[0]:.4f}, {quat1[1]:.4f}, {quat1[2]:.4f}, {quat1[3]:.4f}]")

    # IK with safe initial guess
    q_init = np.clip(q_source, model.lower, model.upper)
    q_solution = model.ik(pos1, quat1, q_init=q_init, max_iters=50, tol=1e-4, damping=1e-2)

    if verbose:
        if q_solution is None:
            print("\nIK Result: NOT CONVERGED")
            return 1
        else:
            print("\nIK Result: CONVERGED")

    # Validate solution
    if q_solution is None:
        return 1

    if not np.all(np.isfinite(q_solution)):
        print("ERROR: q_solution contains NaN or Inf")
        return 1

    # Check solution limits
    sol_in_limits = np.all(q_solution >= model.lower - 1e-6) and np.all(
        q_solution <= model.upper + 1e-6
    )
    if verbose:
        print(f"q_solution (rad): {np.round(q_solution, 4)}")
        print(f"Joint limits check: {'PASS' if sol_in_limits else 'FAIL'}")

    # Second FK
    pos2, quat2 = model.fk(q_solution)
    if verbose:
        print(f"\nSecond FK:")
        print(f"  Position (m): [{pos2[0]:.4f}, {pos2[1]:.4f}, {pos2[2]:.4f}]")
        print(f"  Orientation (xyzw): [{quat2[0]:.4f}, {quat2[1]:.4f}, {quat2[2]:.4f}, {quat2[3]:.4f}]")

    # Calculate errors
    pos_error_m = np.linalg.norm(pos1 - pos2)
    pos_error_mm = pos_error_m * 1000
    ori_error_rad = quat_angle_err(quat1, quat2)
    ori_error_deg = math.degrees(ori_error_rad)

    if verbose:
        print(f"\nErrors:")
        print(f"  Position error: {pos_error_mm:.4f} mm (tolerance: {pos_tol_mm} mm)")
        print(f"  Orientation error: {ori_error_deg:.4f} deg (tolerance: {ori_tol_deg} deg)")

    # Check against tolerances
    pos_pass = pos_error_mm <= pos_tol_mm
    ori_pass = ori_error_deg <= ori_tol_deg
    overall_pass = pos_pass and ori_pass and sol_in_limits

    if verbose:
        print(f"\nResult: {'PASS' if overall_pass else 'FAIL'}")
        print("=" * 70)

    if not overall_pass:
        if not pos_pass:
            print(f"FAIL: Position error {pos_error_mm:.4f} mm > {pos_tol_mm} mm")
        if not ori_pass:
            print(f"FAIL: Orientation error {ori_error_deg:.4f} deg > {ori_tol_deg} deg")
        if not sol_in_limits:
            print("FAIL: Joint limits violated")

    return 0 if overall_pass else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline FK→IK→FK validation for openarm_pinocchio_ik"
    )
    parser.add_argument(
        "--side",
        default="right",
        choices=["left", "right"],
        help="Arm side (default: right)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--joints",
        help="7 joint angles in radians, comma-separated",
    )
    group.add_argument(
        "--deg",
        help="7 joint angles in degrees, comma-separated",
    )
    parser.add_argument(
        "--urdf",
        default="/ros2_ws/openarm_ros2/openarm_description/assets/robot/openarm_v1.0/urdf/example/v1.urdf",
        help="Path to URDF file",
    )
    parser.add_argument(
        "--position-tol-mm",
        type=float,
        default=1.0,
        help="Position tolerance in mm (default: 1.0)",
    )
    parser.add_argument(
        "--orientation-tol-deg",
        type=float,
        default=0.1,
        help="Orientation tolerance in degrees (default: 0.1)",
    )
    args = parser.parse_args()

    # Check URDF exists
    if not Path(args.urdf).exists():
        print(f"ERROR: URDF file not found: {args.urdf}")
        return 1

    # Parse joint angles
    if args.deg:
        angles = [float(x) for x in args.deg.split(",")]
        if len(angles) != 7:
            print(f"ERROR: Expected 7 angles, got {len(angles)}")
            return 1
        q = np.deg2rad(angles)
    elif args.joints:
        q = np.array([float(x) for x in args.joints.split(",")])
        if len(q) != 7:
            print(f"ERROR: Expected 7 angles, got {len(q)}")
            return 1
    else:
        q = np.zeros(7)

    return validate_fk_ik_fk(
        urdf_path=args.urdf,
        side=args.side,
        q_source=q,
        pos_tol_mm=args.position_tol_mm,
        ori_tol_deg=args.orientation_tol_deg,
    )


if __name__ == "__main__":
    sys.exit(main())
