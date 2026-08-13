#!/usr/bin/env python3
# Copyright 2026 Enactic, Inc.
# Random FK→IK→FK Batch Validation for openarm_pinocchio_ik.
# This tool validates IK convergence rate, precision, and failure causes
# without modifying source code or connecting to hardware.

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pinocchio as pin

# Import the actual kinematics module from source
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from openarm_pinocchio_ik.kinematics import PinocchioModel


# Result status constants
STATUS_SUCCESS = "SUCCESS"
STATUS_POSE_PASS = "SOLVER_REPORTED_FAILURE_BUT_POSE_PASS"
STATUS_POSE_ERROR = "POSE_ERROR_TOO_LARGE"
STATUS_LIMIT_VIOLATION = "JOINT_LIMIT_VIOLATION"
STATUS_NONFINITE = "NONFINITE_SOLUTION"
STATUS_IK_EXCEPTION = "IK_EXCEPTION"
STATUS_FK_EXCEPTION = "FK_EXCEPTION"
STATUS_MAX_ITERS = "MAX_ITERATIONS"
STATUS_SINGULAR = "NEAR_SINGULAR"


class BenchmarkConfig:
    """Configuration for the benchmark."""

    def __init__(self, args):
        self.samples = args.samples
        self.seed = args.seed
        self.side = args.side
        self.limit_margin_ratio = args.limit_margin_ratio
        self.init_mode = args.init_mode
        self.multi_starts = args.multi_starts
        self.position_tol_mm = args.position_tol_mm
        self.orientation_tol_deg = args.orientation_tol_deg
        self.max_iters = args.max_iters
        self.urdf = args.urdf
        self.output_dir = args.output_dir
        self.min_init_distance_rad = args.min_init_distance_rad
        self.init_noise_std_rad = args.init_noise_std_rad
        self.singular_value_threshold = args.singular_value_threshold
        self.condition_number_threshold = args.condition_number_threshold


class KinematicsMetrics:
    """Metrics calculated from Jacobian."""

    @staticmethod
    def compute_jacobian_metrics(model: PinocchioModel, q7: np.ndarray) -> dict:
        """Compute Jacobian-based metrics for singularity analysis."""
        q = model._full_q(q7)
        pin.forwardKinematics(model.model, model.data, q)
        pin.updateFramePlacement(model.model, model.data, model.ee_fid)

        J = pin.computeFrameJacobian(
            model.model,
            model.data,
            q,
            model.ee_fid,
            pin.ReferenceFrame.LOCAL,
        )
        J7 = J[:, model.q_idx]  # 6 x 7

        # Singular value decomposition
        try:
            singular_values = np.linalg.svd(J7, compute_uv=False)
            min_singular_value = float(np.min(singular_values))
            max_singular_value = float(np.max(singular_values))

            # Handle zero singular values
            if min_singular_value < 1e-10:
                condition_number = float("inf")
            else:
                condition_number = max_singular_value / min_singular_value

            # Manipulability measure (Yoshikawa)
            manipulability = float(np.sqrt(np.linalg.det(J7 @ J7.T)))

            return {
                "min_singular_value": min_singular_value,
                "max_singular_value": max_singular_value,
                "condition_number": condition_number,
                "manipulability": manipulability,
                "singular_values": singular_values.tolist(),
            }
        except np.linalg.LinAlgError:
            return {
                "min_singular_value": 0.0,
                "max_singular_value": float("inf"),
                "condition_number": float("inf"),
                "manipulability": 0.0,
                "singular_values": [],
            }


def quat_angle_err(q1_xyzw, q2_xyzw) -> float:
    """Calculate orientation error between two quaternions (xyzw)."""
    R1 = pin.Quaternion(np.asarray(q1_xyzw, float)).matrix()
    R2 = pin.Quaternion(np.asarray(q2_xyzw, float)).matrix()
    R = R1.T @ R2
    trace_val = np.trace(R)
    angle = math.acos(max(-1.0, min(1.0, (trace_val - 1.0) / 2.0)))
    return angle


class RandomBenchmark:
    """Random FK→IK→FK batch validation."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.model = None
        self.rng = None
        self.results = []
        self.stats = {
            "total_samples": 0,
            "valid_targets": 0,
            "first_try_success": 0,
            "multi_start_recovery": 0,
            "final_success": 0,
            "statuses": {},
            "position_errors_mm": [],
            "orientation_errors_deg": [],
            "solve_times_ms": [],
            "joint_distances_rad": [],
            "iterations_unavailable": True,
            "near_singular_count": 0,
        }

    def initialize(self):
        """Initialize model and random state."""
        print(f"Loading URDF: {self.config.urdf}")
        self.model = PinocchioModel(self.config.urdf, self.config.side)
        self.rng = np.random.default_rng(self.config.seed)

        print(f"\nJoint limits (rad):")
        for i, (lower, upper) in enumerate(zip(self.model.lower, self.model.upper)):
            print(f"  Joint {i+1}: [{lower:.4f}, {upper:.4f}]")

    def get_sampling_range(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute safe sampling range with margin."""
        margin = self.config.limit_margin_ratio * (self.model.upper - self.model.lower)
        lower_safe = self.model.lower + margin
        upper_safe = self.model.upper - margin
        return lower_safe, upper_safe

    def sample_joint_angles(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Sample joint angles within safe bounds."""
        q = self.rng.uniform(lower, upper)
        # Validate
        if not np.all(np.isfinite(q)) or len(q) != 7:
            raise ValueError("Invalid joint sample")
        return q

    def generate_initial_guess(self, q_target: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Generate initial guess based on configured mode."""
        mode = self.config.init_mode

        if mode == "zero":
            q_init = np.zeros(7)
            q_init = np.clip(q_init, lower, upper)
            # Check if clipped
            if np.any(np.zeros(7) < lower) or np.any(np.zeros(7) > upper):
                pass  # Zero was out of limits, now clipped

        elif mode == "perturb_target":
            noise = self.rng.normal(0, self.config.init_noise_std_rad, size=7)
            q_init = q_target + noise
            q_init = np.clip(q_init, lower, upper)

        elif mode == "independent_random":
            max_attempts = 100
            min_dist = self.config.min_init_distance_rad

            for _ in range(max_attempts):
                q_init = self.sample_joint_angles(lower, upper)
                dist = np.linalg.norm(q_init - q_target)
                if dist >= min_dist:
                    break
            # If we exhausted attempts, use the last sample anyway

        else:
            raise ValueError(f"Unknown init_mode: {mode}")

        return q_init

    def check_near_singular(self, metrics: dict) -> bool:
        """Check if configuration is near singular."""
        min_sv = metrics.get("min_singular_value", float("inf"))
        cond_num = metrics.get("condition_number", 0)

        near_singular = False
        if self.config.singular_value_threshold is not None:
            near_singular = near_singular or (min_sv < self.config.singular_value_threshold)
        if self.config.condition_number_threshold is not None:
            if cond_num == float("inf"):
                near_singular = True
            else:
                near_singular = near_singular or (cond_num > self.config.condition_number_threshold)

        return near_singular

    def check_near_limits(self, q: np.ndarray, margin_ratio: float = 0.1) -> bool:
        """Check if configuration is near joint limits."""
        range_span = self.model.upper - self.model.lower
        margin = margin_ratio * range_span

        near_lower = q < (self.model.lower + margin)
        near_upper = q > (self.model.upper - margin)

        return np.any(near_lower | near_upper)

    def run_single_sample(
        self,
        sample_id: int,
        q_target: np.ndarray,
        q_init: np.ndarray,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        target_metrics: dict,
    ) -> dict:
        """Run FK→IK→FK for a single sample."""
        result = {
            "sample_id": sample_id,
            "seed": self.config.seed,
            "side": self.config.side,
            "q_target": q_target.tolist(),
            "q_init": q_init.tolist(),
            "target_position": target_pos.tolist(),
            "target_quaternion": target_quat.tolist(),
            "target_min_singular_value": target_metrics["min_singular_value"],
            "target_condition_number": target_metrics["condition_number"],
            "target_manipulability": target_metrics["manipulability"],
            "final_status": None,  # Will be set later
        }

        # Try multi-start IK
        first_converged = False
        final_converged = False
        attempts_used = 0
        final_q_solution = None

        for attempt in range(self.config.multi_starts):
            attempts_used += 1

            if attempt == 0:
                q_try = q_init.copy()
            else:
                # New independent initial guess for subsequent attempts
                lower, upper = self.get_sampling_range()
                q_try = self.sample_joint_angles(lower, upper)

            try:
                import time
                start_time = time.perf_counter()

                q_solution = self.model.ik(
                    target_pos,
                    target_quat,
                    q_init=q_try,
                    max_iters=self.config.max_iters,
                    tol=1e-4,
                    damping=1e-2,
                )

                solve_time_ms = (time.perf_counter() - start_time) * 1000

            except Exception as e:
                result.update({
                    "final_status": STATUS_IK_EXCEPTION,
                    "exception_message": str(e),
                    "attempts_used": attempts_used,
                    "solve_time_ms": solve_time_ms if 'solve_time_ms' in locals() else 0,
                })
                return result

            # Check convergence
            if q_solution is not None:
                if attempt == 0:
                    first_converged = True
                final_converged = True
                final_q_solution = q_solution
                result["solve_time_ms"] = solve_time_ms
                break

        # Store convergence info
        result["solver_reported_converged"] = final_converged
        result["attempts_used"] = attempts_used
        result["iterations"] = -1  # Unavailable - not returned by IK function

        if not final_converged:
            result["final_status"] = STATUS_MAX_ITERS
            result["q_solution"] = [None] * 7
            return result

        result["q_solution"] = final_q_solution.tolist()

        # Validate solution
        if not np.all(np.isfinite(final_q_solution)):
            result["final_status"] = STATUS_NONFINITE
            return result

        # Check joint limits
        in_limits = np.all(
            (final_q_solution >= self.model.lower - 1e-6) &
            (final_q_solution <= self.model.upper + 1e-6)
        )
        result["joint_limits_pass"] = bool(in_limits)

        if not in_limits:
            result["final_status"] = STATUS_LIMIT_VIOLATION

        # Second FK
        try:
            result_pos, result_quat = self.model.fk(final_q_solution)
        except Exception as e:
            result["final_status"] = STATUS_FK_EXCEPTION
            result["exception_message"] = str(e)
            return result

        result["result_position"] = result_pos.tolist()
        result["result_quaternion"] = result_quat.tolist()

        # Calculate errors
        pos_error_m = np.linalg.norm(target_pos - result_pos)
        pos_error_mm = pos_error_m * 1000
        ori_error_rad = quat_angle_err(target_quat, result_quat)
        ori_error_deg = math.degrees(ori_error_rad)

        result["position_error_mm"] = float(pos_error_mm)
        result["orientation_error_deg"] = float(ori_error_deg)

        # Joint distance (for multi-solution analysis)
        joint_distance = np.linalg.norm(q_target - final_q_solution)
        result["joint_solution_distance_rad"] = float(joint_distance)

        # Check pose tolerance
        pos_pass = pos_error_mm <= self.config.position_tol_mm
        ori_pass = ori_error_deg <= self.config.orientation_tol_deg

        result["final_pose_pass"] = bool(pos_pass and ori_pass)

        # Determine final status
        if result["final_status"] is None:  # No status set yet
            if pos_pass and ori_pass and in_limits:
                result["final_status"] = STATUS_SUCCESS
            elif not in_limits:
                result["final_status"] = STATUS_LIMIT_VIOLATION
            elif not (pos_pass and ori_pass):
                result["final_status"] = STATUS_POSE_ERROR

        # Special case: solver reported failure but pose passes
        if not final_converged and (pos_pass and ori_pass and in_limits):
            result["final_status"] = STATUS_POSE_PASS

        # Compute solution metrics
        solution_metrics = KinematicsMetrics.compute_jacobian_metrics(self.model, final_q_solution)
        result["min_singular_value"] = solution_metrics["min_singular_value"]
        result["jacobian_condition_number"] = solution_metrics["condition_number"]
        result["manipulability"] = solution_metrics["manipulability"]
        result["near_singular"] = self.check_near_singular(solution_metrics)

        return result

    def run(self):
        """Run the full benchmark."""
        print("\n" + "=" * 70)
        print("RANDOM FK→IK→FK BATCH VALIDATION")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  Samples: {self.config.samples}")
        print(f"  Seed: {self.config.seed}")
        print(f"  Side: {self.config.side}")
        print(f"  Init mode: {self.config.init_mode}")
        print(f"  Multi-starts: {self.config.multi_starts}")
        print(f"  Position tolerance: {self.config.position_tol_mm} mm")
        print(f"  Orientation tolerance: {self.config.orientation_tol_deg} deg")
        print(f"  Max iterations: {self.config.max_iters}")
        print(f"  Limit margin ratio: {self.config.limit_margin_ratio}")
        print("=" * 70)

        self.initialize()

        lower, upper = self.get_sampling_range()

        for sample_id in range(1, self.config.samples + 1):
            print(f"\r[{sample_id}/{self.config.samples}]", end="", flush=True)

            try:
                # Generate target from valid joint angles
                q_target = self.sample_joint_angles(lower, upper)

                # FK to get target pose (guaranteed reachable)
                target_pos, target_quat = self.model.fk(q_target)

                # Validate FK result
                if not (np.all(np.isfinite(target_pos)) and np.all(np.isfinite(target_quat))):
                    continue

                # Compute target metrics
                target_metrics = KinematicsMetrics.compute_jacobian_metrics(self.model, q_target)

                # Generate initial guess
                q_init = self.generate_initial_guess(q_target, lower, upper)

                # Run sample
                result = self.run_single_sample(
                    sample_id,
                    q_target,
                    q_init,
                    target_pos,
                    target_quat,
                    target_metrics,
                )

                self.results.append(result)
                self.stats["total_samples"] += 1

                # Update statistics
                if result["solver_reported_converged"]:
                    if sample_id == 1 or result.get("first_try", True):
                        self.stats["first_try_success"] += 1
                    self.stats["final_success"] += 1
                else:
                    # Check if multi-start would have helped
                    if result.get("multi_start_helped", False):
                        self.stats["multi_start_recovery"] += 1

                status = result["final_status"]
                self.stats["statuses"][status] = self.stats["statuses"].get(status, 0) + 1

                # Collect error statistics for successful samples
                if status == STATUS_SUCCESS:
                    self.stats["position_errors_mm"].append(result["position_error_mm"])
                    self.stats["orientation_errors_deg"].append(result["orientation_error_deg"])
                    self.stats["solve_times_ms"].append(result.get("solve_time_ms", 0))
                    self.stats["joint_distances_rad"].append(result["joint_solution_distance_rad"])

                # Track near-singular samples
                if result.get("near_singular", False):
                    self.stats["near_singular_count"] += 1

                self.stats["valid_targets"] += 1

            except Exception as e:
                print(f"\nERROR at sample {sample_id}: {e}")
                continue

        print(f"\n\nCompleted: {self.stats['valid_targets']} valid samples")

    def compute_statistics(self) -> dict:
        """Compute summary statistics."""
        s = self.stats.copy()

        # Success rates
        if s["total_samples"] > 0:
            s["first_try_success_rate"] = s["first_try_success"] / s["total_samples"]
            s["final_success_rate"] = s["final_success"] / s["total_samples"]
        else:
            s["first_try_success_rate"] = 0.0
            s["final_success_rate"] = 0.0

        # Percentile function
        def percentile(data, p):
            if not data:
                return 0.0
            return float(np.percentile(data, p))

        # Position error statistics (mm)
        pos_err = s["position_errors_mm"]
        s["position_error"] = {
            "mean": float(np.mean(pos_err)) if pos_err else 0.0,
            "median": float(np.median(pos_err)) if pos_err else 0.0,
            "std": float(np.std(pos_err)) if pos_err else 0.0,
            "p90": percentile(pos_err, 90),
            "p95": percentile(pos_err, 95),
            "p99": percentile(pos_err, 99),
            "max": float(np.max(pos_err)) if pos_err else 0.0,
        }

        # Orientation error statistics (deg)
        ori_err = s["orientation_errors_deg"]
        s["orientation_error"] = {
            "mean": float(np.mean(ori_err)) if ori_err else 0.0,
            "median": float(np.median(ori_err)) if ori_err else 0.0,
            "std": float(np.std(ori_err)) if ori_err else 0.0,
            "p90": percentile(ori_err, 90),
            "p95": percentile(ori_err, 95),
            "p99": percentile(ori_err, 99),
            "max": float(np.max(ori_err)) if ori_err else 0.0,
        }

        # Solve time statistics (ms)
        solve_times = s["solve_times_ms"]
        s["solve_time_ms"] = {
            "mean": float(np.mean(solve_times)) if solve_times else 0.0,
            "median": float(np.median(solve_times)) if solve_times else 0.0,
            "p95": percentile(solve_times, 95),
            "max": float(np.max(solve_times)) if solve_times else 0.0,
        }

        # Status breakdown
        total = s["total_samples"]
        s["status_breakdown"] = {
            status: {
                "count": count,
                "percentage": count / total if total > 0 else 0.0,
            }
            for status, count in s["statuses"].items()
        }

        return s

    def save_results(self):
        """Save all results to files."""
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config.output_dir) / f"{self.config.side}_seed{self.config.seed}_{self.config.samples}samples_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving results to: {output_dir}")

        # 1. results.csv - all samples
        csv_path = output_dir / "results.csv"
        fieldnames = [
            "sample_id", "seed", "side",
            "q_target_1", "q_target_2", "q_target_3", "q_target_4", "q_target_5", "q_target_6", "q_target_7",
            "q_init_1", "q_init_2", "q_init_3", "q_init_4", "q_init_5", "q_init_6", "q_init_7",
            "q_solution_1", "q_solution_2", "q_solution_3", "q_solution_4", "q_solution_5", "q_solution_6", "q_solution_7",
            "target_position_x", "target_position_y", "target_position_z",
            "target_quaternion_x", "target_quaternion_y", "target_quaternion_z", "target_quaternion_w",
            "result_position_x", "result_position_y", "result_position_z",
            "result_quaternion_x", "result_quaternion_y", "result_quaternion_z", "result_quaternion_w",
            "solver_reported_converged", "final_pose_pass", "final_status",
            "attempts_used", "iterations", "solve_time_ms",
            "position_error_mm", "orientation_error_deg",
            "joint_solution_distance_rad", "joint_limits_pass",
            "min_singular_value", "jacobian_condition_number", "manipulability", "near_singular",
            "exception_message",
        ]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in self.results:
                row = {
                    "sample_id": r["sample_id"],
                    "seed": r["seed"],
                    "side": r["side"],
                    "target_position_x": r["target_position"][0],
                    "target_position_y": r["target_position"][1],
                    "target_position_z": r["target_position"][2],
                    "target_quaternion_x": r["target_quaternion"][0],
                    "target_quaternion_y": r["target_quaternion"][1],
                    "target_quaternion_z": r["target_quaternion"][2],
                    "target_quaternion_w": r["target_quaternion"][3],
                    "exception_message": r.get("exception_message", ""),
                }

                # Add q_target
                for i, v in enumerate(r["q_target"]):
                    row[f"q_target_{i+1}"] = v

                # Add q_init
                for i, v in enumerate(r["q_init"]):
                    row[f"q_init_{i+1}"] = v

                # Add q_solution
                q_sol = r.get("q_solution", [None] * 7)
                for i, v in enumerate(q_sol):
                    row[f"q_solution_{i+1}"] = v if v is not None else ""

                # Add result position
                if "result_position" in r:
                    row["result_position_x"] = r["result_position"][0]
                    row["result_position_y"] = r["result_position"][1]
                    row["result_position_z"] = r["result_position"][2]
                else:
                    row["result_position_x"] = ""
                    row["result_position_y"] = ""
                    row["result_position_z"] = ""

                # Add result quaternion
                if "result_quaternion" in r:
                    row["result_quaternion_x"] = r["result_quaternion"][0]
                    row["result_quaternion_y"] = r["result_quaternion"][1]
                    row["result_quaternion_z"] = r["result_quaternion"][2]
                    row["result_quaternion_w"] = r["result_quaternion"][3]
                else:
                    row["result_quaternion_x"] = ""
                    row["result_quaternion_y"] = ""
                    row["result_quaternion_z"] = ""
                    row["result_quaternion_w"] = ""

                row["solver_reported_converged"] = r.get("solver_reported_converged", False)
                row["final_pose_pass"] = r.get("final_pose_pass", False)
                row["final_status"] = r["final_status"]
                row["attempts_used"] = r.get("attempts_used", 0)
                row["iterations"] = r.get("iterations", -1)
                row["solve_time_ms"] = r.get("solve_time_ms", 0)
                row["position_error_mm"] = r.get("position_error_mm", "")
                row["orientation_error_deg"] = r.get("orientation_error_deg", "")
                row["joint_solution_distance_rad"] = r.get("joint_solution_distance_rad", "")
                row["joint_limits_pass"] = r.get("joint_limits_pass", "")
                row["min_singular_value"] = r.get("min_singular_value", ""),
                row["jacobian_condition_number"] = r.get("jacobian_condition_number", ""),
                row["manipulability"] = r.get("manipulability", ""),
                row["near_singular"] = r.get("near_singular", False),

                writer.writerow(row)

        # 2. summary.json
        summary = self.compute_statistics()
        summary["config"] = {
            "samples": self.config.samples,
            "seed": self.config.seed,
            "side": self.config.side,
            "init_mode": self.config.init_mode,
            "multi_starts": self.config.multi_starts,
            "position_tol_mm": self.config.position_tol_mm,
            "orientation_tol_deg": self.config.orientation_tol_deg,
            "max_iters": self.config.max_iters,
            "limit_margin_ratio": self.config.limit_margin_ratio,
            "urdf": str(self.config.urdf),
        }

        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # 3. summary.md
        self._write_markdown_report(output_dir, summary)

        # 4. failed_cases.csv
        self._write_failed_cases(output_dir)

        # 5. replay_commands.txt
        self._write_replay_commands(output_dir)

        # 6. Visualizations (if matplotlib available)
        self._generate_visualizations(output_dir)

        return output_dir

    def _write_markdown_report(self, output_dir: Path, summary: dict):
        """Write human-readable markdown report."""
        cfg = summary["config"]

        md = f"""# FK→IK→FK Random Validation Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Side | {cfg['side']} |
| Samples | {cfg['samples']} |
| Seed | {cfg['seed']} |
| Init mode | {cfg['init_mode']} |
| Multi-starts | {cfg['multi_starts']} |
| Position tolerance | {cfg['position_tol_mm']} mm |
| Orientation tolerance | {cfg['orientation_tol_deg']} deg |
| Max iterations | {cfg['max_iters']} |
| Limit margin ratio | {cfg['limit_margin_ratio']} |
| URDF | {cfg['urdf']} |

## Why Targets Are Guaranteed Reachable

All test targets are generated using the following process:

1. Sample a random joint configuration `q_target` within verified joint limits
2. Compute target pose `T_target = FK(q_target)`
3. Solve IK from an independent initial guess `q_init`
4. Verify `FK(q_solution) ≈ T_target`

Because `T_target = FK(q_target)`, the target pose is mathematically guaranteed to be reachable
in the current kinematic model. Failures cannot be attributed to "unreachable targets" and must
indicate issues with the IK solver, initialization, or numerical conditioning.

## Overall Results

| Metric | Value |
|--------|-------|
| Total samples | {summary['total_samples']} |
| Valid targets | {summary['valid_targets']} |
| First-try success | {summary['first_try_success']} ({summary.get('first_try_success_rate', 0)*100:.1f}%) |
| Multi-start recovery | {summary['multi_start_recovery']} |
| Final success | {summary['final_success']} ({summary.get('final_success_rate', 0)*100:.1f}%) |
| Near-singular samples | {summary['near_singular_count']} |

"""

        md += "## Failure Breakdown\n\n"
        md += "| Status | Count | Percentage |\n"
        md += "|--------|-------|------------|\n"

        for status, data in summary.get("status_breakdown", {}).items():
            md += f"| {status} | {data['count']} | {data['percentage']*100:.1f}% |\n"

        md += "\n"

        # Error statistics
        pos_err = summary.get("position_error", {})
        ori_err = summary.get("orientation_error", {})

        md += "## Position Error (Successful Samples, mm)\n\n"
        md += "| Statistic | Value |\n"
        md += "|-----------|-------|\n"
        md += f"| Mean | {pos_err.get('mean', 0):.4f} |\n"
        md += f"| Median | {pos_err.get('median', 0):.4f} |\n"
        md += f"| Std | {pos_err.get('std', 0):.4f} |\n"
        md += f"| P90 | {pos_err.get('p90', 0):.4f} |\n"
        md += f"| P95 | {pos_err.get('p95', 0):.4f} |\n"
        md += f"| P99 | {pos_err.get('p99', 0):.4f} |\n"
        md += f"| Max | {pos_err.get('max', 0):.4f} |\n\n"

        md += "## Orientation Error (Successful Samples, deg)\n\n"
        md += "| Statistic | Value |\n"
        md += "|-----------|-------|\n"
        md += f"| Mean | {ori_err.get('mean', 0):.4f} |\n"
        md += f"| Median | {ori_err.get('median', 0):.4f} |\n"
        md += f"| Std | {ori_err.get('std', 0):.4f} |\n"
        md += f"| P90 | {ori_err.get('p90', 0):.4f} |\n"
        md += f"| P95 | {ori_err.get('p95', 0):.4f} |\n"
        md += f"| P99 | {ori_err.get('p99', 0):.4f} |\n"
        md += f"| Max | {ori_err.get('max', 0):.4f} |\n\n"

        # Solve time
        solve_time = summary.get("solve_time_ms", {})
        md += "## Solve Time (Successful Samples, ms)\n\n"
        md += "| Statistic | Value |\n"
        md += "|-----------|-------|\n"
        md += f"| Mean | {solve_time.get('mean', 0):.2f} |\n"
        md += f"| Median | {solve_time.get('median', 0):.2f} |\n"
        md += f"| P95 | {solve_time.get('p95', 0):.2f} |\n"
        md += f"| Max | {solve_time.get('max', 0):.2f} |\n\n"

        md += """## Conclusions

*This is an automated benchmark report. Conclusions should be drawn by comparing
multiple runs with different seeds and configurations.*

### Notes

- Iteration counts are **unavailable** - the IK function does not return this information
- Multi-start attempts can recover some failures but may not be suitable for real-time control
- Near-singular configurations may require specialized handling

"""

        with open(output_dir / "summary.md", "w") as f:
            f.write(md)

    def _write_failed_cases(self, output_dir: Path):
        """Write CSV of only failed cases."""
        failed = [r for r in self.results if r["final_status"] != STATUS_SUCCESS]

        if not failed:
            with open(output_dir / "failed_cases.csv", "w") as f:
                f.write("# No failed cases\n")
            return

        fieldnames = [
            "sample_id", "final_status", "position_error_mm",
            "orientation_error_deg", "near_singular", "attempts_used",
        ] + [f"q_target_{i+1}" for i in range(7)] + [f"q_init_{i+1}" for i in range(7)]

        with open(output_dir / "failed_cases.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in failed:
                row = {
                    "sample_id": r["sample_id"],
                    "final_status": r["final_status"],
                    "position_error_mm": r.get("position_error_mm", ""),
                    "orientation_error_deg": r.get("orientation_error_deg", ""),
                    "near_singular": r.get("near_singular", False),
                    "attempts_used": r.get("attempts_used", 0),
                }
                for i, v in enumerate(r["q_target"]):
                    row[f"q_target_{i+1}"] = v
                for i, v in enumerate(r["q_init"]):
                    row[f"q_init_{i+1}"] = v
                writer.writerow(row)

    def _write_replay_commands(self, output_dir: Path):
        """Write replay commands for failed cases."""
        failed = [r for r in self.results if r["final_status"] != STATUS_SUCCESS]

        if not failed:
            with open(output_dir / "replay_commands.txt", "w") as f:
                f.write("# No failed cases to replay\n")
            return

        with open(output_dir / "replay_commands.txt", "w") as f:
            f.write("# Replay commands for failed cases\n")
            f.write("# Use with fk_ik_fk_validate.py for single-case debugging\n\n")

            for r in failed:
                sid = r["sample_id"]
                q_target = ",".join(f"{v:.6f}" for v in r["q_target"])
                q_init = ",".join(f"{v:.6f}" for v in r["q_init"])

                f.write(f"# Sample {sid}: {r['final_status']}\n")
                f.write(f"# q_target: {q_target}\n")
                f.write(f"# q_init: {q_init}\n")

                # Command to replay with FK-IK-FK validator
                f.write(f"./run_fk_ik_fk.sh --joints {q_target} --side {self.config.side} \\\n")
                f.write(f"    --position-tol-mm {self.config.position_tol_mm} --orientation-tol-deg {self.config.orientation_tol_deg}\n")
                f.write("\n")

    def _generate_visualizations(self, output_dir: Path):
        """Generate visualization PNGs if matplotlib is available."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            print("Note: matplotlib not available, skipping visualizations")
            return

        # Prepare data
        pos_errors = [r.get("position_error_mm", np.nan) for r in self.results if r.get("position_error_mm") is not None]
        ori_errors = [r.get("orientation_error_deg", np.nan) for r in self.results if r.get("orientation_error_deg") is not None]
        cond_nums = [r.get("jacobian_condition_number", np.nan) for r in self.results if r.get("jacobian_condition_number") is not None]
        min_svs = [r.get("min_singular_value", np.nan) for r in self.results if r.get("min_singular_value") is not None]
        attempts = [r.get("attempts_used", 0) for r in self.results]

        # Filter valid data
        def valid_data(data, threshold=1e6):
            return [x for x in data if np.isfinite(x) and x < threshold]

        pos_errors = valid_data(pos_errors)
        ori_errors = valid_data(ori_errors)
        cond_nums = valid_data(cond_nums)
        min_svs = valid_data(min_svs)

        if not pos_errors:
            return  # No valid data to plot

        # 1. Position error histogram
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(pos_errors, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Position Error (mm)')
        ax.set_ylabel('Frequency')
        ax.set_title('Position Error Distribution (Successful Samples)')
        ax.axvline(self.config.position_tol_mm, color='r', linestyle='--', label='Tolerance')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "position_error_histogram.png", dpi=150)
        plt.close()

        # 2. Orientation error histogram
        if ori_errors:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(ori_errors, bins=50, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Orientation Error (deg)')
            ax.set_ylabel('Frequency')
            ax.set_title('Orientation Error Distribution (Successful Samples)')
            ax.axvline(self.config.orientation_tol_deg, color='r', linestyle='--', label='Tolerance')
            ax.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "orientation_error_histogram.png", dpi=150)
            plt.close()

        # 3. Attempts distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(attempts, bins=range(1, max(attempts)+2), edgecolor='black', alpha=0.7)
        ax.set_xlabel('Attempts Used')
        ax.set_ylabel('Frequency')
        ax.set_title('Multi-Start Attempts Distribution')
        plt.tight_layout()
        plt.savefig(output_dir / "attempts_distribution.png", dpi=150)
        plt.close()

        # 4. Condition number vs success
        if cond_nums:
            successes = [1 if r.get("final_status") == STATUS_SUCCESS else 0 for r in self.results if r.get("jacobian_condition_number") is not None and np.isfinite(r.get("jacobian_condition_number", np.inf)) and r.get("jacobian_condition_number", np.inf) < 1e6]
            conds = [r.get("jacobian_condition_number", np.nan) for r in self.results if r.get("jacobian_condition_number") is not None and np.isfinite(r.get("jacobian_condition_number", np.inf)) and r.get("jacobian_condition_number", np.inf) < 1e6]

            if conds:
                fig, ax = plt.subplots(figsize=(8, 5))
                scatter = ax.scatter(conds, successes, alpha=0.5, c=['green' if s else 'red' for s in successes])
                ax.set_xlabel('Jacobian Condition Number')
                ax.set_ylabel('Success (1) / Failure (0)')
                ax.set_title('Condition Number vs Success')
                ax.set_ylim(-0.1, 1.1)
                plt.tight_layout()
                plt.savefig(output_dir / "condition_number_vs_success.png", dpi=150)
                plt.close()

        print(f"Visualizations saved to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Random FK→IK→FK batch validation for openarm_pinocchio_ik"
    )

    # Basic parameters
    parser.add_argument("--samples", type=int, default=100, help="Number of random samples (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--side", default="right", choices=["left", "right"], help="Arm side (default: right)")
    parser.add_argument("--urdf", default="/ros2_ws/openarm_ros2/openarm_description/assets/robot/openarm_v1.0/urdf/example/v1.urdf", help="Path to URDF file")

    # Sampling parameters
    parser.add_argument("--limit-margin-ratio", type=float, default=0.05, help="Joint limit margin ratio (default: 0.05)")

    # IK initialization
    parser.add_argument("--init-mode", default="independent_random", choices=["independent_random", "zero", "perturb_target"],
                        help="Initial guess generation mode (default: independent_random)")
    parser.add_argument("--min-init-distance-rad", type=float, default=0.5, help="Minimum distance between init and target for independent_random (default: 0.5)")
    parser.add_argument("--init-noise-std-rad", type=float, default=0.2, help="Noise std for perturb_target mode (default: 0.2)")

    # Multi-start
    parser.add_argument("--multi-starts", type=int, default=1, help="Number of independent initial guesses (default: 1)")

    # Tolerances
    parser.add_argument("--position-tol-mm", type=float, default=1.0, help="Position tolerance in mm (default: 1.0)")
    parser.add_argument("--orientation-tol-deg", type=float, default=0.1, help="Orientation tolerance in deg (default: 0.1)")

    # IK parameters
    parser.add_argument("--max-iters", type=int, default=50, help="Maximum IK iterations (default: 50)")

    # Singularity thresholds
    parser.add_argument("--singular-value-threshold", type=float, default=1e-3, help="Singular value threshold for near-singular detection (default: 1e-3)")
    parser.add_argument("--condition-number-threshold", type=float, default=100, help="Condition number threshold for near-singular detection (default: 100)")

    # Output
    parser.add_argument("--output-dir", default="/ros2_ws/openarm_pinocchio_ik/.offline_verify/random_benchmark_results", help="Output directory for results")

    return parser.parse_args()


def main():
    args = parse_args()

    # Check URDF exists
    if not Path(args.urdf).exists():
        print(f"ERROR: URDF file not found: {args.urdf}")
        return 1

    # Create benchmark
    config = BenchmarkConfig(args)
    benchmark = RandomBenchmark(config)

    # Run
    try:
        benchmark.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Save results
    output_dir = benchmark.save_results()

    # Print summary
    stats = benchmark.compute_statistics()
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total samples: {stats['total_samples']}")
    print(f"Success rate: {stats.get('final_success_rate', 0)*100:.1f}%")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
