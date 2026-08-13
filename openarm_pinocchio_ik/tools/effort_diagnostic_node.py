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

"""READ-ONLY diagnostic node for external force estimation from joint_states.

This node subscribes to /joint_states and performs diagnostic calculations
for external force estimation using Pinocchio dynamics model.

WARNING: This node does NOT publish ANY robot commands.
It is for diagnostic and visualization purposes only.

The relationship between effort and external force is uncertain:
- Hardware gravity compensation may be enabled at firmware level
- /joint_states.effort may be pre-compensation, post-compensation, or other
- This node computes multiple candidate estimates for comparison

Author: Diagnostic Tool
"""

from __future__ import annotations

import math
import time
import traceback
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pinocchio as pin
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header


# -----------------------------------------------------------------------------
# Enums and Data Classes
# -----------------------------------------------------------------------------

class DiagnosticState(Enum):
    """States of the diagnostic node."""
    INIT = "initializing"
    WAITING_FIRST_MSG = "waiting_for_first_joint_state"
    CALIBRATING = "calibrating_baseline"
    READY = "ready"
    STALE = "stale_data"
    ERROR = "error"


@dataclass
class ForceCandidates:
    """Candidate force estimates (3D translation only)."""
    # From raw effort
    f_raw: np.ndarray  # 3x1
    f_raw_filtered: np.ndarray  # 3x1

    # From effort - gravity_model
    f_minus_g: np.ndarray  # 3x1
    f_minus_g_filtered: np.ndarray  # 3x1

    # From raw - baseline (if calibrated)
    f_raw_zeroed: np.ndarray | None  # 3x1 or None
    f_raw_zeroed_filtered: np.ndarray | None  # 3x1 or None

    # From effort - g - baseline (if calibrated)
    f_minus_g_zeroed: np.ndarray | None  # 3x1 or None
    f_minus_g_zeroed_filtered: np.ndarray | None  # 3x1 or None

    # XY (2D) candidates
    f_xy_raw_zeroed: np.ndarray | None  # 2x1 or None
    f_xy_minus_g_zeroed: np.ndarray | None  # 2x1 or None


@dataclass
class BaselineData:
    """Baseline calibration data."""
    effort_baseline: np.ndarray | None  # 7x1 or None
    effort_minus_g_baseline: np.ndarray | None  # 7x1 or None
    samples_collected: int
    max_velocity_rad_s: float
    is_valid: bool


@dataclass
class JacobianData:
    """Jacobian and singularity metrics."""
    Jv: np.ndarray  # 3x7 translation Jacobian
    singular_values: np.ndarray  # 3 values
    sigma_min: float
    condition_number: float
    is_reliable: bool

    # XY (2D) data
    J_xy: np.ndarray  # 2x7 XY Jacobian (rows 0,1 of Jv)
    xy_singular_values: np.ndarray  # 2 values
    xy_sigma_min: float
    xy_condition_number: float
    xy_is_reliable: bool


@dataclass
class DiagnosticMetrics:
    """Current diagnostic metrics."""
    timestamp: float
    data_age_sec: float
    max_velocity_rad_s: float
    max_effort_nm: float
    tcp_position: np.ndarray  # 3x1
    baseline: BaselineData
    jacobian: JacobianData
    forces: ForceCandidates
    warnings: list[str]


# -----------------------------------------------------------------------------
# First-Order Low-Pass Filter
# -----------------------------------------------------------------------------

class FirstOrderLPF:
    """First-order low-pass filter for 2D or 3D vector."""

    def __init__(self, cutoff_hz: float, dim: int = 3, dt_nominal: float = 0.02):
        """Initialize filter.

        Args:
            cutoff_hz: Cutoff frequency in Hz. Use 0 to disable filtering.
            dim: Dimension of vector to filter (2 or 3).
            dt_nominal: Nominal time step for initialization.
        """
        self.cutoff_hz = cutoff_hz
        self.dim = dim
        self.alpha = 0.0
        self.x_filtered = np.zeros(dim)
        self.initialized = False
        self._compute_alpha(dt_nominal)

    def _compute_alpha(self, dt: float) -> float:
        """Compute filter coefficient alpha from dt and cutoff."""
        if self.cutoff_hz <= 0.0 or dt <= 0.0:
            return 1.0  # No filtering
        rc = 1.0 / (2.0 * math.pi * self.cutoff_hz)
        return dt / (rc + dt)

    def reset(self, initial_value: np.ndarray | None = None) -> np.ndarray:
        """Reset filter state and return the reset value.

        Args:
            initial_value: Value to reset to. If None, uses zeros.

        Returns:
            The reset filter state (copy of initial_value or zeros).
        """
        if initial_value is not None:
            self.x_filtered = initial_value.copy()
        else:
            self.x_filtered = np.zeros(self.dim)
        self.initialized = False
        return self.x_filtered.copy()

    def update(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Update filter with new sample.

        Args:
            x: New sample (2D or 3D).
            dt: Time since last update.

        Returns:
            Filtered value.
        """
        if not self.initialized:
            self.x_filtered = x.copy()
            self.initialized = True
            return self.x_filtered

        # Recompute alpha if dt changed significantly
        alpha = self._compute_alpha(dt)
        self.x_filtered = alpha * x + (1.0 - alpha) * self.x_filtered
        return self.x_filtered


# -----------------------------------------------------------------------------
# Main Diagnostic Node
# -----------------------------------------------------------------------------

class EffortDiagnosticNode(Node):
    """READ-ONLY diagnostic node for joint effort analysis.

    This node:
        - Subscribes to /joint_states
        - Computes Pinocchio gravity model
        - Computes translation Jacobian
        - Estimates multiple candidate external forces
        - Publishes NO robot commands

    Topics:
        Subscribes: /joint_states (sensor_msgs/JointState)
        Publishes: NONE
    """

    def __init__(self) -> None:
        super().__init__("effort_diagnostic")

        # -----------------------------------------------------------------
        # Declare Parameters
        # -----------------------------------------------------------------
        self.declare_parameter("side", "right")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("print_rate_hz", 2.0)
        self.declare_parameter("joint_state_timeout_sec", 0.5)
        self.declare_parameter("damping_lambda", 0.01)
        self.declare_parameter("filter_cutoff_hz", 2.0)
        self.declare_parameter("stationary_velocity_threshold_rad_s", 0.01)
        self.declare_parameter("calibration_samples", 100)
        self.declare_parameter("enable_baseline_calibration", True)
        self.declare_parameter("effort_warning_threshold_nm", 10.0)
        self.declare_parameter("force_warning_threshold_n", 50.0)
        self.declare_parameter("condition_number_warning", 100.0)
        self.declare_parameter("minimum_singular_value_warning", 0.01)
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("use_local_world_aligned", True)

        # Get parameters with validation
        params = self._get_and_validate_parameters()

        # Store parameters
        self.side = params["side"]
        self.joint_states_topic = params["joint_states_topic"]
        self.print_rate_hz = params["print_rate_hz"]
        self.joint_state_timeout = params["joint_state_timeout_sec"]
        self.damping_lambda = params["damping_lambda"]
        self.filter_cutoff = params["filter_cutoff_hz"]
        self.stationary_threshold = params["stationary_velocity_threshold_rad_s"]
        self.calib_samples = params["calibration_samples"]
        self.enable_calibration = params["enable_baseline_calibration"]
        self.effort_warn_thresh = params["effort_warning_threshold_nm"]
        self.force_warn_thresh = params["force_warning_threshold_n"]
        self.cond_warn_thresh = params["condition_number_warning"]
        self.sv_warn_thresh = params["minimum_singular_value_warning"]
        self.use_lwa = params["use_local_world_aligned"]

        # TCP frame name
        self.tcp_frame = f"openarm_{self.side}_hand_tcp"

        # Joint names for this arm
        self.joint_names = [f"openarm_{self.side}_joint{i}" for i in range(1, 8)]

        # -----------------------------------------------------------------
        # Initialize State
        # -----------------------------------------------------------------
        self.state = DiagnosticState.INIT
        self.last_valid_time = 0.0
        self.last_print_time = 0.0
        self.msg_count = 0
        self.warnings = {}  # dict: category -> (current_message, last_update_time)

        # Current joint data
        self.current_q = np.zeros(7)
        self.current_dq = np.zeros(7)
        self.current_effort = np.zeros(7)
        self.has_velocity = False
        self.has_effort = False
        self.all_joints_present = False

        # Real dt tracking for filtering
        self.last_filter_time = None  # Will be set on first valid callback

        # Import kinematics from parent package
        import sys
        sys.path.insert(0, "/ros2_ws/openarm_pinocchio_ik/src")
        from openarm_pinocchio_ik.kinematics import PinocchioModel

        # Initialize Pinocchio model
        urdf = params["urdf_path"]
        self.get_logger().info(f"Loading URDF from: {urdf}")
        self.model = PinocchioModel(urdf, self.side)

        # Verify TCP frame exists
        if self.model.ee_fid >= len(self.model.model.frames):
            self.get_logger().error(
                f"TCP frame '{self.tcp_frame}' not found in model"
            )
            raise RuntimeError(f"TCP frame '{self.tcp_frame}' not found")

        # Baseline calibration data
        self.baseline = BaselineData(
            effort_baseline=None,
            effort_minus_g_baseline=None,
            samples_collected=0,
            max_velocity_rad_s=0.0,
            is_valid=False
        )

        # Filters for each force candidate (3D and 2D)
        self.filter_f_raw = FirstOrderLPF(self.filter_cutoff, dim=3)
        self.filter_f_minus_g = FirstOrderLPF(self.filter_cutoff, dim=3)
        self.filter_f_raw_zeroed: FirstOrderLPF | None = None
        self.filter_f_minus_g_zeroed: FirstOrderLPF | None = None

        # -----------------------------------------------------------------
        # Create Subscriptions (NO PUBLISHERS)
        # -----------------------------------------------------------------
        # Use sensor data QoS for best effort reception
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.js_sub = self.create_subscription(
            JointState,
            self.joint_states_topic,
            self._on_joint_state,
            qos
        )

        # Diagnostic print timer
        self.create_timer(1.0 / self.print_rate_hz, self._print_diagnostics)

        # -----------------------------------------------------------------
        # Print Startup Banner
        # -----------------------------------------------------------------
        self._print_startup_banner()

        # Transition state
        self.state = DiagnosticState.WAITING_FIRST_MSG

    # ---------------------------------------------------------------------
    # Parameter Validation
    # ---------------------------------------------------------------------

    def _get_and_validate_parameters(self) -> dict:
        """Get and validate all parameters."""
        params = {}

        # side
        side = self.get_parameter("side").value
        if side not in ("left", "right"):
            self.get_logger().error(f"Invalid side '{side}', must be 'left' or 'right'")
            raise ValueError(f"Invalid side: {side}")
        params["side"] = side

        # joint_states_topic
        params["joint_states_topic"] = self.get_parameter("joint_states_topic").value

        # print_rate_hz
        pr = float(self.get_parameter("print_rate_hz").value)
        if pr <= 0.0 or pr > 100.0:
            self.get_logger().warn(f"Invalid print_rate_hz {pr}, using 2.0")
            pr = 2.0
        params["print_rate_hz"] = pr

        # joint_state_timeout_sec
        jt = float(self.get_parameter("joint_state_timeout_sec").value)
        if jt <= 0.0:
            self.get_logger().warn(f"Invalid timeout {jt}, using 0.5")
            jt = 0.5
        params["joint_state_timeout_sec"] = jt

        # damping_lambda
        params["damping_lambda"] = float(self.get_parameter("damping_lambda").value)

        # filter_cutoff_hz
        fc = float(self.get_parameter("filter_cutoff_hz").value)
        if fc < 0.0:
            self.get_logger().warn(f"Invalid cutoff {fc}, using 2.0")
            fc = 2.0
        params["filter_cutoff_hz"] = fc

        # stationary_velocity_threshold_rad_s
        svt = float(self.get_parameter("stationary_velocity_threshold_rad_s").value)
        if svt < 0.0:
            self.get_logger().warn(f"Invalid threshold {svt}, using 0.01")
            svt = 0.01
        params["stationary_velocity_threshold_rad_s"] = svt

        # calibration_samples
        cs = int(self.get_parameter("calibration_samples").value)
        if cs < 10:
            self.get_logger().warn(f"Invalid calibration samples {cs}, using 100")
            cs = 100
        params["calibration_samples"] = cs

        # enable_baseline_calibration
        params["enable_baseline_calibration"] = bool(
            self.get_parameter("enable_baseline_calibration").value
        )

        # effort_warning_threshold_nm
        params["effort_warning_threshold_nm"] = float(
            self.get_parameter("effort_warning_threshold_nm").value
        )

        # force_warning_threshold_n
        params["force_warning_threshold_n"] = float(
            self.get_parameter("force_warning_threshold_n").value
        )

        # condition_number_warning
        params["condition_number_warning"] = float(
            self.get_parameter("condition_number_warning").value
        )

        # minimum_singular_value_warning
        params["minimum_singular_value_warning"] = float(
            self.get_parameter("minimum_singular_value_warning").value
        )

        # urdf_path - use default from kinematics if empty
        urdf = self.get_parameter("urdf_path").value
        if not urdf:
            urdf = (
                "/ros2_ws/install/openarm_description/share/openarm_description/"
                "assets/robot/openarm_v1.0/urdf/example/v1.urdf"
            )
        params["urdf_path"] = urdf

        # use_local_world_aligned
        params["use_local_world_aligned"] = bool(
            self.get_parameter("use_local_world_aligned").value
        )

        return params

    # ---------------------------------------------------------------------
    # Startup Banner
    # ---------------------------------------------------------------------

    def _print_startup_banner(self) -> None:
        """Print comprehensive startup information."""
        self.get_logger().info("=" * 70)
        self.get_logger().info("EFFORT DIAGNOSTIC NODE - READ ONLY")
        self.get_logger().info("=" * 70)
        self.get_logger().info("")
        self.get_logger().info("This node publishes NO robot commands.")
        self.get_logger().info("It only reads /joint_states for diagnostic analysis.")
        self.get_logger().info("")
        self.get_logger().info(f"Side: {self.side}")
        self.get_logger().info(f"TCP Frame: {self.tcp_frame}")
        self.get_logger().info(f"Joint order: {self.joint_names}")
        self.get_logger().info(f"URDF: {self.model.model.name}")
        self.get_logger().info("")
        self.get_logger().info(f"Subscribing to: {self.joint_states_topic}")
        self.get_logger().info(f"Joint state timeout: {self.joint_state_timeout:.2f} sec")
        self.get_logger().info("")
        self.get_logger().info(f"Damping lambda: {self.damping_lambda}")
        self.get_logger().info(f"Filter cutoff: {self.filter_cutoff} Hz")
        self.get_logger().info(f"Stationary threshold: {self.stationary_threshold} rad/s")
        self.get_logger().info("")
        if self.enable_calibration:
            self.get_logger().info("Baseline calibration: ENABLED")
            self.get_logger().info(f"  Samples required: {self.calib_samples}")
            self.get_logger().info("")
            self.get_logger().info("!!! IMPORTANT !!!")
            self.get_logger().info("DO NOT touch the robot during calibration!")
            self.get_logger().info("Wait for 'Baseline calibration complete' message.")
        else:
            self.get_logger().info("Baseline calibration: DISABLED")
        self.get_logger().info("")
        self.get_logger().info("=" * 70)

    # ---------------------------------------------------------------------
    # Joint State Callback
    # ---------------------------------------------------------------------

    def _on_joint_state(self, msg: JointState) -> None:
        """Process incoming joint state message."""
        now = time.monotonic()

        # Check for array length consistency
        if len(msg.name) != len(msg.position):
            self._add_warning("joint_state", "name/position length mismatch")
            return
        if msg.velocity and len(msg.velocity) != len(msg.name):
            self._add_warning("joint_state", "name/velocity length mismatch")
            return
        if msg.effort and len(msg.effort) != len(msg.name):
            self._add_warning("joint_state", "name/effort length mismatch")
            return

        # Check for NaN/inf in position
        if any(not math.isfinite(v) for v in msg.position):
            self._add_warning("joint_state", "NaN/inf in position")
            return

        # Update presence flags
        self.has_velocity = bool(msg.velocity) and all(
            math.isfinite(v) for v in msg.velocity
        )
        self.has_effort = bool(msg.effort) and all(
            math.isfinite(v) for v in msg.effort
        )

        # Check if all required joints are present
        try:
            q = np.array([
                msg.position[msg.name.index(n)]
                for n in self.joint_names
            ], dtype=float)

            if self.has_velocity:
                dq = np.array([
                    msg.velocity[msg.name.index(n)]
                    for n in self.joint_names
                ], dtype=float)
            else:
                dq = np.zeros(7)

            if self.has_effort:
                effort = np.array([
                    msg.effort[msg.name.index(n)]
                    for n in self.joint_names
                ], dtype=float)
            else:
                effort = np.zeros(7)

            self.all_joints_present = True
        except ValueError as e:
            self._add_warning("joint_state", f"Missing joints: {e}")
            self.all_joints_present = False
            return

        # Update current state
        self.current_q = q
        self.current_dq = dq
        self.current_effort = effort
        self.last_valid_time = now
        self.msg_count += 1

        # Clear joint_state warning - message is now valid
        self._remove_warning("joint_state")

        # Update state if was waiting
        if self.state == DiagnosticState.WAITING_FIRST_MSG:
            self.state = DiagnosticState.CALIBRATING if self.enable_calibration else DiagnosticState.READY

        # Perform calibration if in calibrating state
        if self.state == DiagnosticState.CALIBRATING and self.enable_calibration:
            self._perform_calibration_step()

        # Recover from stale state
        if self.state == DiagnosticState.STALE:
            self.get_logger().info("Data received, recovering from stale state")
            self.state = DiagnosticState.READY if not self.enable_calibration or self.baseline.is_valid else DiagnosticState.CALIBRATING

    # ---------------------------------------------------------------------
    # Baseline Calibration
    # ---------------------------------------------------------------------

    def _perform_calibration_step(self) -> None:
        """Perform one step of baseline calibration."""
        # Check if we can calibrate (need velocity data)
        if not self.has_velocity:
            if self.baseline.samples_collected == 0:
                self.get_logger().warn(
                    "Cannot calibrate: no velocity data in /joint_states"
                )
            return

        # Check if stationary
        max_vel = np.max(np.abs(self.current_dq))
        if max_vel > self.stationary_threshold:
            # Not stationary, don't collect
            return

        # Compute gravity model
        g = self.model.gravity(self.current_q)

        # Accumulate baseline
        if self.baseline.effort_baseline is None:
            self.baseline.effort_baseline = np.zeros(7)
            self.baseline.effort_minus_g_baseline = np.zeros(7)

        # Simple moving average
        n = self.baseline.samples_collected
        self.baseline.effort_baseline = (
            n * self.baseline.effort_baseline + self.current_effort
        ) / (n + 1)
        self.baseline.effort_minus_g_baseline = (
            n * self.baseline.effort_minus_g_baseline + (self.current_effort - g)
        ) / (n + 1)

        self.baseline.samples_collected += 1
        self.baseline.max_velocity_rad_s = max(
            self.baseline.max_velocity_rad_s, max_vel
        )

        # Check if calibration complete
        if self.baseline.samples_collected >= self.calib_samples:
            self.baseline.is_valid = True
            self.state = DiagnosticState.READY
            self.get_logger().info("")
            self.get_logger().info("=" * 50)
            self.get_logger().info("Baseline calibration COMPLETE")
            self.get_logger().info("=" * 50)
            self.get_logger().info(f"Samples collected: {self.baseline.samples_collected}")
            self.get_logger().info(f"Max velocity during calibration: {self.baseline.max_velocity_rad_s:.4f} rad/s")
            self.get_logger().info("")
            self.get_logger().info("Baseline effort (N·m):")
            for i, name in enumerate(self.joint_names):
                self.get_logger().info(f"  {name}: {self.baseline.effort_baseline[i]:.4f}")
            self.get_logger().info("")
            self.get_logger().info("You may now slowly push the TCP to observe force estimates.")
            self.get_logger().info("=" * 50)

            # Initialize zeroed filters (3D)
            self.filter_f_raw_zeroed = FirstOrderLPF(self.filter_cutoff, dim=3)
            self.filter_f_minus_g_zeroed = FirstOrderLPF(self.filter_cutoff, dim=3)

    # ---------------------------------------------------------------------
    # Computation Methods
    # ---------------------------------------------------------------------

    def _compute_jacobian_data(self, q: np.ndarray) -> JacobianData:
        """Compute translation Jacobian and singularity metrics (3D and XY 2D)."""
        # Get full configuration
        full_q = self.model._full_q(q)

        # Compute frame Jacobian
        # Try LOCAL_WORLD_ALIGNED first, fall back to LOCAL if not supported
        try:
            if self.use_lwa:
                ref_frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            else:
                ref_frame = pin.ReferenceFrame.LOCAL
        except AttributeError:
            # Pinocchio version may not have LOCAL_WORLD_ALIGNED
            ref_frame = pin.ReferenceFrame.LOCAL

        J = pin.computeFrameJacobian(
            self.model.model,
            self.model.data,
            full_q,
            self.model.ee_fid,
            ref_frame
        )

        # Extract 7 joint columns
        J7 = J[:, self.model.q_idx]  # 6x7

        # Translation Jacobian is first 3 rows (for LOCAL and LOCAL_WORLD_ALIGNED)
        # For WORLD frame, it would be different, but we're using LWA or LOCAL
        Jv = J7[:3, :]  # 3x7

        # XY Jacobian: rows 0 and 1 (X and Y in world-aligned frame)
        J_xy = Jv[:2, :]  # 2x7

        # Compute 3D singular values
        sv = np.linalg.svd(Jv, compute_uv=False)

        sigma_min = np.min(sv)
        sigma_max = np.max(sv)

        if sigma_min > 1e-10:
            cond = sigma_max / sigma_min
        else:
            cond = float('inf')

        is_reliable = (
            sigma_min >= self.sv_warn_thresh and
            cond <= self.cond_warn_thresh
        )

        # Compute XY (2D) singular values
        sv_xy = np.linalg.svd(J_xy, compute_uv=False)

        sigma_min_xy = np.min(sv_xy)
        sigma_max_xy = np.max(sv_xy)

        if sigma_min_xy > 1e-10:
            cond_xy = sigma_max_xy / sigma_min_xy
        else:
            cond_xy = float('inf')

        xy_is_reliable = (
            sigma_min_xy >= self.sv_warn_thresh and
            cond_xy <= self.cond_warn_thresh
        )

        return JacobianData(
            Jv=Jv,
            singular_values=sv,
            sigma_min=sigma_min,
            condition_number=cond,
            is_reliable=is_reliable,
            J_xy=J_xy,
            xy_singular_values=sv_xy,
            xy_sigma_min=sigma_min_xy,
            xy_condition_number=cond_xy,
            xy_is_reliable=xy_is_reliable
        )

    def _estimate_force_from_torque(
        self,
        tau: np.ndarray,
        Jv: np.ndarray
    ) -> np.ndarray:
        """Estimate 3D force from 7 joint torques using damped least squares.

        Uses: F = solve(J @ J.T + lambda^2 * I, J @ tau)

        Args:
            tau: 7 joint torques
            Jv: 3x7 translation Jacobian

        Returns:
            3D force vector
        """
        # Compute J @ J.T + lambda^2 * I
        JJt_lambda = Jv @ Jv.T + (self.damping_lambda ** 2) * np.eye(3)

        # Compute J @ tau
        J_tau = Jv @ tau

        # Solve for force
        try:
            F = np.linalg.solve(JJt_lambda, J_tau)
        except np.linalg.LinAlgError:
            F = np.zeros(3)

        return F

    def _estimate_xy_force_from_torque(
        self,
        tau: np.ndarray,
        J_xy: np.ndarray
    ) -> np.ndarray:
        """Estimate 2D XY force from 7 joint torques using damped least squares.

        Uses: F_xy = solve(J_xy @ J_xy.T + lambda^2 * I_2, J_xy @ tau)

        Args:
            tau: 7 joint torques
            J_xy: 2x7 XY translation Jacobian

        Returns:
            2D force vector [X, Y]

        Raises:
            np.linalg.LinAlgError: If matrix is singular and solve fails.
            ValueError: If input dimensions are incorrect.
            FloatingPointError: If non-finite values encountered.
        """
        # Compute J @ J.T + lambda^2 * I (2x2 matrix)
        JJt_lambda = J_xy @ J_xy.T + (self.damping_lambda ** 2) * np.eye(2)

        # Compute J @ tau (2x1 vector)
        J_tau = J_xy @ tau

        # Solve for force - let exceptions propagate to caller
        F_xy = np.linalg.solve(JJt_lambda, J_tau)

        return F_xy

    def _compute_force_candidates(self) -> tuple[ForceCandidates, JacobianData]:
        """Compute all candidate force estimates (3D and XY 2D)."""
        # Get Jacobian data
        jac = self._compute_jacobian_data(self.current_q)

        # Compute gravity model
        g = self.model.gravity(self.current_q)

        # Candidate A: raw effort (3D)
        f_raw = self._estimate_force_from_torque(self.current_effort, jac.Jv)

        # Candidate B: effort - gravity_model (3D)
        tau_minus_g = self.current_effort - g
        f_minus_g = self._estimate_force_from_torque(tau_minus_g, jac.Jv)

        # Compute real dt for filtering (once per call, shared by all filters)
        now = time.monotonic()

        # Determine dt and reset decision for this cycle
        if self.last_filter_time is None:
            dt = None
            reset_filters = True
        else:
            dt = now - self.last_filter_time
            reset_filters = (
                not math.isfinite(dt)
                or dt <= 0.0
                or dt > 1.0
            )

        # Helper function to update a filter with pre-computed dt/reset decision
        def update_filter(filt, value):
            """Update filter with value, using pre-computed dt/reset decision.

            Args:
                filt: The filter to update.
                value: Current input value.

            Returns:
                Filtered output.
            """
            if reset_filters:
                return filt.reset(value)
            else:
                return filt.update(value, dt)

        # Update main 3D filters (all use same dt/reset decision)
        f_raw_filt = update_filter(self.filter_f_raw, f_raw)
        f_minus_g_filt = update_filter(self.filter_f_minus_g, f_minus_g)

        # Candidates with baseline (if available)
        f_raw_zeroed = None
        f_raw_zeroed_filt = None
        f_minus_g_zeroed = None
        f_minus_g_zeroed_filt = None

        # XY (2D) candidates
        f_xy_raw_zeroed = None
        f_xy_minus_g_zeroed = None

        if self.baseline.is_valid and self.baseline.effort_baseline is not None:
            # Candidate C: raw - baseline (3D)
            tau_raw_zeroed = self.current_effort - self.baseline.effort_baseline
            f_raw_zeroed = self._estimate_force_from_torque(tau_raw_zeroed, jac.Jv)
            if self.filter_f_raw_zeroed:
                f_raw_zeroed_filt = update_filter(self.filter_f_raw_zeroed, f_raw_zeroed)

            # Candidate D: effort - g - baseline_g (3D)
            if self.baseline.effort_minus_g_baseline is not None:
                tau_minus_g_zeroed = tau_minus_g - self.baseline.effort_minus_g_baseline
                f_minus_g_zeroed = self._estimate_force_from_torque(tau_minus_g_zeroed, jac.Jv)
                if self.filter_f_minus_g_zeroed:
                    f_minus_g_zeroed_filt = update_filter(self.filter_f_minus_g_zeroed, f_minus_g_zeroed)

            # XY (2D) candidates
            # XY Raw - baseline
            try:
                f_xy_raw_zeroed = self._estimate_xy_force_from_torque(tau_raw_zeroed, jac.J_xy)
                if not all(math.isfinite(v) for v in f_xy_raw_zeroed):
                    f_xy_raw_zeroed = None
                    self._add_warning("force_xy_raw", "XY Raw force: NaN/inf detected")
                else:
                    self._remove_warning("force_xy_raw")
            except (np.linalg.LinAlgError, ValueError, FloatingPointError) as e:
                f_xy_raw_zeroed = None
                self._add_warning("force_xy_raw", f"XY Raw computation failed: {type(e).__name__}")

            # XY Eff-g - baseline
            if self.baseline.effort_minus_g_baseline is not None:
                try:
                    f_xy_minus_g_zeroed = self._estimate_xy_force_from_torque(tau_minus_g_zeroed, jac.J_xy)
                    if not all(math.isfinite(v) for v in f_xy_minus_g_zeroed):
                        f_xy_minus_g_zeroed = None
                        self._add_warning("force_xy_minus_g", "XY Eff-g force: NaN/inf detected")
                    else:
                        self._remove_warning("force_xy_minus_g")
                except (np.linalg.LinAlgError, ValueError, FloatingPointError) as e:
                    f_xy_minus_g_zeroed = None
                    self._add_warning("force_xy_minus_g", f"XY Eff-g computation failed: {type(e).__name__}")

        # Update filter time AFTER all filters have been processed
        self.last_filter_time = now

        return ForceCandidates(
            f_raw=f_raw,
            f_raw_filtered=f_raw_filt,
            f_minus_g=f_minus_g,
            f_minus_g_filtered=f_minus_g_filt,
            f_raw_zeroed=f_raw_zeroed,
            f_raw_zeroed_filtered=f_raw_zeroed_filt,
            f_minus_g_zeroed=f_minus_g_zeroed,
            f_minus_g_zeroed_filtered=f_minus_g_zeroed_filt,
            f_xy_raw_zeroed=f_xy_raw_zeroed,
            f_xy_minus_g_zeroed=f_xy_minus_g_zeroed
        ), jac

    # ---------------------------------------------------------------------
    # Warning Management
    # ---------------------------------------------------------------------

    def _add_warning(self, category: str, message: str) -> None:
        """Add a warning (replaces previous warning of same category).

        Active warnings track current conditions, not historical occurrences.
        Same category warnings are replaced (not accumulated).
        """
        now = time.monotonic()
        if category not in self.warnings or self.warnings[category][0] != message:
            # New warning or updated message for this category
            self.warnings[category] = (message, now)
            self.get_logger().warn(f"[{category}] {message}")
        else:
            # Same message, just update time
            self.warnings[category] = (message, now)

    def _remove_warning(self, category: str) -> None:
        """Remove a warning by category when condition normalizes."""
        if category in self.warnings:
            del self.warnings[category]

    def _update_active_warnings(self) -> None:
        """Update active warnings based on current conditions.

        This is called periodically to refresh which warnings are currently active.
        """

    def _check_warnings(
        self,
        jac: JacobianData,
        forces: ForceCandidates,
        g: np.ndarray
    ) -> None:
        """Check for warning conditions and update active warnings.

        Active warnings represent current conditions, not historical occurrences.
        Warnings are removed when conditions return to normal.
        """
        # Check effort magnitude
        max_effort = np.max(np.abs(self.current_effort))
        if max_effort > self.effort_warn_thresh:
            self._add_warning(
                "effort",
                f"High effort detected: {max_effort:.2f} N·m"
            )
        else:
            self._remove_warning("effort")

        # Check force magnitude (3D raw and minus_g)
        for name, force in [
            ("force_raw", forces.f_raw),
            ("force_minus_g", forces.f_minus_g)
        ]:
            f_norm = np.linalg.norm(force)
            if f_norm > self.force_warn_thresh:
                self._add_warning(
                    name,
                    f"High force: {f_norm:.1f} N"
                )
            else:
                self._remove_warning(name)

        # Check 3D Jacobian condition
        if not jac.is_reliable:
            self._add_warning(
                "jacobian_3d",
                f"Poor 3D conditioning: cond={jac.condition_number:.1f}, "
                f"sigma_min={jac.sigma_min:.4f}"
            )
        else:
            self._remove_warning("jacobian_3d")

        # Check XY Jacobian condition
        if not jac.xy_is_reliable:
            self._add_warning(
                "jacobian_xy",
                f"Poor XY conditioning: cond={jac.xy_condition_number:.1f}, "
                f"sigma_min={jac.xy_sigma_min:.4f}"
            )
        else:
            self._remove_warning("jacobian_xy")

        # Check for stale data
        now = time.monotonic()
        age = now - self.last_valid_time
        if age > self.joint_state_timeout:
            if self.state != DiagnosticState.STALE:
                self.get_logger().warn(f"Joint state data stale: {age:.2f} sec")
                self.state = DiagnosticState.STALE
                # Reset filters on stale recovery
                self.filter_f_raw.reset()
                self.filter_f_minus_g.reset()
                if self.filter_f_raw_zeroed:
                    self.filter_f_raw_zeroed.reset()
                if self.filter_f_minus_g_zeroed:
                    self.filter_f_minus_g_zeroed.reset()
                # Reset filter time tracking
                self.last_filter_time = None
            self._add_warning("stale", f"Data stale: {age:.1f} sec")
        else:
            self._remove_warning("stale")

    # ---------------------------------------------------------------------
    # Diagnostic Printing
    # ---------------------------------------------------------------------

    def _print_diagnostics(self) -> None:
        """Print periodic diagnostic summary."""
        if self.state == DiagnosticState.WAITING_FIRST_MSG:
            self.get_logger().info("Waiting for first /joint_states message...")
            return

        if self.state == DiagnosticState.STALE:
            self.get_logger().warn("Data stale - no valid /joint_states received")
            return

        if not self.all_joints_present or not self.has_effort:
            self.get_logger().warn("Insufficient joint data (missing joints or no effort)")
            return

        # Compute candidates
        try:
            forces, jac = self._compute_force_candidates()
        except Exception as e:
            self.get_logger().error(
                f"Computation error ({type(e).__name__}): {e}\n"
                f"{traceback.format_exc()}"
            )
            return

        # Compute gravity for display
        g = self.model.gravity(self.current_q)

        # Compute TCP position
        tcp_pos, _ = self.model.fk(self.current_q)

        # Check warnings
        self._check_warnings(jac, forces, g)

        # Print summary
        now = time.monotonic()
        age = now - self.last_valid_time
        max_dq = np.max(np.abs(self.current_dq))
        max_eff = np.max(np.abs(self.current_effort))

        print()
        print("=" * 70)
        print(f"EFFORT DIAGNOSTIC - {self.state.value.upper()}")
        print("=" * 70)
        print(f"Data age: {age:.3f} sec | Messages: {self.msg_count}")
        print(f"Max |dq|: {max_dq:.4f} rad/s | Max effort: {max_eff:.2f} N·m")
        print()

        # Baseline status
        if self.enable_calibration:
            if self.baseline.is_valid:
                print(f"Baseline: VALID (samples={self.baseline.samples_collected})")
            else:
                n = self.baseline.samples_collected
                print(f"Baseline: CALIBRATING ({n}/{self.calib_samples} samples)")

        # TCP position
        print(f"TCP xyz (m): [{tcp_pos[0]:.4f}, {tcp_pos[1]:.4f}, {tcp_pos[2]:.4f}]")
        print()

        # Jacobian info
        print(f"Jacobian singular values: {jac.singular_values}")
        print(f"Condition number: {jac.condition_number:.2f} | ", end="")
        print(f"Reliable: {'YES' if jac.is_reliable else 'NO'}")
        print()

        # Effort vs gravity
        print("Joint Torques (N·m):")
        print(f"{'Joint':<20} {'Effort':>12} {'Gravity':>12} {'Eff-g':>12}")
        print("-" * 60)
        for i, name in enumerate(self.joint_names):
            print(f"{name:<20} {self.current_effort[i]:>12.4f} {g[i]:>12.4f} {self.current_effort[i]-g[i]:>12.4f}")
        print()

        # Force candidates
        print("Estimated TCP Forces (N):")
        print(f"{'Candidate':<30} {'X':>10} {'Y':>10} {'Z':>10} {'|F|':>10}")
        print("-" * 70)

        candidates = [
            ("Raw effort", forces.f_raw, forces.f_raw_filtered),
            ("Effort - gravity_model", forces.f_minus_g, forces.f_minus_g_filtered),
        ]

        if self.baseline.is_valid and forces.f_raw_zeroed is not None:
            candidates.extend([
                ("Raw - baseline", forces.f_raw_zeroed, forces.f_raw_zeroed_filtered),
                ("Eff-g - baseline", forces.f_minus_g_zeroed, forces.f_minus_g_zeroed_filtered),
            ])

        for name, f_raw, f_filt in candidates:
            f_norm = np.linalg.norm(f_raw)
            f_filt_norm = np.linalg.norm(f_filt)
            print(f"{name:<30} {f_raw[0]:>10.2f} {f_raw[1]:>10.2f} {f_raw[2]:>10.2f} {f_norm:>10.2f}")
            if self.filter_cutoff > 0:
                print(f"  {'(filtered)':<28} {f_filt[0]:>10.2f} {f_filt[1]:>10.2f} {f_filt[2]:>10.2f} {f_filt_norm:>10.2f}")

        print("=" * 70)
        print()

        # XY Horizontal Diagnostic (2D)
        print("XY Horizontal Diagnostic")
        print("-" * 70)
        print(f"XY singular values: {jac.xy_singular_values}")
        print(f"XY condition number: {jac.xy_condition_number:.2f} | ", end="")
        print(f"XY Reliable: {'YES' if jac.xy_is_reliable else 'NO'}")
        print()

        # XY Force candidates (only when baseline is valid)
        if self.baseline.is_valid:
            print(f"{'Candidate':<30} {'X':>10} {'Y':>10} {'|Fxy|':>10}")
            print("-" * 70)

            # XY Raw - baseline
            if forces.f_xy_raw_zeroed is not None:
                fxy = forces.f_xy_raw_zeroed
                fxy_norm = np.linalg.norm(fxy)
                print(f"{'XY Raw - baseline':<30} {fxy[0]:>10.2f} {fxy[1]:>10.2f} {fxy_norm:>10.2f}")
            else:
                print(f"{'XY Raw - baseline':<30} INVALID (NaN/inf)")

            # XY Eff-g - baseline
            if forces.f_xy_minus_g_zeroed is not None:
                fxy = forces.f_xy_minus_g_zeroed
                fxy_norm = np.linalg.norm(fxy)
                print(f"{'XY Eff-g - baseline':<30} {fxy[0]:>10.2f} {fxy[1]:>10.2f} {fxy_norm:>10.2f}")
            else:
                print(f"{'XY Eff-g - baseline':<30} INVALID (NaN/inf)")
        else:
            print("XY Force Candidates: CALIBRATING (baseline not yet valid)")

        print("=" * 70)

        # Print active warnings periodically
        if self.warnings:
            print()
            print("Active warnings:")
            for category, (msg, _) in sorted(self.warnings.items()):
                print(f"  - [{category}] {msg}")

    # -------------------------------------------------------------------------
    # Shutdown
    # -------------------------------------------------------------------------

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'warnings'):
            self.warnings.clear()


def main() -> None:
    """Main entry point."""
    rclpy.init()

    node = EffortDiagnosticNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # Only call shutdown if context is still valid
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
