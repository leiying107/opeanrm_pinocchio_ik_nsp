#!/usr/bin/env python3
# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to the consent to to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Read-only shadow monitor for TCP collision detection during motion.

This node subscribes to /joint_states and JointTrajectoryControllerState,
computes collision metrics using Pinocchio, and optionally reports shadow
collision events WITHOUT sending any robot commands.

Estimated forces are uncalibrated - this node only provides relative metrics.

Author: Shadow Monitor
"""

from __future__ import annotations

import csv
import math
import sys
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pinocchio as pin
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from control_msgs.msg import JointTrajectoryControllerState
from sensor_msgs.msg import JointState

if TYPE_CHECKING:
    try:
        from pinocchio.visualize import MeshcatVisualizer
    except ImportError:
        MeshcatVisualizer = None


# -----------------------------------------------------------------------------
# Enums and Data Classes
# -----------------------------------------------------------------------------

class NodeState(Enum):
    """Node operation states."""
    INIT = "initializing"
    CALIBRATING = "calibrating_baseline"
    READY = "ready"
    STALE = "stale_data"
    ERROR = "error"


class ShadowState(Enum):
    """Shadow collision states."""
    MONITOR_ONLY = "monitor_only"
    ARMED = "armed"
    SHADOW_WARNING = "shadow_warning"
    SHADOW_TRIPPED = "shadow_tripped"


class MotionPhase(Enum):
    """Motion phase classification."""
    STATIONARY = "stationary"
    MOVING = "moving"
    UNKNOWN = "unknown"


class TripLatch(Enum):
    """Trip latch state."""
    NOT_TRIPPED = "not_tripped"
    WARNING_LATCH = "warning_latched"
    TRIPPED = "tripped"


@dataclass
class JointData:
    """Joint state data."""
    q: np.ndarray  # 7x1 positions (rad)
    dq: np.ndarray  # 7x1 velocities (rad/s)
    effort: np.ndarray  # 7x1 efforts (uncalibrated N·m)
    valid: bool = False
    timestamp: float = 0.0  # monotonic
    ros_time: float = 0.0


@dataclass
class ControllerStateData:
    """Controller state data."""
    reference_position: np.ndarray | None = None  # 7x1 desired positions
    feedback_position: np.ndarray | None = None  # 7x1 actual positions
    position_error: np.ndarray | None = None  # 7x1 tracking errors
    reference_velocity: np.ndarray | None = None  # 7x1 desired velocities
    feedback_velocity: np.ndarray | None = None  # 7x1 actual velocities
    velocity_error: np.ndarray | None = None  # 7x1 velocity errors
    valid: bool = False
    timestamp: float = 0.0
    ros_time: float = 0.0
    fields_available: list[str] = field(default_factory=list)


@dataclass
class ArmBaseline:
    """Calibrated baseline data."""
    effort_baseline: np.ndarray | None = None  # 7x1
    effort_minus_g_baseline: np.ndarray | None = None  # 7x1
    reference_q: np.ndarray | None = None  # 7x1
    reference_tcp_position: np.ndarray | None = None  # 3x1
    samples: int = 0
    start_time: float = 0.0
    std_effort: np.ndarray | None = None  # 7x1
    std_effort_minus_g: np.ndarray | None = None  # 7x1
    max_effort_delta: float = 0.0
    max_effort_minus_g_delta: float = 0.0
    sum_effort_sq: np.ndarray | None = None  # 7x1  # For std computation
    sum_effort_minus_g_sq: np.ndarray | None = None  # 7x1
    is_valid: bool = False


@dataclass
class XYJacobianData:
    """XY Jacobian data."""
    J_xy: np.ndarray  # 2x7
    singular_values: np.ndarray  # 2 values
    sigma_min: float
    condition_number: float
    is_reliable: bool


@dataclass
class CollisionMetrics:
    """Computed collision metrics."""
    tcp_position: np.ndarray  # 3x1
    gravity: np.ndarray  # 7x1
    tau_raw_zeroed: np.ndarray  # 7x1
    tau_minus_g_zeroed: np.ndarray  # 7x1
    fxy_raw_zeroed: np.ndarray  # 2x1
    fxy_minus_g_zeroed: np.ndarray  # 2x1
    external_raw: np.ndarray  # 2x1  # with sign correction
    external_minus_g: np.ndarray  # 2x1
    jacobian_data: XYJacobianData
    measured_tcp_delta: np.ndarray  # 3x1  # TCP delta from baseline


@dataclass
class OnlineStats:
    """Online statistics."""
    count: int = 0
    sum: float = 0.0
    sum_sq: float = 0.0
    maximum: float = -float('inf')
    minimum: float = float('inf')

    def update(self, value: float) -> None:
        """Update statistics with new value."""
        if not math.isfinite(value):
            return
        self.count += 1
        self.sum += value
        self.sum_sq += value * value
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    @property
    def mean(self) -> float:
        """Mean value."""
        if self.count == 0:
            return float('nan')
        return self.sum / self.count

    @property
    def rms(self) -> float:
        """RMS value."""
        if self.count == 0:
            return float('nan')
        n = self.count
        return math.sqrt(self.sum_sq / n)

    def reset(self) -> None:
        """Reset statistics."""
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.maximum = -float('inf')
        self.minimum = float('inf')


@dataclass
class ArmOnlineStats:
    """Per-arm online statistics by motion phase."""
    stationary: dict[str, OnlineStats] = field(default_factory=dict)
    moving: dict[str, OnlineStats] = field(default_factory=dict)
    all_samples: dict[str, OnlineStats] = field(default_factory=dict)

    def get_phase_stats(self, phase: MotionPhase) -> dict[str, OnlineStats]:
        """Get stats for a phase."""
        if phase == MotionPhase.STATIONARY:
            return self.stationary
        elif phase == MotionPhase.MOVING:
            return self.moving
        return self.all_samples

    def reset(self) -> None:
        """Reset all statistics."""
        for stats_dict in [self.stationary, self.moving, self.all_samples]:
            for stats in stats_dict.values():
                stats.reset()


@dataclass
class ArmShadowState:
    """Shadow state for one arm."""
    state: ShadowState = ShadowState.MONITOR_ONLY
    trip_latch: TripLatch = TripLatch.NOT_TRIPPED
    warning_start_time: float = 0.0
    trip_start_time: float = 0.0
    warning_count: int = 0
    trip_count: int = 0


@dataclass
class ArmState:
    """Complete state for one arm."""
    joint_data: JointData | None = None
    controller_data: ControllerStateData = field(default_factory=ControllerStateData)
    baseline: ArmBaseline = field(default_factory=ArmBaseline)
    metrics: CollisionMetrics | None = None
    shadow_state: ArmShadowState = field(default_factory=ArmShadowState)
    online_stats: ArmOnlineStats = field(default_factory=ArmOnlineStats)
    motion_phase: MotionPhase = MotionPhase.UNKNOWN
    node_state: NodeState = NodeState.INIT
    invalid_reason: str = ""


@dataclass
class CSVRow:
    """One CSV output row."""
    monotonic_time: float
    ros_time: float
    side: str
    node_state: str
    motion_phase: str
    joint_state_age: float
    controller_state_age: float
    q1: float
    q2: float
    q3: float
    q4: float
    q5: float
    q6: float
    q7: float
    dq1: float
    dq2: float
    dq3: float
    dq4: float
    dq5: float
    dq6: float
    dq7: float
    effort1: float
    effort2: float
    effort3: float
    effort4: float
    effort5: float
    effort6: float
    effort7: float
    gravity1: float
    gravity2: float
    gravity3: float
    gravity4: float
    gravity5: float
    gravity6: float
    gravity7: float
    tau_raw_zeroed1: float
    tau_raw_zeroed2: float
    tau_raw_zeroed3: float
    tau_raw_zeroed4: float
    tau_raw_zeroed5: float
    tau_raw_zeroed6: float
    tau_raw_zeroed7: float
    tau_minus_g_zeroed1: float
    tau_minus_g_zeroed2: float
    tau_minus_g_zeroed3: float
    tau_minus_g_zeroed4: float
    tau_minus_g_zeroed5: float
    tau_minus_g_zeroed6: float
    tau_minus_g_zeroed7: float
    max_abs_dq: float
    raw_residual_l2: float
    raw_residual_max_abs: float
    minus_g_residual_l2: float
    minus_g_residual_max_abs: float
    tcp_x: float
    tcp_y: float
    tcp_z: float
    measured_tcp_dx: float
    measured_tcp_dy: float
    measured_tcp_dz: float
    fxy_raw_x: float
    fxy_raw_y: float
    fxy_raw_norm: float
    fxy_minus_g_x: float
    fxy_minus_g_y: float
    fxy_minus_g_norm: float
    external_raw_x: float
    external_raw_y: float
    external_raw_norm: float
    external_minus_g_x: float
    external_minus_g_y: float
    external_minus_g_norm: float
    xy_sigma_min: float
    xy_condition: float
    xy_reliable: int
    ref_pos1: float
    ref_pos2: float
    ref_pos3: float
    ref_pos4: float
    ref_pos5: float
    ref_pos6: float
    ref_pos7: float
    fb_pos1: float
    fb_pos2: float
    fb_pos3: float
    fb_pos4: float
    fb_pos5: float
    fb_pos6: float
    fb_pos7: float
    pos_err1: float
    pos_err2: float
    pos_err3: float
    pos_err4: float
    pos_err5: float
    pos_err6: float
    pos_err7: float
    ref_vel1: float
    ref_vel2: float
    ref_vel3: float
    ref_vel4: float
    ref_vel5: float
    ref_vel6: float
    ref_vel7: float
    fb_vel1: float
    fb_vel2: float
    fb_vel3: float
    fb_vel4: float
    fb_vel5: float
    fb_vel6: float
    fb_vel7: float
    vel_err1: float
    vel_err2: float
    vel_err3: float
    vel_err4: float
    vel_err5: float
    vel_err6: float
    vel_err7: float
    tracking_pos_err_l2: float
    tracking_pos_err_max_abs: float
    tracking_vel_err_l2: float
    tracking_vel_err_max_abs: float
    controller_fields_available: str
    shadow_warning: int
    shadow_tripped: int
    invalid_reason: str


# -----------------------------------------------------------------------------
# Pure Math Functions
# -----------------------------------------------------------------------------

def estimate_xy_force_from_torque(
    tau: np.ndarray,
    J_xy: np.ndarray,
    damping_lambda: float = 0.01,
) -> np.ndarray:
    """Estimate 2D XY force from 7 joint torques using damped least squares.

    Args:
        tau: 7 joint torques
        J_xy: 2x7 XY Jacobian
        damping_lambda: Damping factor

    Returns:
        2D force vector [Fx, Fy]
    """
    # Compute J @ J.T + lambda^2 * I (2x2 matrix)
    JJt_lambda = J_xy @ J_xy.T + (damping_lambda ** 2) * np.eye(2)

    # Compute J @ tau (2x1 vector)
    J_tau = J_xy @ tau

    # Solve for force
    try:
        F_xy = np.linalg.solve(JJt_lambda, J_tau)
    except np.linalg.LinAlgError:
        F_xy = np.zeros(2)

    return F_xy


def compute_xy_jacobian_data(
    model: pin.Model,
    data: pin.Data,
    ee_fid: int,
    q_idx: list[int],
    full_q: np.ndarray,
    use_local_world_aligned: bool = True,
    condition_max: float = 50.0,
    sigma_min_threshold: float = 0.02,
) -> XYJacobianData:
    """Compute XY Jacobian and reliability metrics.

    Returns:
        XYJacobianData with J_xy, singular values, condition number, reliability
    """
    # Try LOCAL_WORLD_ALIGNED first, fall back to LOCAL
    try:
        if use_local_world_aligned:
            ref_frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        else:
            ref_frame = pin.ReferenceFrame.LOCAL
    except AttributeError:
        ref_frame = pin.ReferenceFrame.LOCAL

    J = pin.computeFrameJacobian(
        model, data, full_q, ee_fid, ref_frame
    )

    J7 = J[:, q_idx]  # 6x7
    Jv = J7[:3, :]  # 3x7 translation Jacobian
    J_xy = Jv[:2, :]  # 2x7 XY Jacobian (rows 0,1)

    # Compute 2D singular values
    sv = np.linalg.svd(J_xy, compute_uv=False)

    sigma_min = np.min(sv)
    sigma_max = np.max(sv)

    if sigma_min > 1e-10:
        cond = sigma_max / sigma_min
    else:
        cond = float('inf')

    is_reliable = (
        sigma_min >= sigma_min_threshold and
        cond <= condition_max
    )

    return XYJacobianData(
        J_xy=J_xy,
        singular_values=sv,
        sigma_min=sigma_min,
        condition_number=cond,
        is_reliable=is_reliable,
    )


def nan_to_str(x: float | None) -> str:
    """Convert value to string, NaN becomes empty."""
    if x is None or not math.isfinite(x):
        return ""
    return str(x)


# -----------------------------------------------------------------------------
# Main Node
# -----------------------------------------------------------------------------

class TCPCollisionMonitorShadowNode(Node):
    """Read-only shadow monitor for TCP collision detection.

    This node:
        - Subscribes to /joint_states and JTC state
        - Computes collision metrics
        - Optionally reports shadow collision events
        - NEVER sends robot commands
    """

    def __init__(self) -> None:
        super().__init__("tcp_collision_monitor_shadow")

        # Initialize cleanup-related attributes first (for safe partial initialization)
        self.csv_file: Optional[Path] = None
        self.csv_writer: Optional[csv.writer] = None
        self.csv_row_count = 0
        self.cleanup_done = False

        # Declare and get parameters
        self._declare_parameters()
        params = self._get_parameters()

        # Store key parameters
        self.side = params["side"]
        self.sides = ["left", "right"] if self.side == "both" else [self.side]
        self.joint_state_topic = params["joint_state_topic"]
        self.joint_state_timeout = params["joint_state_timeout_sec"]
        self.controller_state_timeout = params["controller_state_timeout_sec"]
        self.sample_rate = params["sample_rate_hz"]
        self.print_rate = params["print_rate_hz"]
        self.urdf_path = params["urdf_path"]
        self.stationary_threshold = params["stationary_velocity_threshold_rad_s"]

        # Calibration parameters
        self.calibration_duration = params["calibration_duration_sec"]
        self.calibration_min_samples = params["calibration_min_samples"]

        # Force estimation parameters
        self.force_damping = params["force_damping"]
        self.force_sign_x = params["force_sign_x"]
        self.force_sign_y = params["force_sign_y"]
        self.xy_condition_max = params["xy_condition_max"]
        self.xy_sigma_min_threshold = params["xy_sigma_min"]

        # Shadow trip parameters
        self.enable_shadow_trip = params["enable_shadow_trip"]
        self.trip_force_source = params["trip_force_source"]
        self.warning_force_norm = params["warning_force_norm"]
        self.trip_force_norm = params["trip_force_norm"]
        self.hard_trip_force_norm = params["hard_trip_force_norm"]
        self.trip_hold_time = params["trip_hold_time_sec"]
        self.clear_hold_time = params["clear_hold_time_sec"]

        # CSV parameters
        self.csv_path = params["csv_path"]
        self.flush_interval = params["flush_interval_sec"]

        # Initialize Pinocchio models for each side
        self.models: dict[str, any] = {}
        for side in self.sides:
            sys.path.insert(0, "/ros2_ws/openarm_pinocchio_ik/src")
            from openarm_pinocchio_ik.kinematics import PinocchioModel
            self.models[side] = PinocchioModel(self.urdf_path, side)

        # Initialize arm states
        self.arm_states: dict[str, ArmState] = {
            side: ArmState() for side in self.sides
        }

        # Initialize online stats keys
        for side in self.sides:
            state = self.arm_states[side]
            for metric in [
                "fxy_raw_norm", "fxy_minus_g_norm",
                "raw_residual_l2", "minus_g_residual_l2",
                "tracking_position_error_l2",
                "tracking_velocity_error_l2",
                "max_abs_dq",
                "joint_state_age", "controller_state_age",
            ]:
                state.online_stats.stationary[metric] = OnlineStats()
                state.online_stats.moving[metric] = OnlineStats()
                state.online_stats.all_samples[metric] = OnlineStats()

        # Track last times
        self.last_joint_state_time: dict[str, float] = {side: 0.0 for side in self.sides}
        self.last_controller_state_time: dict[str, float] = {side: 0.0 for side in self.sides}
        self.last_print_time = 0.0
        self.last_flush_time = 0.0

        if self.csv_path:
            self._init_csv()

        # Create subscriptions
        self._create_subscriptions()

        # Create timers
        sample_period = 1.0 / self.sample_rate
        print_period = 1.0 / self.print_rate
        self.create_timer(sample_period, self._sample_timer_callback)
        self.create_timer(print_period, self._print_timer_callback)

        # Flush timer
        if self.csv_path and self.flush_interval > 0:
            self.create_timer(self.flush_interval, self._flush_timer_callback)

        # Print startup banner
        self._print_startup_banner()

        # Set initial state
        for side in self.sides:
            self.arm_states[side].node_state = NodeState.CALIBRATING

    def _declare_parameters(self) -> None:
        """Declare all parameters."""
        # Basic
        self.declare_parameter("side", "both")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("left_controller_state_topic",
                             "/left_joint_trajectory_controller/state")
        self.declare_parameter("right_controller_state_topic",
                             "/right_joint_trajectory_controller/state")
        self.declare_parameter("urdf_path",
                             "/ros2_ws/openarm_ros2/openarm_description/assets/robot/openarm_v1.0/urdf/example/v1.urdf")
        self.declare_parameter("sample_rate_hz", 100.0)
        self.declare_parameter("print_rate_hz", 1.0)
        self.declare_parameter("joint_state_timeout_sec", 0.2)
        self.declare_parameter("controller_state_timeout_sec", 0.2)
        self.declare_parameter("stationary_velocity_threshold_rad_s", 0.03)

        # Baseline
        self.declare_parameter("calibration_duration_sec", 2.0)
        self.declare_parameter("calibration_min_samples", 100)

        # Force estimation
        self.declare_parameter("force_damping", 0.01)
        self.declare_parameter("force_sign_x", -1.0)
        self.declare_parameter("force_sign_y", -1.0)
        self.declare_parameter("xy_condition_max", 50.0)
        self.declare_parameter("xy_sigma_min", 0.02)

        # Shadow trip
        self.declare_parameter("enable_shadow_trip", False)
        self.declare_parameter("trip_force_source", "minus_g")
        self.declare_parameter("warning_force_norm", 0.0)
        self.declare_parameter("trip_force_norm", 0.0)
        self.declare_parameter("hard_trip_force_norm", 0.0)
        self.declare_parameter("trip_hold_time_sec", 0.03)
        self.declare_parameter("clear_hold_time_sec", 0.2)

        # Logging
        self.declare_parameter("csv_path", "")
        self.declare_parameter("flush_interval_sec", 1.0)

    def _get_parameters(self) -> dict:
        """Get and validate parameters."""
        params = {}

        # side
        side = self.get_parameter("side").value
        if side not in ("left", "right", "both"):
            self.get_logger().error(f"Invalid side '{side}'")
            raise ValueError(f"side must be 'left', 'right', or 'both'")
        params["side"] = side

        # Topics
        params["joint_state_topic"] = self.get_parameter("joint_state_topic").value
        params["left_controller_state_topic"] = self.get_parameter(
            "left_controller_state_topic").value
        params["right_controller_state_topic"] = self.get_parameter(
            "right_controller_state_topic").value

        # URDF and rates
        params["urdf_path"] = self.get_parameter("urdf_path").value
        params["sample_rate_hz"] = float(self.get_parameter("sample_rate_hz").value)
        params["print_rate_hz"] = float(self.get_parameter("print_rate_hz").value)
        params["joint_state_timeout_sec"] = float(
            self.get_parameter("joint_state_timeout_sec").value)
        params["controller_state_timeout_sec"] = float(
            self.get_parameter("controller_state_timeout_sec").value)
        params["stationary_velocity_threshold_rad_s"] = float(
            self.get_parameter("stationary_velocity_threshold_rad_s").value)

        # Calibration
        params["calibration_duration_sec"] = float(
            self.get_parameter("calibration_duration_sec").value)
        params["calibration_min_samples"] = int(
            self.get_parameter("calibration_min_samples").value)

        # Force estimation
        params["force_damping"] = float(self.get_parameter("force_damping").value)
        params["force_sign_x"] = float(self.get_parameter("force_sign_x").value)
        params["force_sign_y"] = float(self.get_parameter("force_sign_y").value)
        params["xy_condition_max"] = float(self.get_parameter("xy_condition_max").value)
        params["xy_sigma_min"] = float(self.get_parameter("xy_sigma_min").value)

        # Shadow trip
        params["enable_shadow_trip"] = bool(self.get_parameter("enable_shadow_trip").value)

        # Trip force source: selects which metric feeds every trip threshold.
        # Invalid values fail startup loudly - no silent fallback.
        trip_force_source = self.get_parameter("trip_force_source").value
        if trip_force_source not in ("raw", "minus_g"):
            self.get_logger().error(
                f"Invalid trip_force_source '{trip_force_source}' - "
                f"must be 'raw' or 'minus_g'"
            )
            raise ValueError(
                f"trip_force_source must be 'raw' or 'minus_g', "
                f"got '{trip_force_source}'"
            )
        params["trip_force_source"] = trip_force_source

        params["warning_force_norm"] = float(self.get_parameter("warning_force_norm").value)
        params["trip_force_norm"] = float(self.get_parameter("trip_force_norm").value)
        params["hard_trip_force_norm"] = float(self.get_parameter("hard_trip_force_norm").value)
        params["trip_hold_time_sec"] = float(self.get_parameter("trip_hold_time_sec").value)
        params["clear_hold_time_sec"] = float(self.get_parameter("clear_hold_time_sec").value)

        # CSV
        params["csv_path"] = self.get_parameter("csv_path").value
        params["flush_interval_sec"] = float(self.get_parameter("flush_interval_sec").value)

        return params

    def _create_subscriptions(self) -> None:
        """Create ROS subscriptions."""
        # Sensor data QoS for joint states
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Joint states subscription
        self.js_sub = self.create_subscription(
            JointState,
            self.joint_state_topic,
            self._on_joint_state,
            sensor_qos,
        )

        # Controller state subscriptions
        if "left" in self.sides:
            self.create_subscription(
                JointTrajectoryControllerState,
                self.get_parameter("left_controller_state_topic").value,
                lambda msg: self._on_controller_state(msg, "left"),
                10,
            )

        if "right" in self.sides:
            self.create_subscription(
                JointTrajectoryControllerState,
                self.get_parameter("right_controller_state_topic").value,
                lambda msg: self._on_controller_state(msg, "right"),
                10,
            )

    def _on_joint_state(self, msg: JointState) -> None:
        """Handle incoming joint state message."""
        now = time.monotonic()

        # Check array consistency
        if (len(msg.name) != len(msg.position) or
            (msg.velocity and len(msg.velocity) != len(msg.name)) or
            (msg.effort and len(msg.effort) != len(msg.name))):
            return

        # Check for NaN/inf
        if any(not math.isfinite(v) for v in msg.position):
            return

        ros_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Update each arm
        for side in self.sides:
            model = self.models[side]
            joint_names = [f"openarm_{side}_joint{i}" for i in range(1, 8)]

            try:
                q = np.array([
                    msg.position[msg.name.index(n)] for n in joint_names
                ], dtype=float)

                dq = np.zeros(7)
                if msg.velocity and all(math.isfinite(v) for v in msg.velocity):
                    dq = np.array([
                        msg.velocity[msg.name.index(n)] for n in joint_names
                    ], dtype=float)

                effort = np.zeros(7)
                if msg.effort and all(math.isfinite(v) for v in msg.effort):
                    effort = np.array([
                        msg.effort[msg.name.index(n)] for n in joint_names
                    ], dtype=float)
                else:
                    # No effort data
                    continue

                # Update arm state
                state = self.arm_states[side]
                state.joint_data = JointData(
                    q=q, dq=dq, effort=effort, valid=True,
                    timestamp=now, ros_time=ros_time,
                )
                self.last_joint_state_time[side] = now

            except ValueError:
                # Missing joints
                continue

    def _on_controller_state(
        self,
        msg: JointTrajectoryControllerState,
        side: str,
    ) -> None:
        """Handle controller state message."""
        now = time.monotonic()
        ros_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        state = self.arm_states[side]
        model = self.models[side]
        joint_names = [f"openarm_{side}_joint{i}" for i in range(1, 8)]

        # Track available fields
        available_fields = []

        # Extract reference/desired positions
        ref_pos = None
        if hasattr(msg, 'reference') and msg.reference and msg.reference.positions:
            try:
                ref_pos = np.array([
                    msg.reference.positions[msg.joint_names.index(n)]
                    for n in joint_names
                ], dtype=float)
                available_fields.append("reference")
            except (ValueError, AttributeError):
                pass
        elif hasattr(msg, 'desired') and msg.desired and msg.desired.positions:
            try:
                ref_pos = np.array([
                    msg.desired.positions[msg.joint_names.index(n)]
                    for n in joint_names
                ], dtype=float)
                available_fields.append("desired")
            except (ValueError, AttributeError):
                pass

        # Extract feedback/actual positions
        fb_pos = None
        if hasattr(msg, 'feedback') and msg.feedback and msg.feedback.positions:
            try:
                fb_pos = np.array([
                    msg.feedback.positions[msg.joint_names.index(n)]
                    for n in joint_names
                ], dtype=float)
                available_fields.append("feedback")
            except (ValueError, AttributeError):
                pass
        elif hasattr(msg, 'actual') and msg.actual and msg.actual.positions:
            try:
                fb_pos = np.array([
                    msg.actual.positions[msg.joint_names.index(n)]
                    for n in joint_names
                ], dtype=float)
                available_fields.append("actual")
            except (ValueError, AttributeError):
                pass

        # Extract position error
        pos_err = None
        if hasattr(msg, 'error') and msg.error and msg.error.positions:
            try:
                pos_err = np.array([
                    msg.error.positions[msg.joint_names.index(n)]
                    for n in joint_names
                ], dtype=float)
                available_fields.append("error")
            except (ValueError, AttributeError):
                pass

        # Extract reference/desired velocities
        ref_vel = None
        if hasattr(msg, 'reference') and msg.reference and msg.reference.velocities:
            try:
                ref_vel = np.array([
                    msg.reference.velocities[msg.joint_names.index(n)]
                    for n in joint_names
                ], dtype=float)
            except (ValueError, AttributeError):
                pass
        elif hasattr(msg, 'desired') and msg.desired and msg.desired.velocities:
            try:
                ref_vel = np.array([
                    msg.desired.velocities[msg.joint_names.index(n)]
                    for n in joint_names
                ], dtype=float)
            except (ValueError, AttributeError):
                pass

        # Extract feedback/actual velocities
        fb_vel = None
        if hasattr(msg, 'feedback') and msg.feedback and msg.feedback.velocities:
            try:
                fb_vel = np.array([
                    msg.feedback.velocities[msg.joint_names.index(n)]
                    for n in joint_names
                ], dtype=float)
            except (ValueError, AttributeError):
                pass
        elif hasattr(msg, 'actual') and msg.actual and msg.actual.velocities:
            try:
                fb_vel = np.array([
                    msg.actual.velocities[msg.joint_names.index(n)]
                    for n in joint_names
                ], dtype=float)
            except (ValueError, AttributeError):
                pass

        # Extract velocity error
        vel_err = None
        if hasattr(msg, 'error') and msg.error and msg.error.velocities:
            try:
                vel_err = np.array([
                    msg.error.velocities[msg.joint_names.index(n)]
                    for n in joint_names
                ], dtype=float)
            except (ValueError, AttributeError):
                pass

        state.controller_data = ControllerStateData(
            reference_position=ref_pos,
            feedback_position=fb_pos,
            position_error=pos_err,
            reference_velocity=ref_vel,
            feedback_velocity=fb_vel,
            velocity_error=vel_err,
            valid=True,
            timestamp=now,
            ros_time=ros_time,
            fields_available=available_fields,
        )
        self.last_controller_state_time[side] = now

    def _sample_timer_callback(self) -> None:
        """Main sample timer callback."""
        now = time.monotonic()

        for side in self.sides:
            state = self.arm_states[side]
            model = self.models[side]

            # Check if we have joint data yet
            if state.joint_data is None:
                state.node_state = NodeState.ERROR
                state.invalid_reason = "no_joint_data"
                self._reset_trip_timer(side)  # Reset timer on error
                continue

            # Check data validity
            if not state.joint_data.valid:
                state.node_state = NodeState.ERROR
                state.invalid_reason = "no_joint_data"
                self._reset_trip_timer(side)  # Reset timer on error
                continue

            # Check timeout
            js_age = now - state.joint_data.timestamp
            if js_age > self.joint_state_timeout:
                state.node_state = NodeState.STALE
                state.invalid_reason = f"joint_state_timeout_{js_age:.2f}s"
                self._reset_trip_timer(side)  # Reset timer on stale data
                continue

            controller_age = now - state.controller_data.timestamp
            if controller_age > self.controller_state_timeout:
                # Controller timeout is not critical - continue monitoring
                pass

            # Calibrate if needed
            if state.node_state == NodeState.CALIBRATING:
                self._perform_calibration(side, now)
                if state.baseline.is_valid:
                    state.node_state = NodeState.READY

            # Compute metrics
            metrics = self._compute_metrics(side)

            if metrics is None:
                state.node_state = NodeState.ERROR
                state.invalid_reason = "metric_computation_failed"
                self._reset_trip_timer(side)  # Reset timer on error
                continue

            state.metrics = metrics

            # Classify motion phase
            max_abs_dq = np.max(np.abs(state.joint_data.dq))
            if max_abs_dq <= self.stationary_threshold:
                state.motion_phase = MotionPhase.STATIONARY
            else:
                state.motion_phase = MotionPhase.MOVING

            # Check for motion and reset baseline if needed
            if (state.baseline.is_valid and
                state.motion_phase == MotionPhase.MOVING):
                # Motion detected - baseline stays fixed
                pass

            # Update online stats
            self._update_online_stats(side)

            # Shadow trip check (handles timer reset internally)
            self._check_shadow_trip(side, now)

            # CSV write
            if self.csv_writer is not None:
                self._write_csv_row(side, now)

            # Update node state based on baseline validity
            # (ERROR/STALE already continued above, CALIBRATING handled above)
            if state.baseline.is_valid:
                state.node_state = NodeState.READY
            else:
                state.node_state = NodeState.CALIBRATING

    def _perform_calibration(self, side: str, now: float) -> None:
        """Perform one calibration step."""
        state = self.arm_states[side]

        if state.joint_data is None:
            return
        if not state.joint_data.valid:
            return

        # Check if stationary
        max_abs_dq = np.max(np.abs(state.joint_data.dq))
        if max_abs_dq > self.stationary_threshold:
            return  # Only calibrate when stationary

        # Initialize baseline if needed
        if state.baseline.effort_baseline is None:
            state.baseline.effort_baseline = np.zeros(7)
            state.baseline.effort_minus_g_baseline = np.zeros(7)
            state.baseline.reference_q = state.joint_data.q.copy()
            state.baseline.start_time = now
            state.baseline.std_effort = np.zeros(7)
            state.baseline.std_effort_minus_g = np.zeros(7)
            state.baseline.sum_effort_sq = np.zeros(7)
            state.baseline.sum_effort_minus_g_sq = np.zeros(7)

        # Compute gravity
        model = self.models[side]
        gravity = model.gravity(state.joint_data.q)

        # Compute TCP position for baseline
        full_q = model._full_q(state.joint_data.q)
        pin.framesForwardKinematics(model.model, model.data, full_q)
        oMf = model.data.oMf[model.ee_fid]
        tcp_position = oMf.translation.copy()

        # Accumulate baseline (simple moving average)
        n = state.baseline.samples
        new_n = n + 1

        # Update mean using incremental formula
        state.baseline.effort_baseline = (
            n * state.baseline.effort_baseline + state.joint_data.effort
        ) / new_n
        state.baseline.effort_minus_g_baseline = (
            n * state.baseline.effort_minus_g_baseline +
            (state.joint_data.effort - gravity)
        ) / new_n
        state.baseline.reference_tcp_position = tcp_position

        # Update sum of squares for std computation
        # Using: sum((x_i - mean)^2) incrementally
        # For new sample x and new mean new_mean:
        # sum_sq += (x - old_mean) * (x - new_mean)
        effort = state.joint_data.effort
        effort_minus_g = effort - gravity

        if n == 0:
            state.baseline.sum_effort_sq = np.zeros(7)
            state.baseline.sum_effort_minus_g_sq = np.zeros(7)
        else:
            old_effort_baseline = (
                (n * state.baseline.effort_baseline - effort) / n
                if n > 0 else np.zeros(7)
            )
            old_effort_minus_g_baseline = (
                (n * state.baseline.effort_minus_g_baseline - effort_minus_g) / n
                if n > 0 else np.zeros(7)
            )

            state.baseline.sum_effort_sq += (
                (effort - old_effort_baseline) *
                (effort - state.baseline.effort_baseline)
            )
            state.baseline.sum_effort_minus_g_sq += (
                (effort_minus_g - old_effort_minus_g_baseline) *
                (effort_minus_g - state.baseline.effort_minus_g_baseline)
            )

        # Update max delta
        delta = np.abs(effort - state.baseline.effort_baseline)
        state.baseline.max_effort_delta = max(
            state.baseline.max_effort_delta, np.max(delta)
        )

        delta_minus_g = np.abs(effort_minus_g - state.baseline.effort_minus_g_baseline)
        state.baseline.max_effort_minus_g_delta = max(
            state.baseline.max_effort_minus_g_delta, np.max(delta_minus_g)
        )

        state.baseline.samples += 1

        # Check completion and compute final std
        elapsed = now - state.baseline.start_time
        if (elapsed >= self.calibration_duration and
            state.baseline.samples >= self.calibration_min_samples):
            # Compute final standard deviation
            if state.baseline.samples > 1:
                state.baseline.std_effort = np.sqrt(
                    state.baseline.sum_effort_sq / (state.baseline.samples - 1)
                )
                state.baseline.std_effort_minus_g = np.sqrt(
                    state.baseline.sum_effort_minus_g_sq / (state.baseline.samples - 1)
                )

            state.baseline.is_valid = True
            self.get_logger().info(
                f"{side.upper()}: Baseline calibration complete "
                f"({state.baseline.samples} samples, "
                f"max_dq={max_abs_dq:.4f} rad/s)"
            )

    def _compute_metrics(self, side: str) -> CollisionMetrics | None:
        """Compute collision metrics for one arm."""
        state = self.arm_states[side]
        model = self.models[side]

        if state.joint_data is None:
            return None
        if not state.joint_data.valid:
            return None

        # Get current joint data
        q = state.joint_data.q
        dq = state.joint_data.dq
        effort = state.joint_data.effort

        try:
            # Compute gravity
            gravity = model.gravity(q)

            # Compute FK
            full_q = model._full_q(q)
            pin.framesForwardKinematics(model.model, model.data, full_q)
            oMf = model.data.oMf[model.ee_fid]
            tcp_position = oMf.translation.copy()

            # Compute XY Jacobian
            jac_data = compute_xy_jacobian_data(
                model.model, model.data, model.ee_fid, model.q_idx,
                full_q, True, self.xy_condition_max, self.xy_sigma_min_threshold,
            )

            # Compute residuals
            if state.baseline.is_valid and state.baseline.effort_baseline is not None:
                tau_raw_zeroed = effort - state.baseline.effort_baseline
                if state.baseline.effort_minus_g_baseline is not None:
                    tau_minus_g_zeroed = (
                        effort - gravity - state.baseline.effort_minus_g_baseline
                    )
                else:
                    tau_minus_g_zeroed = effort - gravity
            else:
                tau_raw_zeroed = effort - gravity  # Use gravity as reference
                tau_minus_g_zeroed = np.zeros(7)

            # Estimate XY forces
            fxy_raw_zeroed = estimate_xy_force_from_torque(
                tau_raw_zeroed, jac_data.J_xy, self.force_damping
            )
            fxy_minus_g_zeroed = estimate_xy_force_from_torque(
                tau_minus_g_zeroed, jac_data.J_xy, self.force_damping
            )

            # Apply force signs
            external_raw = np.array([
                self.force_sign_x * fxy_raw_zeroed[0],
                self.force_sign_y * fxy_raw_zeroed[1]
            ])
            external_minus_g = np.array([
                self.force_sign_x * fxy_minus_g_zeroed[0],
                self.force_sign_y * fxy_minus_g_zeroed[1]
            ])

            # Compute measured TCP delta from baseline
            if state.baseline.reference_tcp_position is not None:
                measured_tcp_delta = tcp_position - state.baseline.reference_tcp_position
            else:
                measured_tcp_delta = np.zeros(3)

            return CollisionMetrics(
                tcp_position=tcp_position,
                gravity=gravity,
                tau_raw_zeroed=tau_raw_zeroed,
                tau_minus_g_zeroed=tau_minus_g_zeroed,
                fxy_raw_zeroed=fxy_raw_zeroed,
                fxy_minus_g_zeroed=fxy_minus_g_zeroed,
                external_raw=external_raw,
                external_minus_g=external_minus_g,
                jacobian_data=jac_data,
                measured_tcp_delta=measured_tcp_delta,
            )

        except Exception as e:
            self.get_logger().error(f"Metric computation failed: {e}")
            return None

    def _update_online_stats(self, side: str) -> None:
        """Update online statistics."""
        state = self.arm_states[side]
        metrics = state.metrics

        if metrics is None:
            return
        if state.joint_data is None:
            return

        now = time.monotonic()
        js_age = now - state.joint_data.timestamp
        controller_age = now - state.controller_data.timestamp

        # Get phase stats
        phase_stats = state.online_stats.get_phase_stats(state.motion_phase)
        all_stats = state.online_stats.all_samples

        # Helper to update all three
        def update_metric(name: str, value: float) -> None:
            if math.isfinite(value):
                phase_stats[name].update(value)
                all_stats[name].update(value)

        # Update key metrics
        update_metric("fxy_raw_norm", np.linalg.norm(metrics.fxy_raw_zeroed))
        update_metric("fxy_minus_g_norm", np.linalg.norm(metrics.fxy_minus_g_zeroed))
        update_metric("raw_residual_l2", np.linalg.norm(metrics.tau_raw_zeroed))
        update_metric("minus_g_residual_l2", np.linalg.norm(metrics.tau_minus_g_zeroed))
        update_metric("max_abs_dq", np.max(np.abs(state.joint_data.dq)))
        update_metric("joint_state_age", js_age)
        update_metric("controller_state_age", controller_age)

        # Tracking errors
        if state.controller_data.position_error is not None:
            pos_err_l2 = np.linalg.norm(state.controller_data.position_error)
            update_metric("tracking_position_error_l2", pos_err_l2)

        if state.controller_data.velocity_error is not None:
            vel_err_l2 = np.linalg.norm(state.controller_data.velocity_error)
            update_metric("tracking_velocity_error_l2", vel_err_l2)

    def _reset_trip_timer(self, side: str) -> None:
        """Reset trip timer for an arm (but keep TRIPPED latched)."""
        shadow = self.arm_states[side].shadow_state
        if shadow.trip_latch != TripLatch.TRIPPED:
            shadow.trip_start_time = 0.0

    def _check_shadow_trip(self, side: str, now: float) -> None:
        """Check shadow trip conditions."""
        state = self.arm_states[side]
        metrics = state.metrics
        shadow = state.shadow_state

        # Reset trip timer on invalid data (but keep TRIPPED latched)
        should_reset_timer = False

        if metrics is None:
            should_reset_timer = True
        elif not metrics.jacobian_data.is_reliable:
            should_reset_timer = True
        elif not state.baseline.is_valid:
            should_reset_timer = True
        elif not self.enable_shadow_trip:
            should_reset_timer = True
        elif self.trip_force_norm <= 0:
            should_reset_timer = True

        # Reset timer if needed (but don't clear TRIPPED latch)
        if should_reset_timer:
            if shadow.trip_latch != TripLatch.TRIPPED:
                shadow.trip_start_time = 0.0
            return

        # Single force norm for every threshold check below. The source is
        # selected once by trip_force_source - never recomputed per branch.
        if self.trip_force_source == "minus_g":
            force_norm = np.linalg.norm(metrics.external_minus_g)
        else:  # "raw"
            force_norm = np.linalg.norm(metrics.external_raw)

        if not math.isfinite(force_norm):
            if shadow.trip_latch != TripLatch.TRIPPED:
                shadow.trip_start_time = 0.0
            return

        # Set initial state
        if shadow.state == ShadowState.MONITOR_ONLY:
            shadow.state = ShadowState.ARMED

        already_tripped = shadow.trip_latch == TripLatch.TRIPPED

        # Hard trip: immediate trip, but only one NOT_TRIPPED -> TRIPPED
        # transition per collision event. Once TRIPPED it is never re-counted,
        # re-logged, or re-latched.
        if (not already_tripped and
            self.hard_trip_force_norm > 0 and
            force_norm >= self.hard_trip_force_norm):
            shadow.trip_latch = TripLatch.TRIPPED
            shadow.trip_count += 1
            shadow.state = ShadowState.SHADOW_TRIPPED
            shadow.trip_start_time = now
            self.get_logger().error(
                f"{side.upper()}: SHADOW TRIPPED (hard) - force={force_norm:.2f}"
            )
            return

        # Once TRIPPED, the latch persists. No re-logging, no side effects.
        if already_tripped:
            return

        # Normal trip: force >= trip threshold sustained for trip_hold_time.
        # While force is below trip, any pending hold timer is reset.
        if force_norm >= self.trip_force_norm:
            if shadow.trip_start_time == 0.0:
                shadow.trip_start_time = now
            elif now - shadow.trip_start_time >= self.trip_hold_time:
                shadow.trip_latch = TripLatch.TRIPPED
                shadow.trip_count += 1
                shadow.state = ShadowState.SHADOW_TRIPPED
                self.get_logger().error(
                    f"{side.upper()}: SHADOW TRIPPED - force={force_norm:.2f}"
                )
        else:
            shadow.trip_start_time = 0.0

        # Warning latch - independent of trip hold logic. Must run even when
        # force is only above warning (i.e. below trip), so no early return
        # may skip it.
        #   warning <= force < trip: WARNING_LATCH
        #   force < warning sustained clear_hold_time: clear WARNING_LATCH
        if self.warning_force_norm > 0 and force_norm >= self.warning_force_norm:
            shadow.warning_start_time = 0.0  # cancel any pending clear timer
            shadow.warning_count += 1
            # Never let warning overwrite a TRIPPED latch (e.g. when the
            # normal-trip hold expired on this very sample).
            if shadow.trip_latch != TripLatch.TRIPPED:
                shadow.trip_latch = TripLatch.WARNING_LATCH
                shadow.state = ShadowState.SHADOW_WARNING
        elif shadow.trip_latch == TripLatch.WARNING_LATCH:
            # Force < warning: clear after clear_hold_time sustained below
            if shadow.warning_start_time == 0.0:
                shadow.warning_start_time = now
            elif now - shadow.warning_start_time >= self.clear_hold_time:
                shadow.trip_latch = TripLatch.NOT_TRIPPED
                shadow.state = ShadowState.ARMED
                shadow.warning_start_time = 0.0

    def _print_startup_banner(self) -> None:
        """Print startup banner."""
        self.get_logger().info("=" * 70)
        self.get_logger().info("TCP COLLISION MONITOR SHADOW - STRICTLY READ ONLY")
        self.get_logger().info("=" * 70)
        self.get_logger().info("")
        self.get_logger().info("This node publishes NO robot commands.")
        self.get_logger().info("This node does NOT stop trajectories.")
        self.get_logger().info("This node does NOT control controllers.")
        self.get_logger().info("This node only records motion background and shadow collision metrics.")
        self.get_logger().info("Estimated force units are uncalibrated.")
        self.get_logger().info("")
        self.get_logger().info(f"Side: {self.side}")
        self.get_logger().info(f"URDF: {self.urdf_path}")
        self.get_logger().info(f"Joint state topic: {self.joint_state_topic}")

        if "left" in self.sides:
            self.get_logger().info(
                f"Left controller state: {self.get_parameter('left_controller_state_topic').value}"
            )
        if "right" in self.sides:
            self.get_logger().info(
                f"Right controller state: {self.get_parameter('right_controller_state_topic').value}"
            )

        self.get_logger().info("")
        self.get_logger().info(f"Sample rate: {self.sample_rate} Hz")
        self.get_logger().info(f"Print rate: {self.print_rate} Hz")
        self.get_logger().info(f"Calibration: {self.calibration_duration}s / {self.calibration_min_samples} samples")
        self.get_logger().info(f"Force damping: {self.force_damping}")
        self.get_logger().info(f"XY condition max: {self.xy_condition_max}, sigma_min: {self.xy_sigma_min_threshold}")
        self.get_logger().info(f"Force signs: X={self.force_sign_x}, Y={self.force_sign_y}")
        self.get_logger().info("")
        self.get_logger().info(f"Shadow trip: {'ENABLED' if self.enable_shadow_trip else 'DISABLED'}")
        if self.enable_shadow_trip:
            self.get_logger().info(f"  Trip force source: {self.trip_force_source}")
            self.get_logger().info(f"  Warning threshold: {self.warning_force_norm}")
            self.get_logger().info(f"  Trip threshold: {self.trip_force_norm}")
            self.get_logger().info(f"  Hard trip threshold: {self.hard_trip_force_norm}")
        self.get_logger().info("")
        self.get_logger().info(f"CSV: {'ENABLED' if self.csv_path else 'DISABLED'}")
        if self.csv_path:
            self.get_logger().info(f"  Path: {self.csv_path}")
        self.get_logger().info("")
        self.get_logger().info("=" * 70)

    def _print_timer_callback(self) -> None:
        """Print periodic summary."""
        now = time.monotonic()

        for side in self.sides:
            state = self.arm_states[side]

            # Print calibration status
            if state.node_state == NodeState.CALIBRATING:
                n = state.baseline.samples
                if state.joint_data is not None and state.joint_data.valid:
                    max_dq = np.max(np.abs(state.joint_data.dq))
                else:
                    max_dq = 0.0
                self.get_logger().info(
                    f"{side.upper()} CALIBRATING: samples={n}/{self.calibration_min_samples}, max_dq={max_dq:.4f}"
                )
                continue

            # Print error/waiting status
            if state.joint_data is None:
                self.get_logger().info(
                    f"{side.upper()}: WAITING FOR JOINT DATA - no /joint_states received"
                )
                continue

            if state.metrics is None:
                if state.node_state == NodeState.ERROR:
                    self.get_logger().info(
                        f"{side.upper()}: {state.node_state.value.upper()} - {state.invalid_reason}"
                    )
                continue

            metrics = state.metrics
            js_age = now - state.joint_data.timestamp
            ctrl_age = now - state.controller_data.timestamp

            # Build summary
            lines = [
                "=" * 70,
                f"{side.upper()} | {state.shadow_state.state.value.upper()} | {state.motion_phase.value.upper()}",
                "=" * 70,
                f"Joint age: {js_age:.3f}s | Controller age: {ctrl_age:.3f}s",
                f"Baseline: {'VALID' if state.baseline.is_valid else 'CALIBRATING'}",
                f"XY reliable: {'YES' if metrics.jacobian_data.is_reliable else 'NO'}",
                f"Condition: {metrics.jacobian_data.condition_number:.1f}",
                f"Sigma min: {metrics.jacobian_data.sigma_min:.4f}",
                "",
                "TCP world (m):",
                f"  x={metrics.tcp_position[0]:.4f} y={metrics.tcp_position[1]:.4f} z={metrics.tcp_position[2]:.4f}",
                "",
                "Effort residual (uncalibrated):",
                f"  raw L2={np.linalg.norm(metrics.tau_raw_zeroed):.2f}",
                f"  raw max={np.max(np.abs(metrics.tau_raw_zeroed)):.2f}",
                f"  minus-g L2={np.linalg.norm(metrics.tau_minus_g_zeroed):.2f}",
                f"  minus-g max={np.max(np.abs(metrics.tau_minus_g_zeroed)):.2f}",
                "",
                "Estimated XY force (uncalibrated):",
                f"  raw: Fx={metrics.fxy_raw_zeroed[0]:.2f} Fy={metrics.fxy_raw_zeroed[1]:.2f} "
                f"norm={np.linalg.norm(metrics.fxy_raw_zeroed):.2f}",
                f"  minus-g: Fx={metrics.fxy_minus_g_zeroed[0]:.2f} Fy={metrics.fxy_minus_g_zeroed[1]:.2f} "
                f"norm={np.linalg.norm(metrics.fxy_minus_g_zeroed):.2f}",
                "",
            ]

            # Add tracking info
            if state.controller_data.position_error is not None:
                pos_err_l2 = np.linalg.norm(state.controller_data.position_error)
                pos_err_max = np.max(np.abs(state.controller_data.position_error))
                lines.append("Tracking:")
                lines.append(f"  position error L2={pos_err_l2:.4f} max={pos_err_max:.4f}")
            else:
                lines.append("Tracking: N/A")

            # Add shadow info
            lines.extend([
                "",
                "Shadow:",
                f"  warning={'YES' if state.shadow_state.state == ShadowState.SHADOW_WARNING else 'NO'}",
                f"  tripped={'YES' if state.shadow_state.trip_latch == TripLatch.TRIPPED else 'NO'}",
                "NO ROBOT STOP COMMAND WAS SENT.",
                "=" * 70,
            ])

            for line in lines:
                self.get_logger().info(line)

    def _init_csv(self) -> None:
        """Initialize CSV file."""
        try:
            csv_path = Path(self.csv_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)

            self.csv_file = csv_path
            self.csv_writer = csv.writer(csv_path.open('w', newline=''))

            # Write header
            header = [
                "monotonic_time", "ros_time", "side", "node_state", "motion_phase",
                "joint_state_age", "controller_state_age",
                "q1", "q2", "q3", "q4", "q5", "q6", "q7",
                "dq1", "dq2", "dq3", "dq4", "dq5", "dq6", "dq7",
                "effort1", "effort2", "effort3", "effort4", "effort5", "effort6", "effort7",
                "gravity1", "gravity2", "gravity3", "gravity4", "gravity5", "gravity6", "gravity7",
                "tau_raw_zeroed1", "tau_raw_zeroed2", "tau_raw_zeroed3", "tau_raw_zeroed4",
                "tau_raw_zeroed5", "tau_raw_zeroed6", "tau_raw_zeroed7",
                "tau_minus_g_zeroed1", "tau_minus_g_zeroed2", "tau_minus_g_zeroed3", "tau_minus_g_zeroed4",
                "tau_minus_g_zeroed5", "tau_minus_g_zeroed6", "tau_minus_g_zeroed7",
                "max_abs_dq",
                "raw_residual_l2", "raw_residual_max_abs",
                "minus_g_residual_l2", "minus_g_residual_max_abs",
                "tcp_x", "tcp_y", "tcp_z",
                "measured_tcp_dx", "measured_tcp_dy", "measured_tcp_dz",
                "fxy_raw_x", "fxy_raw_y", "fxy_raw_norm",
                "fxy_minus_g_x", "fxy_minus_g_y", "fxy_minus_g_norm",
                "external_raw_x", "external_raw_y", "external_raw_norm",
                "external_minus_g_x", "external_minus_g_y", "external_minus_g_norm",
                "xy_sigma_min", "xy_condition", "xy_reliable",
                "ref_pos1", "ref_pos2", "ref_pos3", "ref_pos4", "ref_pos5", "ref_pos6", "ref_pos7",
                "fb_pos1", "fb_pos2", "fb_pos3", "fb_pos4", "fb_pos5", "fb_pos6", "fb_pos7",
                "pos_err1", "pos_err2", "pos_err3", "pos_err4", "pos_err5", "pos_err6", "pos_err7",
                "ref_vel1", "ref_vel2", "ref_vel3", "ref_vel4", "ref_vel5", "ref_vel6", "ref_vel7",
                "fb_vel1", "fb_vel2", "fb_vel3", "fb_vel4", "fb_vel5", "fb_vel6", "fb_vel7",
                "vel_err1", "vel_err2", "vel_err3", "vel_err4", "vel_err5", "vel_err6", "vel_err7",
                "tracking_pos_err_l2", "tracking_pos_err_max_abs",
                "tracking_vel_err_l2", "tracking_vel_err_max_abs",
                "controller_fields_available",
                "shadow_warning", "shadow_tripped",
                "invalid_reason",
            ]
            self.csv_writer.writerow(header)
            self.get_logger().info(f"CSV header written to {self.csv_path}")

        except Exception as e:
            self.get_logger().error(f"Failed to initialize CSV: {e}")
            self.csv_file = None
            self.csv_writer = None

    def _write_csv_row(self, side: str, now: float) -> None:
        """Write one CSV row."""
        if self.csv_writer is None:
            return

        state = self.arm_states[side]
        metrics = state.metrics

        if metrics is None:
            return

        if state.joint_data is None:
            return

        try:
            # Helper for 7-element arrays
            def arr7(arr: np.ndarray | None) -> list:
                if arr is None:
                    return [""] * 7
                return [nan_to_str(x) for x in arr]

            js_age = now - state.joint_data.timestamp
            ctrl_age = now - state.controller_data.timestamp

            row = CSVRow(
                monotonic_time=now,
                ros_time=state.joint_data.ros_time,
                side=side,
                node_state=state.node_state.value,
                motion_phase=state.motion_phase.value,
                joint_state_age=js_age,
                controller_state_age=ctrl_age,
                # Joint positions
                q1=nan_to_str(state.joint_data.q[0]),
                q2=nan_to_str(state.joint_data.q[1]),
                q3=nan_to_str(state.joint_data.q[2]),
                q4=nan_to_str(state.joint_data.q[3]),
                q5=nan_to_str(state.joint_data.q[4]),
                q6=nan_to_str(state.joint_data.q[5]),
                q7=nan_to_str(state.joint_data.q[6]),
                # Joint velocities
                dq1=nan_to_str(state.joint_data.dq[0]),
                dq2=nan_to_str(state.joint_data.dq[1]),
                dq3=nan_to_str(state.joint_data.dq[2]),
                dq4=nan_to_str(state.joint_data.dq[3]),
                dq5=nan_to_str(state.joint_data.dq[4]),
                dq6=nan_to_str(state.joint_data.dq[5]),
                dq7=nan_to_str(state.joint_data.dq[6]),
                # Joint efforts
                effort1=nan_to_str(state.joint_data.effort[0]),
                effort2=nan_to_str(state.joint_data.effort[1]),
                effort3=nan_to_str(state.joint_data.effort[2]),
                effort4=nan_to_str(state.joint_data.effort[3]),
                effort5=nan_to_str(state.joint_data.effort[4]),
                effort6=nan_to_str(state.joint_data.effort[5]),
                effort7=nan_to_str(state.joint_data.effort[6]),
                # Gravity
                gravity1=nan_to_str(metrics.gravity[0]),
                gravity2=nan_to_str(metrics.gravity[1]),
                gravity3=nan_to_str(metrics.gravity[2]),
                gravity4=nan_to_str(metrics.gravity[3]),
                gravity5=nan_to_str(metrics.gravity[4]),
                gravity6=nan_to_str(metrics.gravity[5]),
                gravity7=nan_to_str(metrics.gravity[6]),
                # Residuals
                tau_raw_zeroed1=nan_to_str(metrics.tau_raw_zeroed[0]),
                tau_raw_zeroed2=nan_to_str(metrics.tau_raw_zeroed[1]),
                tau_raw_zeroed3=nan_to_str(metrics.tau_raw_zeroed[2]),
                tau_raw_zeroed4=nan_to_str(metrics.tau_raw_zeroed[3]),
                tau_raw_zeroed5=nan_to_str(metrics.tau_raw_zeroed[4]),
                tau_raw_zeroed6=nan_to_str(metrics.tau_raw_zeroed[5]),
                tau_raw_zeroed7=nan_to_str(metrics.tau_raw_zeroed[6]),
                tau_minus_g_zeroed1=nan_to_str(metrics.tau_minus_g_zeroed[0]),
                tau_minus_g_zeroed2=nan_to_str(metrics.tau_minus_g_zeroed[1]),
                tau_minus_g_zeroed3=nan_to_str(metrics.tau_minus_g_zeroed[2]),
                tau_minus_g_zeroed4=nan_to_str(metrics.tau_minus_g_zeroed[3]),
                tau_minus_g_zeroed5=nan_to_str(metrics.tau_minus_g_zeroed[4]),
                tau_minus_g_zeroed6=nan_to_str(metrics.tau_minus_g_zeroed[5]),
                tau_minus_g_zeroed7=nan_to_str(metrics.tau_minus_g_zeroed[6]),
                max_abs_dq=nan_to_str(np.max(np.abs(state.joint_data.dq))),
                raw_residual_l2=nan_to_str(np.linalg.norm(metrics.tau_raw_zeroed)),
                raw_residual_max_abs=nan_to_str(np.max(np.abs(metrics.tau_raw_zeroed))),
                minus_g_residual_l2=nan_to_str(np.linalg.norm(metrics.tau_minus_g_zeroed)),
                minus_g_residual_max_abs=nan_to_str(np.max(np.abs(metrics.tau_minus_g_zeroed))),
                # TCP position
                tcp_x=nan_to_str(metrics.tcp_position[0]),
                tcp_y=nan_to_str(metrics.tcp_position[1]),
                tcp_z=nan_to_str(metrics.tcp_position[2]),
                # TCP delta from baseline
                measured_tcp_dx=nan_to_str(metrics.measured_tcp_delta[0]),
                measured_tcp_dy=nan_to_str(metrics.measured_tcp_delta[1]),
                measured_tcp_dz=nan_to_str(metrics.measured_tcp_delta[2]),
                # XY forces
                fxy_raw_x=nan_to_str(metrics.fxy_raw_zeroed[0]),
                fxy_raw_y=nan_to_str(metrics.fxy_raw_zeroed[1]),
                fxy_raw_norm=nan_to_str(np.linalg.norm(metrics.fxy_raw_zeroed)),
                fxy_minus_g_x=nan_to_str(metrics.fxy_minus_g_zeroed[0]),
                fxy_minus_g_y=nan_to_str(metrics.fxy_minus_g_zeroed[1]),
                fxy_minus_g_norm=nan_to_str(np.linalg.norm(metrics.fxy_minus_g_zeroed)),
                external_raw_x=nan_to_str(metrics.external_raw[0]),
                external_raw_y=nan_to_str(metrics.external_raw[1]),
                external_raw_norm=nan_to_str(np.linalg.norm(metrics.external_raw)),
                external_minus_g_x=nan_to_str(metrics.external_minus_g[0]),
                external_minus_g_y=nan_to_str(metrics.external_minus_g[1]),
                external_minus_g_norm=nan_to_str(np.linalg.norm(metrics.external_minus_g)),
                # Jacobian
                xy_sigma_min=nan_to_str(metrics.jacobian_data.sigma_min),
                xy_condition=nan_to_str(metrics.jacobian_data.condition_number),
                xy_reliable=1 if metrics.jacobian_data.is_reliable else 0,
                # Controller state
                ref_pos1=nan_to_str(state.controller_data.reference_position[0]) if state.controller_data.reference_position is not None else "",
                ref_pos2=nan_to_str(state.controller_data.reference_position[1]) if state.controller_data.reference_position is not None else "",
                ref_pos3=nan_to_str(state.controller_data.reference_position[2]) if state.controller_data.reference_position is not None else "",
                ref_pos4=nan_to_str(state.controller_data.reference_position[3]) if state.controller_data.reference_position is not None else "",
                ref_pos5=nan_to_str(state.controller_data.reference_position[4]) if state.controller_data.reference_position is not None else "",
                ref_pos6=nan_to_str(state.controller_data.reference_position[5]) if state.controller_data.reference_position is not None else "",
                ref_pos7=nan_to_str(state.controller_data.reference_position[6]) if state.controller_data.reference_position is not None else "",
                fb_pos1=nan_to_str(state.controller_data.feedback_position[0]) if state.controller_data.feedback_position is not None else "",
                fb_pos2=nan_to_str(state.controller_data.feedback_position[1]) if state.controller_data.feedback_position is not None else "",
                fb_pos3=nan_to_str(state.controller_data.feedback_position[2]) if state.controller_data.feedback_position is not None else "",
                fb_pos4=nan_to_str(state.controller_data.feedback_position[3]) if state.controller_data.feedback_position is not None else "",
                fb_pos5=nan_to_str(state.controller_data.feedback_position[4]) if state.controller_data.feedback_position is not None else "",
                fb_pos6=nan_to_str(state.controller_data.feedback_position[5]) if state.controller_data.feedback_position is not None else "",
                fb_pos7=nan_to_str(state.controller_data.feedback_position[6]) if state.controller_data.feedback_position is not None else "",
                pos_err1=nan_to_str(state.controller_data.position_error[0]) if state.controller_data.position_error is not None else "",
                pos_err2=nan_to_str(state.controller_data.position_error[1]) if state.controller_data.position_error is not None else "",
                pos_err3=nan_to_str(state.controller_data.position_error[2]) if state.controller_data.position_error is not None else "",
                pos_err4=nan_to_str(state.controller_data.position_error[3]) if state.controller_data.position_error is not None else "",
                pos_err5=nan_to_str(state.controller_data.position_error[4]) if state.controller_data.position_error is not None else "",
                pos_err6=nan_to_str(state.controller_data.position_error[5]) if state.controller_data.position_error is not None else "",
                pos_err7=nan_to_str(state.controller_data.position_error[6]) if state.controller_data.position_error is not None else "",
                ref_vel1=nan_to_str(state.controller_data.reference_velocity[0]) if state.controller_data.reference_velocity is not None else "",
                ref_vel2=nan_to_str(state.controller_data.reference_velocity[1]) if state.controller_data.reference_velocity is not None else "",
                ref_vel3=nan_to_str(state.controller_data.reference_velocity[2]) if state.controller_data.reference_velocity is not None else "",
                ref_vel4=nan_to_str(state.controller_data.reference_velocity[3]) if state.controller_data.reference_velocity is not None else "",
                ref_vel5=nan_to_str(state.controller_data.reference_velocity[4]) if state.controller_data.reference_velocity is not None else "",
                ref_vel6=nan_to_str(state.controller_data.reference_velocity[5]) if state.controller_data.reference_velocity is not None else "",
                ref_vel7=nan_to_str(state.controller_data.reference_velocity[6]) if state.controller_data.reference_velocity is not None else "",
                fb_vel1=nan_to_str(state.controller_data.feedback_velocity[0]) if state.controller_data.feedback_velocity is not None else "",
                fb_vel2=nan_to_str(state.controller_data.feedback_velocity[1]) if state.controller_data.feedback_velocity is not None else "",
                fb_vel3=nan_to_str(state.controller_data.feedback_velocity[2]) if state.controller_data.feedback_velocity is not None else "",
                fb_vel4=nan_to_str(state.controller_data.feedback_velocity[3]) if state.controller_data.feedback_velocity is not None else "",
                fb_vel5=nan_to_str(state.controller_data.feedback_velocity[4]) if state.controller_data.feedback_velocity is not None else "",
                fb_vel6=nan_to_str(state.controller_data.feedback_velocity[5]) if state.controller_data.feedback_velocity is not None else "",
                fb_vel7=nan_to_str(state.controller_data.feedback_velocity[6]) if state.controller_data.feedback_velocity is not None else "",
                vel_err1=nan_to_str(state.controller_data.velocity_error[0]) if state.controller_data.velocity_error is not None else "",
                vel_err2=nan_to_str(state.controller_data.velocity_error[1]) if state.controller_data.velocity_error is not None else "",
                vel_err3=nan_to_str(state.controller_data.velocity_error[2]) if state.controller_data.velocity_error is not None else "",
                vel_err4=nan_to_str(state.controller_data.velocity_error[3]) if state.controller_data.velocity_error is not None else "",
                vel_err5=nan_to_str(state.controller_data.velocity_error[4]) if state.controller_data.velocity_error is not None else "",
                vel_err6=nan_to_str(state.controller_data.velocity_error[5]) if state.controller_data.velocity_error is not None else "",
                vel_err7=nan_to_str(state.controller_data.velocity_error[6]) if state.controller_data.velocity_error is not None else "",
                tracking_pos_err_l2=nan_to_str(np.linalg.norm(state.controller_data.position_error)) if state.controller_data.position_error is not None else "",
                tracking_pos_err_max_abs=nan_to_str(np.max(np.abs(state.controller_data.position_error))) if state.controller_data.position_error is not None else "",
                tracking_vel_err_l2=nan_to_str(np.linalg.norm(state.controller_data.velocity_error)) if state.controller_data.velocity_error is not None else "",
                tracking_vel_err_max_abs=nan_to_str(np.max(np.abs(state.controller_data.velocity_error))) if state.controller_data.velocity_error is not None else "",
                controller_fields_available=",".join(state.controller_data.fields_available),
                shadow_warning=1 if state.shadow_state.state == ShadowState.SHADOW_WARNING else 0,
                shadow_tripped=1 if state.shadow_state.trip_latch == TripLatch.TRIPPED else 0,
                invalid_reason=state.invalid_reason,
            )

            self.csv_writer.writerow([
                row.monotonic_time, row.ros_time, row.side, row.node_state, row.motion_phase,
                row.joint_state_age, row.controller_state_age,
                row.q1, row.q2, row.q3, row.q4, row.q5, row.q6, row.q7,
                row.dq1, row.dq2, row.dq3, row.dq4, row.dq5, row.dq6, row.dq7,
                row.effort1, row.effort2, row.effort3, row.effort4, row.effort5, row.effort6, row.effort7,
                row.gravity1, row.gravity2, row.gravity3, row.gravity4, row.gravity5, row.gravity6, row.gravity7,
                row.tau_raw_zeroed1, row.tau_raw_zeroed2, row.tau_raw_zeroed3, row.tau_raw_zeroed4,
                row.tau_raw_zeroed5, row.tau_raw_zeroed6, row.tau_raw_zeroed7,
                row.tau_minus_g_zeroed1, row.tau_minus_g_zeroed2, row.tau_minus_g_zeroed3, row.tau_minus_g_zeroed4,
                row.tau_minus_g_zeroed5, row.tau_minus_g_zeroed6, row.tau_minus_g_zeroed7,
                row.max_abs_dq,
                row.raw_residual_l2, row.raw_residual_max_abs,
                row.minus_g_residual_l2, row.minus_g_residual_max_abs,
                row.tcp_x, row.tcp_y, row.tcp_z,
                row.measured_tcp_dx, row.measured_tcp_dy, row.measured_tcp_dz,
                row.fxy_raw_x, row.fxy_raw_y, row.fxy_raw_norm,
                row.fxy_minus_g_x, row.fxy_minus_g_y, row.fxy_minus_g_norm,
                row.external_raw_x, row.external_raw_y, row.external_raw_norm,
                row.external_minus_g_x, row.external_minus_g_y, row.external_minus_g_norm,
                row.xy_sigma_min, row.xy_condition, row.xy_reliable,
                row.ref_pos1, row.ref_pos2, row.ref_pos3, row.ref_pos4, row.ref_pos5, row.ref_pos6, row.ref_pos7,
                row.fb_pos1, row.fb_pos2, row.fb_pos3, row.fb_pos4, row.fb_pos5, row.fb_pos6, row.fb_pos7,
                row.pos_err1, row.pos_err2, row.pos_err3, row.pos_err4, row.pos_err5, row.pos_err6, row.pos_err7,
                row.ref_vel1, row.ref_vel2, row.ref_vel3, row.ref_vel4, row.ref_vel5, row.ref_vel6, row.ref_vel7,
                row.fb_vel1, row.fb_vel2, row.fb_vel3, row.fb_vel4, row.fb_vel5, row.fb_vel6, row.fb_vel7,
                row.vel_err1, row.vel_err2, row.vel_err3, row.vel_err4, row.vel_err5, row.vel_err6, row.vel_err7,
                row.tracking_pos_err_l2, row.tracking_pos_err_max_abs,
                row.tracking_vel_err_l2, row.tracking_vel_err_max_abs,
                row.controller_fields_available,
                row.shadow_warning, row.shadow_tripped,
                row.invalid_reason,
            ])

            self.csv_row_count += 1

        except Exception as e:
            self.get_logger().error(f"CSV write failed: {e}")
            self.csv_writer = None
            self.csv_file = None

    def _flush_timer_callback(self) -> None:
        """Periodically flush CSV."""
        if self.csv_file is not None:
            try:
                self.csv_file.flush()
            except Exception:
                pass

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self._cleanup()

    def _cleanup(self) -> None:
        """Close CSV and print stats. Safe to call multiple times and with partial initialization."""
        if self.cleanup_done:
            return

        # Close CSV file
        if hasattr(self, 'csv_file') and self.csv_file is not None:
            try:
                self.csv_file.close()
                if hasattr(self, 'get_logger'):
                    self.get_logger().info(
                        f"CSV closed: {self.csv_row_count} rows written"
                    )
            except Exception:
                pass
            self.csv_file = None
            self.csv_writer = None

        self.cleanup_done = True

        # Print session statistics
        if not hasattr(self, 'get_logger'):
            return
        if not hasattr(self, 'sides'):
            return

        self.get_logger().info("")
        self.get_logger().info("=" * 70)
        self.get_logger().info("SESSION STATISTICS")
        self.get_logger().info("=" * 70)

        for side in self.sides:
            state = self.arm_states[side]
            stats = state.online_stats

            self.get_logger().info("")
            self.get_logger().info(f"{side.upper()}:")
            self.get_logger().info(f"  Calibration: {'VALID' if state.baseline.is_valid else 'INCOMPLETE'}")
            if state.baseline.is_valid:
                self.get_logger().info(f"    Samples: {state.baseline.samples}")

            # Print stats for ALL phase
            all_stats = stats.all_samples
            self.get_logger().info(f"  Total samples: {all_stats['fxy_raw_norm'].count if 'fxy_raw_norm' in all_stats else 0}")

            for phase_name, phase_stats in [
                ("STATIONARY", stats.stationary),
                ("MOVING", stats.moving),
            ]:
                if 'fxy_raw_norm' in phase_stats and phase_stats['fxy_raw_norm'].count > 0:
                    self.get_logger().info(f"  {phase_name}: {phase_stats['fxy_raw_norm'].count} samples")
                    self.get_logger().info(f"    Fxy raw: max={phase_stats['fxy_raw_norm'].maximum:.2f}, "
                                           f"RMS={phase_stats['fxy_raw_norm'].rms:.2f}")

            # Shadow stats
            shadow = state.shadow_state
            self.get_logger().info(f"  Shadow: {shadow.state.value}")
            self.get_logger().info(f"    Warnings: {shadow.warning_count}")
            self.get_logger().info(f"    Trips: {shadow.trip_count}")

        self.get_logger().info("=" * 70)


def main() -> None:
    """Main entry point."""
    rclpy.init()

    node = TCPCollisionMonitorShadowNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Node error: {e}\n{traceback.format_exc()}")
    finally:
        node._cleanup()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
