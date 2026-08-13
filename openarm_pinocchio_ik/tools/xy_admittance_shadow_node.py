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

"""READ-ONLY XY admittance shadow node for virtual target visualization.

This node subscribes to /joint_states and computes a virtual XY admittance target
using estimated external forces from joint efforts. It does NOT send any robot commands.

The virtual target is visualized in MeshCat and terminal output for verification
before connecting to a real admittance controller.

Estimated force units; no F/T sensor calibration.
"""

import csv
import math
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

import numpy as np
import pinocchio as pin
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState

if TYPE_CHECKING:
    try:
        from pinocchio.visualize import MeshcatVisualizer
    except ImportError:
        MeshcatVisualizer = None  # type: ignore

try:
    import meshcat
except ImportError:
    meshcat = None  # type: ignore


# -----------------------------------------------------------------------------
# Enums and Data Classes
# -----------------------------------------------------------------------------

class ShadowState(Enum):
    """States of the shadow node."""
    INIT = "initializing"
    WAITING_FIRST_MSG = "waiting_for_first_joint_state"
    CALIBRATING = "calibrating_baseline"
    READY = "ready"
    STALE = "stale_data"
    PAUSED = "paused_safety_gate"
    ERROR = "error"


@dataclass
class BaselineData:
    """Baseline calibration data."""
    effort_baseline: Optional[np.ndarray]  # 7x1 or None
    effort_minus_g_baseline: Optional[np.ndarray]  # 7x1 or None
    reference_q: Optional[np.ndarray]  # 7x1 or None
    reference_tcp_position: Optional[np.ndarray]  # 3x1 or None
    reference_tcp_rotation: Optional[np.ndarray]  # 4x1 quaternion or None
    samples_collected: int
    start_time: Optional[float]
    is_valid: bool


@dataclass
class XYJacobianData:
    """XY Jacobian and reliability metrics."""
    J_xy: np.ndarray  # 2x7
    singular_values: np.ndarray  # 2 values
    sigma_min: float
    condition_number: float
    is_reliable: bool


@dataclass
class AdmittanceState:
    """Virtual admittance state."""
    displacement: np.ndarray  # 2x1 [dx, dy] in meters
    velocity: np.ndarray  # 2x1 [vx, vy] in m/s


@dataclass
class GateStatus:
    """Safety gate status."""
    ok: bool
    reason: Optional[str]


# -----------------------------------------------------------------------------
# Pure 2D Admittance Calculator
# -----------------------------------------------------------------------------

class Admittance2D:
    """Pure 2D admittance model calculator."""

    def __init__(
        self,
        mass_x: float = 2.0,
        mass_y: float = 2.0,
        damping_x: float = 90.0,
        damping_y: float = 90.0,
        stiffness_x: float = 1000.0,
        stiffness_y: float = 1000.0,
        max_virtual_offset_m: float = 0.003,
        max_virtual_velocity_mps: float = 0.02,
        max_virtual_acceleration_mps2: float = 0.2,
        max_integration_dt_sec: float = 0.05,
    ):
        self.mass = np.array([mass_x, mass_y], dtype=float)
        self.damping = np.array([damping_x, damping_y], dtype=float)
        self.stiffness = np.array([stiffness_x, stiffness_y], dtype=float)
        self.max_offset = max_virtual_offset_m
        self.max_velocity = max_virtual_velocity_mps
        self.max_acceleration = max_virtual_acceleration_mps2
        self.max_dt = max_integration_dt_sec

        self.state = AdmittanceState(
            displacement=np.zeros(2),
            velocity=np.zeros(2),
        )
        self.last_update_time: Optional[float] = None

    def reset(self) -> None:
        """Reset admittance state to zero."""
        self.state.displacement = np.zeros(2)
        self.state.velocity = np.zeros(2)
        self.last_update_time = None

    def update(
        self,
        force_used: np.ndarray,
        dt: Optional[float],
    ) -> AdmittanceState:
        """Update admittance state with applied force.

        Args:
            force_used: 2D force vector after deadzone and clipping [Fx, Fy].
            dt: Time step in seconds. If None or invalid, state is reset.

        Returns:
            Updated admittance state.
        """
        # Check dt validity
        if dt is None or not math.isfinite(dt) or dt <= 0.0 or dt > self.max_dt:
            # Invalid dt - reset state
            self.reset()
            return self.state

        # Compute acceleration: a = (F - D*v - K*x) / M
        acceleration = (
            force_used
            - self.damping * self.state.velocity
            - self.stiffness * self.state.displacement
        ) / self.mass

        # Clip acceleration
        acceleration = np.clip(
            acceleration, -self.max_acceleration, self.max_acceleration
        )

        # Semi-implicit Euler integration
        self.state.velocity += acceleration * dt
        self.state.velocity = np.clip(
            self.state.velocity, -self.max_velocity, self.max_velocity
        )

        self.state.displacement += self.state.velocity * dt

        # Clip displacement by norm
        disp_norm = np.linalg.norm(self.state.displacement)
        if disp_norm > self.max_offset:
            self.state.displacement *= self.max_offset / disp_norm
            # Zero velocity component pointing outside boundary
            if disp_norm > 0:
                outward_dir = self.state.displacement / disp_norm
                v_outward = np.dot(self.state.velocity, outward_dir)
                if v_outward > 0:
                    self.state.velocity -= v_outward * outward_dir

        return self.state


# -----------------------------------------------------------------------------
# Main Shadow Node
# -----------------------------------------------------------------------------

class XYAdmittanceShadowNode(Node):
    """READ-ONLY XY admittance shadow node.

    This node:
        - Subscribes to /joint_states
        - Estimates XY external force from joint efforts
        - Computes virtual 2D admittance target
        - Visualizes in MeshCat and terminal
        - Publishes NO robot commands
    """

    def __init__(self) -> None:
        super().__init__("xy_admittance_shadow")

        # -----------------------------------------------------------------
        # Declare Parameters
        # -----------------------------------------------------------------
        self.declare_parameter("side", "left")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("joint_state_timeout_sec", 0.2)
        self.declare_parameter("stationary_velocity_threshold_rad_s", 0.03)
        self.declare_parameter("shadow_rate_hz", 100.0)
        self.declare_parameter("print_rate_hz", 1.0)
        self.declare_parameter("meshcat_rate_hz", 20.0)

        # Baseline parameters
        self.declare_parameter("calibration_duration_sec", 2.0)
        self.declare_parameter("calibration_min_samples", 100)

        # Force estimation
        self.declare_parameter("force_damping", 0.01)
        self.declare_parameter("force_source", "raw_zeroed")
        self.declare_parameter("force_sign_x", -1.0)
        self.declare_parameter("force_sign_y", -1.0)
        self.declare_parameter("force_deadzone_x", 0.5)
        self.declare_parameter("force_deadzone_y", 0.5)
        self.declare_parameter("max_force_norm", 8.0)

        # XY reliability
        self.declare_parameter("xy_condition_max", 50.0)
        self.declare_parameter("xy_sigma_min", 0.02)

        # Admittance parameters
        self.declare_parameter("mass_x", 2.0)
        self.declare_parameter("mass_y", 2.0)
        self.declare_parameter("damping_x", 90.0)
        self.declare_parameter("damping_y", 90.0)
        self.declare_parameter("stiffness_x", 1000.0)
        self.declare_parameter("stiffness_y", 1000.0)

        # Safety limits
        self.declare_parameter("max_virtual_offset_m", 0.003)
        self.declare_parameter("max_virtual_velocity_mps", 0.02)
        self.declare_parameter("max_virtual_acceleration_mps2", 0.2)
        self.declare_parameter("max_reference_tcp_deviation_m", 0.02)
        self.declare_parameter("max_reference_joint_deviation_rad", 0.20)
        self.declare_parameter("max_integration_dt_sec", 0.05)

        # Visualization and logging
        self.declare_parameter("enable_meshcat", True)
        self.declare_parameter("csv_path", "")

        # Get parameters
        params = self._get_and_validate_parameters()

        # Store parameters
        self.side = params["side"]
        self.joint_state_topic = params["joint_state_topic"]
        self.joint_state_timeout = params["joint_state_timeout_sec"]
        self.stationary_threshold = params["stationary_velocity_threshold_rad_s"]
        self.shadow_rate = params["shadow_rate_hz"]
        self.print_rate = params["print_rate_hz"]
        self.meshcat_rate = params["meshcat_rate_hz"]

        # Baseline
        self.calib_duration = params["calibration_duration_sec"]
        self.calib_min_samples = params["calibration_min_samples"]

        # Force estimation
        self.force_damping = params["force_damping"]
        self.force_source = params["force_source"]
        self.force_sign = np.array([params["force_sign_x"], params["force_sign_y"]])
        self.force_deadzone = np.array([params["force_deadzone_x"], params["force_deadzone_y"]])
        self.max_force = params["max_force_norm"]

        # XY reliability
        self.xy_cond_max = params["xy_condition_max"]
        self.xy_sv_min = params["xy_sigma_min"]

        # Admittance
        self.mass = np.array([params["mass_x"], params["mass_y"]])
        self.damping = np.array([params["damping_x"], params["damping_y"]])
        self.stiffness = np.array([params["stiffness_x"], params["stiffness_y"]])
        self.max_offset = params["max_virtual_offset_m"]
        self.max_vel = params["max_virtual_velocity_mps"]
        self.max_acc = params["max_virtual_acceleration_mps2"]
        self.max_ref_tcp_dev = params["max_reference_tcp_deviation_m"]
        self.max_ref_joint_dev = params["max_reference_joint_deviation_rad"]
        self.max_dt = params["max_integration_dt_sec"]

        # Visualization
        self.enable_meshcat = params["enable_meshcat"]
        self.csv_path = params["csv_path"]

        # TCP frame and joint names
        self.tcp_frame = f"openarm_{self.side}_hand_tcp"
        self.joint_names = [f"openarm_{self.side}_joint{i}" for i in range(1, 8)]

        # -----------------------------------------------------------------
        # Initialize State
        # -----------------------------------------------------------------
        self.state = ShadowState.INIT
        self.last_valid_time = 0.0
        self.last_print_time = 0.0
        self.last_meshcat_time = 0.0
        self.msg_count = 0

        # Current joint data
        self.current_q = np.zeros(7)
        self.current_dq = np.zeros(7)
        self.current_effort = np.zeros(7)
        self.has_velocity = False
        self.has_effort = False
        self.all_joints_present = False

        # Current TCP data
        self.current_tcp_position = np.zeros(3)
        self.current_tcp_rotation = np.zeros(4)  # quaternion xyzw

        # Baseline
        self.baseline = BaselineData(
            effort_baseline=None,
            effort_minus_g_baseline=None,
            reference_q=None,
            reference_tcp_position=None,
            reference_tcp_rotation=None,
            samples_collected=0,
            start_time=None,
            is_valid=False,
        )

        # Import kinematics
        import sys
        sys.path.insert(0, "/ros2_ws/openarm_pinocchio_ik/src")
        from openarm_pinocchio_ik.kinematics import PinocchioModel

        # Initialize Pinocchio model
        urdf = params["urdf_path"]
        self.get_logger().info(f"Loading URDF from: {urdf}")
        self.model = PinocchioModel(urdf, self.side)

        # Verify TCP frame
        if self.model.ee_fid >= len(self.model.model.frames):
            self.get_logger().error(f"TCP frame '{self.tcp_frame}' not found")
            raise RuntimeError(f"TCP frame '{self.tcp_frame}' not found")

        # Admittance calculator
        self.admittance = Admittance2D(
            mass_x=self.mass[0],
            mass_y=self.mass[1],
            damping_x=self.damping[0],
            damping_y=self.damping[1],
            stiffness_x=self.stiffness[0],
            stiffness_y=self.stiffness[1],
            max_virtual_offset_m=self.max_offset,
            max_virtual_velocity_mps=self.max_vel,
            max_virtual_acceleration_mps2=self.max_acc,
            max_integration_dt_sec=self.max_dt,
        )

        # Current gate status
        self.gate_status = GateStatus(ok=False, reason="Initializing")

        # Current force and limit status
        self.current_force_estimated = np.zeros(2)
        self.current_force_external = np.zeros(2)
        self.current_force_used = np.zeros(2)
        self.force_in_deadzone = np.array([False, False])
        self.force_clipped = False
        self.offset_clipped = False

        # CSV writer
        self.csv_file = None
        self.csv_writer = None
        if self.csv_path:
            try:
                self.csv_file = open(self.csv_path, 'w', newline='')
                self.csv_writer = csv.writer(self.csv_file)
                self._write_csv_header()
            except Exception as e:
                self.get_logger().warn(f"Failed to open CSV file: {e}")
                self.csv_file = None

        # MeshCat viewer (created later if enabled)
        self.meshcat_viewer: Optional[MeshcatShadowView] = None

        # -----------------------------------------------------------------
        # Create Subscriptions (NO PUBLISHERS)
        # -----------------------------------------------------------------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.js_sub = self.create_subscription(
            JointState,
            self.joint_state_topic,
            self._on_joint_state,
            qos
        )

        # Timers
        self.shadow_timer = self.create_timer(
            1.0 / self.shadow_rate,
            self._shadow_update
        )
        self.print_timer = self.create_timer(
            1.0 / self.print_rate,
            self._print_summary
        )
        if self.enable_meshcat:
            self.meshcat_timer = self.create_timer(
                1.0 / self.meshcat_rate,
                self._update_meshcat
            )

        # -----------------------------------------------------------------
        # Print Startup Banner
        # -----------------------------------------------------------------
        self._print_startup_banner()

        # Transition state
        self.state = ShadowState.WAITING_FIRST_MSG

    # ---------------------------------------------------------------------
    # Parameter Validation
    # ---------------------------------------------------------------------

    def _get_and_validate_parameters(self) -> dict:
        """Get and validate all parameters."""
        params = {}

        # side
        side = self.get_parameter("side").value
        if side not in ("left", "right"):
            self.get_logger().error(f"Invalid side '{side}'")
            raise ValueError(f"Invalid side: {side}")
        params["side"] = side

        # Simple numeric parameters
        params["joint_state_topic"] = self.get_parameter("joint_state_topic").value
        params["joint_state_timeout_sec"] = float(self.get_parameter("joint_state_timeout_sec").value)
        params["stationary_velocity_threshold_rad_s"] = float(self.get_parameter("stationary_velocity_threshold_rad_s").value)
        params["shadow_rate_hz"] = float(self.get_parameter("shadow_rate_hz").value)
        params["print_rate_hz"] = float(self.get_parameter("print_rate_hz").value)
        params["meshcat_rate_hz"] = float(self.get_parameter("meshcat_rate_hz").value)

        # Baseline
        params["calibration_duration_sec"] = float(self.get_parameter("calibration_duration_sec").value)
        params["calibration_min_samples"] = int(self.get_parameter("calibration_min_samples").value)

        # Force
        params["force_damping"] = float(self.get_parameter("force_damping").value)
        params["force_source"] = self.get_parameter("force_source").value
        if params["force_source"] not in ("raw_zeroed", "minus_g_zeroed"):
            self.get_logger().warn(f"Invalid force_source, using raw_zeroed")
            params["force_source"] = "raw_zeroed"
        params["force_sign_x"] = float(self.get_parameter("force_sign_x").value)
        params["force_sign_y"] = float(self.get_parameter("force_sign_y").value)
        params["force_deadzone_x"] = float(self.get_parameter("force_deadzone_x").value)
        params["force_deadzone_y"] = float(self.get_parameter("force_deadzone_y").value)
        params["max_force_norm"] = float(self.get_parameter("max_force_norm").value)

        # XY reliability
        params["xy_condition_max"] = float(self.get_parameter("xy_condition_max").value)
        params["xy_sigma_min"] = float(self.get_parameter("xy_sigma_min").value)

        # Admittance
        params["mass_x"] = float(self.get_parameter("mass_x").value)
        params["mass_y"] = float(self.get_parameter("mass_y").value)
        params["damping_x"] = float(self.get_parameter("damping_x").value)
        params["damping_y"] = float(self.get_parameter("damping_y").value)
        params["stiffness_x"] = float(self.get_parameter("stiffness_x").value)
        params["stiffness_y"] = float(self.get_parameter("stiffness_y").value)

        # Limits
        params["max_virtual_offset_m"] = float(self.get_parameter("max_virtual_offset_m").value)
        params["max_virtual_velocity_mps"] = float(self.get_parameter("max_virtual_velocity_mps").value)
        params["max_virtual_acceleration_mps2"] = float(self.get_parameter("max_virtual_acceleration_mps2").value)
        params["max_reference_tcp_deviation_m"] = float(self.get_parameter("max_reference_tcp_deviation_m").value)
        params["max_reference_joint_deviation_rad"] = float(self.get_parameter("max_reference_joint_deviation_rad").value)
        params["max_integration_dt_sec"] = float(self.get_parameter("max_integration_dt_sec").value)

        # Visualization
        params["enable_meshcat"] = bool(self.get_parameter("enable_meshcat").value)
        params["csv_path"] = self.get_parameter("csv_path").value

        # URDF path
        urdf = self.get_parameter("urdf_path").value
        if not urdf:
            urdf = (
                "/ros2_ws/install/openarm_description/share/openarm_description/"
                "assets/robot/openarm_v1.0/urdf/example/v1.urdf"
            )
        params["urdf_path"] = urdf

        return params

    # ---------------------------------------------------------------------
    # Startup Banner
    # ---------------------------------------------------------------------

    def _print_startup_banner(self) -> None:
        """Print comprehensive startup information."""
        self.get_logger().info("=" * 70)
        self.get_logger().info("XY ADMITTANCE SHADOW NODE - READ ONLY")
        self.get_logger().info("=" * 70)
        self.get_logger().info("")
        self.get_logger().info("This node publishes NO robot commands.")
        self.get_logger().info("This node does NOT run IK.")
        self.get_logger().info("This node does NOT control any controller.")
        self.get_logger().info("It only computes and visualizes a virtual XY target.")
        self.get_logger().info("")
        self.get_logger().info("Estimated force units; no F/T sensor calibration.")
        self.get_logger().info("")
        self.get_logger().info(f"Side: {self.side}")
        self.get_logger().info(f"TCP Frame: {self.tcp_frame}")
        self.get_logger().info(f"Joint names: {self.joint_names}")
        self.get_logger().info("")
        self.get_logger().info(f"Force source: {self.force_source}")
        self.get_logger().info(f"Force signs: x={self.force_sign[0]}, y={self.force_sign[1]}")
        self.get_logger().info(f"Deadzones: x={self.force_deadzone[0]}, y={self.force_deadzone[1]}")
        self.get_logger().info(f"Max force norm: {self.max_force}")
        self.get_logger().info("")
        self.get_logger().info(f"Admittance M: [{self.mass[0]}, {self.mass[1]}]")
        self.get_logger().info(f"Admittance D: [{self.damping[0]}, {self.damping[1]}]")
        self.get_logger().info(f"Admittance K: [{self.stiffness[0]}, {self.stiffness[1]}]")
        self.get_logger__.info("")
        self.get_logger().info(f"Max virtual offset: {self.max_offset*1000} mm")
        self.get_logger().info(f"Max virtual velocity: {self.max_vel*1000} mm/s")
        self.get_logger__.info(f"Max reference TCP deviation: {self.max_ref_tcp_dev*1000} mm")
        self.get_logger().info(f"Max reference joint deviation: {self.max_ref_joint_dev} rad")
        self.get_logger__.info("")
        self.get_logger().info(f"Calibration: min {self.calib_min_samples} samples, {self.calib_duration} sec")
        self.get_logger__.info(f"XY reliability: cond<={self.xy_cond_max}, sigma_min>={self.xy_sv_min}")
        self.get_logger__.info("")
        self.get_logger__.info(f"MeshCat: {'enabled' if self.enable_meshcat else 'disabled'}")
        self.get_logger__.info(f"CSV: {'enabled' if self.csv_path else 'disabled'}")
        self.get_logger__.info("")
        self.get_logger__.info("=" * 70)

    # ---------------------------------------------------------------------
    # CSV Handling
    # ---------------------------------------------------------------------

    def _write_csv_header(self) -> None:
        """Write CSV header."""
        if self.csv_writer:
            self.csv_writer.writerow([
                "monotonic_time", "ros_time", "state", "gate_ok", "gate_reason",
                "data_age", "tcp_x", "tcp_y", "tcp_z",
                "measured_dx", "measured_dy", "measured_dz",
                "force_estimated_x", "force_estimated_y",
                "force_external_x", "force_external_y",
                "force_used_x", "force_used_y",
                "virtual_dx", "virtual_dy", "virtual_vx", "virtual_vy",
                "virtual_target_x", "virtual_target_y", "virtual_target_z",
                "xy_sigma_min", "xy_condition", "force_clipped", "offset_clipped",
            ])

    def _write_csv_row(
        self,
        now: float,
        ros_time: float,
        gate: GateStatus,
        jac_data: Optional[XYJacobianData],
    ) -> None:
        """Write a row to CSV."""
        if not self.csv_writer or self.csv_file is None:
            return

        try:
            virtual_target = self._get_virtual_target()
            measured_delta = self.current_tcp_position - (
                self.baseline.reference_tcp_position if self.baseline.reference_tcp_position is not None
                else self.current_tcp_position
            )

            self.csv_writer.writerow([
                f"{now:.6f}",
                f"{ros_time:.6f}",
                self.state.value,
                "OK" if gate.ok else "NO",
                gate.reason if gate.reason else "",
                f"{now - self.last_valid_time:.6f}" if self.last_valid_time > 0 else "",
                f"{self.current_tcp_position[0]:.6f}",
                f"{self.current_tcp_position[1]:.6f}",
                f"{self.current_tcp_position[2]:.6f}",
                f"{measured_delta[0]*1000:.4f}",
                f"{measured_delta[1]*1000:.4f}",
                f"{measured_delta[2]*1000:.4f}",
                f"{self.current_force_estimated[0]:.4f}",
                f"{self.current_force_estimated[1]:.4f}",
                f"{self.current_force_external[0]:.4f}",
                f"{self.current_force_external[1]:.4f}",
                f"{self.current_force_used[0]:.4f}",
                f"{self.current_force_used[1]:.4f}",
                f"{self.admittance.state.displacement[0]*1000:.4f}",
                f"{self.admittance.state.displacement[1]*1000:.4f}",
                f"{self.admittance.state.velocity[0]*1000:.4f}",
                f"{self.admittance.state.velocity[1]*1000:.4f}",
                f"{virtual_target[0]:.6f}",
                f"{virtual_target[1]:.6f}",
                f"{virtual_target[2]:.6f}",
                f"{jac_data.sigma_min:.6f}" if jac_data else "",
                f"{jac_data.condition_number:.4f}" if jac_data else "",
                "1" if self.force_clipped else "0",
                "1" if self.offset_clipped else "0",
            ])
            self.csv_file.flush()
        except Exception:
            pass  # Don't crash on CSV write errors

    # ---------------------------------------------------------------------
    # Joint State Callback
    # ---------------------------------------------------------------------

    def _on_joint_state(self, msg: JointState) -> None:
        """Process incoming joint state message."""
        now = time.monotonic()

        # Basic validation
        if len(msg.name) != len(msg.position):
            return
        if msg.velocity and len(msg.velocity) != len(msg.name):
            return
        if msg.effort and len(msg.effort) != len(msg.name):
            return

        # Check for finite values
        if any(not math.isfinite(v) for v in msg.position):
            return

        self.has_velocity = bool(msg.velocity) and all(
            math.isfinite(v) for v in msg.velocity
        )
        self.has_effort = bool(msg.effort) and all(
            math.isfinite(v) for v in msg.effort
        )

        # Extract joint data
        try:
            q = np.array([
                msg.position[msg.name.index(n)]
                for n in self.joint_names
            ], dtype=float)

            dq = np.array([
                msg.velocity[msg.name.index(n)]
                for n in self.joint_names
            ], dtype=float) if self.has_velocity else np.zeros(7)

            effort = np.array([
                msg.effort[msg.name.index(n)]
                for n in self.joint_names
            ], dtype=float) if self.has_effort else np.zeros(7)

            self.all_joints_present = True
        except ValueError:
            self.all_joints_present = False
            return

        # Update current state
        self.current_q = q
        self.current_dq = dq
        self.current_effort = effort
        self.last_valid_time = now
        self.msg_count += 1

        # Compute FK
        try:
            self.current_tcp_position, self.current_tcp_rotation = self.model.fk(q)
        except Exception:
            return

        # Update state
        if self.state == ShadowState.WAITING_FIRST_MSG:
            self.state = ShadowState.CALIBRATING
            self.baseline.start_time = now

        # Calibration
        if self.state == ShadowState.CALIBRATING:
            self._perform_calibration_step(now, dq)

        # Recover from stale
        if self.state == ShadowState.STALE:
            self.get_logger().info("Data received, recovering from stale")
            self.state = ShadowState.READY if self.baseline.is_valid else ShadowState.CALIBRATING
            if self.state == ShadowState.CALIBRATING:
                self.baseline.start_time = now

    # ---------------------------------------------------------------------
    # Calibration
    # ---------------------------------------------------------------------

    def _perform_calibration_step(self, now: float, dq: np.ndarray) -> None:
        """Perform one step of baseline calibration."""
        if not self.has_velocity:
            return

        # Check if stationary
        max_vel = np.max(np.abs(dq))
        if max_vel > self.stationary_threshold:
            # Moving - reset calibration
            if self.baseline.samples_collected > 0:
                self.get_logger().info("Arm moved during calibration, restarting")
            self.baseline.samples_collected = 0
            self.baseline.start_time = now
            return

        # Accumulate baseline
        if self.baseline.effort_baseline is None:
            self.baseline.effort_baseline = np.zeros(7)
            self.baseline.effort_minus_g_baseline = np.zeros(7)

        g = self.model.gravity(self.current_q)
        n = self.baseline.samples_collected
        self.baseline.effort_baseline = (
            n * self.baseline.effort_baseline + self.current_effort
        ) / (n + 1)
        self.baseline.effort_minus_g_baseline = (
            n * self.baseline.effort_minus_g_baseline + (self.current_effort - g)
        ) / (n + 1)

        self.baseline.samples_collected += 1

        # Check completion
        elapsed = now - (self.baseline.start_time or now)
        if (elapsed >= self.calib_duration and
            self.baseline.samples_collected >= self.calib_min_samples):
            self.baseline.is_valid = True
            self.baseline.reference_q = self.current_q.copy()
            self.baseline.reference_tcp_position = self.current_tcp_position.copy()
            self.baseline.reference_tcp_rotation = self.current_tcp_rotation.copy()
            self.state = ShadowState.READY
            self.get_logger().info("")
            self.get_logger().info("=" * 50)
            self.get_logger().info("Baseline calibration COMPLETE")
            self.get_logger__.info("=" * 50)
            self.get_logger__.info(f"Samples collected: {self.baseline.samples_collected}")
            self.get_logger__.info(f"Elapsed time: {elapsed:.2f} sec")
            self.get_logger__.info("")
            self.get_logger__.info("Reference TCP (m):")
            self.get_logger__.info(f"  x={self.baseline.reference_tcp_position[0]:.4f}")
            self.get_logger__.info(f"  y={self.baseline.reference_tcp_position[1]:.4f}")
            self.get_logger__.info(f"  z={self.baseline.reference_tcp_position[2]:.4f}")
            self.get_logger__.info("=" * 50)

            # Initialize MeshCat if enabled
            if self.enable_meshcat:
                self._init_meshcat()

    # ---------------------------------------------------------------------
    # Jacobian Computation
    # ---------------------------------------------------------------------

    def _compute_xy_jacobian(self, q: np.ndarray) -> Optional[XYJacobianData]:
        """Compute XY Jacobian and reliability metrics."""
        try:
            full_q = self.model._full_q(q)

            J = pin.computeFrameJacobian(
                self.model.model,
                self.model.data,
                full_q,
                self.model.ee_fid,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )

            J7 = J[:, self.model.q_idx]  # 6x7
            Jv = J7[:3, :]  # 3x7
            J_xy = Jv[:2, :]  # 2x7

            sv = np.linalg.svd(J_xy, compute_uv=False)

            sigma_min = np.min(sv)
            sigma_max = np.max(sv)

            if sigma_min > 1e-10:
                cond = sigma_max / sigma_min
            else:
                cond = float('inf')

            is_reliable = (
                np.all(np.isfinite(sv)) and
                sigma_min >= self.xy_sv_min and
                cond <= self.xy_cond_max
            )

            return XYJacobianData(
                J_xy=J_xy,
                singular_values=sv,
                sigma_min=sigma_min,
                condition_number=cond,
                is_reliable=is_reliable,
            )
        except Exception:
            return None

    # ---------------------------------------------------------------------
    # Force Estimation
    # ---------------------------------------------------------------------

    def _estimate_xy_force(
        self,
        jac: XYJacobianData,
    ) -> np.ndarray:
        """Estimate XY force from current effort using baseline.

        Returns:
            2D force vector [Fx, Fy] before sign correction.
        """
        if not self.baseline.is_valid or self.baseline.effort_baseline is None:
            return np.zeros(2)

        # Select force source
        if self.force_source == "raw_zeroed":
            tau_zeroed = self.current_effort - self.baseline.effort_baseline
        else:  # minus_g_zeroed
            if self.baseline.effort_minus_g_baseline is None:
                return np.zeros(2)
            g = self.model.gravity(self.current_q)
            tau_zeroed = (self.current_effort - g) - self.baseline.effort_minus_g_baseline

        # Damped least squares
        JJt = jac.J_xy @ jac.J_xy.T + (self.force_damping ** 2) * np.eye(2)
        J_tau = jac.J_xy @ tau_zeroed

        try:
            F_xy = np.linalg.solve(JJt, J_tau)
        except np.linalg.LinAlgError:
            F_xy = np.zeros(2)

        return F_xy

    # ---------------------------------------------------------------------
    # Safety Gate
    # ---------------------------------------------------------------------

    def _check_gate(self, jac: Optional[XYJacobianData]) -> GateStatus:
        """Check all safety gate conditions."""
        # Baseline valid
        if not self.baseline.is_valid:
            return GateStatus(ok=False, reason="Baseline not valid")

        # Data fresh
        if time.monotonic() - self.last_valid_time > self.joint_state_timeout:
            return GateStatus(ok=False, reason="Joint state timeout")

        # Joints present
        if not self.all_joints_present:
            return GateStatus(ok=False, reason="Missing joints")

        # Jacobian valid
        if jac is None:
            return GateStatus(ok=False, reason="Jacobian computation failed")

        # Jacobian shape
        if jac.J_xy.shape != (2, 7):
            return GateStatus(ok=False, reason="Jacobian shape invalid")

        # Singular values finite
        if not np.all(np.isfinite(jac.singular_values)):
            return GateStatus(ok=False, reason="Singular values not finite")

        # Reliability
        if not jac.is_reliable:
            return GateStatus(
                ok=False,
                reason=f"XY unreliable: sigma_min={jac.sigma_min:.4f}, cond={jac.condition_number:.2f}"
            )

        # Reference TCP deviation
        if self.baseline.reference_tcp_position is not None:
            tcp_dev = np.linalg.norm(self.current_tcp_position - self.baseline.reference_tcp_position)
            if tcp_dev > self.max_ref_tcp_dev:
                return GateStatus(
                    ok=False,
                    reason=f"TCP deviation {tcp_dev*1000:.1f} mm > {self.max_ref_tcp_dev*1000:.0f} mm"
                )

        # Reference joint deviation
        if self.baseline.reference_q is not None:
            joint_dev = np.max(np.abs(self.current_q - self.baseline.reference_q))
            if joint_dev > self.max_ref_joint_dev:
                return GateStatus(
                    ok=False,
                    reason=f"Joint deviation {joint_dev:.2f} rad > {self.max_ref_joint_dev:.2f} rad"
                )

        return GateStatus(ok=True, reason=None)

    # ---------------------------------------------------------------------
    # Admittance Update (Shadow Timer)
    # ---------------------------------------------------------------------

    def _shadow_update(self) -> None:
        """Main shadow update loop."""
        now = time.monotonic()

        # State checks
        if self.state == ShadowState.WAITING_FIRST_MSG:
            return

        if self.state == ShadowState.CALIBRATING:
            return

        if not self.all_joints_present or not self.has_effort:
            self.gate_status = GateStatus(ok=False, reason="Missing joint data")
            self.admittance.reset()
            return

        # Compute Jacobian
        jac = self._compute_xy_jacobian(self.current_q)

        # Check gate
        self.gate_status = self._check_gate(jac)

        if not self.gate_status.ok:
            self.state = ShadowState.PAUSED
            self.admittance.reset()
            return

        self.state = ShadowState.READY

        # Estimate force
        F_estimated = self._estimate_xy_force(jac)
        self.current_force_estimated = F_estimated

        # Apply sign
        F_external = self.force_sign * F_estimated
        self.current_force_external = F_external

        # Apply deadzone
        F_used = np.zeros(2)
        self.force_in_deadzone = np.abs(F_external) <= self.force_deadzone
        for i in range(2):
            if self.force_in_deadzone[i]:
                F_used[i] = 0.0
            else:
                F_used[i] = np.sign(F_external[i]) * (np.abs(F_external[i]) - self.force_deadzone[i])

        # Apply force limit
        force_norm = np.linalg.norm(F_used)
        self.force_clipped = force_norm > self.max_force
        if self.force_clipped:
            F_used *= self.max_force / force_norm

        self.current_force_used = F_used

        # Compute dt
        if self.admittance.last_update_time is None:
            dt = None
        else:
            dt = now - self.admittance.last_update_time

        self.admittance.last_update_time = now

        # Update admittance
        self.offset_clipped = False
        adm_state = self.admittance.update(F_used, dt)
        if np.linalg.norm(adm_state.displacement) >= self.max_offset * 0.999:
            self.offset_clipped = True

        # Write CSV
        if self.csv_writer:
            ros_time = self.last_valid_time
            self._write_csv_row(now, ros_time, self.gate_status, jac)

    # ---------------------------------------------------------------------
    # Virtual Target
    # ---------------------------------------------------------------------

    def _get_virtual_target(self) -> np.ndarray:
        """Get virtual target world position."""
        if self.baseline.reference_tcp_position is None:
            return self.current_tcp_position.copy()

        virtual = self.baseline.reference_tcp_position.copy()
        virtual[0] += self.admittance.state.displacement[0]
        virtual[1] += self.admittance.state.displacement[1]
        # Z fixed at reference
        return virtual

    # ---------------------------------------------------------------------
    # MeshCat Visualization
    # ---------------------------------------------------------------------

    def _init_meshcat(self) -> None:
        """Initialize MeshCat viewer."""
        if meshcat is None:
            self.get_logger().warn("MeshCat not installed, skipping visualization")
            return

        try:
            self.meshcat_viewer = MeshcatShadowView()
            self.meshcat_viewer.setup()
            self.get_logger().info(f"MeshCat viewer URL: {self.meshcat_viewer.get_url()}")
        except Exception as e:
            self.get_logger().warn(f"Failed to setup MeshCat: {e}")
            self.meshcat_viewer = None

    def _update_meshcat(self) -> None:
        """Update MeshCat visualization."""
        if self.meshcat_viewer is None:
            return

        now = time.monotonic()
        if now - self.last_meshcat_time < 1.0 / self.meshcat_rate:
            return
        self.last_meshcat_time = now

        try:
            virtual_target = self._get_virtual_target()
            ref_tcp = self.baseline.reference_tcp_position if self.baseline.reference_tcp_position is not None else self.current_tcp_position

            self.meshcat_viewer.update(
                reference=ref_tcp,
                actual=self.current_tcp_position,
                virtual=virtual_target,
                state=self.state,
            )
        except Exception:
            pass  # Don't crash on visualization errors

    # ---------------------------------------------------------------------
    # Terminal Output
    # ---------------------------------------------------------------------

    def _print_summary(self) -> None:
        """Print periodic summary."""
        if self.state == ShadowState.WAITING_FIRST_MSG:
            self.get_logger().info("Waiting for first /joint_states message...")
            return

        now = time.monotonic()
        age = now - self.last_valid_time if self.last_valid_time > 0 else 0

        # Calibration progress
        if self.state == ShadowState.CALIBRATING:
            elapsed = now - (self.baseline.start_time or now)
            max_dq = np.max(np.abs(self.current_dq)) if self.has_velocity else 0
            self.get_logger().info(
                f"CALIBRATING: elapsed={elapsed:.1f}/{self.calib_duration} sec, "
                f"samples={self.baseline.samples_collected}/{self.calib_min_samples}, "
                f"max_dq={max_dq:.4f}"
            )
            return

        # Paused state
        if self.state == ShadowState.PAUSED:
            self.get_logger().warn(f"PAUSED: {self.gate_status.reason}")
            return

        # Stale state
        if self.state == ShadowState.STALE:
            self.get_logger().warn("STALE: No valid joint states")
            return

        # Normal output
        virtual_target = self._get_virtual_target()
        ref_tcp = self.baseline.reference_tcp_position if self.baseline.reference_tcp_position is not None else self.current_tcp_position
        measured_delta = self.current_tcp_position - ref_tcp

        # Get Jacobian data for display
        jac = self._compute_xy_jacobian(self.current_q)

        print()
        print("=" * 70)
        print(f"XY ADMITTANCE SHADOW - {self.state.value.upper()}")
        print("=" * 70)
        print(f"Data age: {age:.3f} s")
        print(f"Baseline: {'VALID' if self.baseline.is_valid else 'CALIBRATING'}")
        print(f"Gate: {'OK' if self.gate_status.ok else 'NO'}")
        if jac and jac.is_reliable:
            print(f"XY condition: {jac.condition_number:.2f} | sigma_min: {jac.sigma_min:.4f} | Reliable: YES")
        elif jac:
            print(f"XY condition: {jac.condition_number:.2f} | sigma_min: {jac.sigma_min:.4f} | Reliable: NO")
        print()

        print(f"TCP world (m):")
        print(f"  x={self.current_tcp_position[0]:.4f}")
        print(f"  y={self.current_tcp_position[1]:.4f}")
        print(f"  z={self.current_tcp_position[2]:.4f}")
        print()

        print(f"Measured TCP delta (mm):")
        print(f"  dx={measured_delta[0]*1000:.2f}")
        print(f"  dy={measured_delta[1]*1000:.2f}")
        print(f"  dz={measured_delta[2]*1000:.2f}")
        print()

        print(f"Estimated force before sign:")
        print(f"  Fx={self.current_force_estimated[0]:.2f}")
        print(f"  Fy={self.current_force_estimated[1]:.2f}")
        print()

        print(f"External force after sign:")
        print(f"  Fx={self.current_force_external[0]:.2f}")
        print(f"  Fy={self.current_force_external[1]:.2f}")
        print()

        print(f"Force after deadzone/clip:")
        print(f"  Fx={self.current_force_used[0]:.2f}")
        print(f"  Fy={self.current_force_used[1]:.2f}")
        print(f"  deadzone_x={'YES' if self.force_in_deadzone[0] else 'NO'}")
        print(f"  deadzone_y={'YES' if self.force_in_deadzone[1] else 'NO'}")
        print(f"  clipped={'YES' if self.force_clipped else 'NO'}")
        print()

        print(f"Virtual admittance:")
        print(f"  dx={self.admittance.state.displacement[0]*1000:.2f} mm")
        print(f"  dy={self.admittance.state.displacement[1]*1000:.2f} mm")
        print(f"  vx={self.admittance.state.velocity[0]*1000:.2f} mm/s")
        print(f"  vy={self.admittance.state.velocity[1]*1000:.2f} mm/s")
        print()

        print(f"Virtual target world (m):")
        print(f"  x={virtual_target[0]:.4f}")
        print(f"  y={virtual_target[1]:.4f}")
        print(f"  z={virtual_target[2]:.4f}")
        print()

        print(f"Limits:")
        print(f"  force_clip={'YES' if self.force_clipped else 'NO'}")
        print(f"  offset_clip={'YES' if self.offset_clipped else 'NO'}")
        print("=" * 70)

    # ---------------------------------------------------------------------
    # Shutdown
    # ---------------------------------------------------------------------

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'csv_file') and self.csv_file is not None:
            try:
                self.csv_file.close()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# MeshCat Visualization
# -----------------------------------------------------------------------------

class MeshCatShadowView:
    """Simple MeshCat viewer for shadow mode visualization."""

    def __init__(self) -> None:
        """Initialize viewer (MeshCat setup deferred to setup())."""
        self.vis = None
        self.url = ""

    def setup(self) -> None:
        """Setup MeshCat viewer."""
        if meshcat is None:
            return

        self.vis = meshcat.Visualizer(zurl_url="http://127.0.0.1:7000")

        # Create reference TCP point (green)
        self.vis["reference_tcp"].set_object(
            meshcat.geometry.Sphere(0.015), meshcat.geometry.MeshLambertMaterial(color=0x00ff00)
        )

        # Create actual TCP point (blue)
        self.vis["actual_tcp"].set_object(
            meshcat.geometry.Sphere(0.015), meshcat.geometry.MeshLambertMaterial(color=0x0000ff)
        )

        # Create virtual target TCP point (red)
        self.vis["virtual_target_tcp"].set_object(
            meshcat.geometry.Sphere(0.015), meshcat.geometry.MeshLambertMaterial(color=0xff0000)
        )

        # Create line segments
        self.vis["reference_to_virtual"].set_object(
            meshcat.geometry.Line(
                np.array([[0, 0, 0], [0, 0, 0]]).T,
                meshcat.geometry.MeshLambertMaterial(color=0x00ff00)
            )
        )
        self.vis["actual_to_virtual"].set_object(
            meshcat.geometry.Line(
                np.array([[0, 0, 0], [0, 0, 0]]).T,
                meshcat.geometry.MeshLambertMaterial(color=0xff00ff)
            )
        )

        self.url = self.vis.url()
        if self.url:
            print(f"MeshCat URL: {self.url}")

    def get_url(self) -> str:
        """Get MeshCat viewer URL."""
        return self.url

    def update(
        self,
        reference: np.ndarray,
        actual: np.ndarray,
        virtual: np.ndarray,
        state: ShadowState,
    ) -> None:
        """Update visualization.

        Args:
            reference: Reference TCP position [x, y, z].
            actual: Current actual TCP position [x, y, z].
            virtual: Virtual target position [x, y, z].
            state: Current shadow state.
        """
        if self.vis is None:
            return

        try:
            # Update sphere positions
            self.vis["reference_tcp"].set_transform(
                meshcat.transformations.translation_matrix(reference)
            )
            self.vis["actual_tcp"].set_transform(
                meshcat.transformations.translation_matrix(actual)
            )
            self.vis["virtual_target_tcp"].set_transform(
                meshcat.transformations.translation_matrix(virtual)
            )

            # Update line segments
            ref_to_virtual = np.vstack([reference, virtual]).T
            self.vis["reference_to_virtual"].set_object(
                meshcat.geometry.Line(
                    ref_to_virtual,
                    meshcat.geometry.MeshLambertMaterial(color=0x00ff00)
                )
            )

            actual_to_virtual = np.vstack([actual, virtual]).T
            self.vis["actual_to_virtual"].set_object(
                meshcat.geometry.Line(
                    actual_to_virtual,
                    meshcat.geometry.MeshLambertMaterial(color=0xff00ff)
                )
            )

            # Optional: State-based coloring could go here
            # but text display in MeshCat is unreliable

        except Exception:
            pass  # Don't crash on visualization errors


def main() -> None:
    """Main entry point."""
    rclpy.init()

    node = XYAdmittanceShadowNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        # Check if this is the shutdown-related RuntimeError
        if "Unable to convert call argument to Python object" in str(e):
            # This is expected during shutdown
            pass
        else:
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
