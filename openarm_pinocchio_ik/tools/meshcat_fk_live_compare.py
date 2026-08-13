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

"""Read-only ROS2 + Pinocchio + MeshCat live trajectory comparison visualizer.

This script subscribes to:
  - /joint_states (actual robot feedback)
  - /right_joint_trajectory_controller/joint_trajectory (target trajectory)
  - /left_joint_trajectory_controller/joint_trajectory (target trajectory)

It displays two robot models in MeshCat:
  - "openarm_target": The commanded/target trajectory
  - "openarm_actual": The actual robot state from /joint_states

This is a READ-ONLY visualizer. It does NOT send any commands to the robot.

Example:
    python3 tools/meshcat_fk_live_compare.py --mode side_by_side --rate 10
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from typing import Any

import numpy as np

try:
    import pinocchio as pin
    from pinocchio.visualize import MeshcatVisualizer
except ImportError:
    print("ERROR: Pinocchio not installed. Install with: pip install pinocchio")
    raise SystemExit(1)

try:
    import meshcat
except ImportError:
    print("ERROR: MeshCat not installed. Install with: pip install meshcat")
    raise SystemExit(1)

import rclpy
from builtin_interfaces.msg import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory


def package_uri_to_path(uri: str) -> str:
    """Convert package:// URIs to absolute file paths.

    Args:
        uri: A package URI like "package://openarm_description/assets/..."

    Returns:
        Absolute file path, or original URI if resolution fails.
    """
    if not uri.startswith("package://"):
        return uri

    try:
        from ament_index_python.packages import get_package_share_directory

        # Remove "package://" prefix
        without_prefix = uri[len("package://") :]
        # Split on first '/' to get package name
        parts = without_prefix.split("/", 1)
        if len(parts) != 2:
            return uri

        package_name, relative_path = parts
        pkg_path = get_package_share_directory(package_name)
        return os.path.join(pkg_path, relative_path)
    except Exception:
        return uri


class TrajectoryBuffer:
    """Stores and interpolates a JointTrajectory for smooth playback."""

    def __init__(self) -> None:
        self.points: list[tuple[float, dict[str, float]]] = []  # (time_from_start, joint_map)
        self.joint_names: list[str] = []
        self.start_time: float | None = None  # ROS time in seconds
        self.last_q: dict[str, float] = {}

    def set_trajectory(self, traj: JointTrajectory, ros_time: float) -> None:
        """Store a new trajectory and record its start time."""
        self.points = []
        self.joint_names = list(traj.joint_names)
        self.start_time = ros_time

        for pt in traj.points:
            t = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
            if len(pt.positions) != len(traj.joint_names):
                continue
            joint_map = dict(zip(traj.joint_names, pt.positions))
            self.points.append((t, joint_map))

    def interpolate(self, ros_time: float, joint_names: list[str]) -> dict[str, float]:
        """Get interpolated joint positions at current ROS time.

        Args:
            ros_time: Current ROS time in seconds
            joint_names: All joint names to output (may include uncontrolled joints)

        Returns:
            Dict mapping joint name to position (rad)
        """
        if self.start_time is None or not self.points:
            return {}

        elapsed = ros_time - self.start_time
        if elapsed < 0:
            return {}

        # Find surrounding trajectory points
        before: tuple[float, dict[str, float]] | None = None
        after: tuple[float, dict[str, float]] | None = None

        for t, joint_map in self.points:
            if t <= elapsed:
                before = (t, joint_map)
            if t >= elapsed and after is None:
                after = (t, joint_map)
                break

        # If before the first point, use first point
        if before is None and after is not None:
            return {j: after[1].get(j, 0.0) for j in joint_names}

        # If after the last point, use last point (hold final)
        if after is None and before is not None:
            return {j: before[1].get(j, 0.0) for j in joint_names}

        # If exactly on a point
        if before is not None and after is not None and before[0] == after[0]:
            return {j: before[1].get(j, 0.0) for j in joint_names}

        # Linear interpolation
        if before is not None and after is not None:
            t0, q0 = before
            t1, q1 = after
            alpha = (elapsed - t0) / (t1 - t0) if t1 > t0 else 0.0
            result: dict[str, float] = {}
            for j in joint_names:
                v0 = q0.get(j, 0.0)
                v1 = q1.get(j, 0.0)
                result[j] = v0 + alpha * (v1 - v0)
            return result

        return {}


class MeshCatVisualizer(Node):
    """Read-only node that visualizes target vs actual robot states in MeshCat."""

    def __init__(
        self,
        mode: str = "side_by_side",
        rate: int = 30,
        robot_description_node: str = "/robot_state_publisher",
        side: str = "both",
        separation: float = 1.0,
        debug_values: bool = False,
    ) -> None:
        super().__init__("meshcat_fk_live_compare")

        self.mode = mode
        self.rate = rate
        self.side = side
        self.separation = separation
        self.debug_values = debug_values

        # Note: parameter client is created in fetch_robot_description() function
        # We don't store it as instance variable

        # Subscriptions
        self.js_sub = self.create_subscription(
            JointState, "/joint_states", self._on_joint_states, qos_profile_sensor_data
        )
        self.right_traj_sub = self.create_subscription(
            JointTrajectory,
            "/right_joint_trajectory_controller/joint_trajectory",
            lambda msg: self._on_trajectory(msg, "right"),
            10,
        )
        self.left_traj_sub = self.create_subscription(
            JointTrajectory,
            "/left_joint_trajectory_controller/joint_trajectory",
            lambda msg: self._on_trajectory(msg, "left"),
            10,
        )

        # State storage
        self.actual_q: dict[str, float] = {}
        self.target_q: dict[str, float] = {}
        self.right_traj = TrajectoryBuffer()
        self.left_traj = TrajectoryBuffer()
        self.first_js_received = False
        self.initial_q: dict[str, float] = {}

        # Diagnostic mode
        self.js_count = 0
        self.right_traj_count = 0
        self.left_traj_count = 0
        self.actual_display_count = 0
        self.target_display_count = 0
        self.last_debug_log_time = 0.0

        # Timing tracking
        self.start_js_time = 0.0
        self.last_js_time = 0.0

        # Model placeholders (loaded after robot_description)
        self.model: pin.Model | None = None
        self.data: pin.ModelData | None = None
        self.visual_model: pin.GeometryModel | None = None
        self.collision_model: pin.GeometryModel | None = None
        self.visual_data: pin.GeometryData | None = None
        self.collision_data: pin.GeometryData | None = None

        # MeshCat viewer
        self.vis: meshcat.Visualizer | None = None

        # Pinocchio MeshcatVisualizer instances for target and actual
        self.target_viz: MeshcatVisualizer | None = None
        self.actual_viz: MeshcatVisualizer | None = None

        # Joint names for full bimanual model
        self.all_joint_names: list[str] = []

        # Required joint names for initialization (only arm joints, not finger/mimic)
        self.required_joint_names: list[str] = []
        self._init_required_joints(side)

        # TCP frame names
        self.tcp_frames: dict[str, str] = {
            "right": "openarm_right_hand_tcp",
            "left": "openarm_left_hand_tcp",
        }
        self.tcp_fids: dict[str, int] = {}

        # Timing
        self.last_log_time = 0.0

        self.get_logger().info(
            f"READ-ONLY VISUALIZER: no command will be sent to the robot."
        )

    def _init_required_joints(self, side: str) -> None:
        """Initialize required joint names for initialization check.

        Only arm joints are required, not finger or mimic joints.

        Args:
            side: 'both', 'right', or 'left'
        """
        if side == "both":
            self.required_joint_names = [
                "openarm_left_joint1",
                "openarm_left_joint2",
                "openarm_left_joint3",
                "openarm_left_joint4",
                "openarm_left_joint5",
                "openarm_left_joint6",
                "openarm_left_joint7",
                "openarm_right_joint1",
                "openarm_right_joint2",
                "openarm_right_joint3",
                "openarm_right_joint4",
                "openarm_right_joint5",
                "openarm_right_joint6",
                "openarm_right_joint7",
            ]
        elif side == "right":
            self.required_joint_names = [
                "openarm_right_joint1",
                "openarm_right_joint2",
                "openarm_right_joint3",
                "openarm_right_joint4",
                "openarm_right_joint5",
                "openarm_right_joint6",
                "openarm_right_joint7",
            ]
        else:  # left
            self.required_joint_names = [
                "openarm_left_joint1",
                "openarm_left_joint2",
                "openarm_left_joint3",
                "openarm_left_joint4",
                "openarm_left_joint5",
                "openarm_left_joint6",
                "openarm_left_joint7",
            ]

    def load_robot_model(self, urdf_path: str) -> bool:
        """Load Pinocchio model from URDF and setup MeshCat.

        Args:
            urdf_path: Path to URDF file

        Returns:
            True if successful
        """
        try:
            # Build Pinocchio model (mimic=True for finger joints)
            try:
                self.model = pin.buildModelFromUrdf(urdf_path, True)
            except TypeError:
                self.model = pin.buildModelFromUrdf(urdf_path)
            self.data = self.model.createData()

            # Get all joint names
            self.all_joint_names = [self.model.names[i] for i in range(1, self.model.njoints)]

            # Load both visual and collision geometry models for MeshCat
            self.visual_model = pin.buildGeomFromUrdf(
                self.model, urdf_path, pin.GeometryType.VISUAL
            )
            self.collision_model = pin.buildGeomFromUrdf(
                self.model, urdf_path, pin.GeometryType.COLLISION
            )
            self.visual_data = self.visual_model.createData()
            self.collision_data = self.collision_model.createData()

            # Resolve mesh file paths (package:// -> absolute paths)
            for go in self.visual_model.geometryObjects:
                go.meshPath = package_uri_to_path(go.meshPath)
            for go in self.collision_model.geometryObjects:
                go.meshPath = package_uri_to_path(go.meshPath)

            # Find TCP frame IDs
            for side_name in ["right", "left"]:
                tcp_name = self.tcp_frames[side_name]
                try:
                    fid = self.model.getFrameId(tcp_name)
                    self.tcp_fids[side_name] = fid
                except ValueError:
                    # Fallback to link7
                    fallback = f"openarm_{side_name}_link7"
                    try:
                        fid = self.model.getFrameId(fallback)
                        self.tcp_fids[side_name] = fid
                        self.tcp_frames[side_name] = fallback
                        self.get_logger().warn(
                            f"TCP frame {tcp_name} not found, using {fallback}"
                        )
                    except ValueError:
                        self.tcp_fids[side_name] = -1
                        self.get_logger().warn(f"No TCP frame found for {side_name}")

            self.get_logger().info(f"Loaded model with {len(self.all_joint_names)} joints")

            # Setup MeshCat
            self._setup_meshcat()

            return True

        except Exception as e:
            self.get_logger().error(f"Failed to load robot model: {e}")
            return False

    def _setup_meshcat(self) -> None:
        """Initialize MeshCat viewer with two robot instances."""
        if self.model is None or self.visual_model is None or self.collision_model is None:
            return

        # Create a single shared MeshCat viewer (no auto-open browser)
        self.vis = meshcat.Visualizer()

        # Create MeshcatVisualizer for target robot
        self.target_viz = MeshcatVisualizer(
            self.model,
            collision_model=self.collision_model,
            visual_model=self.visual_model,
        )
        self.target_viz.initViewer(viewer=self.vis, open=False)
        self.target_viz.loadViewerModel(rootNodeName="openarm_target")

        # Create MeshcatVisualizer for actual robot
        self.actual_viz = MeshcatVisualizer(
            self.model,
            collision_model=self.collision_model,
            visual_model=self.visual_model,
        )
        self.actual_viz.initViewer(viewer=self.vis, open=False)
        self.actual_viz.loadViewerModel(rootNodeName="openarm_actual")

        # Apply offset for side_by_side mode
        if self.mode == "side_by_side":
            import meshcat.transformations as tf

            # Offset target to the left
            target_T = tf.translation_matrix([-self.separation / 2, 0, 0])
            self.vis["openarm_target"].set_transform(target_T)

            # Offset actual to the right
            actual_T = tf.translation_matrix([self.separation / 2, 0, 0])
            self.vis["openarm_actual"].set_transform(actual_T)

        # Print MeshCat URL
        url = self.vis.url()
        self.get_logger().info(f"MeshCat viewer URL: {url}")
        self.get_logger().info(
            "Access via VS Code Ports (PORTS tab) or SSH port forwarding."
        )

    def _on_joint_states(self, msg: JointState) -> None:
        """Handle incoming /joint_states messages."""
        ros_time = self.get_clock().now().nanoseconds / 1e9

        # Store by name (not index)
        joint_map = dict(zip(msg.name, msg.position))

        # Always update actual_q with new joint values
        for name, pos in joint_map.items():
            self.actual_q[name] = pos

        self.js_count += 1

        # Track timing
        if self.start_js_time == 0.0:
            self.start_js_time = ros_time
        self.last_js_time = ros_time

        # Initialize target and store initial state on first complete message
        if not self.first_js_received:
            # Check if we have all required arm joints (not all model joints)
            expected = set(self.required_joint_names)
            received = set(joint_map.keys())

            # First time receiving any JointState: log once
            if not hasattr(self, "_logged_first_js"):
                self.get_logger().info(
                    f"Received JointState with {len(msg.name)} names: {msg.name[:10]}"
                    + ("..." if len(msg.name) > 10 else "")
                )
                self._logged_first_js = True

            if expected.issubset(received):
                self.first_js_received = True
                # Store initial state for fallback reference
                self.initial_q = dict(joint_map)
                # Initialize target with a DEEP COPY of actual state
                self.target_q = self.actual_q.copy()
                self.get_logger().info(
                    f"Received required joint states; visualization initialized."
                )
            else:
                # Log missing required joints at most once per second
                ros_time = self.get_clock().now().nanoseconds / 1e9
                if not hasattr(self, "_last_missing_log_time"):
                    self._last_missing_log_time = 0.0
                if ros_time - self._last_missing_log_time >= 1.0:
                    missing = sorted(expected - received)
                    self.get_logger().info(
                        f"waiting for required /joint_states: "
                        f"received={sorted(received)[:5]}... "
                        f"missing_required={missing}"
                    )
                    self._last_missing_log_time = ros_time

    def _on_trajectory(self, msg: JointTrajectory, side: str) -> None:
        """Handle incoming JointTrajectory messages."""
        ros_time = self.get_clock().now().nanoseconds / 1e9
        if side == "right":
            self.right_traj.set_trajectory(msg, ros_time)
            self.right_traj_count += 1
        else:
            self.left_traj.set_trajectory(msg, ros_time)
            self.left_traj_count += 1

    def _update_target_from_trajectories(self, ros_time: float) -> None:
        """Update target_q by interpolating trajectories.

        Only updates joints that have active trajectory data.
        Joints without trajectory data keep their previous target_q values.
        """
        # Get interpolated values from both trajectories
        # Pass the joint names from the trajectory, not all_names
        right_q = self.right_traj.interpolate(
            ros_time, self.right_traj.joint_names
        )
        left_q = self.left_traj.interpolate(ros_time, self.left_traj.joint_names)

        # Update target_q with trajectory values
        # Only update joints that have trajectory data
        for j, val in right_q.items():
            self.target_q[j] = val
        for j, val in left_q.items():
            self.target_q[j] = val

    def _q_dict_to_vector(self, q_dict: dict[str, float]) -> np.ndarray:
        """Convert joint name dict to Pinocchio configuration vector."""
        if self.model is None:
            return np.array([])

        q = pin.neutral(self.model)
        for i in range(1, self.model.njoints):
            name = self.model.names[i]
            jid = self.model.getJointId(name)
            idx = self.model.idx_qs[jid]
            if idx >= 0 and name in q_dict:
                q[idx] = q_dict[name]
        return q

    def _compute_tcp_error(
        self, side: str, q_target_vec: np.ndarray, q_actual_vec: np.ndarray
    ) -> float:
        """Compute TCP position error for one arm (mm)."""
        if (
            self.model is None
            or self.data is None
            or side not in self.tcp_fids
            or self.tcp_fids[side] < 0
        ):
            return 0.0

        fid = self.tcp_fids[side]

        # FK for target
        pin.framesForwardKinematics(self.model, self.data, q_target_vec)
        target_pos = self.data.oMf[fid].translation.copy()

        # FK for actual
        pin.framesForwardKinematics(self.model, self.data, q_actual_vec)
        actual_pos = self.data.oMf[fid].translation.copy()

        # Position error in mm
        return float(np.linalg.norm(target_pos - actual_pos) * 1000.0)

    def update(self) -> None:
        """Main update loop: update models, compute errors, refresh MeshCat."""
        if (
            not self.first_js_received
            or self.model is None
            or self.target_viz is None
            or self.actual_viz is None
        ):
            if self.get_clock().now().nanoseconds / 1e9 - self.last_log_time > 1.0:
                self.get_logger().info("waiting for /joint_states...")
                self.last_log_time = self.get_clock().now().nanoseconds / 1e9
            return

        ros_time = self.get_clock().now().nanoseconds / 1e9

        # Update target from trajectory interpolation
        self._update_target_from_trajectories(ros_time)

        # Build configuration vectors
        q_target = self._q_dict_to_vector(self.target_q)
        q_actual = self._q_dict_to_vector(self.actual_q)

        # Display target robot
        self.target_viz.display(q_target)
        self.target_display_count += 1

        # Display actual robot
        self.actual_viz.display(q_actual)
        self.actual_display_count += 1

        # Forward kinematics for error computation (reusing data)
        pin.forwardKinematics(self.model, self.data, q_actual)
        pin.updateFramePlacements(self.model, self.data)

        # Compute errors
        max_joint_err = 0.0
        for j in self.all_joint_names:
            if j in self.target_q and j in self.actual_q:
                err = abs(self.target_q[j] - self.actual_q[j])
                max_joint_err = max(max_joint_err, np.rad2deg(err))

        right_tcp_err = 0.0
        left_tcp_err = 0.0

        if self.side in ("both", "right"):
            right_tcp_err = self._compute_tcp_error("right", q_target, q_actual)
        if self.side in ("both", "left"):
            left_tcp_err = self._compute_tcp_error("left", q_target, q_actual)

        # Debug logging (throttled to every 2 seconds when enabled)
        if self.debug_values and ros_time - self.last_debug_log_time >= 2.0:
            # Get key joint values for debugging
            actual_right_j2 = self.actual_q.get("openarm_right_joint2", 0.0)
            actual_right_j4 = self.actual_q.get("openarm_right_joint4", 0.0)
            target_right_j2 = self.target_q.get("openarm_right_joint2", 0.0)
            target_right_j4 = self.target_q.get("openarm_right_joint4", 0.0)

            # Check if target_q and actual_q share the same object
            target_actual_same_object = self.target_q is self.actual_q

            # Check if q vectors share memory
            shares_memory = np.shares_memory(q_target, q_actual) if q_target is not q_actual else True

            self.get_logger().info(
                f"[debug] js_count={self.js_count}, "
                f"right_traj_count={self.right_traj_count}, "
                f"left_traj_count={self.left_traj_count}, "
                f"actual_right_j2={actual_right_j2:.4f} rad, "
                f"actual_right_j4={actual_right_j4:.4f} rad, "
                f"target_right_j2={target_right_j2:.4f} rad, "
                f"target_right_j4={target_right_j4:.4f} rad, "
                f"q_actual_idx_values={q_actual[:5].tolist() if len(q_actual) >= 5 else q_actual.tolist()}..., "
                f"q_target_idx_values={q_target[:5].tolist() if len(q_target) >= 5 else q_target.tolist()}..., "
                f"target_actual_same_object={target_actual_same_object}, "
                f"q_vectors_share_memory={shares_memory}, "
                f"actual_display_count={self.actual_display_count}, "
                f"target_display_count={self.target_display_count}"
            )
            self.last_debug_log_time = ros_time

        # Throttled logging (every 2 seconds)
        if ros_time - self.last_log_time >= 2.0:
            # Compute state age
            age_ms = (ros_time - self.last_js_time) * 1000 if hasattr(self, 'last_js_time') else 9999
            if age_ms < 300:
                state = "LIVE"
            elif age_ms < 1000:
                state = "STALE"
            elif age_ms < 5000:
                state = "DELAYED"
            else:
                state = "OFFLINE"

            # Compute approximate Hz
            hz = self.js_count / (ros_time - self.start_js_time) if hasattr(self, 'start_js_time') and ros_time > self.start_js_time else 0

            self.get_logger().info(
                f"[state] js={state}, hz={hz:.1f}, age={age_ms:.0f} ms, "
                f"right_traj={self.right_traj_count}, left_traj={self.left_traj_count}, "
                f"max_err={max_joint_err:.2f} deg, "
                f"right_tcp={right_tcp_err:.1f} mm, left_tcp={left_tcp_err:.1f} mm"
            )
            self.last_log_time = ros_time


def fetch_robot_description(node: MeshCatVisualizer, timeout_sec: float = 5.0) -> str | None:
    """Fetch robot_description parameter from robot_state_publisher.

    Args:
        node: ROS2 node
        timeout_sec: Timeout for parameter fetch

    Returns:
        URDF string or None if failed
    """
    import rclpy.node
    from rcl_interfaces.srv import GetParameters

    # Use synchronous parameter client
    client = node.create_client(GetParameters, "/robot_state_publisher/get_parameters")

    if not client.wait_for_service(timeout_sec=timeout_sec):
        node.get_logger().error(
            "robot_state_publisher not available. Is the launch running?"
        )
        return None

    request = GetParameters.Request()
    request.names = ["robot_description"]

    try:
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
        response = future.result()
        if response and response.values:
            return response.values[0].string_value
    except Exception as e:
        node.get_logger().error(f"Failed to get robot_description: {e}")

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only MeshCat visualizer: target trajectory vs actual robot state"
    )
    parser.add_argument(
        "--mode",
        choices=["side_by_side", "overlay"],
        default="side_by_side",
        help="Display mode: side_by_side (customer demos) or overlay (debugging)",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=10,
        help="Update rate in Hz (default: 10)",
    )
    parser.add_argument(
        "--robot-description-node",
        default="/robot_state_publisher",
        help="Node name to read robot_description from",
    )
    parser.add_argument(
        "--side",
        choices=["both", "right", "left"],
        default="both",
        help="Which arm side to emphasize (both sides still loaded)",
    )
    parser.add_argument(
        "--separation",
        type=float,
        default=1.0,
        help="Base separation in meters for side_by_side mode",
    )
    parser.add_argument(
        "--debug-values",
        action="store_true",
        help="Enable diagnostic logging of joint values (throttled to every 2 seconds)",
    )
    args = parser.parse_args()

    rclpy.init()

    node = MeshCatVisualizer(
        mode=args.mode,
        rate=args.rate,
        robot_description_node=args.robot_description_node,
        side=args.side,
        separation=args.separation,
        debug_values=args.debug_values,
    )

    # Fetch robot_description
    node.get_logger().info("Fetching robot_description from /robot_state_publisher...")
    urdf_str = fetch_robot_description(node)

    if not urdf_str:
        node.get_logger().error("Failed to get robot_description")
        node.destroy_node()
        rclpy.shutdown()
        return

    # Write to temp file for Pinocchio
    temp_urdf = tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False)
    try:
        temp_urdf.write(urdf_str)
        temp_urdf_path = temp_urdf.name
    finally:
        temp_urdf.close()

    node.get_logger().info(f"Loading robot model from {temp_urdf_path}")

    # Load model
    if not node.load_robot_model(temp_urdf_path):
        node.get_logger().error("Failed to load robot model")
        try:
            os.unlink(temp_urdf_path)
        except OSError:
            pass
        node.destroy_node()
        rclpy.shutdown()
        return

    # Setup update timer
    timer = node.create_timer(1.0 / args.rate, node.update)

    # Spin
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        node.get_logger().info("Visualizer running. Press Ctrl+C to exit.")
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        # Cleanup
        executor.shutdown()
        timer.cancel()
        node.destroy_node()
        rclpy.shutdown()

        # Delete temp file
        try:
            os.unlink(temp_urdf_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
