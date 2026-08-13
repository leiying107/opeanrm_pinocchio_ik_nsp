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

"""Read-only ROS2 + Dash + Plotly live trajectory comparison dashboard.

This script subscribes to:
  - /joint_states (actual robot feedback)
  - /right_joint_trajectory_controller/joint_trajectory (target trajectory)
  - /left_joint_trajectory_controller/joint_trajectory (target trajectory)

This is a READ-ONLY dashboard. It does NOT send any commands to the robot.

Example:
    python3 tools/meshcat_fk_dashboard.py --host 0.0.0.0 --port 8050

Demo mode (offline, no ROS2 required):
    python3 tools/meshcat_fk_dashboard.py --demo
"""

from __future__ import annotations

import argparse
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory
except ImportError:
    print("WARNING: ROS2 not available. Only --demo mode will work.")
    rclpy = None

try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output, State
    import plotly.graph_objs as go
except ImportError:
    print("ERROR: Dash not installed. Install with: pip install dash plotly")
    raise SystemExit(1)

# Joint names for each side
RIGHT_JOINTS = [
    "openarm_right_joint1",
    "openarm_right_joint2",
    "openarm_right_joint3",
    "openarm_right_joint4",
    "openarm_right_joint5",
    "openarm_right_joint6",
    "openarm_right_joint7",
]
LEFT_JOINTS = [
    "openarm_left_joint1",
    "openarm_left_joint2",
    "openarm_left_joint3",
    "openarm_left_joint4",
    "openarm_left_joint5",
    "openarm_left_joint6",
    "openarm_left_joint7",
]

ALL_JOINTS = RIGHT_JOINTS + LEFT_JOINTS

# TCP offsets (approximate, for demo only)
# In production, these would come from FK
TCP_OFFSET_RIGHT = np.array([0.0, 0.0, 0.25])  # meters from J7
TCP_OFFSET_LEFT = np.array([0.0, 0.0, 0.25])

# Buffer duration for plots (seconds)
PLOT_DURATION = 20.0

# Demo mode trajectory parameters
DEMO_RIGHT_J2_START = 0.0
DEMO_RIGHT_J2_END = -np.radians(10)  # -10 degrees
DEMO_RIGHT_J4_START = 0.0
DEMO_RIGHT_J4_END = np.radians(10)  # +10 degrees
DEMO_CYCLE_DURATION = 14.0  # seconds per full cycle
DEMO_DELAY = 0.2  # actual follows target with delay

# Demo cycle timing (seconds)
DEMO_HOLD_START = 2.0
DEMO_MOVE_OUT_START = 2.0
DEMO_MOVE_OUT_END = 6.0
DEMO_HOLD_END = 2.0
DEMO_MOVE_BACK_START = 8.0
DEMO_MOVE_BACK_END = 12.0
DEMO_HOLD_FINAL = 2.0

# Color palette for 7 joints (distinct colors)
JOINT_COLORS = [
    "#1f77b4",  # blue - J1
    "#ff7f0e",  # orange - J2
    "#2ca02c",  # green - J3
    "#d62728",  # red - J4
    "#9467bd",  # purple - J5
    "#8c564b",  # brown - J6
    "#e377c2",  # pink - J7
]


def smoothstep(u: float) -> float:
    """Smooth step interpolation: 3u^2 - 2u^3.

    Args:
        u: Normalized value in [0, 1]

    Returns:
        Smoothed value in [0, 1]
    """
    return u * u * (3 - 2 * u)


class EventLog:
    """Thread-safe event log for dashboard."""

    def __init__(self, max_events: int = 100):
        self.events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self.lock = threading.Lock()

    def add(self, event_type: str, message: str) -> None:
        with self.lock:
            self.events.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "type": event_type,
                "message": message,
            })

    def get_all(self) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.events)


class RobotState:
    """Thread-safe container for robot state data."""

    def __init__(self):
        self.lock = threading.Lock()

        # Joint states
        self.actual_q: dict[str, float] = {j: 0.0 for j in ALL_JOINTS}
        self.target_q: dict[str, float] = {j: 0.0 for j in ALL_JOINTS}
        self.initial_q: dict[str, float] = {j: 0.0 for j in ALL_JOINTS}

        # Timing
        self.last_js_time: float | None = None
        self.js_count: int = 0
        self.start_time: float | None = None

        # Trajectory counts
        self.right_traj_count: int = 0
        self.left_traj_count: int = 0

        # TCP positions (simplified, for demo)
        self.right_tcp_target: np.ndarray = np.zeros(3)
        self.right_tcp_actual: np.ndarray = np.zeros(3)
        self.left_tcp_target: np.ndarray = np.zeros(3)
        self.left_tcp_actual: np.ndarray = np.zeros(3)

    def update_joint_state(self, msg: JointState | None) -> None:
        with self.lock:
            now = time.time()
            if self.start_time is None:
                self.start_time = now

            self.last_js_time = now
            self.js_count += 1

            if msg is not None:
                joint_map = dict(zip(msg.name, msg.position))
                for name, pos in joint_map.items():
                    if name in ALL_JOINTS:
                        self.actual_q[name] = pos

                # Initialize initial_q on first message
                if all(v == 0.0 for v in self.initial_q.values()):
                    for j in ALL_JOINTS:
                        if j in self.actual_q:
                            self.initial_q[j] = self.actual_q[j]

    def update_trajectory(
        self, msg: JointTrajectory | None, side: str
    ) -> None:
        with self.lock:
            if side == "right":
                self.right_traj_count += 1
            else:
                self.left_traj_count += 1

            if msg is not None and msg.points:
                # Get latest point
                pt = msg.points[-1]
                joint_map = dict(zip(msg.joint_names, pt.positions))
                for j, val in joint_map.items():
                    if j in ALL_JOINTS:
                        self.target_q[j] = val

    def get_state_summary(self, demo_mode: bool = False) -> dict[str, Any]:
        with self.lock:
            now = time.time()

            # In demo mode, always show DEMO state
            if demo_mode:
                state = "DEMO"
            else:
                age = (
                    now - self.last_js_time
                    if self.last_js_time is not None
                    else float("inf")
                )

                # Determine state
                if age == float("inf"):
                    state = "OFFLINE"
                elif age < 0.3:
                    state = "LIVE"
                elif age < 1.0:
                    state = "DELAYED"
                else:
                    state = "STALE"

            # Compute Hz (Demo Rate in demo mode)
            hz = 0.0
            if self.start_time is not None and self.js_count > 0:
                elapsed = now - self.start_time
                if elapsed > 0:
                    hz = self.js_count / elapsed

            # Compute max joint error
            max_err = 0.0
            for j in ALL_JOINTS:
                err = abs(self.target_q[j] - self.actual_q[j])
                max_err = max(max_err, np.rad2deg(err))

            # Compute TCP errors (simplified)
            right_tcp_err = np.linalg.norm(self.right_tcp_target - self.right_tcp_actual) * 1000
            left_tcp_err = np.linalg.norm(self.left_tcp_target - self.left_tcp_actual) * 1000

            return {
                "state": state,
                "hz": hz,
                "age_ms": 0.0 if demo_mode else (
                    (now - self.last_js_time) * 1000 if self.last_js_time is not None else float("inf")
                ),
                "right_traj_count": self.right_traj_count,
                "left_traj_count": self.left_traj_count,
                "max_joint_error": max_err,
                "right_tcp_error": right_tcp_err,
                "left_tcp_error": left_tcp_err,
            }

    def get_joint_data(self, side: str, joint_idx: int) -> dict[str, float]:
        joints = RIGHT_JOINTS if side == "right" else LEFT_JOINTS
        if 0 <= joint_idx < len(joints):
            name = joints[joint_idx]
            with self.lock:
                return {
                    "target": np.rad2deg(self.target_q[name]),
                    "actual": np.rad2deg(self.actual_q[name]),
                    "error": np.rad2deg(
                        self.target_q[name] - self.actual_q[name]
                    ),
                    "relative": np.rad2deg(
                        self.actual_q[name] - self.initial_q[name]
                    ),
                }
        return {"target": 0.0, "actual": 0.0, "error": 0.0, "relative": 0.0}

    def get_all_joint_data(self, side: str) -> list[dict[str, Any]]:
        joints = RIGHT_JOINTS if side == "right" else LEFT_JOINTS
        result = []
        with self.lock:
            for i, name in enumerate(joints):
                result.append({
                    "joint": f"J{i+1}",
                    "name": name,
                    "target": np.rad2deg(self.target_q[name]),
                    "actual": np.rad2deg(self.actual_q[name]),
                    "error": np.rad2deg(self.target_q[name] - self.actual_q[name]),
                    "relative": np.rad2deg(self.actual_q[name] - self.initial_q[name]),
                })
        return result


class DataBuffer:
    """Thread-safe circular buffer for time-series data."""

    def __init__(self, duration: float = PLOT_DURATION):
        self.duration = duration
        # Separate buffer per side, storing all joints
        self.data: dict[str, deque[dict[str, Any]]] = {"right": deque(), "left": deque()}
        self.lock = threading.Lock()

    def add(self, timestamp: float, side: str, joint_data: list[dict[str, Any]]) -> None:
        with self.lock:
            now = time.time()
            # joint_data is list of dicts with keys: joint_idx, target, actual, error, relative
            entry = {"time": timestamp, "joints": joint_data}
            self.data[side].append(entry)

            # Remove old data
            cutoff = now - self.duration
            while self.data[side] and self.data[side][0]["time"] < cutoff:
                self.data[side].popleft()

    def get(self, side: str) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.data.get(side, []))


class ROS2Node(Node):
    """Read-only ROS2 node for subscribing to robot state."""

    def __init__(self, state: RobotState, events: EventLog):
        if rclpy is None:
            raise RuntimeError("ROS2 not available")
        super().__init__("meshcat_fk_dashboard")

        self.state = state
        self.events = events

        # Subscriptions (read-only)
        self.js_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self._on_joint_states,
            qos_profile_sensor_data,
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

        self.get_logger().info("READ-ONLY DASHBOARD: no commands sent to robot")
        self.first_js = False

    def _on_joint_states(self, msg: JointState) -> None:
        self.state.update_joint_state(msg)

        if not self.first_js:
            self.events.add("info", "First JointState received")
            self.first_js = True

    def _on_trajectory(self, msg: JointTrajectory, side: str) -> None:
        self.state.update_trajectory(msg, side)
        self.events.add(
            "trajectory", f"{side.capitalize()} trajectory received ({len(msg.points)} points)"
        )


class DemoModeGenerator:
    """Generates deterministic demo data for offline testing.

    Uses smooth cyclic motion with no sudden jumps.
    """

    def __init__(self, state: RobotState, events: EventLog):
        self.state = state
        self.events = events
        self.start_time = time.time()
        self.running = True
        self.cycle_count = 0
        self.logged_start = False

    def update(self) -> None:
        now = time.time()
        elapsed = now - self.start_time

        # Cycle position (0 to DEMO_CYCLE_DURATION)
        cycle_time = elapsed % DEMO_CYCLE_DURATION

        # Track cycles for logging
        current_cycle = int(elapsed / DEMO_CYCLE_DURATION)
        if current_cycle > self.cycle_count:
            self.cycle_count = current_cycle
            self.events.add("demo", f"Demo cycle {self.cycle_count + 1} started")

        # First cycle start notification
        if not self.logged_start and elapsed > 0.5:
            self.events.add("demo", "Demo trajectory started")
            self.logged_start = True

        # Calculate target position based on cycle phase
        j2_target = 0.0
        j4_target = 0.0

        if cycle_time < DEMO_HOLD_START:
            # Phase 1: Hold at start (0-2s)
            j2_target = DEMO_RIGHT_J2_START
            j4_target = DEMO_RIGHT_J4_START
        elif cycle_time < DEMO_MOVE_OUT_END:
            # Phase 2: Move to target (2-6s)
            phase_time = cycle_time - DEMO_MOVE_OUT_START
            phase_duration = DEMO_MOVE_OUT_END - DEMO_MOVE_OUT_START
            u = phase_time / phase_duration
            u_smooth = smoothstep(u)
            j2_target = DEMO_RIGHT_J2_START + (DEMO_RIGHT_J2_END - DEMO_RIGHT_J2_START) * u_smooth
            j4_target = DEMO_RIGHT_J4_START + (DEMO_RIGHT_J4_END - DEMO_RIGHT_J4_START) * u_smooth
        elif cycle_time < DEMO_MOVE_BACK_START:
            # Phase 3: Hold at target (6-8s)
            j2_target = DEMO_RIGHT_J2_END
            j4_target = DEMO_RIGHT_J4_END
        elif cycle_time < DEMO_MOVE_BACK_END:
            # Phase 4: Move back to start (8-12s)
            phase_time = cycle_time - DEMO_MOVE_BACK_START
            phase_duration = DEMO_MOVE_BACK_END - DEMO_MOVE_BACK_START
            u = phase_time / phase_duration
            u_smooth = smoothstep(u)
            j2_target = DEMO_RIGHT_J2_END + (DEMO_RIGHT_J2_START - DEMO_RIGHT_J2_END) * u_smooth
            j4_target = DEMO_RIGHT_J4_END + (DEMO_RIGHT_J4_START - DEMO_RIGHT_J4_END) * u_smooth
        else:
            # Phase 5: Hold at end (12-14s)
            j2_target = DEMO_RIGHT_J2_START
            j4_target = DEMO_RIGHT_J4_START

        # Actual follows with slight delay (smoothed)
        delayed_time = (elapsed - DEMO_DELAY) % DEMO_CYCLE_DURATION

        j2_actual = 0.0
        j4_actual = 0.0

        if delayed_time < DEMO_HOLD_START:
            j2_actual = DEMO_RIGHT_J2_START
            j4_actual = DEMO_RIGHT_J4_START
        elif delayed_time < DEMO_MOVE_OUT_END:
            phase_time = delayed_time - DEMO_MOVE_OUT_START
            phase_duration = DEMO_MOVE_OUT_END - DEMO_MOVE_OUT_START
            u = phase_time / phase_duration
            u_smooth = smoothstep(u)
            j2_actual = DEMO_RIGHT_J2_START + (DEMO_RIGHT_J2_END - DEMO_RIGHT_J2_START) * u_smooth
            j4_actual = DEMO_RIGHT_J4_START + (DEMO_RIGHT_J4_END - DEMO_RIGHT_J4_START) * u_smooth
        elif delayed_time < DEMO_MOVE_BACK_START:
            j2_actual = DEMO_RIGHT_J2_END
            j4_actual = DEMO_RIGHT_J4_END
        elif delayed_time < DEMO_MOVE_BACK_END:
            phase_time = delayed_time - DEMO_MOVE_BACK_START
            phase_duration = DEMO_MOVE_BACK_END - DEMO_MOVE_BACK_START
            u = phase_time / phase_duration
            u_smooth = smoothstep(u)
            j2_actual = DEMO_RIGHT_J2_END + (DEMO_RIGHT_J2_START - DEMO_RIGHT_J2_END) * u_smooth
            j4_actual = DEMO_RIGHT_J4_END + (DEMO_RIGHT_J4_START - DEMO_RIGHT_J4_END) * u_smooth
        else:
            j2_actual = DEMO_RIGHT_J2_START
            j4_actual = DEMO_RIGHT_J4_START

        # Update state
        with self.state.lock:
            self.state.target_q["openarm_right_joint2"] = j2_target
            self.state.target_q["openarm_right_joint4"] = j4_target
            self.state.actual_q["openarm_right_joint2"] = j2_actual
            self.state.actual_q["openarm_right_joint4"] = j4_actual

            # Simulate timing
            self.state.last_js_time = now
            if self.state.start_time is None:
                self.state.start_time = now
            self.state.js_count += 1


def create_dashboard_layout(state: RobotState, demo_mode: bool) -> html.Div:
    """Create the Dash dashboard layout."""

    # Demo mode banner
    demo_banner = html.Div(
        [
            html.H3(
                "DEMO DATA — NOT CONNECTED TO ROBOT",
                style={
                    "color": "#ff6b6b",
                    "text-align": "center",
                    "margin": "10px",
                },
            )
        ]
        if demo_mode
        else html.Div()
    )

    # Status cards
    status_card_style = {
        "background": "#f8f9fa",
        "border": "1px solid #dee2e6",
        "border-radius": "8px",
        "padding": "15px",
        "margin": "5px",
        "min-width": "150px",
        "text-align": "center",
    }

    status_cards = html.Div(
        [
            html.Div(
                [
                    html.H6("JointState Status", style={"margin-bottom": "5px"}),
                    html.H3(id="status-state", children="OFFLINE", style={"margin": "0"}),
                ],
                style=status_card_style,
            ),
            html.Div(
                [
                    html.H6(id="status-hz-label", children="JointState Hz", style={"margin-bottom": "5px"}),
                    html.H3(id="status-hz", children="0.0", style={"margin": "0"}),
                ],
                style=status_card_style,
            ),
            html.Div(
                [
                    html.H6("Last Age (ms)", style={"margin-bottom": "5px"}),
                    html.H3(id="status-age", children="N/A", style={"margin": "0"}),
                ],
                style=status_card_style,
            ),
            html.Div(
                [
                    html.H6("Right Traj Count", style={"margin-bottom": "5px"}),
                    html.H3(id="status-right-traj", children="0", style={"margin": "0"}),
                ],
                style=status_card_style,
            ),
            html.Div(
                [
                    html.H6("Left Traj Count", style={"margin-bottom": "5px"}),
                    html.H3(id="status-left-traj", children="0", style={"margin": "0"}),
                ],
                style=status_card_style,
            ),
            html.Div(
                [
                    html.H6("Max Joint Error (deg)", style={"margin-bottom": "5px"}),
                    html.H3(id="status-max-err", children="0.00", style={"margin": "0"}),
                ],
                style=status_card_style,
            ),
            html.Div(
                [
                    html.H6("Right TCP Error (mm)", style={"margin-bottom": "5px"}),
                    html.H3(id="status-right-tcp", children="0.0", style={"margin": "0"}),
                ],
                style=status_card_style,
            ),
            html.Div(
                [
                    html.H6("Left TCP Error (mm)", style={"margin-bottom": "5px"}),
                    html.H3(id="status-left-tcp", children="0.0", style={"margin": "0"}),
                ],
                style=status_card_style,
            ),
        ],
        style={
            "display": "flex",
            "flex-wrap": "wrap",
            "justify-content": "center",
            "margin": "20px 0",
        },
    )

    # Demo status banner (dynamic content based on mode)
    demo_status_banner = html.Div(
        id="demo-status-banner",
        children="",
        style={"display": "none"},
    )

    # Control for plot selection
    controls = html.Div(
        [
            html.Div(
                [
                    html.Label("Side:"),
                    dcc.Dropdown(
                        id="side-dropdown",
                        options=[
                            {"label": "Right Arm", "value": "right"},
                            {"label": "Left Arm", "value": "left"},
                        ],
                        value="right",
                        clearable=False,
                        style={"width": "200px"},
                    ),
                ],
                style={"margin": "10px"},
            ),
            html.Div(
                [
                    html.Label("Joint:"),
                    dcc.Dropdown(
                        id="joint-dropdown",
                        options=[
                            {"label": f"J{i+1}", "value": i}
                            for i in range(len(RIGHT_JOINTS))
                        ],
                        value=0,
                        clearable=False,
                        style={"width": "200px"},
                    ),
                ],
                style={"margin": "10px"},
            ),
        ],
        style={"display": "flex", "justify-content": "center", "margin": "20px 0"},
    )

    # Time series plots
    plot_style = {"height": "250px", "margin": "10px"}

    plots = html.Div(
        [
            html.Div(
                [
                    html.H5("All Joint Angles"),
                    dcc.Graph(id="plot-position", style=plot_style),
                ]
            ),
            html.Div(
                [html.H5("All Joint Errors"), dcc.Graph(id="plot-error", style=plot_style)]
            ),
            html.Div(
                [
                    html.H5("All Joint Relative Changes"),
                    dcc.Graph(id="plot-relative", style=plot_style),
                ]
            ),
            html.Div(
                [html.H5("TCP Position Error (mm)"), dcc.Graph(id="plot-tcp", style=plot_style)]
            ),
        ],
        style={"display": "grid", "grid-template-columns": "1fr 1fr", "margin": "20px"},
    )

    # All joints table
    table_style = {
        "margin": "20px auto",
        "width": "90%",
        "border-collapse": "collapse",
    }

    table = html.Div(
        [
            html.H4("All Joints Overview", style={"text-align": "center"}),
            html.Table(
                [
                    html.Thead(
                        [
                            html.Tr(
                                [
                                    html.Th("Joint", style={"border": "1px solid #ddd", "padding": "8px"}),
                                    html.Th("Target (deg)", style={"border": "1px solid #ddd", "padding": "8px"}),
                                    html.Th("Actual (deg)", style={"border": "1px solid #ddd", "padding": "8px"}),
                                    html.Th("Error (deg)", style={"border": "1px solid #ddd", "padding": "8px"}),
                                    html.Th("Relative (deg)", style={"border": "1px solid #ddd", "padding": "8px"}),
                                ]
                            )
                        ]
                    ),
                    html.Tbody(id="joint-table-body"),
                ],
                style=table_style,
            ),
        ],
        style={"margin": "20px 0"},
    )

    # Event log
    events_div = html.Div(
        [
            html.H4("Event Log", style={"text-align": "center"}),
            html.Div(
                id="event-log",
                style={
                    "height": "200px",
                    "overflow-y": "auto",
                    "background": "#f8f9fa",
                    "border": "1px solid #dee2e6",
                    "border-radius": "8px",
                    "padding": "10px",
                    "margin": "20px auto",
                    "width": "90%",
                    "font-family": "monospace",
                    "font-size": "12px",
                },
            ),
        ]
    )

    return html.Div(
        [
            demo_banner,
            html.H1(
                "OpenArm FK Trajectory Dashboard",
                style={"text-align": "center", "margin": "20px 0"},
            ),
            status_cards,
            demo_status_banner,
            controls,
            plots,
            table,
            events_div,
            dcc.Interval(id="interval-component", interval=200, n_intervals=0),
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Read-only Dash dashboard for trajectory comparison"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host to bind dashboard server"
    )
    parser.add_argument(
        "--port", type=int, default=8050, help="Port for dashboard server"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode (offline, no ROS2 required)",
    )
    args = parser.parse_args()

    # Shared state
    state = RobotState()
    events = EventLog()
    buffer = DataBuffer()

    demo_mode = args.demo

    # ROS2 setup (skip in demo mode)
    ros2_node = None
    executor_thread = None

    if not demo_mode:
        if rclpy is None:
            print("ERROR: ROS2 requested but not available. Use --demo for offline mode.")
            return

        rclpy.init()
        ros2_node = ROS2Node(state, events)

        # Spin in background thread
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(ros2_node)

        def spin():
            try:
                executor.spin()
            except Exception:
                pass

        executor_thread = threading.Thread(target=spin, daemon=True)
        executor_thread.start()

        print(f"ROS2 node initialized, waiting for /joint_states...")
    else:
        print("DEMO MODE: generating offline data")

        # Initialize demo generator
        demo_gen = DemoModeGenerator(state, events)

        def demo_updater():
            while demo_gen.running:
                demo_gen.update()
                time.sleep(0.05)  # 20 Hz

        threading.Thread(target=demo_updater, daemon=True).start()

    # Create Dash app
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.layout = create_dashboard_layout(state, demo_mode)

    # Callbacks
    @app.callback(
        [
            Output("status-state", "children"),
            Output("status-state", "style"),
            Output("status-hz", "children"),
            Output("status-hz-label", "children"),
            Output("status-age", "children"),
            Output("status-right-traj", "children"),
            Output("status-left-traj", "children"),
            Output("status-max-err", "children"),
            Output("status-right-tcp", "children"),
            Output("status-left-tcp", "children"),
            Output("demo-status-banner", "children"),
            Output("demo-status-banner", "style"),
        ],
        [Input("interval-component", "n_intervals")],
    )
    def update_status(n):
        summary = state.get_state_summary(demo_mode=demo_mode)

        # Color coding
        state_colors = {
            "LIVE": {"color": "#28a745"},
            "DELAYED": {"color": "#ffc107"},
            "STALE": {"color": "#fd7e14"},
            "OFFLINE": {"color": "#dc3545"},
            "DEMO": {"color": "#6c757d"},
        }

        # Hz label depends on mode
        hz_label = "Demo Rate" if demo_mode else "JointState Hz"

        # Demo status banner
        if demo_mode:
            demo_banner = "SIMULATED DATA — ROBOT NOT CONNECTED"
            demo_banner_style = {
                "color": "#dc3545",
                "font-weight": "bold",
                "text-align": "center",
                "margin": "5px",
                "font-size": "14px",
            }
        else:
            demo_banner = ""
            demo_banner_style = {"display": "none"}

        return (
            summary["state"],
            state_colors.get(summary["state"], {}),
            f"{summary['hz']:.1f}",
            hz_label,
            f"{summary['age_ms']:.0f}" if not demo_mode else "N/A",
            summary["right_traj_count"],
            summary["left_traj_count"],
            f"{summary['max_joint_error']:.2f}",
            f"{summary['right_tcp_error']:.1f}",
            f"{summary['left_tcp_error']:.1f}",
            demo_banner,
            demo_banner_style,
        )

    @app.callback(
        [
            Output("plot-position", "figure"),
            Output("plot-error", "figure"),
            Output("plot-relative", "figure"),
            Output("plot-tcp", "figure"),
        ],
        [
            Input("interval-component", "n_intervals"),
            Input("side-dropdown", "value"),
        ],
    )
    def update_plots(n, side):
        # Add current data for all joints to buffer
        now = time.time()
        all_joint_data = state.get_all_joint_data(side)
        buffer.add(now, side, all_joint_data)

        # Get historical data for selected side
        hist = buffer.get(side)

        if not hist:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                height=200, margin=dict(l=0, r=0, t=30, b=30)
            )
            return empty_fig, empty_fig, empty_fig, empty_fig

        times = [time.time() - d["time"] for d in hist]
        times = [-t for t in times]  # Show as "seconds ago"

        # Position plot: All J1-J7 Target and Actual (14 traces)
        pos_fig = go.Figure()
        for j_idx in range(7):
            color = JOINT_COLORS[j_idx]
            # Extract data for this joint
            target_vals = [d["joints"][j_idx]["target"] for d in hist]
            actual_vals = [d["joints"][j_idx]["actual"] for d in hist]

            # Target (dashed)
            pos_fig.add_trace(
                go.Scatter(
                    x=times,
                    y=target_vals,
                    name=f"J{j_idx+1} Target",
                    mode="lines",
                    line=dict(color=color, dash="dash"),
                )
            )
            # Actual (solid)
            pos_fig.add_trace(
                go.Scatter(
                    x=times,
                    y=actual_vals,
                    name=f"J{j_idx+1} Actual",
                    mode="lines",
                    line=dict(color=color, dash="solid"),
                )
            )

        pos_fig.update_layout(
            title=f"All Joint Angles — {side.capitalize()}",
            xaxis_title="Seconds Ago",
            yaxis_title="Angle (deg)",
            margin=dict(l=0, r=0, t=30, b=30),
            height=200,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
            ),
            hovermode="x unified",
        )
        # Set uirevision to preserve zoom/legend state
        pos_fig.update_layout(uirevision=f"{side}_position")

        # Error plot: All J1-J7 Error (7 traces)
        err_fig = go.Figure()
        for j_idx in range(7):
            color = JOINT_COLORS[j_idx]
            error_vals = [d["joints"][j_idx]["error"] for d in hist]

            err_fig.add_trace(
                go.Scatter(
                    x=times,
                    y=error_vals,
                    name=f"J{j_idx+1} Error",
                    mode="lines",
                    line=dict(color=color),
                )
            )

        err_fig.update_layout(
            title="All Joint Errors",
            xaxis_title="Seconds Ago",
            yaxis_title="Error (deg)",
            margin=dict(l=0, r=0, t=30, b=30),
            height=200,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
            ),
            hovermode="x unified",
        )
        err_fig.update_layout(uirevision=f"{side}_error")

        # Relative plot: All J1-J7 Relative Change (7 traces)
        rel_fig = go.Figure()
        for j_idx in range(7):
            color = JOINT_COLORS[j_idx]
            rel_vals = [d["joints"][j_idx]["relative"] for d in hist]

            rel_fig.add_trace(
                go.Scatter(
                    x=times,
                    y=rel_vals,
                    name=f"J{j_idx+1} Relative",
                    mode="lines",
                    line=dict(color=color),
                )
            )

        rel_fig.update_layout(
            title=f"All Joint Relative Changes — {side.capitalize()}",
            xaxis_title="Seconds Ago",
            yaxis_title="Change (deg)",
            margin=dict(l=0, r=0, t=30, b=30),
            height=200,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
            ),
            hovermode="x unified",
        )
        rel_fig.update_layout(uirevision=f"{side}_relative")

        # TCP plot (unchanged)
        tcp_fig = go.Figure()
        tcp_err = state.get_state_summary()
        tcp_fig.add_trace(
            go.Scatter(
                x=[0],
                y=[tcp_err["right_tcp_error"] if side == "right" else tcp_err["left_tcp_error"]],
                name="TCP Error",
                mode="markers",
                marker=dict(size=10),
            )
        )
        tcp_fig.update_layout(
            title=f"{side.capitalize()} TCP Position Error",
            xaxis_title="Current",
            yaxis_title="Error (mm)",
            margin=dict(l=0, r=0, t=30, b=30),
            height=200,
        )
        tcp_fig.update_layout(uirevision=f"{side}_tcp")

        return pos_fig, err_fig, rel_fig, tcp_fig

    @app.callback(
        Output("joint-table-body", "children"),
        [Input("interval-component", "n_intervals"), Input("side-dropdown", "value")],
    )
    def update_table(n, side):
        rows = state.get_all_joint_data(side)

        return [
            html.Tr(
                [
                    html.Td(r["joint"], style={"border": "1px solid #ddd", "padding": "8px"}),
                    html.Td(
                        f"{r['target']:.2f}",
                        style={"border": "1px solid #ddd", "padding": "8px"},
                    ),
                    html.Td(
                        f"{r['actual']:.2f}",
                        style={"border": "1px solid #ddd", "padding": "8px"},
                    ),
                    html.Td(
                        f"{r['error']:.3f}",
                        style={
                            "border": "1px solid #ddd",
                            "padding": "8px",
                            "color": "red" if abs(r["error"]) > 1.0 else "green",
                        },
                    ),
                    html.Td(
                        f"{r['relative']:.2f}",
                        style={"border": "1px solid #ddd", "padding": "8px"},
                    ),
                ]
            )
            for r in rows
        ]

    @app.callback(
        Output("event-log", "children"),
        [Input("interval-component", "n_intervals")],
    )
    def update_events(n):
        all_events = events.get_all()
        return [
            html.Div(
                f"[{e['time']}] {e['type'].upper()}: {e['message']}",
                style={"margin": "2px 0"},
            )
            for e in all_events
        ]

    # Print URL
    url = f"http://{args.host}:{args.port}/"
    print(f"Dashboard URL: {url}")
    if demo_mode:
        print("DEMO DATA — NOT CONNECTED TO ROBOT")

    try:
        app.run(host=args.host, port=args.port, debug=False)
    finally:
        # Cleanup
        if demo_mode:
            state.running = False
        if ros2_node is not None:
            ros2_node.destroy_node()
        if executor_thread is not None:
            executor_thread.join(timeout=1.0)
        if not demo_mode and rclpy is not None:
            rclpy.shutdown()


if __name__ == "__main__":
    main()
