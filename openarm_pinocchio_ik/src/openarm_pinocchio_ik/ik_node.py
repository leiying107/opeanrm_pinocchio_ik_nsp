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

"""ROS2 node: target EE pose -> IK -> joint_trajectory + gravity compensation.

Subscribes to a target end-effector pose (geometry_msgs/PoseStamped), solves IK
with Pinocchio each cycle, and publishes:
  * the solved joint positions to ``/<side>_joint_trajectory_controller/joint_trajectory``
  * the gravity torques to ``/<side>_forward_effort_controller/commands``

Run one instance per arm side (left / right). Warm-starts IK from /joint_states.
"""

from __future__ import annotations

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from openarm_pinocchio_ik.kinematics import PinocchioModel

_DEFAULT_URDF = (
    "/ros2_ws/install/openarm_description/share/openarm_description/"
    "assets/robot/openarm_v1.0/urdf/example/v1.urdf"
)


class IKNode(Node):
    def __init__(self) -> None:
        super().__init__("openarm_pinocchio_ik")

        self.declare_parameter("side", "right")
        self.declare_parameter("urdf_path", _DEFAULT_URDF)
        self.declare_parameter("control_hz", 50.0)
        self.declare_parameter("ik_max_iters", 50)
        self.declare_parameter("ik_tol", 1e-4)
        self.declare_parameter("ik_damping", 1e-2)
        self.declare_parameter("max_step_rad", 0.05)
        self.declare_parameter("enable_gravity_comp", True)

        side = self.get_parameter("side").value
        urdf_path = self.get_parameter("urdf_path").value
        self.max_step = float(self.get_parameter("max_step_rad").value)
        self.enable_gravity = bool(self.get_parameter("enable_gravity_comp").value)
        self.ik_kwargs = dict(
            max_iters=int(self.get_parameter("ik_max_iters").value),
            tol=float(self.get_parameter("ik_tol").value),
            damping=float(self.get_parameter("ik_damping").value),
        )

        self.model = PinocchioModel(urdf_path, side)
        self.joint_names = [f"openarm_{side}_joint{i}" for i in range(1, 8)]
        self.current_q = (self.model.lower + self.model.upper) / 2.0
        self.target_pose = None  # geometry_msgs/Pose

        self.create_subscription(
            PoseStamped, f"/openarm_{side}_target_pose", self._on_pose, 10
        )
        self.create_subscription(JointState, "/joint_states", self._on_js, qos_profile_sensor_data)
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            f"/{side}_joint_trajectory_controller/joint_trajectory",
            10,
        )
        self.effort_pub = self.create_publisher(
            Float64MultiArray,
            f"/{side}_forward_effort_controller/commands",
            10,
        )

        hz = float(self.get_parameter("control_hz").value)
        self.create_timer(1.0 / hz, self._loop)
        self.get_logger().info(
            f"IK node ready: side={side}, hz={hz}, gravity_comp={self.enable_gravity}"
        )

    def _on_js(self, msg: JointState) -> None:
        try:
            self.current_q = np.array(
                [msg.position[msg.name.index(n)] for n in self.joint_names]
            )
        except ValueError:
            pass  # not all joints present yet

    def _on_pose(self, msg: PoseStamped) -> None:
        self.target_pose = msg.pose

    def _loop(self) -> None:
        if self.target_pose is None:
            return
        p = self.target_pose.position
        o = self.target_pose.orientation
        target_pos = np.array([p.x, p.y, p.z])
        target_quat_xyzw = np.array([o.x, o.y, o.z, o.w])

        q_target = self.model.ik(
            target_pos, target_quat_xyzw, q_init=self.current_q, **self.ik_kwargs
        )
        if q_target is None:
            self.get_logger().warn("IK not converged", throttle_duration_sec=1.0)
            return

        # rate-limit the commanded step for safety on real hardware
        step = np.clip(q_target - self.current_q, -self.max_step, self.max_step)
        q_cmd = self.current_q + step

        traj = JointTrajectory()
        traj.joint_names = self.joint_names
        pt = JointTrajectoryPoint()
        pt.positions = q_cmd.tolist()
        pt.time_from_start = Duration(sec=0, nanosec=200_000_000)  # 0.2s
        traj.points = [pt]
        self.traj_pub.publish(traj)

        if self.enable_gravity:
            g = self.model.gravity(self.current_q)
            self.effort_pub.publish(Float64MultiArray(data=g.tolist()))


def main() -> None:
    rclpy.init()
    node = IKNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
