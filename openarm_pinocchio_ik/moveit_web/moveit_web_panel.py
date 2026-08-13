#!/usr/bin/env python3
# Copyright 2026 Enactic, Inc.
# SPDX-License-Identifier: Apache-2.0
"""OpenArm native-MoveIt web panel (minimal).

A pure ``move_group`` CLIENT: talks only to ``/move_action`` (MoveGroup action),
``/compute_fk`` (service) and ``/joint_states`` (topic). It is **hardware- and
ROS-domain-agnostic**: the SAME script drives fake hardware now (ROS_DOMAIN_ID=42)
and real hardware later (once you take over CAN) -- only the launch command and
domain change, not this code.

Run (fake, isolated from the live product stack):
    # terminal A -- official move_group + fake ros2_control (domain 42)
    source /opt/ros/humble/setup.bash && source /ros2_ws/openarm_ros2/install/setup.bash
    ROS_DOMAIN_ID=42 ros2 launch openarm_bimanual_moveit_config demo.launch.py \
        arm_type:=openarm_v1.0
    # terminal B -- this panel
    source /opt/ros/humble/setup.bash && source /ros2_ws/openarm_ros2/install/setup.bash
    ROS_DOMAIN_ID=42 /ros2_ws/openarm_nsp_ws/venv-openarm-ik/bin/python \
        /ros2_ws/openarm_pinocchio_ik/moveit_web/moveit_web_panel.py
    # -> open http://<this-machine-ip>:8050

Real hardware (later): stop/swap the live stack so ros2_control owns CAN, then run
the same panel on that domain. on_activate sweeps to zero, keep velocity scaling low.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from moveit_msgs.srv import GetPositionFK
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (MotionPlanRequest, Constraints, JointConstraint,
                             RobotState, PositionConstraint, OrientationConstraint,
                             PlanningOptions)
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Quaternion
from std_msgs.msg import Header
from scipy.spatial.transform import Rotation

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go

ROOT = "openarm_body_link0"
SIDES = {"left": "left_arm", "right": "right_arm"}
JOINTS = {s: [f"openarm_{s}_joint{i}" for i in range(1, 8)] for s in ("left", "right")}
TIP = {s: f"openarm_{s}_link7" for s in ("left", "right")}
# SRDF group_state values (same numbers for both arms, applied to that arm's joints)
NAMED = {"home": [0.0] * 7, "hands_up": [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]}
ERR = {1: "SUCCESS", 99999: "FAILURE", -1: "PLANNING_FAILED", -2: "INVALID_MOTION_PLAN",
       -6: "TIMED_OUT", -12: "GOAL_IN_COLLISION", -13: "GOAL_VIOLATES_PATH_CONSTRAINTS",
       -16: "INVALID_GOAL_CONSTRAINTS", -27: "GOAL_STATE_INVALID", -28: "UNRECOGNIZED_GOAL_TYPE",
       -31: "NO_IK_SOLUTION", -10: "START_STATE_IN_COLLISION", -11: "START_STATE_VIOLATES_PATH_CONSTRAINTS",
       -23: "ROBOT_STATE_STALE", -30: "ABORT"}


def err_name(c):
    return ERR.get(int(c), f"code={c}")


def fmt_time(t):
    if isinstance(t, (int, float)):
        return float(t)
    return getattr(t, "sec", 0) + getattr(t, "nanosec", 0) * 1e-9


# --------------------------------------------------------------------------- ROS
class Ros:
    """Holds node + clients; spins in a daemon thread. Dash threads poll futures."""

    def __init__(self):
        self.node = Node("moveit_web_panel")
        self.js_lock = threading.Lock()
        self.latest_js: dict[str, float] = {}
        self.node.create_subscription(JointState, "/joint_states", self._on_js, 50)
        self.fk = self.node.create_client(GetPositionFK, "/compute_fk")
        self.mg = ActionClient(self.node, MoveGroup, "/move_action")
        self.exe = SingleThreadedExecutor()
        self.exe.add_node(self.node)
        self._t = threading.Thread(target=self.exe.spin, daemon=True)
        self._t.start()

    def _on_js(self, msg: JointState):
        with self.js_lock:
            for n, p in zip(msg.name, msg.position):
                self.latest_js[n] = float(p)

    def current_joints(self, side):
        with self.js_lock:
            return [self.latest_js.get(n, 0.0) for n in JOINTS[side]]

    def ready(self):
        return self.mg.server_is_ready() and self.fk.service_is_ready()

    def fk_pose(self, side):
        """Return (xyz, rpy) of the arm tip, or None."""
        if not self.fk.service_is_ready():
            return None
        req = GetPositionFK.Request()
        req.fk_link_names = [TIP[side]]
        req.header.frame_id = ROOT
        js = JointState()
        js.name = list(self.latest_js.keys()) if self.latest_js else JOINTS[side]
        js.position = list(self.latest_js.values()) if self.latest_js else [0.0] * 7
        rs = RobotState(); rs.joint_state = js
        req.robot_state = rs
        fut = self.fk.call_async(req)
        if not self._wait(fut, 0.5):
            return None
        r = fut.result()
        if r.error_code.val != 1 or not r.pose_stamped:
            return None
        p = r.pose_stamped[0].pose.position
        q = r.pose_stamped[0].pose.orientation
        rpy = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")
        return ((round(p.x, 3), round(p.y, 3), round(p.z, 3)),
                (round(float(rpy[0]), 3), round(float(rpy[1]), 3), round(float(rpy[2]), 3)))

    @staticmethod
    def _wait(fut, timeout):
        end = time.time() + timeout
        while time.time() < end:
            if fut.done():
                return True
            time.sleep(0.01)
        return fut.done()

    # ----- goal builders -----------------------------------------------------
    @staticmethod
    def _joint_constraints(side, q):
        c = Constraints()
        for n, v in zip(JOINTS[side], q):
            jc = JointConstraint()
            jc.joint_name = n
            jc.position = float(v)
            jc.tolerance_above = 1e-3
            jc.tolerance_below = 1e-3
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        return c

    @staticmethod
    def _pose_constraints(side, xyz, rpy):
        c = Constraints()
        tip = TIP[side]
        # position: tiny box at target point
        pc = PositionConstraint()
        pc.link_name = tip
        pc.header = Header(frame_id=ROOT)
        box = SolidPrimitive(); box.type = SolidPrimitive.BOX
        box.dimensions = [0.001, 0.001, 0.001]
        pc.constraint_region.primitives = [box]
        bp = Pose(); bp.position.x, bp.position.y, bp.position.z = xyz
        bp.orientation.w = 1.0
        pc.constraint_region.primitive_poses = [bp]
        pc.weight = 1.0
        c.position_constraints.append(pc)
        # orientation
        oc = OrientationConstraint()
        oc.link_name = tip
        oc.header = Header(frame_id=ROOT)
        qx, qy, qz, qw = Rotation.from_euler("xyz", rpy).as_quat()
        oc.orientation = Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw))
        oc.absolute_x_axis_tolerance = oc.absolute_y_axis_tolerance = oc.absolute_z_axis_tolerance = 1e-2
        oc.weight = 1.0
        c.orientation_constraints.append(oc)
        return c

    def send(self, side, goal_type, named, joint_str, pose_str,
             plan_only, planner_id, ptime, vel_scale):
        """Send one MoveGroup goal. Returns a result dict."""
        if not self.mg.server_is_ready():
            return {"ok": False, "err": "move_group /move_action not ready"}
        req = MotionPlanRequest()
        req.group_name = SIDES[side]
        req.planner_id = planner_id or "RRTConnect"
        req.num_planning_attempts = 5
        req.allowed_planning_time = float(ptime or 5.0)
        req.max_velocity_scaling_factor = float(vel_scale or 0.1)
        req.max_acceleration_scaling_factor = float(vel_scale or 0.1)
        req.start_state = RobotState(); req.start_state.is_diff = True  # use current
        try:
            if goal_type == "named":
                req.goal_constraints.append(self._joint_constraints(side, NAMED[named]))
            elif goal_type == "joint":
                q = [float(x) for x in joint_str.replace("[", "").replace("]", "").split(",")]
                if len(q) != 7:
                    return {"ok": False, "err": f"joint needs 7 values, got {len(q)}"}
                req.goal_constraints.append(self._joint_constraints(side, q))
            elif goal_type == "pose":
                vals = [float(x) for x in pose_str.replace("[", "").replace("]", "").split(",")]
                if len(vals) != 6:
                    return {"ok": False, "err": f"pose needs x,y,z,r,p,y (6), got {len(vals)}"}
                req.goal_constraints.append(self._pose_constraints(side, vals[:3], vals[3:6]))
            else:
                return {"ok": False, "err": f"unknown goal type {goal_type}"}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "err": f"parse error: {e}"}

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = bool(plan_only)

        t0 = time.time()
        gh_fut = self.mg.send_goal_async(goal)
        if not self._wait(gh_fut, 15):
            return {"ok": False, "err": "goal send timeout"}
        gh = gh_fut.result()
        if not gh.accepted:
            return {"ok": False, "err": "goal REJECTED by server"}
        res_fut = gh.get_result_async()
        if not self._wait(res_fut, 90):
            return {"ok": False, "err": "result timeout"}
        res = res_fut.result().result
        tr = res.planned_trajectory.joint_trajectory
        pts = len(tr.points) if tr and tr.points else 0
        return {"ok": True, "code": res.error_code.val, "err": err_name(res.error_code.val),
                "ptime": fmt_time(res.planning_time), "pts": pts, "wall": time.time() - t0,
                "plan_only": bool(plan_only)}


# --------------------------------------------------------------------------- Dash
rclpy.init()
ros = Ros()
HIST: list[dict] = []
HIST_LOCK = threading.Lock()

app = dash.Dash(__name__, title="OpenArm MoveIt Panel")
app.layout = html.Div([dcc.Interval(id="tick", interval=500), html.Div([
    html.H2("OpenArm 原生 MoveIt 测试面板"),
    html.Div([html.B("状态: "), html.Span(id="status", style={"color": "#888"})]),
    html.Div([html.B("当前末端位姿: "), html.Span(id="ee", style={"fontFamily": "monospace"})]),
    html.Hr(),
    html.Div([
        html.Label("规划组"),
        dcc.Dropdown(id="side", options=[{"label": "左 left_arm", "value": "left"},
                                         {"label": "右 right_arm", "value": "right"}],
                     value="left", clearable=False, style={"width": "160px"}),
    ], style={"display": "flex", "gap": "24px", "alignItems": "center", "marginBottom": "8px"}),
    html.Div([
        html.Label("目标类型"),
        dcc.Dropdown(id="gtype", options=[{"label": "命名状态", "value": "named"},
                                          {"label": "关节目标 (rad)", "value": "joint"},
                                          {"label": "位姿目标 (x,y,z,r,p,y)", "value": "pose"}],
                     value="named", clearable=False, style={"width": "260px"}),
    ], style={"marginBottom": "8px"}),
    html.Div([
        html.Span("命名状态", style={"width": "90px", "display": "inline-block"}),
        dcc.Dropdown(id="named", options=[{"label": k, "value": k} for k in NAMED],
                     value="home", clearable=False, style={"width": "140px"}),
    ], id="named-row", style={"marginBottom": "8px"}),
    html.Div([
        html.Span("关节值", style={"width": "90px", "display": "inline-block"}),
        dcc.Input(id="jval", value="0,0,0,0,0,0,0", style={"width": "320px", "fontFamily": "monospace"}),
        html.Button("取当前", id="jcur", n_clicks=0, style={"marginLeft": "8px"}),
    ], id="joint-row", style={"display": "none", "marginBottom": "8px"}),
    html.Div([
        html.Span("位姿", style={"width": "90px", "display": "inline-block"}),
        dcc.Input(id="pval", value="0.2,0.05,0.4,0,0,0", style={"width": "320px", "fontFamily": "monospace"}),
        html.Button("取当前末端", id="pcur", n_clicks=0, style={"marginLeft": "8px"}),
    ], id="pose-row", style={"display": "none", "marginBottom": "8px"}),
    html.Div([
        html.Label("planner"), dcc.Input(id="planner", value="RRTConnect", style={"width": "110px"}),
        html.Label("planning_time(s)", style={"marginLeft": "12px"}), dcc.Input(id="ptime", value="5", style={"width": "60px"}),
        html.Label("vel/accel scaling", style={"marginLeft": "12px"}), dcc.Input(id="vscale", value="0.1", style={"width": "60px"}),
        dcc.Checklist(id="ponly", options=[{"label": "仅规划(plan_only)", "value": "p"}], style={"marginLeft": "12px"}),
    ], style={"display": "flex", "gap": "6px", "alignItems": "center", "marginBottom": "10px"}),
    html.Button("▶ 规划 / 执行", id="go", n_clicks=0,
                style={"fontSize": "16px", "padding": "8px 20px", "background": "#2e7d32", "color": "white"}),
    html.Div(id="result", style={"marginTop": "10px", "fontWeight": "bold", "fontFamily": "monospace"}),
    html.Hr(),
    html.H4("关节角 (当前, rad)"),
    dcc.Graph(id="jchart", config={"displayModeBar": False}, style={"height": "240px"}),
    html.H4("历史"),
    html.Div(id="history"),
])])


def _show_row(gtype):
    return {"named": ("block", "none", "none"), "joint": ("none", "flex", "none"),
            "pose": ("none", "none", "flex")}.get(gtype, ("none", "none", "none"))


@app.callback(Output("named-row", "style"), Output("joint-row", "style"), Output("pose-row", "style"),
              Input("gtype", "value"))
def _switch(gtype):
    n, j, p = _show_row(gtype)
    base = {"marginBottom": "8px"}
    return {**base, "display": n}, {**base, "display": j}, {**base, "display": p}


@app.callback(Output("status", "children"), Output("status", "style"),
              Output("ee", "children"), Output("jchart", "figure"),
              Input("tick", "n_intervals"), Input("side", "value"))
def _tick(_, side):
    ok = ros.ready()
    status = ("move_group + compute_fk 在线 ✓" if ok else "等待 move_group / compute_fk …")
    col = {"color": "#2e7d32"} if ok else {"color": "#c62828"}
    ee = ros.fk_pose(side)
    ee_txt = f"xyz=({ee[0][0]},{ee[0][1]},{ee[0][2]}) rpy=({ee[1][0]},{ee[1][1]},{ee[1][2]})" if ee else "(unavailable)"
    q = ros.current_joints(side)
    names = [n.replace(f"openarm_{side}_", "") for n in JOINTS[side]]
    fig = {"data": [go.Bar(x=names, y=q, marker_color="#1565c0")],
           "layout": {"margin": {"l": 40, "r": 10, "t": 10, "b": 30}, "yaxis": {"range": [-3.2, 3.2]}}}
    return status, col, ee_txt, fig


@app.callback(Output("jval", "value"), Input("jcur", "n_clicks"), State("side", "value"),
              prevent_initial_call=True)
def _fill_joints(_, side):
    return ",".join(f"{x:.4f}" for x in ros.current_joints(side))


@app.callback(Output("pval", "value"), Input("pcur", "n_clicks"), State("side", "value"),
              prevent_initial_call=True)
def _fill_pose(_, side):
    ee = ros.fk_pose(side)
    if not ee:
        return dash.no_update
    return f"{ee[0][0]},{ee[0][1]},{ee[0][2]},{ee[1][0]},{ee[1][1]},{ee[1][2]}"


@app.callback(Output("result", "children"), Output("history", "children"),
              Input("go", "n_clicks"),
              State("side", "value"), State("gtype", "value"), State("named", "value"),
              State("jval", "value"), State("pval", "value"), State("ponly", "value"),
              State("planner", "value"), State("ptime", "value"), State("vscale", "value"),
              prevent_initial_call=True)
def _go(_, side, gtype, named, jval, pval, ponly, planner, ptime, vscale):
    res = ros.send(side, gtype, named, jval or "", pval or "", bool(ponly), planner, ptime, vscale)
    if not res.get("ok"):
        txt = f"❌ {res.get('err', 'error')}"
    else:
        tag = "规划(未执行)" if res["plan_only"] else "规划+执行"
        good = res["code"] == 1
        txt = f"{'✅' if good else '⚠️'} [{tag}] {res['err']}  | planning_time={res['ptime']:.3f}s  | 轨迹点={res['pts']}  | wall={res['wall']:.2f}s"
    with HIST_LOCK:
        HIST.insert(0, {"side": side, "gtype": gtype, "named": named,
                        "code": res.get("err", res.get("err", "?")),
                        "ptime": res.get("ptime", "-"), "pts": res.get("pts", "-"),
                        "ok": res.get("ok")})
        del HIST[12:]
        rows = [html.Tr([html.Th("side"), html.Th("type"), html.Th("named/goal"),
                         html.Th("result"), html.Th("time(s)"), html.Th("pts")])]
        for h in HIST:
            col = "#2e7d32" if (h["code"] == "SUCCESS") else "#c62828"
            rows.append(html.Tr([
                html.Td(h["side"]), html.Td(h["gtype"]), html.Td(h.get("named", "")),
                html.Td(h["code"], style={"color": col}),
                html.Td(f"{h['ptime']:.3f}" if isinstance(h["ptime"], (int, float)) else str(h["ptime"])),
                html.Td(str(h["pts"])),
            ]))
    return txt, html.Table(rows, style={"borderCollapse": "collapse", "fontSize": "13px"})


if __name__ == "__main__":
    # give move_group a moment, then serve
    time.sleep(1.0)
    (getattr(app, "run", None) or app.run_server)(host="0.0.0.0", port=8050, debug=False)
