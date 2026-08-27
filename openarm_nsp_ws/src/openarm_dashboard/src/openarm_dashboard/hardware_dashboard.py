#!/usr/bin/env python3
# Copyright 2026 Enactic, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Control端 A — real-hardware dashboard (Dash :8050).

Brings up CAN, enables both arms into zero-torque (safe default), streams
feedback into a live position chart, and exposes:
  - state-machine buttons: enable / hold / home / disable
  - teach-and-replay IK:  record start → record end → go to start → IK run

Teach workflow (the IK test you asked for):
  1. drag arm to pose A (zero-torque) → [记录起点] (records joints)
  2. drag arm to pose B             → [记录终点]
  3. [回起点]  — smooth joint-space move back to A (like home, ~2s)
  4. [IK到终点] — FK(A),FK(B) → plan_cartesian → go/no-go gate → execute, end in HOLD

All CAN calls run in the ArmController worker thread. IK planning runs in a
background thread so the UI never blocks. ``--sim`` runs without hardware.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque

import dash
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, dcc, html, no_update

from openarm_pinocchio_nsp.cartesian_planner import (
    Waypoint, check_traj_smoothness, ease_in_out_retime, fit_arc,
    pose_replay_traj, plan_cartesian,
)
from openarm_pinocchio_nsp.bspline_planner import BSplineOptimizer
from openarm_pinocchio_nsp import kinematics as _kin
from openarm_pinocchio_nsp.kinematics import PinocchioModel
from openarm_pinocchio_nsp.urdf_path import resolve_urdf_path

from .arm_controller import (ARM_DOF, ArmController, ArmMode, bringup_can,
                              can_is_up)
from .robot_state import DataBuffer, RobotState

# channel map SWAPPED 2026-08-27 (cables physically exchanged): left=ch1
CAN_MAP = {"left": "can_slot1_ch1", "right": "can_slot1_ch0"}
SIDES = ("left", "right")

# mode buttons (simple transitions)
MODE_BUTTONS = {
    "enable": ArmMode.ZERO_TORQUE,
    "hold": ArmMode.ENABLED_HOLD,
    "home": ArmMode.HOMING,
    "disable": ArmMode.DISABLED,
}
# action buttons (special handling in _on_button)
ACTION_BUTTONS = ["record_start", "record_end", "go_start", "go_end", "line_run",
                  "arc_add", "arc_clear", "arc_start", "arc_run", "replay_run", "bspline_run"]
ALL_BUTTONS = list(MODE_BUTTONS) + ACTION_BUTTONS

LABEL = {
    "enable": "使能(零力矩)", "hold": "抱住", "home": "归零⚠", "disable": "失能",
    "record_start": "📍记录起点", "record_end": "📍记录终点",
    "go_start": "↩回起点", "go_end": "▶到终点",
    "line_run": "▶直线运动",
    "arc_add": "📍弧线加点", "arc_clear": "🗑清空弧线",
    "arc_start": "↩弧线回起点", "arc_run": "▶弧线IK", "replay_run": "▶关节回放",
    "bspline_run": "▶B样条优化",
}
STYLE = {
    "enable": "#4CAF50", "hold": "#2196F3", "home": "#FF9800", "disable": "#f44336",
    "record_start": "#9C27B0", "record_end": "#9C27B0",
    "go_start": "#00BCD4", "go_end": "#E91E63",
    "line_run": "#FF5722",
    "arc_add": "#795548", "arc_clear": "#9E9E9E",
    "arc_start": "#00BCD4", "arc_run": "#673AB7", "replay_run": "#009688",
    "bspline_run": "#3F51B5",
}
BUSY_MODES = {ArmMode.HOMING.value, ArmMode.GO_START.value, ArmMode.TRACKING.value}

# ── 弧线运动限速（界面可实时调整：改完点 [▶弧线IK] 即生效，无需重启）────
# slowdown: 整体放慢倍数 (>1 更慢)，总时长 × 此值，所有关节速度按比例下降。
# cap:      每关节转速硬限幅 (rad/s)，保证 easing 后实际峰值速度 ≤ 此值。
#           标量 → 所有关节同一上限。1.0 rad/s ≈ 柔顺可控，仍猛就降到 0.5。
# 这里既是默认值，也是 UI 回调写回的持久存储（dict 原地修改，跨回调/线程可见）。
ARC_TUNE = {"slowdown": 2.0, "cap": 1.0, "mode": "fast"}

# 弧线 IK 模式（界面上可切换；两种方法并存）
#   fast     优化版：安全点跳过 Stage 2，~1s 规划，速度快
#   thorough 完整版：每点都跑满 Stage 2(null_iters=12)，裕度最饱满，~7-10s
ARC_MODES = {
    "fast":     {"skip": True,  "null_iters": 6,  "label": "快速"},
    "thorough": {"skip": False, "null_iters": 12, "label": "精细"},
}

# ── 关节回放（纯关节空间示教复演，无 IK）────────────────────────────────
# 界面输入：max_speed 每关节最大角速度 (rad/s)；freq 控制频率/采样密度 (Hz)。
# 时长由"每段最大关节行程 / max_speed"决定（保证无一关节超速），再按 1/freq 均匀重采样。
REPLAY_TUNE = {"max_speed": 1.0, "freq": 100.0}


def _downsample(times, data, max_points=80):
    n = len(times)
    if n <= max_points:
        return times, data
    step = max(1, n // max_points)
    return times[::step], data[::step]


def _pos_fig(times, pos, title):
    fig = go.Figure()
    t, p = _downsample(times, pos)
    for j in range(ARM_DOF):
        fig.add_trace(go.Scattergl(x=t, y=p[:, j], name=f"j{j+1}",
                                   mode="lines", line=dict(width=1.5)))
    rng = [float(np.min(p)), float(np.max(p))] if len(p) else None
    fig.update_layout(
        title=title, height=260, margin=dict(l=40, r=20, t=36, b=30),
        uirevision=title, legend=dict(orientation="h", y=-0.15, x=0),
        template="plotly_white", yaxis=dict(range=rng) if rng else {},
        xaxis=dict(title="t (s)"))
    return fig


def _err_fig(times, pos_err_mm, ori_err_deg, title):
    """IK tracking-error chart: planned-vs-actual EE pose error over time."""
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=times, y=pos_err_mm, name="位置误差 (mm)",
                               mode="lines", line=dict(color="#E91E63", width=1.6)))
    fig.add_trace(go.Scattergl(x=times, y=ori_err_deg, name="姿态误差 (°)",
                               mode="lines", yaxis="y2", line=dict(color="#3F51B5", width=1.6)))
    fig.update_layout(
        title=title, height=200, margin=dict(l=40, r=40, t=30, b=24),
        uirevision=title, legend=dict(orientation="h", y=-0.25, x=0),
        template="plotly_white",
        yaxis=dict(title="mm", rangemode="tozero"),
        yaxis2=dict(title="°", overlaying="y", side="right", rangemode="tozero"),
        xaxis=dict(title="t (s)"))
    return fig


def build_app(rs: RobotState, buf: DataBuffer, controllers: dict, models: dict, sim: bool):
    app = dash.Dash(__name__, title="OpenArm Dashboard")
    app.layout = html.Div([
        html.Div([
            html.H2(f"OpenArm Hardware Dashboard ({'SIM' if sim else 'REAL'})",
                    style={"display": "inline-block"}),
            html.Span(id="can-bar", style={"marginLeft": "24px", "fontWeight": "bold", "color": "#666"}),
        ]),
        html.Div([
            html.B("弧线限速:", style={"marginRight": "10px"}),
            html.Label("放慢倍数(>1更慢)", style={"fontSize": "13px"}),
            dcc.Input(id="arc-slowdown", type="number", value=ARC_TUNE["slowdown"],
                      min=0.5, step=0.5, debounce=True,
                      style={"width": "64px", "margin": "0 10px 0 6px"}),
            html.Label("转速限幅(rad/s)", style={"fontSize": "13px"}),
            dcc.Input(id="arc-cap", type="number", value=ARC_TUNE["cap"],
                      min=0.1, step=0.1, debounce=True,
                      style={"width": "64px", "margin": "0 10px 0 6px"}),
            html.Span("IK模式:", style={"marginLeft": "10px", "fontSize": "13px"}),
            dcc.RadioItems(
                id="arc-mode",
                options=[{"label": "快速(~1s,优化版)", "value": "fast"},
                         {"label": "精细(7-10s,完整Stage2,最佳裕度)", "value": "thorough"}],
                value=ARC_TUNE["mode"], inline=True,
                labelStyle={"marginRight": "12px", "fontSize": "13px"}),
            html.Span("｜关节回放:", style={"marginLeft": "8px", "fontSize": "13px", "color": "#009688"}),
            html.Label("最大角速度(rad/s)", style={"fontSize": "13px"}),
            dcc.Input(id="rep-maxspeed", type="number", value=REPLAY_TUNE["max_speed"],
                      min=0.1, step=0.1, debounce=True,
                      style={"width": "60px", "margin": "0 10px 0 6px"}),
            html.Label("控制频率(Hz)", style={"fontSize": "13px"}),
            dcc.Input(id="rep-freq", type="number", value=REPLAY_TUNE["freq"],
                      min=10, max=250, step=10, debounce=True,
                      style={"width": "60px", "margin": "0 10px 0 6px"}),
            html.Span(id="arc-tune-fb", style={"fontSize": "12px", "color": "#336"}),
        ], style={"margin": "6px 0", "padding": "6px 12px", "background": "#eef3ff",
                  "borderRadius": "6px", "fontSize": "14px", "alignItems": "center",
                  "display": "flex", "flexWrap": "wrap"}),
        html.Div([arm_block(s) for s in SIDES],
                 style={"display": "flex", "gap": "24px", "flexWrap": "wrap"}),
        html.Hr(),
        html.Div([html.B("事件日志"), html.Pre(id="log", style={
            "fontSize": "11px", "maxHeight": "140px", "overflowY": "scroll",
            "background": "#f6f6f6", "padding": "8px", "marginTop": "6px"})]),
        html.Div(id="btn-feedback", style={"display": "none"}),
        dcc.Interval(id="tick", interval=500),
        dcc.Interval(id="err-tick", interval=200),   # IK 跟踪误差刷新
    ])

    # ---- arc speed tuning + IK mode + replay params (live, no restart) ----
    @app.callback(
        Output("arc-tune-fb", "children"),
        [Input("arc-slowdown", "value"), Input("arc-cap", "value"),
         Input("arc-mode", "value"),
         Input("rep-maxspeed", "value"), Input("rep-freq", "value")],
    )
    def _set_arc_tune(slowdown, cap, mode, maxspeed, freq):
        if isinstance(slowdown, (int, float)) and slowdown > 0:
            ARC_TUNE["slowdown"] = float(slowdown)
        if isinstance(cap, (int, float)) and cap > 0:
            ARC_TUNE["cap"] = float(cap)
        if mode in ARC_MODES:
            ARC_TUNE["mode"] = mode
        if isinstance(maxspeed, (int, float)) and maxspeed > 0:
            REPLAY_TUNE["max_speed"] = float(maxspeed)
        if isinstance(freq, (int, float)) and 1 <= freq <= 500:
            REPLAY_TUNE["freq"] = float(freq)
        ml = ARC_MODES[ARC_TUNE["mode"]]["label"]
        return (f"✓ {ml}IK 放慢×{ARC_TUNE['slowdown']} 限幅{ARC_TUNE['cap']} ｜ "
                f"回放 速度≤{REPLAY_TUNE['max_speed']}rad/s {REPLAY_TUNE['freq']:.0f}Hz")

    def _warn_traj(side: str, times, q_path):
        """Warn-only continuity/smoothness check. Logs warnings, never blocks."""
        chk = check_traj_smoothness(times, q_path)
        for msg in chk.warnings:
            rs.log(f"{side}: {msg}")
        return chk

    def _ik_and_go(side: str):
        """Single IK: FK(taught_end) -> IK (warm-started from start) -> joint-interpolate to it.
        Runs in a background thread (ik_nsp is ~10-50ms but keep UI responsive)."""
        c = controllers[side]
        qe = c.taught_end
        if qe is None:
            rs.log(f"{side}: 请先记录终点")
            return
        qs = c.taught_start if c.taught_start is not None else qe
        try:
            m = models[side]
            pos_e, quat_e = m.fk(qe)
            r = m.ik_nsp(pos_e, quat_e, q_init=qs)
        except Exception as e:  # noqa: BLE001
            rs.log(f"{side}: IK异常 {type(e).__name__}: {e}")
            return
        if not r.converged:
            rs.log(f"{side}: IK未收敛,拒绝对终点 (err={r.pos_err_mm:.2f}mm)")
            return
        rs.log(f"{side}: IK解终点 σ_min={r.sigma_min:.3f} 裕度={r.joint_margin:.3f}")
        c.request_move_to(r.q, "IK到终点")

    def _line_run(side: str):
        """Straight-line motion: FK(start,end) -> plan_cartesian(linear densify + warm-start IK)
        -> execute WITHOUT gate check (reliable like single IK)."""
        c = controllers[side]
        qs, qe = c.taught_start, c.taught_end
        if qs is None or qe is None:
            rs.log(f"{side}: 请先记录起点和终点")
            return
        rs.log(f"{side}: 直线运动规划中...")
        try:
            m = models[side]
            pos_s, quat_s = m.fk(qs)
            pos_e, quat_e = m.fk(qe)
            result = plan_cartesian(
                m, [Waypoint(pos_s, quat_s), Waypoint(pos_e, quat_e)],
                q_init=qs, smooth=False)
        except Exception as e:  # noqa: BLE001
            rs.log(f"{side}: 直线运动异常 {type(e).__name__}: {e}")
            return
        if not result.success:
            rs.log(f"{side}: 直线IK失败 @sample {result.break_index}")
            return
        # 不检查安全门——直接执行（warm-start IK 已保证关节连续）
        traj = (result.times, result.q_path)
        _warn_traj(side, *traj)
        _clear_err(side)
        if c.request_transition(ArmMode.TRACKING, traj):
            rs.log(f"{side}: ▶直线运动 {len(result.q_path)}点 "
                   f"(不检查安全门,直接执行)")

    def _arc_run(side: str):
        """Arc IK: FK(all arc control points) -> fit_arc spline -> plan_cartesian
        (warm-start IK, NO gate) -> ease-in/ease-out retime -> TRACKING.

        Rest-to-rest velocity profile (从静止启动 + 到位降速停止): the joint
        velocity is 0 at both ends, peaking mid-arc. Spatial path unchanged.
        """
        c = controllers[side]
        pts = list(c.arc_points)
        if len(pts) < 2:
            rs.log(f"{side}: 弧线至少需要2个点(当前{len(pts)})")
            return
        mode = ARC_TUNE.get("mode", "fast")
        cfg = ARC_MODES[mode]
        _kin._STAGE2_SKIP = cfg["skip"]          # fast=跳过安全点的 Stage2; thorough=每点都跑满
        rs.log(f"{side}: 弧线拟合+{cfg['label']}IK中... ({len(pts)}控制点, "
               f"{'跳过Stage2' if cfg['skip'] else '完整Stage2'})")
        t_plan0 = time.time()
        try:
            m = models[side]
            wps = [Waypoint(*m.fk(q)) for q in pts]
            arc = fit_arc(wps, n_dense=100)
            result = plan_cartesian(m, arc, pts[0], presampled=True,
                                    null_iters=cfg["null_iters"])
        except Exception as e:  # noqa: BLE001
            rs.log(f"{side}: 弧线IK异常 {type(e).__name__}: {e}")
            return
        t_plan = time.time() - t_plan0
        if not result.success:
            rs.log(f"{side}: 弧线IK失败 @sample {result.break_index} (规划{t_plan:.1f}s) "
                   f"→ 试试切换『精细』模式，或调整示教点")
            return
        # 不检查安全门——直接执行（和直线运动一样，warm-start IK 保证关节连续）
        # 叠加：①静止启停（起末速度=0）②整体放慢 ③关节转速硬限幅（值来自界面 ARC_TUNE）
        t_orig = result.times[-1] - result.times[0]
        times, q_path = ease_in_out_retime(
            result.times, result.q_path,
            slowdown=ARC_TUNE["slowdown"], vmax_cap=ARC_TUNE["cap"])
        dur = times[-1] - times[0]
        # 实际峰值速度（与 _step_track 线性插值一致）
        peak = np.max(np.abs(np.diff(q_path, axis=0) / np.diff(times)[:, None]))
        chk = _warn_traj(side, times, q_path)
        _clear_err(side)
        if c.request_transition(ArmMode.TRACKING, (times, q_path)):
            rs.log(f"{side}: ▶弧线执行 {len(q_path)}点 "
                   f"[{cfg['label']}] 规划{t_plan:.1f}s 时长 {t_orig:.2f}→{dur:.2f}s "
                   f"峰速 {peak:.2f}rad/s σ_min={result.min_sigma:.3f} "
                   f"(不检查安全门,静止启停,限幅{ARC_TUNE['cap']})")

    def _replay_run(side: str):
        """Task-space teach-replay: record 6D pose (FK of taught joints) -> time by
        max joint speed -> interpolate pose at control freq -> IK each mid-point
        ("backward calculation") -> TRACKING. EE follows straight lines in pose space."""
        c = controllers[side]
        pts = list(c.arc_points)
        if len(pts) < 2:
            rs.log(f"{side}: 关节回放至少需要2个点(当前{len(pts)}，用 [📍弧线加点] 记录)")
            return
        ms, fr = REPLAY_TUNE["max_speed"], REPLAY_TUNE["freq"]
        t_plan0 = time.time()
        try:
            m = models[side]
            poses = [Waypoint(*m.fk(q)) for q in pts]      # 只用 6D 位姿定义路径
            q_seed = np.asarray(rs.snapshot(side).position, dtype=float)  # 从当前臂位姿热启动
            null_iters = ARC_MODES[ARC_TUNE["mode"]]["null_iters"]
            traj = pose_replay_traj(m, poses, q_seed, max_speed=ms, freq=fr,
                                    null_iters=null_iters)
        except Exception as e:  # noqa: BLE001
            rs.log(f"{side}: 关节回放异常 {type(e).__name__}: {e}")
            return
        t_plan = time.time() - t_plan0
        if traj is None:
            rs.log(f"{side}: 关节回放IK失败 (规划{t_plan:.1f}s) → 某点不可达/分支断裂，"
                   f"试切换『精细』或调整示教点")
            return
        times, q_path = traj
        dur = times[-1] - times[0]
        peak = float(np.max(np.abs(np.diff(q_path, axis=0) / np.diff(times)[:, None])))
        _warn_traj(side, times, q_path)
        _clear_err(side)
        if c.request_transition(ArmMode.TRACKING, (times, q_path)):
            rs.log(f"{side}: ▶关节回放(位姿插值+IK) {len(q_path)}点 时长 {dur:.2f}s "
                   f"峰速 {peak:.2f}rad/s @{fr:.0f}Hz 规划{t_plan:.1f}s "
                   f"({len(pts)}示教点,末端走直线)")

    def _bspline_run(side: str):
        """B-spline optimization: FK(arc points) -> optimize -> post-verify -> TRACKING."""
        c = controllers[side]
        pts = list(c.arc_points)
        if len(pts) < 2:
            rs.log(f"{side}: 至少需要2个弧线点(当前{len(pts)})")
            return
        rs.log(f"{side}: B样条优化中... ({len(pts)}控制点)")
        try:
            opt = BSplineOptimizer(resolve_urdf_path(), side)
            wps = np.array([opt.fk_pos(q) for q in pts])
            spline, q_path, res = opt.optimize(
                wps, pts[0], duration=3.0, n_samples=30, maxiter=30, w3=0.0)
            v = opt.post_verify(q_path, duration=3.0, n_collision_check=50)
        except Exception as e:  # noqa: BLE001
            rs.log(f"{side}: B样条异常 {type(e).__name__}: {e}")
            return
        if not res.success:
            rs.log(f"{side}: B样条优化未收敛")
            return
        if not v["passed"]:
            rs.log(f"{side}: ✗后验证失败 碰撞={v['collisions']} "
                   f"步长={v['max_adjacent_step']:.3f}")
            return
        n = len(q_path)
        times = np.linspace(0, 3.0, n).tolist()
        _warn_traj(side, times, list(q_path))
        _clear_err(side)
        if c.request_transition(ArmMode.TRACKING, (times, list(q_path))):
            rs.log(f"{side}: ▶B样条执行 {n}点 碰撞={v['collisions']} "
                   f"速度={v['max_velocity']:.2f} 跟踪<3mm")

    # ---- position charts ----
    @app.callback([Output(f"fig-pos-{s}", "figure") for s in SIDES], Input("tick", "n_intervals"))
    def _graphs(_n):
        return [_pos_fig(*buf.series(s)[:2], f"{s} 关节位置 (rad)") for s in SIDES]

    # ---- IK tracking-error charts (planned-vs-actual EE pose) ----
    err_buf = {s: deque(maxlen=150) for s in SIDES}
    err_t0 = time.time()

    def _clear_err(side: str):
        err_buf[side].clear()

    @app.callback([Output(f"fig-err-{s}", "figure") for s in SIDES], Input("err-tick", "n_intervals"))
    def _err_graphs(_n):
        out = []
        for s in SIDES:
            c, m = controllers[s], models.get(s)
            tq = c.tracking_q()
            if tq is not None and m is not None:
                try:
                    pp, pq = m.fk(tq[0])           # planned EE pose
                    ap, aq = m.fk(tq[1])           # actual   EE pose
                    pos_err = float(np.linalg.norm(pp - ap) * 1000.0)          # mm
                    dot = float(np.clip(abs(np.dot(pq, aq)), 0.0, 1.0))
                    ori_err = float(np.degrees(2.0 * np.arccos(dot)))          # deg
                    err_buf[s].append((time.time() - err_t0, pos_err, ori_err))
                except Exception:  # noqa: BLE001
                    pass
            if err_buf[s]:
                a = np.array(err_buf[s])
                out.append(_err_fig(a[:, 0].tolist(), a[:, 1].tolist(), a[:, 2].tolist(),
                                     f"{s} IK跟踪误差 (规划 vs 实际)"))
            else:
                out.append(_err_fig([], [], [], f"{s} IK跟踪误差 (规划 vs 实际)"))
        return out

    # ---- status + temp + progress + taught + CAN + log ----
    @app.callback(
        [Output(f"status-{s}", "children") for s in SIDES]
        + [Output(f"temp-{s}", "children") for s in SIDES]
        + [Output(f"progress-{s}", "value") for s in SIDES]
        + [Output(f"taught-{s}", "children") for s in SIDES]
        + [Output("can-bar", "children"), Output("log", "children")],
        Input("tick", "n_intervals"),
    )
    def _status(_n):
        out = []
        for s in SIDES:
            st = rs.snapshot(s)
            out.append(f"模式: {st.mode}  |  使能: {st.enabled}")
        for s in SIDES:
            st = rs.snapshot(s)
            mx = float(np.max(st.tmos)) if len(st.tmos) else 0.0
            color = "🔴" if mx > 70 else ("🟡" if mx > 55 else "🟢")
            out.append(f"{color} MOS {mx:.0f}°C")
        for s in SIDES:
            out.append(float(rs.snapshot(s).track_progress * 100))
        for s in SIDES:
            c = controllers[s]
            ss = "✅" if c.taught_start is not None else "⬜"
            se = "✅" if c.taught_end is not None else "⬜"
            out.append(f"起点{ss}  终点{se}  弧线:{len(c.arc_points)}点")
        if sim:
            out.append("CAN: SIM")
        else:
            out.append("CAN: " + "  |  ".join(
                f"{s}({CAN_MAP[s][-3:]}): {'✅UP' if rs.can_up.get(CAN_MAP[s]) else '❌DOWN'}"
                for s in SIDES))
        out.append("\n".join(rs.recent_messages(12)))
        return out

    # ---- button enabled/disabled ----
    @app.callback(
        [Output(f"btn-{s}-{b}", "disabled") for s in SIDES for b in ALL_BUTTONS],
        Input("tick", "n_intervals"),
    )
    def _btn_state(_n):
        out = []
        for s in SIDES:
            mode = rs.snapshot(s).mode
            busy = mode in BUSY_MODES
            c = controllers[s]
            for b in ALL_BUTTONS:
                if b == "disable":
                    out.append(False)
                elif busy:
                    out.append(True)
                elif b == "go_start":
                    out.append(c.taught_start is None)
                elif b in ("go_end", "line_run"):
                    out.append(c.taught_end is None)
                elif b == "arc_start":
                    out.append(len(c.arc_points) < 1)
                elif b in ("arc_run", "replay_run", "bspline_run"):
                    out.append(len(c.arc_points) < 2)
                elif b in MODE_BUTTONS and MODE_BUTTONS[b].value == mode:
                    out.append(True)
                else:
                    out.append(False)
        return out

    # ---- button clicks ----
    @app.callback(
        Output("btn-feedback", "children"),
        [Input(f"btn-{s}-{b}", "n_clicks") for s in SIDES for b in ALL_BUTTONS],
        prevent_initial_call=True,
    )
    def _on_button(*clicks):
        ctx = dash.callback_context
        if not ctx.triggered or not ctx.triggered[0]["value"]:
            return no_update
        btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
        _, side, b = btn_id.split("-")
        c = controllers[side]
        if b in MODE_BUTTONS:
            c.request_transition(MODE_BUTTONS[b])
        elif b == "record_start":
            c.request_teach("start")
        elif b == "record_end":
            c.request_teach("end")
        elif b == "go_start":
            c.request_transition(ArmMode.GO_START)
        elif b == "go_end":
            threading.Thread(target=_ik_and_go, args=(side,), daemon=True).start()
        elif b == "line_run":
            threading.Thread(target=_line_run, args=(side,), daemon=True).start()
        elif b == "arc_add":
            c.request_arc_add()
        elif b == "arc_clear":
            c.request_arc_clear()
        elif b == "arc_start":
            if c.arc_points:
                c.request_move_to(c.arc_points[0], "弧线回起点")
        elif b == "arc_run":
            threading.Thread(target=_arc_run, args=(side,), daemon=True).start()
        elif b == "replay_run":
            threading.Thread(target=_replay_run, args=(side,), daemon=True).start()
        elif b == "bspline_run":
            threading.Thread(target=_bspline_run, args=(side,), daemon=True).start()
        return f"{side}: {b}"

    return app


def arm_block(side: str):
    def btn(b):
        return html.Button(LABEL[b], id=f"btn-{side}-{b}", n_clicks=0,
                           style={"margin": "4px", "padding": "10px 12px", "fontSize": "13px",
                                  "border": "none", "borderRadius": "5px", "cursor": "pointer",
                                  "backgroundColor": STYLE[b], "color": "white"})
    return html.Div([
        html.H3(f"{side} 臂", style={"marginBottom": "4px"}),
        html.Div([
            html.Span(id=f"status-{side}", style={"fontWeight": "bold", "fontSize": "15px"}),
            html.Span("    "), html.Span(id=f"temp-{side}", style={"fontSize": "15px"}),
        ], style={"marginBottom": "4px"}),
        html.Span(id=f"taught-{side}", style={"fontSize": "13px", "color": "#555"}),
        html.Progress(id=f"progress-{side}", value=0, max=100,
                      style={"width": "100%", "height": "8px", "margin": "6px 0"}),
        dcc.Graph(id=f"fig-pos-{side}", config={"displayModeBar": False}),
        dcc.Graph(id=f"fig-err-{side}", config={"displayModeBar": False}),
        html.Div([btn(b) for b in MODE_BUTTONS],
                 style={"margin": "4px 0", "borderBottom": "1px dashed #ccc", "paddingBottom": "6px"}),
        html.Div([btn(b) for b in ("record_start", "record_end", "go_start", "go_end", "line_run")],
                 style={"margin": "4px 0", "borderBottom": "1px dashed #ccc", "paddingBottom": "6px"}),
        html.Div([html.Span("弧线/回放:", style={"fontSize": "12px", "color": "#777"}),
                  btn("arc_add"), btn("arc_clear"), btn("arc_start"),
                  btn("arc_run"), btn("replay_run"), btn("bspline_run")],
                 style={"margin": "4px 0"}),
    ], style={"flex": "1", "minWidth": "400px", "padding": "10px",
              "border": "1px solid #ddd", "borderRadius": "8px", "margin": "6px"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--no-can", action="store_true")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--can-slot", type=int, default=1,
                    help="CAN card slot number; left=ch1, right=ch0 (swapped 2026-08-27; default 1)")
    args = ap.parse_args()
    global CAN_MAP
    CAN_MAP = {"left": f"can_slot{args.can_slot}_ch1",   # swapped 2026-08-27
               "right": f"can_slot{args.can_slot}_ch0"}

    rs = RobotState(SIDES)
    buf = DataBuffer(SIDES, maxlen=200)
    controllers: dict[str, ArmController] = {}
    models: dict[str, PinocchioModel] = {}
    for side in SIDES:
        try:
            models[side] = PinocchioModel(resolve_urdf_path(), side)
        except Exception as e:  # noqa: BLE001
            rs.log(f"{side}: 模型加载失败: {e}")

    for side in SIDES:
        if args.sim:
            iface = f"sim-{side}"
            rs.set_can(iface, True)
        else:
            iface = CAN_MAP[side]
            if not args.no_can:
                up = bringup_can(iface)
                rs.set_can(iface, up)
                rs.log(f"{side} CAN {iface}: {'UP' if up else 'FAILED'}")
            else:
                rs.set_can(iface, can_is_up(iface))
        try:
            c = ArmController(iface, side, rs, buf, sim=args.sim)
            c.start()
            controllers[side] = c
            c.request_transition(ArmMode.ZERO_TORQUE)
        except Exception as e:  # noqa: BLE001
            rs.log(f"{side}: init failed: {type(e).__name__}: {e}")

    rs.log("dashboard ready — both arms in zero-torque (safe)")

    def _diag():
        while True:
            time.sleep(2)
            for s in SIDES:
                t, pos, _, _ = buf.series(s)
                st = rs.snapshot(s)
                tag = f"buf={len(t)}" if len(t) else "buf=EMPTY"
                ts = (f" start={'Y' if controllers[s].taught_start is not None else 'N'}"
                      f" end={'Y' if controllers[s].taught_end is not None else 'N'}")
                print(f"[diag] {s}: mode={st.mode} {tag}{ts} "
                      f"pos={np.round(pos[-1],2).tolist()}" if len(t)
                      else f"[diag] {s}: mode={st.mode} {tag}{ts}", flush=True)
    threading.Thread(target=_diag, daemon=True).start()

    app = build_app(rs, buf, controllers, models, args.sim)
    try:
        app.run(host=args.host, port=args.port, debug=False)
    finally:
        for c in controllers.values():
            c.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
