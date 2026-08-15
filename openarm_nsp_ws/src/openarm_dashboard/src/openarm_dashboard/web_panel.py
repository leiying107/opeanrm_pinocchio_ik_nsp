#!/usr/bin/env python3
# Copyright 2026 Enactic, Inc.
# SPDX-License-Identifier: Apache-2.0
"""OpenArm real-hardware web panel — FastAPI + SSE + canvas (smooth).

Replaces the Dash frontend of ``hardware_dashboard`` with the openarm_panel
pattern: a persistent **SSE** stream pushes a tiny JSON snapshot ~20×/s; the
browser renders with native ``<canvas>`` (no Plotly); commands are one-shot
REST POSTs. Same backend as the Dash version — every IK/planner/CAN function is
preserved verbatim (only the UI transport changed).

Run (real hw):  ros2 run openarm_dashboard web_panel
Sim (no hw):    ros2 run openarm_dashboard web_panel --sim
Browser:        http://<host>:8050
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import uvicorn
from scipy.spatial.transform import Rotation as _Rot
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from openarm_pinocchio_nsp.cartesian_planner import (
    Waypoint, check_traj_smoothness, ease_in_out_retime, fit_arc,
    pose_replay_traj, plan_cartesian,
)
from openarm_pinocchio_nsp.bspline_planner import BSplineOptimizer
from openarm_pinocchio_nsp import kinematics as _kin
from openarm_pinocchio_nsp.kinematics import PinocchioModel
from openarm_pinocchio_nsp.urdf_path import resolve_urdf_path

from .arm_controller import ARM_DOF, ArmController, ArmMode, bringup_can, can_is_up
from .robot_state import DataBuffer, RobotState

SIDES = ("left", "right")
CAN_MAP = {"left": "can_slot1_ch0", "right": "can_slot1_ch1"}  # ch0=left, ch1=right; --can-slot overrides

# live-tunable (mirrors hardware_dashboard)
ARC_TUNE = {"slowdown": 2.0, "cap": 1.0, "mode": "fast"}
ARC_MODES = {"fast": {"skip": True, "null_iters": 6, "label": "快速"},
             "thorough": {"skip": False, "null_iters": 12, "label": "精细"}}
REPLAY_TUNE = {"max_speed": 1.0, "freq": 100.0}
MODE_ACTIONS = {"enable": ArmMode.ZERO_TORQUE, "hold": ArmMode.ENABLED_HOLD,
                "home": ArmMode.HOMING, "disable": ArmMode.DISABLED}

STREAM_HZ = 20.0

# module-level backend handles (set in main / lifespan)
rs: RobotState | None = None
buf: DataBuffer | None = None
controllers: dict[str, ArmController] = {}
models: dict[str, PinocchioModel] = {}
err_buf: dict[str, deque] = {s: deque(maxlen=160) for s in SIDES}


# ============================================================ backend actions
def _warn_traj(side, times, q_path):
    chk = check_traj_smoothness(times, q_path)
    for msg in chk.warnings:
        rs.log(f"{side}: {msg}")
    return chk


def _ik_and_go(side):
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
        rs.log(f"{side}: IK未收敛 (err={r.pos_err_mm:.2f}mm)")
        return
    rs.log(f"{side}: IK解终点 σ_min={r.sigma_min:.3f}")
    c.request_move_to(r.q, "IK到终点")


def _line_run(side):
    c = controllers[side]
    qs, qe = c.taught_start, c.taught_end
    if qs is None or qe is None:
        rs.log(f"{side}: 请先记录起点和终点")
        return
    rs.log(f"{side}: 直线运动规划中...")
    t0 = time.time()
    try:
        m = models[side]
        result = plan_cartesian(m, [Waypoint(*m.fk(qs)), Waypoint(*m.fk(qe))],
                                q_init=qs, smooth=False)
    except Exception as e:  # noqa: BLE001
        rs.log(f"{side}: 直线运动异常 {type(e).__name__}: {e}")
        return
    t_plan = time.time() - t0
    if not result.success:
        rs.log(f"{side}: 直线IK失败 @sample {result.break_index} (规划{t_plan:.1f}s)")
        return
    times, q_path = result.times, result.q_path
    dur = times[-1] - times[0]
    peak = float(np.max(np.abs(np.diff(q_path, axis=0) / np.diff(times)[:, None])))
    _warn_traj(side, times, q_path)
    err_buf[side].clear()
    if c.request_transition(ArmMode.TRACKING, (times, q_path)):
        rs.log(f"{side}: ▶直线运动 {len(q_path)}点 规划{t_plan:.1f}s "
               f"时长{dur:.2f}s 峰速{peak:.2f}rad/s σ_min={result.min_sigma:.3f} "
               f"(不检查安全门)")


def _arc_run(side):
    c = controllers[side]
    pts = list(c.arc_points)
    if len(pts) < 2:
        rs.log(f"{side}: 弧线至少需要2个点(当前{len(pts)})")
        return
    cfg = ARC_MODES[ARC_TUNE.get("mode", "fast")]
    _kin._STAGE2_SKIP = cfg["skip"]
    rs.log(f"{side}: 弧线拟合+{cfg['label']}IK中... ({len(pts)}点)")
    t0 = time.time()
    try:
        m = models[side]
        wps = [Waypoint(*m.fk(q)) for q in pts]
        arc = fit_arc(wps, n_dense=100)
        result = plan_cartesian(m, arc, pts[0], presampled=True, null_iters=cfg["null_iters"])
    except Exception as e:  # noqa: BLE001
        rs.log(f"{side}: 弧线IK异常 {type(e).__name__}: {e}")
        return
    t_plan = time.time() - t0
    if not result.success:
        rs.log(f"{side}: 弧线IK失败 @sample {result.break_index} (规划{t_plan:.1f}s)")
        return
    times, q_path = ease_in_out_retime(result.times, result.q_path,
                                       slowdown=ARC_TUNE["slowdown"], vmax_cap=ARC_TUNE["cap"])
    dur = times[-1] - times[0]
    peak = np.max(np.abs(np.diff(q_path, axis=0) / np.diff(times)[:, None]))
    _warn_traj(side, times, q_path)
    err_buf[side].clear()
    if c.request_transition(ArmMode.TRACKING, (times, q_path)):
        rs.log(f"{side}: ▶弧线执行 {len(q_path)}点 [{cfg['label']}] 规划{t_plan:.1f}s "
               f"时长{dur:.2f}s 峰速{peak:.2f}rad/s σ_min={result.min_sigma:.3f}")


def _replay_run(side):
    c = controllers[side]
    pts = list(c.arc_points)
    if len(pts) < 2:
        rs.log(f"{side}: 关节回放至少需要2个点")
        return
    ms, fr = REPLAY_TUNE["max_speed"], REPLAY_TUNE["freq"]
    t0 = time.time()
    try:
        m = models[side]
        poses = [Waypoint(*m.fk(q)) for q in pts]
        q_seed = np.asarray(rs.snapshot(side).position, dtype=float)
        traj = pose_replay_traj(m, poses, q_seed, max_speed=ms, freq=fr,
                                null_iters=ARC_MODES[ARC_TUNE["mode"]]["null_iters"])
    except Exception as e:  # noqa: BLE001
        rs.log(f"{side}: 关节回放异常 {type(e).__name__}: {e}")
        return
    t_plan = time.time() - t0
    if traj is None:
        rs.log(f"{side}: 关节回放IK失败 (规划{t_plan:.1f}s)")
        return
    times, q_path = traj
    dur = times[-1] - times[0]
    peak = float(np.max(np.abs(np.diff(q_path, axis=0) / np.diff(times)[:, None])))
    _warn_traj(side, times, q_path)
    err_buf[side].clear()
    if c.request_transition(ArmMode.TRACKING, (times, q_path)):
        rs.log(f"{side}: ▶关节回放 {len(q_path)}点 时长{dur:.2f}s 峰速{peak:.2f}rad/s @{fr:.0f}Hz")


def _bspline_run(side):
    c = controllers[side]
    pts = list(c.arc_points)
    if len(pts) < 2:
        rs.log(f"{side}: 至少需要2个弧线点")
        return
    rs.log(f"{side}: B样条优化中... ({len(pts)}点)")
    try:
        opt = BSplineOptimizer(resolve_urdf_path(), side)
        wps = np.array([opt.fk_pos(q) for q in pts])
        spline, q_path, res = opt.optimize(wps, pts[0], duration=3.0, n_samples=30, maxiter=30, w3=0.0)
        v = opt.post_verify(q_path, duration=3.0, n_collision_check=50)
    except Exception as e:  # noqa: BLE001
        rs.log(f"{side}: B样条异常 {type(e).__name__}: {e}")
        return
    if not res.success:
        rs.log(f"{side}: B样条优化未收敛")
        return
    if not v["passed"]:
        rs.log(f"{side}: ✗后验证失败 碰撞={v['collisions']}")
        return
    n = len(q_path)
    times = np.linspace(0, 3.0, n).tolist()
    _warn_traj(side, times, list(q_path))
    err_buf[side].clear()
    if c.request_transition(ArmMode.TRACKING, (times, list(q_path))):
        rs.log(f"{side}: ▶B样条执行 {n}点 碰撞={v['collisions']}")


# background action dispatch (run IK in a thread so the request returns instantly)
def _spawn(fn, side):
    threading.Thread(target=fn, args=(side,), daemon=True).start()


# ============================================================ snapshot for SSE
def _snapshot() -> dict:
    out = {"arms": {}, "log": rs.recent_messages(14),
           "tune": {"slowdown": ARC_TUNE["slowdown"], "cap": ARC_TUNE["cap"],
                    "mode": ARC_TUNE["mode"], "maxspeed": REPLAY_TUNE["max_speed"],
                    "freq": REPLAY_TUNE["freq"]}}
    for s in SIDES:
        st = rs.snapshot(s)
        c = controllers.get(s)
        err = None
        m = models.get(s)
        tq = c.tracking_q() if c else None
        if tq is not None and m is not None:
            try:
                pp, pq = m.fk(tq[0]); ap, aq = m.fk(tq[1])
                d = (ap - pp) * 1000.0                       # actual−planned, mm, body frame (x前 y左 z上)
                rpy = (_Rot.from_quat(list(aq)) * _Rot.from_quat(list(pq)).inv()
                       ).as_euler("xyz", degrees=True)      # [roll,pitch,yaw]°, actual-vs-planned
                err = {"pos": [round(float(v), 2) for v in d],
                       "ori": [round(float(v), 3) for v in rpy],
                       "pmag": round(float(np.linalg.norm(d)), 2),
                       "omag": round(float(np.linalg.norm(rpy)), 3)}
            except Exception:  # noqa: BLE001
                pass
        out["arms"][s] = {
            "mode": st.mode, "enabled": st.enabled,
            "tmos_max": float(np.max(st.tmos)) if len(st.tmos) else 0.0,
            "taught": [c.taught_start is not None, c.taught_end is not None] if c else [False, False],
            "arc_n": len(c.arc_points) if c else 0,
            "progress": float(st.track_progress),
            "can_up": bool(rs.can_up.get(CAN_MAP[s], False)),
            "joints": [round(float(x), 4) for x in st.position],
            "err": err,
            "grav_on": c.gravity_on if c else False,
            "grav_scale": round(float(c.grav_scale), 2) if c else 0.0,
        }
    return out


# ============================================================ FastAPI
STATIC = Path(__file__).parent / "static"
app = FastAPI(title="OpenArm Web Panel")
if STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/", response_class=HTMLResponse)
def _index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/state")
def _state():
    return _snapshot()


@app.get("/sse/stream")
async def _sse(request: Request):
    async def gen():
        while True:
            if await request.is_disconnected():
                return
            yield f"data: {_json_dumps(_snapshot())}\n\n"
            await asyncio.sleep(1.0 / STREAM_HZ)
    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _ok(ok=True, **kw):
    return JSONResponse({"ok": bool(ok), **kw})


@app.post("/mode")
async def _mode(req: Request):
    b = await req.json()
    side, action = b.get("side"), b.get("action")
    if side not in SIDES or action not in MODE_ACTIONS:
        return _ok(False, err="bad args")
    controllers[side].request_transition(MODE_ACTIONS[action])
    return _ok()


@app.post("/teach")
async def _teach(req: Request):
    b = await req.json()
    side, which = b.get("side"), b.get("which")
    if side not in SIDES or which not in ("start", "end"):
        return _ok(False, err="bad args")
    controllers[side].request_teach(which)
    return _ok()


@app.post("/action")
async def _action(req: Request):
    """One-shot buttons: go_start | go_end | line_run | arc_add | arc_clear |
    arc_start | arc_run | replay_run | bspline_run."""
    b = await req.json()
    side, op = b.get("side"), b.get("op")
    if side not in SIDES or not op:
        return _ok(False, err="bad args")
    c = controllers[side]
    if op == "go_start":
        c.request_transition(ArmMode.GO_START)
    elif op == "go_end":
        _spawn(_ik_and_go, side)
    elif op == "line_run":
        _spawn(_line_run, side)
    elif op == "arc_add":
        c.request_arc_add()
    elif op == "arc_clear":
        c.request_arc_clear()
    elif op == "arc_start":
        if c.arc_points:
            c.request_move_to(c.arc_points[0], "弧线回起点")
    elif op == "arc_run":
        _spawn(_arc_run, side)
    elif op == "replay_run":
        _spawn(_replay_run, side)
    elif op == "bspline_run":
        _spawn(_bspline_run, side)
    else:
        return _ok(False, err=f"unknown op {op}")
    return _ok()


@app.post("/tune")
async def _tune(req: Request):
    b = await req.json()
    try:
        if isinstance(b.get("slowdown"), (int, float)) and b["slowdown"] > 0:
            ARC_TUNE["slowdown"] = float(b["slowdown"])
        if isinstance(b.get("cap"), (int, float)) and b["cap"] > 0:
            ARC_TUNE["cap"] = float(b["cap"])
        if b.get("mode") in ARC_MODES:
            ARC_TUNE["mode"] = b["mode"]
        if isinstance(b.get("maxspeed"), (int, float)) and b["maxspeed"] > 0:
            REPLAY_TUNE["max_speed"] = float(b["maxspeed"])
        if isinstance(b.get("freq"), (int, float)) and 1 <= b["freq"] <= 500:
            REPLAY_TUNE["freq"] = float(b["freq"])
    except Exception as e:  # noqa: BLE001
        return _ok(False, err=str(e))
    return _ok()


@app.post("/gravity")
async def _gravity(req: Request):
    """Toggle gravity comp + set scale (0..1.5). Effective only when enabled."""
    b = await req.json()
    side = b.get("side")
    if side not in SIDES:
        return _ok(False, err="bad side")
    controllers[side].set_gravity(bool(b.get("on", False)), b.get("scale", 0.0))
    return _ok()


def _json_dumps(obj):
    import json
    return json.dumps(obj, separators=(",", ":"))


# ============================================================ main
def _bringup(sim, can_slot, no_can):
    global rs, buf, CAN_MAP
    rs = RobotState(SIDES)
    buf = DataBuffer(SIDES, maxlen=400)
    CAN_MAP = {"left": f"can_slot{can_slot}_ch0", "right": f"can_slot{can_slot}_ch1"}
    for side in SIDES:
        try:
            models[side] = PinocchioModel(resolve_urdf_path(), side)
        except Exception as e:  # noqa: BLE001
            rs.log(f"{side}: 模型加载失败: {e}")
    for side in SIDES:
        if sim:
            iface = f"sim-{side}"
            rs.set_can(iface, True)
        else:
            iface = CAN_MAP[side]
            if not no_can:
                up = bringup_can(iface)
                rs.set_can(iface, up)
                rs.log(f"{side} CAN {iface}: {'UP' if up else 'FAILED'}")
            else:
                rs.set_can(iface, can_is_up(iface))
        try:
            c = ArmController(iface, side, rs, buf, sim=sim)
            c.start()
            controllers[side] = c
            c.request_transition(ArmMode.ZERO_TORQUE)
        except Exception as e:  # noqa: BLE001
            rs.log(f"{side}: init failed: {type(e).__name__}: {e}")
    rs.log("web panel ready — both arms zero-torque (safe)")


@app.on_event("shutdown")
def _shutdown():
    for c in controllers.values():
        with contextlib.suppress(Exception):
            c.stop()


def main() -> int:
    global CAN_MAP
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--no-can", action="store_true")
    ap.add_argument("--can-slot", type=int, default=1)
    args = ap.parse_args()
    _bringup(args.sim, args.can_slot, args.no_can)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    sys.exit(main())
