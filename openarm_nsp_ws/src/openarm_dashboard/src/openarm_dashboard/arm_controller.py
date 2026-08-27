# Copyright 2026 Enactic, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-arm hardware controller: state machine + 250 Hz worker thread.

Wraps ``openarm_can`` directly (no ros2_control). All CAN calls happen in the
worker thread — Dash buttons only push commands onto a queue.

State machine + teach workflow:
    DISABLED --enable-->  ZERO_TORQUE (safe default: free, draggable)
    ZERO_TORQUE --hold--> ENABLED_HOLD (PD holds current pose)
    [teach_start/teach_end] record current joints as start/end waypoint
    ZERO_TORQUE/HOLD --home--> HOMING --done--> HOLD (at zero)
    *_HOLD --go_start--> GO_START --done--> HOLD (at taught start pose)
    *_HOLD --track(traj)--> TRACKING --done--> HOLD (at end pose)
    any --disable--> DISABLED
"""

from __future__ import annotations

import bisect
import enum
import os
import queue
import subprocess
import threading
import time
from collections import deque

import numpy as np

from .robot_state import ARM_DOF, DataBuffer, RobotState

try:
    from openarm_can import (  # type: ignore
        MITParam, CallbackMode, MotorType, OpenArm,
    )
    _HAVE_CAN = True
except ImportError:
    _HAVE_CAN = False

ARM_TYPES = (
    [MotorType.DM8009, MotorType.DM8009, MotorType.DM4340,
     MotorType.DM4340, MotorType.DM4310, MotorType.DM4310, MotorType.DM4310]
    if _HAVE_CAN else []
)
ARM_SEND_IDS = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]
ARM_RECV_IDS = [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]
GRIPPER_SEND_ID, GRIPPER_RECV_ID = 0x08, 0x18
ARM_KP = [70.0, 70.0, 70.0, 60.0, 10.0, 10.0, 10.0]
ARM_KD = [2.75, 2.5, 2.0, 2.0, 0.7, 0.6, 0.5]
ZERO_POSITION = np.zeros(ARM_DOF)
MOVE_DUR = 2.0   # seconds for home / go_start interpolation

TMOS_WARN, TMOS_ERROR = 70, 85


class ArmMode(enum.Enum):
    DISABLED = "DISABLED"
    ZERO_TORQUE = "ZERO_TORQUE"
    ENABLED_HOLD = "ENABLED_HOLD"
    HOMING = "HOMING"
    GO_START = "GO_START"
    GO_END = "GO_END"
    TRACKING = "TRACKING"
    IMPEDANCE = "IMPEDANCE"   # Cartesian 6D spring-damper at the EE
    IMP_TRACK = "IMP_TRACK"   # impedance playback: anchor follows a recorded
                              # trajectory; external push pauses the clock


_BUSY = {ArmMode.HOMING, ArmMode.GO_START, ArmMode.GO_END, ArmMode.TRACKING,
         ArmMode.IMP_TRACK}


def bringup_can(iface: str) -> bool:
    subprocess.run(["ip", "link", "set", iface, "down"], stderr=subprocess.DEVNULL)
    r = subprocess.run(
        ["ip", "link", "set", iface, "up", "type", "can",
         "bitrate", "1000000", "dbitrate", "5000000", "fd", "on", "restart-ms", "1"],
        stderr=subprocess.DEVNULL,
    )
    return r.returncode == 0


def can_is_up(iface: str) -> bool:
    try:
        with open(f"/sys/class/net/{iface}/operstate") as f:
            return f.read().strip() in ("up", "unknown")
    except FileNotFoundError:
        return False


# 250 Hz control-loop CSV: one row per tick (real hw); ring buffer keeps the
# last 30 s in memory for event dumps. Header must match _fmt_row.
CTRL_HEADER = ("t,mode,enabled,grav_on,grav_scale,"
               + ",".join(f"q{i}" for i in range(ARM_DOF))
               + "," + ",".join(f"dq{i}" for i in range(ARM_DOF))
               + "," + ",".join(f"taus{i}" for i in range(ARM_DOF))
               + "," + ",".join(f"tauc{i}" for i in range(ARM_DOF))
               + "," + ",".join(f"tmos{i}" for i in range(ARM_DOF))
               + ",dx0,dx1,dx2,fest0,fest1,fest2,sigma,ramp,buckle,push")
RING_N = 7500      # rows @ 250 Hz = 30 s


class ArmController:
    def __init__(self, iface: str, side: str, robot_state: RobotState,
                 buffer: DataBuffer, sim: bool = False, with_gripper: bool = True,
                 record_dir: str | None = None):
        self.iface, self.side, self.sim = iface, side, sim
        self.rs, self.buf = robot_state, buffer
        self.with_gripper = with_gripper
        # control-loop recording (250 Hz CSV via a separate writer thread —
        # the 4 ms loop only does queue.put_nowait, never disk IO)
        self._rec_q: queue.Queue | None = None
        self._rec_thread: threading.Thread | None = None
        self._rec_drops = 0
        self._ring: deque = deque(maxlen=RING_N)
        self._last_cmd_tau = np.zeros(ARM_DOF)
        if record_dir:
            try:
                os.makedirs(record_dir, exist_ok=True)
                self._rec_path = os.path.join(record_dir, f"ctrl_{side}.csv")
                self._rec_q = queue.Queue(maxsize=40000)
                with open(self._rec_path, "w") as f:
                    f.write(CTRL_HEADER + "\n")
            except OSError as e:  # noqa: BLE001
                print(f"{side}: recording unavailable: {e}", flush=True)
                self._rec_q = None
        self.mode = ArmMode.DISABLED
        self.enabled = False
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._hold_q = np.zeros(ARM_DOF)
        # taught waypoints (set via teach commands in worker thread)
        self.taught_start: np.ndarray | None = None
        self.taught_end: np.ndarray | None = None
        self.arc_points: list[np.ndarray] = []   # taught arc control points (joint angles)
        # generic interpolated-move state (used by HOMING / GO_START)
        self._move_target = np.zeros(ARM_DOF)
        self._move_start = np.zeros(ARM_DOF)
        self._move_t0 = 0.0
        self._move_dur = MOVE_DUR
        # tracking trajectory
        self._traj: tuple[list[float], list[np.ndarray]] | None = None
        self._traj_t0 = 0.0
        # latest planned vs actual joint angles (for IK tracking-error display)
        self._last_planned_q: np.ndarray | None = None
        self._last_actual_q: np.ndarray | None = None
        self._last_actual_dq: np.ndarray = np.zeros(ARM_DOF)
        # sim state
        self._sim_pos = np.zeros(ARM_DOF)
        self._sim_vel = np.zeros(ARM_DOF)
        self._sim_tmos = np.full(ARM_DOF, 34.0)
        # gravity compensation (KDL-equivalent G(q) feedforward; off until enabled)
        self.gravity_on = False
        self.grav_scale = 0.0
        self._grav = None
        try:
            from .gravity import GravityComp
            self._grav = GravityComp()
        except Exception as e:  # noqa: BLE001
            print(f"{self.side}: gravity model unavailable: {type(e).__name__}: {e}", flush=True)
        # cartesian impedance (spring-damper at the EE; G(q) added separately
        # via _grav_tau — the impedance law itself contains NO gravity term)
        self._loop_dt = 0.05 if sim else 0.004
        self._imp = None
        self._imp_plant = None
        self._imp_kd = np.zeros(ARM_DOF)   # motor-side joint damping floor
        self._imp_tmax = np.array([54.0, 54.0, 28.0, 28.0, 10.0, 10.0, 10.0])
        self._imp_diag: dict | None = None
        # singularity-guard thresholds (module-level import so the command
        # handlers always see them even when model construction failed)
        from .impedance import SIG_WARN, SIG_CRIT
        self._sig_warn, self._sig_crit = SIG_WARN, SIG_CRIT
        self._imp_slow = 0          # consecutive slow impedance ticks (watchdog)
        self._imp_brake_until = 0.0  # monotonic deadline for post-exit damping
        # velocity / joint-limit trip-wires (exit-to-PD, NOT software limiting)
        self._imp_vtrip = 4.0       # rad/s -> count toward exit
        self._imp_vslow = 0         # windowed over-speed counter (decay, no reset)
        # inward axial-push detector state (buckle assist)
        self._last_actual_tau = np.zeros(ARM_DOF)
        self._imp_tau_res = np.zeros(ARM_DOF)   # LPF'd tau_sensed - tau_cmd
        self._imp_push_ct = 0
        # trajectory record / impedance playback (traj_rec.py)
        self._trec = None              # TrajRecorder while recording
        self._trec_t0 = 0.0
        self._trec_last = 0.0          # last 50Hz sample time
        self._traj = None              # loaded TrajData
        self._traj_name = ""
        self._kr_saved = None          # pre-replay Kr (restored on exit)
        self._mode_t0 = 0.0            # last enable time (guard settle window)
        self._tplay = 0.0              # replay clock (trajectory seconds)
        self._tpaused = False
        self._tresume_ct = 0
        self._trate = 0.5              # rate multiplier (UI)
        self._imp_lower = np.full(ARM_DOF, -3.05)
        self._imp_upper = np.full(ARM_DOF, 3.05)
        try:
            from .impedance import CartesianImpedance, IMP_KD, TMAX as IMP_TMAX
            self._imp_kd = IMP_KD
            from openarm_pinocchio_nsp.kinematics import PinocchioModel
            from openarm_pinocchio_nsp.urdf_path import resolve_urdf_path
            self._imp = CartesianImpedance(PinocchioModel(resolve_urdf_path(), side))
            self._imp_tmax = IMP_TMAX
            self._imp_lower = self._imp.m.lower.copy()   # URDF joint limits
            self._imp_upper = self._imp.m.upper.copy()
            if sim:
                from .impedance import ImpedanceSimPlant
                self._imp_plant = ImpedanceSimPlant(side)
        except Exception as e:  # noqa: BLE001
            print(f"{self.side}: impedance unavailable: {type(e).__name__}: {e}", flush=True)
        if not sim:
            if not _HAVE_CAN:
                raise RuntimeError("openarm_can not available; use sim=True")
            self.oa = OpenArm(iface, True)
            self.oa.init_arm_motors(ARM_TYPES, ARM_SEND_IDS, ARM_RECV_IDS)
            if with_gripper:
                self.oa.init_gripper_motor(MotorType.DM4310, GRIPPER_SEND_ID, GRIPPER_RECV_ID)

    # ------------------------------------------------- lifecycle
    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._work_loop, daemon=True, name=f"arm-{self.side}")
        self._thread.start()
        if self._rec_q is not None:
            self._rec_thread = threading.Thread(
                target=self._rec_writer, daemon=True, name=f"rec-{self.side}")
            self._rec_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.enabled:
            self._disable()
        self._flush_rec()

    # ------------------------------------------------- control-loop recording
    def _rec_writer(self) -> None:
        with open(self._rec_path, "a") as f:
            while True:
                row = self._rec_q.get()
                if row is None:          # sentinel: flush & exit
                    f.flush()
                    return
                f.write(row)

    def _fmt_row(self, t: float, q, dq, tau_s, tmos) -> str:
        st = self._imp_diag
        cols = [f"{t:.4f}", self.mode.value, str(int(self.enabled)),
                str(int(self.gravity_on)), f"{self.grav_scale:.2f}"]
        cols += [f"{v:.6g}" for v in q]
        cols += [f"{v:.6g}" for v in dq]
        cols += [f"{v:.4g}" for v in tau_s]
        cols += [f"{v:.4g}" for v in self._last_cmd_tau]
        cols += [f"{v:.1f}" for v in tmos]
        if st and st.get("dx") is not None:
            cols += [f"{v:.3g}" for v in st["dx"]]
            cols += [f"{v:.3g}" for v in st["fest"]]
            cols += [f"{st['sigma']:.4g}", f"{st['ramp']:.2f}",
                     f"{st.get('buckle', 0):.2f}",
                     str(int(self._imp.push_boost > 0.0)) if self._imp else "0"]
        else:
            cols += ["nan"] * 10
        return ",".join(cols) + "\n"

    def _record(self, q, dq, tau_s, tmos) -> None:
        if self._rec_q is None:
            return
        row = self._fmt_row(time.time(), q, dq, tau_s, tmos)
        self._ring.append(row)
        try:
            self._rec_q.put_nowait(row)
        except queue.Full:
            self._rec_drops += 1

    def _dump_event(self, why: str) -> None:
        """Write the last 5 s of the ring to an event CSV (debug a shake/exit)."""
        if self._rec_q is None or not self._ring:
            return
        name = os.path.join(os.path.dirname(self._rec_path),
                            f"event_{self.side}_{time.strftime('%H%M%S')}.csv")
        try:
            n = int(5.0 / max(0.05 if self.sim else 0.004, 1e-6))
            with open(name, "w") as f:
                f.write(f"# reason: {why}\n")
                f.write(CTRL_HEADER + "\n")
                f.writelines(list(self._ring)[-n:])
            self.rs.log(f"{self.side}: 事件日志已存 {os.path.basename(name)}")
        except OSError:  # noqa: BLE001
            pass

    def _flush_rec(self) -> None:
        if self._rec_q is None:
            return
        try:
            self._rec_q.put(None)
        except queue.Full:
            pass
        if self._rec_thread:
            self._rec_thread.join(timeout=2.0)
        if self._rec_drops:
            self.rs.log(f"{self.side}: 记录丢弃 {self._rec_drops} 行(磁盘慢)")
        if self._ring:
            tail = os.path.join(os.path.dirname(self._rec_path),
                                f"ring_tail_{self.side}.csv")
            try:
                with open(tail, "w") as f:
                    f.write(CTRL_HEADER + "\n")
                    f.writelines(self._ring)
            except OSError:  # noqa: BLE001
                pass

    def request_transition(self, mode: ArmMode, trajectory=None) -> bool:
        if self.mode in _BUSY and mode is not ArmMode.DISABLED:
            self.rs.log(f"{self.side}: busy ({self.mode.value}), reject {mode.value}")
            return False
        self._q.put(("mode", mode, trajectory))
        return True

    def request_teach(self, which: str) -> bool:
        """which = 'start' | 'end'. Records current joints as a waypoint."""
        self._q.put(("teach", which, None))
        return True

    def request_move_to(self, q_target, label: str = "moving") -> bool:
        """Smooth joint interpolation to an arbitrary target (e.g. an IK solution)."""
        if self.mode in _BUSY:
            self.rs.log(f"{self.side}: busy ({self.mode.value}), reject move_to")
            return False
        self._q.put(("move_to", np.asarray(q_target, dtype=float), label))
        return True

    def request_arc_add(self) -> bool:
        """Append current joints as an arc control point."""
        self._q.put(("arc_add", None, None))
        return True

    def request_arc_clear(self) -> bool:
        self._q.put(("arc_clear", None, None))
        return True

    def request_impedance(self, on: bool, params: dict | None = None) -> bool:
        """Toggle Cartesian impedance. params: preset/kx/zeta/leak (live-retunable)."""
        self._q.put(("impedance", on, dict(params or {})))
        return True

    def request_imp_push(self, force3, dur: float = 1.0) -> bool:
        """Virtual EE push force (N, world frame) — sim verification only."""
        self._q.put(("imp_push", force3, dur))
        return True

    def request_traj(self, op: str, payload: dict | None = None) -> bool:
        """Trajectory record/replay ops (see _do_traj)."""
        self._q.put(("traj", op, dict(payload or {})))
        return True

    # ------------------------------------------------- worker
    def _work_loop(self) -> None:
        rate = 0.05 if self.sim else 0.004
        while not self._stop.is_set():
            t0 = time.time()
            try:
                while True:
                    cmd = self._q.get_nowait()
                    try:
                        if cmd[0] == "mode":
                            self._process(cmd[1], cmd[2])
                        elif cmd[0] == "teach":
                            self._do_teach(cmd[1])
                        elif cmd[0] == "move_to":
                            self._do_move_to(cmd[1], cmd[2])
                        elif cmd[0] == "arc_add":
                            self._do_arc_add()
                        elif cmd[0] == "arc_clear":
                            self._do_arc_clear()
                        elif cmd[0] == "impedance":
                            self._do_impedance(cmd[1], cmd[2])
                        elif cmd[0] == "imp_push":
                            self._do_imp_push(cmd[1], cmd[2])
                        elif cmd[0] == "traj":
                            self._do_traj(cmd[1], cmd[2])
                    except Exception as e:  # noqa: BLE001
                        msg = f"{self.side}: cmd error {type(e).__name__}: {e}"
                        print(msg, flush=True)
                        self.rs.log(msg)
            except queue.Empty:
                pass
            try:
                self._step()
            except Exception as e:  # noqa: BLE001
                msg = f"{self.side}: loop error {type(e).__name__}: {e}"
                print(msg, flush=True)
                self.rs.log(msg)
                # a failed tick in IMPEDANCE means NO torque frame went out —
                # never keep the mode alive on a broken loop; drop to PD hold
                if self.mode in (ArmMode.IMPEDANCE, ArmMode.IMP_TRACK):
                    try:
                        self._exit_impedance(f"loop error {type(e).__name__} → 抱住")
                    except Exception:  # noqa: BLE001 - last resort
                        self._disable()
            dt = time.time() - t0
            if dt < rate:
                time.sleep(rate - dt)

    def _do_teach(self, which: str) -> None:
        q = self._read_pos()
        if which == "start":
            self.taught_start = q
            self.rs.log(f"{self.side}: 起点已记录 {np.round(q, 2).tolist()}")
        else:
            self.taught_end = q
            self.rs.log(f"{self.side}: 终点已记录 {np.round(q, 2).tolist()}")

    def _do_move_to(self, q_target: np.ndarray, label: str) -> None:
        if not self.enabled:
            self._enable()
        self._begin_move(q_target, ArmMode.GO_END, label)

    def _do_arc_add(self) -> None:
        q = self._read_pos()
        self.arc_points.append(q)
        self.rs.log(f"{self.side}: 弧线加点 #{len(self.arc_points)} {np.round(q, 2).tolist()}")

    def _do_arc_clear(self) -> None:
        self.arc_points.clear()
        self.rs.log(f"{self.side}: 弧线已清空")

    def _do_impedance(self, on: bool, params: dict) -> None:
        if self._imp is None:
            self.rs.log(f"{self.side}: 阻抗模型不可用")
            return
        if on:
            if not self.enabled:
                self.rs.log(f"{self.side}: 请先使能再开阻抗")
                return
            if self.mode in _BUSY:
                self.rs.log(f"{self.side}: busy ({self.mode.value}), 拒绝开阻抗")
                return
            if self.mode == ArmMode.IMPEDANCE:
                # already on: live retune only — do NOT re-anchor the spring
                self._imp.set_params(**params)
                self.rs.log(f"{self.side}: 阻抗调参 Kx={self._imp.kx:.0f}N/m "
                            f"ζ={self._imp.zeta:.2f} 漏速={self._imp.leak:.2f}/s")
                return
            # entry gate: refuse to anchor at/near a kinematic singularity —
            # there the J^T spring loses the singular direction entirely
            # (unbounded drift, silent elbow reversal, F_est under-reporting)
            q = self._read_pos()
            sig = float(np.linalg.svd(
                self._imp.m.jacobian6(q), compute_uv=False)[-1])
            if sig < self._sig_warn:
                self.rs.log(f"{self.side}: 拒绝开阻抗——σ_min={sig:.3f}<{self._sig_warn} "
                            f"(奇异/伸直位)。先屈肘(肘关节≈1.2rad)再开")
                return
            # (anchor capture happens in start() below)
            self._imp.set_params(**params)
            # impedance without gravity develops a steady-state sag (the spring
            # carries the arm's weight) — force gravity comp on
            if not self.gravity_on or self.grav_scale < 1.0:
                self.gravity_on = True
                self.grav_scale = max(self.grav_scale, 1.0)
                self.rs.log(f"{self.side}: 阻抗自动开启重力补偿 scale={self.grav_scale:.2f}")
            self._imp.start(q)          # anchor = current pose (q from gate above)
            if self._imp_plant is not None:
                self._imp_plant.reset(q)
            self.mode = ArmMode.IMPEDANCE
            self.rs.set_mode(self.side, "IMPEDANCE", True)
            self.rs.log(f"{self.side}: 阻抗ON [{self._imp.preset}] Kx={self._imp.kx:.0f}N/m "
                        f"Kr={self._imp.kr:.0f}Nm/rad ζ={self._imp.zeta:.2f} "
                        f"漏速={self._imp.leak:.2f}/s σ_min={sig:.3f} (纯力矩+重力)")
        elif self.mode == ArmMode.IMPEDANCE:
            self._exit_impedance("阻抗OFF → 抱住当前位姿")

    def _do_imp_push(self, force3, dur: float) -> None:
        if self._imp_plant is None:
            self.rs.log(f"{self.side}: 虚拟推力仅 --sim 模式可用")
            return
        if self.mode != ArmMode.IMPEDANCE:
            self.rs.log(f"{self.side}: 请先开启阻抗再用虚拟推力")
            return
        self._imp_plant.set_push(force3, dur, time.monotonic())
        self.rs.log(f"{self.side}: 虚拟推力 {np.round(np.asarray(force3, float), 1).tolist()}N "
                    f"× {dur:.1f}s")

    # ------------------------------------------------------ trajectory ops
    def _do_traj(self, op: str, p: dict) -> None:
        """record_start/record_stop/load/replay/stop/home (traj_rec.py)."""
        from . import traj_rec
        if op == "record_start":
            if self.mode != ArmMode.ZERO_TORQUE or not self.enabled:
                self.rs.log(f"{self.side}: 录制需先使能+零力矩（当前 {self.mode.value}）")
                return
            self._trec = traj_rec.TrajRecorder()
            self._trec.start(self._read_pos())
            self._trec_last = time.monotonic()
            self._trec_t0 = time.time()
            self.rs.log(f"{self.side}: 轨迹录制开始（拖动机械臂，最长120s）")
        elif op == "record_stop":
            if self._trec is None:
                self.rs.log(f"{self.side}: 未在录制")
                return
            try:
                self._traj = self._trec.stop()
                self._traj.side = self.side
                path = self._traj.save(p.get("name", ""), self.side)
                self._traj_name = path.name
                self.rs.log(f"{self.side}: 录制完成 {self._traj.times[-1]:.1f}s "
                            f"{len(self._traj.times)}点 → {path.name}")
            except ValueError as e:
                self.rs.log(f"{self.side}: 录制失败: {e}")
            finally:
                self._trec = None
        elif op == "load":
            try:
                self._traj = traj_rec.TrajData.load(
                    traj_rec.TRAJ_DIR / p["name"])
                if self._traj.side and self._traj.side != self.side:
                    self.rs.log(f"{self.side}: 轨迹属于{self._traj.side}臂，已拒载")
                    self._traj = None
                    return
                self._traj_name = p["name"]
                self.rs.log(f"{self.side}: 已加载 {p['name']} "
                            f"({self._traj.times[-1]:.1f}s)")
            except (KeyError, ValueError, OSError) as e:
                self.rs.log(f"{self.side}: 加载失败: {e}")
        elif op == "replay":
            self._begin_imp_track(p)
        elif op == "stop":
            if self.mode == ArmMode.IMP_TRACK:
                self._exit_impedance("轨迹回放停止 → 抱住")
            elif self._trec is not None:
                self._trec = None
                self.rs.log(f"{self.side}: 录制已取消")
        elif op == "home":
            if self._traj is None:
                self.rs.log(f"{self.side}: 先加载轨迹")
                return
            if not self.enabled:
                self.rs.log(f"{self.side}: 请先使能")
                return
            if self.mode in _BUSY:
                self.rs.log(f"{self.side}: busy ({self.mode.value})")
                return
            q0 = self._traj.q[0]
            dq = float(np.max(np.abs(q0 - self._read_pos())))
            dur = float(np.clip(dq / 0.3, 12.0, 60.0))   # ≤0.3 rad/s, floored
            self._begin_move(q0, ArmMode.GO_START, "traj home", duration=dur)
            self.rs.log(f"{self.side}: 匀速回轨迹起点 {dur:.0f}s (|Δq|max={np.degrees(dq):.0f}°)")

    def _begin_imp_track(self, p: dict) -> None:
        """Enter IMP_TRACK: anchor the impedance spring on the trajectory
        start and let _step_imp_track walk the anchor along it."""
        if self._imp is None:
            self.rs.log(f"{self.side}: 阻抗模型不可用")
            return
        if not self.enabled:
            self.rs.log(f"{self.side}: 请先使能")
            return
        if self.mode in _BUSY:
            self.rs.log(f"{self.side}: busy ({self.mode.value})")
            return
        if self._traj is None:
            self.rs.log(f"{self.side}: 先加载/录制轨迹")
            return
        q_now = self._read_pos()
        d0 = float(np.max(np.abs(self._traj.q[0] - q_now)))
        if d0 > 0.15:
            self.rs.log(f"{self.side}: 距轨迹起点 {np.degrees(d0):.0f}° (>8.6°)，"
                        f"先按[回起点]再回放")
            return
        sig = float(np.linalg.svd(
            self._imp.m.jacobian6(q_now), compute_uv=False)[-1])
        if sig < self._sig_warn:
            self.rs.log(f"{self.side}: 拒绝回放——σ_min={sig:.3f}<{self._sig_warn} 奇异位")
            return
        self._trate = float(np.clip(p.get("rate", self._trate), 0.2, 1.0))
        params = {k: p[k] for k in ("preset", "kx", "zeta") if k in p}
        self._imp.set_params(**params)
        # JOINT-SPACE replay: silence the EE ORIENTATION spring for the whole
        # replay (wrist/config tracking is carried by the joint springs with
        # q_post = q_ref). Kr is restored on exit (_exit_impedance).
        self._kr_saved = self._imp.kr
        self._imp.kr = 0.0
        if not self.gravity_on or self.grav_scale < 1.0:
            self.gravity_on = True
            self.grav_scale = max(self.grav_scale, 1.0)
            self.rs.log(f"{self.side}: 回放自动开启重力补偿 scale={self.grav_scale:.2f}")
        self._imp.start(self._traj.q[0])
        if self._imp_plant is not None:
            self._imp_plant.reset(q_now)
        self._tplay = 0.0
        self._tpaused = False
        self._tresume_ct = 0
        self.mode = ArmMode.IMP_TRACK
        self.rs.set_mode(self.side, "IMP_TRACK", True)
        dur = self._traj.play_duration(self._trate)
        self.rs.log(f"{self.side}: 轨迹回放开始 [{self._imp.preset}] "
                    f"Kx={self._imp.kx:.0f} ζ={self._imp.zeta:.2f} "
                    f"{self._trate:.1f}x 预计{dur:.0f}s")

    def _step_imp_track(self) -> None:
        """One IMP_TRACK tick: advance the replay clock (unless paused by an
        external push), move the impedance anchor to q_ref(t), then run the
        stock impedance tick (all guards/trip-wires live).

        JOINT-SPACE replay (user decision 2026-08-26): the recorded joint
        angles ARE the reference. The EE POSITION spring (x_des from FK of
        q_ref) provides push-compliance at the hand; the wrist/config
        tracking is carried by the JOINT-space springs (posture q_post +
        config-return, both = q_ref) instead of the EE ORIENTATION spring —
        during a shoulder sweep the FK orientation rotates at the shoulder
        rate even with wrist joints held, which fed the ZOH-capped wrist
        orientation loop into an 8.5 Hz blow-up (hw 2026-08-26). Kr is set
        to 0 for the whole replay (restored on exit)."""
        from .traj_rec import ease_rate
        if self._imp is None or self._traj is None:
            self._exit_impedance("轨迹数据丢失 → 抱住")
            return
        dur = self._traj.play_duration(self._trate)
        # this method runs ONCE per loop tick, so the clock advances by the
        # FULL tick each call. (A previous "substep clock" made the clock 12x
        # too slow in sim — R2 timeouts with progress ~0.99.) The anchor
        # simply jumps by rate*loop_dt per tick, which the spring filters.
        dt_clk = self._loop_dt
        # deviation measured on the RAW (unweighted) anchor error so the
        # singular-direction fade cannot hide a real displacement
        raw_dx = np.linalg.norm(self._imp.x_des - np.asarray(
            self._imp.m.fk(self._last_actual_q)[0], float))
        if raw_dx > 0.05:
            if not self._tpaused:
                self._tpaused = True
                self.rs.log(f"{self.side}: 外力偏离 {raw_dx*1000:.0f}mm → 轨迹暂停")
            self._tresume_ct = 0
        elif self._tpaused:
            # resume just below the pause threshold (a 2–5 cm dead zone would
            # trap the clock paused forever — sim probe finding); hold the
            # requirement for 0.3s so a bouncing boundary doesn't chatter
            self._tresume_ct += 1
            if raw_dx < 0.045 and self._tresume_ct >= 15:
                self._tpaused = False
                self.rs.log(f"{self.side}: 已回位 → 轨迹继续")
        else:
            self._tplay += ease_rate(self._tplay, dur, self._trate) * dt_clk
        if self._tplay >= self._traj.times[-1]:
            # hold the anchor at the END pose until the arm catches up (the
            # spring lags a moving anchor by design), then exit to PD
            q_ref = self._traj.q[-1]
            x, quat = self._imp.m.fk(q_ref)
            self._imp.x_des = np.asarray(x, float).copy()
            self._imp.q_post = q_ref
            lag = float(np.max(np.abs(q_ref - self._last_actual_q)))
            if lag < 0.10 or self._tplay > self._traj.times[-1] + 5.0:
                self.rs.update_arm(self.side, track_progress=1.0)
                self._exit_impedance("轨迹回放完成 → 抱住终点")
            return
        q_ref = self._traj.sample_at(self._tplay)
        # move the POSITION anchor along the trajectory (joint-space replay:
        # wrist orientation is carried by q_post, NOT by R_des — see docstring)
        x, quat = self._imp.m.fk(q_ref)
        self._imp.x_des = np.asarray(x, float).copy()
        self._imp.q_post = q_ref
        self.rs.update_arm(self.side, track_progress=self._tplay / self._traj.times[-1])
        self._step_impedance()

    def _exit_impedance(self, why: str) -> None:
        """Leave impedance the safe way: capture pose, revert to motor-side PD hold.

        If the arm is moving FAST at exit (e.g. the watchdog froze damping for
        tens of ms while the user pushed), a stiff PD hold at the CURRENT pose
        + high velocity = a hard brake + overshoot "rebound". Ramp the hold
        target over ~0.5 s from the frozen pose to a soft landing pose instead
        of stepping kp on instantly — RECOVER_MOVE style damping-first stop."""
        # restore the EE orientation spring if a joint-space replay zeroed it
        if getattr(self, "_kr_saved", None) is not None and self._imp is not None:
            self._imp.kr = self._kr_saved
            self._kr_saved = None
        self._hold_q = self._read_pos()
        vmax = float(np.max(np.abs(self._last_actual_dq))) if self._last_actual_dq is not None else 0.0
        if vmax > 2.0 and not self.sim:
            # moving fast: brake phase — hold target = frozen pose, but walk
            # gain in via a short GO-style ramp by pretending we just moved
            # here (hold PD with gravity stays on; kd is small so also emit
            # pure-torque damping for the first 0.5 s via _imp_brake)
            self._imp_brake_until = time.monotonic() + 0.5
            self.rs.log(f"{self.side}: 退出时高速({vmax:.1f}rad/s) → 0.5s阻尼刹车")
        self.mode = ArmMode.ENABLED_HOLD
        self._imp_diag = None
        self.rs.set_mode(self.side, "ENABLED_HOLD", True)
        self.rs.log(f"{self.side}: {why}")
        self._dump_event(why)     # 5 s pre-exit control data for post-mortem

    def _process(self, mode: ArmMode, trajectory) -> None:
        # any transition away from IMPEDANCE/IMP_TRACK is an implicit
        # impedance-exit (motion requests auto-fall-back to motor-side PD)
        if mode not in (ArmMode.IMPEDANCE, ArmMode.IMP_TRACK):
            self._imp_diag = None
        if mode == ArmMode.DISABLED:
            self._disable()
            self.mode = ArmMode.DISABLED
            self.rs.set_mode(self.side, "DISABLED", False)
            self.rs.log(f"{self.side}: disabled")
            return
        if not self.enabled:
            self._enable()
        if mode == ArmMode.ZERO_TORQUE:
            self.mode = ArmMode.ZERO_TORQUE
            self.rs.set_mode(self.side, "ZERO_TORQUE", True)
            self.rs.log(f"{self.side}: zero-torque (free, draggable)")
        elif mode == ArmMode.ENABLED_HOLD:
            self._hold_q = self._read_pos()
            self.mode = ArmMode.ENABLED_HOLD
            self.rs.set_mode(self.side, "ENABLED_HOLD", True)
            self.rs.log(f"{self.side}: holding pose")
        elif mode == ArmMode.HOMING:
            self._begin_move(ZERO_POSITION, ArmMode.HOMING, "homing to zero (will move!)")
        elif mode == ArmMode.GO_START:
            if self.taught_start is None:
                self.rs.log(f"{self.side}: 未记录起点，无法回起点")
                return
            self._begin_move(self.taught_start, ArmMode.GO_START, "returning to taught start")
        elif mode == ArmMode.GO_END:
            if self.taught_end is None:
                self.rs.log(f"{self.side}: 未记录终点，无法到终点")
                return
            self._begin_move(self.taught_end, ArmMode.GO_END, "moving to taught end")
        elif mode == ArmMode.TRACKING:
            if trajectory is None:
                self.rs.log(f"{self.side}: TRACKING needs a trajectory")
                return
            self._traj = trajectory
            self._traj_t0 = time.time()
            self.mode = ArmMode.TRACKING
            self.rs.set_mode(self.side, "TRACKING", True)
            self.rs.log(f"{self.side}: tracking trajectory")

    def _begin_move(self, target: np.ndarray, mode: ArmMode, msg: str,
                    duration: float | None = None) -> None:
        self._move_target = np.asarray(target, dtype=float).copy()
        self._move_start = self._read_pos()
        self._move_t0 = time.time()
        self._move_dur = duration if duration else MOVE_DUR
        self.mode = mode
        self.rs.set_mode(self.side, mode.value, True)
        self.rs.log(f"{self.side}: {msg} (will move!)")

    def _enable(self) -> None:
        if self.sim:
            self.enabled = True
            return
        self.oa.set_callback_mode_all(CallbackMode.STATE)
        self.oa.enable_all()
        time.sleep(0.1)
        self.oa.recv_all()
        zero = [MITParam(0, 0, 0, 0, 0)] * ARM_DOF
        for _ in range(30):
            self.oa.get_arm().mit_control_all(zero)
            self.oa.recv_all()
        self.enabled = True
        # telemetry settle window for the encoder-jump guard (see _step)
        self._mode_t0 = time.monotonic()

    def _disable(self) -> None:
        self.gravity_on = False
        if self.sim:
            self.enabled = False
            return
        for _ in range(3):
            self.oa.disable_all()
            time.sleep(0.1)
            self.oa.recv_all()
        self.enabled = False

    def _read_pos(self) -> np.ndarray:
        if self.sim:
            return self._sim_pos.copy()
        motors = self.oa.get_arm().get_motors()
        return np.array([m.get_position() for m in motors[:ARM_DOF]])

    # ------------------------------------------------- gravity
    def _grav_tau(self, q=None) -> np.ndarray:
        """Gravity feedforward (7-vec) added to the MIT tau. Zeros if off/disabled."""
        if (not self.enabled or not self.gravity_on or self.grav_scale <= 0
                or self._grav is None):
            return np.zeros(ARM_DOF)
        try:
            return self._grav.torque(self.side, self._read_pos() if q is None else q,
                                     self.grav_scale)
        except Exception:  # noqa: BLE001
            return np.zeros(ARM_DOF)

    def set_gravity(self, on: bool, scale: float) -> None:
        """Toggle gravity comp + set scale (0..1.5). Effective only when enabled."""
        self.gravity_on = bool(on) and self.enabled
        try:
            self.grav_scale = float(max(0.0, min(1.5, scale)))
        except (TypeError, ValueError):
            pass
        self.rs.log(f"{self.side}: 重力补偿 {'ON' if self.gravity_on else 'OFF'} "
                    f"scale={self.grav_scale:.2f}")

    # ------------------------------------------------- step
    def _step(self) -> None:
        pos, vel, tau, tmos, trotor = self._sense()
        # ENCODER RE-INDEX GUARD: a failing CAN bus (hw 2026-08-27 06:42 —
        # right ch1 error storm, bus went silent) made all 7 motors re-acquire
        # their position estimates with new multi-turn offsets: EVERY joint
        # "jumped" 0.3–3.1 rad within one 4 ms tick while velocity feedback
        # read ~0 — physically impossible motion. With gravity comp on, the
        # model G flipped sign at the new (wrong) q and actively DROVE the
        # shoulder a full revolution into the body. Guard: any per-tick joint
        # jump that no joint can physically traverse (>25 rad/s effective)
        # while the sensed speed is far below it = position estimate corrupt
        # → kill gravity feedforward immediately (pure torque would be wrong
        # too — disable the arm) and demand re-enable.
        if self._last_actual_q is not None and not self.sim:
            # real hw only: the sim test harness teleports _sim_pos between
            # scripted poses, which would false-trip the guard. SKIP the first
            # 0.5 s after any transition into an enabled mode: the position
            # baseline is stale there (motors report zeros until the first
            # telemetry frame lands — hw 2026-08-27 07:34: every enable
            # false-tripped on the resting j7 offset vs the zero baseline and
            # disabled the arm instantly, "cannot enable")
            settled = (time.monotonic() - self._mode_t0) > 0.5
            if settled and self.enabled:
                step = np.abs(pos - self._last_actual_q)
                bad = step > 0.1   # >25 rad/s at 250Hz — no joint can do this
                quiet = np.max(np.abs(vel)) < 8.0  # sensors say NOT flying
                if np.any(bad) and quiet:
                    j = int(np.argmax(step)) + 1
                    self.rs.log(
                        f"{self.side}: ⛔ 编码器跳变 j{j} +{step.max():.2f}rad/周期"
                        f"(速度反馈仅{np.max(np.abs(vel)):.1f}rad/s) — 位置估计损坏，"
                        f"急停。请检查电机/CAN后重新上电使能")
                    self._dump_event("encoder-jump")
                    self._disable()
                    self.mode = ArmMode.DISABLED
                    self.rs.set_mode(self.side, "DISABLED", False)
                    return
                self.rs.set_mode(self.side, "DISABLED", False)
                return
        self._last_actual_q = pos
        self._last_actual_dq = vel
        self._last_actual_tau = tau
        self.rs.update_arm(self.side, position=pos, velocity=vel, torque=tau,
                           tmos=tmos, trotor=trotor)
        self.buf.append(self.side, pos, tau, tmos)
        if np.max(tmos) > TMOS_ERROR and self.mode != ArmMode.DISABLED:
            self.rs.log(f"{self.side}: OVERHEAT {float(np.max(tmos)):.0f}°C → disable")
            self._dump_event("overheat")
            self._disable()
            self.mode = ArmMode.DISABLED
            self.rs.set_mode(self.side, "DISABLED", False)
            return
        self._actuate()
        # trajectory recording: sample at 50 Hz WALL time (the sim loop runs
        # at 20 Hz — a tick-count gate would sample 4x too slowly there)
        if self._trec is not None:
            now = time.monotonic()
            if now - self._trec_last >= 0.02:
                self._trec_last = now
                self._trec.sample(pos)
        self._record(pos, vel, tau, tmos)

    def _sense(self):
        if self.sim:
            return self._sim_sense()
        self.oa.recv_all()
        motors = self.oa.get_arm().get_motors()
        pos = np.array([m.get_position() for m in motors[:ARM_DOF]])
        vel = np.array([m.get_velocity() for m in motors[:ARM_DOF]])
        tau = np.array([m.get_torque() for m in motors[:ARM_DOF]])
        tmos = np.array([m.get_state_tmos() for m in motors[:ARM_DOF]], dtype=float)
        trotor = np.array([m.get_state_trotor() for m in motors[:ARM_DOF]], dtype=float)
        return pos, vel, tau, tmos, trotor

    def _actuate(self) -> None:
        if not self.enabled:
            self._last_cmd_tau[:] = 0.0
            return
        if self.mode == ArmMode.ZERO_TORQUE:
            if not self.sim:
                g = self._grav_tau()
                self._last_cmd_tau = g
                self.oa.get_arm().mit_control_all(
                    [MITParam(0, 0, 0, 0, g[i]) for i in range(ARM_DOF)])
            else:
                self._last_cmd_tau[:] = 0.0
        elif self.mode == ArmMode.ENABLED_HOLD:
            if self.sim:
                self._sim_pos += 0.1 * (self._hold_q - self._sim_pos)
                self._last_cmd_tau[:] = 0.0
            elif time.monotonic() < self._imp_brake_until:
                # post-impedance-exit brake window: arm left impedance at high
                # speed; add pure-torque joint damping on top of the (weak-kd)
                # hold so the stop is gradual instead of a PD hard brake that
                # overshoots and "rebounds"
                g = self._grav_tau()
                brake = -np.array([4.0, 4.0, 3.0, 3.0, 2.0, 2.0, 2.0]) \
                    * self._last_actual_dq
                lim = np.array([0.3, 0.3, 0.3, 0.3, 0.15, 0.15, 0.15]) \
                    * np.array([0.23, 0.24, 0.06, 0.08, 0.003, 0.004, 0.004]) \
                    / self._loop_dt
                tau = np.clip(self._hold_pd_tau() + np.clip(brake, -lim, lim) + g,
                              -self._imp_tmax, self._imp_tmax)
                self._last_cmd_tau = tau
                self.oa.get_arm().mit_control_all(
                    [MITParam(0, 0, 0, 0, float(tau[i])) for i in range(ARM_DOF)])
            else:
                self._mit_hold(self._hold_q)
        elif self.mode in (ArmMode.HOMING, ArmMode.GO_START, ArmMode.GO_END):
            self._step_move()
        elif self.mode == ArmMode.TRACKING:
            self._step_track()
        elif self.mode == ArmMode.IMPEDANCE:
            self._step_impedance()
        elif self.mode == ArmMode.IMP_TRACK:
            self._step_imp_track()

    def _step_impedance(self) -> None:
        """Cartesian impedance tick: tau = J^T(K.d - D.xdot) + null + G(q).

        Sent as PURE torque (MITParam kp=kd=0) — the same real-hw-validated
        path as zero-torque+gravity. Watchdog: slow/failed compute falls back
        to motor-side PD hold. In sim, the control law is sub-stepped at 250 Hz
        inside the 20 Hz loop (a 800 N/m spring sampled at 20 Hz is unstable;
        real hw runs the loop at 250 Hz natively).
        """
        if self._imp is None:
            self._exit_impedance("阻抗模型丢失 → 抱住")
            return
        if self.sim:
            if self._imp_plant is None:
                self._sim_vel[:] = 0.0   # no plant: freeze in place
                return
            n = max(1, int(round(self._loop_dt / 0.004)))   # ~12 sub-ticks @ 250 Hz
            h = self._loop_dt / n
            t_now = time.monotonic()
            diag = None
            for _ in range(n):
                q = self._imp_plant.q + np.random.normal(0, 0.005, ARM_DOF)
                dq = self._imp_plant.v + np.random.normal(0, 0.01, ARM_DOF)
                try:
                    tau_imp, diag = self._imp.torque(q, dq, h)
                except Exception as e:  # noqa: BLE001
                    self.rs.log(f"{self.side}: 阻抗计算异常 {type(e).__name__}: {e} → 抱住")
                    self._exit_impedance("阻抗异常 → 抱住")
                    return
                tau = np.clip(tau_imp + self._grav_tau(q), -self._imp_tmax,
                              self._imp_tmax)
                # (no separate kd emulation: commanded damping already includes
                #  the capped KD_J floor; a ZOH-emulated motor kd diverged in
                #  sim on the ~0.003 kg m^2 wrist inertia)
                self._imp_plant.step(tau, t_now, h, substeps=2)
                self._last_cmd_tau = tau
            self._sim_pos = self._imp_plant.q.copy()
            self._sim_vel = self._imp_plant.v.copy()
            self._imp_diag = diag
            if diag is not None and diag["sigma"] < self._sig_crit:
                self._exit_impedance(
                    f"σ_min={diag['sigma']:.3f}<{self._sig_crit} 奇异硬保护 → 抱住")
            return
        # ---- real hardware: one 250 Hz tick ----
        # watchdog: a SINGLE >8ms tick trips too easily — OpenBLAS thread-pool
        # sync + Python GC occasionally stall one tick by 5–15ms on the 8-core
        # RK3588 under load (measured: p99 tick 10ms, ~2.7% of ticks, module
        # otherwise healthy). One missed 4ms tick is harmless (pure torque, no
        # integration); THREE consecutive slow ticks = real stall -> exit.
        t0 = time.perf_counter()
        try:
            tau_imp, diag = self._imp.torque(
                self._last_actual_q, self._last_actual_dq, self._loop_dt)
        except Exception as e:  # noqa: BLE001
            self.rs.log(f"{self.side}: 阻抗计算异常 {type(e).__name__}: {e} → 抱住")
            self._exit_impedance("阻抗异常 → 抱住")
            return
        dt_tick = time.perf_counter() - t0
        if dt_tick > 0.008:
            self._imp_slow += 1
        else:
            self._imp_slow = 0
        if self._imp_slow >= 3:
            self._exit_impedance(f"阻抗连续3次超时({dt_tick*1000:.0f}ms) → 抱住")
            return
        self._imp_diag = diag
        if diag["sigma"] < self._sig_crit:
            # PERSISTENCE gate: brief sigma dips happen when a push carries the
            # arm THROUGH an interior low-sigma pocket (sim: 0.7s crossing with
            # clean recovery after) — exiting on the first sub-critical tick
            # aborted recoverable transits. Require a sustained violation.
            self._imp_siglow = getattr(self, "_imp_siglow", 0) + 1
            if self._imp_siglow >= 125:    # 0.5 s sustained @250Hz
                self._exit_impedance(
                    f"σ_min={diag['sigma']:.3f}<{self._sig_crit} 奇异硬保护(持续0.5s) → 抱住")
                return
        else:
            self._imp_siglow = 0
        # --- velocity / joint-limit trip-wires (real hw only) -----------------
        # NOT a software speed limiter: commanded damping is ZOH-capped at
        # 0.2*I/dt (0.15 Nm at the wrist) and CANNOT stop a fast arm, and
        # motor-side kd caused the first real-hw shake incident. Instead,
        # sustained over-speed or approach to a URDF joint limit exits to the
        # motor-side PD hold (kp=70 — a far harder brake than any software
        # damping, on the already-validated path).
        if not self.sim:
            vmax = float(np.max(np.abs(self._last_actual_dq)))
            if vmax > self._imp_vtrip:
                self._imp_vslow += 1
            else:
                self._imp_vslow = max(0, self._imp_vslow - 1)   # decay, not reset
            # CONSECUTIVE-tick counting was defeated by a 15 Hz limit cycle
            # (each half-period dropped below threshold and reset the count —
            # measured: 8 bursts of 44-104ms in 1.5s, none reaching 200ms).
            # Windowed count with decay trips on oscillation too: >75 of the
            # last 250 ticks (0.3 s window, 30% duty) above threshold = exit.
            if self._imp_vslow >= 75:
                self._exit_impedance(
                    f"关节超速{vmax:.1f}rad/s(窗口累计0.3s) → 抱住")
                return
            q_now = self._last_actual_q
            margin = float(np.min(np.minimum(
                q_now - self._imp_lower, self._imp_upper - q_now)))
            if margin < 0.10:   # 5.7° from a hard stop
                self._exit_impedance(
                    f"逼近关节限位({np.degrees(margin):.1f}°) → 抱住")
                return
            # --- inward axial-push detector (buckle assist) --------------------
            # tau_sensed - tau_cmd ≈ J^T F_ext: an inward push along the
            # (nearly lost) radial direction projects onto v_dir as sigma*|F|
            # (verified numerically: 20N @ sigma=0.042 -> 0.849 Nm = theory).
            # Residual is LPF'd (10 Hz); decay counter over a 0.15 Nm (3-sigma)
            # threshold; noise floor measured at 1sigma≈0.05 Nm in the field.
            if diag.get("vdir") is not None and diag["sigma"] < 0.12:
                raw = self._last_actual_tau - self._last_cmd_tau
                a = self._loop_dt / (self._loop_dt + 1.0 / (2.0 * np.pi * 10.0))
                self._imp_tau_res += a * (raw - self._imp_tau_res)
                p = float(self._imp_tau_res @ diag["vdir"])
                if p > 0.15:
                    self._imp_push_ct += 1
                else:
                    self._imp_push_ct = max(0, self._imp_push_ct - 1)
                if self._imp_push_ct >= 25:   # 0.1 s sustained
                    self._imp_push_ct = 0
                    self._imp.push_boost = 0.5
                    self.rs.log(f"{self.side}: 检测到轴向内推 → 屈肘协助0.5s")
        tau = np.clip(tau_imp + self._grav_tau(), -self._imp_tmax, self._imp_tmax)
        self._last_cmd_tau = tau
        # kp=0 + kd=IMP_KD(=0 on real hw): pure torque, the validated path
        self.oa.get_arm().mit_control_all(
            [MITParam(0, float(self._imp_kd[i]), 0, 0, float(tau[i]))
             for i in range(ARM_DOF)])

    def imp_state(self) -> dict | None:
        """Impedance status for the web panel snapshot (None = unavailable)."""
        if self._imp is None:
            return None
        d = self._imp_diag
        return {
            "on": self.mode == ArmMode.IMPEDANCE,
            "preset": self._imp.preset,
            "kx": round(float(self._imp.kx), 1),
            "kr": round(float(self._imp.kr), 1),
            "zeta": round(float(self._imp.zeta), 2),
            "leak": round(float(self._imp.leak), 2),
            "dx": [round(float(v), 1) for v in d["dx"]] if d else None,
            "dth": [round(float(v), 2) for v in d["dth"]] if d else None,
            "fest": [round(float(v), 1) for v in d["fest"]] if d else None,
            "qdot": round(float(d["qdot"]), 2) if d else None,
            "ramp": d.get("ramp") if d else None,
            "sigma": d.get("sigma") if d else None,
            "buckle": d.get("buckle") if d else None,   # flexion-pull blend 0-1
            "frozen": d.get("frozen") if d else None,   # singular dir under-actuated
            "push": bool(getattr(self._imp, "push_boost", 0.0) > 0.0),
        }

    def traj_state(self) -> dict:
        """Trajectory record/playback status for the web panel snapshot.

        ``self._traj`` is DUAL-PURPOSE: a ``TrajData`` (recorder playback,
        has ``.times``) or a ``(times, q_path)`` tuple (TRACKING from the
        dashboard planners). Snapshot fields only make sense for the former.
        """
        is_td = hasattr(self._traj, "times")
        return {
            "rec": self._trec is not None,
            "rec_s": round(time.time() - self._trec_t0, 1) if self._trec else 0.0,
            "loaded": self._traj_name,
            "dur": round(float(self._traj.times[-1]), 1) if is_td else 0.0,
            "playing": self.mode == ArmMode.IMP_TRACK,
            "paused": self._tpaused,
            "progress": (round(self._tplay / self._traj.times[-1], 3)
                         if is_td else 0.0),
            "rate": round(self._trate, 2),
        }

    def _hold_pd_tau(self) -> np.ndarray:
        """The PD part of _mit_hold (no gravity) — reused by the brake window."""
        kp = np.asarray(ARM_KP, float)
        kd = np.asarray(ARM_KD, float)
        return kp * (self._hold_q - self._last_actual_q) - kd * self._last_actual_dq

    def _mit_hold(self, target: np.ndarray) -> None:
        g = self._grav_tau()
        params = [MITParam(ARM_KP[i], ARM_KD[i], target[i], 0, g[i]) for i in range(ARM_DOF)]
        self.oa.get_arm().mit_control_all(params)
        self._last_cmd_tau = (np.asarray(ARM_KP, float) * (np.asarray(target, float) - self._last_actual_q)
                              - np.asarray(ARM_KD, float) * self._last_actual_dq + g)

    def _step_move(self) -> None:
        """Smooth linear interpolation _move_start -> _move_target over _move_dur."""
        t = time.time() - self._move_t0
        if t >= self._move_dur:
            self._hold_q = self._move_target.copy()
            self.mode = ArmMode.ENABLED_HOLD
            self.rs.set_mode(self.side, "ENABLED_HOLD", True)
            self.rs.update_arm(self.side, track_progress=1.0)
            self.rs.log(f"{self.side}: reached target, holding")
            return
        frac = t / self._move_dur
        target = self._move_start + frac * (self._move_target - self._move_start)
        self._last_planned_q = target
        self.rs.update_arm(self.side, track_progress=frac)
        if self.sim:
            self._sim_pos = target.copy()
        else:
            self._mit_hold(target)

    def _step_track(self) -> None:
        times, positions = self._traj
        t = time.time() - self._traj_t0
        if t >= times[-1]:
            self._hold_q = np.array(positions[-1], dtype=float)
            self.mode = ArmMode.ENABLED_HOLD
            self.rs.set_mode(self.side, "ENABLED_HOLD", True)
            self.rs.update_arm(self.side, track_progress=1.0)
            self.rs.log(f"{self.side}: trajectory done, holding end pose")
            return
        idx = bisect.bisect_right(times, t)
        i0, i1 = max(0, idx - 1), min(len(times) - 1, idx)
        t0, t1 = times[i0], times[i1]
        frac = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        p0 = np.array(positions[i0], dtype=float)
        p1 = np.array(positions[i1], dtype=float)
        target = p0 + frac * (p1 - p0)
        self._last_planned_q = target
        self.rs.update_arm(self.side, track_progress=t / times[-1])
        if self.sim:
            self._sim_pos = target.copy()
        else:
            self._mit_hold(target)

    def tracking_q(self) -> tuple[np.ndarray, np.ndarray] | None:
        """(planned_q, actual_q) for the IK tracking-error display.
        During motion (HOMING/GO_*/TRACKING): the interpolated target. While
        holding (ENABLED_HOLD): the held target — so the display KEEPS showing
        the STEADY-STATE error after motion stops (doesn't drain). None only
        when disabled / idle (zero-torque)."""
        if not self.enabled or self._last_actual_q is None:
            return None
        if self.mode in _BUSY:
            planned = self._last_planned_q
        elif self.mode == ArmMode.ENABLED_HOLD:
            planned = self._hold_q
        else:
            return None
        if planned is None:
            return None
        return planned, self._last_actual_q

    # ------------------------------------------------- sim
    def _sim_sense(self):
        pos = self._sim_pos + np.random.normal(0, 0.005, ARM_DOF)
        vel = self._sim_vel + np.random.normal(0, 0.01, ARM_DOF)
        tau = np.random.normal(0, 0.05, ARM_DOF)
        tmos = np.clip(self._sim_tmos + np.random.normal(0, 0.2, ARM_DOF), 30, 46)
        self._sim_tmos = tmos
        return pos, vel, tau, tmos.copy(), tmos.copy() - 3.0
