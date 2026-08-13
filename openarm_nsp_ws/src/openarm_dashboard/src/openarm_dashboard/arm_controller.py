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
import queue
import subprocess
import threading
import time

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


_BUSY = {ArmMode.HOMING, ArmMode.GO_START, ArmMode.GO_END, ArmMode.TRACKING}


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


class ArmController:
    def __init__(self, iface: str, side: str, robot_state: RobotState,
                 buffer: DataBuffer, sim: bool = False, with_gripper: bool = True):
        self.iface, self.side, self.sim = iface, side, sim
        self.rs, self.buf = robot_state, buffer
        self.with_gripper = with_gripper
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
        # tracking trajectory
        self._traj: tuple[list[float], list[np.ndarray]] | None = None
        self._traj_t0 = 0.0
        # latest planned vs actual joint angles (for IK tracking-error display)
        self._last_planned_q: np.ndarray | None = None
        self._last_actual_q: np.ndarray | None = None
        # sim state
        self._sim_pos = np.zeros(ARM_DOF)
        self._sim_tmos = np.full(ARM_DOF, 34.0)
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

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.enabled:
            self._disable()

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

    def _process(self, mode: ArmMode, trajectory) -> None:
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

    def _begin_move(self, target: np.ndarray, mode: ArmMode, msg: str) -> None:
        self._move_target = np.asarray(target, dtype=float).copy()
        self._move_start = self._read_pos()
        self._move_t0 = time.time()
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

    def _disable(self) -> None:
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

    # ------------------------------------------------- step
    def _step(self) -> None:
        pos, vel, tau, tmos, trotor = self._sense()
        self._last_actual_q = pos
        self.rs.update_arm(self.side, position=pos, velocity=vel, torque=tau,
                           tmos=tmos, trotor=trotor)
        self.buf.append(self.side, pos, tau, tmos)
        if np.max(tmos) > TMOS_ERROR and self.mode != ArmMode.DISABLED:
            self.rs.log(f"{self.side}: OVERHEAT {float(np.max(tmos)):.0f}°C → disable")
            self._disable()
            self.mode = ArmMode.DISABLED
            self.rs.set_mode(self.side, "DISABLED", False)
            return
        self._actuate()

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
            return
        if self.mode == ArmMode.ZERO_TORQUE:
            if not self.sim:
                self.oa.get_arm().mit_control_all([MITParam(0, 0, 0, 0, 0)] * ARM_DOF)
        elif self.mode == ArmMode.ENABLED_HOLD:
            if self.sim:
                self._sim_pos += 0.1 * (self._hold_q - self._sim_pos)
            else:
                self._mit_hold(self._hold_q)
        elif self.mode in (ArmMode.HOMING, ArmMode.GO_START, ArmMode.GO_END):
            self._step_move()
        elif self.mode == ArmMode.TRACKING:
            self._step_track()

    def _mit_hold(self, target: np.ndarray) -> None:
        params = [MITParam(ARM_KP[i], ARM_KD[i], target[i], 0, 0) for i in range(ARM_DOF)]
        self.oa.get_arm().mit_control_all(params)

    def _step_move(self) -> None:
        """Smooth linear interpolation _move_start -> _move_target over MOVE_DUR."""
        t = time.time() - self._move_t0
        if t >= MOVE_DUR:
            self._hold_q = self._move_target.copy()
            self.mode = ArmMode.ENABLED_HOLD
            self.rs.set_mode(self.side, "ENABLED_HOLD", True)
            self.rs.update_arm(self.side, track_progress=1.0)
            self.rs.log(f"{self.side}: reached target, holding")
            return
        frac = t / MOVE_DUR
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
        """Latest (planned_q, actual_q) during a moving mode (HOMING / GO_* /
        TRACKING), or None when idle. The UI FKs both to show tracking error."""
        if self.mode in _BUSY and self._last_planned_q is not None and self._last_actual_q is not None:
            return self._last_planned_q, self._last_actual_q
        return None

    # ------------------------------------------------- sim
    def _sim_sense(self):
        pos = self._sim_pos + np.random.normal(0, 0.005, ARM_DOF)
        vel = np.random.normal(0, 0.01, ARM_DOF)
        tau = np.random.normal(0, 0.05, ARM_DOF)
        tmos = np.clip(self._sim_tmos + np.random.normal(0, 0.2, ARM_DOF), 30, 46)
        self._sim_tmos = tmos
        return pos, vel, tau, tmos.copy(), tmos.copy() - 3.0
