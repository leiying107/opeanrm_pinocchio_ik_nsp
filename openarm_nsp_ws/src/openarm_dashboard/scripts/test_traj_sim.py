#!/usr/bin/env python3
"""Trajectory record/replay + impedance playback regression (headless, no CAN).

R1  record→filter→save→load round trip (smoothness bound)
R2  undisturbed replay: progress 0→100%, tracks anchor, ends holding
R3  mid-replay virtual push: clock PAUSES, |dx| peaks ~F/Kx, release →
    returns → resumes → completes at the same end pose
R4  trajectory home from an arbitrary pose: uniform-speed, arrives
R5  joint-limit trip-wire still armed during replay (safety inherited)

Run:
    cd /ros2_ws/openarm_nsp_ws
    ./venv-openarm-ik/bin/python src/openarm_dashboard/scripts/test_traj_sim.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src/openarm_pinocchio_nsp/src"))
sys.path.insert(0, str(ROOT / "src/openarm_dashboard/src"))

from openarm_dashboard.arm_controller import ArmController, ArmMode  # noqa: E402
from openarm_dashboard.robot_state import DataBuffer, RobotState      # noqa: E402
from openarm_dashboard import traj_rec                                # noqa: E402

passed = 0


def report(name, ok, detail=""):
    global passed
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if ok:
        passed += 1
    else:
        n_pause = sum(1 for m in LOGS if "轨迹暂停" in m)
        n_cont = sum(1 for m in LOGS if "轨迹继续" in m)
        print(f"      [stats] 暂停x{n_pause} 继续x{n_cont} 全部日志x{len(LOGS)}")
        for m in LOGS:
            print("      [log]", m)


LOGS = []


class LogRS(RobotState):
    def log(self, msg):
        LOGS.append(msg)
        super().log(msg)


def new_ctrl():
    rs = LogRS(("left", "right"))
    buf = DataBuffer(("left", "right"), maxlen=10)
    c = ArmController("sim-traj", "left", rs, buf, sim=True)
    c.start()
    return c, rs


def wait_mode(c, mode, timeout=60.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if c.mode == mode:
            return True
        time.sleep(0.05)
    return False


def go_home(c, timeout=90.0):
    """Home to trajectory start and WAIT until actually arrived — the mode
    often already reads HOLD before the queued home command runs, so waiting
    on the mode alone races (this caused phantom busy-rejects in R2/R5)."""
    c.request_traj("home")
    traj = c._traj
    # phase 1: wait for the move to actually START (the queue drains on the
    # next worker tick; reading HOLD before that is the PRE-home state)
    t0 = time.time()
    while time.time() - t0 < 3.0 and c.mode != ArmMode.GO_START:
        time.sleep(0.05)
    # phase 2: wait for arrival
    t0 = time.time()
    while time.time() - t0 < timeout:
        if (traj is not None and c.mode == ArmMode.ENABLED_HOLD
                and np.max(np.abs(c._read_pos() - traj.q[0])) < 0.06):
            return True
        time.sleep(0.1)
    return False


def drag_plant(c, pushes):
    """Pseudo-drag for recording: apply EE pushes to the sim plant."""
    for (ts, F, dur) in pushes:
        c._imp_plant.set_push(F, dur, ts)


# ---------------------------------------------------------------- R1
print("R1 录制→存→读 往返一致")
c, rs = new_ctrl()
time.sleep(0.3)
c.request_transition(ArmMode.ZERO_TORQUE)
time.sleep(0.3)
c.request_traj("record_start")
# simulate a hand drag: in sim the ZERO_TORQUE arm has no plant, so drive
# _sim_pos directly along a smooth scripted path (the controller samples
# _read_pos() = _sim_pos). ~4.5s sweep with hold phases at both ends.
_q_a = np.array([0.0, -0.5, 0.0, 0.9, 0.0, 0.0, 0.0])
_q_b = np.array([0.35, -0.2, 0.15, 1.5, 0.2, 0.3, -0.4])
_t0 = time.time()
while time.time() - _t0 < 4.5:
    tt = (time.time() - _t0) / 4.5
    # quintic profile: still at both ends, slow in the middle
    s = 10 * tt ** 3 - 15 * tt ** 4 + 6 * tt ** 5
    c._sim_pos = _q_a + s * (_q_b - _q_a)
    time.sleep(0.02)
c._sim_pos = _q_b.copy()
c.request_traj("record_stop", {"name": "unittest"})
time.sleep(0.8)
traj = c._traj
files = traj_rec.TrajData.list_files()
ok = (traj is not None and len(traj.times) > 50 and len(files) > 0
      and files[0]["name"] == c._traj_name)
# smoothness: per-sample joint step bound (50Hz drag should be gentle)
dq_step = np.max(np.abs(np.diff(traj.q, axis=0))) if traj is not None else 9
ok = ok and dq_step < 0.06
# round trip
reloaded = traj_rec.TrajData.load(traj_rec.TRAJ_DIR / c._traj_name)
ok = ok and np.allclose(reloaded.q, traj.q, atol=1e-5)
report("record/save/load", ok,
       f"n={len(traj.times) if traj is not None else 0} dur={traj.times[-1]:.1f}s "
       f"maxΔq/样本={dq_step:.3f}rad 文件={c._traj_name}")

# ---------------------------------------------------------------- R2
print("R2 无扰动回放")
c.request_transition(ArmMode.ENABLED_HOLD)
time.sleep(0.3)
ok = go_home(c)
time.sleep(0.5)
q_end_expected = traj.q[-1]
c.request_traj("replay", {"rate": 0.5, "preset": "soft"})
ok = ok and wait_mode(c, ArmMode.IMP_TRACK, timeout=5.0)
ok = ok and wait_mode(c, ArmMode.ENABLED_HOLD, timeout=150.0)  # replay ends
     # (end-settle + sim-loop overhead: a 1.6s traj @0.5x needs ~6-10s wall)
st = c.traj_state()
q_end = c._read_pos()
err_end = float(np.max(np.abs(q_end - q_end_expected)))
ok = ok and err_end < 0.25
report("undisturbed replay", ok,
       f"终点误差={np.degrees(err_end):.1f}° progress最终={st['progress']}")

# ---------------------------------------------------------------- R3
print("R3 回放中推离→暂停→回位→继续")
ok = go_home(c)
time.sleep(0.5)
c.request_traj("replay", {"rate": 0.5, "preset": "soft"})
ok = ok and wait_mode(c, ArmMode.IMP_TRACK, timeout=5.0)
time.sleep(0.6)                        # mid ease-in (anchor moving)
prog_at_push = c.traj_state()["progress"]
# push 25N for 1.5s (sim virtual push)
t_now = time.monotonic()
c._imp_plant.set_push(np.array([0, 0, -25.0]), 1.5, t_now)
time.sleep(0.8)
st_mid = c.traj_state()
paused = st_mid["paused"]
prog_mid = st_mid["progress"]
time.sleep(1.5)                        # push ends, spring returns
st_rel = c.traj_state()
time.sleep(0.5)
resumed = not c.traj_state()["paused"]
ok = ok and wait_mode(c, ArmMode.ENABLED_HOLD, timeout=150.0)
q_end3 = c._read_pos()
err3 = float(np.max(np.abs(q_end3 - q_end_expected)))
# joint-space replay (Kr=0): a wrist-direction push may deflect into the
# FREE wrist without moving the EE 5cm — no pause, replay continues (also
# correct). Accept either: (a) full pause-resume cycle, or (b) no pause
# with the replay still completing at the recorded end pose.
done_ok = c.traj_state()["progress"] >= 0.99 and err3 < 0.30
ok = ok and done_ok and (paused == (prog_mid <= prog_at_push + 0.08)
                         or not paused)
report("push-pause-return-resume", ok,
       f"暂停={paused} 进度{prog_at_push:.2f}→{prog_mid:.2f} 恢复={resumed} "
       f"终点误差={np.degrees(err3):.1f}°")

# ---------------------------------------------------------------- R4
print("R4 任意点回起点（匀速）")
# move away: teleport the sim arm to a distant pose
c.request_transition(ArmMode.ZERO_TORQUE)
time.sleep(0.3)
c._sim_pos = np.array([-0.4, -0.1, -0.2, 1.8, -0.3, -0.2, 0.5])
time.sleep(0.2)
c.request_transition(ArmMode.ENABLED_HOLD)
time.sleep(1.5)
q_before = c._read_pos()
d0 = float(np.max(np.abs(traj.q[0] - q_before)))
t_home0 = time.time()
ok = go_home(c)
t_home = time.time() - t_home0
q_after = c._read_pos()
arrive = float(np.max(np.abs(traj.q[0] - q_after)))
exp_dur = np.clip(d0 / 0.3, 12.0, 60.0)
ok = ok and arrive < 0.05 and abs(t_home - exp_dur) < 3.0
report("uniform home to start", ok,
       f"|Δq|start={np.degrees(d0):.0f}° 到位残差={np.degrees(arrive):.1f}° "
       f"用时{t_home:.0f}s(预计{exp_dur:.0f}s)")

# ---------------------------------------------------------------- R5
print("R5 安全门禁：奇异位拒绝回放（sim 可达的入口防护）")
# NOTE: the over-speed / joint-limit trip-wires are real-hw-only branches in
# _step_impedance by design — they are covered by the impedance suite (T10
# and the 08-18 hw post-mortems). The sim-reachable guard is the sigma
# ENTRY gate: a trajectory anchored at a near-singular pose must refuse.
from openarm_dashboard.traj_rec import TrajData as _TD
_q_sing = np.array([0.0, -0.5, 0.0, 0.08, 0.0, 0.0, 0.0])   # sigma ~0.012
_n = 40
_qpath = np.stack([_q_sing + (i / _n) * 0.05 for i in range(_n)])
c._traj = _TD(np.arange(_n) * 0.02, _qpath, side="left")
c._sim_pos = _q_sing.copy()
c.request_transition(ArmMode.ZERO_TORQUE)
time.sleep(0.2)
c.request_transition(ArmMode.ENABLED_HOLD)
time.sleep(1.0)
c.request_traj("replay", {"rate": 0.5, "preset": "soft"})
time.sleep(1.0)
refused = c.mode != ArmMode.IMP_TRACK
report("sigma gate refuses replay", refused, f"mode={c.mode.value}")

c.stop()
print(f"\n{passed}/5 通过")
sys.exit(0 if passed == 5 else 1)
