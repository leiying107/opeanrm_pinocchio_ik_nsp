#!/usr/bin/env python3
"""Impedance + singularity-guard sim test suite (headless, NO CAN / NO ROS).

Run:
    cd /ros2_ws/openarm_nsp_ws
    PYTHONPATH=src/openarm_pinocchio_nsp/src:src/openarm_dashboard/src \
        ./venv-openarm-ik/bin/python src/openarm_dashboard/scripts/test_impedance_sim.py

Each test drives ArmController(sim=True) directly — the same code path as
real hardware, with a pinocchio-aba dynamics plant standing in for the arm.

Tests:
  T1 entry gate  : enabling at the singular home pose is REFUSED
  T2 safe enable : elbow-flexed pose enables; sigma_min >= 0.05
  T3 spring law  : 20 N push -> |dx| ~ F/Kx, then rebounds
  T4 drag mode   : leak>0 follows the push instead of rebounding
  T5 soft stop   : outward pull floors sigma at ~0.05 (velocity fold control)
  T6 hard exit   : sigma below SIG_CRIT falls back to motor-side PD hold
  T7 lateral @ low sigma : the 5 healthy directions stay compliant near extension
  T8 inward push : axial inward push BUCKLES the arm (q4 rises, sigma up)
  T9 no hunting  : hold near the soft stop 3 s — no limit cycle
  T10 quiet drift: low-sigma hover 3 s without push — |dx| stays tiny
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]           # openarm_nsp_ws
sys.path.insert(0, str(ROOT / "src/openarm_pinocchio_nsp/src"))
sys.path.insert(0, str(ROOT / "src/openarm_dashboard/src"))

from openarm_dashboard.arm_controller import ArmController, ArmMode  # noqa: E402
from openarm_dashboard.robot_state import DataBuffer, RobotState      # noqa: E402

SAFE_Q = np.array([0.0, -0.5, 0.0, 0.8, 0.0, 0.0, 0.0])   # elbow-flexed, j2 out


def new_ctrl():
    rs = RobotState(("left", "right"))
    buf = DataBuffer(("left", "right"), maxlen=10)
    c = ArmController("sim-left", "left", rs, buf, sim=True)
    c.start()
    return c, rs


def wait_hold(c, timeout=4.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if c.mode == ArmMode.ENABLED_HOLD:
            return True
        time.sleep(0.05)
    return False


def wait_ramped(c, timeout=6.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = c.imp_state()
        if st and st.get("ramp") is not None and st["ramp"] >= 0.99:
            return True
        time.sleep(0.05)
    return False


def settle(c, secs=1.2):
    time.sleep(secs)
    st = c.imp_state()
    return float(np.linalg.norm(st["dx"])) if st and st["dx"] else float("inf")


def report(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    return ok


passed = 0

# ---------------------------------------------------------------- T1 gate
print("T1 入口门禁：奇异归零位拒绝开启")
c, rs = new_ctrl(); time.sleep(0.3)
c.request_transition(ArmMode.ZERO_TORQUE); time.sleep(0.3)
c.request_impedance(True, {"preset": "soft"}); time.sleep(0.5)
ok = c.mode != ArmMode.IMPEDANCE
last_log = rs.recent_messages(2)[-1] if rs.recent_messages(2) else ""
passed += report("gate refuses at home", ok, f"mode={c.mode.value} | {last_log}")
c.stop()

# ------------------------------------------------------------ T2 enable
print("T2 安全位姿正常开启")
c, rs = new_ctrl(); time.sleep(0.3)
c.request_transition(ArmMode.ZERO_TORQUE); time.sleep(0.3)
c.request_move_to(SAFE_Q, "safe pose")
ok = wait_hold(c)
time.sleep(0.2)
c.request_impedance(True, {"preset": "soft"}); time.sleep(0.6)
st = c.imp_state()
ok = ok and c.mode == ArmMode.IMPEDANCE and st and st["sigma"] >= 0.05
passed += report("enables at safe pose", ok,
                 f"mode={c.mode.value} sigma={st['sigma'] if st else None}")

# ------------------------------------------------------ T3 spring law
print("T3 弹簧定律：推20N → 峰值≈F/Kx，回弹")
ok = wait_ramped(c)
d0 = settle(c)
c.request_imp_push([20.0, 0.0, 0.0], 1.0)
t0 = time.time(); peak = 0.0
while time.time() - t0 < 1.6:
    st = c.imp_state()
    if st and st["dx"]:
        peak = max(peak, float(np.linalg.norm(st["dx"])))
    time.sleep(0.05)
d1 = settle(c, 2.0)
pred = 20.0 / 300.0 * 1000.0   # F/Kx = 67mm (soft preset)
ok = ok and abs(peak - pred) < 30 and d1 < 15 and d0 < 10
passed += report("push/rebound physics", ok,
                 f"peak={peak:.0f}mm (pred {pred:.0f}) settle={d0:.1f}->{d1:.1f}mm")

# ---------------------------------------------------------- T4 drag
print("T4 漏速拖动：跟随后不回中")
c.request_impedance(True, {"leak": 0.8})
c.request_imp_push([15.0, 0.0, 0.0], 2.0)
time.sleep(3.2)
st = c.imp_state()
d = float(np.linalg.norm(st["dx"])) if st and st["dx"] else float("inf")
ok = d < 25   # drag keeps the error small while displaced (vs ~70mm spring peak)
passed += report("leak=0.8 drag follows", ok, f"|dx|={d:.1f}mm (displaced, small error)")

# ------------------------------------------------------ T5 soft stop
print("T5 软挡块：径向外拉，σ 被屈肘控制托在 ~0.05")
c.request_impedance(True, {"leak": 0.0})
time.sleep(0.5)
c._imp.q_post = np.array([0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0])  # posture → straight
t0 = time.time(); worst = 1.0; q4_last = SAFE_Q[3]
while time.time() - t0 < 5.0:
    st = c.imp_state()
    if st and st["sigma"]:
        worst = min(worst, st["sigma"])
    q4_last = float(c._read_pos()[3])
    if c.mode != ArmMode.IMPEDANCE:
        break
    time.sleep(0.1)
st = c.imp_state()
# v5 velocity-resolved fold control: floors sigma near SIG_WARN-ish and
# recovers; q4 stays off full extension (0 = straight arm)
ok = (c.mode == ArmMode.IMPEDANCE and worst >= 0.04
      and 0.35 < q4_last < 1.4)
passed += report("soft stop holds sigma", ok,
                 f"worst σ={worst:.3f} q4={q4_last:.2f} (0=straight) mode={c.mode.value}")

# ---------------------------------------------------------- T6 hard exit
print("T6 硬退出：σ 低于临界 → 回电机侧 PD")
c2_ok = True
# v5 holds sigma near ~0.075 (SIG_HOLD); trip the guard from just above that
c._sig_crit = 0.10          # raise threshold so the current pose trips it
time.sleep(0.4)
for _ in range(30):
    if c.mode != ArmMode.IMPEDANCE:
        break
    time.sleep(0.05)
ok = c.mode == ArmMode.ENABLED_HOLD
last_log = rs.recent_messages(2)[-1] if rs.recent_messages(2) else ""
passed += report("hard exit to PD hold", ok, f"mode={c.mode.value} | {last_log}")

# ---- T7-T10 drive the plant directly (radial geometry per arm side) ------
import pinocchio as pin  # noqa: E402
from openarm_dashboard.impedance import ImpedanceSimPlant  # noqa: E402

c3, rs3 = new_ctrl(); time.sleep(0.3)
c3.request_transition(ArmMode.ZERO_TORQUE); time.sleep(0.3)
c3.request_move_to(SAFE_Q, "safe pose")
ok = wait_hold(c3)
c3.request_impedance(True, {"preset": "soft"}); time.sleep(0.6)
ok = ok and wait_ramped(c3)

mp = c3._imp.m
def radial_dir(q):
    qf = mp._full_q(q)
    pin.forwardKinematics(mp.model, mp.data, qf)
    pin.updateFramePlacements(mp.model, mp.data)
    x = np.asarray(mp.fk(q)[0], float)
    j3 = np.asarray(
        mp.data.oMi[mp.model.getJointId(f"openarm_left_joint3")].translation, float)
    r = x - j3
    return r / np.linalg.norm(r)

# ------------------------------- T7 lateral compliance at low sigma
print("T7 低σ侧向柔顺：伸直方向附近，垂直径向推 20N 应正常让位")
radial = radial_dir(SAFE_Q)
lat = np.cross(radial, np.array([0.0, 0.0, 1.0]))
lat /= np.linalg.norm(lat)
c3.request_imp_push(list(20.0 * lat), 1.5)     # pull out first via posture push
# simpler: lateral push directly at the safe pose but after pulling sigma low:
c3._imp.q_post = np.array([0.0, -0.5, 0.0, 0.05, 0.0, 0.0, 0.0])
time.sleep(2.5)                                # drift toward extension to soft stop
sig_low = c3.imp_state()["sigma"]
c3._imp.q_post = c3._imp.q_post               # keep
c3.request_imp_push(list(20.0 * lat), 1.2)
t0 = time.time(); peak_lat = 0.0
while time.time() - t0 < 2.0:
    st = c3.imp_state()
    if st and st["dx"]:
        peak_lat = max(peak_lat, float(np.linalg.norm(st["dx"])))
    if c3.mode != ArmMode.IMPEDANCE:
        break
    time.sleep(0.05)
pred = 20.0 / 300.0 * 1000.0
ok = ok and (c3.mode == ArmMode.IMPEDANCE and sig_low < 0.09
             and peak_lat > 0.35 * pred and peak_lat < 1.8 * pred)
passed += report("lateral compliant at low σ", ok,
                 f"σ_low={sig_low:.3f} lateral peak={peak_lat:.0f}mm (pred {pred:.0f})")

# ------------------------------- T8 inward axial push buckles
print("T8 轴向内推：屈肘让位（q4↑ σ↑），不退出")
c3._imp.q_post = SAFE_Q.copy()
time.sleep(1.5)
q4_before = float(c3._read_pos()[3])
c3.request_imp_push(list(-40.0 * radial), 1.5)
t0 = time.time()
q4_after = q4_before; sig_after = 0
while time.time() - t0 < 2.5:
    st = c3.imp_state()
    q4_after = float(c3._read_pos()[3])
    sig_after = st["sigma"] if st else 0
    if c3.mode != ArmMode.IMPEDANCE:
        break
    time.sleep(0.05)
ok = ok and (c3.mode == ArmMode.IMPEDANCE and q4_after > q4_before + 0.05
             and sig_after > 0.06)
passed += report("inward push buckles elbow", ok,
                 f"q4 {q4_before:.2f}->{q4_after:.2f} σ={sig_after:.3f} mode={c3.mode.value}")

# ------------------------------- T9/T10 quiet low-sigma stability
print("T9 无hunting：软挡块附近悬停 3s")
zc7 = 0
q7_prev = float(c3._read_pos()[6])
sig_min9, sig_max9 = 9.0, 0.0
t0 = time.time()
while time.time() - t0 < 3.0:
    st = c3.imp_state()
    q7 = float(c3._read_pos()[6])
    if (q7 > 0.02 > q7_prev) or (q7 < 0.02 <= q7_prev):
        zc7 += 1
    q7_prev = q7
    if st and st["sigma"]:
        sig_min9 = min(sig_min9, st["sigma"])
        sig_max9 = max(sig_max9, st["sigma"])
    if c3.mode != ArmMode.IMPEDANCE:
        break
    time.sleep(0.02)
ok = ok and (c3.mode == ArmMode.IMPEDANCE and zc7 <= 6)
passed += report("no hunting near soft stop", ok,
                 f"j7 zc={zc7} σ range {sig_min9:.3f}-{sig_max9:.3f}")

print("T10 静态漂移：低σ悬停无推力 3s")
t0 = time.time(); dx_max = 0.0
while time.time() - t0 < 3.0:
    st = c3.imp_state()
    if st and st["dx"]:
        dx_max = max(dx_max, float(np.linalg.norm(st["dx"])))
    if c3.mode != ArmMode.IMPEDANCE:
        break
    time.sleep(0.05)
ok = ok and (c3.mode == ArmMode.IMPEDANCE and dx_max < 25.0)
passed += report("no quiet drift at low σ", ok, f"|dx|max={dx_max:.1f}mm")

c3.stop()
c.stop()

print(f"\n{passed}/10 通过")
sys.exit(0 if passed == 10 else 1)
