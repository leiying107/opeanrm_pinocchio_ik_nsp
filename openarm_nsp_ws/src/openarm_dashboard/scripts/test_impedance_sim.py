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
  T5 soft escape : forced drift toward straight-arm stops at sigma ~= 0.05
  T6 hard exit   : sigma below SIG_CRIT falls back to motor-side PD hold
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]           # openarm_nsp_ws
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

# ------------------------------------------------------ T5 soft escape
print("T5 软逃逸：强制拉向伸直臂，σ 在 0.05 边界被拉住")
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
# wide-band escape (v4): blends from SIG_BLEND_HI=0.12, so the equilibrium
# sits FURTHER from singularity than the old narrow band (0.062 vs 0.050)
ok = (c.mode == ArmMode.IMPEDANCE and worst >= 0.055
      and 0.55 < q4_last < 0.95)   # held off straight-arm, no hard exit
passed += report("escape holds sigma boundary", ok,
                 f"worst σ={worst:.3f} q4={q4_last:.2f} (0=straight) mode={c.mode.value}")

# ---------------------------------------------------------- T6 hard exit
print("T6 硬退出：σ 低于临界 → 回电机侧 PD")
c2_ok = True
c._sig_crit = 0.07          # raise threshold so the current pose trips it
time.sleep(0.4)             # next tick: torque() returns sigma=0.065 < 0.07
for _ in range(30):
    if c.mode != ArmMode.IMPEDANCE:
        break
    time.sleep(0.05)
ok = c.mode == ArmMode.ENABLED_HOLD
last_log = rs.recent_messages(2)[-1] if rs.recent_messages(2) else ""
passed += report("hard exit to PD hold", ok, f"mode={c.mode.value} | {last_log}")
c.stop()

print(f"\n{passed}/6 通过")
sys.exit(0 if passed == 6 else 1)
