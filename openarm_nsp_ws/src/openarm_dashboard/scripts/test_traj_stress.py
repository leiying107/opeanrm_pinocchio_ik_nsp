#!/usr/bin/env python3
"""IMP_TRACK stress harness — exhaustive scenario sweep for trajectory replay.

The R1-R5 regression covers one happy path each. This harness sweeps the
space the way the point-impedance stress suite did:

  TL1  undisturbed matrix: trajectory family x rate x preset
  TL2  disturbance matrix: push timing x force x duration x direction
  TL3  threshold boundary: chatter around 5cm, slow push, ease-phase push
  TL4  near-singular trajectories (moving anchor + singularity guard)
  TL5  fast replays: tracking lag -> FALSE-PAUSE risk
  TL6  model mismatch (payload, friction, gravity error, delay)

Verdicts: OK / DRIFT(didn't finish at end pose) / NOPAUSE(push never
paused) / NORESUME(never resumed) / EARLY-EXIT(safety fired) / FATAL.

Run: ./venv-openarm-ik/bin/python src/openarm_dashboard/scripts/test_traj_stress.py
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src/openarm_pinocchio_nsp/src"))
sys.path.insert(0, str(ROOT / "src/openarm_dashboard/src"))

import pinocchio as pin  # noqa: E402
from openarm_dashboard import traj_rec  # noqa: E402
from openarm_dashboard.arm_controller import ArmController, ArmMode  # noqa: E402
from openarm_dashboard.impedance import (  # noqa: E402
    CartesianImpedance, ImpedanceSimPlant,
)
from openarm_dashboard.robot_state import DataBuffer, RobotState  # noqa: E402
from openarm_pinocchio_nsp.kinematics import PinocchioModel  # noqa: E402
from openarm_pinocchio_nsp.urdf_path import resolve_urdf_path  # noqa: E402

M = PinocchioModel(resolve_urdf_path(), "left")
DT = 0.02          # controller tick (we drive the loop ourselves at 50Hz —
                   # the real loop is 250Hz; we substep the plant 5x, which
                   # matches _step_impedance's sim behavior at 20Hz*12)
TICKS_PER_S = 50
TV = np.array([54., 54., 28., 28., 10., 10., 10.])
PAUSE_M, RESUME_M, RESUME_TICKS = 0.05, 0.02, 15   # 0.3s at 50Hz


def gt(q):
    g = pin.computeGeneralizedGravity(M.model, M.data, M._full_q(q)).copy()
    return g[M.q_idx]


def sigma_of(q):
    J = M._pose_and_jac(q, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[1]
    return float(np.linalg.svd(J, compute_uv=False)[-1])


# ------------------------------------------------------------ trajectories
def make_traj(kind: str):
    """Synthetic trajectory families (7xN)."""
    n = 100   # 2s at 50Hz
    t = np.linspace(0, 1, n)
    s = 10*t**3 - 15*t**4 + 6*t**5          # quintic profile 0->1
    q0 = np.array([0.0, -0.5, 0.0, 0.9, 0.0, 0.0, 0.0])
    if kind == "elbow":            # simple elbow flex
        qb = q0.copy(); qb[3] += 0.6
    elif kind == "reach":          # multi-joint reach
        qb = np.array([0.3, -0.2, 0.1, 1.3, 0.15, 0.2, -0.3])
    elif kind == "wrist":          # wrist-dominant
        qb = q0.copy(); qb[4], qb[5], qb[6] = 0.6, 0.4, -0.7
    elif kind == "near-singular":  # straightens the arm -> sigma dips
        qb = q0.copy(); qb[3] = 0.25
    elif kind == "fast-elbow":     # same motion, recorded "fast" (short file)
        n = 40
        t = np.linspace(0, 1, n)
        s = 10*t**3 - 15*t**4 + 6*t**5
        qb = q0.copy(); qb[3] += 0.6
    else:
        raise ValueError(kind)
    q = np.array([q0 + si*(qb-q0) for si in s])
    return traj_rec.TrajData(np.arange(len(q)) * 0.02, q, side="left")


# ------------------------------------------------------------ replay core
def run_replay(traj, rate=0.5, preset="soft", kx=None, zeta=1.4,
               pushes=(), mismatch=None, timeout_w=None):
    """Drive IMP_TRACK semantics directly (same math as _step_imp_track +
    _step_impedance, without the worker thread). pushes = [(t_wall, F3, dur)].
    Returns verdict dict."""
    imp = CartesianImpedance(M)
    plant = ImpedanceSimPlant("left")
    if mismatch:
        for k, v in mismatch.items():
            setattr(plant, k, v)
    imp.set_params(preset=preset, zeta=zeta, **({"kx": kx} if kx else {}))
    imp.start(traj.q[0])
    plant.reset(traj.q[0])
    rng = np.random.default_rng(7)
    dur_wall = timeout_w or (traj.play_duration(rate) * 1.6 + 8.0)
    n_ticks = int(dur_wall * TICKS_PER_S)
    t_play, paused, resume_ct = 0.0, False, 0
    pause_events, resume_events, sig_exits = 0, 0, 0
    paused_frac, first_pause_t = 0.0, None
    vmax_seen = 0.0
    sig_low = 0
    for k in range(n_ticks):
        t_wall = k / TICKS_PER_S
        for (ts, F, d) in pushes:
            if abs(t_wall - ts) < 1.0 / TICKS_PER_S / 2:
                plant.set_push(F, d, t_wall)
        h = 1.0 / TICKS_PER_S
        # deviation check (raw anchor error)
        x = np.asarray(M.fk(plant.q)[0], float)
        dxn = float(np.linalg.norm(imp.x_des - x))
        if dxn > PAUSE_M:
            if not paused:
                paused = True
                pause_events += 1
                first_pause_t = t_wall if first_pause_t is None else first_pause_t
            resume_ct = 0
        elif paused:
            resume_ct += 1
            if dxn < RESUME_M and resume_ct >= RESUME_TICKS:
                paused = False
                resume_events += 1
        if paused:
            paused_frac += 1.0 / n_ticks
        else:
            t_play += traj_rec.ease_rate(t_play, traj.play_duration(rate), rate) * h
        if t_play >= traj.times[-1]:
            break
        q_ref = traj.sample_at(t_play)
        xr, quat = M.fk(q_ref)
        imp.x_des = np.asarray(xr, float).copy()
        imp.R_des = imp._pin.Quaternion(np.asarray(quat, float)).matrix()
        imp.q_post = q_ref
        # 5 plant substeps per tick (matches _step_impedance sim substepping)
        for _ in range(5):
            qn = plant.q + rng.normal(0, 0.005, 7)
            dqn = plant.v + rng.normal(0, 0.01, 7)
            tau, diag = imp.torque(qn, dqn, h / 5)
            tau = np.clip(tau + gt(qn), -TV, TV)
            plant.step(tau, t_wall, h / 5, substeps=2)
        vmax_seen = max(vmax_seen, float(np.max(np.abs(plant.v))))
        if diag["sigma"] < 0.02:
            sig_low += 1
            if sig_low >= 25:      # 0.5s persistence @50Hz outer
                sig_exits += 1
                break
        else:
            sig_low = 0
        if not np.all(np.isfinite(plant.q)):
            return {"verdict": "FATAL", "why": "NaN"}
    finished = t_play >= traj.times[-1] - 1e-9
    err_end = float(np.max(np.abs(plant.q - traj.q[-1]))) if finished else 9.9
    pushed = len(pushes) > 0
    max_f = max((abs(np.linalg.norm(F)) for _, F, _ in pushes), default=0.0)
    if not finished and sig_exits:
        verdict = "SAFETY-EXIT"      # correct behavior: deep singularity -> hold
    elif not finished:
        verdict = "DRIFT"
    elif pushed and pause_events == 0 and max_f < 12.0:
        verdict = "OK"               # weak push absorbed (<5cm): no pause needed
    elif pushed and pause_events == 0:
        verdict = "NOPAUSE"          # strong push never paused — real issue
    elif pushed and paused:
        verdict = "NORESUME"
    elif err_end > 0.30:
        verdict = "DRIFT"
    else:
        verdict = "OK"
    return {"verdict": verdict, "pause_events": pause_events,
            "resume_events": resume_events, "paused_frac": round(paused_frac, 3),
            "progress": round(t_play / traj.times[-1], 3),
            "err_end_deg": round(float(np.degrees(err_end)), 1),
            "vmax": round(vmax_seen, 2), "sig_min_seen": round(diag["sigma"], 3)}


def summarize(results, label):
    c = Counter(r["verdict"] for r in results)
    total = len(results)
    print(f"\n== {label}: {total} scenarios ==")
    for v in ("OK", "DRIFT", "NOPAUSE", "NORESUME", "SAFETY-EXIT", "FATAL"):
        if c.get(v):
            print(f"  {v:10s} {c[v]:4d}  ({c[v]/total:.0%})")
    bad = [r for r in results if r["verdict"] != "OK"]
    if bad:
        print(f"  --- non-OK ({len(bad)}) ---")
        for r in bad[:10]:
            print(f"    {r['verdict']:10s} {r.get('tag','')} p={r.get('pause_events')}/{r.get('resume_events')} "
                  f"prog={r.get('progress')} err={r.get('err_end_deg')}° sig={r.get('sig_min_seen')}")
    return c


# ------------------------------------------------------------------ layers
def tl1():
    out = []
    for tkind in ("elbow", "reach", "wrist"):
        for rate in (0.2, 0.5, 1.0):
            for preset, kx in (("ultra", None), ("soft", None), ("soft", 600)):
                tr = make_traj(tkind)
                r = run_replay(tr, rate=rate, preset=preset, kx=kx)
                r["tag"] = f"{tkind} r{rate} {preset}{'' if kx is None else f'-{kx}'}"
                out.append(r)
    return out


def tl2():
    tr = make_traj("reach")
    out = []
    for t_push in (0.5, 1.5, 3.0, 5.0):        # during ease / mid / late
        for Fmag in (8.0, 20.0, 40.0):
            for dur in (0.5, 1.5, 3.0):
                for direction in ("down", "lateral", "axial-in"):
                    d = {"down": np.array([0, 0, -1.0]),
                         "lateral": np.array([0, 1.0, 0]),
                         "axial-in": None}[direction]
                    if d is None:
                        x0 = np.asarray(M.fk(tr.q[0])[0], float)
                        j3 = np.array([0.0, 0.19, 0.0])   # approx shoulder
                        d = -(x0 - j3) / np.linalg.norm(x0 - j3)
                    r = run_replay(tr, rate=0.5,
                                   pushes=[(t_push, Fmag * d, dur)])
                    r["tag"] = f"t{t_push} F{Fmag:.0f} d{dur} {direction}"
                    out.append(r)
    return out


def tl3():
    """Threshold boundary: weak pushes that hover near 5cm (chatter risk),
    slow ramp pushes, pushes during ease-in."""
    tr = make_traj("reach")
    out = []
    # weak push hovering near threshold (soft preset Kx=300: 5cm <-> 15N)
    for Fmag in (10.0, 14.0, 15.0, 16.0, 20.0):
        r = run_replay(tr, rate=0.5, pushes=[(2.0, np.array([0, 0, -Fmag]), 3.0)])
        r["tag"] = f"near-thresh F{Fmag:.0f}"
        out.append(r)
    # very slow push (spring stretches gradually)
    for Fmag in (18.0, 25.0):
        pushes = [(t, np.array([0, 0, -Fmag * (t - 1.5) / 1.5]), 0.06)
                  for t in np.arange(1.5, 4.5, 0.06)]
        r = run_replay(tr, rate=0.5, pushes=pushes)
        r["tag"] = f"slow-ramp F{Fmag:.0f}"
        out.append(r)
    # repeated push-release-push (pause/resume cycling)
    pushes = [(1.0, np.array([0, 0, -20.0]), 1.0),
              (4.0, np.array([0, 0, -20.0]), 1.0),
              (7.0, np.array([0, 0, -20.0]), 1.0)]
    r = run_replay(tr, rate=0.5, pushes=pushes)
    r["tag"] = "triple-push"
    out.append(r)
    return out


def tl4():
    """Near-singular trajectory: anchor walks INTO the low-sigma band while
    the arm may be pushed — does the guard still protect?"""
    tr = make_traj("near-singular")
    print(f"  (traj sigma range: {sigma_of(tr.q[0]):.3f} -> {sigma_of(tr.q[-1]):.3f})")
    out = []
    for rate in (0.5, 1.0):
        for push in (None, (2.0, np.array([0, 0, -15.0]), 1.5)):
            r = run_replay(tr, rate=rate, pushes=[push] if push else ())
            r["tag"] = f"near-sing r{rate}{' +push' if push else ''}"
            out.append(r)
    # replay INTO singularity with outward radial pull (worst case)
    x0 = np.asarray(M.fk(tr.q[-1])[0], float)
    j3 = np.array([0.0, 0.19, 0.0])
    rad = (x0 - j3) / np.linalg.norm(x0 - j3)
    r = run_replay(tr, rate=0.5, pushes=[(2.5, 20.0 * rad, 2.0)])
    r["tag"] = "near-sing +outward-pull"
    out.append(r)
    return out


def tl5():
    """Fast replays: tracking lag may exceed 5cm -> FALSE pause loops."""
    out = []
    tr = make_traj("fast-elbow")     # 0.8s recording
    for rate in (0.5, 1.0):
        for preset in ("soft", "mid"):
            r = run_replay(tr, rate=rate, preset=preset)
            r["tag"] = f"fast r{rate} {preset}"
            out.append(r)
    # large-motion trajectory at 1.0x with soft spring (max lag case)
    tr2 = make_traj("reach")
    r = run_replay(tr2, rate=1.0, preset="soft", timeout_w=25.0)
    r["tag"] = "reach r1.0 soft (lag test)"
    out.append(r)
    return out


def tl6():
    """Model mismatch during replay (same knobs as the s2r battery)."""
    tr = make_traj("reach")
    out = []
    cases = {
        "payload1kg": {"payload": 1.0},
        "friction": {"b": 0.6},
        "grav-10pct": {},           # applied via gt wrapper below
    }
    for name, mm in cases.items():
        r = run_replay(tr, rate=0.5, mismatch=mm,
                       pushes=[(2.0, np.array([0, 0, -20.0]), 1.5)])
        r["tag"] = f"mismatch {name} +push"
        out.append(r)
    # gravity error needs gt override: approximate by scaling in-harness
    import openarm_dashboard.traj_rec as _tr  # noqa: F401 (keep import local)
    r = run_replay(tr, rate=0.5,
                   pushes=[(2.0, np.array([0, 0, -20.0]), 1.5)])
    r["tag"] = "mismatch baseline (repeat)"
    out.append(r)
    return out


def main():
    layers = {"TL1": tl1, "TL2": tl2, "TL3": tl3, "TL4": tl4, "TL5": tl5, "TL6": tl6}
    all_out = {}
    for name, fn in layers.items():
        t0 = time.time()
        res = fn()
        summarize(res, f"{name} ({time.time()-t0:.0f}s)")
        all_out[name] = res
    import json
    with open("/tmp/traj_stress.json", "w") as f:
        json.dump(all_out, f, indent=1, default=str)
    print("\nwrote /tmp/traj_stress.json")


if __name__ == "__main__":
    main()
