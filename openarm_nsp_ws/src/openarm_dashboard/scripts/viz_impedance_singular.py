#!/usr/bin/env python3
"""Sweep the two singular-entry scenarios and dump JSON for visualization.

Scenario A — ENABLE at a near-singular pose (entry gate bypassed):
  j4 swept 0.20..0.75 (sigma 0.017..0.067), each pose run quiet 7 s.
  Question: does v5 pull the arm OUT of the singular region by itself,
  hold, or hard-exit?

Scenario B — ENABLED at a safe pose, then PUSHED INTO singularity:
  40 N radial-outward push at t=2 for 2 s, from j4 = 1.4 (sigma 0.105).
  Question: where does sigma floor, what does the elbow do, does it
  recover after release?

Dumps /tmp/imp_singular_sweep.json with full time series.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src/openarm_pinocchio_nsp/src"))
sys.path.insert(0, str(ROOT / "src/openarm_dashboard/src"))

import pinocchio as pin  # noqa: E402
from openarm_dashboard.impedance import CartesianImpedance, ImpedanceSimPlant  # noqa: E402
from openarm_pinocchio_nsp.kinematics import PinocchioModel  # noqa: E402
from openarm_pinocchio_nsp.urdf_path import resolve_urdf_path  # noqa: E402

m = PinocchioModel(resolve_urdf_path(), "left")
DT = 0.004
TMAXV = np.array([54., 54., 28., 28., 10., 10., 10.])


def gt(q):
    g = pin.computeGeneralizedGravity(m.model, m.data, m._full_q(q)).copy()
    return g[m.q_idx]


def sigma_of(q):
    J = m._pose_and_jac(q, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[1]
    return float(np.linalg.svd(J, compute_uv=False)[-1])


def simulate(q0, pushes=(), dur=7.0, preset="soft", seed=3):
    imp = CartesianImpedance(m)
    plant = ImpedanceSimPlant("left")
    imp.set_params(preset=preset)
    imp.start(q0)
    plant.reset(q0)
    plant._push = None
    plant._push_until = 0.0
    rng = np.random.default_rng(seed)
    ts, sig, q4s, buck, dxs, vs = [], [], [], [], [], []
    outcome = "held"
    for k in range(int(dur / DT)):
        t = k * DT
        for (t0, F, d_) in pushes:
            if abs(t - t0) < DT / 2:
                plant.set_push(F, d_, t)
        q = plant.q + rng.normal(0, 0.005, 7)
        dq = plant.v + rng.normal(0, 0.01, 7)
        tau, d = imp.torque(q, dq, DT)
        plant.step(np.clip(tau + gt(q), -TMAXV, TMAXV), t, DT, substeps=2)
        ts.append(t)
        sig.append(d["sigma"])
        q4s.append(float(plant.q[3]))
        buck.append(d["buckle"])
        dxs.append(float(np.linalg.norm(d["dx"])))   # diag dx is ALREADY mm
        vs.append(float(np.max(np.abs(plant.v))))
        if d["sigma"] < 0.02:
            outcome = f"hard-exit@{t:.2f}s"
            break
    return {"t": ts, "sigma": sig, "q4": q4s, "buckle": buck,
            "dx_mm": dxs, "vmax": vs, "outcome": outcome}


def radial_of(q):
    qf = m._full_q(q)
    pin.forwardKinematics(m.model, m.data, qf)
    pin.updateFramePlacements(m.model, m.data)
    x = np.asarray(m.fk(q)[0], float)
    j3 = np.asarray(
        m.data.oMi[m.model.getJointId("openarm_left_joint3")].translation, float)
    r = x - j3
    return r / np.linalg.norm(r)


out = {"meta": {"dt": DT, "preset": "soft", "seed": 3,
                "gate": 0.05, "hold": 0.055, "blend_hi": 0.085, "crit": 0.02}}

# ---- Scenario A: enable at increasing depths of near-singular -------------
out["A"] = []
for j4v in (0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15):
    q0 = np.array([0.0, -0.5, 0.0, j4v, 0.0, 0.0, 0.0])
    s0 = sigma_of(q0)
    r = simulate(q0, dur=7.0)
    out["A"].append({"j4_start": j4v, "sigma_start": round(s0, 4), **r})
    print(f"A j4={j4v:.2f} σ0={s0:.3f} -> {r['outcome']}, "
          f"σ_end={r['sigma'][-1]:.3f} q4_end={r['q4'][-1]:.2f}")

# ---- Scenario B: safe enable, pushed out hard, release -------------------
out["B"] = []
q_safe = np.array([0.0, -0.5, 0.0, 1.4, 0.0, 0.0, 0.0])   # σ=0.105
rad = radial_of(q_safe)
for Fmag in (10.0, 20.0, 40.0, 60.0):
    r = simulate(q_safe, pushes=[(2.0, Fmag * rad, 2.0)], dur=8.0)
    out["B"].append({"force_N": Fmag, **r})
    print(f"B F={Fmag:.0f}N -> {r['outcome']}, "
          f"σ_min={min(r['sigma']):.3f} σ_end={r['sigma'][-1]:.3f}")

with open("/tmp/imp_singular_sweep.json", "w") as f:
    json.dump(out, f)
print("wrote /tmp/imp_singular_sweep.json")
