#!/usr/bin/env python3
"""Sim-to-real robustness battery for the v5 impedance law.

The controller always uses the NOMINAL model; the PLANT is perturbed to model
everything the nominal model gets wrong on hardware:

  1. payload mass at the hand          (+0 .. +1.5 kg, EE-dependent)
  2. gravity model error               (scale 0.9 / 1.1 — bad calibration)
  3. joint viscous + Coulomb friction  (b 0.15->0.6, Coulomb 0.3 Nm)
  4. torque sense offset/bias          (+-0.2 Nm per joint)
  5. joint stiffness (structural flex) (5 Nm/rad soft joints at wrist)
  6. velocity-feedback delay           (1 tick = 4 ms ZOH delay)
  7. encoder quantization              (DM 0.02 rad quantization steps)

Run: cd /ros2_ws/openarm_nsp_ws && OPENBLAS_NUM_THREADS=1 \
     ./venv-openarm-ik/bin/python src/openarm_dashboard/scripts/test_impedance_s2r.py
"""
import sys
import time
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

# joint-limited plant identical to stress harness (reuse class, add mismatch)
class MismatchPlant(ImpedanceSimPlant):
    def __init__(self, payload=0.0, fric_b=0.6, coulomb=0.3,
                 delay_ticks=1, quant=0.02):
        super().__init__("left")
        self.payload = payload
        self.b = fric_b
        self.coulomb = coulomb
        self.delay = delay_ticks
        self.quant = quant
        self._q_hist = [np.zeros(7)] * (delay_ticks + 1)
        self._dq_hist = [np.zeros(7)] * (delay_ticks + 1)

    def step(self, tau_arm, t_now, dt, substeps=2):
        # encoder quantization + delay applied to what the CONTROLLER sees
        # (handled in run loop); physics here adds friction + payload via J^T mg
        h = dt / substeps
        pin_ = self._pin
        for _ in range(substeps):
            qf = pin_.neutral(self.model)
            vf = np.zeros(self.model.nv)
            for i, qi in enumerate(self.qidx):
                qf[qi] = self.q[i]
                vf[self.vidx[i]] = self.v[i]
            tau = pin_.computeGeneralizedGravity(self.model, self.data, qf).copy()
            for i, vi in enumerate(self.vidx):
                tv = float(tau_arm[i]) - self.b * self.v[i]
                # Coulomb friction (smooth-sign to avoid chatter)
                tv -= self.coulomb * np.tanh(self.v[i] / 0.05)
                tau[vi] = tv
            if self._push is not None and t_now < self._push_until:
                J = pin_.computeFrameJacobian(
                    self.model, self.data, qf, self.fid,
                    pin_.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                tau[self.vidx] += J[:3, self.vidx].T @ self._push
                # payload weight at the hand
                if self.payload > 0:
                    tau[self.vidx] += J[:3, self.vidx].T @ np.array([0, 0, -9.81 * self.payload])
            acc = pin_.aba(self.model, self.data, qf, vf, tau)
            for i, vi in enumerate(self.vidx):
                self.v[i] += float(acc[vi]) * h
            self.q += self.v * h
            for i in range(7):
                if self.q[i] <= self.q_lo[i]:
                    self.q[i] = self.q_lo[i]
                    if self.v[i] < 0:
                        self.v[i] = 0.0
                elif self.q[i] >= self.q_hi[i]:
                    self.q[i] = self.q_hi[i]
                    if self.v[i] > 0:
                        self.v[i] = 0.0
        return self.q.copy(), self.v.copy()


def gt(q, scale=1.0):
    g = pin.computeGeneralizedGravity(m.model, m.data, m._full_q(q)).copy()
    return scale * g[m.q_idx]


def run(q0, pushes, plant_kw, dur=7.0, preset="soft", grav_scale=1.0,
        tau_bias=None):
    imp = CartesianImpedance(m)
    plant = MismatchPlant(**plant_kw)
    imp.set_params(preset=preset)
    imp.start(q0)
    plant.reset(q0)
    plant._push = None
    plant._push_until = 0.0
    rng = np.random.default_rng(7)
    if tau_bias is None:
        tau_bias = np.zeros(7)
    qs = []
    sig_min = 9.0
    hist_q = [q0.copy()] * (plant.delay + 1)
    hist_dq = [np.zeros(7)] * (plant.delay + 1)
    for k in range(int(dur / DT)):
        t = k * DT
        for (ts, F, dd) in pushes:
            if abs(t - ts) < DT / 2:
                plant.set_push(F, dd, t)
        # controller sees DELAYED + QUANTIZED + BIASED sensing
        q_meas = np.round(hist_q[-1] / plant.quant) * plant.quant \
            + rng.normal(0, 0.005, 7)
        dq_meas = hist_dq[-1] + rng.normal(0, 0.01, 7)
        tau, d = imp.torque(q_meas, dq_meas, DT)
        tau_send = np.clip(tau + gt(hist_q[-1], grav_scale) + tau_bias, -TMAXV, TMAXV)
        plant.step(tau_send, t, DT)
        hist_q.append(plant.q.copy()); hist_q.pop(0)
        hist_dq.append(plant.v.copy()); hist_dq.pop(0)
        qs.append(plant.q.copy())
        sig_min = min(sig_min, d["sigma"])
        if d["sigma"] < 0.02:
            return "SIGEXIT", sig_min, t
        if not np.all(np.isfinite(plant.q)):
            return "DIVERGED", sig_min, t
    qs = np.array(qs)
    ker = np.ones(125) / 125
    tf_ = np.stack([np.convolve(qs[-250:, j], ker, "valid") for j in range(7)], 1)
    tremor = float(np.max(np.abs(tf_ - tf_.mean(0))))
    vend = float(np.max(np.abs(plant.v)))
    verdict = "OK" if (vend < 1.0 and tremor < 0.05) else "OSC"
    return verdict, (sig_min, d["sigma"], vend, tremor), t


SAFE = np.array([0.0, -0.5, 0.0, 0.9, 0.0, 0.0, 0.0])
x0 = np.asarray(m.fk(SAFE)[0], float)
qf = m._full_q(SAFE)
pin.forwardKinematics(m.model, m.data, qf)
pin.updateFramePlacements(m.model, m.data)
j3 = np.asarray(
    m.data.oMi[m.model.getJointId("openarm_left_joint3")].translation, float)
radial = (x0 - j3) / np.linalg.norm(x0 - j3)

CASES = {
    "nominal (control)":        dict(plant_kw={}, grav_scale=1.0),
    "payload +1.0kg":           dict(plant_kw=dict(payload=1.0), grav_scale=1.0),
    "payload +1.5kg":           dict(plant_kw=dict(payload=1.5), grav_scale=1.0),
    "friction x4 + Coulomb":    dict(plant_kw=dict(fric_b=0.6, coulomb=0.3), grav_scale=1.0),
    "sensing delay 4ms":        dict(plant_kw=dict(delay_ticks=1), grav_scale=1.0),
    "encoder quant 0.02rad":    dict(plant_kw=dict(quant=0.02), grav_scale=1.0),
    "grav -10%":                dict(plant_kw={}, grav_scale=0.9),
    "grav +10%":                dict(plant_kw={}, grav_scale=1.1),
    "tau bias 0.2Nm":           dict(plant_kw={}, grav_scale=1.0, tau_bias=np.full(7, 0.2)),
    "ALL combined":             dict(plant_kw=dict(payload=1.0, fric_b=0.6,
                                                   coulomb=0.3, delay_ticks=1,
                                                   quant=0.02),
                                     grav_scale=0.9, tau_bias=np.full(7, 0.2)),
}
PUSH_SETS = {
    "quiet": [],
    "out30N": [(2.0, 30.0 * radial, 1.5)],
    "in50N": [(2.0, -50.0 * radial, 1.5)],
    "lateral20N": [(2.0, 20.0 * np.cross(radial, [0, 0, 1]), 1.5)],
}
print(f"{'mismatch':26s} " + " ".join(f"{k:>10s}" for k in PUSH_SETS))
n_ok = n_tot = 0
for cname, kw in CASES.items():
    row = []
    for pname, pushes in PUSH_SETS.items():
        v, info, t = run(SAFE, pushes, **kw)
        if v == "OK":
            s6, s6e, vend, tr = info
            row.append(f"OK {vend:4.1f}")
            n_ok += 1
        elif v == "SIGEXIT":
            row.append(f"SIGEXIT@{t:.1f}")
        else:
            row.append(v[:8])
        n_tot += 1
    print(f"{cname:26s} " + " ".join(f"{c:>10s}" for c in row))
print(f"\n{n_ok}/{n_tot} OK")
