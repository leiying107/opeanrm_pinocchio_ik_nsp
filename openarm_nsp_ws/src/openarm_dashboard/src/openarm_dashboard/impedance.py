# Copyright 2026 Enactic, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cartesian 6D impedance control for OpenArm (spring-damper at the EE).

    tau = J_v^T (Kx*dx - Dx*vx) + J_w^T (Kr*dth - Dr*w)      # task space
        + N^T (Kq*(q_post - q) - Dq*qdot)                    # null-space posture

* NO gravity term here — G(q) is added separately via ``_grav_tau()`` /
  GravityComp (real-hw validated). Adding it here would double-count.
* J = PinocchioModel.jacobian6(q, LOCAL_WORLD_ALIGNED) — world-aligned frame,
  same URDF/EE frame (hand_tcp) as the IK + tracking-error display.
* dx = x_des - x (world); dth = log3(R_des R^T) (world rotation vector).
* vx, w = J qdot — motor velocity feedback, 1st-order low-passed (~30 Hz).
* Damping from the critical ratio zeta: Dx = 2*zeta*sqrt(Kx*m_hat),
  Dr = 2*zeta*sqrt(Kr*I_hat).
* Leak (drag mode): the anchor x_des/R_des slowly follows the actual pose —
  leak=0 frozen spring (push -> rebound), leak>0 compliant drag.
* Safety: |dx| <= 0.10 m, |dth| <= 0.5 rad clamped; tau clamped to +/-TMAX;
  |qdot| beyond QDOT_SOFT attenuates tau (soft-stop).
"""
from __future__ import annotations

import numpy as np

try:  # kinematics lives in the workspace PYTHONPATH (web_panel / hardware_dashboard)
    from openarm_pinocchio_nsp.ik_nsp import damped_pseudoinverse
    _HAVE_NSP = True
except ImportError:  # pragma: no cover - controller guards this module's import
    _HAVE_NSP = False

# (Kx N/m, Kr Nm/rad) — isotropic presets; UI retunes Kx/zeta/leak live.
# "ultra" is the FIRST-CONTACT preset for real hardware: soft enough that
# velocity-feedback noise cannot shake the wrist even before any tuning.
PRESETS = {"ultra": (100.0, 3.0), "soft": (300.0, 8.0),
           "mid": (800.0, 20.0), "stiff": (1500.0, 40.0)}
PRESET_LABELS = {"ultra": "极软", "soft": "软", "mid": "中", "stiff": "硬"}
# per-joint torque clamp (Nm): DM8009, DM8009, DM4340, DM4340, DM4310 x3
TMAX = np.array([54.0, 54.0, 28.0, 28.0, 10.0, 10.0, 10.0])
NULL_KQ = np.array([20.0, 20.0, 20.0, 15.0, 5.0, 5.0, 5.0])   # posture spring
NULL_DQ = np.array([1.5, 1.5, 1.2, 1.0, 0.3, 0.3, 0.3])       # posture damper
M_HAT, I_HAT = 2.0, 0.05        # effective mass (kg) / inertia (kg m^2) for damping
DX_MAX, DTH_MAX = 0.10, 0.5     # error clamps (m / rad)
QDOT_SOFT = 3.0                 # rad/s — attenuate the SPRING terms beyond this
DQ_LPF_HZ = 15.0                # velocity low-pass cutoff (Hz) — suppresses
                                # DM velocity-feedback noise before it reaches
                                # the damping terms (noise*kd = torque buzz)
LAMBDA_SQ = 1e-4                # null-space projector damping (0.01^2)
# --- singularity guarding ------------------------------------------------
# sigma_min thresholds from workspace sampling (kinematics.py): P10=0.041,
# median=0.078. At sigma < SIG_WARN the J^T mapping loses the singular
# direction: the spring stops resisting along it (unbounded drift, silent
# config flips like elbow reversal) and F_est=K*dx under-reports contact.
SIG_WARN = 0.05   # soft escape: bias the null-space posture toward elbow flex
SIG_CRIT = 0.02   # hard exit: fall back to motor-side PD hold
# soft-escape blend band. Blending only below SIG_WARN caused boundary
# HUNTING on real hw: a pose hovering at sigma ~0.052-0.058 (just past the
# entry gate) switched the escape on/off every few ticks -> sustained 15 Hz
# limit cycle (2026-08-18 incident). Widening the band to start at
# SIG_BLEND_HI (with a slew limit on the blend factor) keeps the escape
# authority monotonic in sigma — no cliff to hunt across.
SIG_BLEND_HI = 0.12   # blend starts smoothly here (≈ P90 of workspace sigma)
BLEND_SLEW = 0.02     # max blend change per tick (full swing >= 0.5 s @250Hz)
# elbow-flexion escape direction: j4 is NOT mirrored between arms (FK-verified:
# j4=+0.8 flexes BOTH arms' EE forward +0.155), and sigma climbs monotonically
# with |j4| on both sides (0 at full extension -> 0.132 at j4=2). The escape
# posture moves j4 toward a safe target instead of a fixed offset, so it also
# works from flexed poses (direction picks the nearest safe side).
Q_POST_SAFE = np.array([0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0])  # σ≈0.096
# motor-side MIT kd (dq_des=0). ZERO on real hardware: the real-hw-validated
# pure-torque path (zero-torque + gravity) runs kp=0/kd=0 and is smooth, while
# ANY motor-side kd multiplies the DM motors' quantized velocity feedback into
# torque noise (0.4 * a few rad/s of wrist velocity noise = volts-scale buzz on
# a 0.003 kg m^2 wrist) — the violent shake on first real-hw impedance enable
# traced to this. Do not raise without re-validating on hardware.
IMP_KD = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# commanded joint damping floor (inside the sampled-stability cap below)
KD_J = np.array([0.8, 0.8, 0.6, 0.6, 0.3, 0.3, 0.3])
# sampled-stability cap on the DAMPING torque: zero-order-hold damping is
# unstable when kd*h/I >= 2, and the OpenArm wrist inertia is only ~0.003 kg m^2
# (diag(M) = [.23,.24,.06,.08,.003,.004,.004]) — so every commanded damping term
# is clipped per joint to KAPPA*I_i/dt, keeping the loop gain far below the
# edge by construction. Springs are NOT capped (they set equilibrium, not rate).
KAPPA = 0.2
# stiffness ramp-in after impedance ON: kx/kr grow 0 -> target over RAMP_S so
# the spring engages gently instead of stepping in at full stiffness (a step
# engagement with any anchor error or hand contact jerks the arm).
RAMP_S = 2.0


class CartesianImpedance:
    """Per-arm 6D Cartesian impedance law (stateless in q; holds the anchor)."""

    def __init__(self, model):
        import pinocchio as pin  # numpy<2 via the dashboard venv
        if not _HAVE_NSP:
            raise ImportError("openarm_pinocchio_nsp not importable")
        self._pin = pin
        self.m = model                       # PinocchioModel for this side
        self.preset = "ultra"
        self.kx, self.kr = PRESETS[self.preset]
        self.zeta = 1.2
        self.leak = 0.0
        self._age = 0.0               # seconds since ON — drives the ramp-in
        self.x_des: np.ndarray | None = None
        self.R_des: np.ndarray | None = None
        self.q_post: np.ndarray | None = None
        self._dq_f = np.zeros(7)             # low-passed joint velocity
        self._inertia = np.full(7, 0.01)     # joint inertia diag (set at start())
        self.sigma = float("inf")            # latest sigma_min (UI/diag)
        self._blend = 0.0                    # escape blend factor (slew-limited)

    # ------------------------------------------------------------- params
    def set_params(self, preset=None, kx=None, zeta=None, leak=None) -> None:
        if preset in PRESETS:
            self.preset = preset
            self.kx, self.kr = PRESETS[preset]
        if kx is not None:
            self.kx = float(np.clip(float(kx), 100.0, 2000.0))
        if zeta is not None:
            self.zeta = float(np.clip(float(zeta), 0.5, 1.5))
        if leak is not None:
            self.leak = float(np.clip(float(leak), 0.0, 2.0))

    def start(self, q) -> None:
        """Capture the current pose as the spring anchor (impedance ON)."""
        x, quat = self.m.fk(q)
        self.x_des = np.asarray(x, float).copy()
        self.R_des = self._pin.Quaternion(np.asarray(quat, float)).matrix()
        self.q_post = np.asarray(q, float).copy()
        self._dq_f[:] = 0.0
        self._age = 0.0
        self._blend = 0.0
        # joint-space inertia diagonal at the anchor (for the damping cap);
        # crba returns the upper triangle only
        M = self._pin.crba(self.m.model, self.m.data, self.m._full_q(q)).copy()
        Mb = M[np.ix_(self.m.q_idx, self.m.q_idx)]
        Mb = np.triu(Mb) + np.triu(Mb, 1).T
        self._inertia = np.maximum(np.diag(Mb), 0.002)

    def _leak_update(self, x: np.ndarray, R: np.ndarray, dt: float) -> None:
        """Drag mode: anchor slowly follows the actual pose (leak in 1/s)."""
        if self.leak <= 0 or dt <= 0:
            return
        f = 1.0 - float(np.exp(-self.leak * dt))   # exact per-tick fraction
        self.x_des += f * (x - self.x_des)
        e = self._pin.log3(R @ self.R_des.T)
        if np.linalg.norm(e) > 1e-9:
            self.R_des = self._pin.exp3(f * e) @ self.R_des

    # ------------------------------------------------------------- control
    def torque(self, q, dq, dt: float) -> tuple[np.ndarray, dict]:
        """One control tick -> (tau WITHOUT gravity, diag dict for the UI)."""
        pin = self._pin
        q = np.asarray(q, float)
        dq = np.asarray(dq, float)
        self._age += dt
        ramp = min(1.0, self._age / RAMP_S)          # 0 -> 1 over RAMP_S
        ramp = ramp * ramp * (3.0 - 2.0 * ramp)      # smoothstep: gentle slope
        alpha = dt / (dt + 1.0 / (2.0 * np.pi * DQ_LPF_HZ)) if dt > 0 else 1.0
        self._dq_f += alpha * (dq - self._dq_f)

        # one FK pass yields placement AND Jacobian (jacobian6/fk would each
        # rerun forwardKinematics — 2x the model work per tick)
        oMf, J = self.m._pose_and_jac(q, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        self.sigma = float(np.linalg.svd(J, compute_uv=False)[-1])
        x = np.asarray(oMf.translation, float)
        R = np.asarray(oMf.rotation, float)
        self._leak_update(x, R, dt)

        dx = self.x_des - x
        nx = float(np.linalg.norm(dx))
        if nx > DX_MAX:
            dx *= DX_MAX / nx
        dth = pin.log3(self.R_des @ R.T)
        nt = float(np.linalg.norm(dth))
        if nt > DTH_MAX:
            dth *= DTH_MAX / nt

        vx = J[:3] @ self._dq_f
        w = J[3:] @ self._dq_f
        Dx = 2.0 * self.zeta * np.sqrt(self.kx * M_HAT)
        Dr = 2.0 * self.zeta * np.sqrt(self.kr * I_HAT)
        # soft-stop: attenuate only the SPRING terms beyond QDOT_SOFT — the
        # damping terms must stay full-strength (attenuating them too creates
        # a limit cycle: fast motion weakens the very damping that stops it)
        vmax = float(np.max(np.abs(self._dq_f)))
        s = min(1.0, QDOT_SOFT / vmax) if vmax > QDOT_SOFT else 1.0
        kx_e, kr_e = ramp * self.kx, ramp * self.kr   # effective (ramped) stiffness
        tau_spring = (J[:3].T @ (s * kx_e * dx)
                      + J[3:].T @ (s * kr_e * dth))
        tau_damp = (J[:3].T @ (-Dx * vx)
                    + J[3:].T @ (-Dr * w))

        # null-space posture hold (keeps the elbow swivel from drifting);
        # near a singularity the posture target BLENDS toward the elbow-flexed
        # safe posture. Blend authority is MONOTONIC in sigma with a wide band
        # [SIG_CRIT, SIG_BLEND_HI] plus a per-tick slew limit — a narrow band
        # ending at SIG_WARN hunted on real hw (pose hovering at ~0.055
        # switched the escape on/off every few ticks -> 15 Hz limit cycle).
        # The EE spring keeps holding while the redundancy climbs out.
        Jpinv = damped_pseudoinverse(J, LAMBDA_SQ)
        N = np.eye(7) - Jpinv @ J
        b_req = float(np.clip(
            (SIG_BLEND_HI - self.sigma) / (SIG_BLEND_HI - SIG_CRIT), 0.0, 1.0))
        self._blend += float(np.clip(b_req - self._blend, -BLEND_SLEW, BLEND_SLEW))
        target = (1.0 - self._blend) * self.q_post + self._blend * Q_POST_SAFE
        tau_spring = tau_spring + N.T @ (s * ramp * NULL_KQ * (target - q))
        tau_damp = tau_damp + N.T @ (-NULL_DQ * self._dq_f)

        # commanded joint damping floor + per-joint sampled-stability cap:
        # |tau_damp_i| <= KAPPA*I_i/dt keeps zero-order-hold damping far below
        # the instability edge kd*h/I = 2 (the wrist inertia is ~0.003 kg m^2)
        tau_damp = tau_damp - KD_J * self._dq_f
        lim = KAPPA * self._inertia / max(dt, 1e-4)
        tau = tau_spring + np.clip(tau_damp, -lim, lim)

        tau = np.clip(tau, -TMAX, TMAX)
        diag = {
            "dx": dx * 1000.0,                # mm, world frame (x前 y左 z上)
            "dth": np.degrees(dth),           # deg (R P Y components)
            "fest": kx_e * dx,                # N — contact-force estimate K*dx
            "qdot": vmax,                     # rad/s (filtered)
            "ramp": round(ramp, 2),           # 0->1 stiffness ramp-in progress
            "sigma": round(self.sigma, 3),    # sigma_min (singularity guard)
        }
        return tau, diag


class ImpedanceSimPlant:
    """Dynamics plant for ONE arm on v1_simple.urdf (sim-only verification).

    Every joint defaults to pure gravity hold (other arm stays put); OUR arm's
    joints get tau_arm (the controller's FULL commanded torque — it already
    contains G(q) + impedance) minus viscous friction. Integrated with
    semi-implicit Euler (substeps per tick). External pushes enter as
    tau_ext = J_v^T F_push (world frame) at the hand — lets the spring
    response be tested with no hardware and no browser.
    """

    def __init__(self, side: str):
        import pinocchio as pin
        from .gravity import _resolve_urdf
        path = _resolve_urdf()
        if not path:
            raise FileNotFoundError("v1_simple.urdf not found for sim plant")
        self._pin = pin
        self.model = pin.buildModelFromUrdf(path)
        self.model.gravity.linear = np.array([0.0, 0.0, -9.81])
        self.data = self.model.createData()
        jid = [self.model.getJointId(f"openarm_{side}_joint{k}") for k in range(1, 8)]
        self.qidx = [int(self.model.idx_qs[i]) for i in jid]
        self.vidx = [int(self.model.idx_vs[i]) for i in jid]
        self.fid = self.model.getFrameId(f"openarm_{side}_hand")
        self.q = np.zeros(7)
        self.v = np.zeros(7)
        self.b = 0.15                 # joint viscous friction (Nm s/rad)
        self._push: np.ndarray | None = None
        self._push_until = 0.0        # time.monotonic() deadline

    def reset(self, q7) -> None:
        self.q = np.asarray(q7, float).copy()
        self.v[:] = 0.0

    def set_push(self, force3, dur: float, t_now: float) -> None:
        self._push = np.asarray(force3, float)
        self._push_until = t_now + float(dur)

    def step(self, tau_arm, t_now: float, dt: float, substeps: int = 4):
        pin = self._pin
        h = dt / substeps
        for _ in range(substeps):
            qf = pin.neutral(self.model)
            vf = np.zeros(self.model.nv)
            for i, qi in enumerate(self.qidx):
                qf[qi] = self.q[i]
                vf[self.vidx[i]] = self.v[i]
            # default: pure gravity on every joint; OUR joints OVERWRITE with
            # the commanded torque (which already includes its own G term —
            # adding here as well would double-count gravity)
            tau = pin.computeGeneralizedGravity(self.model, self.data, qf).copy()
            for i, vi in enumerate(self.vidx):
                tau[vi] = float(tau_arm[i]) - self.b * self.v[i]
            if self._push is not None and t_now < self._push_until:
                J = pin.computeFrameJacobian(
                    self.model, self.data, qf, self.fid,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                Jarm = J[:3, self.vidx]
                tau[self.vidx] += Jarm.T @ self._push
            acc = pin.aba(self.model, self.data, qf, vf, tau)
            for i, vi in enumerate(self.vidx):
                self.v[i] += float(acc[vi]) * h
            self.q += self.v * h
        return self.q.copy(), self.v.copy()


if __name__ == "__main__":  # sanity self-check, no hardware
    import time as _time
    import sys
    sys.path.insert(0, "/ros2_ws/openarm_nsp_ws/src/openarm_pinocchio_nsp/src")
    from openarm_pinocchio_nsp.kinematics import PinocchioModel
    from openarm_pinocchio_nsp.urdf_path import resolve_urdf_path

    for side in ("left", "right"):
        m = PinocchioModel(resolve_urdf_path(), side)
        imp = CartesianImpedance(m)
        q0 = np.array([0.0, 0.6, 0.0, 1.1, 0.0, 0.7, 0.0])
        imp.start(q0)
        imp.set_params(preset="soft")
        # push the EE 5 cm along +x via the anchor and check tau reacts
        imp.x_des = imp.x_des + np.array([0.05, 0.0, 0.0])
        tau, diag = imp.torque(q0, np.zeros(7), 0.004)
        print(f"{side}: tau={np.round(tau, 2)}")
        print(f"       dx={np.round(diag['dx'], 1)}mm fest={np.round(diag['fest'], 1)}N")
