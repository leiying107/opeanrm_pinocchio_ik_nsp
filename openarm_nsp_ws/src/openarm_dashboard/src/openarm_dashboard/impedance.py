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
                       # REFERENCE values at Kx=300 (soft). Scaled by kx/300
                       # at runtime: with ultra (Kx=100) the unscaled 20 Nm/rad
                       # overpowered the weak task spring and held the EE
                       # 20-45mm off anchor after release (hw 2026-08-19:
                       # posture ~1.6 Nm vs task ~0.8 Nm) — "回弹弱/回不到位".
NULL_KQ_REF_KX = 300.0          # the Kx the reference NULL_KQ was tuned for
NULL_KQ_SING = 3.0              # posture-spring multiplier at full flex blend
                               # (restores anti-fold resistance near sigma
                               # gates without touching healthy-pose residual)
K_CFG_RETURN = np.array([5.0, 5.0, 4.0, 8.0, 2.0, 2.0, 2.0])  # Nm/rad
                       # UNPROJECTED config-return spring (NOT passed through
                       # N^T). The null-space projection removes ~99% of the
                       # j4 posture pull (elbow rotation moves the EE, so N
                       # treats it as task-space motion), leaving only the
                       # weak task spring at ultra — total ~0.4-0.7 Nm vs
                       # elbow stiction 1-2 Nm under load: the elbow parked
                       # wherever the user left it (hw 2026-08-20 07:05,
                       # elbow-only test). This small direct spring returns
                       # the WHOLE config; capped per joint below so it can
                       # never dominate the task spring.
CFG_SHALLOW_GAIN = 8.0          # Nm/rad — equals K_CFG_RETURN: shallow-boost was tried (12) and reverted — it re-stiffened the buckle (L5/T8 regressions); silkiness comes from the damper
                       # CFG_SHALLOW_RAD displacement. hw 2026-08-20 08:02:
                       # shallow elbow rotations (0.16-0.18 rad) returned only
                       # 0-46% — the linear 8 Nm/rad gives 1.3-1.5 Nm, right
                       # AT the stiction edge: breaks away, creeps, re-sticks
                       # (Stribeck crawl). 12 Nm/rad at shallow disp breaks
                       # cleanly (deep disp unchanged — cap binds first).
CFG_SHALLOW_RAD = 0.25          # rad — shallow-displacement profile extent
C_CFG_RETURN = np.zeros(7)  # Nm·s/rad — DISABLED after sweep: 2.0 (L5 6→4, L4 OSC up), 1.0 and 0.6 still cost L5; the ZOH-capped damping interacts with the fold sweep. Zero = v8-exact. Re-enable only with per-joint retune + full stress.
                       # return-spring DAMPER (disabled; see above). The idea was:
                       # breakaway overshoots and re-sticks partway (the
                       # crawl above). A velocity term makes the return a
                       # glide. Well inside the ZOH edge (kd < 2I/h).
CFG_RETURN_CAP = 5.0            # Nm — per-joint cap on the return spring
                       # (keeps it a stiction-breaker, not a position lock)
NULL_DQ = np.array([1.5, 1.5, 1.2, 1.0, 0.3, 0.3, 0.3])       # posture damper
M_HAT, I_HAT = 2.0, 0.05        # effective mass (kg) / inertia (kg m^2) for
                       # damping. I_HAT 0.05->0.15 (2026-08-19 hw session 12:09):
                       # wrist j6/j7 flapped ±0.1-0.4 rad at ~1 Hz in ALL presets
                       # — the orientation spring Kr=8 through J^T gives j7 only
                       # ~0.2 Nm·s/rad of rotational damping vs 0.36 needed for
                       # critical (wrist inertia 0.004 kg·m²) => ζ≈0.55. The
                       # commanded torque itself reversed sign each cycle (the
                       # wrist tracked it — underdamped spring, NOT friction
                       # stick-slip). I_HAT 0.15 puts ζ≈0.95 by construction.
                       # Sim friction sweep at 0.1-0.2: harmless (zc=0).
DX_MAX, DTH_MAX = 0.10, 0.5     # error clamps (m / rad)
QDOT_SOFT = 3.0                 # rad/s — attenuate the SPRING terms beyond this
DQ_LPF_HZ = 10.0                # velocity low-pass cutoff (Hz) — suppresses
                                # DM velocity-feedback noise before it reaches
                                # the damping terms (noise*kd = torque buzz).
                                # 15 Hz still passed the noise that produced a
                                # 22 Hz torque loop on hw (band energy in the
                                # command rivaled the signal); 10 Hz kills it
LAMBDA_SQ = 1e-4                # null-space projector damping (0.01^2)
# --- singularity guarding (v5: direction-aware) ----------------------------
# Singularities are DIRECTIONAL: at full extension only the radial direction
# (shoulder->hand axis) is lost — the other 5 task directions stay perfectly
# controllable (verified: |u_min . radial| = 0.999..1.000 across the sweep;
# 95% of random poses with sigma<0.06 are radial-aligned). v4 treated sigma
# as a scalar and blended the whole posture target toward Q_POST_SAFE, which
# polluted the entire null space and fought the task spring (2026-08-18
# 15 Hz hunting incident). v5 instead:
#   1. weights the TASK spring/damper per direction (W = U diag(w) U^T,
#      w_i = sigma_i^2/(sigma_i^2 + SIG_DIR^2)) — the singular direction fades
#      out smoothly, the other 5 stay at full stiffness;
#   2. pushes q along the FOLD direction v_dir = Vt[5] (the joint direction
#      of the smallest singular value — NOT Vt[6], which is the exact null
#      space/elbow swivel and orthogonal to it). Moving along v_dir flexes the
#      elbow with shoulder compensation while the hand stays ~fixed (|J v_dir|
#      = sigma6), so the task spring is not provoked. A soft stop: pulling the
#      arm toward extension meets a growing flexion pull (soft end-stop), and
#      an INWARD axial push buckles the arm into flexion like a human arm.
SIG_WARN = 0.05   # entry gate: refuse to ANCHOR below this (controller-side)
SIG_CRIT = 0.02   # hard exit: fall back to motor-side PD hold (last resort)
SIG_BLEND_HI = 0.085  # flexion control engages smoothly from here down.
                      # 0.12 was TOO WIDE: at sigma 0.07 (workspace median!)
                      # the fold law fought the task spring at half authority
                      # and held a 97 mm steady offset (stress L1 DRIFT cluster)
SIG_HOLD = 0.055      # fold-control sigma target — just above the entry gate.
                      # 0.065 overreached: poses that equilibrate at 0.06
                      # (perfectly safe) kept a permanent flex pull and a
                      # 97 mm EE offset (stalemate vs the task spring)
SIG_DEAD = 0.012      # sigma deadband around SIG_HOLD: within it the position
                      # term is zero (near-target = satisfied) — without this
                      # a pose stuck at σ=0.054 held a permanent 97 mm offset
BLEND_SLEW = 0.04     # max blend change per tick (full swing >= 0.25 s @250Hz)
SIG_DIR = 0.05        # direction weight sigma scale (w=0.5 at this sigma)
K_SIG = 2.5           # 1/s — sigma-rate reference gain (sigma_dot_ref law)
C_SIG = 30.0          # Nm·s — fold-rate torque gain (velocity-resolved law;
                      # swept in sim: 3 cannot hold a 30 N outward pull, 30
                      # floors σ at ~0.05 and returns — no buzz at any gain
                      # because the output requires an active σ-rate error)
K_PUSH_BOOST = 3.0    # multiplier on K_SIG while an inward push is detected
VFOLD_LPF_HZ = 5.0    # fold-rate LPF — the 15 Hz dq filter still passes
                      # noise that resonated the wrist at ~70 Hz in sim
V_DIR_LPF_HZ = 15.0   # low-pass on v_dir (SVD sign flips with wrist motion)
# elbow-flexion sign reference: j4 is NOT mirrored between arms (FK-verified)
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
# is clipped per joint to KAPPA*I_i/dt (with the floor below), keeping the loop
# gain far below the edge by construction. Springs are NOT capped (they set
# equilibrium, not rate).
KAPPA = 0.2
DAMP_CAP_FLOOR = 0.20  # Nm — wrist damping-torque cap floor
                       # alone gives the wrist only 0.15 Nm (0.2*0.003/0.004),
                       # but the ~1 Hz wrist limit cycle measured on hw
                       # (2026-08-19 12:09, j7 ±0.1-0.4 rad, ALL presets)
                       # needed up to 0.18 Nm to arrest — the cap made the
                       # oscillation UNDAMPABLE. Stability margin: a 0.35 Nm
                       # cap at v=1 rad/s implies kd=0.35 << ZOH edge 2I/h=1.5.
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
        self.zeta = 1.4   # default 1.4: with directional mass the wrist gets
                          # real critical damping; 1.2 left it ~ζ0.6 (hw 1Hz LC)
        self.leak = 0.0
        self._age = 0.0               # seconds since ON — drives the ramp-in
        self.x_des: np.ndarray | None = None
        self.R_des: np.ndarray | None = None
        self.q_post: np.ndarray | None = None
        self._dq_f = np.zeros(7)             # low-passed joint velocity
        self._inertia = np.full(7, 0.01)     # joint inertia diag (set at start())
        self.sigma = float("inf")            # latest sigma_min (UI/diag)
        self._blend = 0.0                    # flexion-pull blend (slew-limited)
        self._v_dir = np.zeros(7)            # filtered fold direction (Vt[5])
        self._sigma_prev = float("inf")      # previous sigma (rate estimate)
        self._dsig_f = 0.0                   # LPF'd sigma rate
        self.push_boost = 0.0                # seconds of inward-push boost left
        self._Mbb = None                     # anchor 7x7 inertia (set in start)

    # ------------------------------------------------------------- params
    def set_params(self, preset=None, kx=None, zeta=None, leak=None) -> None:
        if preset in PRESETS:
            self.preset = preset
            self.kx, self.kr = PRESETS[preset]
        if kx is not None:
            self.kx = float(np.clip(float(kx), 100.0, 2000.0))
        if zeta is not None:
            # HARD CAP 1.5: with the directional (operational-space mass)
            # damping, ζ>1.5 destabilizes deterministically across seeds
            # (vend 7.2 vs 0.12 at 1.4, sim sweep 2026-08-19) — the heavy-
            # direction damping through Jᵀ fights the per-joint ZOH cap.
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
        self._v_dir[:] = 0.0
        self._sigma_prev = float("inf")
        self._dsig_f = 0.0
        self.push_boost = 0.0
        # joint-space inertia diagonal at the anchor (for the damping cap);
        # crba returns the upper triangle only
        M = self._pin.crba(self.m.model, self.m.data, self.m._full_q(q)).copy()
        Mb = M[np.ix_(self.m.q_idx, self.m.q_idx)]
        Mb = np.triu(Mb) + np.triu(Mb, 1).T
        self._inertia = np.maximum(np.diag(Mb), 0.002)
        # FULL 7x7 arm block (inverted each tick for the operational-space
        # mass used by the directional Cartesian damping) — fresh from crba
        self._Mbb = Mb.copy()

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
        # SVD WITH vectors: u_min/sigma/v_dir drive the direction-aware guard.
        # try/except fallback: with the hard-stop plant the arm can transiently
        # reach degenerate states where LAPACK fails to converge — fall back to
        # the values-only SVD for that tick (guard still gets sigma).
        try:
            U, sv, Vt = np.linalg.svd(J)
        except np.linalg.LinAlgError:
            U = np.eye(6)
            sv = np.linalg.svd(J, compute_uv=False)
            Vt = np.eye(7)
        self.sigma = float(sv[-1])
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

        # --- direction weighting W = U diag(w) U^T, w_i = s_i^2/(s_i^2+SIG_DIR^2):
        # the singular (radial) direction's spring/damper fades smoothly to 0
        # as sigma falls; the 5 controllable directions stay at full stiffness.
        # J rows are [translation(3); rotation(3)] in the same world-aligned
        # frame, so the 6x6 W mixes both blocks consistently.
        w = sv ** 2 / (sv ** 2 + SIG_DIR ** 2)
        W = (U * w) @ U.T                       # 6x6, symmetric
        err = np.concatenate([dx, dth])         # 6-vector task error
        err_w = W @ err
        dx_w, dth_w = err_w[:3], err_w[3:]

        vx = J[:3] @ self._dq_f
        wv = J[3:] @ self._dq_f
        # Cartesian apparent mass from the LIVE operational-space inertia
        # (J M⁻¹ Jᵀ)⁻¹ diagonal: the wrist directions carry only ~0.05-0.3 kg,
        # far below the old flat M_HAT=2 kg — a flat mass left the wrist at
        # effective ζ≈0.3 (hw: sustained 1 Hz limit cycle, 2026-08-19).
        # Eigenvalues of the 3x3 block, clamped to a sane band.
        Minv = np.linalg.inv(self._Mbb) if self._Mbb is not None else None
        if Minv is not None:
            lam = np.linalg.eigvalsh(J[:3] @ Minv @ J[:3].T)
            m_dir = np.clip(1.0 / np.maximum(lam, 1.0 / 50.0), 0.2, 20.0)
        else:
            m_dir = np.full(3, M_HAT)
        Dr = 2.0 * self.zeta * np.sqrt(self.kr * I_HAT)
        # soft-stop: attenuate only the SPRING terms beyond QDOT_SOFT — the
        # damping terms must stay full-strength (attenuating them too creates
        # a limit cycle: fast motion weakens the very damping that stops it)
        vmax = float(np.max(np.abs(self._dq_f)))
        s = min(1.0, QDOT_SOFT / vmax) if vmax > QDOT_SOFT else 1.0
        kx_e, kr_e = ramp * self.kx, ramp * self.kr   # effective (ramped) stiffness
        tau_spring = (J[:3].T @ (s * kx_e * dx_w)
                      + J[3:].T @ (s * kr_e * dth_w))
        vel = np.concatenate([vx, wv])
        vel_w = W @ vel
        # Translational damping in the SVD principal frame: U3ᵀ maps world
        # velocity into the 3 principal directions, each damped by its own
        # critical gain (from the directional mass), then mapped back. This
        # damps the light wrist axes properly without over-damping the heavy
        # carry directions — a single scalar Dx could only do one or the other.
        U3 = U[:3, :3]
        v_p = U3.T @ vel_w[:3]
        d_p = 2.0 * self.zeta * np.sqrt(self.kx * m_dir) * v_p
        tau_damp = J[:3].T @ (-U3 @ d_p)
        tau_damp = tau_damp + J[3:].T @ (-Dr * vel_w[3:])

        # --- flexion control along the fold direction v_dir = Vt[5] ----------
        # (smallest-singular-value joint direction; Vt[6] is the swivel null
        # space and orthogonal). LPF'd with sign continuity — SVD flips signs
        # discontinuously, and a flipping direction would buzz the arm.
        v_raw = Vt[5]
        if np.linalg.norm(self._v_dir) < 1e-6:
            self._v_dir = v_raw.copy()
        else:
            if float(v_raw @ self._v_dir) < 0.0:
                v_raw = -v_raw                      # sign-align first
            a_v = dt / (dt + 1.0 / (2.0 * np.pi * V_DIR_LPF_HZ)) if dt > 0 else 1.0
            self._v_dir += a_v * (v_raw - self._v_dir)
            self._v_dir /= np.linalg.norm(self._v_dir)
        v_f = self._v_dir
        # flexion sign: toward the safe (elbow-flexed) posture along v_f
        sgn = float(np.sign(v_f @ (Q_POST_SAFE - q))) or 1.0
        b_req = float(np.clip(
            (SIG_BLEND_HI - self.sigma) / (SIG_BLEND_HI - SIG_CRIT), 0.0, 1.0))
        self._blend += float(np.clip(b_req - self._blend, -BLEND_SLEW, BLEND_SLEW))
        self.push_boost = max(0.0, self.push_boost - dt)

        # VELOCITY-RESOLVED 1-D law on the fold coordinate (stress-harness
        # lessons: position/force-based pulls either buzzed the wrist (v_f has
        # ~0.4 weight on the 0.003 kg m^2 j7 → open-loop force = 70 Hz buzz)
        # or whipped the fold (strong saturated force + lagged damper = limit
        # cycle). This law outputs torque ∝ RATE ERROR ONLY:
        #   sigma_dot_ref = -K_SIG * (sigma - SIG_HOLD)      (pull sigma up to
        #                                                     SIG_HOLD, no action
        #                                                     while above)
        #   dsigma/dt  ≈ dσ/dq·q̇  — measured from the LPF'd sigma difference
        #   tau_flex   = v_f·sgn·C_SIG·(sigma_dot_ref - dsigma/dt)
        # While sigma > SIG_HOLD the reference is <= 0 and only DAMPS fold
        # motion; crossing below it commands a bounded fold-OUT rate. Being
        # purely rate-based it cannot excite a position resonance, and its
        # magnitude is bounded by K_SIG·band by construction.
        a_v2 = dt / (dt + 1.0 / (2.0 * np.pi * VFOLD_LPF_HZ)) if dt > 0 else 1.0
        if np.isfinite(self._sigma_prev):
            dsig = (self.sigma - self._sigma_prev) / max(dt, 1e-4)
            self._dsig_f += a_v2 * (dsig - self._dsig_f)
        self._sigma_prev = self.sigma
        # deadband: a pose whose spring equilibrium sits just below SIG_HOLD
        # (measured: σ=0.054 vs target 0.055, q4 unable to rise) would keep a
        # permanent flex pull and a ~97 mm EE offset — treat near-target as ON
        sig_err = min(0.0, self.sigma - SIG_HOLD)    # <=0 below the hold point
        if sig_err > -SIG_DEAD:
            sig_err = 0.0
        boost = (1.0 + (K_PUSH_BOOST - 1.0) * min(1.0, self.push_boost / 0.5)
                 if self.push_boost > 0.0 else 1.0)
        sig_dot_ref = -K_SIG * sig_err * boost       # >0: climb sigma back up
        # authority handback, DIRECTION-AWARE: gate only when the flex itself
        # is dragging the EE away from the anchor. NOTE the sign: dx points
        # TOWARD the anchor (spring pull-back), so the flex being the DRIVER
        # means its EE motion (ee_fold) points AWAY from the anchor, i.e.
        # OPPOSITE dx (align < 0). An external outward pull stretches the
        # spring the same direction the fold pushes back (align > 0) — full
        # authority there. Gating on raw |dx| alone disarmed the soft stop
        # exactly under load (stress: outward pull dove to sigma 0.021).
        ee_fold = (J[:3] @ v_f) * sgn                # EE velocity per unit fold rate
        align = float(ee_fold @ dx) / max(nx * np.linalg.norm(ee_fold), 1e-9)
        drag_gate = 1.0 if align >= 0.0 else max(0.0, 1.0 + align)
        tau_flex = (v_f * sgn * C_SIG * (sig_dot_ref - self._dsig_f)
                    * self._blend * drag_gate)
        # NOTE: a kx-scaled c_sig (sqrt(300/kx), to restore soft-stop margin
        # after the posture-scaling fix) was tried and REVERTED — at near-
        # gate poses it pushed the loop onto a chaotic boundary (same
        # scenario/noise flipped between 0.36 / 2.2 / 70 rad/s across
        # runs). Flat C_SIG stays; the sigma floor under 40 N (0.033) never
        # crossed SIG_CRIT standalone and the hw v-trip-wire covers the rest.

        # null-space posture hold (keeps the elbow swivel from drifting).
        Jpinv = damped_pseudoinverse(J, LAMBDA_SQ)
        N = np.eye(7) - Jpinv @ J
        kq_eff = NULL_KQ * (self.kx / NULL_KQ_REF_KX)   # keep posture spring
        # proportional to the task spring so the anchor wins the post-release
        # equilibrium (hw: unscaled, ultra held 25mm residual; scaled → 11mm,
        # soft → 1mm). Two alternatives measured WORSE and rejected: posture
        # error saturation (16mm — caps the pullback that returns q) and
        # task-first yield (14mm — leaves the task spring alone to drag the
        # whole null space at Kx=100). Scaling keeps BOTH springs pulling.
        # SINGULARITY FLOOR: restore strength near the fold — the scaled-down
        # posture spring let lateral pushes cross SIG_CRIT (stress: L1 SIGEXIT
        # 32→104, all side-pushes at 20-40 N from j4 0.6-1.6). The floor only
        # engages as the flex blend rises, so the residual fix stays intact at
        # healthy poses.
        kq_eff = kq_eff * (1.0 + (NULL_KQ_SING - 1.0) * self._blend)
        tau_post = N.T @ (s * ramp * kq_eff * (self.q_post - q)
                          - NULL_DQ * self._dq_f)
        # RETURN EXEMPTION (hw 2026-08-19 12:09): the old unconditional
        # v_f-component removal (tau_post -= (tau_post·v_f)v_f·blend) let an
        # elbow push strand j4: 30% of the posture pullback was subtracted
        # along v_f AND the fold damping opposed the return, while j1 kept
        # its full spring — the arm returned via j1 only (measured: tauc4
        # +2.18 Nm net toward home, dq4 ≈ 0 for 6 s). Now the yield applies
        # ONLY while the flex pull is actually pulling OUT (sig_err < 0);
        # when sigma is above the hold target the fold law outputs pure
        # damping and must not cannibalize the posture return.
        if sig_err < 0.0:
            tau_post -= (float(tau_post @ v_f) * v_f) * self._blend
        # UNPROJECTED config-return spring: breaks joint stiction that the
        # null-space-projected posture spring cannot (see K_CFG_RETURN note).
        # FULL authority while the EE is settled near anchor (<30 mm) — that
        # is the parked-elbow state; fades 30→70 mm so the task spring owns
        # large displacements (a linear-from-zero gate halved the authority
        # exactly at the parked |dx|~35-45 mm and failed to break stiction).
        # full authority below 30 mm, fading to zero at 70 mm. (45 mm was
        # tried for the lingering-elbow feel but re-introduced end-of-scenario
        # return creep in stress L5 — reverted; the gated damper below is the
        # silkiness lever, not the band)
        cfg_gate = float(np.clip(1.0 - (nx - 0.03) / 0.04, 0.0, 1.0))
        # YIELD to interaction: while an inward push is detected, the fold
        # law is commanding, OR the user is actively MOVING the arm along
        # the fold direction (measured via the filtered fold-rate — works
        # in sim too, unlike the torque-residual detector), the return
        # spring steps aside; it re-engages as motion settles. Without this
        # the return spring fought the buckle feature (regression T8).
        if self.push_boost > 0.0 or sig_err < 0.0:
            cfg_gate = 0.0
        cfg_gate *= float(np.clip(1.0 - np.abs(self._dsig_f) / 0.03, 0.1, 1.0))
        # progressive profile: FLAT boosted gain inside the shallow band (a
        # linear-from-zero ramp put 8.16 Nm/rad at 0.17 rad — no better than
        # the base 8), base gain beyond
        qerr = self.q_post - q
        k_cfg = np.where(np.abs(qerr) < CFG_SHALLOW_RAD,
                         np.maximum(K_CFG_RETURN, CFG_SHALLOW_GAIN),
                         K_CFG_RETURN)
        # damper joins the damping bucket (subject to the same ZOH cap) so the
        # return glides instead of overshoot-and-restick. Shares the spring's
        # INTERACTION gates (not just cfg_gate): ungated it fought the inward
        # buckle (stress v9: L5 48→32 — the 2 Nm·s/rad j4 damper opposed the
        # fold even while the spring had yielded).
        if self.push_boost > 0.0 or sig_err < 0.0:
            cfg_damp_gate = 0.0
        else:
            cfg_damp_gate = cfg_gate * float(
                np.clip(1.0 - np.abs(self._dsig_f) / 0.03, 0.1, 1.0))
        tau_damp = tau_damp - cfg_damp_gate * C_CFG_RETURN * self._dq_f
        tau_cfg = np.clip(k_cfg * qerr * s * ramp,
                          -CFG_RETURN_CAP, CFG_RETURN_CAP) * cfg_gate
        tau_spring = tau_spring + tau_post + tau_flex + tau_cfg

        # commanded joint damping floor + per-joint sampled-stability cap:
        # |tau_damp_i| <= KAPPA*I_i/dt keeps zero-order-hold damping far below
        # the instability edge kd*h/I = 2 (the wrist inertia is ~0.003 kg m^2)
        tau_damp = tau_damp - KD_J * self._dq_f
        lim = np.maximum(KAPPA * self._inertia / max(dt, 1e-4), DAMP_CAP_FLOOR)
        tau = tau_spring + np.clip(tau_damp, -lim, lim)

        tau = np.clip(tau, -TMAX, TMAX)
        diag = {
            "dx": dx_w * 1000.0,           # mm, world (weighted: singular dir fades)
            "dth": np.degrees(dth_w),      # deg (weighted)
            "fest": kx_e * dx_w,           # N — contact-force estimate (weighted)
            "qdot": vmax,                  # rad/s (filtered)
            "ramp": round(ramp, 2),        # 0->1 stiffness ramp-in progress
            "sigma": round(self.sigma, 3), # sigma_min (singularity guard)
            "buckle": round(self._blend, 2),   # flexion-pull blend 0-1
            "frozen": self.sigma < SIG_DIR,    # singular dir under-actuated
            "vdir": v_f,                       # fold direction (debug)
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
        # URDF joint limits with a stiff hard-stop model — the earlier plant
        # had NO limits, so a hard inward push folded q4 past 2.44 rad into
        # nonphysical territory and the stress harness flagged phantom
        # singularity exits. Real hardware stops at the joint limit.
        self.q_lo = np.array([self.model.lowerPositionLimit[
            self.model.idx_qs[j]] for j in jid], float)
        self.q_hi = np.array([self.model.upperPositionLimit[
            self.model.idx_qs[j]] for j in jid], float)
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
            # URDF hard stops as a PROJECTION (not stiff springs — a stiff
            # explicit spring chatters at any sane k): position clamps to the
            # limit, outward velocity is absorbed by restitution-free impact
            # (v→0 leaving the stop; inward motion still allowed to separate)
            for i in range(7):
                if self.q[i] <= self.q_lo[i]:
                    self.q[i] = self.q_lo[i]
                    if self.v[i] < 0.0:
                        self.v[i] = 0.0
                elif self.q[i] >= self.q_hi[i]:
                    self.q[i] = self.q_hi[i]
                    if self.v[i] > 0.0:
                        self.v[i] = 0.0
            if not np.all(np.isfinite(self.q)):
                raise FloatingPointError("sim plant state diverged")
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
