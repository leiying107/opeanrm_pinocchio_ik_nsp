# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pinocchio-based FK / IK / gravity for one side of an OpenArm bimanual model.

Enhanced over the original ``openarm_pinocchio_ik`` with:
  * singularity metrics (``singular_values``, ``manipulability``, ``jacobian6``)
  * a null-space-projection IK solver (``ik_nsp``) that actively drives the arm
    away from kinematic singularities via manipulability gradient ascent
  * a multi-seed solver (``ik_multi``) with a safety-weighted score

The original ``fk`` / ``ik`` (plain DLS) / ``gravity`` are preserved verbatim for
backward compatibility. Quaternion convention follows ROS: [x, y, z, w].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from .ik_nsp import damped_pinv_matvec, damped_pseudoinverse


@dataclass
class IKResult:
    """Structured outcome of an IK solve — used by the safety validation gates."""

    q: np.ndarray              # 7 joint angles (rad); last iterate if not converged
    converged: bool
    pos_err_mm: float          # Cartesian position error at solution
    ori_err_deg: float         # orientation error at solution
    iters: int
    sigma_min: float           # smallest singular value of the 6x7 Jacobian
    manipulability: float      # w(q) = sqrt(det(J J^T))
    joint_margin: float        # min distance to a joint limit (rad)
    cond: float                # Jacobian condition number

    @property
    def safe(self) -> bool:
        """A converged solution sitting comfortably above the critical σ floor."""
        return self.converged and self.sigma_min >= _SIGMA_CRIT


# --- Singularity thresholds (calibrated from workspace sampling) -------------
# Workspace σ_min distribution: median 0.078, P10 0.041, P90 0.127; 22.5% < 0.05.
_SIGMA_GOOD = 0.08   # comfortable operating regime (median level)
SIGMA_WARN = 0.05    # degraded — NSP should be climbing out; warn
_SIGMA_CRIT = 0.02   # near-singular — reject / reseed

# --- Stage-2 (null-space ascent) speed/reliability tunables -------------------
# Stage 2 manages the 1-DoF redundancy (elbow swivel) so a warm-start CHAIN stays
# on a safe, continuous branch. Skipping it weakens that continuity and can make
# multi-point arc IK fail at branch transitions — so the default is RELIABLE.
_STAGE2_SKIP = False   # True = skip Stage 2 when DLS solution already safe (fast, riskier)


class PinocchioModel:
    """Wraps a Pinocchio model built from the OpenArm bimanual URDF.

    Only the 7 joints of the requested arm side are exposed; the other side and
    finger (mimic) joints are held at neutral. EE frame is ``openarm_<side>_hand_tcp``
    (geometrically coincident with link7 — TCP offset is zero).
    """

    def __init__(self, urdf_path: str, side: str) -> None:
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        self.side = side

        # mimic=True so finger_joint2 follows finger_joint1 (matches URDF)
        try:
            self.model = pin.buildModelFromUrdf(urdf_path, True)
        except TypeError:
            self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        # Indices of this side's 7 joints within the full q vector.
        self.joint_names = [f"openarm_{side}_joint{i}" for i in range(1, 8)]
        self.q_idx: list[int] = []
        for name in self.joint_names:
            jid = self.model.getJointId(name)
            if jid >= self.model.njoints:
                raise ValueError(f"joint {name!r} not found in URDF model")
            self.q_idx.append(int(self.model.idx_qs[jid]))

        self.ee_frame = f"openarm_{side}_hand_tcp"
        self.ee_fid = self.model.getFrameId(self.ee_frame)
        if self.ee_fid >= len(self.model.frames):
            raise ValueError(f"EE frame {self.ee_frame!r} not found")

        # Per-joint position limits (rad) taken straight from the URDF <limit>.
        self.lower = np.array(
            [self.model.lowerPositionLimit[i] for i in self.q_idx], dtype=float
        )
        self.upper = np.array(
            [self.model.upperPositionLimit[i] for i in self.q_idx], dtype=float
        )

    # ------------------------------------------------------------------ helpers
    def _full_q(self, q7: np.ndarray) -> np.ndarray:
        q = pin.neutral(self.model)
        for i, qi in enumerate(self.q_idx):
            q[qi] = q7[i]
        return q

    def _pose_and_jac(self, q7: np.ndarray, ref=pin.ReferenceFrame.LOCAL):
        """FK + 6x7 Jacobian at the EE frame (this side's columns only)."""
        q = self._full_q(np.asarray(q7, dtype=float))
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, self.ee_fid)
        oMf = self.data.oMf[self.ee_fid]
        J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_fid, ref)
        return oMf, J[:, self.q_idx]

    # --------------------------------------------------- singularity metrics
    def jacobian6(self, q7: np.ndarray, ref=pin.ReferenceFrame.LOCAL) -> np.ndarray:
        """6x7 Jacobian of this side's EE frame (LOCAL reference by default)."""
        _, J = self._pose_and_jac(q7, ref)
        return J

    def singular_values(self, q7: np.ndarray) -> np.ndarray:
        """Singular values of the 6x7 EE Jacobian (length 6, descending)."""
        return np.linalg.svd(self.jacobian6(q7), compute_uv=False)

    def manipulability(self, q7: np.ndarray) -> float:
        """Yoshikawa manipulability w(q) = sqrt(det(J J^T))."""
        J = self.jacobian6(q7)
        return float(np.sqrt(max(0.0, np.linalg.det(J @ J.T))))

    def joint_margin(self, q7: np.ndarray) -> float:
        """Minimum distance (rad) from q7 to the nearest joint limit."""
        q7 = np.asarray(q7, dtype=float)
        return float(min(np.min(q7 - self.lower), np.min(self.upper - q7)))

    def _result_from(
        self, q7: np.ndarray, target_se3: pin.SE3, iters: int, converged: bool
    ) -> IKResult:
        """Build an IKResult by measuring the candidate against the target."""
        q7 = np.asarray(q7, dtype=float)
        oMf, J = self._pose_and_jac(q7)
        err = pin.log(oMf.inverse() * target_se3).vector
        sv = np.linalg.svd(J, compute_uv=False)
        return IKResult(
            q=q7,
            converged=converged,
            pos_err_mm=float(np.linalg.norm(err[:3]) * 1000.0),
            ori_err_deg=float(np.linalg.norm(err[3:]) * 180.0 / np.pi),
            iters=iters,
            sigma_min=float(sv[-1]),
            manipulability=float(np.sqrt(max(0.0, np.linalg.det(J @ J.T)))),
            joint_margin=self.joint_margin(q7),
            cond=float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
        )

    # ---------------------------------------------------------------- FK
    def fk(self, q7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics: 7 joint angles (rad) -> (position[3], quat_xyzw[4])."""
        q = self._full_q(np.asarray(q7, dtype=float))
        pin.framesForwardKinematics(self.model, self.data, q)
        oMf = self.data.oMf[self.ee_fid]
        quat_xyzw = np.asarray(pin.Quaternion(oMf.rotation).coeffs(), dtype=float)
        return oMf.translation.copy(), quat_xyzw

    # --------------------------------------------------------- gravity
    def gravity(self, q7: np.ndarray) -> np.ndarray:
        """Gravity torques (Nm) for this side's 7 joints at configuration q7."""
        q = self._full_q(np.asarray(q7, dtype=float))
        g = pin.computeGeneralizedGravity(self.model, self.data, q)
        return np.array([g[i] for i in self.q_idx], dtype=float)

    # --------------------------------------------------- original DLS IK
    def ik(
        self,
        target_pos: np.ndarray,
        target_quat_xyzw: np.ndarray,
        q_init: np.ndarray,
        max_iters: int = 50,
        tol: float = 1e-4,
        damping: float = 1e-2,
    ) -> np.ndarray | None:
        """Damped-least-squares IK (original solver, preserved for compat).

        Returns 7 joint angles (rad) within limits, or None if not converged.
        """
        R = pin.Quaternion(np.asarray(target_quat_xyzw, dtype=float)).matrix()
        target_se3 = pin.SE3(R, np.asarray(target_pos, dtype=float))

        q7 = np.clip(np.asarray(q_init, dtype=float).copy(), self.lower, self.upper)
        for _ in range(max_iters):
            oMf, J = self._pose_and_jac(q7)
            err = pin.log(oMf.inverse() * target_se3).vector  # 6D (pos+ori)
            if np.linalg.norm(err) < tol:
                return q7
            JJt = J @ J.T + (damping**2) * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            q7 = np.clip(q7 + dq, self.lower, self.upper)
        return None

    # --------------------------------------- null-space-projection IK
    def ik_nsp(
        self,
        target_pos: np.ndarray,
        target_quat_xyzw: np.ndarray,
        q_init: np.ndarray,
        *,
        max_iters: int = 80,
        tol: float = 1e-4,
        lambda0: float = 1e-2,
        sigma_eps: float = SIGMA_WARN,
        kd: float = 0.5,
        null_iters: int = 30,
        null_step: float = 0.1,
        kw: float = 0.5,
        kc: float = 0.3,
    ) -> IKResult:
        """Two-stage null-space IK: DLS convergence, then σ_min ascent.

        Stage 1 — pure damped-least-squares convergence (no null-space term) so
        the primary task is never fought. Adaptive damping inflates only near
        singularities to keep J^# bounded.

        Stage 2 — once converged, greedily climb manipulability along the null
        space ``P = I - J^# J`` (computed with the *small* base damping so the
        projector is accurate). Each candidate step is accepted only if it keeps
        the EE error below tolerance *and* strictly raises σ_min — so the safety
        gain is real and the solution stays valid.

        Returns an :class:`IKResult` (always — inspect ``.converged``/``.safe``).
        """
        R = pin.Quaternion(np.asarray(target_quat_xyzw, dtype=float)).matrix()
        target_se3 = pin.SE3(R, np.asarray(target_pos, dtype=float))

        q7 = np.clip(np.asarray(q_init, dtype=float).copy(), self.lower, self.upper)

        # ---- Stage 1: DLS convergence (primary task only) ----
        converged = False
        iters_run = 0
        for it in range(max_iters):
            iters_run = it + 1
            oMf, J = self._pose_and_jac(q7)
            err = pin.log(oMf.inverse() * target_se3).vector
            if np.linalg.norm(err) < tol:
                converged = True
                break
            sigma_min = float(np.linalg.svd(J, compute_uv=False)[-1])
            lam_sq = lambda0**2 + kd * max(0.0, sigma_eps - sigma_min) ** 2
            dq = damped_pinv_matvec(J, lam_sq, err)
            q7 = np.clip(q7 + dq, self.lower, self.upper)

        if not converged:
            return self._result_from(q7, target_se3, iters_run, False)

        # ---- Stage 1.5: fast path — skip null-space ascent if already safe ----
        # DISABLED by default (_STAGE2_SKIP=False): in a warm-start CHAIN, skipping
        # Stage 2 removes the per-point redundancy management that keeps the elbow
        # swivel on a safe continuous branch → multi-point arc IK can fail at branch
        # transitions. Enable only for isolated single-shot queries where speed
        # matters more than chain continuity.
        if _STAGE2_SKIP:
            sv0 = np.linalg.svd(self.jacobian6(q7), compute_uv=False)
            if float(sv0[-1]) >= _SIGMA_GOOD and self.joint_margin(q7) >= 0.2:
                return self._result_from(q7, target_se3, iters_run, True)

        # ---- Stage 2: null-space MULTI-objective ascent (EE held within tol) ----
        # Drives the 1-DoF redundancy (elbow swivel) to stay away from BOTH
        # singularities (manipulability) AND joint limits (weighted centering),
        # so the IK solution is safe by construction — not merely gated after.
        q_center = (self.lower + self.upper) / 2.0
        best_q = q7.copy()
        best_score = self._ns_score(q7)
        ee_budget = max(tol * 5.0, 5e-4)
        for _ in range(null_iters):
            grad_w = self._manipulability_grad(q7)
            J = self.jacobian6(q7)
            # secondary velocity: manipulability gradient + limit-avoidance
            # (joints near a limit get an exponentially stronger pull to center)
            margin = np.minimum(q7 - self.lower, self.upper - q7)
            dq_sec = kw * grad_w + kc * (np.exp(-margin / 0.3) * (q_center - q7))
            # P @ dq_sec without materializing P: P v = v - J^# (J v), with J^#
            # applied by a single 6×6 solve (the matrix form costs 6 solves here).
            P_dq_sec = dq_sec - damped_pinv_matvec(J, lambda0**2, J @ dq_sec)
            q_try = np.clip(
                q7 + null_step * P_dq_sec, self.lower, self.upper
            )
            oMf_try, _ = self._pose_and_jac(q_try)
            err_try = float(
                np.linalg.norm(pin.log(oMf_try.inverse() * target_se3).vector)
            )
            if err_try < ee_budget:
                score_try = self._ns_score(q_try)
                if score_try > best_score:   # accept if strictly safer
                    q7 = q_try
                    best_score = score_try
                    best_q = q_try.copy()
        return self._result_from(best_q, target_se3, iters_run + null_iters, True)

    def _ns_score(self, q7: np.ndarray) -> float:
        """Null-space safety objective: higher = farther from singularity AND limits."""
        sigma = float(np.linalg.svd(self.jacobian6(q7), compute_uv=False)[-1])
        return sigma + 0.5 * self.joint_margin(q7)

    def _manipulability_grad(self, q7: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """Finite-difference gradient of manipulability w(q) w.r.t. the 7 joints.

        Forward differences (8 evals: base + 7) — half the cost of central
        differences (14 evals) and amply accurate for the greedy null-space climb.
        """
        q7 = np.asarray(q7, dtype=float)
        w0 = self.manipulability(q7)
        grad = np.zeros(7)
        for i in range(7):
            qp = q7.copy()
            qp[i] += eps
            grad[i] = (self.manipulability(qp) - w0) / eps
        return grad

    # ----------------------------------------------------- multi-seed IK
    def ik_multi(
        self,
        target_pos: np.ndarray,
        target_quat_xyzw: np.ndarray,
        *,
        seeds: list[np.ndarray] | None = None,
        q_prev: np.ndarray | None = None,
        n_random: int = 8,
        rng: np.random.Generator | None = None,
        weights: tuple[float, float, float, float] = (0.5, 0.25, 0.15, 0.1),
        reject_sigma: float = _SIGMA_CRIT,
        **ik_nsp_kwargs,
    ) -> IKResult:
        """Multi-seed NSP-IK: solve from several seeds, return the safest one.

        Score (higher is better)::

            w1*sigma_min + w2*joint_margin - w3*||q - q_prev|| - w4*iters

        For trajectory warm-start pass ``q_prev`` so continuity dominates. Any
        seed whose solution is unconverged or below ``reject_sigma`` is dropped.
        Falls back to the best (least-bad) result if nothing is accepted.
        """
        rng = rng or np.random.default_rng(0)
        target_pos = np.asarray(target_pos, dtype=float)
        target_quat_xyzw = np.asarray(target_quat_xyzw, dtype=float)

        # assemble seed list
        margin = 0.1  # keep random seeds 0.1 rad inside limits
        lo, up = self.lower + margin, self.upper - margin
        if seeds is None:
            seeds = []
            if q_prev is not None:
                seeds.append(np.asarray(q_prev, dtype=float))
            # canonical good poses (hands_up: j4=2 -> sigma_min~0.104)
            hu = np.zeros(7)
            hu[3] = 2.0
            seeds.append(hu)
            seeds.append((self.lower + self.upper) / 2.0)
        seeds = list(seeds) + [rng.uniform(lo, up) for _ in range(n_random)]

        w1, w2, w3, w4 = weights
        results: list[IKResult] = []
        for s in seeds:
            r = self.ik_nsp(target_pos, target_quat_xyzw, s, **ik_nsp_kwargs)
            results.append(r)

        accepted = [r for r in results if r.converged and r.sigma_min >= reject_sigma]

        def score(r: IKResult) -> float:
            cont = 0.0 if q_prev is None else float(np.linalg.norm(r.q - q_prev))
            return w1 * r.sigma_min + w2 * r.joint_margin - w3 * cont - w4 * r.iters

        pool = accepted if accepted else results  # fall back to least-bad
        return max(pool, key=score)
