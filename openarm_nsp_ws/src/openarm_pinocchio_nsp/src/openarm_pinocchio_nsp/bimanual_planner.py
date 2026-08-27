# Copyright 2026 Enactic, Inc.
# SPDX-License-Identifier: Apache-2.0
"""14-D bimanual B-spline coordinated planning with self-collision avoidance.

Both arms are optimized TOGETHER: control points are 14-vectors (left 7 +
right 7), so the trajectory can exploit time-phasing (one arm yields while
the other crosses) instead of treating the partner as a frozen obstacle.
Collision cost uses the FULL pair set from BimanualCollisionChecker —
cross-arm (L↔R) and arm↔body pairs included, which bspline_planner's
single-arm optimizer never checked.

Cost (SLSQP over M×14 control points):
    w1 · Σ ‖FK_L(q_L(t_i)) − x_L(t_i)‖²  +  ‖FK_R(q_R(t_i)) − x_R(t_i)‖²   tracking
  + w2 · Σ ‖Δ²C‖²                                                        smoothness
  + w3 · Σ viol(t_i)²                                                     collision

Collision penalty per sample: the worst boolean margin violation among the
163 checked pairs, measured at increasing margins (ladder), so the gradient
points away from ANY approaching geometry, not just penetration.
"""

from __future__ import annotations

import time

import numpy as np
import pinocchio as pin
from scipy.interpolate import BSpline
from scipy.optimize import minimize

from .collision import BimanualCollisionChecker
from .urdf_path import resolve_urdf_path


class _BudgetExhausted(Exception):
    """Raised internally when time_budget is exceeded mid-optimization."""


class BimanualBSplinePlanner:
    """Joint-space 14-D B-spline optimizer for coordinated two-arm motion."""

    def __init__(self, urdf_path: str | None = None,
                 checker: BimanualCollisionChecker | None = None):
        self.chk = checker or BimanualCollisionChecker(urdf_path)
        self.model = self.chk.model
        self.data = self.model.createData()
        self.ee_fid = {
            side: self.model.getFrameId(f"openarm_{side}_hand_tcp")
            for side in ("left", "right")
        }

    # ------------------------------------------------------------- FK helpers
    def ee_pos(self, q16: np.ndarray, side: str) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, q16)
        pin.updateFramePlacement(self.model, self.data, self.ee_fid[side])
        return self.data.oMf[self.ee_fid[side]].translation.copy()

    def split(self, q14: np.ndarray):
        """14-vector → (q16, q_left7, q_right7)."""
        ql, qr = q14[:7], q14[7:]
        return self.chk.make_q(left=ql, right=qr), ql, qr

    # ------------------------------------------------------- collision ladder
    def collision_penalty(self, q14: np.ndarray, ladder=(0.03, 0.02, 0.01)) -> float:
        """Piecewise penalty: 0 if clear at the comfort rung, rising per rung.

        Faster than exact distances (~0.4 ms/rung) and differentiable enough
        for SLSQP's finite differences.
        """
        q16, _, _ = self.split(q14)
        pen = 0.0
        for level, m in enumerate(ladder):
            if self.chk.check(q16, m).in_collision:
                pen += (len(ladder) - level) ** 2
        return pen

    # ------------------------------------------------------------------ plan
    def plan(self, way_left, way_right, q0_left, q0_right, duration=4.0,
             n_samples=40, M=10, w1=1.0, w2=1.0, w3=20.0, maxiter=60,
             time_budget: float | None = None, verbose=False):
        """Optimize a 14-D B-spline through per-arm Cartesian waypoints.

        ``time_budget`` (seconds) caps wall-clock: SLSQP checks it between
        iterations and returns the best iterate so far when exceeded — on the
        1.8 GHz ARM board the numeric-gradient cost (~0.5 s/iter at M=10)
        dominates, so callers typically pass 120–300 s.
        """
        """Optimize a 14-D B-spline through per-arm Cartesian waypoints.

        way_left / way_right : (N,3) EE position tracks per arm (same N,
                               parametrized by normalized arc length s∈[0,1]).
        q0_left / q0_right   : start configurations (spline begins here).
        Returns (times, q14_path, scipy_result).
        """
        wl = np.asarray(way_left, float)
        wr = np.asarray(way_right, float)
        assert wl.shape[0] == wr.shape[0], "waypoint counts must match"
        N = wl.shape[0]

        # arc-length parametrisation per arm
        def arclen(w):
            s = np.zeros(len(w))
            for i in range(1, len(w)):
                s[i] = s[i-1] + np.linalg.norm(w[i] - w[i-1])
            return s / s[-1] if s[-1] > 1e-9 else np.linspace(0, 1, len(w))
        sl, sr = arclen(wl), arclen(wr)

        def target_at(s, w, sarr):
            return np.array([np.interp(s, sarr, w[:, k]) for k in range(3)])

        # knots for a clamped cubic B-spline with M control points over [0,1]
        p = 3
        n_int = M - p
        t_int = np.linspace(0, 1, max(2, n_int + 1))[1:-1]
        knots = np.concatenate([[0]*(p+1), t_int, [1]*(p+1)])

        q0_14 = np.concatenate([np.asarray(q0_left, float),
                                np.asarray(q0_right, float)])
        # endpoint: IK to last waypoints (simple DLS on full model)
        qe_14 = np.concatenate([self._ik3(wl[-1], q0_left, "left"),
                                self._ik3(wr[-1], q0_right, "right")])
        c0 = np.tile(q0_14, (M, 1)).flatten()
        lo = np.tile(self._lo14(), M)
        up = np.tile(self._up14(), M)
        # pin first 3 control points at the start config (q̇(0)=q̈(0)=0) and
        # last 3 at the IK endpoint (q̇(T)=q̈(T)=0)
        for k in range(3):
            lo[k*14:(k+1)*14] = up[k*14:(k+1)*14] = q0_14
        for k in range(M-3, M):
            lo[k*14:(k+1)*14] = up[k*14:(k+1)*14] = qe_14
        bounds = list(zip(lo, up))

        s_arr = np.linspace(0, 1, n_samples)

        def eval_c(c_flat):
            C = c_flat.reshape(M, 14)
            return BSpline(knots, C, p)(s_arr)      # (n_samples, 14)

        calls = [0]

        def cost(c_flat):
            calls[0] += 1
            if time_budget is not None and time.time() - t_start > time_budget:
                raise _BudgetExhausted(c_flat)
            Q = eval_c(c_flat)
            C = c_flat.reshape(M, 14)
            track = 0.0
            lo14, up14 = self._lo14(), self._up14()
            for i, s in enumerate(s_arr):
                q16, _, _ = self.split(np.clip(Q[i], lo14, up14))
                tl = np.array([np.interp(s, sl, wl[:, k]) for k in range(3)])
                tr = np.array([np.interp(s, sr, wr[:, k]) for k in range(3)])
                track += float(np.sum((self.ee_pos(q16, "left") - tl)**2))
                track += float(np.sum((self.ee_pos(q16, "right") - tr)**2))
            smooth = float(np.sum(np.diff(C, n=2, axis=0)**2))
            coll = 0.0
            for i in range(0, n_samples, 2):
                coll += self.collision_penalty(Q[i])
            if verbose and calls[0] % 20 == 0:
                print(f"  [iter {calls[0]:4d}] track={track:.5f} smooth={smooth:.4f} coll={coll:.1f}")
            return w1*track + w2*smooth + w3*coll

        res = None
        t_start = time.time()
        try:
            res = minimize(cost, c0, method="SLSQP", bounds=bounds,
                           options={"maxiter": maxiter, "ftol": 1e-6, "disp": False})
        except _BudgetExhausted as e:
            class _R:  # best-effort result on budget exhaustion
                success = False
                nit = calls[0]
                fun = float("nan")
                x = np.asarray(e.args[0], float)
            res = _R()
            if verbose:
                print(f"  [budget] stopped after {calls[0]} evals "
                      f"({time.time()-t_start:.0f}s)")
        Q = eval_c(res.x)
        times = np.linspace(0, duration, n_samples)
        return times.tolist(), Q, res

    # ---------------------------------------------------------------- helpers
    def _lo14(self):
        return np.concatenate([
            np.array([self.model.lowerPositionLimit[i] for i in self.chk.q_idx["left"]]),
            np.array([self.model.lowerPositionLimit[i] for i in self.chk.q_idx["right"]])])

    def _up14(self):
        return np.concatenate([
            np.array([self.model.upperPositionLimit[i] for i in self.chk.q_idx["left"]]),
            np.array([self.model.upperPositionLimit[i] for i in self.chk.q_idx["right"]])])

    def _ik3(self, target_xyz, q_seed, side, iters=80, tol=1e-4):
        """3-D position IK for one arm (orientation-free endpoint fit).

        The position rows must come from the LOCAL_WORLD_ALIGNED Jacobian —
        LOCAL rows are EE-frame velocities and drive the DLS step the wrong
        way whenever the tool is rotated.
        """
        q_idx = self.chk.q_idx[side]
        lo = np.array([self.model.lowerPositionLimit[i] for i in q_idx])
        up = np.array([self.model.upperPositionLimit[i] for i in q_idx])
        q7 = np.clip(np.asarray(q_seed, float).copy(), lo, up)
        tgt = np.asarray(target_xyz, float)
        for _ in range(iters):
            q16 = self.chk.make_q(**{side: q7})
            pin.forwardKinematics(self.model, self.data, q16)
            pin.updateFramePlacement(self.model, self.data, self.ee_fid[side])
            err = tgt - self.data.oMf[self.ee_fid[side]].translation
            if np.linalg.norm(err) < tol:
                break
            J = pin.computeFrameJacobian(self.model, self.data, q16,
                                         self.ee_fid[side],
                                         pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J3 = J[:3][:, q_idx]
            dq = J3.T @ np.linalg.solve(J3 @ J3.T + 1e-4*np.eye(3), err)
            q7 = np.clip(q7 + dq, lo, up)
        return q7
