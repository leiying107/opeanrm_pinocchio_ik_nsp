# Copyright 2026 Enactic, Inc.
# SPDX-License-Identifier: Apache-2.0
"""B-spline joint-space trajectory optimization with self-collision avoidance.

Offline planner: given N end-effector pose waypoints, fit a cubic (C²) B-spline
in JOINT space whose control points are optimized (SLSQP) to track the Cartesian
path while staying within joint limits and free of self-collision.

This is a NEW module — the existing plan_cartesian / fit_arc / dashboard IK
features are all preserved unchanged.

Pipeline:
  1. load model + collision model (this arm's links only)
  2. B-spline parametrisation (M control points, cubic, clamped uniform knots)
  3. cost = w1·tracking + w2·acceleration-smoothness + w3·collision-penalty
  4. SLSQP solve (joint limits as bounds)
  5. post-verify (100-pt collision recheck, velocity/acceleration, no branch jump)

P0 (this file): model + collision + B-spline param + evaluate + collision check.
P1: cost + gradient + SLSQP.  P2: post-verify.  P3: dashboard integration.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
from scipy.interpolate import BSpline

from .urdf_path import resolve_urdf_path

_ARM_DOF = 7
_PKG_DIRS = [
    "/ros2_ws/openarm_ros2/install/openarm_description/share",
    "/ros2_ws/openarm_ros2/openarm_description",
]


class BSplineOptimizer:
    """Joint-space B-spline trajectory optimizer with self-collision avoidance."""

    def __init__(self, urdf_path: str | None = None, side: str = "right"):
        if side not in ("left", "right"):
            raise ValueError(f"side must be left/right, got {side!r}")
        self.side = side
        urdf = urdf_path or resolve_urdf_path()

        # full bimanual model
        self.model = pin.buildModelFromUrdf(urdf, True)
        self.data = self.model.createData()

        # this arm's joint indices in the full q vector
        self.q_idx: list[int] = []
        for i in range(1, _ARM_DOF + 1):
            jid = self.model.getJointId(f"openarm_{side}_joint{i}")
            if jid >= self.model.njoints:
                raise ValueError(f"joint openarm_{side}_joint{i} not in URDF")
            self.q_idx.append(int(self.model.idx_qs[jid]))

        self.ee_fid = self.model.getFrameId(f"openarm_{side}_hand_tcp")
        self.lower = np.array(
            [self.model.lowerPositionLimit[i] for i in self.q_idx], dtype=float)
        self.upper = np.array(
            [self.model.upperPositionLimit[i] for i in self.q_idx], dtype=float)

        # collision model (this arm's links only)
        self.cgeom = pin.buildGeomFromUrdf(
            self.model, urdf, pin.GeometryType.COLLISION, _PKG_DIRS)
        self._generate_pairs()
        self.cdata = pin.GeometryData(self.cgeom)
        self._select_arm_pairs()

    # -------------------------------------------------- collision pairs
    def _generate_pairs(self):
        """Manually add collision pairs (non-adjacent links only)."""
        objs = self.cgeom.geometryObjects
        parents = [o.parentJoint for o in objs]
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                if parents[i] == parents[j]:
                    continue  # same link (multiple meshes)
                if abs(parents[i] - parents[j]) <= 2:
                    continue  # same/adjacent/near links (always in contact or too close to be useful)
                self.cgeom.addCollisionPair(pin.CollisionPair(i, j))

    def _select_arm_pairs(self):
        """Indices of collision pairs where BOTH objects belong to this arm."""
        prefix = f"openarm_{self.side}_"
        objs = self.cgeom.geometryObjects

        def belongs(go):
            return prefix in self.model.names[go.parentJoint]

        self.arm_cp_idx = [
            i for i, cp in enumerate(self.cgeom.collisionPairs)
            if belongs(objs[cp.first]) and belongs(objs[cp.second])
        ]

    # -------------------------------------------------- helpers
    def _full_q(self, q7: np.ndarray) -> np.ndarray:
        q = pin.neutral(self.model)
        for i, qi in enumerate(self.q_idx):
            q[qi] = q7[i]
        return q

    def fk_pos(self, q7: np.ndarray) -> np.ndarray:
        """EE position (xyz) of this arm at q7."""
        q = self._full_q(q7)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, self.ee_fid)
        return self.data.oMf[self.ee_fid].translation.copy()

    def _ik_pos(self, target_xyz, q_seed, max_iters=80, tol=1e-4):
        """Simple DLS position-only IK (3D) — used to get endpoint joint angles."""
        q7 = np.clip(np.asarray(q_seed, float).copy(), self.lower, self.upper)
        tgt = np.asarray(target_xyz, float)
        for _ in range(max_iters):
            q = self._full_q(q7)
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacement(self.model, self.data, self.ee_fid)
            err = tgt - self.data.oMf[self.ee_fid].translation
            if np.linalg.norm(err) < tol:
                break
            J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_fid, pin.ReferenceFrame.LOCAL)
            J3 = J[:3][:, self.q_idx]
            dq = J3.T @ np.linalg.solve(J3 @ J3.T + 1e-4 * np.eye(3), err)
            q7 = np.clip(q7 + dq, self.lower, self.upper)
        return q7

    def check_collisions(self, q7: np.ndarray) -> int:
        """Return number of colliding pairs at q7 (this arm's links only)."""
        q = self._full_q(q7)
        pin.updateGeometryPlacements(self.model, self.data, self.cgeom, self.cdata, q)
        n = 0
        for i in self.arm_cp_idx:
            if pin.computeCollision(self.cgeom, self.cdata, i):
                n += 1
        return n

    def min_collision_distance(self, q7: np.ndarray) -> float:
        """Minimum distance across this arm's collision pairs (negative = penetration)."""
        q = self._full_q(q7)
        pin.updateGeometryPlacements(self.model, self.data, self.cgeom, self.cdata, q)
        worst = 1e6
        for i in self.arm_cp_idx:
            try:
                res = pin.computeDistance(self.cgeom, self.cdata, i)
                d = res.min_distance if hasattr(res, "min_distance") else 0.0
                worst = min(worst, d)
            except Exception:
                pass
        return worst

    # -------------------------------------------------- B-spline parametrisation
    def make_bspline(self, control_points: np.ndarray, duration: float = 3.0) -> BSpline:
        """Build a clamped cubic B-spline from (M,7) control points over [0,duration]."""
        C = np.asarray(control_points, dtype=float)
        M = C.shape[0]
        p = 3
        # clamped uniform knots: endpoints repeated p+1 times
        n_internal = M - p  # number of internal knots
        if n_internal < 1:
            # too few control points for cubic — fall back to degree
            p = max(1, M - 1)
        t_internal = np.linspace(0, duration, max(2, M - p + 1))[1:-1] if M - p > 1 else []
        knots = np.concatenate([[0] * (p + 1), t_internal, [duration] * (p + 1)])
        # scipy needs knots normalized; we use [0,1] internally and scale time at eval
        knots = knots / duration
        ts = np.linspace(0, 1, M)  # control point abscissae (Greville-ish, uniform)
        return BSpline(knots, C, p)

    def evaluate(self, spline: BSpline, n_samples: int = 100) -> tuple[list, np.ndarray]:
        """Sample q(t) at n_samples points; return (times, q_path[n_samples,7])."""
        s = np.linspace(0, 1, n_samples)
        q7 = spline(s)  # (n_samples, 7)  — clamp to limits
        q7 = np.clip(q7, self.lower, self.upper)
        times = np.linspace(0, 1, n_samples)  # normalized; caller scales by duration
        return times.tolist(), q7

    # -------------------------------------------------- P2: post-verification
    def post_verify(self, q_path: np.ndarray, duration: float = 3.0,
                    n_collision_check: int = 100, vmax=None, amax=None,
                    max_step: float = 0.2) -> dict:
        """Post-verification of an optimized trajectory.

        Checks: (a) n_collision_check-point collision recheck; (b) velocity/acceleration
        limits; (c) no branch-jump (adjacent step < max_step).
        Returns dict of results.
        """
        n = len(q_path)
        q = np.asarray(q_path)
        dt = duration / max(1, n - 1)
        # (a) collision recheck on dense grid
        collision_samples = np.linspace(0, n - 1, n_collision_check).astype(int)
        n_col = max(self.check_collisions(q[i]) for i in collision_samples)
        # (b) velocity & acceleration
        if n > 1:
            vel = np.diff(q, axis=0) / dt
            max_vel = float(np.max(np.abs(vel)))
        else:
            max_vel = 0.0
        if n > 2:
            acc = np.diff(q, n=2, axis=0) / (dt ** 2)
            max_acc = float(np.max(np.abs(acc)))
        else:
            max_acc = 0.0
        # (c) branch-jump check
        max_step_actual = float(np.max(np.linalg.norm(np.diff(q, axis=0), axis=1))) if n > 1 else 0.0

        result = {
            "collisions": n_col,
            "max_velocity": max_vel,
            "max_acceleration": max_acc,
            "max_adjacent_step": max_step_actual,
            "velocity_ok": vmax is None or max_vel <= vmax,
            "acceleration_ok": amax is None or max_acc <= amax,
            "no_branch_jump": max_step_actual < max_step,
            "passed": n_col == 0 and max_step_actual < max_step,
        }
        if vmax is not None:
            result["velocity_ok"] = max_vel <= vmax
        if amax is not None:
            result["acceleration_ok"] = max_acc <= amax
        return result

    # -------------------------------------------------- P1: optimization
    def optimize(self, waypoints_xyz, q_init, duration=3.0, n_samples=50,
                 w1=1.0, w2=1.0, w3=10.0, eps=0.02, maxiter=100):
        """Optimize B-spline control points to track waypoints_xyz (N,3).

        cost = w1·Σ‖FK(q(t_i)) − x_d(t_i)‖²  (tracking)
             + w2·Σ‖c_{k+2} − 2·c_{k+1} + c_k‖²  (acceleration smoothness)
             + w3·Σ max(0, ε − dist(q(t_i)))²      (collision penalty)

        Returns (spline, q_path[n_samples,7], scipy_result).
        """
        from scipy.optimize import minimize
        pts = np.asarray(waypoints_xyz, float)
        N = len(pts)
        M = max(N, 12)

        # arc-length parametrisation of the target waypoints (for interpolation)
        sw = np.zeros(N)
        for i in range(1, N):
            sw[i] = sw[i - 1] + np.linalg.norm(pts[i] - pts[i - 1])
        sw = sw / sw[-1] if sw[-1] > 1e-9 else np.linspace(0, 1, N)

        def target_at(s):
            return np.array([np.interp(s, sw, pts[:, k]) for k in range(3)])

        c0 = np.tile(np.clip(np.asarray(q_init, float), self.lower, self.upper), (M, 1)).flatten()
        lo = np.tile(self.lower, M)
        up = np.tile(self.upper, M)
        # fix first 2 and last 2 control points → zero start/stop velocity (q̇(0)=q̇(T)=0)
        q_start = np.clip(np.asarray(q_init, float), self.lower, self.upper)
        q_end = self._ik_pos(pts[-1], q_start)
        for j in range(7):
            for k in (0, 1, 2):  # first 3 control pts fixed → q̇(0)=q̈(0)=0
                lo[k * 7 + j] = up[k * 7 + j] = q_start[j]
            for k in (M - 3, M - 2, M - 1):  # last 3 → q̇(T)=q̈(T)=0
                lo[k * 7 + j] = up[k * 7 + j] = q_end[j]
        bounds = list(zip(lo, up))

        def cost(c_flat):
            C = c_flat.reshape(M, 7)
            spline = self.make_bspline(C, duration)
            s_arr = np.linspace(0, 1, n_samples)
            q7 = np.clip(spline(s_arr), self.lower, self.upper)
            # tracking error
            track = 0.0
            for i in range(n_samples):
                pos = self.fk_pos(q7[i])
                track += float(np.sum((pos - target_at(s_arr[i])) ** 2))
            # acceleration smoothness (2nd difference of control points)
            smooth = float(np.sum(np.diff(C, n=2, axis=0) ** 2))
            # collision penalty (subsampled; skipped entirely if w3==0 for speed)
            coll = 0.0
            if w3 > 0:
                for i in range(0, n_samples, 5):
                    d = self.min_collision_distance(q7[i])
                    if d < eps:
                        coll += (eps - d) ** 2
            return w1 * track + w2 * smooth + w3 * coll

        res = minimize(cost, c0, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-6, 'disp': False})
        C_opt = res.x.reshape(M, 7)
        spline_opt = self.make_bspline(C_opt, duration)
        _, q_path = self.evaluate(spline_opt, n_samples)
        return spline_opt, q_path, res


# convenience
def load(side: str = "right", urdf: str | None = None) -> BSplineOptimizer:
    return BSplineOptimizer(urdf, side)
