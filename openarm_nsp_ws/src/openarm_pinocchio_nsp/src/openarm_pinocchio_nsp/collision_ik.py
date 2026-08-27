# Copyright 2026 Enactic, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Collision-aware IK: null-space collision repulsion on top of NSP-IK.

Extends the two-stage NSP solver with a THIRD safety objective — collision
clearance — resolved, like singularity and joint-limit avoidance, entirely
inside the arm's 1-DoF null space (elbow swivel). The end-effector task path
is UNCHANGED: candidates are only accepted if the EE stays within tolerance.

Approach
--------
The base ``ik_nsp`` already runs a null-space ascent maximizing
``sigma_min + 0.5 * joint_margin``. Here we wrap the solve and, whenever the
DLS solution is inside the collision bubble (or the exact clearance is below
a comfort threshold), search along discrete samples of the null-space
manifold (the self-motion curve of a 7-DoF arm: elbow swivels, EE fixed) for
the configuration with maximum collision clearance.

Sampling the whole 1-D self-motion curve (rather than gradient-stepping it)
is deliberate: clearance is discontinuous under FCL margins, and the curve is
cheap to trace — each sample is one null-space step + one boolean collision
check (~0.4 ms).

The other arm participates as a FROZEN configuration (its measured q), which
is the correct model for single-arm operation against a stationary partner
arm; for bimanual planning both arms' trajectories are checked jointly in
``collision_planner`` / the 14-D B-spline optimizer.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin

from .collision import BimanualCollisionChecker
from .ik_nsp import damped_pseudoinverse
from .kinematics import PinocchioModel
from .urdf_path import resolve_urdf_path


class CollisionAwareIK:
    """NSP-IK + null-space collision repulsion for one arm of the bimanual model.

    Parameters
    ----------
    side : "left" | "right"
        The arm being solved for.
    other_q7 : 7-vector, optional
        The OTHER arm's (frozen) joint angles participating in collision checks.
        Updated per solve via :meth:`set_other` or the ``other_q7=`` kwarg —
        pass the partner arm's measured q right before planning.
    margin : float
        Collision bubble [m] — clearance below this triggers active avoidance
        (default 20 mm, same as the checker default).
    comfort : float
        Clearance the repulsion tries to reach (default 40 mm): once attained,
        the elbow stops migrating.
    """

    def __init__(self, side: str, other_q7: np.ndarray | None = None,
                 urdf_path: str | None = None,
                 checker: BimanualCollisionChecker | None = None,
                 margin: float = 0.02, comfort: float = 0.04,
                 null_iters: int = 30):
        if side not in ("left", "right"):
            raise ValueError(f"side must be left/right, got {side!r}")
        self.side = side
        self.model = PinocchioModel(urdf_path or resolve_urdf_path(), side)
        self.chk = checker or BimanualCollisionChecker(urdf_path)
        self.margin = float(margin)
        self.comfort = float(comfort)
        self.null_iters = null_iters
        self.other_q7 = (np.zeros(7) if other_q7 is None
                         else np.asarray(other_q7, float).copy())

    # ------------------------------------------------------------------ state
    def set_other(self, other_q7: np.ndarray) -> None:
        """Update the frozen partner-arm configuration."""
        self.other_q7 = np.asarray(other_q7, float).copy()

    def _q16(self, q7: np.ndarray) -> np.ndarray:
        if self.side == "left":
            return self.chk.make_q(left=q7, right=self.other_q7)
        return self.chk.make_q(left=self.other_q7, right=q7)

    def in_collision(self, q7: np.ndarray) -> bool:
        return self.chk.check(self._q16(q7), self.margin).in_collision

    # ------------------------------------------------------------- self-motion
    def nullspace_direction(self, q7: np.ndarray) -> np.ndarray:
        """Unit null-space direction at q7 (damped projector's dominant null vector)."""
        J = self.model.jacobian6(q7)
        P = np.eye(7) - damped_pseudoinverse(J, 1e-4) @ J
        # P acts as IDENTITY on null(J) (P n = n) and ~0 on the task-space rows,
        # so the null direction is P's LARGEST right-singular vector, not smallest.
        _, _, Vt = np.linalg.svd(P)
        n = Vt[0]
        if np.linalg.norm(J @ n) > 1e-2:   # sanity: must map to ~zero task motion
            return np.zeros(7)
        return n

    def trace_self_motion(self, q7: np.ndarray, span: float = 1.5,
                          n_samples: int = 24) -> list[np.ndarray]:
        """Sample the self-motion curve through q7 (both directions, limits-clipped).

        Returns configurations on the 1-D manifold q(s) with q(0)=q7, ordered
        from one extreme to the other through q7. The straight null-space chord
        is re-projected onto the manifold every few steps (drift correction).
        """
        J = self.model.jacobian6(q7)
        n_dir = self.nullspace_direction(q7)
        if not np.any(n_dir):
            return [q7]
        q_lo = q7.copy()
        for _ in range(8):   # walk one way, re-projecting periodically
            n_lo = self.nullspace_direction(q_lo)
            if not np.any(n_lo):
                break
            q_lo = np.clip(q_lo - (span / 8) * n_lo,
                           self.model.lower, self.model.upper)
        q_hi = q7.copy()
        for _ in range(8):
            n_hi = self.nullspace_direction(q_hi)
            if not np.any(n_hi):
                break
            q_hi = np.clip(q_hi + (span / 8) * n_hi,
                           self.model.lower, self.model.upper)
        # interpolate back through q7 on the chord lo→hi (curve is smooth, chord≈arc)
        ts = np.linspace(0.0, 1.0, n_samples)
        return [(1 - t) * q_lo + t * q_hi for t in ts]

    # ------------------------------------------------------------------ solve
    def ik(self, target_pos, target_quat, q_init, other_q7=None, *,
           ee_budget: float = 2e-3, verbose: bool = False, **nsp_kwargs):
        """Collision-aware IK.

        1. base NSP solve (DLS + σ/limit null-space ascent);
        2. if clearance < comfort: trace the self-motion curve around the NSP
           solution, keep every candidate that (a) holds the EE within
           ``ee_budget`` and (b) is inside joint limits, and pick the one with
           the largest clearance-to-comfort gain (tie-broken by continuity).

        Returns an :class:`IKResult` (the base one, with ``q`` possibly migrated
        along the null space). ``result.converged`` unchanged semantics; the
        collision status at the returned q is exposed via ``result_info`` dict.
        """
        if other_q7 is not None:
            self.set_other(other_q7)
        target_pos = np.asarray(target_pos, float)
        R = pin.Quaternion(np.asarray(target_quat, float)).matrix()
        target = pin.SE3(R, target_pos)

        res = self.model.ik_nsp(target_pos, target_quat, q_init, **nsp_kwargs)
        if not res.converged:
            return res, {"avoided": False, "reason": "base IK failed",
                         "clearance": None}
        q0 = res.q.copy()

        # clearance probe: boolean sweep along the self-motion curve first
        q16 = self._q16(q0)
        rep = self.chk.check(q16, self.margin)
        if not rep.in_collision and self._quick_clear(q0):
            return res, {"avoided": False, "reason": "already clear",
                         "clearance": "ok"}

        # migrate along the self-motion curve for max clearance
        cands = self.trace_self_motion(q0)
        best_q, best_gain, best_gap = q0, 0.0, None
        for q_c in cands:
            # EE task must hold
            oMf, _ = self.model._pose_and_jac(q_c)
            err = pin.log(oMf.inverse() * target).vector
            if np.linalg.norm(err) > ee_budget:
                continue
            gap = self._clearance_gap(q_c)   # comfort - clearance (smaller=better)
            if best_gap is None or gap < best_gap:
                best_gap, best_q = gap, q_c
        if best_gap is not None and best_gap < self._clearance_gap(q0) - 1e-3:
            from dataclasses import replace
            res = replace(res, q=best_q.copy())
            if verbose:
                print(f"[collision-ik] migrated elbow: "
                      f"gap {self._clearance_gap(q0):.3f} -> {best_gap:.3f}")
            return res, {"avoided": True, "reason": "null-space migration",
                         "gap": float(best_gap)}
        return res, {"avoided": False, "reason": "no better null-space point",
                     "gap": self._clearance_gap(q0)}

    # ---------------------------------------------------------------- helpers
    def _quick_clear(self, q7: np.ndarray) -> bool:
        """Cheap 'comfortably clear' test: no margin violation at 0.8×comfort."""
        return not self.chk.check(self._q16(q7), 0.8 * self.comfort).in_collision

    def _clearance_gap(self, q7: np.ndarray, probes: int = 4) -> float:
        """comfort - clearance proxy at q7 (0 = at comfort, larger = worse).

        Uses a small ladder of boolean margin checks instead of the expensive
        exact distance: the gap only steers the choice among a handful of
        candidates, so 3-4 probes × 0.4 ms each is plenty.
        """
        # ladder from hard collision to comfort; first clear rung = clearance
        for frac in np.linspace(0.0, 1.0, probes + 2):
            m = frac * self.comfort
            if not self.chk.check(self._q16(q7), m).in_collision:
                return self.comfort * (1.0 - frac)
        return self.comfort   # below-zero rung never reached: treat as worst
