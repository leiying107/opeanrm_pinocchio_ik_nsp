# Copyright 2026 Enactic, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Collision-aware Cartesian planning: route around, degrade gracefully.

Three strategies, in order of preference:

1. **Null-space migration** (per-waypoint, no path change) — for each IK sample
   that lands inside the collision bubble, slide along the 1-DoF self-motion
   curve (elbow swivel) to restore clearance. End-effector path unchanged.
   Handles "the elbow would clip the obstacle while the EE path is fine".

2. **Waypoint detour** (path deformation) — when the EE PATH itself passes
   through the partner arm, insert a via point offset perpendicular to the
   segment (direction chosen away from the obstacle's closest point) and
   re-plan the two sub-segments. Bounded recursion depth.

3. **Truncation** — if neither works (target truly unreachable), return the
   collision-free prefix plus a diagnostic, so the executor can still perform
   the safe part of the motion.

All checks use the full bimanual model: the partner arm participates as a
frozen obstacle at its measured configuration (single-arm operation) or as a
moving trajectory (bimanual planning — pass ``other_path`` aligned in time).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pinocchio as pin

from .cartesian_planner import Waypoint, plan_cartesian
from .collision import BimanualCollisionChecker
from .collision_ik import CollisionAwareIK
from .kinematics import PinocchioModel
from .urdf_path import resolve_urdf_path


@dataclass
class CollisionPlanResult:
    """Outcome of a collision-aware plan."""
    ok: bool                      # full path collision-free (strategy 1/2 succeeded)
    truncated: bool               # path cut before the obstacle (strategy 3)
    q_path: list = field(default_factory=list)     # (n,7) executed joint path
    times: list = field(default_factory=list)
    n_points_requested: int = 0
    first_violation: int | None = None            # index in the ORIGINAL path
    strategy: str = ""            # "nullspace" | "detour" | "truncate" | "clean"
    detour_added: list = field(default_factory=list)  # via points inserted
    notes: list = field(default_factory=list)

    def __len__(self):
        return len(self.q_path)


class CollisionAwarePlanner:
    """plan_cartesian + self-collision avoidance for one arm of the bimanual model."""

    def __init__(self, side: str, other_q7: np.ndarray | None = None,
                 urdf_path: str | None = None,
                 checker: BimanualCollisionChecker | None = None,
                 margin: float = 0.02):
        self.side = side
        self.chk = checker or BimanualCollisionChecker(urdf_path)
        self.cik = CollisionAwareIK(side, other_q7, urdf_path,
                                    checker=self.chk, margin=margin)
        self.model: PinocchioModel = self.cik.model
        self.margin = margin

    # ---------------------------------------------------------------- helpers
    def _q16s(self, q7_path, other_q7=None):
        other = self.cik.other_q7 if other_q7 is None else other_q7
        if self.side == "left":
            return [self.chk.make_q(left=q, right=other) for q in q7_path]
        return [self.chk.make_q(left=other, right=q) for q in q7_path]

    def set_other(self, other_q7: np.ndarray):
        self.cik.set_other(other_q7)

    # ------------------------------------------------------------------ plan
    def plan_line(self, pos_from, quat_from, pos_to, quat_to, q_init,
                  other_q7=None, max_detours: int = 2, verbose=False,
                  **plan_kwargs) -> CollisionPlanResult:
        """Plan a straight EE line with self-collision avoidance.

        Returns a CollisionPlanResult; ``ok=True`` means every sample is clear
        at ``margin``; ``truncated=True`` means the safe prefix only.
        """
        if other_q7 is not None:
            self.set_other(other_q7)

        base = plan_cartesian(self.model,
                              [Waypoint(np.asarray(pos_from, float), np.asarray(quat_from, float)),
                               Waypoint(np.asarray(pos_to, float), np.asarray(quat_to, float))],
                              q_init=np.asarray(q_init, float), **plan_kwargs)
        return self._resolve(base, [pos_from, pos_to], q_init,
                             max_detours, verbose, **plan_kwargs)

    # ------------------------------------------------------------- strategies
    def _resolve(self, base, endpoints, q_init, max_detours, verbose, **plan_kwargs):
        # A "clean" result must ALSO actually reach the goal: plan_cartesian
        # returns success=False with a partial q_path when IK breaks mid-chain
        # (behind-the-back starts hit this — the chain dies at the torso), and
        # a collision-free prefix is NOT a solution to the task.
        pos_to = np.asarray(endpoints[-1], float)
        reached = False
        if base.q_path:
            pos_end, _ = self.model.fk(base.q_path[-1])
            reached = float(np.linalg.norm(pos_end - pos_to)) <= 0.005
        if (not getattr(base, "success", True)) or not reached:
            if verbose:
                why = "IK break" if not getattr(base, "success", True) else "goal not reached"
                print(f"[planner] base plan unusable ({why}, "
                      f"{len(base.q_path)} pts, FK(end) {np.round(pos_end if base.q_path else [0,0,0],3)} "
                      f"vs target {np.round(pos_to,3)})")
            # fall through to detour; if nothing works, truncate the partial
            # path at ITS first collision too — a prefix past the violation is
            # no safer than a full colliding path.
            if max_detours > 0:
                detour = self._detour(base, endpoints, q_init, max_detours,
                                      verbose, **plan_kwargs)
                if detour is not None and detour.ok:
                    return detour
            part = list(base.q_path)
            n = len(part)
            pv = self.chk.check_trajectory(self._q16s(part), self.margin)
            safe_end = (pv - 1) if pv is not None else (n - 1)
            return CollisionPlanResult(False, True, part[:max(1, safe_end + 1)],
                                       list(base.times)[:max(1, safe_end + 1)],
                                       n, pv, "truncate",
                                       notes=["base plan broke before goal"])

        q_path = list(base.q_path)
        n = len(q_path)
        viol = self.chk.check_trajectory(self._q16s(q_path), self.margin)
        if viol is None:
            return CollisionPlanResult(True, False, q_path, list(base.times),
                                       n, None, "clean")

        # strategy 1: null-space migration on violating samples (path unchanged)
        fixed = self._nullspace_repair(q_path, verbose=verbose)
        if fixed is not None:
            return CollisionPlanResult(True, False, fixed, list(base.times),
                                       n, viol, "nullspace")

        # strategy 2: detour via perpendicular offset (recursed per sub-segment)
        if max_detours > 0:
            detour = self._detour(base, endpoints, q_init, max_detours,
                                  verbose, **plan_kwargs)
            if detour is not None and detour.ok:
                return detour

        # strategy 3: truncate at first violation
        safe_end = viol - 1 if viol is not None else n - 1
        res = CollisionPlanResult(False, True, q_path[:max(1, safe_end + 1)],
                                  list(base.times)[:max(1, safe_end + 1)],
                                  n, viol, "truncate",
                                  notes=[f"truncated at {viol}/{n}"])
        return res

    def _nullspace_repair(self, q_path, verbose=False):
        """Migrate violating IK samples along their self-motion curves.

        Returns the repaired (n,7) path if EVERY sample ends clear, else None.
        Continuity: a migrated sample must stay within 0.35 rad of its neighbor,
        and the final path is fully re-verified. Fast path: only samples that
        actually violate are traced (clean ones pass through untouched).
        """
        q_arr = [np.asarray(q, float) for q in q_path]
        # quick pre-scan: which samples violate?
        violating = [i for i, q in enumerate(q_arr)
                     if self.chk.check(self._q16s([q])[0], self.margin).in_collision]
        if not violating:
            return q_arr
        out = list(q_arr)
        prev = None
        for i in violating:
            q = q_arr[i]
            prev = out[i - 1] if i > 0 else None
            cands = self.cik.trace_self_motion(q, span=1.0, n_samples=15)
            cands_sorted = sorted(cands, key=lambda c: float(np.linalg.norm(c - q)))
            cand = None
            for c in cands_sorted:
                if (not self.chk.check(self._q16s([c])[0], self.margin).in_collision
                        and (prev is None or np.linalg.norm(c - prev) < 0.35)):
                    cand = c
                    break
            if cand is None:
                return None
            out[i] = cand
        viol = self.chk.check_trajectory(self._q16s(out), self.margin)
        if viol is not None:
            return None
        if verbose:
            print(f"[planner] null-space repair migrated {len(violating)}/{len(q_arr)} samples")
        return out

    def _detour(self, base, endpoints, q_init, max_detours, verbose, **plan_kwargs):
        """Insert a via point offset perpendicular to the violating segment.

        The via point splits the segment in two; each sub-segment is solved and
        validated INDEPENDENTLY (recursing with ``max_detours - 1``), and the
        result concatenates both — a sub-segment's success never masquerades as
        the whole path's.
        """
        pts = [np.asarray(e, float) for e in endpoints]
        pos_from, pos_to = pts[0], pts[-1]
        quat = self._last_quat
        seg = pos_to - pos_from
        L = np.linalg.norm(seg)
        if L < 1e-6:
            return None
        u = seg / L
        # perpendicular in the horizontal plane first (elbows go up/out, not down)
        perp = np.array([-u[1], u[0], 0.0])
        if np.linalg.norm(perp) < 0.05:
            perp = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        perp /= np.linalg.norm(perp)

        mid = 0.5 * (pos_from + pos_to)
        for sign in (1.0, -1.0):
            for dist in (0.08, 0.12, 0.18):
                via = mid + sign * dist * perp
                try:
                    plan1 = plan_cartesian(self.model,
                                           [Waypoint(pos_from, quat), Waypoint(via, quat)],
                                           q_init=np.asarray(q_init, float), **plan_kwargs)
                    sub1 = self._resolve(plan1, [pos_from, via], q_init,
                                         max_detours - 1, verbose, **plan_kwargs)
                    if not sub1.ok:
                        continue
                    # anchor segment 2 at the ACTUAL end of segment 1 (which may
                    # itself have detoured away from the nominal via)
                    q_mid = np.asarray(sub1.q_path[-1], float)
                    pos_mid, quat_mid = self.model.fk(q_mid)
                    plan2 = plan_cartesian(self.model,
                                           [Waypoint(pos_mid, quat_mid),
                                            Waypoint(pos_to, quat)],
                                           q_init=q_mid, **plan_kwargs)
                    sub2 = self._resolve(plan2, [pos_mid, pos_to], q_mid,
                                         max_detours - 1, verbose, **plan_kwargs)
                    if not sub2.ok:
                        continue
                except Exception:
                    continue
                q_path = list(sub1.q_path) + list(sub2.q_path[1:])
                t_off = sub1.times[-1] if sub1.times else 0.0
                times = list(sub1.times) + [t + t_off for t in list(sub2.times)[1:]]
                res = CollisionPlanResult(True, False, q_path, times,
                                          len(base.q_path), None, "detour",
                                          detour_added=[via.tolist()])
                if verbose:
                    print(f"[planner] detour via {np.round(via,3)} "
                          f"(sign={sign:+.0f}, d={dist}, {len(q_path)} pts)")
                return res
        return None

    def _quat_at(self, endpoints):
        """Orientation carried through the plan (caller's quat via plan_line_quat)."""
        return self._last_quat

    # ---------------------------------------------------------------- quoting
    def plan_line_quat(self, pos_from, quat, pos_to, q_init, **kw):
        self._last_quat = np.asarray(quat, float)
        return self.plan_line(pos_from, quat, pos_to, quat, q_init, **kw)
