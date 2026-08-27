# Copyright 2026 Enactic, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bimanual self-collision checking via Pinocchio + hpp-fcl.

Covers ALL collision categories on the OpenArm v1.0 bimanual model:
  - per-arm self-collision (arm vs itself)
  - LEFT vs RIGHT arm (cross-arm, 78 pairs)
  - arms vs central body/base (body_link0, 48 pairs)

Design notes
------------
- Works on the FULL bimanual q vector (q16: 7+7 arm joints + 2 finger joints),
  so the same checker serves single-arm planning (other arm frozen) and
  future bimanual coordinated planning (both arms moving).
- Adjacent-link pairs are suppressed by KINEMATIC-TREE ancestry (parent /
  grandparent), NOT by joint-id numeric difference — the numeric rule used by
  bspline_planner.py wrongly keeps shoulder-shell vs upper-arm (joint ids far
  apart on the bimanual tree) permanently ~0.5 mm apart and drops every
  cross-arm pair.
- Boolean collision checks with security margins run ~0.4 ms (GJK early-out);
  exact min-distance queries are mesh-mesh expensive (~ms-100ms) and are
  reserved for offline reporting, with an AABB prefilter to skip distant pairs.

API summary
-----------
    from openarm_pinocchio_nsp.collision import BimanualCollisionChecker
    chk = BimanualCollisionChecker()                 # resolves URDF like the rest of the package
    chk = BimanualCollisionChecker(other_side="left")  # convenience: single-arm mode helpers
    rep = chk.check(q16)                             # CollisionReport
    rep.in_collision, rep.worst_pair, rep.min_distance
    rep = chk.check(q16, margin=0.02)                # 20 mm bubble
    d, name_a, name_b = chk.min_distance(q16)        # exact (slow, offline)
    idx = chk.check_trajectory(q16_path)             # first violating sample index (None = clean)
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin

from .urdf_path import resolve_urdf_path

_PKG_DIRS = [
    "/ros2_ws/openarm_ros2/install/openarm_description/share",
    "/ros2_ws/openarm_ros2/openarm_description",
]

_ARM_DOF = 7


class CollisionReport:
    """Result of one boolean collision check."""

    __slots__ = ("in_collision", "margin", "n_violating", "worst_pair", "min_distance")

    def __init__(self, in_collision: bool, margin: float, n_violating: int,
                 worst_pair=None, min_distance: float = None):
        self.in_collision = in_collision
        self.margin = margin
        self.n_violating = n_violating
        self.worst_pair = worst_pair          # (name_a, name_b) or None
        self.min_distance = min_distance      # only set when asked for (expensive)

    def __repr__(self):
        if self.in_collision:
            return (f"CollisionReport(COLLISION margin={self.margin*1000:.0f}mm "
                    f"n={self.n_violating} worst={self.worst_pair})")
        return f"CollisionReport(clear, margin={self.margin*1000:.0f}mm)"


class BimanualCollisionChecker:
    """Self-collision checker for the full OpenArm bimanual model.

    Parameters
    ----------
    urdf_path : str, optional
        URDF to build from (defaults to resolve_urdf_path()).
    margin : float
        Default security margin [m] for boolean checks (default 20 mm).
    """

    # Categories for reporting / per-category margins
    CAT_ARM_SELF = "arm_self"      # links of the same arm
    CAT_CROSS_ARM = "cross_arm"    # left link vs right link
    CAT_ARM_BODY = "arm_body"      # arm link vs central body
    CAT_OTHER = "other"

    def __init__(self, urdf_path: str | None = None, margin: float = 0.02):
        self.urdf = urdf_path or resolve_urdf_path()
        self.default_margin = float(margin)

        # full bimanual model (mimic joints for fingers)
        try:
            self.model = pin.buildModelFromUrdf(self.urdf, True)
        except TypeError:
            self.model = pin.buildModelFromUrdf(self.urdf)
        self.data = self.model.createData()

        # collision geometries (URDF ships per-link collision STLs incl. body)
        self.cgeom = pin.buildGeomFromUrdf(
            self.model, self.urdf, pin.GeometryType.COLLISION, _PKG_DIRS)

        # joint index bookkeeping for both arms
        self.q_idx: dict[str, list[int]] = {}
        for side in ("left", "right"):
            idx = []
            for i in range(1, _ARM_DOF + 1):
                jid = self.model.getJointId(f"openarm_{side}_joint{i}")
                if jid >= self.model.njoints:
                    raise ValueError(f"joint openarm_{side}_joint{i} not in URDF")
                idx.append(int(self.model.idx_qs[jid]))
            self.q_idx[side] = idx

        self._build_pair_set()

        self.cdata_by_margin: dict[float, pin.GeometryData] = {}

    # ------------------------------------------------------------------ pairs
    def _build_pair_set(self):
        """All-pairs minus kinematically-adjacent ones (tree ancestry rule).

        Adjacent = one geometry's nearest movable joint lies within 2 levels of
        the other's joint ancestry (self, parent, grandparent), universe included.
        This suppresses bolted-together shells (body/link0 share `universe`,
        left/right link0 can never move relative to each other) and consecutive
        arm links, while KEEPING genuinely checkable pairs like body↔forearm or
        shoulder-shell↔upper-arm.

        Extra rule: the two fingers of one gripper are a parallel pair that is
        permanently near/touching by construction — excluded (they close on
        purpose). Cross-side fingers and fingers vs the OTHER arm remain.
        """
        self.cgeom.addAllCollisionPairs()
        objs = self.cgeom.geometryObjects

        def ancestors(jid):
            """{joint, parent, grandparent} — universe (0) included."""
            chain, j = [], jid
            while len(chain) < 3:
                chain.append(j)
                if j == 0:
                    break
                j = self.model.parents[j]
            return set(chain)

        anc_cache = {i: ancestors(o.parentJoint) for i, o in enumerate(objs)}
        frame_cache = [self.model.frames[o.parentFrame].name for o in objs]

        def same_gripper(i, j):
            fi, fj = frame_cache[i], frame_cache[j]
            return ("finger" in fi and "finger" in fj
                    and fi.split("_")[1] == fj.split("_")[1])

        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                cp = pin.CollisionPair(i, j)
                if not self.cgeom.existCollisionPair(cp):
                    continue
                if (objs[i].parentJoint in anc_cache[j]
                        or objs[j].parentJoint in anc_cache[i]
                        or same_gripper(i, j)):
                    self.cgeom.removeCollisionPair(cp)

        self.n_pairs = len(self.cgeom.collisionPairs)
        self._classify_pairs()

    def _classify_pairs(self):
        """Tag every pair with a category for reporting and per-category margins.

        Uses parent FRAME names (e.g. 'openarm_left_link0') — parentJoint names
        are useless here because every link before the first movable joint of an
        arm collapses onto `universe` in Pinocchio.
        """
        objs = self.cgeom.geometryObjects
        frame_cache = [self.model.frames[o.parentFrame].name for o in objs]

        def arm_of(fname):
            if "openarm_left" in fname:
                return "L"
            if "openarm_right" in fname:
                return "R"
            return "B"  # central body / world

        self.pair_category: list[str] = []
        self.cat_counts = {}
        for cp in self.cgeom.collisionPairs:
            a, b = arm_of(frame_cache[cp.first]), arm_of(frame_cache[cp.second])
            key = "".join(sorted(a + b))
            cat = {
                "LL": self.CAT_ARM_SELF,
                "RR": self.CAT_ARM_SELF,
                "LR": self.CAT_CROSS_ARM,
                "BL": self.CAT_ARM_BODY,
                "BR": self.CAT_ARM_BODY,
            }.get(key, self.CAT_OTHER)
            self.pair_category.append(cat)
            self.cat_counts[cat] = self.cat_counts.get(cat, 0) + 1

    # ------------------------------------------------------------- q helpers
    def neutral_q(self) -> np.ndarray:
        return pin.neutral(self.model)

    def make_q(self, left: np.ndarray = None, right: np.ndarray = None,
               q_base: np.ndarray = None) -> np.ndarray:
        """Build full q16 from per-arm 7-vectors on top of q_base (default neutral)."""
        q = (q_base if q_base is not None else pin.neutral(self.model)).copy()
        if left is not None:
            for k, qi in enumerate(self.q_idx["left"]):
                q[qi] = left[k]
        if right is not None:
            for k, qi in enumerate(self.q_idx["right"]):
                q[qi] = right[k]
        return q

    def arm_q(self, q16: np.ndarray, side: str) -> np.ndarray:
        return np.array([q16[i] for i in self.q_idx[side]])

    # ------------------------------------------------------------------ data
    def _cdata(self, margin: float) -> pin.GeometryData:
        """GeometryData with per-pair security margins, cached per margin value."""
        margin = float(margin)
        if margin not in self.cdata_by_margin:
            cd = pin.GeometryData(self.cgeom)
            if margin > 0.0:
                M = np.zeros((len(self.cgeom.geometryObjects),
                              len(self.cgeom.geometryObjects)))
                for cp in self.cgeom.collisionPairs:
                    M[cp.first, cp.second] = margin
                    M[cp.second, cp.first] = margin
                cd.setSecurityMargins(self.cgeom, M, True)
            self.cdata_by_margin[margin] = cd
        return self.cdata_by_margin[margin]

    # ---------------------------------------------------------------- checks
    def check(self, q16: np.ndarray, margin: float | None = None) -> CollisionReport:
        """Boolean collision check (with security margin) at full-bimanual q.

        ~0.4 ms at margin=20 mm (GJK early-out makes margined checks FASTER
        than raw ones). Returns CollisionReport; also caches violating pairs.
        """
        margin = self.default_margin if margin is None else float(margin)
        cdata = self._cdata(margin)
        pin.updateGeometryPlacements(self.model, self.data, self.cgeom, cdata, np.asarray(q16, float))
        pin.computeCollisions(self.cgeom, cdata, False)

        violating = []
        for k, res in enumerate(cdata.collisionResults):
            if res.isCollision():
                cp = self.cgeom.collisionPairs[k]
                objs = self.cgeom.geometryObjects
                violating.append((objs[cp.first].name, objs[cp.second].name,
                                  self.pair_category[k]))
        if violating:
            return CollisionReport(True, margin, len(violating), violating[0])
        return CollisionReport(False, margin, 0, None)

    def min_distance(self, q16: np.ndarray, upper_bound: float = 0.10) -> tuple:
        """Exact minimum distance [m] across checked pairs (negative = penetration).

        Expensive (mesh-mesh distance) — offline use only. `upper_bound` prunes
        pairs farther than this (AABB prefilter + GJK upper bound), 0.10 m default.
        Returns (min_distance, name_a, name_b); (inf, None, None) if no pair.
        """
        cdata = self._cdata(0.0)
        pin.updateGeometryPlacements(self.model, self.data, self.cgeom, cdata, np.asarray(q16, float))
        pin.computeDistances(self.model, self.data, self.cgeom, cdata, np.asarray(q16, float))
        best, pair = np.inf, None
        for k, res in enumerate(cdata.distanceResults):
            d = res.min_distance
            if d < best:
                best = d
                cp = self.cgeom.collisionPairs[k]
                objs = self.cgeom.geometryObjects
                pair = (objs[cp.first].name, objs[cp.second].name)
        return float(best), *(pair if pair else (None, None))

    def check_trajectory(self, q16_path, margin: float | None = None,
                         start: int = 0) -> int | None:
        """First violating sample index in a q16 trajectory, or None if clean.

        Boolean margin check per sample; ~0.4 ms × n_samples.
        """
        for i in range(start, len(q16_path)):
            rep = self.check(np.asarray(q16_path[i], float), margin)
            if rep.in_collision:
                return i
        return None

    # ------------------------------------------------------------- reporting
    def sweep_distance(self, q16_path) -> tuple[list, list]:
        """Per-sample exact min distance over a trajectory (offline, slow).

        Returns (min_dist_series, worst_pair_at_argmin).
        """
        dists, worst = [], None
        for q in q16_path:
            d, a, b = self.min_distance(q)
            dists.append(d)
        if dists:
            k = int(np.argmin(dists))
            d, a, b = self.min_distance(np.asarray(q16_path[k], float))
            worst = (a, b)
        return dists, worst

    # -------------------------------------------------------------- metrics
    @staticmethod
    def summarize(counts: dict) -> str:
        return " | ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
