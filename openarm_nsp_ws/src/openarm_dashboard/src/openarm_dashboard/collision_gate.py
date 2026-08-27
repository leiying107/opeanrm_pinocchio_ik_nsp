# Copyright 2026 Enactic, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared self-collision gate for the dashboard (single BimanualCollisionChecker).

Design rules (in order):
  1. NEVER block the panel: any failure inside the gate logs and ALLOWS the
     motion (fail-open) — the collision module must not become a new way to
     brick the UI. Hard safety remains the existing thermal/disable paths.
  2. Master switch, default ON: ``POST /collision {on: false}`` restores the
     pre-collision behavior of every gated entry point.
  3. The OTHER arm participates at its MEASURED configuration (read from the
     peer controller) — the correct model for single-arm operation against a
     parked partner.
  4. Runtime guard runs OUTSIDE the 250 Hz loop (5 Hz thread, boolean check
     ~0.4 ms) and only RAISES the red flag + logs; it never disables motors —
     hands are on the robot in zero-torque/impedance modes.

Gate results:
    GatePass(reason)      — proceed
    GateReject(reason)    — refuse the motion (log tells which pair/where)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

MARGIN = 0.02          # 20 mm bubble (validated in sim: 0 false positives at
                       # the calibration poses, GJK-with-margin is the fast path)
GUARD_HZ = 5.0         # runtime monitor rate (0.4 ms check, off the 250 Hz loop)


@dataclass
class GateResult:
    ok: bool
    reason: str = ""
    detail: dict | None = None

    @classmethod
    def pass_(cls, reason=""):
        return cls(True, reason)

    @classmethod
    def reject(cls, reason, detail=None):
        return cls(False, reason, detail)


class CollisionGate:
    """Singleton gate + runtime monitor over both ArmControllers."""

    def __init__(self):
        self.enabled = True          # master switch (default ON)
        self.available = False       # checker built successfully
        self._chk = None
        self._controllers: dict = {}     # side -> ArmController (weak refs not needed:
                                         # gate lives as long as the panel process)
        self._lock = threading.Lock()
        self._guard_thread: threading.Thread | None = None
        self._stop = threading.Event()
        # guard state (UI red flag)
        self.guard_hit = False           # inside margin right now
        self.guard_pair: tuple | None = None
        self.guard_last_t = 0.0          # monotonic time of last transition
        self._log = print                # replaced by attach() with rs.log

    # ------------------------------------------------------------- lifecycle
    def build(self) -> bool:
        """Construct the checker (2 s, once at panel startup)."""
        try:
            from openarm_pinocchio_nsp.collision import BimanualCollisionChecker
            self._chk = BimanualCollisionChecker()
            self.available = True
            return True
        except Exception as e:  # noqa: BLE001 — fail-open by design
            self._log(f"[collision] checker 构建失败，检测停用(fail-open): "
                      f"{type(e).__name__}: {e}")
            self.available = False
            return False

    def attach(self, controllers: dict, log=None) -> None:
        """Register controllers (for the peer arm's measured q) + start guard."""
        self._controllers = controllers
        if log is not None:
            self._log = log
        if self.available and self._guard_thread is None:
            self._stop.clear()
            self._guard_thread = threading.Thread(
                target=self._guard_loop, daemon=True, name="collision-guard")
            self._guard_thread.start()

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------ primitives
    def _peer_q(self, side: str) -> np.ndarray | None:
        """Measured 7-vector of the OTHER arm (None if unreadable)."""
        peer = "right" if side == "left" else "left"
        c = self._controllers.get(peer)
        if c is None or c._last_actual_q is None:
            return None
        return np.asarray(c._last_actual_q, float)

    def _arm_q(self, side: str) -> np.ndarray | None:
        c = self._controllers.get(side)
        if c is None or c._last_actual_q is None:
            return None
        return np.asarray(c._last_actual_q, float)

    def _q16(self, side: str, q7: np.ndarray) -> np.ndarray | None:
        peer = self._peer_q(side)
        if peer is None:
            return None
        if side == "left":
            return self._chk.make_q(left=q7, right=peer)
        return self._chk.make_q(left=peer, right=q7)

    def _pair_str(self, pair) -> str:
        if not pair:
            return ""
        a = pair[0].replace("openarm_", "").replace("_0", "")
        b = pair[1].replace("openarm_", "").replace("_0", "")
        return f"{a}↔{b}"

    # ----------------------------------------------------------------- gates
    def gate_config(self, side: str, q7: np.ndarray, label: str) -> GateResult:
        """Gate a static configuration (endpoints, home targets).

        Checks the TARGET config (not the path) — for joint-interpolated moves
        the straight joint path between two clear configs stays clear in
        practice (validated: the 40-scenario stress had zero path violations
        between gated endpoints); Cartesian paths are gated separately.
        """
        if not self.enabled or not self.available:
            return GateResult.pass_("gate off/unavailable")
        try:
            q16 = self._q16(side, q7)
            if q16 is None:
                return GateResult.pass_("peer q unreadable")
            rep = self._chk.check(q16, MARGIN)
            if rep.in_collision:
                return GateResult.reject(
                    f"{label} 目标位姿碰撞 {self._pair_str(rep.worst_pair)}",
                    {"pair": rep.worst_pair, "n": rep.n_violating})
            return GateResult.pass_()
        except Exception as e:  # noqa: BLE001 — fail-open
            self._log(f"[collision] gate异常(fail-open): {type(e).__name__}: {e}")
            return GateResult.pass_("gate error")

    def gate_trajectory(self, side: str, q_path, label: str) -> GateResult:
        """Gate a joint trajectory: clean → pass; violating index → reject
        (callers that can repair/detour do so BEFORE calling this)."""
        if not self.enabled or not self.available:
            return GateResult.pass_("gate off/unavailable")
        try:
            peer = self._peer_q(side)
            if peer is None:
                return GateResult.pass_("peer q unreadable")
            path16 = []
            for q in q_path:
                if side == "left":
                    path16.append(self._chk.make_q(left=q, right=peer))
                else:
                    path16.append(self._chk.make_q(left=peer, right=q))
            viol = self._chk.check_trajectory(path16, MARGIN)
            if viol is not None:
                rep = self._chk.check(path16[viol], MARGIN)
                return GateResult.reject(
                    f"{label} 轨迹第{viol + 1}/{len(path16)}点碰撞 "
                    f"{self._pair_str(rep.worst_pair)}",
                    {"index": viol, "pair": rep.worst_pair})
            return GateResult.pass_()
        except Exception as e:  # noqa: BLE001
            self._log(f"[collision] gate异常(fail-open): {type(e).__name__}: {e}")
            return GateResult.pass_("gate error")

    # ---------------------------------------------------------- runtime guard
    def _guard_loop(self) -> None:
        period = 1.0 / GUARD_HZ
        while not self._stop.wait(period):
            if not self.enabled:
                continue
            try:
                ql, qr = self._arm_q("left"), self._arm_q("right")
                if ql is None or qr is None:
                    continue
                rep = self._chk.check(self._chk.make_q(left=ql, right=qr), MARGIN)
                hit = rep.in_collision
                if hit and not self.guard_hit:        # rising edge → log once
                    self._log(f"⚠ [collision] 实时监护: 双臂间距<20mm "
                              f"({self._pair_str(rep.worst_pair)})")
                if hit != self.guard_hit:
                    self.guard_last_t = time.monotonic()
                self.guard_hit = hit
                self.guard_pair = rep.worst_pair if hit else None
            except Exception:  # noqa: BLE001 — guard must never die
                pass

    # -------------------------------------------------------------------- UI
    def state(self) -> dict:
        """Snapshot for the SSE stream."""
        return {
            "on": self.enabled and self.available,
            "avail": self.available,
            "hit": self.guard_hit,
            "pair": (self.guard_pair[0].replace("openarm_", "").replace("_0", "")
                     + "↔" + self.guard_pair[1].replace("openarm_", "").replace("_0", ""))
                    if self.guard_pair else "",
        }

    def set_enabled(self, on: bool) -> None:
        self.enabled = bool(on)
        if not on:
            self.guard_hit = False
            self.guard_pair = None


# module-level singleton (both panels share ONE checker — 700 ms to build, and
# the URDF/pair set is process-global anyway)
gate = CollisionGate()
