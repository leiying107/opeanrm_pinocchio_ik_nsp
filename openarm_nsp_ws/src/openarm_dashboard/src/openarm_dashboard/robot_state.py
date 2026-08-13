# Copyright 2026 Enactic, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Thread-safe state containers shared between the CAN worker threads and the
Dash UI thread. Pattern adapted from ``meshcat_fk_dashboard.py``.

- :class:`RobotState` holds the latest snapshot per arm (position/velocity/
  torque/temperatures/mode) under a lock.
- :class:`DataBuffer` keeps short rolling time-series for plotting.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

ARM_DOF = 7


@dataclass
class ArmSnapshot:
    position: np.ndarray = field(default_factory=lambda: np.zeros(ARM_DOF))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(ARM_DOF))
    torque: np.ndarray = field(default_factory=lambda: np.zeros(ARM_DOF))
    tmos: np.ndarray = field(default_factory=lambda: np.full(ARM_DOF, 25.0))   # MOS temp °C
    trotor: np.ndarray = field(default_factory=lambda: np.full(ARM_DOF, 25.0))  # rotor temp °C
    mode: str = "DISABLED"
    enabled: bool = False
    timestamp: float = 0.0
    track_progress: float = 0.0   # 0..1 during TRACKING


class RobotState:
    """Latest-per-arm snapshot, plus a rolling event log. Thread-safe."""

    def __init__(self, sides=("left", "right")):
        self._lock = threading.Lock()
        self.sides = tuple(sides)
        self.arms: dict[str, ArmSnapshot] = {s: ArmSnapshot() for s in sides}
        self.can_up: dict[str, bool] = {}      # iface -> up
        self.messages: deque[str] = deque(maxlen=200)

    def update_arm(self, side: str, **fields) -> None:
        with self._lock:
            snap = self.arms[side]
            for k, v in fields.items():
                setattr(snap, k, v)
            snap.timestamp = time.time()

    def set_mode(self, side: str, mode: str, enabled: bool | None = None) -> None:
        with self._lock:
            self.arms[side].mode = mode
            if enabled is not None:
                self.arms[side].enabled = enabled

    def set_can(self, iface: str, up: bool) -> None:
        with self._lock:
            self.can_up[iface] = up

    def log(self, msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        with self._lock:
            self.messages.append(f"[{stamp}] {msg}")

    def snapshot(self, side: str) -> ArmSnapshot:
        with self._lock:
            s = self.arms[side]
            # return a shallow copy so the UI reads a stable point-in-time view
            return ArmSnapshot(
                position=s.position.copy(),
                velocity=s.velocity.copy(),
                torque=s.torque.copy(),
                tmos=s.tmos.copy(),
                trotor=s.trotor.copy(),
                mode=s.mode,
                enabled=s.enabled,
                timestamp=s.timestamp,
                track_progress=s.track_progress,
            )

    def all_can_up(self) -> bool:
        with self._lock:
            return bool(self.can_up) and all(self.can_up.values())

    def recent_messages(self, n: int = 12) -> list[str]:
        with self._lock:
            return list(self.messages)[-n:]


class DataBuffer:
    """Rolling time-series for one or more arms, for Plotly charts."""

    def __init__(self, sides=("left", "right"), maxlen: int = 400):
        self._lock = threading.Lock()
        self.sides = tuple(sides)
        self.maxlen = maxlen
        # per side: deque of (t, position[7], torque[7], tmos_mean)
        self._data: dict[str, deque] = {s: deque(maxlen=maxlen) for s in sides}
        self._t0 = time.time()

    def append(self, side: str, position: np.ndarray, torque: np.ndarray,
               tmos: np.ndarray) -> None:
        t = time.time() - self._t0
        with self._lock:
            self._data[side].append((t, position.copy(), torque.copy(), float(np.mean(tmos))))

    def series(self, side: str):
        """Return (times, positions[Nx7], torques[Nx7], tmos_mean[N]) for plotting."""
        with self._lock:
            d = list(self._data[side])
        if not d:
            return [], np.zeros((0, ARM_DOF)), np.zeros((0, ARM_DOF)), []
        times = [row[0] for row in d]
        pos = np.array([row[1] for row in d])
        tau = np.array([row[2] for row in d])
        tmos = [row[3] for row in d]
        return times, pos, tau, tmos
