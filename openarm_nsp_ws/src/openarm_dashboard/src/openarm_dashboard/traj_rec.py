# Copyright 2026 Enactic, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Trajectory record / persist / retime for impedance playback (traj playback).

Pipeline: hand-drag the arm in zero-torque → TrajRecorder samples joint
angles at 50 Hz → on stop: smoothing filter + uniform resample + head/tail
stillness trim → TrajData (times + q[N][7]) → JSON file under
log/trajectories/ → replay retimes the clock by a rate slider with quintic
ease-in/out at both ends. The controller consumes q_ref(t) and moves the
impedance anchor along it (arm_controller IMP_TRACK); nothing here touches
the control law.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

REC_DT = 0.02            # 50 Hz sample period (every 5th 250Hz tick)
SMOOTH_WIN_S = 0.15      # moving-average window for the raw drag samples
TRIM_S = 0.25            # head/tail stillness trim threshold window
TRIM_QRAD = 0.01         # a sample is "still" if |Δq| vs window start < this
MAX_POINTS = 6000        # 120 s at 50 Hz — recording hard cap
PLAY_EASE_S = 1.0        # quintic ease-in/out duration at each end of replay

TRAJ_DIR = Path(__file__).resolve().parents[3] / "log" / "trajectories"


class TrajRecorder:
    """Collects 50 Hz joint samples while the user drags the arm."""

    def __init__(self):
        self.buf: list[np.ndarray] = []
        self.t0 = 0.0
        self.recording = False

    def start(self, q: np.ndarray) -> None:
        self.buf = [np.asarray(q, float).copy()]
        self.t0 = time.time()
        self.recording = True

    def sample(self, q: np.ndarray) -> None:
        if self.recording and len(self.buf) < MAX_POINTS:
            self.buf.append(np.asarray(q, float).copy())

    def stop(self) -> "TrajData":
        """Finish recording and post-process into a TrajData."""
        self.recording = False
        raw = np.array(self.buf)
        self.buf = []
        return TrajData.from_raw(raw, REC_DT)


class TrajData:
    """A smoothed, uniformly-sampled joint trajectory (times in seconds)."""

    def __init__(self, times: np.ndarray, q: np.ndarray, side: str = ""):
        self.times = np.asarray(times, float)
        self.q = np.asarray(q, float)
        self.side = side

    # ------------------------------------------------------------ build
    @staticmethod
    def from_raw(raw: np.ndarray, dt: float) -> "TrajData":
        """Post-process raw drag samples: smooth → resample → trim."""
        if len(raw) < 10:
            raise ValueError(f"录制太短（{len(raw)} 个样本，至少需要 10 个）")
        # moving-average filter (window spans SMOOTH_WIN_S, min 3 taps)
        win = max(3, int(round(SMOOTH_WIN_S / dt)) | 1)
        ker = np.ones(win) / win
        qf = np.stack([np.convolve(raw[:, j], ker, mode="same") for j in
                       range(raw.shape[1])], axis=1)
        # "same"-convolution edge ramps: drop one window at each end
        edge = win // 2
        qf = qf[edge:-edge]
        # uniform time base (the raw cadence is near-uniform already; this
        # makes interpolation exact regardless of jitter)
        t_raw = np.arange(len(qf)) * dt
        t_new = np.arange(0, t_raw[-1] + 1e-9, dt)
        qs = np.stack([np.interp(t_new, t_raw, qf[:, j]) for j in
                       range(qf.shape[1])], axis=1)
        qs = _trim_still(qs, dt)
        if len(qs) < 10:
            raise ValueError("裁剪静止段后轨迹太短，录一段更长的动作")
        return TrajData(np.arange(len(qs)) * dt, qs)

    # ------------------------------------------------------------- io
    def save(self, name: str = "", side: str = "") -> Path:
        TRAJ_DIR.mkdir(parents=True, exist_ok=True)
        stem = time.strftime("traj_%Y%m%d_%H%M%S")
        if name:
            safe = "".join(c for c in name if c.isalnum() or c in "-_")
            if safe:
                stem += f"_{safe}"
        path = TRAJ_DIR / f"{stem}.json"
        data = {
            "version": 1,
            "side": side or self.side,
            "dt": float(self.times[1] - self.times[0]) if len(self.times) > 1 else REC_DT,
            "n": int(len(self.times)),
            "duration": float(self.times[-1]),
            "q": [[round(float(v), 6) for v in row] for row in self.q],
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    @staticmethod
    def load(path: str | Path) -> "TrajData":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        if obj.get("version") != 1:
            raise ValueError(f"未知轨迹文件版本: {obj.get('version')}")
        q = np.array(obj["q"], float)
        if q.ndim != 2 or q.shape[1] != 7:
            raise ValueError("轨迹文件关节维度不对（应为 N×7）")
        dt = float(obj["dt"])
        return TrajData(np.arange(len(q)) * dt, q, side=obj.get("side", ""))

    @staticmethod
    def list_files() -> list[dict]:
        if not TRAJ_DIR.exists():
            return []
        out = []
        for p in sorted(TRAJ_DIR.glob("traj_*.json"), reverse=True):
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                out.append({"name": p.name, "duration": round(float(obj.get("duration", 0)), 1),
                            "n": int(obj.get("n", 0)), "side": obj.get("side", "")})
            except (ValueError, OSError):
                continue
        return out

    # ---------------------------------------------------------- replay
    def sample_at(self, t: float) -> np.ndarray:
        """Linear-interpolated q at replay time t (clamped at both ends)."""
        i1 = int(np.searchsorted(self.times, t))
        if i1 <= 0:
            return self.q[0].copy()
        if i1 >= len(self.times):
            return self.q[-1].copy()
        i0 = i1 - 1
        f = (t - self.times[i0]) / (self.times[i1] - self.times[i0])
        return self.q[i0] + f * (self.q[i1] - self.q[i0])

    def play_duration(self, rate: float) -> float:
        """Wall-clock duration of a replay at the given rate multiplier."""
        return float(self.times[-1]) / max(rate, 0.05)


def _trim_still(q: np.ndarray, dt: float) -> np.ndarray:
    """Drop leading/trailing windows where the arm was not yet / no longer
    moving (the user needs a moment to grab and release the arm)."""
    win = max(2, int(round(TRIM_S / dt)))
    speed = np.max(np.abs(np.diff(q, axis=0)), axis=1)      # per-sample joint speed
    pad = np.concatenate([speed[:1], speed, speed[-1:]])    # keep alignment
    moving = pad > TRIM_QRAD / dt * 0.5                     # ~half-threshold
    i0, i1 = 0, len(q)
    for i in range(len(q) - win):
        if moving[i:i + win].any():
            i0 = i
            break
    for i in range(len(q) - 1, win, -1):
        if moving[max(0, i - win):i].any():
            i1 = i
            break
    return q[i0:max(i1 + 1, i0 + 1)]


def ease_rate(t_play: float, dur: float, rate: float) -> float:
    """Instantaneous replay-rate FACTOR (multiplier on `rate`), quintic-eased
    over PLAY_EASE_S of TRAJECTORY time at both ends (speed bump
    s'(τ)=30τ²(1−τ)²/1.875). A 15% floor keeps the integrated clock moving
    at the very start/end — the pure bump is exactly 0 at τ=0, which
    deadlocks a clock whose rate is its own integral (found in sim R2).
    The ease window is in TRAJECTORY seconds, so slow rates (0.2x) stretch
    the wall-time ease proportionally — by design: a slow replay should
    also start/stop slowly."""
    e = min(PLAY_EASE_S, dur / 4.0)
    if t_play < e:
        s = t_play / e
    elif t_play > dur - e:
        s = max(0.0, (dur - t_play) / e)
    else:
        return rate
    f = 30 * s ** 2 * (1 - s) ** 2 / 1.875
    return rate * max(f, 0.15)
