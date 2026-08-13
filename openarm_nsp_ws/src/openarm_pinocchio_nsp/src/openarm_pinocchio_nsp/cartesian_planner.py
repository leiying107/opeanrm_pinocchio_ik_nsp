# Copyright 2026 Enactic, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Offline Cartesian trajectory planning via chained null-space IK.

Given a list of Cartesian waypoints (position + orientation), this densifies
them (linear position + SLERP orientation), solves IK at each sample with
warm-starting (the previous solution seeds the next), and time-parameterises
the joint path against per-joint velocity limits.

The warm-start chain is where the null-space solver earns its keep: each
sample's solution feeds the next, so the elbow swivel (the 1-DoF redundancy)
stays continuous and the σ_min profile is actively held away from singularities
along the whole path — something single-shot IK cannot do.

Pure computation (no rclpy) so it is easy to unit-test and to replay offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pinocchio as pin

from .kinematics import IKResult, PinocchioModel, SIGMA_WARN

# OpenArm v1.0 max joint velocities (rad/s), from joint_limits.yaml.
# Order: joint1..joint7. Used for time parameterisation.
DEFAULT_VMAX = np.array([16.75, 16.75, 5.45, 5.45, 20.94, 20.94, 20.94])


@dataclass
class Waypoint:
    pos: np.ndarray          # [x, y, z] metres
    quat_xyzw: np.ndarray    # [x, y, z, w]

    @classmethod
    def from_arrays(cls, pos, quat) -> "Waypoint":
        return cls(np.asarray(pos, float), np.asarray(quat, float))


@dataclass
class PlanResult:
    success: bool
    q_path: list[np.ndarray] = field(default_factory=list)   # per-sample joint angles
    times: list[float] = field(default_factory=list)         # time-from-start per sample (s)
    diagnostics: list[dict] = field(default_factory=list)    # per-sample σ_min, margin, err
    break_index: int | None = None                           # first failed sample, if any

    # go/no-go summary ----------------------------------------------------
    @property
    def min_sigma(self) -> float:
        return min((d["sigma_min"] for d in self.diagnostics), default=0.0)

    @property
    def min_margin(self) -> float:
        return min((d["joint_margin"] for d in self.diagnostics), default=0.0)

    @property
    def passed_gate(self) -> bool:
        """All samples solved, σ_min≥σ_warn everywhere, margin>0.1 rad."""
        return (
            self.success
            and self.min_sigma >= SIGMA_WARN
            and self.min_margin > 0.1
        )


# ------------------------------------------------------------------ utils
def _slerp(q0_xyzw: np.ndarray, q1_xyzw: np.ndarray, t: float) -> np.ndarray:
    """SLERP between two xyzw quaternions (Pinocchio backend, shortest arc)."""
    qa = pin.Quaternion(np.asarray(q0_xyzw, float))
    qb = pin.Quaternion(np.asarray(q1_xyzw, float))
    # ensure shortest path
    if qa.dot(qb) < 0:
        qb = pin.Quaternion(-qb.coeffs())
    qa.normalize()
    qb.normalize()
    angle = float(np.arccos(np.clip(qa.dot(qb), -1.0, 1.0)))
    if angle < 1e-6:
        return qa.coeffs().copy()
    s = np.sin(angle)
    out = (np.sin((1 - t) * angle) * qa.coeffs()
           + np.sin(t * angle) * qb.coeffs()) / s
    return out


def _slerp_quat_chain(quats: np.ndarray, s_ctrl: np.ndarray, sd: float) -> np.ndarray:
    """SLERP along control quaternions by arc-length parameter sd in [0,1]."""
    quats = np.asarray(quats, float)
    n = len(quats)
    if sd <= 0.0:
        return quats[0]
    if sd >= 1.0:
        return quats[-1]
    idx = int(np.searchsorted(s_ctrl, sd)) - 1
    idx = max(0, min(idx, n - 2))
    s0, s1 = s_ctrl[idx], s_ctrl[idx + 1]
    t = (sd - s0) / (s1 - s0) if s1 > s0 else 0.0
    return _slerp(quats[idx], quats[idx + 1], t)


def fit_arc(waypoints: list[Waypoint], n_dense: int = 100) -> list[Waypoint]:
    """Fit a smooth arc THROUGH all control points (position spline + orientation slerp).

    Position: interpolating B-spline (scipy make_interp_spline) over arc-length param.
    Orientation: piecewise SLERP along control quaternions.
    Returns ``n_dense`` Waypoints sampling the fitted arc at s∈[0,1].
    """
    from scipy.interpolate import make_interp_spline
    n = len(waypoints)
    if n < 2:
        return list(waypoints)
    pos = np.array([w.pos for w in waypoints], dtype=float)
    quat = np.array([w.quat_xyzw for w in waypoints], dtype=float)
    # drop near-duplicate control points (would create duplicate spline knots)
    keep = [0]
    for i in range(1, n):
        if float(np.linalg.norm(pos[i] - pos[keep[-1]])) > 1e-6:
            keep.append(i)
    pos, quat, n = pos[keep], quat[keep], len(keep)
    if n < 2:
        return [Waypoint(pos[0].copy(), quat[0].copy()) for _ in range(n_dense)]
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = s[i - 1] + float(np.linalg.norm(pos[i] - pos[i - 1]))
    if s[-1] < 1e-9:
        return [Waypoint(pos[0].copy(), quat[0].copy()) for _ in range(n_dense)]
    s = s / s[-1]
    k = min(3, n - 1)
    spline = make_interp_spline(s, pos, k=k, axis=0)
    sd = np.linspace(0.0, 1.0, n_dense)
    pos_d = spline(sd)
    quat_d = np.array([_slerp_quat_chain(quat, s, t) for t in sd])
    return [Waypoint(pos_d[i], quat_d[i]) for i in range(n_dense)]


def densify(
    waypoints: list[Waypoint],
    max_pos_step: float = 0.005,   # 5 mm Cartesian resolution
    max_ori_step: float = np.deg2rad(1.0),
) -> list[Waypoint]:
    """Insert intermediate samples so consecutive EE poses move in small steps."""
    if len(waypoints) == 1:
        return list(waypoints)
    out = [waypoints[0]]
    for a, b in zip(waypoints, waypoints[1:]):
        dpos = float(np.linalg.norm(b.pos - a.pos))
        dori = float(np.linalg.norm(_slerp(a.quat_xyzw, b.quat_xyzw, 1.0)
                                    - a.quat_xyzw))
        n = max(1, int(np.ceil(max(dpos / max_pos_step, dori / max_ori_step))))
        for k in range(1, n + 1):
            t = k / n
            out.append(Waypoint(
                pos=(1 - t) * a.pos + t * b.pos,
                quat_xyzw=_slerp(a.quat_xyzw, b.quat_xyzw, t),
            ))
    return out


def _se3_lerp(a, b, t):
    """Geodesic interpolation on SE(3): M(t) = a · exp(t · log(a⁻¹·b)). Pinocchio."""
    return a.act(pin.exp6(t * pin.log6(a.inverse() * b)))


def densify_se3(waypoints: list[Waypoint], n_per_seg: int = 30) -> list[Waypoint]:
    """Smooth SE(3) geodesic interpolation (Pinocchio). Couples position + orientation
    so the EE follows a geodesic — smoother than independent linear position + slerp."""
    if len(waypoints) == 1:
        return list(waypoints)
    out = []
    for a, b in zip(waypoints, waypoints[1:]):
        Ma = pin.SE3(pin.Quaternion(np.asarray(a.quat_xyzw, float)).matrix(),
                     np.asarray(a.pos, float))
        Mb = pin.SE3(pin.Quaternion(np.asarray(b.quat_xyzw, float)).matrix(),
                     np.asarray(b.pos, float))
        for k in range(n_per_seg):
            M = _se3_lerp(Ma, Mb, k / n_per_seg)
            out.append(Waypoint(M.translation.copy(),
                                np.asarray(pin.Quaternion(M.rotation).coeffs(), float)))
    out.append(waypoints[-1])
    return out


def time_parameterise(
    q_path: list[np.ndarray], vmax: np.ndarray, safety: float = 0.3
) -> list[float]:
    """Assign cumulative times so per-joint velocity ≤ vmax·safety per segment."""
    times = [0.0]
    for qa, qb in zip(q_path, q_path[1:]):
        dq = np.abs(qb - qa)
        dt = float(np.max(dq / (vmax * safety))) if len(q_path) > 1 else 0.0
        dt = max(dt, 1e-3)  # min 1 ms per sample
        times.append(times[-1] + dt)
    return times


def _spline_smooth(q_path: list[np.ndarray], times: list[float], n_dense: int):
    """Cubic-spline resample per joint -> C2 continuity (position/velocity/acceleration)."""
    from scipy.interpolate import CubicSpline
    q = np.asarray(q_path)
    t = np.asarray(times)
    cs = CubicSpline(t, q, axis=0)
    td = np.linspace(t[0], t[-1], n_dense)
    return td.tolist(), [np.asarray(row, float) for row in cs(td)]


def ease_in_out_retime(
    times: list[float],
    q_path: list[np.ndarray],
    n_dense: int | None = None,
    slowdown: float = 1.0,
    min_duration: float | None = None,
    vmax_cap: float | np.ndarray | None = None,
) -> tuple[list[float], list[np.ndarray]]:
    """Re-time a trajectory to a rest-to-rest (ease-in / ease-out) velocity profile.

    The spatial path is left unchanged; only the *timing* is reshaped so the
    joint velocity is 0 at both ends (从静止启动 + 到位降速停止) and peaks
    mid-motion. This is a pure post-processing step applied AFTER plan_cartesian,
    so the existing planner logic is untouched.

    Uses the quintic smooth-step s(τ)=10τ³−15τ⁴+6τ⁵, whose derivative
    s'(0)=s'(1)=0. Output times are UNIFORM; output positions are the original
    spatial path sampled at the eased arc parameter. Because the consumer
    (_step_track) linearly interpolates between uniform-time samples, the
    per-segment velocity tracks s'(τ) — flat at the ends, 1.875× the average
    in the middle.

    Speed control (任选其一或同时):
      slowdown     —— 整体放慢倍数 (>1 更慢)，总时长 × slowdown。
      min_duration —— 总时长下限 (s)，强制更慢。
      vmax_cap     —— 每关节转速硬限幅 (rad/s)，标量或 (7,) 数组。保证 easing
                      后的实际峰值速度 ≤ cap。因 eased 位置增量 Δq 与 T 无关、
                      只拉伸时间轴，故可闭式求满足限幅的最小时长：
                          peak_v_j = Δq_max_j · (n-1) / T ≤ cap_j
                          T ≥ Δq_max_j · (n-1) / cap_j   (取所有关节最大者)

    Returns (new_times, new_q_path).
    """
    from scipy.interpolate import CubicSpline
    t = np.asarray(times, float)
    q = np.asarray(q_path, float)
    if len(t) < 2 or t[-1] <= t[0]:
        return list(t), [np.asarray(r, float) for r in q]
    T0 = float(t[-1] - t[0])
    u = (t - t[0]) / T0                      # normalized arc parameter of the input
    cs = CubicSpline(u, q, axis=0)           # spatial path q(u), path unchanged
    n = n_dense or len(q)
    tau = np.linspace(0.0, 1.0, n)           # uniform output time parameter
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5   # smooth-step: s'(0)=s'(1)=0
    q_new = cs(s)                            # eased positions (T-independent)

    # --- total duration: base slowdown / floor, then enlarge for the velocity cap ---
    T = T0 * float(slowdown)
    if min_duration is not None:
        T = max(T, float(min_duration))
    if vmax_cap is not None:
        cap = np.asarray(vmax_cap, float)    # scalar or (7,)
        dq_max = np.abs(np.diff(q_new, axis=0)).max(axis=0)  # (7,) worst eased Δq per joint
        # peak_v_j over the uniform-time segments = dq_max_j · (n-1) / T  ≤  cap_j
        T = max(T, float(np.max(dq_max * (n - 1) / cap)))

    t_new = (t[0] + tau * T).tolist()        # UNIFORM output times over the chosen duration
    return t_new, [np.asarray(row, float) for row in q_new]


def joint_replay_traj(
    points: list[np.ndarray],
    max_speed: float,
    freq: float,
) -> tuple[list[float], list[np.ndarray]] | None:
    """Joint-space teach-and-replay trajectory through recorded configurations.

    No IK — just replays the taught joint angles, so it never hits the
    branch/singularity failures Cartesian arc IK can. Timing is set by the
    user's max-speed cap:

      per segment i:   dt_i = max_j|q[i+1]-q[i]|_j / max_speed
                       (the joint that moves the most in that segment sets the
                        pace; every other joint moves proportionally slower, so
                        NO joint ever exceeds max_speed)
      total time:      T = Σ dt_i

    Then the whole path is resampled UNIFORMLY at the control frequency
    (dt = 1/freq) via linear interpolation within each segment. Returns
    (times, q_path) ready for the TRACKING worker, or None if < 2 points.

    Note: the actual motor command rate is the worker loop (250 Hz real); `freq`
    sets how densely the path is sampled so _step_track's linear interpolation
    between samples is accurate. Pick freq ≤ 250 on real hardware.
    """
    pts = np.asarray([np.asarray(p, dtype=float).ravel() for p in points], dtype=float)
    n = len(pts)
    if n < 2 or max_speed <= 0 or freq <= 0:
        return None
    dt = 1.0 / float(freq)
    # per-segment bottleneck time (largest single-joint move / max_speed)
    seg_dt = np.abs(np.diff(pts, axis=0)).max(axis=1) / float(max_speed)
    seg_dt = np.maximum(seg_dt, dt)                 # at least one control tick per segment
    cum_t = np.concatenate([[0.0], np.cumsum(seg_dt)])
    T = float(cum_t[-1])
    # uniform output grid at the control frequency, clipped to [0, T]
    n_out = max(2, int(round(T / dt)) + 1)
    t_grid = np.clip(np.arange(n_out) * dt, 0.0, T)
    # which segment each output time falls in, and linear interpolation within it
    seg_idx = np.clip(np.searchsorted(cum_t, t_grid, side="right") - 1, 0, n - 2)
    t0, t1 = cum_t[seg_idx], cum_t[seg_idx + 1]
    frac = np.clip((t_grid - t0) / np.maximum(t1 - t0, 1e-12), 0.0, 1.0)
    q_path = pts[seg_idx] + frac[:, None] * (pts[seg_idx + 1] - pts[seg_idx])
    return t_grid.tolist(), [np.asarray(r, dtype=float) for r in q_path]


def pose_replay_traj(
    model,
    poses: list[Waypoint],
    q_seed: np.ndarray,
    max_speed: float,
    freq: float,
    null_iters: int = 6,
    fine_per_seg: int = 30,
) -> tuple[list[float], list[np.ndarray]] | None:
    """Task-space teach-replay: record 6D poses, interpolate at the control
    frequency, IK ("backward calculation") each mid-point to get joint angles.

    The END EFFECTOR follows a straight line in pose space between taught points
    (linear position + SLERP orientation); every tick's joints come from IK — so
    the path is defined by the recorded 6D pose, not by joint angles.

    Both user constraints hold simultaneously:
      • control frequency  → output ticks are UNIFORM at dt = 1/freq
      • max angle speed    → each tick advances exactly max_speed·dt of joint
        travel, so no joint exceeds max_speed

    This is done by arc-length resampling in "max-joint-move" space:
      1. Fine-sample each segment's pose (lerp pos + SLERP quat), IK each
         (warm-start chain from ``q_seed`` = current arm joints → no branch jump)
         → a dense spatial joint path q_fine.
      2. Measure cumulative max-joint-move along q_fine (the arc length that
         bounds speed).
      3. Resample q_fine at uniform arc steps of ``step = max_speed/freq`` →
         one output point per control tick. Each tick's joint move ≈ step, so
         the speed ≈ max_speed, and times are exactly 0, dt, 2dt, …

    Returns (times, q_path) ready for TRACKING, or None if any IK fails.
    """
    n = len(poses)
    if n < 2 or max_speed <= 0 or freq <= 0:
        return None
    dt = 1.0 / float(freq)
    step = max_speed * dt          # max joint move permitted in one control tick

    def _ik(wp, q_prev):
        r = model.ik_nsp(wp.pos, wp.quat_xyzw, q_init=q_prev, null_iters=null_iters)
        if not r.converged:
            r = model.ik_multi(wp.pos, wp.quat_xyzw, q_prev=q_prev)
        return r

    # 1. fine-sample the pose path + IK (warm-start) → dense joint path q_fine
    q_prev = np.asarray(q_seed, dtype=float)
    q_fine: list[np.ndarray] = []
    for seg in range(n - 1):
        A, B = poses[seg], poses[seg + 1]
        k0 = 0 if seg == 0 else 1           # skip duplicate segment boundary
        for k in range(k0, fine_per_seg + 1):
            f = k / fine_per_seg
            pos = (1.0 - f) * A.pos + f * B.pos
            quat = _slerp(A.quat_xyzw, B.quat_xyzw, f)
            r = _ik(Waypoint(pos, quat), q_prev)
            if not r.converged:
                return None
            q_fine.append(r.q)
            q_prev = r.q
    q_fine = np.asarray(q_fine)            # (M, 7)

    # 2. cumulative max-joint-move arc length
    dq_seg = np.max(np.abs(np.diff(q_fine, axis=0)), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(dq_seg)])
    total = float(arc[-1])
    if total <= 0:
        return None

    # 3. resample at uniform arc step → uniform-time ticks, each ≤ step joint move
    n_out = max(2, int(np.ceil(total / step)) + 1)
    target = np.minimum(np.arange(n_out) * step, total)
    idx = np.clip(np.searchsorted(arc, target, side="right") - 1, 0, len(arc) - 2)
    a0, a1 = arc[idx], arc[idx + 1]
    frac = np.clip((target - a0) / np.maximum(a1 - a0, 1e-12), 0.0, 1.0)
    q_path = q_fine[idx] + frac[:, None] * (q_fine[idx + 1] - q_fine[idx])
    q_path[-1] = q_fine[-1]                # snap final tick to exact end config
    times = (np.arange(n_out) * dt).tolist()
    return times, [np.asarray(r, dtype=float) for r in q_path]


@dataclass
class TrajCheck:
    """Warn-only smoothness/continuity report for a solved joint trajectory.

    Never blocks execution — callers just log ``warnings``. Designed to catch
    IK branch jumps (the main failure mode of warm-start chains), which show up
    as one segment whose step is a large outlier vs the rest.
    """
    warnings: list[str] = field(default_factory=list)
    max_step: float = 0.0          # largest adjacent |Δq| (rad), any joint
    max_velocity: float = 0.0      # largest joint speed (rad/s)
    jump_index: int | None = None  # segment index of the worst outlier step


def check_traj_smoothness(times, q_path, vmax: np.ndarray = DEFAULT_VMAX) -> TrajCheck:
    """Warn-only continuity/smoothness check on a (times, q_path) trajectory.

    Detects, WITHOUT blocking:
      1. IK branch jumps — a single adjacent step that's a big outlier vs the
         step distribution (MAD z-score). The classic warm-start-chain failure.
      2. Absolute large steps (>0.5 rad between consecutive points) → real-arm jerk.
      3. Any joint exceeding its velocity limit (speed/vmax > 90%).
    """
    q = np.asarray(q_path, dtype=float)
    t = np.asarray(times, dtype=float)
    rep = TrajCheck()
    if len(q) < 3 or len(t) < 2:
        return rep
    step = np.abs(np.diff(q, axis=0)).max(axis=1)        # (N-1,) max-joint step/segment
    dt = np.diff(t)
    rep.max_step = float(step.max())
    rep.max_velocity = float((step / np.maximum(dt, 1e-9)).max())

    # 1. branch-jump outlier (robust z-score on the step distribution)
    med = float(np.median(step))
    mad = float(np.median(np.abs(step - med))) + 1e-9
    z = np.abs(step - med) / mad
    ji = int(np.argmax(z))
    if z[ji] > 6.0 and step[ji] > 0.15:
        rep.jump_index = ji
        rep.warnings.append(
            f"⚠ 第{ji}→{ji+1}步关节跳变 {step[ji]:.3f}rad (中位{med:.4f}, 离群z={min(z[ji],99):.0f})，"
            f"疑似IK分支跳变→实机可能抖动")

    # 2. absolute large single step
    if rep.max_step > 0.5:
        k = int(np.argmax(step))
        rep.warnings.append(
            f"⚠ 相邻步长过大 {rep.max_step:.3f}rad @{k}→{k+1} (>0.5rad)，轨迹不平滑")

    # 3. per-joint velocity vs its limit
    v_per = np.abs(np.diff(q, axis=0)) / np.maximum(dt, 1e-9)[:, None]   # (N-1,7)
    ratio = v_per / np.asarray(vmax, dtype=float)[None, :]
    wr, wj = int(np.argmax(ratio)), 0
    if ratio.size:
        wr = int(np.unravel_index(np.argmax(ratio), ratio.shape)[1])
    if ratio.max() > 0.9:
        rep.warnings.append(
            f"⚠ 关节{wr+1}峰值速度 {v_per[:,wr].max():.2f}rad/s "
            f"达其vmax({vmax[wr]:.2f})的{ratio.max()*100:.0f}%")
    return rep


# ------------------------------------------------------------- main solver
def plan_cartesian(
    model: PinocchioModel,
    waypoints: list[Waypoint],
    q_init: np.ndarray,
    *,
    max_pos_step: float = 0.005,
    max_ori_step: float = np.deg2rad(1.0),
    vmax: np.ndarray = DEFAULT_VMAX,
    vel_safety: float = 0.3,
    n_random: int = 5,
    null_iters: int = 6,
    smooth: bool = True,
    n_dense: int = 80,
    presampled: bool = False,
) -> PlanResult:
    """Plan a smooth joint trajectory following the Cartesian waypoints.

    Two-layer smoothing:
      1. Cartesian: SE(3) geodesic interpolation between waypoints (Pinocchio).
      2. Joint: cubic-spline resample of IK solutions -> C2 continuity
         (position/velocity/acceleration), so motion is smooth, not piecewise.
    Set ``smooth=False`` for the old linear-densify behaviour.
    Set ``presampled=True`` when waypoints are already a dense arc (e.g. from
    ``fit_arc``); the SE(3)/linear densify is skipped.
    """
    if presampled:
        samples = list(waypoints)
    elif smooth:
        samples = densify_se3(waypoints)
    else:
        samples = densify(waypoints, max_pos_step, max_ori_step)
    q_prev = np.asarray(q_init, float)
    q_path: list[np.ndarray] = []
    diags: list[dict] = []

    for i, wp in enumerate(samples):
        # Along a chain the previous solution warm-starts the next, so a single
        # ik_nsp is enough (and ~10x faster than multi-seed). Multi-seed is only
        # a fallback for the rare sample where warm-start diverges.
        r: IKResult = model.ik_nsp(
            wp.pos, wp.quat_xyzw, q_init=q_prev, null_iters=null_iters
        )
        if not r.converged:
            r = model.ik_multi(
                wp.pos, wp.quat_xyzw, q_prev=q_prev, n_random=n_random
            )
        if not r.converged:
            return PlanResult(
                success=False, q_path=q_path, diagnostics=diags, break_index=i
            )
        q_path.append(r.q)
        diags.append({
            "sigma_min": r.sigma_min,
            "joint_margin": r.joint_margin,
            "pos_err_mm": r.pos_err_mm,
            "ori_err_deg": r.ori_err_deg,
        })
        q_prev = r.q  # warm-start the next sample

    times = time_parameterise(q_path, vmax, vel_safety)
    if smooth and len(q_path) > 3:
        # cubic-spline resample -> C2 continuous joint trajectory
        times, q_path = _spline_smooth(q_path, times, n_dense)
        diags = [{
            "sigma_min": float(model.singular_values(q)[-1]),
            "joint_margin": model.joint_margin(q),
            "pos_err_mm": 0.0,
            "ori_err_deg": 0.0,
        } for q in q_path]
    return PlanResult(success=True, q_path=q_path, times=times, diagnostics=diags)
