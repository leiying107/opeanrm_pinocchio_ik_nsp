# Copyright 2026 Enactic, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive self-collision simulation suite.

Groups (each prints a PASS/FAIL line and appends to the summary):
  A. Checker calibration    — known poses (home/safe/folded) & boundary sweeps
  B. Random sweep           — hundreds of random within-limit bimanual poses;
                              no FALSE positives on clearly-separated poses,
                              reliable detection on folded/intruding ones
  C. Trajectory checking    — clean vs colliding paths, first-violation index
  D. Collision-aware IK     — elbow migration on tight-clearance targets
  E. Detour planning        — mid-path obstacle: route around, endpoint exact,
                              truncation when target itself is blocked
  F. Bimanual coordinated   — 14-D B-spline keeps clearance through a
                              center-convergence motion

Run:
    source ../../venv-openarm-ik/bin/activate
    python v_collision_suite.py [--quick] [--group A]
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, "/ros2_ws/openarm_nsp_ws/src/openarm_pinocchio_nsp/src")

from openarm_pinocchio_nsp.collision import BimanualCollisionChecker  # noqa: E402
from openarm_pinocchio_nsp.collision_planner import CollisionAwarePlanner  # noqa: E402
from openarm_pinocchio_nsp.kinematics import PinocchioModel  # noqa: E402
from openarm_pinocchio_nsp.urdf_path import resolve_urdf_path  # noqa: E402
from openarm_pinocchio_nsp.cartesian_planner import plan_cartesian, Waypoint  # noqa: E402

MARGIN = 0.02
results: list[tuple[str, bool, str]] = []


def report(name: str, ok: bool, detail: str):
    results.append((name, ok, detail))
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}  {name:38s} {detail}")


# ------------------------------------------------------------------ Group A
def group_a(chk: BimanualCollisionChecker):
    print("\n=== Group A: checker calibration ===")
    q_home = chk.make_q()
    rep = chk.check(q_home, MARGIN)
    report("A1 home pose clear @20mm", not rep.in_collision, str(rep))

    q_fold = chk.make_q(left=np.array([0, 0.1745, 0, 0, 0, 0, 0]),
                        right=np.array([0, -0.1745, 0, 0, 0, 0, 0]))
    rep = chk.check(q_fold, MARGIN)
    report("A2 folded-shoulder detected", rep.in_collision, str(rep))

    q_safe = chk.make_q(left=np.array([0, -0.5, 0, 0.8, 0, 0, 0]),
                        right=np.array([0, 0.5, 0, 0.8, 0, 0, 0]))
    rep = chk.check(q_safe, MARGIN)
    d, a, b = chk.min_distance(q_safe)
    report("A3 safety-doc pose clear", not rep.in_collision,
           f"min {d*1000:.0f}mm ({a.replace('openarm_','')}↔{b.replace('openarm_','')})")

    # hard collision: verified torso-intruding pose (random-searched, margin=0)
    q_hard = chk.make_q(left=np.array([0.28, 0.03, 1.43, 2.08, 1.21, 0.29, 0.16]))
    rep = chk.check(q_hard, 0.0)
    report("A4 torso-intrusion hard collision", rep.in_collision, str(rep))

    # boundary: shoulder j2 sweep — clearance must go from clear → collision
    lo = np.array([chk.model.lowerPositionLimit[i] for i in chk.q_idx["left"]])
    hi = np.array([chk.model.upperPositionLimit[i] for i in chk.q_idx["left"]])
    rng = np.random.default_rng(0)
    n_col = 0
    for _ in range(40):
        qr = rng.uniform(lo, hi)
        q = chk.make_q(left=np.array([0, 0.17, 0, rng.uniform(0, 2.4), 0, 0, 0]),
                       right=qr)
        if chk.check(q, MARGIN).in_collision:
            n_col += 1
    report("A5 shoulder-at-limit risky", n_col > 0,
           f"{n_col}/40 folded-left poses violate")


# ------------------------------------------------------------------ Group B
def group_b(chk: BimanualCollisionChecker, n: int):
    print(f"\n=== Group B: random sweep ({n} poses) ===")
    rng = np.random.default_rng(42)
    lo_l = np.array([chk.model.lowerPositionLimit[i] for i in chk.q_idx["left"]])
    hi_l = np.array([chk.model.upperPositionLimit[i] for i in chk.q_idx["left"]])
    lo_r, hi_r = -hi_l, -lo_l  # right mirrors left

    n_colliding = 0
    t0 = time.time()
    for _ in range(n):
        ql = rng.uniform(lo_l, hi_l)
        qr = rng.uniform(lo_r, hi_r)
        if chk.check(chk.make_q(left=ql, right=qr), MARGIN).in_collision:
            n_colliding += 1
    dt = time.time() - t0
    print(f"  {n} random poses: {n_colliding} colliding "
          f"({100*n_colliding/n:.1f}%), {dt/n*1000:.2f} ms/check")
    report("B1 detection rate sane", 0 < n_colliding < n,
           f"{n_colliding}/{n}")
    report("B2 check speed", dt / n < 0.01, f"{dt/n*1000:.2f} ms/check")

    # symmetric arms-out poses must ALWAYS be clear
    n_bad = 0
    for _ in range(60):
        j4 = rng.uniform(0.4, 2.2)
        j6 = rng.uniform(-0.7, 0.7)
        ql = np.array([0, -0.6, 0, j4, 0, j6, 0])
        qr = np.array([0, 0.6, 0, j4, 0, j6, 0])
        if chk.check(chk.make_q(left=ql, right=qr), MARGIN).in_collision:
            n_bad += 1
    report("B3 arms-out symmetric false positives", n_bad == 0,
           f"{n_bad}/60 flagged")


# ------------------------------------------------------------------ Group C
def group_c(chk: BimanualCollisionChecker):
    print("\n=== Group C: trajectory checking ===")
    m = PinocchioModel(resolve_urdf_path(), "left")
    q0 = np.array([0, -0.5, 0, 0.8, 0, 0, 0])
    p0, quat0 = m.fk(q0)
    qr = np.load("/tmp/conflict_obs4.npy") if __import__("os").path.exists(
        "/tmp/conflict_obs4.npy") else np.array([0.85, -0.02, -1.0, 1.33, -0.93, 0.63, 0.81])

    res = plan_cartesian(m, [Waypoint(p0, quat0),
                             Waypoint(np.array([0.276, 0.229, 0.608]), quat0)],
                         q_init=q0, smooth=False)
    viol = chk.check_trajectory([chk.make_q(left=q, right=qr) for q in res.q_path],
                                margin=MARGIN)
    report("C1 conflict path flagged", viol is not None, f"first viol {viol}/{len(res.q_path)}")

    # clean path: same motion, right arm parked far
    qr_safe = np.array([0, 0.5, 0, 0.8, 0, 0, 0])
    viol2 = chk.check_trajectory([chk.make_q(left=q, right=qr_safe) for q in res.q_path],
                                 margin=MARGIN)
    report("C2 same path clear vs parked arm", viol2 is None, f"viol={viol2}")


# ------------------------------------------------------------------ Group D
def group_d(chk: BimanualCollisionChecker):
    print("\n=== Group D: collision-aware IK ===")
    from openarm_pinocchio_nsp.collision_ik import CollisionAwareIK
    m = PinocchioModel(resolve_urdf_path(), "left")
    q0 = np.array([0, -0.5, 0, 0.8, 0, 0, 0])
    p0, quat0 = m.fk(q0)
    qr = np.array([0.85, -0.02, -1.0, 1.33, -0.93, 0.63, 0.81])

    cik = CollisionAwareIK("left", other_q7=qr, checker=chk)
    # near-miss target: naive converges tight; aware should restore comfort
    n_improved = 0
    n_tot = 0
    for dy in (0.12, 0.16, 0.20, 0.24):
        pos_e = np.array([0.276, 0.03 + dy, 0.608])
        r0 = m.ik_nsp(pos_e, quat0, q_init=q0)
        if not r0.converged:
            continue
        n_tot += 1
        d0, _, _ = chk.min_distance(chk.make_q(left=r0.q, right=qr))
        r1, _ = cik.ik(pos_e, quat0, q_init=q0)
        d1, _, _ = chk.min_distance(chk.make_q(left=r1.q, right=qr))
        if d1 >= d0 - 1e-3 and r1.pos_err_mm < 2.0:
            n_improved += 1
        print(f"    dy={dy:.2f}: naive {d0*1000:5.1f}mm → aware {d1*1000:5.1f}mm "
              f"(err {r1.pos_err_mm:.2f}mm)")
    report("D1 elbow migration keeps/raises clearance", n_improved == n_tot,
           f"{n_improved}/{n_tot}")


# ------------------------------------------------------------------ Group E
def group_e(chk: BimanualCollisionChecker):
    print("\n=== Group E: detour planning ===")
    m = PinocchioModel(resolve_urdf_path(), "left")
    q0 = np.array([0, -0.5, 0, 0.8, 0, 0, 0])
    p0, quat0 = m.fk(q0)
    qr = np.array([0.85, -0.02, -1.0, 1.33, -0.93, 0.63, 0.81])
    pos_e = np.array([0.276, 0.229, 0.608])

    pl = CollisionAwarePlanner("left", other_q7=qr, checker=chk)
    t0 = time.time()
    res = pl.plan_line_quat(p0, quat0, pos_e, q0, smooth=False)
    dt = time.time() - t0
    pos_f, _ = m.fk(res.q_path[-1])
    err = np.linalg.norm(pos_f - pos_e) * 1000
    viol = chk.check_trajectory([chk.make_q(left=q, right=qr) for q in res.q_path],
                                margin=MARGIN)
    report("E1 detour found & clean", res.ok and viol is None,
           f"{res.strategy}, {len(res)}pts, endpoint {err:.2f}mm, {dt:.1f}s")

    # unreachable target (inside obstacle cluster): must truncate, not collide
    pos_bad = np.array([0.276, 0.0, 0.608])  # at the obstacle EE
    res2 = pl.plan_line_quat(p0, quat0, pos_bad, q0, smooth=False)
    viol2 = chk.check_trajectory(
        [chk.make_q(left=q, right=qr) for q in res2.q_path], margin=MARGIN)
    report("E2 blocked target truncates safely",
           viol2 is None and (res2.truncated or not res2.ok),
           f"{res2.strategy}, viol={viol2}, {len(res2)}pts")


# ------------------------------------------------------------------ Group F
def group_f(chk: BimanualCollisionChecker, quick: bool):
    print("\n=== Group F: bimanual 14-D coordinated planning ===")
    from openarm_pinocchio_nsp.bimanual_planner import BimanualBSplinePlanner
    bp = BimanualBSplinePlanner(checker=chk)
    q0l = np.array([0, -0.5, 0, 0.8, 0, 0, 0])
    q0r = np.array([0, 0.5, 0, 0.8, 0, 0, 0])
    # both arms converge toward center-front; 240 mm final EE gap (a closer
    # hand-off physically overlaps the wrists: cross-arm pairs collide at any
    # gap ≤ 200 mm with these quats — see sweep in the dev log)
    yg = 0.24
    wl = np.array([[0.15, 0.33, 0.37], [0.20, 0.18, 0.45],
                   [0.24, yg/2 + 0.02, 0.50], [0.20, yg/2, 0.52]])
    wr = np.array([[0.15, -0.33, 0.37], [0.20, -0.18, 0.45],
                   [0.24, -yg/2 - 0.02, 0.50], [0.20, -yg/2, 0.52]])
    t0 = time.time()
    times, Q, res = bp.plan(wl, wr, q0l, q0r, duration=4.0,
                            n_samples=16 if quick else 24, M=8,
                            maxiter=25 if quick else 50,
                            time_budget=120 if quick else 420)
    dt = time.time() - t0
    n_col = sum(1 for q14 in Q if bp.collision_penalty(q14) > 0)
    q16_last, _, _ = bp.split(Q[-1])
    el = bp.ee_pos(q16_last, "left")
    er = bp.ee_pos(q16_last, "right")
    track_err = max(np.linalg.norm(el - wl[-1]), np.linalg.norm(er - wr[-1])) * 1000
    report("F1 center-convergence collision-free", n_col == 0,
           f"{n_col}/{len(Q)} violating, opt {dt:.0f}s, nit={res.nit}")
    report("F2 endpoint tracking", track_err < 25,
           f"max EE err {track_err:.1f}mm")


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--group", default="ABCDEF")
    args = ap.parse_args()

    t_all = time.time()
    print("building checker (URDF + 21 geoms + pair set)...")
    chk = BimanualCollisionChecker()
    print(f"  pairs={chk.n_pairs} cats={chk.cat_counts}")

    if "A" in args.group:
        group_a(chk)
    if "B" in args.group:
        group_b(chk, 80 if args.quick else 300)
    if "C" in args.group:
        group_c(chk)
    if "D" in args.group:
        group_d(chk)
    if "E" in args.group:
        group_e(chk)
    if "F" in args.group:
        group_f(chk, args.quick)

    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n================ SUMMARY ================")
    for name, ok, detail in results:
        print(f"  {'✅' if ok else '❌'} {name:44s} {detail}")
    print(f"  {n_pass}/{len(results)} passed in {time.time()-t_all:.1f}s")
    sys.exit(0 if n_pass == len(results) else 1)


if __name__ == "__main__":
    main()
