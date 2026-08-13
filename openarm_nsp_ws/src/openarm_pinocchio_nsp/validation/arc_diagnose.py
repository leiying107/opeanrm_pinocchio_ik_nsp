#!/usr/bin/env python3
# Copyright 2026 Enactic, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnose why an arc trajectory fails the safety gate.

Shows, for each taught control point AND each point along the fitted arc:
  * σ_min (singularity distance) and joint margin, for THREE paths:
    1. the taught joint angles themselves (what your hand dragged to)
    2. joint-space linear interpolation between taught points
    3. the Cartesian arc (fit_arc + IK) that plan_cartesian produces
  * exactly which samples fail the gate (σ_min<0.05 or margin≤0.1)

This reveals whether the failure is: bad taught poses, the Cartesian arc
crossing a danger zone, or IK picking a worse branch than your hand.

Run:
    python arc_diagnose.py            # built-in test arc
    python arc_diagnose.py --deg "0,0,0,114,0,0,0" "10,0,0,110,0,5,0" "20,0,0,100,0,10,0"
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from openarm_pinocchio_nsp.cartesian_planner import Waypoint, fit_arc, plan_cartesian
from openarm_pinocchio_nsp.kinematics import PinocchioModel, SIGMA_WARN
from openarm_pinocchio_nsp.urdf_path import resolve_urdf_path

GATE_MARGIN = 0.1


def _metrics(model: PinocchioModel, q):
    sv = model.singular_values(q)
    return float(sv[-1]), model.joint_margin(q)


def _flag(smin, margin):
    bad = []
    if smin < SIGMA_WARN:
        bad.append(f"σ_min={smin:.3f}<0.05奇异")
    if margin <= GATE_MARGIN:
        bad.append(f"裕度={margin:.3f}≤0.1限位")
    return "  ✗ " + "; ".join(bad) if bad else "  ✓"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="right")
    ap.add_argument("--deg", nargs="+", default=None,
                    help="control points as degree lists, e.g. '0,0,0,114,0,0,0' '10,...'")
    args = ap.parse_args()

    model = PinocchioModel(resolve_urdf_path(), args.side)

    if args.deg:
        pts_deg = [[float(x) for x in d.split(",")] for d in args.deg]
    else:
        pts_deg = [[0, 0, 0, 114, 0, 0, 0], [10, 0, 0, 110, 0, 5, 0],
                   [20, 0, 0, 100, 0, 10, 0], [15, -10, 0, 95, 0, 15, 5],
                   [5, -15, 0, 100, 0, 20, 0]]
    taught = [np.clip(np.deg2rad(p), model.lower, model.upper) for p in pts_deg]

    print("=" * 70)
    print(f"弧线诊断 — {len(taught)} 个示教控制点 ({args.side} 臂)")
    print("=" * 70)

    # --- 1. taught poses themselves (what your hand dragged to) ---
    print("\n① 示教点本身（你手拖到的关节角，FK 后的指标）:")
    for i, q in enumerate(taught):
        smin, mar = _metrics(model, q)
        print(f"  P{i}: σ_min={smin:.3f}  裕度={mar:.3f}{_flag(smin, mar)}")

    # --- 2. joint-space linear interpolation (no gate, like move_to) ---
    print("\n② 关节空间线性插值（[到终点]走的路径，无gate）:")
    n = 20
    for i in range(len(taught) - 1):
        for k in range(n):
            t = k / n
            q = taught[i] + t * (taught[i + 1] - taught[i])
            smin, mar = _metrics(model, q)
            if smin < SIGMA_WARN or mar <= GATE_MARGIN:
                print(f"  段{i}->{i+1} t={t:.2f}: σ_min={smin:.3f} 裕度={mar:.3f}{_flag(smin,mar)}")
    print("  (只列不安全点；关节插值不检查gate所以照跑)")

    # --- 3. Cartesian arc (fit_arc + IK, what [弧线IK] does) ---
    print("\n③ 笛卡尔弧线（fit_arc+IK，[弧线IK]走的路径，检查gate）:")
    wps = [Waypoint(*model.fk(q)) for q in taught]
    arc = fit_arc(wps, n_dense=100)
    res = plan_cartesian(model, arc, taught[0], presampled=True)
    if not res.success:
        print(f"  IK 失败 @sample {res.break_index}（无解/不可达）")
        return 1
    q_arc = np.array(res.q_path)
    fails = 0
    worst_smin, worst_mar = 1e9, 1e9
    for i, q in enumerate(q_arc):
        smin, mar = _metrics(model, q)
        worst_smin = min(worst_smin, smin)
        worst_mar = min(worst_mar, mar)
        if smin < SIGMA_WARN or mar <= GATE_MARGIN:
            fails += 1
            if fails <= 8:
                print(f"  sample {i:3d}: σ_min={smin:.3f} 裕度={mar:.3f}{_flag(smin,mar)}")
    if fails > 8:
        print(f"  ... 共 {fails} 个不安全点")
    print(f"\n  汇总: {fails}/{len(q_arc)} 点不安全 | 最差 σ_min={worst_smin:.3f} 最差裕度={worst_mar:.3f}")
    print(f"  gate: {'PASS ✅' if res.passed_gate else 'FAIL ❌'}")

    # --- diagnosis ---
    print("\n④ 诊断:")
    taught_smin = min(_metrics(model, q)[0] for q in taught)
    taught_mar = min(_metrics(model, q)[1] for q in taught)
    if taught_mar <= GATE_MARGIN:
        print(f"  → 你的示教点本身裕度就 ≤0.1（{taught_mar:.3f}），靠近限位。")
        print(f"    关节插值不检查所以能跑；笛卡尔gate拦截。建议:示教点选关节中部的姿态。")
    elif worst_smin < SIGMA_WARN:
        print(f"  → 弧线中间点经过奇异区(σ_min低至{worst_smin:.3f})，但你示教点本身OK({taught_smin:.3f})。")
        print(f"    笛卡尔弧线穿过了奇异区，关节插值没穿过。建议:调整中间点让弧线绕开奇异。")
    elif fails == 0:
        print(f"  → 实际上gate是PASS的！如果实机还失败，检查别的环节。")
    else:
        print(f"  → IK在弧线上选了比你手拖更差的构型(裕度{worst_mar:.3f} vs 示教{taught_mar:.3f})。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
