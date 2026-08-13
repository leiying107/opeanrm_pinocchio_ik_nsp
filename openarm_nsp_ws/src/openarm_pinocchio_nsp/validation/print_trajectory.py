#!/usr/bin/env python3
# Copyright 2026 Enactic, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Print a planned Cartesian trajectory as ASCII sparklines + an optional PNG.

Visualises, per sample along the path:
  * EE position (x, y, z) and orientation
  * all 7 joint angles
  * σ_min (singularity distance) and joint margin
  * the EE path in the XY plane

Example::
    python print_trajectory.py --side right --line "0.22,-0.04,0.51 -> 0.34,-0.04,0.51"
    python print_trajectory.py --side right --circle --save traj.png
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pinocchio as pin

from openarm_pinocchio_nsp.cartesian_planner import Waypoint, plan_cartesian
from openarm_pinocchio_nsp.kinematics import PinocchioModel, SIGMA_WARN
from openarm_pinocchio_nsp.urdf_path import resolve_urdf_path

_BARS = " ▁▂▃▄▅▆▇█"


def spark(seq, width=48, lo=None, hi=None):
    """Render a 1-D sequence as a unicode sparkline bar."""
    a = np.asarray(seq, float)
    lo = a.min() if lo is None else lo
    hi = a.max() if hi is None else hi
    if hi - lo < 1e-12:
        hi = lo + 1.0
    # resample to `width` samples
    idx = np.linspace(0, len(a) - 1, width).astype(int)
    norm = (a[idx] - lo) / (hi - lo)
    return "".join(_BARS[min(8, int(v * 8.999))] for v in norm)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="right")
    ap.add_argument("--urdf", default=None)
    ap.add_argument("--line", default=None, help="'x,y,z -> x,y,z'")
    ap.add_argument("--circle", action="store_true", help="horizontal circle r=6cm")
    ap.add_argument("--radius", type=float, default=0.06)
    ap.add_argument("--save", default=None, help="save matplotlib PNG to this path")
    args = ap.parse_args()

    model = PinocchioModel(resolve_urdf_path(args.urdf), args.side)
    q0 = np.clip(np.array([0, 0, 0, 2.0, 0, 0, 0.0]), model.lower, model.upper)
    p0, quat0 = model.fk(q0)

    if args.circle:
        c = p0 + np.array([0.05, 0, 0])
        wps = [Waypoint(c + args.radius * np.array([np.cos(a), np.sin(a), 0]),
                        quat0) for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
    elif args.line:
        a, b = [np.array([float(v) for v in p.split(",")]) for p in args.line.split("->")]
        wps = [Waypoint(a, quat0), Waypoint(b, quat0)]
    else:
        ap.error("specify --line or --circle")

    res = plan_cartesian(model, wps, q0)
    if not res.success:
        print(f"plan failed at sample {res.break_index}")
        return 1

    q = np.array(res.q_path)                 # (N,7)
    ee = np.array([model.fk(qi)[0] for qi in res.q_path])  # (N,3)
    sig = np.array([d["sigma_min"] for d in res.diagnostics])
    mar = np.array([d["joint_margin"] for d in res.diagnostics])

    n = len(res.q_path)
    print("=" * 72)
    print(f"{'CIRCLE' if args.circle else 'LINE'} trajectory — {args.side} arm, {n} samples, "
          f"{res.times[-1]:.2f}s")
    print(f"gate: {'PASS' if res.passed_gate else 'REVIEW'}  "
          f"(σ_min min={sig.min():.3f}, σ_warn={SIGMA_WARN}, margin min={mar.min():.3f} rad)")
    print("=" * 72)

    print("\nEE position (metres) along path — each row is one axis:")
    for i, ax in enumerate("xyz"):
        print(f"  {ax}: [{ee[:,i].min():+.3f} .. {ee[:,i].max():+.3f}]  {spark(ee[:,i])}")

    print("\nJoint angles (rad) along path — each row is one joint:")
    for j in range(7):
        print(f"  q{j+1}: [{q[:,j].min():+.2f} .. {q[:,j].max():+.2f}]  {spark(q[:,j])}")

    print(f"\nσ_min (singularity distance, must stay ≥ {SIGMA_WARN}):")
    print(f"  [{sig.min():.3f} .. {sig.max():.3f}]  {spark(sig, lo=0, hi=max(0.15, sig.max()))}")
    print(f"  {'✓ safe' if sig.min() >= SIGMA_WARN else '✗ ENTERS singular zone'}")

    print(f"\nJoint margin (rad to nearest limit, must stay > 0.1):")
    print(f"  [{mar.min():.3f} .. {mar.max():.3f}]  {spark(mar, lo=0, hi=max(0.5, mar.max()))}")

    # XY plane path
    print("\nEE path in XY plane (top view):")
    _xy_ascii(ee)

    if args.save:
        _save_png(ee, q, sig, mar, res.times, args.save)
        print(f"\nPNG saved: {args.save}")
    return 0


def _xy_ascii(ee):
    """ASCII scatter of EE (x,y); x→ right, y→ up."""
    x, y = ee[:, 0], ee[:, 1]
    W, H = 60, 18
    xn = (x - x.min()) / (x.max() - x.min() + 1e-9)
    yn = (y - y.min()) / (y.max() - y.min() + 1e-9)
    grid = [[" "] * W for _ in range(H)]
    for i in range(len(x)):
        col = min(W - 1, int(xn[i] * (W - 1)))
        row = H - 1 - min(H - 1, int(yn[i] * (H - 1)))
        grid[row][col] = "*"
    # connect with - where adjacent on same row
    print("    +" + "-" * W + "+  → x")
    for r in range(H):
        print(f" y  |" + "".join(grid[r]) + "|")
    print("    +" + "-" * W + "+")


def _save_png(ee, q, sig, mar, times, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(ee[:, 0], ee[:, 1], "-o", ms=3)
    ax[0, 0].set(title="EE XY path", xlabel="x [m]", ylabel="y [m]"); ax[0, 0].grid(); ax[0, 0].set_aspect("equal")
    ax[0, 1].plot(times, q)
    ax[0, 1].set(title="joint angles", xlabel="t [s]", ylabel="rad"); ax[0, 1].grid()
    ax[1, 0].plot(times, sig, "-o", ms=3)
    ax[1, 0].axhline(SIGMA_WARN, color="r", ls="--", label=f"σ_warn={SIGMA_WARN}")
    ax[1, 0].set(title="σ_min along path", xlabel="t [s]"); ax[1, 0].legend(); ax[1, 0].grid()
    ax[1, 1].plot(times, mar, "-o", ms=3)
    ax[1, 1].axhline(0.1, color="r", ls="--", label="margin=0.1")
    ax[1, 1].set(title="joint margin", xlabel="t [s]"); ax[1, 1].legend(); ax[1, 1].grid()
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    sys.exit(main())
