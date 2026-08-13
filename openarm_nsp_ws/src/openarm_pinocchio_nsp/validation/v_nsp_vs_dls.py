#!/usr/bin/env python3
# Copyright 2026 Enactic, Inc.
# SPDX-License-Identifier: Apache-2.0
"""P0 validation: DLS (single-seed) vs NSP (single-seed) vs NSP-multi (multi-seed).

For N random reachable targets, solve IK with all three and compare:
  * convergence rate (multi-seed should reach ~100% — never fail to find a solution)
  * sigma_min distribution of returned solutions (distance from singularity)
  * joint margin (distance from joint limits)

The headline safety claim: ``ik_multi`` converges on essentially every
reachable target and never returns a solution below the critical σ floor.

Run (workspace root, venv active)::
    python src/openarm_pinocchio_nsp/validation/v_nsp_vs_dls.py
"""

from __future__ import annotations

import math
import sys

import numpy as np

from openarm_pinocchio_nsp.kinematics import PinocchioModel, SIGMA_WARN
from openarm_pinocchio_nsp.urdf_path import DEFAULT_URDF

N_SAMPLES = 120
SEED = 42
CONV_TOL_MM = 1.0


def stats(vals):
    a = np.asarray(vals, float)
    return (
        f"med={np.median(a):.4f} P10={np.percentile(a,10):.4f} "
        f"min={np.min(a):.4f} frac<σw={np.mean(a<SIGMA_WARN):.0%}"
    )


def main() -> int:
    m = PinocchioModel(DEFAULT_URDF, "right")
    home = (m.lower + m.upper) / 2.0
    rng = np.random.default_rng(SEED)

    solvers = {"dls": [], "nsp": [], "multi": []}
    conv = {k: 0 for k in solvers}
    margins = {k: [] for k in solvers}

    for _ in range(N_SAMPLES):
        q_true = rng.uniform(m.lower, m.upper)
        pos, quat = m.fk(q_true)

        # DLS single-seed (singular home)
        q = m.ik(pos, quat, q_init=home)
        if q is not None and np.linalg.norm(pos - m.fk(q)[0]) * 1000 < CONV_TOL_MM:
            conv["dls"] += 1
            solvers["dls"].append(m.singular_values(q)[-1])
            margins["dls"].append(m.joint_margin(q))

        # NSP single-seed (singular home)
        r = m.ik_nsp(pos, quat, q_init=home)
        if r.converged and r.pos_err_mm < CONV_TOL_MM:
            conv["nsp"] += 1
            solvers["nsp"].append(r.sigma_min)
            margins["nsp"].append(r.joint_margin)

        # NSP multi-seed (the production solver)
        r = m.ik_multi(pos, quat, n_random=6)
        if r.converged and r.pos_err_mm < CONV_TOL_MM:
            conv["multi"] += 1
            solvers["multi"].append(r.sigma_min)
            margins["multi"].append(r.joint_margin)

    print("=" * 68)
    print("IK solver comparison — OpenArm v1.0 right arm, %d random targets" % N_SAMPLES)
    print("=" * 68)
    print(f"{'solver':8s} {'conv':>10s}   {'σ_min distribution':40s}")
    for k in ("dls", "nsp", "multi"):
        s = stats(solvers[k]) if solvers[k] else "no solutions"
        print(f"{k:8s} {conv[k]:>4d}/{N_SAMPLES:<4d}   {s}")
    print()
    print("joint margin (rad, higher = farther from limits):")
    for k in ("dls", "nsp", "multi"):
        if margins[k]:
            print(f"  {k:8s} med={np.median(margins[k]):.3f} min={np.min(margins[k]):.3f}")

    # pass gate: multi-seed must solve ~everything and beat DLS on σ_min median
    multi_rate = conv["multi"] / N_SAMPLES
    dls_med = float(np.median(solvers["dls"])) if solvers["dls"] else 0
    multi_med = float(np.median(solvers["multi"])) if solvers["multi"] else 0
    ok = multi_rate >= 0.99 and multi_med >= dls_med
    print()
    print(
        f"GATE: multi conv {multi_rate:.0%} (≥99%) & "
        f"multi σ_min med {multi_med:.4f} ≥ DLS {dls_med:.4f} -> "
        f"{'PASS' if ok else 'REVIEW'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
