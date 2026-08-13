#!/usr/bin/env python3
# Copyright 2026 Enactic, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Control端 B — IK + preview dashboard (Dash :8051 + meshcat :7000).

P2 milestone (not yet implemented). Will provide: trajectory-library picker,
plan_cartesian IK with go/no-go gate, meshcat 3D preview, and a button to send
the trajectory to control端 A via the /openarm_dashboard/trajectory topic.

For now only hardware_dashboard (control端 A) is available.
"""

import sys


def main() -> int:
    print(__doc__)
    print("\nik_dashboard is not implemented yet (P2 milestone).")
    print("Use: ros2 run openarm_dashboard hardware_dashboard --sim")
    return 0


if __name__ == "__main__":
    sys.exit(main())
