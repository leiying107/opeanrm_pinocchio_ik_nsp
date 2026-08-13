# opeanrm_pinocchio_ik_nsp

OpenArm v1.0 (7-DoF bimanual, Damiao motors over CAN-FD) **null-space-projection
inverse kinematics (NSP-IK)**, offline Cartesian/arc/replay trajectory planning,
a real-hardware Dash dashboard, and a native-MoveIt test panel.

This repository combines two workspaces:

| Directory | What it is |
|-----------|------------|
| [`openarm_pinocchio_ik/`](./openarm_pinocchio_ik) | The Pinocchio IK package (`openarm_pinocchio_ik`) + offline verification tools + a minimal **native-MoveIt web panel** (`moveit_web/`). |
| [`openarm_nsp_ws/`](./openarm_nsp_ws) | Colcon workspace holding `openarm_pinocchio_nsp` (NSP-IK + planning) and `openarm_dashboard` (real-hardware Dash UI). See its [`USAGE.md`](./openarm_nsp_ws/USAGE.md) for the full manual. |

## Highlights

- **NSP-IK** (`openarm_nsp_ws/src/openarm_pinocchio_nsp`): two-stage solver —
  Stage 1 damped-least-squares convergence, Stage 2 null-space multi-objective
  ascent (manipulability + joint-limit centering) via `P = I − J⁺J`. Built on
  Pinocchio. Avoids the home-singularity failure that KDL (MoveIt's default) hits.
- **Planners**: Cartesian line, arc (circular fit), pose-replay, B-spline; with
  rest-to-rest quintic ease + per-joint velocity cap; warn-only smoothness check.
- **Dashboard** (`openarm_nsp_ws/src/openarm_dashboard`): Dash web UI driving the
  arms over direct CAN (`openarm_can`), 250 Hz worker, teach/line/arc/replay.
- **MoveIt panel** (`openarm_pinocchio_ik/moveit_web/moveit_web_panel.py`): a pure
  `move_group` client (Dash) to interactively test OpenArm's *native* MoveIt
  (plan/execute named/joint/pose goals, show planning time / error / trajectory).

## Environment (the #1 gotcha)

`ros-humble-pinocchio` is built against **numpy < 2**. Run Pinocchio nodes under a
venv with `numpy<2` (the repo expects `venv-openarm-ik` with `--system-site-packages`):

```bash
python3 -m venv --system-site-packages ~/venv-openarm-ik
source ~/venv-openarm-ik/bin/activate && pip install "numpy<2" scipy dash plotly && deactivate
```

Build with the **venv python** (colcon's shebang is system Python otherwise):
```bash
~/venv-openarm-ik/bin/python -m colcon build --symlink-install
```

## Layout

```
opeanrm_pinocchio_ik_nsp/
├── openarm_pinocchio_ik/        # IK package + tools + moveit_web panel
│   ├── src/openarm_pinocchio_ik/
│   ├── moveit_web/moveit_web_panel.py
│   ├── tools/  recovery_reports/  .offline_verify/  test_*.py
│   └── package.xml setup.py
└── openarm_nsp_ws/              # NSP + dashboard workspace
    ├── src/openarm_pinocchio_nsp/   # NSP-IK + planners (see its README.md)
    ├── src/openarm_dashboard/       # real-hw Dash dashboard
    ├── USAGE.md                     # full manual
    └── singularity_map_j1j4.csv
```

> `build/`, `install/`, `log/`, virtualenvs, core dumps, and `*.log` are git-ignored.

— *Target robot: OpenArm v1.0, real hardware. ROS 2 Humble, Ubuntu 22.04 (aarch64).*
