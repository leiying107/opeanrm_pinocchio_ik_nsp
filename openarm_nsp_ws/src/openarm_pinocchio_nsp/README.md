# OpenArm Pinocchio NSP — Trajectory Planning Toolkit

A **planning-only** toolkit for OpenArm v1.0 (7-DoF × 2): null-space-projection
inverse kinematics (NSP-IK) + Cartesian / arc / replay trajectory planning, built
on [Pinocchio](https://github.com/stack-of-tasks/pinocchio). **It computes joint
trajectories — it does NOT drive hardware.** Execution is a separate concern (see
[§6](#6-executing-the-chain)).

Target robot: **OpenArm v1.0**, real hardware, Damiao motors over CAN-FD.

---

## 1. What's inside

| Tool / module | What it does |
|---------------|--------------|
| **`plan_trajectory`** | ★ The headline tool: recorded points + max speed + frequency → a **joint pose chain** (JSON). Pure planner. |
| `plan_cartesian` | Cartesian straight-line / waypoint trajectory (offline, can publish to ros2_control). |
| `ik_node` | ROS2 node exposing NSP-IK. |
| `fk` | Forward kinematics CLI. |
| `move_joints` / `home` | Joint-space move / home (require a controller running). |
| `kinematics.py` | `PinocchioModel` — `fk()`, `ik_nsp()`, `ik_multi()`, singularity metrics. |
| `cartesian_planner.py` | `plan_cartesian`, `fit_arc`, `pose_replay_traj`, `ease_in_out_retime`, `check_traj_smoothness`. |
| `bspline_planner.py` | Joint-space B-spline optimization with self-collision avoidance. |
| `urdf_path.py` | Centralized URDF resolution. |
| `validation/` | Offline validation/diagnostic scripts. |

**Planning / execution separation:** the planners output `(times, q_path)` (or a
JSON chain). Anything that consumes that — ros2_control, a dashboard, a custom
node — is the "executor". This package contains **no CAN code, no motor calls**.

---

## 2. Dependencies & environment (read carefully)

| Dependency | Version / note |
|------------|----------------|
| OS | Ubuntu 22.04 (aarch64 or x86_64) |
| ROS 2 | Humble (`/opt/ros/humble`) |
| Pinocchio | `ros-humble-pinocchio` (3.9.0) — compiled against **numpy 1.x** |
| **numpy** | **must be < 2** (1.26.4). System numpy 2.x makes pinocchio crash (`_ARRAY_API not found`). |
| scipy | any (1.8+). Used for splines / interpolation. |
| URDF | `openarm_description` package (the OpenArm v1.0 `v1.urdf`). |

### ⚠️ The numpy ABI trap (the #1 gotcha)

`ros-humble-pinocchio` is built against numpy 1.x. If you `import pinocchio`
under a Python with **numpy ≥ 2**, it crashes with
`AttributeError: _ARRAY_API not found` (or a segfault). You **must** run these
nodes under a Python that has **numpy < 2**.

**Fix: a dedicated virtualenv** (recommended):
```bash
cd ~/  # or wherever you keep it
python3 -m venv --system-site-packages venv-openarm-ik   # system-site-packages lets it see ROS + colcon
source venv-openarm-ik/bin/activate
pip install "numpy<2" scipy
python -c "import numpy, pinocchio; print(numpy.__version__)"   # → 1.26.4
```
`--system-site-packages` is important: it lets the venv see system-installed
`colcon`, `rclpy`, and `openarm_can`, while the venv's `numpy 1.26.4` shadows
the system `numpy 2.x`.

---

## 3. Install / build

This is a standard `ament_python` ROS 2 package. Drop it into a colcon workspace:

```bash
mkdir -p ~/ik_ws/src && cd ~/ik_ws/src
# (unpack this tar here → produces openarm_pinocchio_nsp/)
cd ~/ik_ws
source /opt/ros/humble/setup.bash
source ~/venv-openarm-ik/bin/activate          # ← venv FIRST (see §2)
python -m colcon build --packages-select openarm_pinocchio_nsp --symlink-install
source install/setup.bash
```

### ⚠️ Build with the venv python, NOT `source activate && colcon build`
`colcon`'s own shebang is `#!/usr/bin/python3`, so `source venv/bin/activate`
does **not** change which Python runs `setuptools` — the generated `ros2 run`
entry-point scripts get a system-Python shebang and crash at launch. Use:
```bash
~/venv-openarm-ik/bin/python -m colcon build --packages-select openarm_pinocchio_nsp --symlink-install
```
Verify: `head -1 install/openarm_pinocchio_nsp/lib/openarm_pinocchio_nsp/plan_trajectory`
should show `#!.../venv-openarm-ik/bin/python`.

### URDF location
`urdf_path.resolve_urdf_path()` looks for `openarm_description` via (in order):
the `OPENARM_URDF` env var → any colcon `install/*/share` on the path → known
fallbacks. If you have the OpenArm ROS 2 stack installed and sourced, it's found
automatically. Otherwise set it explicitly:
```bash
export OPENARM_URDF=/path/to/openarm_description/urdf/openarm_v1_0.urdf
```

---

## 4. The `plan_trajectory` tool (headline feature)

Takes **recorded joint points + a max joint speed + a control frequency** and
outputs a **joint pose chain** (time-stamped joint angles). No execution.

### Input
A JSON file: a list of 7-element joint-angle arrays (radians). These are the
recorded configurations (e.g. taught by dragging the arm). Example `pts.json`:
```json
[[0, 0, 0, 1.5708, 0, 0, 0],
 [0.3491, -0.1745, 0, 1.3963, 0.0873, 0.2618, 0],
 [0.6109, -0.3491, 0.00524, 1.2217, 0.1745, 0.5236, 0]]
```

### Run
```bash
ros2 run openarm_pinocchio_nsp plan_trajectory \
    --points pts.json --max-speed 1.0 --freq 100 --side right \
    --output chain.json
# planned 96-point chain | duration 0.95s | peak 1.00 rad/s | max_step 0.0100 rad -> chain.json
```

| Flag | Meaning | Default |
|------|---------|---------|
| `--points` | input JSON (list of 7 joint angles, rad) | required |
| `--max-speed` | max joint angular speed (rad/s) — hard limit | required |
| `--freq` | control frequency / output sampling density (Hz) | 100 |
| `--side` | `left` or `right` | `right` |
| `--q-seed` | IK warm-start seed (7 angles); default = first point | first point |
| `--null-iters` | IK null-space iterations: `6` fast, `12` thorough | 6 |
| `--output` | output chain JSON path | required |

### What it computes
1. FK each recorded point → 6D pose.
2. Fine-sample each segment in **pose space** (linear position + SLERP
   orientation) → IK each (warm-start chain from `--q-seed`, so no branch jump).
3. Arc-length resample in "max-joint-move" space at step `max_speed/freq` → one
   output point per control tick.

**Two guarantees hold simultaneously:**
- **Uniform time**: output times are exactly `0, 1/freq, 2/freq, …`
- **Max speed**: every joint's speed ≤ `max_speed` (each tick advances exactly
  `max_speed·dt` of joint travel).

Plus: the end-effector follows a **straight line in pose space** between taught
points (because the path is interpolated in pose space, not joint space).

### Output `chain.json` (two formats in one file)
```json
{
  "joint_names": ["openarm_right_joint1", …],          // ros2_control JointTrajectory shape
  "points": [
    {"positions": [7 floats], "time_from_start": {"sec": 0, "nanosec": 0}},
    …
  ],
  "times":    [0.0, 0.01, …],                          // raw form (for any executor)
  "positions":[[7 floats], …],
  "duration": 0.95, "max_speed": 1.0, "freq": 100, "n_points": 96
}
```
A smoothness check runs after planning and prints any `⚠` warnings to **stderr**
(does not block). It flags IK branch jumps (a single huge outlier step) and
over-speed.

---

## 5. Library API (use the planner from Python)

```python
from openarm_pinocchio_nsp.kinematics import PinocchioModel
from openarm_pinocchio_nsp.urdf_path import resolve_urdf_path
from openarm_pinocchio_nsp.plan_trajectory import build_chain, chain_to_joint_trajectory
from openarm_pinocchio_nsp.cartesian_planner import (
    Waypoint, pose_replay_traj, plan_cartesian, fit_arc,
    ease_in_out_retime, check_traj_smoothness,
)

model = PinocchioModel(resolve_urdf_path(), "right")

# --- the planner: points + speed + freq → joint chain ---
points = [q0, q1, q2]                       # each a 7-vector of joint angles
chain = build_chain(model, points, max_speed=1.0, freq=100)
# chain = {"times": [...], "q_path": [[7], ...]}    or None if IK fails

jt = chain_to_joint_trajectory(chain, "right", 1.0, 100)   # JointTrajectory-shape dict

# --- warn-only smoothness check ---
chk = check_traj_smoothness(chain["times"], chain["q_path"])   # TrajCheck(warnings, max_step, …)

# --- other planners (same package) ---
res = plan_cartesian(model, [Waypoint(p0, q0), Waypoint(p1, q1)], q_init=q0)   # Cartesian line
times, q = ease_in_out_retime(res.times, res.q_path, slowdown=2.0, vmax_cap=1.0)
```

---

## 6. Executing the chain

This package does **not** execute. Pick an executor:

### Option A — ros2_control (ROS-native)
Convert `points` to `trajectory_msgs/JointTrajectory` and send to the arm's
`joint_trajectory_controller`:
```python
# pseudocode
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import json
c = json.load(open("chain.json"))
msg = JointTrajectory()
msg.joint_names = c["joint_names"]
for p in c["points"]:
    msg.points.append(JointTrajectoryPoint(
        positions=p["positions"],
        time_from_start=Duration(sec=p["time_from_start"]["sec"],
                                 nanosec=p["time_from_start"]["nanosec"])))
# pub.publish(msg)  → /<side>_joint_trajectory_controller/joint_trajectory
```
Requires ros2_control running for that arm. **Do not** also run a direct-CAN
executor on the same arm simultaneously.

### Option B — custom executor (direct CAN or simulator)
Read `times` and `positions`, linearly interpolate by wall-clock, and send joint
targets to your motor interface at your control rate. The chain is already
time-stamped and speed-limited, so a simple `(time → interpolated q)` follower
suffices.

### Option C — companion dashboard (separate package)
The `openarm_dashboard` package (not included here) is one such executor: its
replay button uses the same `pose_replay_traj` core, then drives CAN.

---

## 7. Other tools (also in this package)

```bash
ros2 run openarm_pinocchio_nsp fk --side right --deg "0,0,0,90,0,0,0"           # forward kinematics
ros2 run openarm_pinocchio_nsp plan_cartesian --side right \
    --line "0.30,0.10,0.40 -> 0.35,0.05,0.45" --quat "0,0,0,1" --output t.json  # Cartesian plan
ros2 run openarm_pinocchio_nsp ik_node                                          # ROS IK service node
```
`validation/` holds diagnostic scripts (singularity maps, arc diagnosis,
closed-loop FK compare, trajectory preview).

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `import pinocchio` → `_ARRAY_API not found` / segfault | numpy ≥ 2 | run under the venv (numpy 1.26.4), §2 |
| `ros2 run …` → `No module named ...` / crashes at launch | entry-point shebang is system Python | rebuild with `venv/bin/python -m colcon build`, §3 |
| URDF not found | `openarm_description` not sourced | `export OPENARM_URDF=/path/to/v1.urdf`, §3 |
| `PLANNING FAILED: some point unreachable` | a taught pose is unreachable / IK branch break | lower `--max-speed` won't help; pick teach points closer in configuration, or raise `--null-iters 12` |
| `⚠ …关节跳变…疑似IK分支跳变` (stderr) | warm-start chain switched branch at one sample | usually still executes; if motion is bad, re-teach points or use `--null-iters 12` |

---

## 9. File map

```
openarm_pinocchio_nsp/
├── README.md                 ← this file
├── package.xml  setup.py  setup.cfg
├── resource/openarm_pinocchio_nsp
├── launch/ik.launch.py
├── src/openarm_pinocchio_nsp/
│   ├── plan_trajectory.py    ← ★ headline planner tool (build_chain, chain_to_joint_trajectory, main)
│   ├── cartesian_planner.py  ← pose_replay_traj, plan_cartesian, fit_arc, ease_in_out_retime, check_traj_smoothness
│   ├── kinematics.py         ← PinocchioModel (fk, ik_nsp, ik_multi, singularity metrics)
│   ├── ik_nsp.py             ← damped pseudoinverse / null-space core
│   ├── bspline_planner.py    ← B-spline optimization (SLSQP + self-collision)
│   ├── urdf_path.py          ← resolve_urdf_path()
│   ├── fk.py ik_node.py move_joints.py home.py plan_cartesian.py  ← other CLI entry points
└── validation/               ← offline diagnostic scripts
```

## 10. Quick start (copy-paste)

```bash
# one-time: venv (numpy<2) + build
python3 -m venv --system-site-packages ~/venv-openarm-ik
source ~/venv-openarm-ik/bin/activate && pip install "numpy<2" scipy && deactivate
mkdir -p ~/ik_ws/src && cd ~/ik_ws/src && tar xzf openarm_pinocchio_nsp_planner.tar.gz
cd ~/ik_ws && source /opt/ros/humble/setup.bash
~/venv-openarm-ik/bin/python -m colcon build --packages-select openarm_pinocchio_nsp --symlink-install
source install/setup.bash

# plan a trajectory (no hardware needed)
cat > pts.json <<'EOF'
[[0,0,0,1.5708,0,0,0],[0.349,-0.175,0,1.396,0.087,0.262,0],[0.611,-0.349,0.005,1.222,0.175,0.524,0]]
EOF
ros2 run openarm_pinocchio_nsp plan_trajectory \
    --points pts.json --max-speed 1.0 --freq 100 --side right --output chain.json
```

— *Planning and execution are deliberately separate. This package plans; your
executor runs.*
