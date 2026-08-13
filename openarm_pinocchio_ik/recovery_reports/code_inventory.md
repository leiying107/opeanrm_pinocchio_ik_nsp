# OpenArm Pinocchio IK - Code Recovery Inventory Report

**Analysis Date:** 2026-07-24
**Analyzed by:** Static Code Analysis (Claude Code)
**Project:** openarm_pinocchio_ik
**Total Files Analyzed:** 11 (excluding __pycache__)
**Total Lines of Code:** ~865 lines

---

## Executive Summary

This directory contains a **ROS2 Python package** for OpenArm robotic arm inverse kinematics using Pinocchio library. The package appears to be a **single, cohesive module** from a single development period (copyright 2026). **No obvious version conflicts or legacy code fragments detected.**

**Overall Status:** ✅ **CODE BASE APPEARS INTACT AND CONSISTENT**

---

## 1. Project Directory Overview

```
openarm_pinocchio_ik/
├── package.xml                    # ROS2 package manifest
├── setup.py                       # Python package setup with entry points
├── setup.cfg                      # Installation configuration
├── resource/
│   └── openarm_pinocchio_ik       # Empty resource marker file
├── src/
│   └── openarm_pinocchio_ik/
│       ├── __init__.py            # Package initialization (2 lines)
│       ├── kinematics.py          # Core FK/IK/gravity library (126 lines)
│       ├── ik_node.py             # ROS2 IK node (150 lines)
│       ├── fk.py                  # FK CLI tool (71 lines)
│       ├── move_joints.py         # Joint movement CLI (151 lines)
│       ├── home.py                # Home position CLI (62 lines)
│       └── __pycache__/           # Compiled Python cache (Python 3.10)
└── test_fk_ik.py                 # Offline validation script (66 lines)
```

**Total Package Size:** ~91KB

---

## 2. File Function Table

| File | Type | Purpose | Inputs | Outputs | Dependencies |
|------|------|---------|--------|----------|--------------|
| `kinematics.py` | Library | Core FK/IK/gravity engine | URDF path, arm side, joint angles | EE pose, joint torques | pinocchio, numpy |
| `ik_node.py` | ROS2 Node | Real-time IK control | Target pose, joint states | Joint trajectories, gravity torques | rclpy, kinematics.py, ROS2 messages |
| `fk.py` | CLI Tool | Forward kinematics calculator | Joint angles (rad/deg) | EE position + orientation | kinematics.py, numpy |
| `move_joints.py` | CLI Tool | Joint-space movement | Target joint angles | Multi-point trajectory | rclpy, numpy, ROS2 messages |
| `home.py` | CLI Tool | Move to home position | Side selection | Trajectory to zero pose | move_joints.py, rclpy |
| `test_fk_ik.py` | Test Script | Offline validation | URDF path | FK/IK accuracy metrics | kinematics.py, pinocchio, numpy |
| `package.xml` | Config | ROS2 package manifest | N/A | Package metadata | N/A |
| `setup.py` | Config | Python package setup | N/A | Console script entry points | setuptools |
| `setup.cfg` | Config | Install paths | N/A | Script directory configuration | N/A |

---

## 3. Core Call Chain Analysis

### 3.1 Primary Call Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENTRY POINTS (4 CLI tools + 1 ROS2 node)      │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
  │  fk.py      │    │ home.py     │    │ ik_node.py  │
  │  (CLI)      │    │  (CLI)      │    │  (ROS2)     │
  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
         │                  │                  │
         │                  ▼                  │
         │         ┌──────────────┐            │
         │         │ move_joints  │            │
         │         │  .py (CLI)   │            │
         │         └──────┬───────┘            │
         │                │                    │
         └────────────────┴────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   kinematics.py      │
              │   (PinocchioModel)   │
              └───────────────────────┘
```

### 3.2 ROS2 Node Call Flow (ik_node.py)

```
┌─────────────────────────────────────────────────────────────┐
│                     ik_node.py (IKNode)                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Subscriptions:                                              │
│    ├─ /openarm_{side}_target_pose (geometry_msgs/PoseStamped)│
│    └─ /joint_states (sensor_msgs/JointState)                │
│                                                               │
│  Publishers:                                                 │
│    ├─ /{side}_joint_trajectory_controller/joint_trajectory   │
│    └─ /{side}_forward_effort_controller/commands            │
│                                                               │
│  Control Loop (50Hz default):                                │
│    1. Read target_pose from subscription                    │
│    2. Call model.ik() to solve inverse kinematics           │
│    3. Rate-limit joint step (max_step_rad=0.05)              │
│    4. Publish joint trajectory to controller                │
│    5. Publish gravity compensation torques (if enabled)     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   kinematics.py      │
              │   PinocchioModel.ik()│
              │   PinocchioModel.fk()│
              │   PinocchioModel.gravity()│
              └───────────────────────┘
```

### 3.3 CLI Tool Call Flows

**fk.py:**
```
CLI args → PinocchioModel() → fk() → print pose
```

**move_joints.py:**
```
CLI args → MoveJoints node → read /joint_states → interpolate → publish trajectory
```

**home.py:**
```
CLI args → MoveJoints (with target=zeros(7)) → publish trajectory
```

---

## 4. Confirmed Existing Modules

### 4.1 Core Kinematics Module ✅

**File:** `src/openarm_pinocchio_ik/kinematics.py`

**Class:** `PinocchioModel`

**Methods:**
- `__init__(urdf_path, side)` - Initialize model with URDF
- `_full_q(q7)` - Internal: construct full configuration vector
- `fk(q7)` - Forward kinematics (7 joints → pose)
- `gravity(q7)` - Gravity compensation torques
- `ik(target_pos, target_quat_xyzw, q_init, ...)` - Inverse kinematics (damped least squares)

**Features:**
- Supports both "left" and "right" arms
- Joint naming: `openarm_{side}_joint1` through `openarm_{side}_joint7`
- End-effector frame: `openarm_{side}_hand_tcp`
- Joint limits enforced from URDF
- Quaternion convention: ROS xyzw format

**Dependencies:**
- `pinocchio` (any version with URDF loading)
- `numpy`

### 4.2 ROS2 IK Node ✅

**File:** `src/openarm_pinocchio_ik/ik_node.py`

**Class:** `IKNode` (inherits from `rclpy.node.Node`)

**Parameters:**
- `side` - "left" or "right"
- `urdf_path` - Path to URDF file
- `control_hz` - Control frequency (default 50Hz)
- `ik_max_iters` - IK max iterations (default 50)
- `ik_tol` - IK tolerance (default 1e-4)
- `ik_damping` - IK damping factor (default 1e-2)
- `max_step_rad` - Safety rate limit (default 0.05 rad)
- `enable_gravity_comp` - Enable gravity compensation (default True)

**ROS2 Topics:**
- **Subscribes:**
  - `/openarm_{side}_target_pose` (geometry_msgs/PoseStamped)
  - `/joint_states` (sensor_msgs/JointState)
- **Publishes:**
  - `/{side}_joint_trajectory_controller/joint_trajectory` (trajectory_msgs/JointTrajectory)
  - `/{side}_forward_effort_controller/commands` (std_msgs/Float64MultiArray)

### 4.3 CLI Tools ✅

**fk.py:** Forward kinematics calculator
**move_joints.py:** Joint-space trajectory generator
**home.py:** Home position shortcut (zeros pose)

### 4.4 Test Module ✅

**File:** `test_fk_ik.py`

**Features:**
- FK ↔ IK round-trip validation
- Random configuration sampling
- Position/orientation error reporting
- Gravity torque sanity check

---

## 5. Suspected Missing Modules

### 5.1 External Dependencies (Required but Not in Package) ⚠️

**ROS2 Workspace Files (Referenced but Not Included):**
- **URDF Path:** `/ros2_ws/install/openarm_description/share/openarm_description/assets/robot/openarm_v1.0/urdf/example/v1.urdf`
  - **Status:** ❌ NOT IN PACKAGE
  - **Impact:** CRITICAL - Cannot run without URDF file
  - **Required:** `openarm_description` ROS2 package

**ROS2 Controller Names (Assumed to Exist):**
- `/{side}_joint_trajectory_controller` - Joint trajectory controller
- `/{side}_forward_effort_controller` - Forward command effort controller
  - **Status:** ❌ NOT IN PACKAGE (assumed to exist in separate controller package)
  - **Impact:** CRITICAL - Required for hardware control

### 5.2 Configuration Files (Not Found) ⚠️

**Missing Expected Files:**
- Launch files (`launch/*.py`) - Not found
- Config files (`config/*.yaml`) - Not found
- CMakeLists.txt or additional build files
- Requirements.txt or setup requirements file

### 5.3 Documentation (Not Found) ℹ️

**Missing:**
- README.md
- LICENSE file (referenced Apache 2.0 in package.xml)
- Usage documentation
- API documentation

---

## 6. Version Conflicts and Interface Risks

### 6.1 Version Analysis ✅

**Copyright Years:** All files dated 2026 (single development period)
**Package Version:** 0.1.0 (consistent across package.xml and setup.py)
**Python Version:** Python 3.10 (based on __pycache__ naming)
**Pinocchio API:** Compatible with both Pinocchio 2.x and 3.x (has fallback)

**Finding:** ✅ **NO VERSION CONFLICTS DETECTED**

### 6.2 Interface Consistency Analysis ✅

**Joint Naming Convention:**
- Pattern: `openarm_{side}_joint{1-7}`
- Used consistently across: kinematics.py, ik_node.py, move_joints.py, fk.py
- **Result:** ✅ CONSISTENT

**Topic Naming Convention:**
- Input pose: `/openarm_{side}_target_pose`
- Joint states: `/joint_states` (global)
- Trajectory controller: `/{side}_joint_trajectory_controller/joint_trajectory`
- Effort controller: `/{side}_forward_effort_controller/commands`
- **Result:** ✅ CONSISTENT

**Quaternion Convention:**
- Format: ROS geometry_msgs (xyzw)
- Documented in kinematics.py docstring
- Used consistently across FK and IK
- **Result:** ✅ CONSISTENT

**Frame ID Convention:**
- End-effector frame: `openarm_{side}_hand_tcp`
- Used consistently in kinematics.py
- **Result:** ✅ CONSISTENT

**Joint Indexing:**
- 7 joints per arm (1-7)
- Index base: 0 (Python)
- No off-by-one errors detected
- **Result:** ✅ CONSISTENT

### 6.3 Entry Point Configuration ✅

**setup.py Entry Points:**
```python
'ik_node = openarm_pinocchio_ik.ik_node:main',
'fk = openarm_pinocchio_ik.fk:main',
'move_joints = openarm_pinocchio_ik.move_joints:main',
'home = openarm_pinocchio_ik.home:main',
```

**Verification:** All target functions exist and have `main()` functions
**Result:** ✅ VALID CONFIGURATION

---

## 7. High-Risk Code Analysis

### 7.1 Hardware Control Risk Assessment ⚠️

**CRITICAL RISK FILES:**

1. **ik_node.py** - 🚨 HIGH RISK
   - **Risk:** Directly controls real hardware via trajectory controllers
   - **Publishes to:** `/{side}_joint_trajectory_controller/joint_trajectory`
   - **Publishes to:** `/{side}_forward_effort_controller/commands`
   - **Safety Features:** Rate-limiting (`max_step_rad=0.05`), IK convergence check
   - **Dependencies:** Real ROS2 controllers, valid URDF

2. **move_joints.py** - 🚨 HIGH RISK
   - **Risk:** Directly commands joint trajectories
   - **Publishes to:** `/{side}_joint_trajectory_controller/joint_trajectory`
   - **Safety Features:** Interpolated trajectories, reads current state first
   - **Dependencies:** Real ROS2 controllers, active `/joint_states`

3. **home.py** - 🚨 HIGH RISK (inherits risk from move_joints.py)
   - **Risk:** Shortcut for `move_joints` with zero target
   - **Same risk profile as move_joints.py**

**MEDIUM RISK FILES:**

4. **kinematics.py** - ⚠️ MEDIUM RISK
   - **Risk:** Core computation library, used by all high-risk tools
   - **Indirect Risk:** Calculation errors propagate to hardware commands
   - **Safety Features:** Joint limit enforcement, IK convergence detection

**LOW RISK FILES:**

5. **fk.py** - ℹ️ LOW RISK
   - **Risk:** Read-only forward kinematics calculation
   - **No hardware interaction**

6. **test_fk_ik.py** - ℹ️ LOW RISK
   - **Risk:** Offline validation only
   - **No hardware interaction**

### 7.2 Safety Mechanisms Found ✅

**Rate Limiting:**
- `max_step_rad = 0.05` in ik_node.py (line 120)
- Prevents sudden large joint movements

**IK Convergence Check:**
- Returns `None` if IK fails to converge
- Checked in ik_node.py (line 115)

**Joint Limit Enforcement:**
- `np.clip(q7, self.lower, self.upper)` in kinematics.py
- IK solution respects URDF limits

**Current State Monitoring:**
- Reads `/joint_states` before commanding
- Warm-starts IK from current configuration

**Gravity Compensation:**
- Optional enable/disable parameter
- Separate from position control

---

## 8. Dependency Analysis

### 8.1 Python Dependencies

**Required:**
- `numpy` - Array operations
- `pinocchio` - Kinematics/dynamics engine
- `rclpy` - ROS2 Python client library
- `setuptools` - Package installation

**ROS2 Message Dependencies:**
- `std_msgs` - Float64MultiArray
- `sensor_msgs` - JointState
- `geometry_msgs` - PoseStamped, Quaternion
- `control_msgs` - (declared in package.xml, not directly used)
- `trajectory_msgs` - JointTrajectory, JointTrajectoryPoint
- `builtin_interfaces` - Duration

### 8.2 External System Dependencies

**Required External Systems:**
- ROS2 installation (ros2_ws)
- openarm_description package (provides URDF)
- Joint trajectory controller (ros2_control)
- Forward effort controller (ros2_control)
- Active ROS2 master/network

**Critical External File:**
```
/ros2_ws/install/openarm_description/share/openarm_description/
  assets/robot/openarm_v1.0/urdf/example/v1.urdf
```

---

## 9. Data Flow Analysis

### 9.1 Input Data Flow

```
┌────────────────────────────────────────────────────────────────┐
│                         INPUTS                                  │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. URDF File (/ros2_ws/.../v1.urdf)                           │
│     ├─ Robot model description                                 │
│     └─ Joint limits, kinematics chain                         │
│                                                                 │
│  2. Target Pose (geometry_msgs/PoseStamped)                   │
│     ├─ Position (x, y, z)                                      │
│     └─ Orientation (quaternion xyzw)                           │
│                                                                 │
│  3. Current Joint States (sensor_msgs/JointState)              │
│     ├─ Current positions (warm-start IK)                      │
│     └─ Safety verification                                    │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### 9.2 Output Data Flow

```
┌────────────────────────────────────────────────────────────────┐
│                         OUTPUTS                                │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Joint Trajectory (trajectory_msgs/JointTrajectory)         │
│     ├─ Target joint positions for 7 joints                    │
│     ├─ Timestamp (0.2s lookahead)                              │
│     └─ Topic: /{side}_joint_trajectory_controller/...         │
│                                                                 │
│  2. Gravity Torques (std_msgs/Float64MultiArray)               │
│     ├─ 7 joint torques for gravity compensation               │
│     └─ Topic: /{side}_forward_effort_controller/commands       │
│                                                                 │
│  3. Console Output (FK CLI)                                   │
│     ├─ EE position (x, y, z)                                  │
│     └─ EE orientation (quaternion xyzw)                        │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## 10. Recommended Recovery Sequence

### Phase 1: Environment Setup (Pre-requisites)

1. **Install ROS2** (if not present)
2. **Install Python Dependencies:**
   ```bash
   pip install numpy pinocchio rclpy setuptools
   ```

3. **Source ROS2 Environment:**
   ```bash
   source /opt/ros/{humble|iron}/setup.bash
   ```

### Phase 2: External Package Recovery

4. **Locate or Recreate openarm_description Package**
   - Required file: `v1.urdf` at expected path
   - Alternative: Update URDF path parameter to actual location

5. **Verify ros2_control Setup**
   - Ensure joint trajectory controllers configured
   - Ensure forward effort controllers configured

### Phase 3: Package Installation

6. **Build and Install This Package:**
   ```bash
   cd /path/to/openarm_pinocchio_ik
   pip install -e .
   ```

7. **Verify Installation:**
   ```bash
   ros2 run openarm_pinocchio_ik fk --help
   ros2 run openarm_pinocchio_ik home --help
   ```

### Phase 4: Offline Testing (No Hardware)

8. **Run Validation Tests:**
   ```bash
   python test_fk_ik.py
   ```

9. **Test FK CLI:**
   ```bash
   ros2 run openarm_pinocchio_ik fk --side right --deg 0,0,0,0,0,0,0
   ```

### Phase 5: ROS2 Node Testing (Hardware Connected)

10. **Start ROS2 Controllers** (separate launch)

11. **Test IK Node (Low Frequency First):**
    ```bash
    ros2 run openarm_pinocchio_ik ik_node --ros-args -p side:=right -p control_hz:=10.0
    ```

12. **Test Home Command:**
    ```bash
    ros2 run openarm_pinocchio_ik home --side right
    ```

### Phase 6: Integration Testing

13. **Publish Target Pose and Verify Movement**
14. **Monitor Gravity Compensation Output**
15. **Safety Check: Emergency Stop Prepared**

---

## 11. Critical Confirmation Checklist

Before running on real hardware, verify:

- [ ] URDF file exists at correct path or update parameter
- [ ] Joint trajectory controller is running and responsive
- [ ] Forward effort controller is running
- [ ] Emergency stop is accessible
- [ ] Joint limits in URDF match hardware limits
- [ ] Control frequency appropriate for hardware
- [ ] Rate limiting (max_step_rad) is appropriate
- [ ] Current state reading from /joint_states is valid
- [ ] Test FK→IK round-trip accuracy acceptable

---

## 12. Potential Issues to Address

### 12.1 Hardcoded Paths

**Issue:** URDF path hardcoded in multiple files
**Files:** kinematics.py (referenced), ik_node.py, fk.py, test_fk_ik.py
**Default Path:** `/ros2_ws/install/openarm_description/share/openarm_description/assets/robot/openarm_v1.0/urdf/example/v1.urdf`

**Recommendation:** Use parameter override or environment variable for flexibility

### 12.2 Missing Error Handling

**Issue:** Limited error handling in IK convergence failure
**Current:** Warning logged, no action
**Recommendation:** Consider configurable failure mode (hold position vs. stop)

### 12.3 No Launch Files

**Issue:** No ROS2 launch files provided
**Impact:** Manual node startup required
**Recommendation:** Create launch/ik_node_launch.py for convenient startup

---

## 13. Code Quality Observations

### Strengths ✅

1. **Clean Architecture:** Separation of concerns (core library vs. ROS2 wrapper vs. CLI tools)
2. **Type Hints:** Function signatures include type annotations
3. **Documentation:** Docstrings present in key modules
4. **Safety Features:** Rate limiting, IK convergence checks, joint limits
5. **Consistency:** Uniform naming conventions throughout
6. **Error Checking:** Value validation on inputs

### Areas for Improvement ℹ️

1. **No Unit Tests:** Only integration test (test_fk_ik.py) present
2. **No Config Files:** Hardcoded defaults could be externalized
3. **Limited Logging:** Could benefit from more structured logging
4. **No Documentation:** README.md or user guide would be helpful

---

## 14. Final Assessment

### Integrity: ✅ PASS
- No file corruption detected
- No obvious missing code segments
- No incomplete implementations

### Consistency: ✅ PASS
- Naming conventions uniform
- Interface contracts consistent
- No conflicting variable definitions

### Completeness: ⚠️ PARTIAL
- Core functionality complete
- External dependencies missing (expected)
- Documentation missing (optional)

### Safety: ⚠️ CONDITIONAL
- Safety mechanisms present
- Dependent on correct URDF limits
- Requires verified controller setup

### Overall Recovery Status: ✅ READY FOR RESTORATION

**This code base appears to be a complete, consistent module from a single development effort.** The missing components are external dependencies (URDF file, ROS2 controllers) that are expected to be provided by the larger OpenArm robot setup, not missing code from this package.

---

## 15. Contact Information

**Package Maintainer:** OpenArm
**Email:** openarm@enactic.ai
**License:** Apache License 2.0
**Copyright:** 2026 Enactic, Inc.

---

**Report Generated:** 2026-07-24
**Analysis Tool:** Claude Code Static Analysis
**Analysis Scope:** openarm_pinocchio_ik directory only
**Status:** ✅ COMPLETE - NO MODIFICATIONS MADE
