# Offline FK URDF Resolution Report

**Date:** 2026-07-24
**Package:** openarm_pinocchio_ik v0.1.0
**Issue:** Convenience script failed due to missing URDF file

---

## 1. Problem Description

### Original Error
```
File /ros2_ws/install/openarm_description/share/openarm_description/assets/robot/openarm_v1.0/urdf/example/v1.urdf does not exist
```

### Root Cause
The convenience script `run_offline_fk.sh` did not specify a URDF path, causing FK to fall back to the hardcoded default in `fk.py`:
```
/ros2_ws/install/openarm_description/share/openarm_description/assets/robot/openarm_v1.0/urdf/example/v1.urdf
```

This path points to the install directory, which does not exist for this package (it's in `openarm_ros2`, not installed globally).

---

## 2. Original Hardcoded Path

**In fk.py, ik_node.py, test_fk_ik.py:**
```python
_DEFAULT_URDF = (
    "/ros2_ws/install/openarm_description/share/openarm_description/"
    "assets/robot/openarm_v1.0/urdf/example/v1.urdf"
)
```

**Why it fails:**
- Points to `/ros2_ws/install/` (global workspace install)
- The actual `openarm_description` package is in `/ros2_ws/openarm_ros2/`
- No `colcon build` has been run at the workspace root to create this install path

---

## 3. URDF Search and Verification

### Candidate URDF Found
**Path:** `/ros2_ws/openarm_ros2/openarm_description/assets/robot/openarm_v1.0/urdf/example/v1.urdf`

**Verification:**
- File exists: ✓
- Size: 29,972 bytes
- Contains all required joints:
  - `openarm_left_joint1` ✓
  - `openarm_left_joint7` ✓
  - `openarm_right_joint1` ✓
  - `openarm_right_joint7` ✓
- Contains all required TCP frames:
  - `openarm_left_hand_tcp` ✓
  - `openarm_right_hand_tcp` ✓

**Pinocchio Loading Test:**
```
Model name: openarm
nq: 16
nv: 16
Required Joints: All FOUND
Required Frames (TCP): All FOUND
URDF validation: PASS
```

### Model Type
- Formal ROS URDF (not MuJoCo converted)
- Bimanual (both left and right arms)
- Base frame: `world`
- Includes mimic joints for gripper fingers

---

## 4. Modified Convenience Script

**File:** `.offline_verify/run_offline_fk.sh`

### URDF Resolution Priority (New Logic)
1. **User-provided `--urdf` argument** (highest priority)
2. **Environment variable `OPENARM_URDF`**
3. **Verified default path** (fallback)

### Default URDF (Now Set in Script)
```bash
DEFAULT_URDF="/ros2_ws/openarm_ros2/openarm_description/assets/robot/openarm_v1.0/urdf/example/v1.urdf"
```

### New Features
- Prints the URDF path being used
- Validates file existence before running FK
- Exits with clear error if URDF not found
- Supports explicit `--urdf` override
- Supports `OPENARM_URDF` environment variable

---

## 5. Test Results

### Test 1: Right Arm FK (Default URDF)
**Command:**
```bash
./.offline_verify/run_offline_fk.sh --side right --deg 0,-30,0,90,0,45,0
```

**Output:**
```
Using URDF: /ros2_ws/openarm_ros2/openarm_description/assets/robot/openarm_v1.0/urdf/example/v1.urdf
side = right
joints (rad)      : [0.0, -0.5236, 0.0, 1.5708, 0.0, 0.7854, 0.0]
EE position   xyz : [0.2160, -0.0435, 0.5075]
EE orientation xyzw: [0.5610, 0.0923, 0.7011, -0.4305]
```

**Exit code:** 0
**Status:** PASS

### Test 2: Left Arm FK (Default URDF)
**Command:**
```bash
./.offline_verify/run_offline_fk.sh --side left --deg 0,0,0,0,0,0,0
```

**Output:**
```
Using URDF: /ros2_ws/openarm_ros2/openarm_description/assets/robot/openarm_v1.0/urdf/example/v1.urdf
side = left
joints (rad)      : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
EE position   xyz : [0.0000, 0.1535, 0.2620]
EE orientation xyzw: [1.0000, 0.0000, 0.0000, 0.0000]
```

**Exit code:** 0
**Status:** PASS

### Test 3: Explicit URDF Override
**Command:**
```bash
./.offline_verify/run_offline_fk.sh \
    --urdf /ros2_ws/openarm_ros2/openarm_description/assets/robot/openarm_v1.0/urdf/example/v1.urdf \
    --side right --deg 0,-30,0,90,0,45,0
```

**Exit code:** 0
**Status:** PASS

### Test 4: Environment Variable URDF
**Command:**
```bash
OPENARM_URDF=/ros2_ws/openarm_ros2/openarm_description/assets/robot/openarm_v1.0/urdf/example/v1.urdf \
./.offline_verify/run_offline_fk.sh --side right --deg 0,-30,0,90,0,45,0
```

**Output:**
```
Using URDF: /ros2_ws/openarm_ros2/openarm_description/assets/robot/openarm_v1.0/urdf/example/v1.urdf
```

**Exit code:** 0
**Status:** PASS

---

## 6. Runtime Environment

| Component | Version | Location |
|-----------|---------|----------|
| Python | 3.10.12 | `.offline_verify/venv/bin/python` |
| NumPy | 1.26.4 | `.offline_verify/venv/lib/python3.10/site-packages/numpy/` |
| Pinocchio | 3.9.0 | `/opt/ros/humble/lib/python3.10/site-packages/pinocchio/` |
| rclpy | OK | ROS2 Humble |
| URDF | v1.urdf | `/ros2_ws/openarm_ros2/openarm_description/...` |

---

## 7. Usage Examples

### Default Usage (uses built-in URDF)
```bash
cd /ros2_ws/openarm_pinocchio_ik
./.offline_verify/run_offline_fk.sh --side right --deg 0,-30,0,90,0,45,0
```

### Explicit URDF Override
```bash
./.offline_verify/run_offline_fk.sh \
    --urdf /path/to/your/robot.urdf \
    --side right --deg 0,-30,0,90,0,45,0
```

### Using Environment Variable
```bash
export OPENARM_URDF=/path/to/your/robot.urdf
./.offline_verify/run_offline_fk.sh --side right --deg 0,-30,0,90,0,45,0
```

---

## 8. Compliance Statement

### Source Code Modifications
**NONE.** Original source files remain unchanged:
- `fk.py` - Unmodified
- `kinematics.py` - Unmodified
- `package.xml` - Unmodified
- `setup.py` - Unmodified
- `setup.cfg` - Unmodified

### Script Modifications
**Modified:** `.offline_verify/run_offline_fk.sh`
- Added URDF path resolution logic
- Added file existence validation
- Added URDF path printing

### System Modifications
**NONE.**
- System Python unchanged
- System NumPy unchanged
- `/ros2_ws/install` untouched
- No global packages installed or modified

### Hardware Operations
**NONE.** Only pure offline FK computation performed.

---

## 9. Summary

### Before Fix
- `run_offline_fk.sh` failed with missing URDF error
- No default URDF configured in script
- Fell back to non-existent hardcoded path in source

### After Fix
- `run_offline_fk.sh` uses verified URDF path
- Prints URDF path being used
- Validates file existence
- Supports multiple URDF specification methods
- All FK tests pass

### Verification Results
| Test | Exit Code | Status |
|------|-----------|--------|
| Right Arm FK | 0 | PASS |
| Left Arm FK | 0 | PASS |
| Explicit URDF | 0 | PASS |
| Environment Variable | 0 | PASS |

---

**Fix Status:** COMPLETE
**Verification:** PASSED
