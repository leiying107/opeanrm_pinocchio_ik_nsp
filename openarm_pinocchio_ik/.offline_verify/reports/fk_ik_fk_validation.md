# FK→IK→FK Validation Report

**Date:** 2026-07-24
**Package:** openarm_pinocchio_ik v0.1.0

---

## 1. Problem Description

### Original Script Issues
The original `test_fk_ik.py` had several limitations:

1. **No argparse:** Cannot pass `--help` or command-line arguments
2. **Hardcoded URDF:** Points to non-existent install path
3. **No flexibility:** Always runs with fixed random seed and 20 iterations

### Hardcoded URDF (Broken)
```python
URDF = (
    "/ros2_ws/install/openarm_description/share/openarm_description/"
    "assets/robot/openarm_v1.0/urdf/example/v1.urdf"
)
```

**Why it fails:**
- Points to `/ros2_ws/install/` (global workspace install)
- The actual `openarm_description` is in `/ros2_ws/openarm_ros2/`
- No global install has been performed

---

## 2. New Validation Program

**Location:** `.offline_verify/fk_ik_fk_validate.py`

### Features
- Full argparse support with `--help`
- Configurable URDF path
- Configurable arm side (left/right)
- Input in degrees or radians
- Configurable tolerances
- Joint limit validation
- Comprehensive output

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--side` | right | Arm side (left/right) |
| `--deg` | - | 7 angles in degrees |
| `--joints` | - | 7 angles in radians |
| `--urdf` | /ros2_ws/openarm_ros2/.../v1.urdf | URDF path |
| `--position-tol-mm` | 1.0 | Position tolerance (mm) |
| `--orientation-tol-deg` | 0.1 | Orientation tolerance (deg) |

### Validation Process
1. Read source joint angles (q_source)
2. Execute FK on q_source → get target pose
3. Execute IK from target pose → get q_solution
4. Check if IK converged
5. Verify q_solution is finite
6. Verify q_solution within joint limits
7. Execute FK on q_solution → get recovered pose
8. Calculate position/orientation errors
9. Compare against tolerances → PASS/FAIL

### Convenience Script
**Location:** `.offline_verify/run_fk_ik_fk.sh`

Features:
- Auto-loads ROS2 Humble
- Activates venv with NumPy 1.26.4
- Verifies environment before running
- Passes all arguments to validation script

---

## 3. URDF Configuration

### Verified URDF
**Path:** `/ros2_ws/openarm_ros2/openarm_description/assets/robot/openarm_v1.0/urdf/example/v1.urdf`

**Model Properties:**
- Name: openarm
- DOF: 16 (bimanual, 7 joints per arm + mimic joints)
- Base frame: world
- TCP frames: openarm_left_hand_tcp, openarm_right_hand_tcp

**Joint Limits (Right Arm):**
| Joint | Lower (rad) | Upper (rad) |
|-------|-------------|-------------|
| 1 | -1.396 | 3.491 |
| 2 | -0.175 | 3.316 |
| 3 | -1.571 | 1.571 |
| 4 | 0.0 | 2.443 |
| 5 | -1.571 | 1.571 |
| 6 | -0.785 | 0.785 |
| 7 | -1.571 | 1.571 |

---

## 4. Validation Results

### Right Arm Test
**Command:**
```bash
./.offline_verify/run_fk_ik_fk.sh --side right --deg 0,-30,0,90,0,45,0
```

**Results:**
| Metric | Value | Status |
|--------|-------|--------|
| q_source (rad) | [0.0, -0.524, 0.0, 1.571, 0.0, 0.785, 0.0] | - |
| IK Convergence | Yes | ✓ |
| Position error | 0.019 mm | PASS (≤1.0 mm) |
| Orientation error | 0.005 deg | PASS (≤0.1 deg) |
| Joint limits | Satisfied | ✓ |
| Overall | - | **PASS** |

**Details:**
- First FK position: [0.2160, -0.0435, 0.5075] m
- First FK orientation (xyzw): [0.5610, 0.0923, 0.7011, -0.4305]
- q_solution: [0.064, -0.175, -0.344, 1.571, 0.320, 0.429, 0.069] rad
- Exit code: 0

### Left Arm Test
**Command:**
```bash
./.offline_verify/run_fk_ik_fk.sh --side left --deg 0,0,0,0,0,0,0
```

**Results:**
| Metric | Value | Status |
|--------|-------|--------|
| q_source (rad) | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] | - |
| IK Convergence | Yes | ✓ |
| Position error | 0.000 mm | PASS (≤1.0 mm) |
| Orientation error | 0.000 deg | PASS (≤0.1 deg) |
| Joint limits | Satisfied | ✓ |
| Overall | - | **PASS** |

**Details:**
- First FK position: [0.0000, 0.1535, 0.2620] m
- First FK orientation (xyzw): [1.0000, 0.0000, 0.0000, 0.0000]
- q_solution: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] rad
- Exit code: 0

---

## 5. Kinematics Interface

### PinocchioModel Methods Used

**Forward Kinematics:**
```python
def fk(q7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Returns: (position[3], quaternion_xyzw[4])
```

**Inverse Kinematics:**
```python
def ik(
    target_pos: np.ndarray,
    target_quat_xyzw: np.ndarray,
    q_init: np.ndarray,
    max_iters: int = 50,
    tol: float = 1e-4,
    damping: float = 1e-2,
) -> np.ndarray | None:
    # Returns: 7 joint angles if converged, None if failed
```

**Joint Properties:**
- `model.lower`: Lower position limits (rad)
- `model.upper`: Upper position limits (rad)
- `model.q_idx`: Joint indices in full model vector

### Orientation Error Calculation
Uses rotation matrix relative rotation angle:
```python
R = R1.T @ R2
angle = acos((trace(R) - 1) / 2)
```
Handles quaternion double-cover (q and -q represent same rotation).

---

## 6. Hardware Calls Detection

**Scanned for:** rclpy, Node, create_publisher, create_subscription

**Result:** NONE FOUND in:
- Original `test_fk_ik.py`
- New `fk_ik_fk_validate.py`
- `kinematics.py`

**Conclusion:** All validation scripts are **PURELY OFFLINE**

---

## 7. Usage Examples

### Basic Usage
```bash
cd /ros2_ws/openarm_pinocchio_ik

# Right arm with degrees
./.offline_verify/run_fk_ik_fk.sh --side right --deg 0,-30,0,90,0,45,0

# Left arm with degrees
./.offline_verify/run_fk_ik_fk.sh --side left --deg 0,0,0,0,0,0,0

# With custom tolerances
./.offline_verify/run_fk_ik_fk.sh \
    --side right \
    --deg 0,-30,0,90,0,45,0 \
    --position-tol-mm 0.5 \
    --orientation-tol-deg 0.05
```

### Help
```bash
./.offline_verify/run_fk_ik_fk.sh --help
```

---

## 8. Source Code Integrity

### Verification Method
Compared SHA256 hashes of all source files (excluding .pyc and core dump) before and after.

### Result
**UNCHANGED**

All original source files remain identical:
- `test_fk_ik.py` - Unmodified
- `kinematics.py` - Unmodified
- `fk.py` - Unmodified
- `ik_node.py` - Unmodified
- `move_joints.py` - Unmodified
- `home.py` - Unmodified
- `package.xml` - Unmodified
- `setup.py` - Unmodified
- `setup.cfg` - Unmodified

### New Files Created (in .offline_verify/)
- `fk_ik_fk_validate.py` - New validation script
- `run_fk_ik_fk.sh` - New convenience wrapper

---

## 9. System Modifications

### What Was Modified
**NONE** in the global system environment

### What Was Created
All within `/ros2_ws/openarm_pinocchio_ik/.offline_verify/`:
- `fk_ik_fk_validate.py`
- `run_fk_ik_fk.sh`
- `reports/test_fk_ik_analysis.txt`
- `reports/source_hashes_after_fk_ik_fix.txt`
- `reports/fk_ik_fk_validation.md`

### What Was NOT Modified
- System Python environment
- System NumPy (remains 2.2.6)
- ROS2 Humble packages
- Original source code
- `/ros2_ws/build`, `/ros2_ws/install`, `/ros2_ws/log`

---

## 10. Summary

### Before Fix
- `test_fk_ik.py` had no argument support
- Hardcoded URDF path was broken
- No flexibility in validation parameters

### After Fix
- New `fk_ik_fk_validate.py` with full argument support
- Verified URDF path configured as default
- Flexible validation with custom tolerances
- Convenience wrapper script for easy usage
- All tests PASS

### Validation Results
| Test | Exit Code | Status |
|------|-----------|--------|
| Right Arm FK→IK→FK | 0 | PASS |
| Left Arm FK→IK→FK | 0 | PASS |

### Compliance
✅ No source code modifications
✅ No system environment modifications
✅ All new files within `.offline_verify/`
✅ No hardware commands executed
✅ Validation successful

---

**Fix Status:** COMPLETE
**Verification:** PASSED
