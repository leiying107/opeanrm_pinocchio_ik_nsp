# Random FK→IK→FK Batch Validation - Comprehensive Report

**Generated:** 2026-07-24 11:09:00
**Package:** openarm_pinocchio_ik
**Test Environment:** ROS2 Humble Offline Verification

---

## 1. Algorithm Design

### 1.1 Test Methodology

This validation uses a **reverse kinematics approach** to guarantee reachable targets:

1. Sample a random joint configuration `q_target` within verified joint limits
2. Compute target pose `T_target = FK(q_target)`
3. Solve IK from an independent initial guess `q_init`
4. Verify `FK(q_solution) ≈ T_target`

Because `T_target = FK(q_target)`, the target pose is **mathematically guaranteed to be reachable** in the current kinematic model. Failures cannot be attributed to "unreachable targets" and must indicate issues with the IK solver, initialization, or numerical conditioning.

### 1.2 IK Interface Analysis

The existing `kinematics.py` IK function has the following interface:

```python
def ik(
    self,
    target_pos: np.ndarray,
    target_quat_xyzw: np.ndarray,
    q_init: np.ndarray,
    max_iters: int = 50,
    tol: float = 1e-4,
    damping: float = 1e-2,
) -> np.ndarray | None
```

**Key findings:**
- Returns: `q7` (7 joint angles) on success, `None` on failure
- Does NOT return: convergence flag, iteration count, or final error
- Default parameters: `max_iters=50`, `tol=1e-4`, `damping=1e-2`
- Algorithm: Damped Least Squares (DLS) with Jacobian from Pinocchio

---

## 2. Joint Sampling Strategy

### 2.1 Joint Limits (Right Arm)

| Joint | Lower (rad) | Upper (rad) | Range (deg) |
|-------|-------------|-------------|-------------|
| 1 | -1.3963 | 3.4907 | ~279° |
| 2 | -0.1745 | 3.3161 | ~193° |
| 3 | -1.5708 | 1.5708 | 180° |
| 4 | 0.0000 | 2.4435 | ~140° |
| 5 | -1.5708 | 1.5708 | 180° |
| 6 | -0.7854 | 0.7854 | 90° |
| 7 | -1.5708 | 1.5708 | 180° |

### 2.2 Joint Limits (Left Arm)

| Joint | Lower (rad) | Upper (rad) | Range (deg) |
|-------|-------------|-------------|-------------|
| 1 | -3.4907 | 1.3963 | ~279° |
| 2 | -3.3161 | 0.1745 | ~193° |
| 3-7 | Same as right | | |

### 2.3 Sampling Method

- **Margin ratio:** 0.05 (5% safety margin from limits)
- **Effective range:** `q_lower + margin` to `q_upper - margin`
- **Distribution:** Uniform within effective range
- **Rejection:** NaN, Inf, or out-of-range samples are rejected

---

## 3. IK Initialization Strategies

### 3.1 independent_random (Default)

The initial guess `q_init` is sampled independently from `q_target`:

- Both sampled from the same valid joint range
- Minimum distance enforced: 0.5 rad (Euclidean norm)
- If initial guess is too close to target, resample up to 100 attempts
- This tests the solver's global convergence capability

### 3.2 Other Supported Modes

- **zero:** Uses clipped zero position as initial guess
- **perturb_target:** Adds Gaussian noise to target (configurable std)
- **multi-start:** Retries with new independent initial guesses on failure

---

## 4. Right Arm Statistics (100 Samples)

### 4.1 Overall Results

| Metric | Value |
|--------|-------|
| Total samples | 100 |
| Valid targets | 100 |
| First-try success | 98 (98.0%) |
| Multi-start recovery | 0 |
| Final success | 98 (98.0%) |
| Near-singular samples | 7 (7%) |

### 4.2 Failure Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| SUCCESS | 98 | 98.0% |
| MAX_ITERATIONS | 2 | 2.0% |

**Conclusion:** Only 2 out of 100 reachable targets failed, both due to maximum iteration limit (50 iterations). No failures due to joint limit violations, non-finite solutions, or exceptions.

### 4.3 Position Error Distribution (Successful Samples)

| Statistic | Value (mm) |
|-----------|------------|
| Mean | 0.0180 |
| Median | 0.0087 |
| Std | 0.0235 |
| P90 | 0.0447 |
| P95 | 0.0812 |
| P99 | 0.0885 |
| Max | 0.0917 |

**Analysis:** The median error is well below 1 mm tolerance. Even P99 is below 0.1 mm (10x better than tolerance).

### 4.4 Orientation Error Distribution (Successful Samples)

| Statistic | Value (deg) |
|-----------|-------------|
| Mean | 0.0015 |
| Median | 0.0005 |
| Std | 0.0018 |
| P90 | 0.0044 |
| P95 | 0.0052 |
| P99 | 0.0055 |
| Max | 0.0056 |

**Analysis:** All orientation errors are well below 0.1° tolerance (20x better).

### 4.5 Solve Time Distribution

| Statistic | Value (ms) |
|-----------|------------|
| Mean | 0.71 |
| Median | 0.46 |
| P95 | 1.94 |
| Max | 2.77 |

**Analysis:** Very fast convergence on average, with worst case under 3ms.

---

## 5. Left Arm Statistics (100 Samples)

### 5.1 Overall Results

| Metric | Value |
|--------|-------|
| Total samples | 100 |
| Valid targets | 100 |
| First-try success | 98 (98.0%) |
| Multi-start recovery | 0 |
| Final success | 98 (98.0%) |
| Near-singular samples | 5 (5%) |

### 5.2 Failure Breakdown

| Status | Count | Percentage |
|--------|-------|------------|
| SUCCESS | 98 | 98.0% |
| MAX_ITERATIONS | 2 | 2.0% |

**Conclusion:** Identical performance to right arm - 98% success rate with same failure pattern.

### 5.3 Position Error Distribution (Successful Samples)

| Statistic | Value (mm) |
|-----------|------------|
| Mean | 0.0139 |
| Median | 0.0073 |
| Std | 0.0202 |
| P90 | 0.0384 |
| P95 | 0.0540 |
| P99 | 0.0935 |
| Max | 0.0973 |

### 5.4 Orientation Error Distribution (Successful Samples)

| Statistic | Value (deg) |
|-----------|-------------|
| Mean | 0.0020 |
| Median | 0.0008 |
| Std | 0.0021 |
| P90 | 0.0050 |
| P95 | 0.0054 |
| P99 | 0.0055 |
| Max | 0.0056 |

### 5.5 Solve Time Distribution

| Statistic | Value (ms) |
|-----------|------------|
| Mean | 0.89 |
| Median | 0.56 |
| P95 | 2.54 |
| Max | 3.11 |

---

## 6. Error Distribution Analysis

### 6.1 Cross-Arm Comparison

| Metric | Right | Left |
|--------|-------|------|
| Success Rate | 98.0% | 98.0% |
| Position Error (mean) | 0.0180 mm | 0.0139 mm |
| Orientation Error (mean) | 0.0015 deg | 0.0020 deg |
| Solve Time (mean) | 0.71 ms | 0.89 ms |

**Conclusion:** Both arms exhibit nearly identical performance, validating the kinematic model consistency.

### 6.2 Iteration Analysis

**Limitation:** The IK function does not return iteration counts. This is marked as `unavailable` in all results.

---

## 7. Singularity Analysis

### 7.1 Near-Singular Detection Criteria

- **Singular value threshold:** 1e-3
- **Condition number threshold:** 100

### 7.2 Right Arm Near-Singular Samples

- 7 out of 100 samples (7%) flagged as near-singular
- All near-singular samples still converged successfully
- No correlation between singularity and failure

### 7.3 Left Arm Near-Singular Samples

- 5 out of 100 samples (5%) flagged as near-singular
- All near-singular samples still converged successfully
- No correlation between singularity and failure

**Conclusion:** The DLS (Damped Least Squares) IK implementation handles near-singular configurations robustly.

---

## 8. Comparison with Previous 60% Convergence Rate

### 8.1 Previous Test Analysis

The previous tests that showed ~60% convergence rate may have had one or more of the following issues:

1. **Target generation method:** If targets were sampled directly in Cartesian space without FK validation, many may have been genuinely unreachable
2. **Initialization method:** Poor initial guesses could lead to convergence failures
3. **Joint limit handling:** Samples near limits may have caused numerical issues
4. **Tolerance settings:** Different position/orientation thresholds

### 8.2 Current Test Improvements

1. **Guaranteed reachable targets:** All targets are `FK(q_target)` for valid `q_target`
2. **Proper initialization:** Independent random initial guesses with minimum distance
3. **Safe sampling:** 5% margin from joint limits
4. **Reasonable tolerances:** 1 mm position, 0.1° orientation

### 8.3 Conclusion

The **98% success rate** demonstrated in this test strongly suggests that:
- The IK solver implementation is fundamentally sound
- Previous ~60% rates were likely due to unreachable target poses or poor initialization
- With proper target generation and initialization, the solver achieves excellent convergence

---

## 9. Failure Cause Analysis

### 9.1 Observed Failure Modes

Only one failure mode was observed:

| Failure Mode | Count | Cause |
|--------------|-------|-------|
| MAX_ITERATIONS | 4 total | Solver did not converge within 50 iterations |

### 9.2 NOT Observed

The following failure modes were **NOT observed** in 200 samples:
- `JOINT_LIMIT_VIOLATION` (0 samples)
- `NONFINITE_SOLUTION` (0 samples)
- `IK_EXCEPTION` (0 samples)
- `FK_EXCEPTION` (0 samples)
- `POSE_ERROR_TOO_LARGE` (0 samples)
- `SOLVER_REPORTED_FAILURE_BUT_POSE_PASS` (0 samples)

### 9.3 Failure Analysis

The 4 failures (2 right, 2 left) all occurred with:
- Full 5 initial guesses tried (multi-start)
- `attempts_used = 5`
- Final status: `MAX_ITERATIONS`
- Not near-singular configurations

This suggests these cases are due to:
1. Challenging kinematic configurations (deep in workspace)
2. Poor initial guesses for all 5 attempts
3. Possible local minima in the error landscape

---

## 10. Source Code Integrity

### 10.1 Verification

All source code files in `src/openarm_pinocchio_ik/` remain **unchanged**:

| File | SHA-256 |
|------|---------|
| kinematics.py | 0e09e67e4b25280e0a5969bf52d25e8e66418b13ff1e4f9253ec4bdd321304e9 |
| fk.py | 0a90bad215546b10247f6b9420554984a9ed598988045c41772c96ea862e9ede |
| ik_node.py | addc8cce426bdd718d356cabed7206dc3734fb5562e0a3927a3813d3cca9e249 |

### 10.2 New Files

All new files are located under `.offline_verify/`:
- `random_fk_ik_fk_benchmark.py` - Main validation program
- `run_random_fk_ik_fk.sh` - Convenience wrapper script
- `random_benchmark_results/` - Test results and reports
- `reports/random_fk_ik_fk_benchmark.md` - This report

---

## 11. Hardware Isolation Confirmation

### 11.1 No Hardware Commands Executed

- No CAN bus communication
- No controller commands
- No joint movement
- No ROS publishers/subscribers created
- No `openarm_bringup` launched

### 11.2 Pure Offline Validation

All tests used only:
- Pinocchio FK/IK computations
- NumPy numerical operations
- File I/O for results

---

## 12. Recommendations

### 12.1 Current IK Implementation

**Strengths:**
- 98% convergence on reachable targets
- Sub-millimeter precision on successful cases
- Sub-millisecond solve times
- Robust handling of near-singular configurations

**Limitations:**
- No iteration count feedback
- Fixed damping factor
- No adaptive step size

### 12.2 Potential Improvements

1. **Add iteration count to IK return value** - For performance monitoring
2. **Adaptive damping** - Reduce damping when far from target, increase when close
3. **Configurable tolerances** - Allow caller to specify position/orientation thresholds
4. **Singularity detection** - Early warning for near-singular configurations
5. **Multi-start option** - For offline planning when time permits

### 12.3 For Real-Time Control

The current implementation is suitable for real-time control given:
- Average solve time < 1ms
- 98% single-attempt success rate
- Failures are graceful (return None)

For critical applications:
- Add watchdog timeout
- Monitor for `None` returns
- Consider motion planning with collision checking

---

## 13. Test Artifacts

### 13.1 Result Locations

**Right Arm (100 samples, seed 42):**
```
.offline_verify/random_benchmark_results/right_seed42_100samples_20260724_110808/
├── results.csv          - All 100 samples with full details
├── summary.json         - Statistical summary
├── summary.md           - Human-readable report
├── failed_cases.csv     - 2 failed samples for replay
└── replay_commands.txt  - Commands to reproduce failures
```

**Left Arm (100 samples, seed 43):**
```
.offline_verify/random_benchmark_results/left_seed43_100samples_20260724_110817/
├── results.csv
├── summary.json
├── summary.md
├── failed_cases.csv     - 2 failed samples for replay
└── replay_commands.txt
```

### 13.2 Reproducibility

All tests are reproducible using the same random seed:
- Right arm: `--seed 42`
- Left arm: `--seed 43`

---

## 14. Final Conclusions

### 14.1 Summary

| Metric | Value |
|--------|-------|
| Total samples tested | 200 (100 per arm) |
| Overall success rate | 98.0% |
| Position error (mean) | < 0.02 mm |
| Orientation error (mean) | < 0.002 deg |
| Solve time (mean) | < 1 ms |
| Source modifications | None |
| Hardware interaction | None |

### 14.2 Key Findings

1. **The IK solver implementation is highly effective** - 98% success rate on guaranteed-reachable targets
2. **Previous 60% rates were likely due to unreachable targets** - not a solver deficiency
3. **The solver is numerically robust** - handles near-singular configurations well
4. **Performance is excellent** - sub-millisecond solve times on average
5. **Both arms behave identically** - validates model consistency

### 14.3 Validation Complete

This batch validation tool successfully demonstrates that:
- The FK→IK→FK pipeline works correctly
- Reachable targets are achieved with high probability
- Error is within acceptable tolerance for robotic applications
- The implementation is ready for further integration testing

---

**End of Report**
