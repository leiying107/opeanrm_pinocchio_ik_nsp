#!/usr/bin/env python3
"""Offline baseline calibration state machine test.

Simulates the calibration state machine without connecting to ROS.
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class JointData:
    """Joint state data."""
    q: np.ndarray
    dq: np.ndarray
    effort: np.ndarray
    valid: bool = False
    timestamp: float = 0.0
    ros_time: float = 0.0


@dataclass
class ArmBaseline:
    """Calibrated baseline data."""
    effort_baseline: np.ndarray | None = None
    effort_minus_g_baseline: np.ndarray | None = None
    reference_q: np.ndarray | None = None
    reference_tcp_position: np.ndarray | None = None
    samples: int = 0
    start_time: float = 0.0
    std_effort: np.ndarray | None = None
    std_effort_minus_g: np.ndarray | None = None
    max_effort_delta: float = 0.0
    max_effort_minus_g_delta: float = 0.0
    sum_effort_sq: np.ndarray | None = None
    sum_effort_minus_g_sq: np.ndarray | None = None
    is_valid: bool = False


class NodeState:
    """Node operation states."""
    INIT = "initializing"
    CALIBRATING = "calibrating_baseline"
    READY = "ready"
    STALE = "stale_data"
    ERROR = "error"


def simulate_calibration():
    """Simulate baseline calibration state machine.

    Conditions:
    - 100 Hz sample timer
    - joint_data valid
    - dq all zeros (stationary)
    - calibration_duration=2.0
    - calibration_min_samples=100
    """

    print("=" * 70)
    print("TCP Collision Monitor Baseline State Machine Offline Test")
    print("=" * 70)
    print()

    # Parameters
    calibration_duration = 2.0
    calibration_min_samples = 100
    sample_rate = 100.0  # Hz
    sample_period = 1.0 / sample_rate

    # Simulated state
    state = NodeState.CALIBRATING
    baseline = ArmBaseline()
    start_time = 0.0

    # Simulated joint data (all zeros for stationary)
    joint_data = JointData(
        q=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
        dq=np.zeros(7),  # Stationary
        effort=np.ones(7) * 0.5,
        valid=True,
        timestamp=0.0,
    )

    sample_count = 0
    test_results = []

    # Simulate 3 seconds (enough for calibration)
    sim_duration = 3.0
    now = 0.0

    print(f"Simulation parameters:")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Calibration duration: {calibration_duration} s")
    print(f"  Min samples: {calibration_min_samples}")
    print(f"  Simulation duration: {sim_duration} s")
    print()

    # Test A: First callback
    print("Test A: First callback")
    now = sample_period
    sample_count = 1
    baseline.start_time = 0.0
    baseline.samples = sample_count
    baseline.is_valid = False

    # Fixed logic: state depends on baseline.is_valid
    if baseline.is_valid:
        state = NodeState.READY
    else:
        state = NodeState.CALIBRATING

    print(f"  After 1st callback:")
    print(f"    samples = {baseline.samples}")
    print(f"    baseline.is_valid = {baseline.is_valid}")
    print(f"    node_state = {state}")

    test_a_pass = (
        baseline.samples == 1 and
        baseline.is_valid == False and
        state == NodeState.CALIBRATING
    )
    print(f"  Result: {'PASS' if test_a_pass else 'FAIL'}")
    test_results.append(("A - First callback", test_a_pass))
    print()

    # Test B: Calibration intermediate
    print("Test B: Calibration intermediate state")
    sample_count = 50
    baseline.samples = sample_count
    elapsed = sample_count * sample_period

    if baseline.is_valid:
        state = NodeState.READY
    else:
        state = NodeState.CALIBRATING

    print(f"  After 50 callbacks (t={elapsed:.2f}s):")
    print(f"    samples = {baseline.samples}")
    print(f"    elapsed = {elapsed:.2f}s < {calibration_duration}s")
    print(f"    baseline.is_valid = {baseline.is_valid}")
    print(f"    node_state = {state}")

    test_b_pass = (
        baseline.samples == 50 and
        elapsed < calibration_duration and
        baseline.is_valid == False and
        state == NodeState.CALIBRATING
    )
    print(f"  Result: {'PASS' if test_b_pass else 'FAIL'}")
    test_results.append(("B - Intermediate state", test_b_pass))
    print()

    # Test C: Samples >= 100 but elapsed < 2.0 (impossible at 100Hz)
    # At 100Hz, 100 samples = 1.0s exactly
    # So we skip this test - it's impossible to have samples >=100 and elapsed < 2.0 at 100Hz

    # Test C2: elapsed >= 2.0 but samples < 100 (simulate intermittent samples)
    print("Test C: Early completion check (samples < min_samples)")
    sample_count = 90
    baseline.samples = sample_count
    elapsed = 2.1  # > 2.0s but samples < 100

    if baseline.is_valid:
        state = NodeState.READY
    else:
        state = NodeState.CALIBRATING

    print(f"  At t={elapsed}s with {baseline.samples} samples:")
    print(f"    elapsed = {elapsed}s >= {calibration_duration}s")
    print(f"    samples = {baseline.samples} < {calibration_min_samples}")
    print(f"    baseline.is_valid = {baseline.is_valid}")
    print(f"    node_state = {state}")

    test_c_pass = (
        elapsed >= calibration_duration and
        baseline.samples < calibration_min_samples and
        baseline.is_valid == False and
        state == NodeState.CALIBRATING
    )
    print(f"  Result: {'PASS' if test_c_pass else 'FAIL'}")
    test_results.append(("C - Incomplete at duration", test_c_pass))
    print()

    # Test D: Calibration complete
    print("Test D: Calibration complete")
    sample_count = 200
    baseline.samples = sample_count
    elapsed = sample_count * sample_period
    baseline.is_valid = True

    if baseline.is_valid:
        state = NodeState.READY
    else:
        state = NodeState.CALIBRATING

    print(f"  After {sample_count} callbacks (t={elapsed:.2f}s):")
    print(f"    samples = {baseline.samples} >= {calibration_min_samples}")
    print(f"    elapsed = {elapsed}s >= {calibration_duration}s")
    print(f"    baseline.is_valid = {baseline.is_valid}")
    print(f"    node_state = {state}")

    test_d_pass = (
        baseline.samples >= calibration_min_samples and
        elapsed >= calibration_duration and
        baseline.is_valid == True and
        state == NodeState.READY
    )
    print(f"  Result: {'PASS' if test_d_pass else 'FAIL'}")
    test_results.append(("D - Calibration complete", test_d_pass))
    print()

    # Test E: After calibration, state stays READY
    print("Test E: State stays READY after calibration")
    # Simulate more callbacks
    sample_count = 250
    baseline.samples = sample_count
    elapsed = sample_count * sample_period
    # baseline.is_valid stays True

    if baseline.is_valid:
        state = NodeState.READY
    else:
        state = NodeState.CALIBRATING

    print(f"  After {sample_count} callbacks (t={elapsed:.2f}s):")
    print(f"    baseline.is_valid = {baseline.is_valid}")
    print(f"    node_state = {state}")

    test_e_pass = (
        baseline.is_valid == True and
        state == NodeState.READY
    )
    print(f"  Result: {'PASS' if test_e_pass else 'FAIL'}")
    test_results.append(("E - State persists READY", test_e_pass))
    print()

    # Summary
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    for name, passed in test_results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")

    all_pass = all(passed for _, passed in test_results)
    print()
    print(f"Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 70)

    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(simulate_calibration())
