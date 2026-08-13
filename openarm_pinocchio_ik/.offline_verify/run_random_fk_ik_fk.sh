#!/bin/bash
# Copyright 2026 Enactic, Inc.
# Convenience script for running random FK→IK→FK batch validation.
# This script sets up the offline verification environment and runs the benchmark.

# Exit on error, but don't fail on unbound variables in sourced scripts
set -e
set -o pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

log_info "Project root: ${PROJECT_ROOT}"

# Change to project root
cd "${PROJECT_ROOT}"

# Source ROS2 environment
if [ -f "/opt/ros/humble/setup.bash" ]; then
    log_info "Sourcing ROS2 Humble environment..."
    source /opt/ros/humble/setup.bash
else
    log_error "ROS2 Humble not found at /opt/ros/humble/setup.bash"
    exit 1
fi

# Activate virtual environment
VENV_DIR="${SCRIPT_DIR}/venv"
if [ -d "${VENV_DIR}" ]; then
    log_info "Activating virtual environment: ${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
else
    log_error "Virtual environment not found at ${VENV_DIR}"
    exit 1
fi

# Source local install setup
INSTALL_SETUP="${SCRIPT_DIR}/install/setup.bash"
if [ -f "${INSTALL_SETUP}" ]; then
    log_info "Sourcing local install setup: ${INSTALL_SETUP}"
    source "${INSTALL_SETUP}"
else
    log_warn "Local install setup not found at ${INSTALL_SETUP}"
fi

# Check NumPy version
NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__.split('.')[0])" 2>/dev/null || echo "unknown")
if [ "${NUMPY_VERSION}" = "unknown" ]; then
    log_error "Failed to check NumPy version"
    exit 1
fi

if [ "${NUMPY_VERSION}" -ge 2 ]; then
    log_error "NumPy 2.x detected (version ${NUMPY_VERSION}.x). This package requires NumPy 1.x."
    exit 1
fi
log_info "NumPy version check passed: ${NUMPY_VERSION}.x"

# Check Pinocchio
if ! python3 -c "import pinocchio" 2>/dev/null; then
    log_error "Pinocchio not found in virtual environment"
    exit 1
fi
PINOCCHIO_VERSION=$(python3 -c "import pinocchio; print(pinocchio.__version__)" 2>/dev/null || echo "unknown")
log_info "Pinocchio version: ${PINOCCHIO_VERSION}"

# Check URDF
URDF_PATH="/ros2_ws/openarm_ros2/openarm_description/assets/robot/openarm_v1.0/urdf/example/v1.urdf"
if [ ! -f "${URDF_PATH}" ]; then
    log_error "URDF file not found: ${URDF_PATH}"
    exit 1
fi
log_info "URDF found: ${URDF_PATH}"

# Check benchmark script
BENCHMARK_SCRIPT="${SCRIPT_DIR}/random_fk_ik_fk_benchmark.py"
if [ ! -f "${BENCHMARK_SCRIPT}" ]; then
    log_error "Benchmark script not found: ${BENCHMARK_SCRIPT}"
    exit 1
fi

# Ensure executable
chmod +x "${BENCHMARK_SCRIPT}"

log_info "Starting random FK→IK→FK batch validation..."
log_info "Command: ${BENCHMARK_SCRIPT} $@"
echo ""

# Run the benchmark
python3 "${BENCHMARK_SCRIPT}" "$@"

exit_code=$?

if [ ${exit_code} -eq 0 ]; then
    log_info "Benchmark completed successfully"
else
    log_error "Benchmark failed with exit code ${exit_code}"
fi

exit ${exit_code}
