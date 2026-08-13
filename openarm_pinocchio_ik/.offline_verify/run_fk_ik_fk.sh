#!/bin/bash
# FK→IK→FK validation runner for openarm_pinocchio_ik
# This script ensures the correct Python environment is used.

set -eo pipefail

# Get script directory and change to package root
# The script is in .offline_verify/, so we go up one level to get to package root
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
cd "$SCRIPT_DIR/.."
PKG_ROOT="$(pwd)"

# Source ROS2 Humble
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source "/opt/ros/humble/setup.bash"
else
    echo "ERROR: ROS2 Humble not found"
    exit 1
fi

# Activate virtual environment
if [ -d "$PKG_ROOT/.offline_verify/venv" ]; then
    source "$PKG_ROOT/.offline_verify/venv/bin/activate"
else
    echo "ERROR: Virtual environment not found"
    exit 1
fi

# Source local install
if [ -f "$PKG_ROOT/.offline_verify/install/setup.bash" ]; then
    source "$PKG_ROOT/.offline_verify/install/setup.bash"
else
    echo "ERROR: Install not found"
    exit 1
fi

# Verify NumPy version
NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "failed")
if [[ "$NUMPY_VERSION" == 2.* ]]; then
    echo "ERROR: NumPy 2.x detected ($NUMPY_VERSION)"
    echo "Pinocchio requires NumPy 1.x. Aborting."
    exit 1
fi

# Verify Pinocchio is available
if ! python -c "import pinocchio" 2>/dev/null; then
    echo "ERROR: Pinocchio not available"
    exit 1
fi

# Run validation
exec python "$PKG_ROOT/.offline_verify/fk_ik_fk_validate.py" "$@"
