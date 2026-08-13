#!/bin/bash
# Offline FK runner for openarm_pinocchio_ik
# This script ensures the correct Python environment (NumPy 1.x) is used.

set -eo pipefail

# Change to package directory
# Get the absolute path of this script
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
cd "$SCRIPT_DIR/.."
PKG_ROOT="$(pwd)"
export PKG_ROOT

# Source ROS2 Humble
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source "/opt/ros/humble/setup.bash"
else
    echo "ERROR: ROS2 Humble not found at /opt/ros/humble/setup.bash"
    exit 1
fi

# Activate virtual environment with compatible NumPy
if [ -d "$PKG_ROOT/.offline_verify/venv" ]; then
    source "$PKG_ROOT/.offline_verify/venv/bin/activate"
else
    echo "ERROR: Virtual environment not found at $PKG_ROOT/.offline_verify/venv"
    echo "Please run the build process first."
    exit 1
fi

# Source the local install
if [ -f "$PKG_ROOT/.offline_verify/install/setup.bash" ]; then
    source "$PKG_ROOT/.offline_verify/install/setup.bash"
else
    echo "ERROR: Install not found at $PKG_ROOT/.offline_verify/install/setup.bash"
    echo "Please build the package first."
    exit 1
fi

# Verify NumPy version
NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "failed")
if [[ "$NUMPY_VERSION" == 2.* ]]; then
    echo "ERROR: NumPy 2.x detected ($NUMPY_VERSION)"
    echo "Pinocchio requires NumPy 1.x. Aborting."
    exit 1
fi

if ! python -c "import numpy" 2>/dev/null; then
    echo "ERROR: NumPy not available"
    exit 1
fi

# Verify Pinocchio is available
if ! python -c "import pinocchio" 2>/dev/null; then
    echo "ERROR: Pinocchio not available"
    exit 1
fi

# Determine URDF path
# Priority: 1) --urdf argument, 2) OPENARM_URDF env var, 3) verified default
DEFAULT_URDF="/ros2_ws/openarm_ros2/openarm_description/assets/robot/openarm_v1.0/urdf/example/v1.urdf"
URDF_PATH=""

# Check if user provided --urdf argument
for arg in "$@"; do
    if [[ "$arg" == "--urdf" ]]; then
        # Next argument should be the path
        # We'll let ros2 run handle it directly
        URDF_PROVIDED="yes"
        break
    fi
done

if [[ -n "${OPENARM_URDF:-}" ]] && [[ "$URDF_PROVIDED" != "yes" ]]; then
    URDF_PATH="$OPENARM_URDF"
elif [[ "$URDF_PROVIDED" != "yes" ]]; then
    URDF_PATH="$DEFAULT_URDF"
fi

# If URDF path is set (not user-provided), verify it exists
if [[ -n "$URDF_PATH" ]]; then
    if [[ ! -f "$URDF_PATH" ]]; then
        echo "ERROR: URDF not found: $URDF_PATH"
        exit 1
    fi
    echo "Using URDF: $URDF_PATH"
    exec ros2 run openarm_pinocchio_ik fk --urdf "$URDF_PATH" "$@"
else
    # User provided --urdf, pass through directly
    exec ros2 run openarm_pinocchio_ik fk "$@"
fi
