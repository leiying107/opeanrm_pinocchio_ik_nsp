# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""URDF path resolution — centralised so the hard-coded-path bug is fixed once.

Resolution order:
  1. ``OPENARM_URDF`` environment variable (explicit override).
  2. ``openarm_description`` package share dir (when ROS is sourced).
  3. Known installed / source candidate paths.
"""

from __future__ import annotations

import os

# v1.0 example URDF, relative to the openarm_description share root.
_URDF_REL = "assets/robot/openarm_v1.0/urdf/example/v1.urdf"

# Candidate roots, checked in order (installed first, then source checkout).
_CANDIDATE_ROOTS = [
    "/ros2_ws/openarm_ros2/install/openarm_description/share/openarm_description",
    "/ros2_ws/install/openarm_description/share/openarm_description",
    "/ros2_ws/openarm_ros2/openarm_description",
]


def resolve_urdf_path(explicit: str | None = None) -> str:
    """Return an existing OpenArm v1.0 URDF path.

    Args:
        explicit: if given and existing, used verbatim (highest priority).

    Raises:
        FileNotFoundError: if no candidate exists.
    """
    if explicit and os.path.isfile(explicit):
        return explicit

    env = os.environ.get("OPENARM_URDF")
    if env and os.path.isfile(env):
        return env

    # 2. ament package share (only available when ROS workspace is sourced)
    try:
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("openarm_description")
        cand = os.path.join(share, _URDF_REL)
        if os.path.isfile(cand):
            return cand
    except Exception:
        pass  # ROS not sourced / package not found — fall through to candidates

    # 3. known candidate roots
    for root in _CANDIDATE_ROOTS:
        cand = os.path.join(root, _URDF_REL)
        if os.path.isfile(cand):
            return cand

    raise FileNotFoundError(
        f"OpenArm v1.0 URDF not found. Set OPENARM_URDF env var or source the "
        f"openarm_ros2 install setup. Tried: explicit={explicit!r}, candidates="
        f"{[os.path.join(r, _URDF_REL) for r in _CANDIDATE_ROOTS]}"
    )


# Convenience: the default path resolved at import time (for argparse defaults).
DEFAULT_URDF = resolve_urdf_path()
