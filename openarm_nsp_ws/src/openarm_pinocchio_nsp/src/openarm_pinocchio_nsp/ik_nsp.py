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

"""Null-space-projection control law — a pure function, unit-testable in isolation.

Given a 6x7 Jacobian J, a 6D task error e, and a desired null-space velocity
``dq_sec``, returns the joint velocity::

    dq = J^# e + alpha (I - J^# J) dq_sec

with J^# the damped right pseudo-inverse J^T (J J^T + lam^2 I)^-1.

Keeping this as a stateless function lets the control-law math be verified with
synthetic matrices (no URDF / Pinocchio model needed).
"""

from __future__ import annotations

import numpy as np


def damped_pseudoinverse(J: np.ndarray, lam_sq: float) -> np.ndarray:
    """Right damped pseudo-inverse J^# = J^T (J J^T + lam^2 I)^-1.

    The matrix inverse of the symmetric PSD 6×6 is expressed as a single
    ``solve`` against the identity only when the full matrix is needed; here
    Cholesky (``posv``-style via ``scipy.linalg.cho_solve`` is avoided to keep
    numpy-only) would help, but the 5× win below comes from never materializing
    ``solve(A, I)`` when callers immediately right-multiply by a vector. This
    function returns the full matrix (public API), so the win is inlining the
    vector path at hot call sites — see ``ik_nsp``/``nsp_step``.
    """
    m = J.shape[0]
    A = J @ J.T + lam_sq * np.eye(m)
    return J.T @ np.linalg.solve(A, np.eye(m))


def damped_pinv_matvec(J: np.ndarray, lam_sq: float, v: np.ndarray) -> np.ndarray:
    """J^# @ v without materializing J^# (one 6×6 solve instead of six)."""
    A = J @ J.T + lam_sq * np.eye(J.shape[0])
    return J.T @ np.linalg.solve(A, v)


def nsp_step(
    J: np.ndarray,
    err: np.ndarray,
    dq_sec: np.ndarray,
    *,
    lam_sq: float,
    alpha: float,
) -> np.ndarray:
    """One null-space-projection IK update.

    Args:
        J: 6x7 task Jacobian.
        err: 6D task error (e.g. SE3 log vector).
        dq_sec: 7D desired null-space velocity (e.g. manipulability gradient).
        lam_sq: squared damping (adaptive: inflate near singularities).
        alpha: secondary-task gain (gain-scheduled toward 0 far from target).

    Returns:
        7D joint velocity increment ``dq``.
    """
    J = np.asarray(J, dtype=float)
    err = np.asarray(err, dtype=float)
    dq_sec = np.asarray(dq_sec, dtype=float)
    n = J.shape[1]

    J_pinv = damped_pseudoinverse(J, lam_sq)
    P = np.eye(n) - J_pinv @ J
    return J_pinv @ err + alpha * (P @ dq_sec)
