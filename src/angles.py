"""Joint angle helpers for MediaPipe Pose (33 landmarks)."""

from __future__ import annotations

import numpy as np

# MediaPipe Pose landmark indices
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

# (name, a, b, c) => angle at point b formed by a-b-c
JOINT_DEFS: list[tuple[str, int, int, int]] = [
    ("left_elbow", LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    ("right_elbow", RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
    ("left_shoulder", LEFT_HIP, LEFT_SHOULDER, LEFT_ELBOW),
    ("right_shoulder", RIGHT_HIP, RIGHT_SHOULDER, RIGHT_ELBOW),
    ("left_hip", LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
    ("right_hip", RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
    ("left_knee", LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
    ("right_knee", RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
]

JOINT_CN = {
    "left_elbow": "左肘",
    "right_elbow": "右肘",
    "left_shoulder": "左肩",
    "right_shoulder": "右肩",
    "left_hip": "左髋",
    "right_hip": "右髋",
    "left_knee": "左膝",
    "right_knee": "右膝",
}

# joint name -> (a, vertex, c) landmark indices for highlight
JOINT_FOCUS = {name: (a, b, c) for name, a, b, c in JOINT_DEFS}

# Skeleton edges for drawing
POSE_CONNECTIONS = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
]


def angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return angle ABC in degrees. NaN if vectors are degenerate."""
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-8 or nbc < 1e-8:
        return float("nan")
    cos = float(np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def frame_angles(landmarks: np.ndarray) -> dict[str, float]:
    """
    landmarks: (33, 3) or (33, 2) array of xyz / xy.
    Returns joint name -> angle degrees.
    """
    out: dict[str, float] = {}
    for name, ia, ib, ic in JOINT_DEFS:
        out[name] = angle_at(landmarks[ia, :2], landmarks[ib, :2], landmarks[ic, :2])
    return out


def sequence_angles(poses: np.ndarray) -> dict[str, np.ndarray]:
    """
    poses: (T, 33, 3)
    Returns joint name -> (T,) angles.
    """
    names = [n for n, *_ in JOINT_DEFS]
    series = {n: np.full(len(poses), np.nan, dtype=np.float64) for n in names}
    for t, lm in enumerate(poses):
        if np.isnan(lm).all():
            continue
        angs = frame_angles(lm)
        for n, v in angs.items():
            series[n][t] = v
    return series
