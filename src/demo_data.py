"""Synthetic MediaPipe-like pose sequences with known teacher/student differences."""

from __future__ import annotations

import numpy as np

from angles import (
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)


def _base_tpose() -> np.ndarray:
    """Normalized [0,1] landmarks for a standing figure facing camera."""
    lm = np.zeros((33, 3), dtype=np.float64)
    # torso
    lm[NOSE] = (0.50, 0.12, 0)
    lm[LEFT_SHOULDER] = (0.40, 0.25, 0)
    lm[RIGHT_SHOULDER] = (0.60, 0.25, 0)
    lm[LEFT_HIP] = (0.43, 0.50, 0)
    lm[RIGHT_HIP] = (0.57, 0.50, 0)
    # arms down by default
    lm[LEFT_ELBOW] = (0.36, 0.38, 0)
    lm[RIGHT_ELBOW] = (0.64, 0.38, 0)
    lm[LEFT_WRIST] = (0.34, 0.50, 0)
    lm[RIGHT_WRIST] = (0.66, 0.50, 0)
    # legs
    lm[LEFT_KNEE] = (0.44, 0.70, 0)
    lm[RIGHT_KNEE] = (0.56, 0.70, 0)
    lm[LEFT_ANKLE] = (0.44, 0.90, 0)
    lm[RIGHT_ANKLE] = (0.56, 0.90, 0)
    return lm


def _raise_right_arm(lm: np.ndarray, amount: float) -> np.ndarray:
    """amount in [0,1]: 0=arm down, 1=arm raised overhead-ish."""
    out = lm.copy()
    # interpolate elbow/wrist upward and outward
    shoulder = out[RIGHT_SHOULDER]
    elbow_down = np.array([0.64, 0.38, 0])
    elbow_up = np.array([0.72, 0.18, 0])
    wrist_down = np.array([0.66, 0.50, 0])
    wrist_up = np.array([0.78, 0.08, 0])
    out[RIGHT_ELBOW] = elbow_down * (1 - amount) + elbow_up * amount
    out[RIGHT_WRIST] = wrist_down * (1 - amount) + wrist_up * amount
    # keep relative to shoulder x-shift a bit
    out[RIGHT_ELBOW, 0] = shoulder[0] + (out[RIGHT_ELBOW, 0] - 0.60)
    out[RIGHT_WRIST, 0] = shoulder[0] + (out[RIGHT_WRIST, 0] - 0.60)
    return out


def _bend_left_knee(lm: np.ndarray, amount: float) -> np.ndarray:
    out = lm.copy()
    knee_straight = np.array([0.44, 0.70, 0])
    knee_bent = np.array([0.40, 0.62, 0])
    ankle_straight = np.array([0.44, 0.90, 0])
    ankle_bent = np.array([0.38, 0.78, 0])
    out[LEFT_KNEE] = knee_straight * (1 - amount) + knee_bent * amount
    out[LEFT_ANKLE] = ankle_straight * (1 - amount) + ankle_bent * amount
    return out


def make_teacher_sequence(n_frames: int = 90, fps: float = 30.0) -> np.ndarray:
    """Arm raise then knee bend choreography."""
    poses = np.zeros((n_frames, 33, 3), dtype=np.float64)
    for t in range(n_frames):
        phase = t / max(n_frames - 1, 1)
        base = _base_tpose()
        # 0-0.5: raise right arm; 0.5-1.0: hold arm + bend left knee
        if phase < 0.5:
            arm = phase / 0.5
            knee = 0.0
        else:
            arm = 1.0
            knee = (phase - 0.5) / 0.5
        frame = _raise_right_arm(base, arm)
        frame = _bend_left_knee(frame, knee)
        poses[t] = frame
    return poses


def make_student_sequence(
    n_frames: int = 90,
    arm_scale: float = 0.65,
    knee_scale: float = 0.85,
    lag_frames: int = 8,
) -> np.ndarray:
    """
    Intentionally imperfect student:
    - smaller arm raise (arm_scale)
    - slightly smaller knee bend
    - delayed by lag_frames
    """
    teacher = make_teacher_sequence(n_frames)
    # Rebuild with scaled amplitudes then lag
    poses = np.zeros_like(teacher)
    for t in range(n_frames):
        phase = t / max(n_frames - 1, 1)
        base = _base_tpose()
        if phase < 0.5:
            arm = (phase / 0.5) * arm_scale
            knee = 0.0
        else:
            arm = arm_scale
            knee = ((phase - 0.5) / 0.5) * knee_scale
        frame = _raise_right_arm(base, arm)
        frame = _bend_left_knee(frame, knee)
        poses[t] = frame

    if lag_frames > 0:
        lagged = np.concatenate([np.repeat(poses[:1], lag_frames, axis=0), poses], axis=0)
        poses = lagged[:n_frames]
    elif lag_frames < 0:
        poses = poses[-lag_frames:]
        pad = np.repeat(poses[-1:], n_frames - len(poses), axis=0)
        poses = np.concatenate([poses, pad], axis=0)
    return poses
