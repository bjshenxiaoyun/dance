"""Extract MediaPipe Pose landmarks from a video."""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_lite.task"


def _make_landmarker(model_path: Path, video_mode: bool = True) -> PoseLandmarker:
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO if video_mode else RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return PoseLandmarker.create_from_options(options)


def extract_poses(
    video_path: str | Path,
    model_path: str | Path | None = None,
    max_frames: int | None = None,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Returns:
      poses: (T, 33, 3) normalized xyz; missing frames are NaN
      fps: video fps
      size: (width, height)
    """
    video_path = Path(video_path)
    model_path = Path(model_path) if model_path else DEFAULT_MODEL
    if not model_path.exists():
        raise FileNotFoundError(f"Pose model not found: {model_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames: list[np.ndarray] = []
    with _make_landmarker(model_path, video_mode=True) as landmarker:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and idx >= max_frames:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(idx * 1000 / fps)
            result = landmarker.detect_for_video(mp_image, ts_ms)
            pose = np.full((33, 3), np.nan, dtype=np.float64)
            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                for i, p in enumerate(lm):
                    pose[i] = (p.x, p.y, p.z)
            frames.append(pose)
            idx += 1

    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {video_path}")
    return np.stack(frames, axis=0), fps, (width, height)
