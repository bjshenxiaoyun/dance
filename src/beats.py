"""Detect musical beats / downbeats from a video soundtrack."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class BeatTrack:
    tempo: float
    beat_times: list[float]
    downbeat_times: list[float]
    source: str

    def to_dict(self) -> dict:
        return asdict(self)


def extract_audio_wav(video_path: str | Path, wav_path: str | Path, sr: int = 22050) -> Path:
    video_path = Path(video_path)
    wav_path = Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-f",
        "wav",
        str(wav_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not wav_path.exists() or wav_path.stat().st_size < 1000:
        raise RuntimeError(
            "无法从视频提取音轨（可能无声音）。"
            f" ffmpeg: {proc.stderr[-400:] if proc.stderr else 'no stderr'}"
        )
    return wav_path


def detect_beats(
    video_path: str | Path,
    sr: int = 22050,
    assume_meter: int = 4,
    max_downbeats: int | None = 24,
) -> BeatTrack:
    """
    Detect beats from teacher video audio.
    Downbeats are estimated as every `assume_meter`-th beat, phase-aligned
    to the strongest onset among the first few bars (common for 4/4 dance music).
    """
    import librosa

    video_path = Path(video_path)
    with tempfile.TemporaryDirectory(prefix="dance_beats_") as tmp:
        wav = Path(tmp) / "audio.wav"
        extract_audio_wav(video_path, wav, sr=sr)
        y, file_sr = librosa.load(str(wav), sr=sr, mono=True)

    if y.size < sr * 0.5:
        raise RuntimeError("音轨过短，无法检测节拍。")

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=file_sr, units="frames")
    # librosa may return tempo as ndarray
    tempo_f = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=file_sr).astype(float)

    if len(beat_times) == 0:
        raise RuntimeError("未检测到节拍，请确认视频有清晰鼓点/节奏。")

    onset_env = librosa.onset.onset_strength(y=y, sr=file_sr)
    # Strength at each beat frame (nearest onset-env frame)
    beat_strength = np.array(
        [float(onset_env[min(int(f), len(onset_env) - 1)]) for f in beat_frames],
        dtype=np.float64,
    )

    # Choose downbeat phase 0..meter-1 that maximizes mean strength of that residue class
    meter = max(2, int(assume_meter))
    best_phase, best_score = 0, -1.0
    for phase in range(meter):
        idxs = np.arange(phase, len(beat_strength), meter)
        if len(idxs) == 0:
            continue
        score = float(np.mean(beat_strength[idxs]))
        if score > best_score:
            best_score, best_phase = score, phase

    down_idx = np.arange(best_phase, len(beat_times), meter)
    downbeat_times = beat_times[down_idx].tolist()
    if max_downbeats is not None:
        downbeat_times = downbeat_times[:max_downbeats]

    return BeatTrack(
        tempo=round(tempo_f, 2),
        beat_times=[round(float(t), 3) for t in beat_times.tolist()],
        downbeat_times=[round(float(t), 3) for t in downbeat_times],
        source=str(video_path),
    )


def beats_to_key_moments(
    beat_times: list[float],
    result,
    fps: float,
    label_prefix: str = "重拍",
):
    """
    Build KeyMoment-like entries at beat times using the worst joint at that frame.
    """
    from angles import JOINT_CN
    from compare import KeyMoment

    moments = []
    if not result.joint_series_diff:
        return moments

    names = list(result.joint_series_diff.keys())
    length = len(next(iter(result.joint_series_diff.values())))

    for i, t_sec in enumerate(beat_times, start=1):
        frame = int(round(t_sec * fps))
        if frame < 0 or frame >= length:
            continue

        worst_name = None
        worst_abs = -1.0
        for name in names:
            d = result.joint_series_diff[name][frame]
            if np.isnan(d):
                continue
            if abs(d) > worst_abs:
                worst_abs = abs(float(d))
                worst_name = name
        if worst_name is None:
            continue

        diff = float(result.joint_series_diff[worst_name][frame])
        t_deg = float(result.teacher_angles[worst_name][frame])
        s_deg = float(result.student_angles[worst_name][frame])
        cn = JOINT_CN.get(worst_name, worst_name)
        direction = "偏大" if diff > 0 else "偏小"
        note = (
            f"{label_prefix}#{i} @ {t_sec:.2f}s：{cn}{direction}——"
            f"老师 {t_deg:.0f}°，你 {s_deg:.0f}°（差 {diff:+.0f}°）"
        )
        moments.append(
            KeyMoment(
                time_sec=round(float(t_sec), 2),
                frame=frame,
                joint=worst_name,
                joint_cn=cn,
                teacher_deg=round(t_deg, 1),
                student_deg=round(s_deg, 1),
                diff_deg=round(diff, 1),
                note=note,
            )
        )
    return moments
