"""Compare teacher vs student pose sequences."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from angles import JOINT_CN, sequence_angles


@dataclass
class KeyMoment:
    time_sec: float
    frame: int
    joint: str
    joint_cn: str
    teacher_deg: float
    student_deg: float
    diff_deg: float
    note: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CompareResult:
    joint_mae: dict[str, float]
    joint_series_diff: dict[str, np.ndarray]
    overall_score: float  # 0-100, higher = more similar
    lag_frames: int
    teacher_angles: dict[str, np.ndarray]
    student_angles: dict[str, np.ndarray]
    suggestions: list[str]
    key_moments: list[KeyMoment]
    fps: float


def _nanmean_abs(a: np.ndarray) -> float:
    v = np.abs(a)
    if np.all(np.isnan(v)):
        return float("nan")
    return float(np.nanmean(v))


def estimate_lag(
    teacher_angles: dict[str, np.ndarray],
    student_angles: dict[str, np.ndarray],
    max_lag: int = 30,
) -> int:
    """
    Estimate student lag (positive = student late) via normalized
    cross-correlation on active joint angle curves.
    """
    activity = []
    for name, series in teacher_angles.items():
        if np.all(np.isnan(series)):
            continue
        activity.append((name, float(np.nanstd(series))))
    activity.sort(key=lambda x: x[1], reverse=True)
    names = [n for n, s in activity[:3] if s > 1.0]
    if not names:
        return 0

    def filled(series: np.ndarray) -> np.ndarray:
        out = series.copy()
        nans = np.isnan(out)
        if nans.all():
            return np.zeros_like(out)
        if nans.any():
            idx = np.where(~nans, np.arange(len(out)), 0)
            np.maximum.accumulate(idx, out=idx)
            out[nans] = out[idx[nans]]
            # leading nans
            first = np.argmax(~np.isnan(series))
            out[:first] = out[first]
        return out

    best_lag, best_score = 0, -1e18
    for name in names:
        t = filled(teacher_angles[name])
        s = filled(student_angles[name])
        n = min(len(t), len(s))
        t, s = t[:n], s[:n]
        for lag in range(-max_lag, max_lag + 1):
            if lag > 0:
                a, b = t[: n - lag], s[lag:]
            elif lag < 0:
                a, b = t[-lag:], s[: n + lag]
            else:
                a, b = t, s
            if len(a) < 12:
                continue
            a = (a - a.mean()) / (a.std() + 1e-8)
            b = (b - b.mean()) / (b.std() + 1e-8)
            score = float(np.dot(a, b) / len(a))
            if score > best_score:
                best_score, best_lag = score, lag
    return best_lag


def align_student(student_angles: dict[str, np.ndarray], lag: int, length: int) -> dict[str, np.ndarray]:
    """
    Shift student series by lag and crop/pad to `length`.
    lag > 0 means student is late: drop the first `lag` frames so timings line up.
    """
    out: dict[str, np.ndarray] = {}
    for name, series in student_angles.items():
        if lag > 0:
            shifted = series[lag:]
        elif lag < 0:
            shifted = np.concatenate([np.full(-lag, np.nan), series])
        else:
            shifted = series
        if len(shifted) < length:
            shifted = np.concatenate([shifted, np.full(length - len(shifted), np.nan)])
        out[name] = shifted[:length]
    return out


def _fmt_time(sec: float) -> str:
    if sec < 0:
        sec = 0.0
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m}:{s:04.1f}" if m else f"{s:.1f}s"


def find_key_moments(
    teacher_angles: dict[str, np.ndarray],
    student_angles: dict[str, np.ndarray],
    joint_series_diff: dict[str, np.ndarray],
    fps: float,
    top_k: int = 8,
    min_abs_diff: float = 15.0,
    min_spacing_sec: float = 0.6,
) -> list[KeyMoment]:
    """Pick peak-error timestamps across joints, with spacing to avoid duplicates."""
    candidates: list[tuple[float, int, str, float, float, float]] = []
    length = len(next(iter(joint_series_diff.values())))
    half_win = max(1, int(round(fps * 0.2)))  # ~0.2s local peak window

    for name, diff in joint_series_diff.items():
        abs_diff = np.abs(diff)
        for t in range(length):
            v = abs_diff[t]
            if np.isnan(v) or v < min_abs_diff:
                continue
            lo, hi = max(0, t - half_win), min(length, t + half_win + 1)
            window = abs_diff[lo:hi]
            if np.nanmax(window) != v:
                continue
            # require local maximum
            candidates.append(
                (
                    float(v),
                    t,
                    name,
                    float(teacher_angles[name][t]),
                    float(student_angles[name][t]),
                    float(diff[t]),
                )
            )

    candidates.sort(key=lambda x: x[0], reverse=True)
    picked: list[KeyMoment] = []
    used_frames: list[int] = []
    min_spacing = max(1, int(round(min_spacing_sec * fps)))

    for mag, t, name, t_deg, s_deg, d_deg in candidates:
        if any(abs(t - u) < min_spacing for u in used_frames):
            continue
        cn = JOINT_CN.get(name, name)
        direction = "偏大（张得更开/抬得更高）" if d_deg > 0 else "偏小（幅度不够）"
        if abs(d_deg) >= 25:
            level = "明显"
        elif abs(d_deg) >= 15:
            level = "较明显"
        else:
            level = "略"
        time_sec = t / max(fps, 1e-6)
        note = (
            f"{_fmt_time(time_sec)}：{cn}{level}{direction}——"
            f"老师 {t_deg:.0f}°，你 {s_deg:.0f}°（差 {d_deg:+.0f}°）"
        )
        picked.append(
            KeyMoment(
                time_sec=round(time_sec, 2),
                frame=t,
                joint=name,
                joint_cn=cn,
                teacher_deg=round(t_deg, 1),
                student_deg=round(s_deg, 1),
                diff_deg=round(d_deg, 1),
                note=note,
            )
        )
        used_frames.append(t)
        if len(picked) >= top_k:
            break

    picked.sort(key=lambda m: m.time_sec)
    return picked


def make_suggestions(
    joint_mae: dict[str, float],
    lag_frames: int,
    fps: float,
    key_moments: list[KeyMoment] | None = None,
) -> list[str]:
    tips: list[str] = []
    valid = [(k, v) for k, v in joint_mae.items() if not np.isnan(v)]
    if not valid:
        return ["未检测到有效人体姿态，请确认视频中人物清晰、全身入镜、光线充足。"]

    if abs(lag_frames) >= 2:
        seconds = abs(lag_frames) / max(fps, 1e-6)
        if lag_frames > 0:
            tips.append(f"整体节奏偏慢约 {seconds:.2f}s（{lag_frames} 帧），建议提前启动动作。")
        else:
            tips.append(f"整体节奏偏快约 {seconds:.2f}s（{-lag_frames} 帧），建议稍晚启动、跟住老师节拍。")

    if key_moments:
        tips.append("关键时间点（按视频时间轴）：")
        for m in key_moments[:6]:
            tips.append(f"  · {m.note}")

    ranked = sorted(valid, key=lambda x: x[1], reverse=True)
    for name, mae in ranked[:3]:
        cn = JOINT_CN.get(name, name)
        if mae < 8:
            continue
        # attach peak time for this joint if available
        peak = next((m for m in (key_moments or []) if m.joint == name), None)
        peak_txt = f"，峰值约在 {_fmt_time(peak.time_sec)}" if peak else ""
        if mae >= 25:
            tips.append(f"{cn} 全程偏差较大（平均约 {mae:.0f}°{peak_txt}），重点对照开合幅度。")
        elif mae >= 15:
            tips.append(f"{cn} 有明显偏差（平均约 {mae:.0f}°{peak_txt}），注意角度与发力位置。")
        else:
            tips.append(f"{cn} 略有偏差（平均约 {mae:.0f}°{peak_txt}），可微调对齐。")

    if not tips:
        tips.append("整体相似度很高，关节角度与节奏都比较接近老师。")
    return tips


def compare_poses(
    teacher_poses: np.ndarray,
    student_poses: np.ndarray,
    fps: float = 30.0,
    align: bool = True,
) -> CompareResult:
    teacher_angles = sequence_angles(teacher_poses)
    student_angles = sequence_angles(student_poses)

    lag = estimate_lag(teacher_angles, student_angles) if align else 0
    length = len(teacher_poses)
    student_aligned = align_student(student_angles, lag, length)

    joint_series_diff: dict[str, np.ndarray] = {}
    joint_mae: dict[str, float] = {}
    for name in teacher_angles:
        diff = student_aligned[name] - teacher_angles[name]
        joint_series_diff[name] = diff
        joint_mae[name] = _nanmean_abs(diff)

    # If too few valid comparisons, overall score should reflect that
    maes = [v for v in joint_mae.values() if not np.isnan(v)]
    mean_mae = float(np.mean(maes)) if maes else 180.0
    overall = float(np.clip(100.0 * (1.0 - mean_mae / 45.0), 0.0, 100.0))

    # Validity: fraction of frames with at least one finite joint on both sides
    teacher_ok = np.any(np.isfinite(np.vstack(list(teacher_angles.values()))), axis=0)
    student_ok = np.any(np.isfinite(np.vstack(list(student_aligned.values()))), axis=0)
    both_ok = float(np.mean(teacher_ok & student_ok)) if length else 0.0
    if both_ok < 0.3:
        overall *= both_ok / 0.3  # penalize sparse detections

    key_moments = find_key_moments(
        teacher_angles, student_aligned, joint_series_diff, fps=fps
    )

    return CompareResult(
        joint_mae=joint_mae,
        joint_series_diff=joint_series_diff,
        overall_score=overall,
        lag_frames=lag,
        teacher_angles=teacher_angles,
        student_angles=student_aligned,
        suggestions=make_suggestions(joint_mae, lag, fps, key_moments),
        key_moments=key_moments,
        fps=fps,
    )
