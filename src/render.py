"""Render stick-figure comparison video and charts."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from angles import JOINT_CN, JOINT_FOCUS, POSE_CONNECTIONS
from compare import CompareResult


def _to_px(lm: np.ndarray, w: int, h: int) -> tuple[int, int] | None:
    if np.isnan(lm[0]) or np.isnan(lm[1]):
        return None
    return int(lm[0] * w), int(lm[1] * h)


def draw_skeleton(
    canvas: np.ndarray,
    landmarks: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 3,
) -> None:
    h, w = canvas.shape[:2]
    for a, b in POSE_CONNECTIONS:
        pa = _to_px(landmarks[a], w, h)
        pb = _to_px(landmarks[b], w, h)
        if pa and pb:
            cv2.line(canvas, pa, pb, color, thickness, cv2.LINE_AA)
    for i in range(len(landmarks)):
        p = _to_px(landmarks[i], w, h)
        if p:
            cv2.circle(canvas, p, 4, color, -1, cv2.LINE_AA)


def error_color(mae: float) -> tuple[int, int, int]:
    """BGR: green -> yellow -> red by error magnitude."""
    t = float(np.clip(mae / 30.0, 0.0, 1.0))
    # green (0,200,0) to red (0,0,220)
    return (0, int(200 * (1 - t)), int(220 * t))


def _fit_panel(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize frame to fit inside (width, height), letterbox with dark padding."""
    canvas = np.full((height, width, 3), 30, dtype=np.uint8)
    if frame is None or frame.size == 0:
        return canvas
    fh, fw = frame.shape[:2]
    scale = min(width / max(fw, 1), height / max(fh, 1))
    nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = (width - nw) // 2
    y0 = (height - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def write_comparison_video(
    teacher_poses: np.ndarray,
    student_poses: np.ndarray,
    result: CompareResult,
    out_path: str | Path,
    fps: float = 30.0,
    size: tuple[int, int] = (480, 540),
    teacher_video: str | Path | None = None,
    student_video: str | Path | None = None,
    original_height: int = 480,
) -> Path:
    """
    Write comparison video.
    If teacher/student video paths are provided, layout is:
      top:    original teacher | original student  (aligned)
      bottom: skeleton teacher | skeleton student
    Otherwise only the skeleton row is written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    show_original = teacher_video is not None and student_video is not None
    divider_h = 4 if show_original else 0
    out_h = (original_height + divider_h + h) if show_original else h
    out_w = w * 2
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h),
    )
    n = min(
        len(teacher_poses),
        len(student_poses),
        len(next(iter(result.joint_series_diff.values()))),
    )
    diffs = np.vstack([np.abs(v[:n]) for v in result.joint_series_diff.values()])
    if diffs.size == 0:
        frame_err = np.zeros(n, dtype=np.float64)
    else:
        with np.errstate(all="ignore"):
            frame_err = np.nanmean(diffs, axis=0)
        frame_err = np.where(np.isnan(frame_err), 0.0, frame_err)

    lag = result.lag_frames
    cap_t = cv2.VideoCapture(str(teacher_video)) if show_original else None
    cap_s = cv2.VideoCapture(str(student_video)) if show_original else None
    # Student late: skip ahead so teacher[t] lines up with student[t+lag]
    if cap_s is not None and lag > 0:
        for _ in range(lag):
            cap_s.read()

    blank = np.full((original_height, w, 3), 40, dtype=np.uint8)

    for t in range(n):
        si = t + lag
        sk_left = np.full((h, w, 3), 30, dtype=np.uint8)
        sk_right = np.full((h, w, 3), 30, dtype=np.uint8)
        draw_skeleton(sk_left, teacher_poses[t], (80, 220, 80), 3)
        sc = error_color(float(frame_err[t]) if not np.isnan(frame_err[t]) else 0.0)
        if 0 <= si < len(student_poses):
            draw_skeleton(sk_right, student_poses[si], sc, 3)

        cv2.putText(sk_left, "Teacher pose", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
        cv2.putText(sk_right, "Student pose", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
        time_txt = f"t={t / max(fps, 1e-6):.1f}s"
        cv2.putText(sk_left, time_txt, (16, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 2)
        err = frame_err[t]
        err_txt = f"err={err:.1f}deg" if not np.isnan(err) else "err=NA"
        cv2.putText(sk_right, err_txt, (16, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sc, 2)
        score_txt = f"score={result.overall_score:.0f}"
        cv2.putText(sk_left, score_txt, (16, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 100), 2)

        if result.key_moments:
            nearest = min(result.key_moments, key=lambda m: abs(m.frame - t))
            if abs(nearest.frame - t) <= max(1, int(fps * 0.25)):
                tip = f"{nearest.joint_cn} {nearest.diff_deg:+.0f}deg"
                cv2.putText(sk_right, tip, (16, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 180, 255), 2)

        bottom = np.hstack([sk_left, sk_right])

        if show_original and cap_t is not None and cap_s is not None:
            ok_t, t_raw = cap_t.read()
            if si < 0:
                ok_s, s_raw = False, None
            else:
                ok_s, s_raw = cap_s.read()

            if ok_t and t_raw is not None:
                overlay_t = t_raw.copy()
                if 0 <= t < len(teacher_poses):
                    draw_skeleton(overlay_t, teacher_poses[t], (80, 220, 80), 2)
                top_left = _fit_panel(overlay_t, w, original_height)
            else:
                top_left = blank.copy()

            if ok_s and s_raw is not None:
                overlay_s = s_raw.copy()
                if 0 <= si < len(student_poses):
                    draw_skeleton(overlay_s, student_poses[si], sc, 2)
                top_right = _fit_panel(overlay_s, w, original_height)
            else:
                top_right = blank.copy()

            cv2.putText(top_left, "Teacher", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 255, 120), 2)
            cv2.putText(top_right, "Student", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 180, 255), 2)
            cv2.putText(top_left, time_txt, (16, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            top = np.hstack([top_left, top_right])
            divider = np.full((divider_h, out_w, 3), 70, dtype=np.uint8)
            frame = np.vstack([top, divider, bottom])
        else:
            frame = bottom

        if frame.shape[0] != out_h or frame.shape[1] != out_w:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(frame)

    if cap_t is not None:
        cap_t.release()
    if cap_s is not None:
        cap_s.release()
    writer.release()
    return out_path

def _setup_cn_font() -> None:
    # Prefer macOS Chinese fonts; fall back silently.
    from matplotlib import font_manager

    candidates = [
        "PingFang SC",
        "Heiti SC",
        "Songti SC",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


def save_joint_chart(result: CompareResult, out_path: str | Path) -> Path:
    _setup_cn_font()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(result.joint_mae.keys())
    vals = [result.joint_mae[n] for n in names]
    labels = [JOINT_CN.get(n, n) for n in names]
    fps = result.fps or 30.0

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    colors = [plt.cm.RdYlGn_r(min(v / 30.0, 1.0)) if not np.isnan(v) else (0.5, 0.5, 0.5) for v in vals]
    axes[0].bar(labels, vals, color=colors)
    axes[0].axhline(15, color="orange", ls="--", lw=1, label="noticeable 15°")
    axes[0].axhline(25, color="red", ls="--", lw=1, label="large 25°")
    axes[0].set_ylabel("MAE (degrees)")
    axes[0].set_title(f"Joint error (score {result.overall_score:.0f}/100, lag {result.lag_frames}f)")
    axes[0].legend(loc="upper right")
    axes[0].tick_params(axis="x", rotation=30)

    ranked = sorted(
        ((k, v) for k, v in result.joint_mae.items() if not np.isnan(v)),
        key=lambda x: x[1],
        reverse=True,
    )[:2]
    for name, _ in ranked:
        series = result.joint_series_diff[name]
        xs = np.arange(len(series)) / max(fps, 1e-6)
        axes[1].plot(xs, series, label=JOINT_CN.get(name, name))
    for m in result.key_moments[:6]:
        axes[1].axvline(m.time_sec, color="red", alpha=0.25, lw=1)
        axes[1].annotate(
            f"{m.joint_cn}\n{m.diff_deg:+.0f}°",
            xy=(m.time_sec, m.diff_deg),
            xytext=(4, 8),
            textcoords="offset points",
            fontsize=7,
            color="crimson",
        )
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("student - teacher (deg)")
    axes[1].set_title("Top joint error over time (red lines = key moments)")
    axes[1].legend()

    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def write_report(result: CompareResult, out_path: str | Path, fps: float) -> Path:
    out_path = Path(out_path)
    lines = [
        "# 舞蹈动作对比报告",
        "",
        f"- 相似度得分: **{result.overall_score:.1f} / 100**",
        f"- 估计节奏滞后: **{result.lag_frames} 帧** ({result.lag_frames / max(fps, 1e-6):.2f}s)",
        "",
        "## 关键时间点对比",
        "",
        "| 时间 | 关节 | 老师 | 学生 | 差值 | 说明 |",
        "|---|---|---:|---:|---:|---|",
    ]
    if result.key_moments:
        for m in result.key_moments:
            lines.append(
                f"| {m.time_sec:.1f}s | {m.joint_cn} | {m.teacher_deg:.0f}° | "
                f"{m.student_deg:.0f}° | {m.diff_deg:+.0f}° | {m.note} |"
            )
    else:
        lines.append("| - | - | - | - | - | 未发现超过阈值的瞬时偏差 |")

    lines.extend(
        [
            "",
            "## 关节平均偏差",
            "",
            "| 关节 | MAE (°) |",
            "|---|---|",
        ]
    )
    for name, mae in sorted(result.joint_mae.items(), key=lambda x: (-(x[1] if not np.isnan(x[1]) else -1))):
        cn = JOINT_CN.get(name, name)
        val = f"{mae:.1f}" if not np.isnan(mae) else "N/A"
        lines.append(f"| {cn} | {val} |")
    lines.extend(["", "## 建议", ""])
    for tip in result.suggestions:
        lines.append(f"- {tip}")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def read_video_frame(video_path: str | Path, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_idx < 0 or (total > 0 and frame_idx >= total):
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _resize_to_height(frame: np.ndarray, height: int = 720) -> np.ndarray:
    h, w = frame.shape[:2]
    if h == height:
        return frame
    scale = height / max(h, 1)
    return cv2.resize(frame, (int(w * scale), height), interpolation=cv2.INTER_AREA)


def _put_cn_text(
    img: np.ndarray,
    text: str,
    xy: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    size: int = 28,
) -> np.ndarray:
    """Draw Chinese text with Pillow; color is BGR."""
    from PIL import Image, ImageDraw, ImageFont

    font_paths = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    font = None
    for fp in font_paths:
        if Path(fp).exists():
            try:
                font = ImageFont.truetype(fp, size=size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    r, g, b = color[2], color[1], color[0]
    # shadow for readability
    draw.text((xy[0] + 1, xy[1] + 1), text, font=font, fill=(0, 0, 0))
    draw.text(xy, text, font=font, fill=(r, g, b))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def highlight_joint(
    canvas: np.ndarray,
    landmarks: np.ndarray,
    joint: str,
    color: tuple[int, int, int] = (0, 80, 255),
) -> None:
    focus = JOINT_FOCUS.get(joint)
    if not focus or landmarks is None or np.isnan(landmarks).all():
        return
    h, w = canvas.shape[:2]
    pts = []
    for idx in focus:
        p = _to_px(landmarks[idx], w, h)
        if p:
            pts.append(p)
            cv2.circle(canvas, p, 10, color, 3, cv2.LINE_AA)
    if len(pts) >= 2:
        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i + 1], color, 5, cv2.LINE_AA)


def save_key_moment_snapshots(
    teacher_video: str | Path,
    student_video: str | Path,
    teacher_poses: np.ndarray,
    student_poses: np.ndarray,
    result: CompareResult,
    out_dir: str | Path,
    panel_height: int = 720,
    moments: list | None = None,
    file_prefix: str = "moment",
    title_prefix: str = "关键差异",
) -> list[Path]:
    """
    Save side-by-side real-frame screenshots for each moment.
    Teacher frame uses moment.frame; student uses moment.frame + lag.
    If `moments` is None, uses result.key_moments.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    lag = result.lag_frames
    use_moments = moments if moments is not None else result.key_moments

    for i, m in enumerate(use_moments, start=1):
        t_idx = m.frame
        s_idx = t_idx + lag
        t_frame = read_video_frame(teacher_video, t_idx)
        s_frame = read_video_frame(student_video, s_idx)
        if t_frame is None and s_frame is None:
            continue
        if t_frame is None:
            t_frame = np.full((panel_height, 480, 3), 40, dtype=np.uint8)
        if s_frame is None:
            s_frame = np.full((panel_height, 480, 3), 40, dtype=np.uint8)

        left = _resize_to_height(t_frame, panel_height)
        right = _resize_to_height(s_frame, panel_height)
        target_w = max(left.shape[1], right.shape[1])

        def pad(img: np.ndarray) -> np.ndarray:
            if img.shape[1] == target_w:
                return img
            pad_total = target_w - img.shape[1]
            left_pad = pad_total // 2
            right_pad = pad_total - left_pad
            return cv2.copyMakeBorder(img, 0, 0, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(30, 30, 30))

        left, right = pad(left), pad(right)

        if 0 <= t_idx < len(teacher_poses):
            draw_skeleton(left, teacher_poses[t_idx], (80, 220, 80), 2)
            highlight_joint(left, teacher_poses[t_idx], m.joint, (0, 220, 255))
        if 0 <= s_idx < len(student_poses):
            draw_skeleton(right, student_poses[s_idx], error_color(abs(m.diff_deg)), 2)
            highlight_joint(right, student_poses[s_idx], m.joint, (0, 80, 255))

        banner_h = 110
        banner = np.full((banner_h, target_w * 2, 3), 24, dtype=np.uint8)
        combo = np.vstack([banner, np.hstack([left, right])])

        title = f"{title_prefix} #{i}  |  {m.time_sec:.2f}s  |  {m.joint_cn} 差 {m.diff_deg:+.0f}°"
        detail = f"老师 {m.teacher_deg:.0f}°   vs   学生 {m.student_deg:.0f}°   |   {m.note}"
        combo = _put_cn_text(combo, title, (20, 18), (230, 240, 255), 30)
        combo = _put_cn_text(combo, detail, (20, 60), (180, 210, 255), 22)
        combo = _put_cn_text(combo, "老师", (24, banner_h + 16), (120, 255, 120), 26)
        combo = _put_cn_text(combo, "学生", (target_w + 24, banner_h + 16), (120, 180, 255), 26)

        out_path = out_dir / f"{file_prefix}_{i:02d}_{m.time_sec:.2f}s_{m.joint}.jpg"
        cv2.imwrite(str(out_path), combo, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        paths.append(out_path)

    return paths
