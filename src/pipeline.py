"""Shared teacher/student comparison pipeline used by both the CLI and the HTTP API."""

from __future__ import annotations

import json
import math
from pathlib import Path

from beats import beats_to_key_moments, detect_beats
from compare import compare_poses
from extract import extract_poses
from render import (
    save_joint_chart,
    save_key_moment_snapshots,
    write_comparison_video,
    write_report,
)


def run_comparison(
    teacher_video: str | Path,
    student_video: str | Path,
    out_dir: str | Path,
    max_frames: int | None = None,
    align: bool = True,
    use_beats: bool = True,
    meter: int = 4,
    max_downbeats: int = 20,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    teacher, fps_t, _ = extract_poses(teacher_video, max_frames=max_frames)
    student, fps_s, _ = extract_poses(student_video, max_frames=max_frames)
    fps = fps_t or fps_s or 30.0

    result = compare_poses(teacher, student, fps=fps, align=align)

    video_path = write_comparison_video(
        teacher,
        student,
        result,
        out_dir / "comparison.mp4",
        fps=fps,
        teacher_video=teacher_video,
        student_video=student_video,
    )
    chart_path = save_joint_chart(result, out_dir / "joint_errors.png")
    report_path = write_report(result, out_dir / "report.md", fps=fps)

    snap_dir = out_dir / "moments"
    snap_paths = save_key_moment_snapshots(
        teacher_video, student_video, teacher, student, result, snap_dir
    )

    beat_info = None
    beat_paths: list[Path] = []
    beat_moments = []
    if use_beats:
        try:
            beat_info = detect_beats(
                teacher_video, assume_meter=meter, max_downbeats=max_downbeats
            )
            beat_moments = beats_to_key_moments(
                beat_info.downbeat_times, result, fps=fps, label_prefix="重拍"
            )
            beat_dir = out_dir / "beats"
            beat_paths = save_key_moment_snapshots(
                teacher_video,
                student_video,
                teacher,
                student,
                result,
                beat_dir,
                moments=beat_moments,
                file_prefix="beat",
                title_prefix="音乐重拍",
            )
        except Exception:
            beat_info = None

    summary = {
        "overall_score": round(result.overall_score, 2),
        "lag_frames": result.lag_frames,
        "joint_mae": {
            k: (None if math.isnan(v) else round(v, 2)) for k, v in result.joint_mae.items()
        },
        "key_moments": [m.to_dict() for m in result.key_moments],
        "beat_track": beat_info.to_dict() if beat_info else None,
        "beat_moments": [m.to_dict() for m in beat_moments],
        "suggestions": result.suggestions,
        "outputs": {
            "video": str(video_path),
            "chart": str(chart_path),
            "report": str(report_path),
            "moments": [str(p) for p in snap_paths],
            "beats": [str(p) for p in beat_paths],
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if snap_paths:
        with report_path.open("a", encoding="utf-8") as f:
            f.write("\n## 关键差异截图\n\n")
            for p in snap_paths:
                rel = p.relative_to(out_dir)
                f.write(f"- `{rel}`\n")
                f.write(f"\n![{p.stem}]({rel.as_posix()})\n\n")

    if beat_paths:
        with report_path.open("a", encoding="utf-8") as f:
            tempo = beat_info.tempo if beat_info else 0
            f.write(f"\n## 音乐重拍截图（约 {tempo:.0f} BPM，按老师音轨）\n\n")
            for p in beat_paths:
                rel = p.relative_to(out_dir)
                f.write(f"- `{rel}`\n")
                f.write(f"\n![{p.stem}]({rel.as_posix()})\n\n")

    return summary
