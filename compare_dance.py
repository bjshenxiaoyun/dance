#!/usr/bin/env python3
"""Compare teacher vs student dance videos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from beats import beats_to_key_moments, detect_beats  # noqa: E402
from compare import compare_poses  # noqa: E402
from extract import extract_poses  # noqa: E402
from render import (  # noqa: E402
    save_joint_chart,
    save_key_moment_snapshots,
    write_comparison_video,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare teacher and student dance videos")
    parser.add_argument("--teacher", required=True, help="Teacher video path")
    parser.add_argument("--student", required=True, help="Student video path")
    parser.add_argument("--out", default=str(ROOT / "output" / "run"), help="Output directory")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit")
    parser.add_argument("--no-align", action="store_true", help="Disable temporal alignment")
    parser.add_argument("--beats", action="store_true", default=True, help="Snapshot on musical downbeats (default on)")
    parser.add_argument("--no-beats", action="store_true", help="Disable beat-based snapshots")
    parser.add_argument("--meter", type=int, default=4, help="Beats per bar for downbeat estimate (default 4)")
    parser.add_argument("--max-downbeats", type=int, default=20, help="Max downbeat screenshots")
    args = parser.parse_args()
    use_beats = not args.no_beats

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting teacher poses...")
    teacher, fps_t, _ = extract_poses(args.teacher, max_frames=args.max_frames)
    print("Extracting student poses...")
    student, fps_s, _ = extract_poses(args.student, max_frames=args.max_frames)
    fps = fps_t or fps_s or 30.0

    print("Comparing...")
    result = compare_poses(teacher, student, fps=fps, align=not args.no_align)

    print("Writing comparison video (original + pose)...")
    video_path = write_comparison_video(
        teacher,
        student,
        result,
        out_dir / "comparison.mp4",
        fps=fps,
        teacher_video=args.teacher,
        student_video=args.student,
    )
    chart_path = save_joint_chart(result, out_dir / "joint_errors.png")
    report_path = write_report(result, out_dir / "report.md", fps=fps)

    snap_dir = out_dir / "moments"
    print("Saving key-moment snapshots...")
    snap_paths = save_key_moment_snapshots(
        args.teacher, args.student, teacher, student, result, snap_dir
    )

    beat_info = None
    beat_paths: list[Path] = []
    beat_moments = []
    if use_beats:
        print("Detecting musical downbeats from teacher audio...")
        try:
            beat_info = detect_beats(
                args.teacher,
                assume_meter=args.meter,
                max_downbeats=args.max_downbeats,
            )
            print(f"  tempo≈{beat_info.tempo:.1f} BPM, beats={len(beat_info.beat_times)}, downbeats={len(beat_info.downbeat_times)}")
            beat_moments = beats_to_key_moments(
                beat_info.downbeat_times, result, fps=fps, label_prefix="重拍"
            )
            beat_dir = out_dir / "beats"
            print("Saving downbeat snapshots...")
            beat_paths = save_key_moment_snapshots(
                args.teacher,
                args.student,
                teacher,
                student,
                result,
                beat_dir,
                moments=beat_moments,
                file_prefix="beat",
                title_prefix="音乐重拍",
            )
        except Exception as e:
            print(f"  节拍检测跳过: {e}")

    summary = {
        "overall_score": round(result.overall_score, 2),
        "lag_frames": result.lag_frames,
        "joint_mae": {
            k: (None if __import__("math").isnan(v) else round(v, 2)) for k, v in result.joint_mae.items()
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
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

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

    print(f"\nScore: {result.overall_score:.1f}/100 | lag={result.lag_frames} frames")
    if result.key_moments:
        print("\n关键时间点:")
        for m in result.key_moments:
            print(f"  [{m.time_sec:5.1f}s] {m.joint_cn}: 老师 {m.teacher_deg:.0f}° / 你 {m.student_deg:.0f}° ({m.diff_deg:+.0f}°)")
    print(f"\n关键截图: {len(snap_paths)} 张 -> {snap_dir}")
    if beat_paths:
        print(f"重拍截图: {len(beat_paths)} 张 -> {out_dir / 'beats'}")
        for m in beat_moments[:8]:
            print(f"  [{m.time_sec:5.2f}s] {m.joint_cn}: 老师 {m.teacher_deg:.0f}° / 你 {m.student_deg:.0f}° ({m.diff_deg:+.0f}°)")
    for tip in result.suggestions:
        print(f"- {tip}")
    print(f"\nOutputs: {out_dir}")


if __name__ == "__main__":
    main()
