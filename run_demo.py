#!/usr/bin/env python3
"""Run synthetic teacher/student comparison to verify the pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from compare import compare_poses  # noqa: E402
from demo_data import make_student_sequence, make_teacher_sequence  # noqa: E402
from render import save_joint_chart, write_comparison_video, write_report  # noqa: E402


def main() -> None:
    fps = 30.0
    n = 90
    # Known injected errors:
    # - arm only reaches 65% of teacher height
    # - knee 85%
    # - delayed by 8 frames (~0.27s)
    teacher = make_teacher_sequence(n, fps)
    student = make_student_sequence(n, arm_scale=0.65, knee_scale=0.85, lag_frames=8)

    result = compare_poses(teacher, student, fps=fps, align=True)

    out_dir = ROOT / "output" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    video_path = write_comparison_video(
        teacher, student, result, out_dir / "comparison.mp4", fps=fps
    )
    chart_path = save_joint_chart(result, out_dir / "joint_errors.png")
    report_path = write_report(result, out_dir / "report.md", fps=fps)

    summary = {
        "overall_score": round(result.overall_score, 2),
        "lag_frames": result.lag_frames,
        "expected_lag_frames": 8,
        "joint_mae": {k: (None if __import__("math").isnan(v) else round(v, 2)) for k, v in result.joint_mae.items()},
        "key_moments": [m.to_dict() for m in result.key_moments],
        "suggestions": result.suggestions,
        "outputs": {
            "video": str(video_path),
            "chart": str(chart_path),
            "report": str(report_path),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Dance compare demo ===")
    print(f"Score: {result.overall_score:.1f}/100")
    print(f"Detected lag: {result.lag_frames} frames (injected: 8)")
    print("Top joint MAE:")
    for name, mae in sorted(result.joint_mae.items(), key=lambda x: -x[1])[:5]:
        print(f"  {name}: {mae:.1f}°")
    print("Suggestions:")
    for tip in result.suggestions:
        print(f"  - {tip}")
    print(f"\nOutputs written to: {out_dir}")


if __name__ == "__main__":
    main()
